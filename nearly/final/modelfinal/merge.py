#!/usr/bin/env python
# mail_input_range_forecast_v2.py
# =========================================================
# INPUT-DRIVEN RANGE FORECAST MODEL
#  - builds quantile + bootstrap models from user-specified
#    mail volumes, with optional economic indicators.
#  - reports MAE *and* R² for median model + ensemble.
# =========================================================

from __future__ import annotations
import warnings, sys, json, logging
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import holidays, joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
CFG: dict = {
    "top_mail_types": [
        "Reject_Ltrs", "Cheque 1099", "Exercise_Converted",
        "SOI_Confirms", "Exch_chks", "ACH_Debit_Enrollment",
        "Transfer", "COA", "NOTC_WITHDRAW", "Repl_Chks"
    ],
    "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
    "bootstrap_samples": 30,
    "top_econ": 5,                   # how many economic columns to keep
    "econ_files": ["econ_orig.csv", "econ_extra.csv"],  # add your filenames
    "output_dir": "dist_input_ranges"
}

# ---------------------------------------------------------
# LOGGING
# ---------------------------------------------------------
LOG = logging.getLogger("mail_input")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | mail_input | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

US_HOLIDAYS = holidays.US()

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date

def _find_file(candidates: list[str]) -> Path:
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError(f"None found: {candidates}")

# ---------------------------------------------------------
# DATA INGEST
# ---------------------------------------------------------
def load_mail_call_data() -> pd.DataFrame:
    """Return daily frame containing mail types + calls_total."""
    # ---------- mail ----------
    mail_path = _find_file(["mail.csv", "data/mail.csv"])
    mail = pd.read_csv(mail_path)
    mail.columns = [c.lower().strip() for c in mail.columns]
    mail["mail_date"] = _to_date(mail["mail_date"])
    mail = mail.dropna(subset=["mail_date"])

    # ---------- calls ----------
    vol_path   = _find_file(["callvolumes.csv", "data/callvolumes.csv"])
    intent_path = _find_file(
        ["callintent.csv", "data/callintent.csv", "callintetn.csv"]
    )

    df_vol = pd.read_csv(vol_path)
    df_vol.columns = [c.lower().strip() for c in df_vol.columns]
    dcol_v = next(c for c in df_vol.columns if "date" in c)
    df_vol[dcol_v] = _to_date(df_vol[dcol_v])
    vol_daily = (
        df_vol.groupby(dcol_v)[df_vol.columns.difference([dcol_v])[0]].sum()
    )

    df_int = pd.read_csv(intent_path)
    df_int.columns = [c.lower().strip() for c in df_int.columns]
    dcol_i = next(c for c in df_int.columns if "date" in c or "conversationstart" in c)
    df_int[dcol_i] = _to_date(df_int[dcol_i])
    int_daily = df_int.groupby(dcol_i).size()

    overlap = vol_daily.index.intersection(int_daily.index)
    if len(overlap) >= 5:
        scale = int_daily.loc[overlap].mean() / vol_daily.loc[overlap].mean()
        LOG.info("Scaled call volumes by factor: %.2f", scale)
        vol_daily *= scale

    calls_total = vol_daily.combine_first(int_daily).sort_index()

    # ---------- pivot mail ----------
    mail_daily = (
        mail.groupby(["mail_date", "mail_type"], as_index=False)["mail_volume"]
        .sum()
        .pivot(index="mail_date", columns="mail_type", values="mail_volume")
        .fillna(0)
    )

    mail_daily.index  = pd.to_datetime(mail_daily.index)
    calls_total.index = pd.to_datetime(calls_total.index)

    # business day filter
    mask = (~mail_daily.index.weekday.isin([5, 6])
            & ~mail_daily.index.isin(US_HOLIDAYS))
    daily = mail_daily.loc[mask].join(
        calls_total.loc[mask].rename("calls_total"), how="inner"
    )
    LOG.info("Daily mail+calls frame: %s", daily.shape)
    return daily

def load_economic_data(daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Load all CSVs in CFG['econ_files'] and align to business-day index."""
    frames = []
    for fname in CFG["econ_files"]:
        path = Path(fname)
        if not path.exists():
            LOG.warning("Economic file %s not found, skipping.", fname)
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        # assume first column is date
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
        df = df.set_index(df.columns[0]).sort_index()
        frames.append(df)
        LOG.info("Loaded econ file %s with %d columns.", fname, df.shape[1])
    if not frames:
        return pd.DataFrame(index=daily_index)
    econ = pd.concat(frames, axis=1)
    econ = econ.reindex(daily_index).ffill()  # forward-fill latest known
    LOG.info("Combined econ frame: %s", econ.shape)
    return econ

# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
def create_features(
    daily: pd.DataFrame,
    econ: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    """Return X (features) and y (next-day calls)."""

    avail_mail = [t for t in CFG["top_mail_types"] if t in daily.columns]
    features, targets = [], []

    for i in range(len(daily) - 1):
        cur = daily.iloc[i]
        nxt = daily.iloc[i + 1]
        row = {}

        # --- mail volumes ---
        for mt in avail_mail:
            row[f"{mt}_volume"] = cur[mt]
        total_mail = sum(cur[mt] for mt in avail_mail)
        row["total_mail_volume"]     = total_mail
        row["log_total_mail_volume"] = np.log1p(total_mail)

        # percentile of today’s mail vs history so far
        hist_total = daily[avail_mail].sum(axis=1).iloc[: i + 1]
        row["mail_percentile"] = (hist_total <= total_mail).mean()

        # --- calendar ---
        dt = daily.index[i]
        row["weekday"]       = dt.weekday()
        row["month"]         = dt.month
        row["is_month_end"]  = int(dt.day > 25)
        row["is_holiday_we"] = int(dt in US_HOLIDAYS)

        # --- recent calls context ---
        recent = daily["calls_total"].iloc[max(0, i - 5) : i + 1]
        row["recent_calls_avg"]   = recent.mean()
        row["recent_calls_trend"] = recent.diff().mean()

        # --- economic indicators (today & lag1) ---
        if econ is not None and not econ.empty:
            for col in econ.columns:
                row[f"{col}_today"] = econ.iloc[i][col]
                row[f"{col}_lag1"]  = econ.iloc[i - 1][col] if i > 0 else np.nan

        features.append(row)
        targets.append(nxt["calls_total"])

    X = pd.DataFrame(features).fillna(method="ffill").fillna(0)
    y = pd.Series(targets, index=X.index, name="next_day_calls")
    LOG.info("Features ready: %d rows × %d cols", *X.shape)
    return X, y

# ---------------------------------------------------------
# ECON SELECTION
# ---------------------------------------------------------
def select_top_econ(daily: pd.DataFrame, econ: pd.DataFrame) -> pd.DataFrame:
    """Pick top-k econ indicators by next-day correlation with calls."""
    if econ.empty:
        return econ
    calls = daily["calls_total"].iloc[1:]  # align with next day
    corrs = {}
    for col in econ.columns:
        same = econ[col].iloc[:-1].corr(calls)
        corrs[col] = abs(same)
    top_cols = sorted(corrs, key=corrs.get, reverse=True)[: CFG["top_econ"]]
    LOG.info("Selected top econ: %s", top_cols)
    return econ[top_cols]

# ---------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------
def train_models(X: pd.DataFrame, y: pd.Series) -> dict:
    """Train quantile, bootstrap and median point models with metrics."""

    split = int(len(X) * 0.8)
    X_tr, X_te = X.iloc[:split], X.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    models: dict[str, object] = {}
    metrics: dict[str, dict]  = {}

    # ------- quantile regressors -------
    for q in CFG["quantiles"]:
        m = QuantileRegressor(quantile=q, alpha=0.1, solver="highs")
        m.fit(X_tr, y_tr)
        y_pred = m.predict(X_te)
        mae = mean_absolute_error(y_te, y_pred)
        # only compute R² for the median (q = 0.5) – others are not centred
        r2 = r2_score(y_te, y_pred) if q == 0.5 else None
        name = f"quantile_{q:.2f}"
        models[name]  = m
        metrics[name] = {"MAE": mae, "R2": r2}
        LOG.info("Quantile %.0f%% – MAE: %.0f%s",
                 q * 100, mae,
                 f", R2: {r2:.3f}" if r2 is not None else "")

    # ------- bootstrap ensemble (RF) -------
    preds_boot = []
    boot_models = []
    for i in range(CFG["bootstrap_samples"]):
        idx = np.random.choice(len(X_tr), len(X_tr), replace=True)
        m = RandomForestRegressor(
            n_estimators=60, max_depth=6,
            min_samples_leaf=3, random_state=i
        )
        m.fit(X_tr.iloc[idx], y_tr.iloc[idx])
        boot_models.append(m)
        preds_boot.append(m.predict(X_te))
    preds_boot = np.vstack(preds_boot)
    y_pred_mean = preds_boot.mean(axis=0)
    mae_boot = mean_absolute_error(y_te, y_pred_mean)
    rmse_boot = np.sqrt(mean_squared_error(y_te, y_pred_mean))
    r2_boot   = r2_score(y_te, y_pred_mean)
    models["bootstrap"] = boot_models
    metrics["bootstrap"] = {
        "MAE": mae_boot, "RMSE": rmse_boot, "R2": r2_boot
    }
    LOG.info("Bootstrap ensemble – R2: %.3f, MAE: %.0f, RMSE: %.0f",
             r2_boot, mae_boot, rmse_boot)
    return models, metrics

# ---------------------------------------------------------
# PREDICTION INTERFACE
# ---------------------------------------------------------
def predict_calls(
    models: dict, mail_inputs: dict[str, int], date_str: str | None = None
) -> tuple[dict, dict]:
    """Return quantile dict + bootstrap summary for given mail volumes."""
    date = (datetime.strptime(date_str, "%Y-%m-%d")
            if date_str else datetime.now() + timedelta(days=1))

    feat = {f"{t}_volume": mail_inputs.get(t, 0)
            for t in CFG["top_mail_types"]}
    total = sum(feat.values())
    feat["total_mail_volume"]     = total
    feat["log_total_mail_volume"] = np.log1p(total)
    feat["mail_percentile"]       = 0.5
    feat["weekday"] = date.weekday()
    feat["month"]   = date.month
    feat["is_month_end"]  = int(date.day > 25)
    feat["is_holiday_we"] = int(date.date() in US_HOLIDAYS)
    feat["recent_calls_avg"]   = 15000
    feat["recent_calls_trend"] = 0
    Xnew = pd.DataFrame([feat])

    # quantile predictions
    q_pred = {}
    for q in CFG["quantiles"]:
        m = models[f"quantile_{q:.2f}"]
        q_pred[f"q{int(q*100)}"] = max(0, m.predict(Xnew)[0])

    # bootstrap predictions
    boot_preds = [m.predict(Xnew)[0] for m in models["bootstrap"]]
    boot_stats = {
        "mean": float(np.mean(boot_preds)),
        "std":  float(np.std(boot_preds)),
        "min":  float(np.min(boot_preds)),
        "max":  float(np.max(boot_preds))
    }
    return q_pred, boot_stats

# ---------------------------------------------------------
# SCENARIO TEST
# ---------------------------------------------------------
def run_scenarios(models: dict, out_dir: Path) -> None:
    scenarios = [
        {"name": "Scenario_A",
         "desc": "Large reject-letter day",
         "mail": {"Reject_Ltrs": 2000, "Cheque 1099": 500}},
        {"name": "Scenario_B",
         "desc": "Typical mixed mail",
         "mail": {"Reject_Ltrs": 800, "Cheque 1099": 1200,
                  "Exercise_Converted": 300}},
        {"name": "Scenario_C",
         "desc": "Low admin mail",
         "mail": {"Transfer": 200, "COA": 150}},
    ]
    res = {}
    for sc in scenarios:
        q, b = predict_calls(models, sc["mail"])
        LOG.info("\n%-12s | %s", sc["name"], sc["desc"])
        LOG.info("  q25-q75 : %.0f – %.0f", q["q25"], q["q75"])
        LOG.info("  q10-q90 : %.0f – %.0f", q["q10"], q["q90"])
        LOG.info("  median  : %.0f",         q["q50"])
        res[sc["name"]] = {"desc": sc["desc"], "mail": sc["mail"],
                           "quantiles": q, "bootstrap": b}
    with open(out_dir / "scenario_results.json", "w") as fp:
        json.dump(res, fp, indent=2)

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main() -> None:
    out = Path(CFG["output_dir"]); out.mkdir(exist_ok=True)

    LOG.info("=== COMPREHENSIVE MAIL-INPUT FORECAST ===")
    daily = load_mail_call_data()

    # ----- economic data (optional) -----
    econ_raw = load_economic_data(daily.index)
    econ_sel = select_top_econ(daily, econ_raw) if not econ_raw.empty else None

    # ----- feature building -----
    X, y = create_features(daily, econ_sel)

    # ----- model training -----
    models, met = train_models(X, y)
    with open(out / "metrics.json", "w") as fp:
        json.dump(met, fp, indent=2)

    joblib.dump(models, out / "mail_input_models.pkl")

    # ----- scenario tests + interactive example -----
    run_scenarios(models, out)

    LOG.info("Results saved to %s", out.resolve())

if __name__ == "__main__":
    main()