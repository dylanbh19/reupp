# simple_mail_forecast_ascii.py
# ==========================================================
# Ultra-simple mail→calls forecast
# • uses top correlated mail types only
# • trains Ridge + RF with log-target
# • avoids numeric blow-ups and JSON errors
# ==========================================================

from __future__ import annotations

import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import timedelta
import holidays
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------
CFG = {
    "top_mail_types": [
        "Reject_Ltrs", "Cheque 1099", "Exercise_Converted",
        "SOI_Confirms", "Exch_chks", "ACH_Debit_Enrollment",
        "Transfer", "COA", "NOTC_WITHDRAW", "Repl_Chks"
    ],
    "forecast_horizons": [1, 3, 7, 14],
    "output_dir": "dist_simple",
}

LOG = logging.getLogger("simple_mail")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)

# ----------------------------------------------------------
# UTILS
# ----------------------------------------------------------
US_HOLIDAYS = holidays.US()


def _to_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.date


def _find_file(cands: list[str]) -> Path | None:
    for p in cands:
        pth = Path(p)
        if pth.exists():
            return pth
    return None


# ----------------------------------------------------------
# DATA LOAD
# ----------------------------------------------------------
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    """return mail dataframe + daily calls series"""
    mail_p = _find_file(["mail.csv", "data/mail.csv"])
    if mail_p is None:
        raise FileNotFoundError("mail csv not found")

    mail = pd.read_csv(mail_p)
    mail.columns = [c.lower().strip() for c in mail.columns]
    mail["mail_date"] = _to_date(mail["mail_date"])
    mail = mail.dropna(subset=["mail_date"])

    vol_p = _find_file(["callvolumes.csv", "data/callvolumes.csv"])
    int_p = _find_file(
        ["callintent.csv", "data/callintent.csv", "callintetn.csv"]
    )
    if vol_p is None or int_p is None:
        raise FileNotFoundError("call csv files not found")

    # legacy daily volume
    df_vol = pd.read_csv(vol_p)
    df_vol.columns = [c.lower().strip() for c in df_vol.columns]
    dcol_v = next(c for c in df_vol.columns if "date" in c)
    df_vol[dcol_v] = _to_date(df_vol[dcol_v])
    vol_daily = df_vol.groupby(dcol_v)[df_vol.columns.difference([dcol_v])[0]].sum()

    # intent – one row per call
    df_int = pd.read_csv(int_p)
    df_int.columns = [c.lower().strip() for c in df_int.columns]
    dcol_i = next(c for c in df_int.columns if "date" in c or "conversation" in c)
    df_int[dcol_i] = _to_date(df_int[dcol_i])
    int_daily = df_int.groupby(dcol_i).size()

    # scale & merge
    overlap = vol_daily.index.intersection(int_daily.index)
    if len(overlap) >= 5:
        scale = int_daily.loc[overlap].mean() / vol_daily.loc[overlap].mean()
        vol_daily *= scale
    calls = vol_daily.combine_first(int_daily).sort_index()
    return mail, calls.rename("calls_total")


# ----------------------------------------------------------
# FEATURES
# ----------------------------------------------------------
def create_simple_features(
    mail: pd.DataFrame, calls: pd.Series
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    # aggregate mail
    mail_daily = (
        mail.groupby(["mail_date", "mail_type"], as_index=False)["mail_volume"]
        .sum()
        .pivot(index="mail_date", columns="mail_type", values="mail_volume")
        .fillna(0)
    )

    top_types = [t for t in CFG["top_mail_types"] if t in mail_daily.columns]
    mail_daily = mail_daily[top_types]

    mail_daily.index = pd.to_datetime(mail_daily.index)
    calls.index = pd.to_datetime(calls.index)

    mask = (
        ~mail_daily.index.weekday.isin([5, 6])
        & ~mail_daily.index.isin(US_HOLIDAYS)
    )
    mail_daily = mail_daily.loc[mask]
    calls = calls.loc[calls.index.isin(mail_daily.index)]

    daily = mail_daily.join(calls, how="inner")

    feats = pd.DataFrame(index=daily.index)
    for t in top_types[:5]:
        feats[f"{t}_lag1"] = daily[t].shift(1)

    total_mail = daily[top_types].sum(axis=1)
    feats["total_mail_lag1"] = total_mail.shift(1)
    feats["total_mail_lag2"] = total_mail.shift(2)
    feats["log_total_mail_lag1"] = np.log1p(total_mail).shift(1)

    feats["weekday"] = daily.index.dayofweek
    feats["month"] = daily.index.month
    feats["is_month_end"] = (daily.index.day > 25).astype(int)

    target_log = np.log1p(daily["calls_total"])

    feats = feats.dropna()
    target_log = target_log.loc[feats.index]

    LOG.info("Features ready: %d rows × %d cols", *feats.shape)
    return feats, target_log, daily["calls_total"].loc[feats.index]


def create_targets(y_log: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=y_log.index)
    for h in CFG["forecast_horizons"]:
        out[f"calls_{h}d"] = y_log.shift(-1).rolling(h).sum()
    return out.dropna()


# ----------------------------------------------------------
# MODEL TRAINING
# ----------------------------------------------------------
def train_one_horizon(
    X: pd.DataFrame, y: pd.Series, horizon: int
) -> tuple[Pipeline, dict]:
    if len(X) < 30:
        return None, {}

    cv = TimeSeriesSplit(n_splits=3)

    models = {
        "Ridge": Pipeline(
            [("scaler", StandardScaler()), ("reg", Ridge(alpha=10.0))]
        ),
        "RF": RandomForestRegressor(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=5,
            random_state=42,
        ),
    }

    metrics = {}
    best_name, best_r2 = None, -np.inf
    for name, model in models.items():
        r2s, rmses = [], []
        for tr, te in cv.split(X):
            model.fit(X.iloc[tr], y.iloc[tr])
            pred = model.predict(X.iloc[te])

            # evaluate **in log space** to avoid overflow
            r2s.append(r2_score(y.iloc[te], pred))
            rmses.append(np.sqrt(mean_squared_error(y.iloc[te], pred)))

        metrics[name] = {
            "R2": float(np.mean(r2s)),
            "R2_std": float(np.std(r2s)),
            "RMSE_log": float(np.mean(rmses)),
        }

        if metrics[name]["R2"] > best_r2:
            best_name, best_r2 = name, metrics[name]["R2"]

        LOG.info("  %s  R2=%.3f  RMSE=%.0f", name, metrics[name]["R2"], metrics[name]["RMSE_log"])

    best_model = models[best_name]
    best_model.fit(X, y)
    return best_model, metrics


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
def main() -> None:
    out = Path(CFG["output_dir"])
    out.mkdir(exist_ok=True)

    LOG.info("Loading data …")
    mail_df, calls_ser = load_data()

    feats, y_log, y_raw = create_simple_features(mail_df, calls_ser)
    targets = create_targets(y_log)

    common = feats.index.intersection(targets.index)
    X = feats.loc[common]
    targets = targets.loc[common]

    LOG.info("Training …")
    metrics_all = {}
    models = {}

    for h in CFG["forecast_horizons"]:
        LOG.info("Horizon %dd", h)
        col = f"calls_{h}d"
        model, m = train_one_horizon(X, targets[col], h)
        if model is None:
            continue
        models[h] = model
        metrics_all[f"{h}d"] = m
        joblib.dump(model, out / f"simple_model_{h}d.pkl")

    # forecasts (repeat last row)
    forecasts = {}
    for h, mdl in models.items():
        last_feat = X.iloc[[-1]]
        pred_log = mdl.predict(last_feat)[0]
        pred_raw = np.expm1(pred_log)
        forecasts[h] = float(np.clip(pred_raw, 0, 1e7))  # sanity clip

    # save metrics WITHOUT model objects
    with open(out / "simple_results.json", "w", encoding="utf-8") as fp:
        json.dump(metrics_all, fp, indent=2)

    with open(out / "forecast_summary.json", "w", encoding="utf-8") as fp:
        json.dump(
            {
                "forecasts": forecasts,
                "feature_count": X.shape[1],
                "sample_count": X.shape[0],
                "top_mail_types": CFG["top_mail_types"][:5],
            },
            fp,
            indent=2,
        )

    LOG.info("Done – results in %s", out.resolve())


if __name__ == "__main__":
    main()