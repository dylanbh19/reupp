# simple_mail_forecast_ascii.py
# =========================================================
# Ultra-simple mail → call forecast
# * only strongest mail types (debug findings)
# * no exotic dependencies
# * ASCII-only text
# =========================================================

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import holidays
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
CFG = {
    "top_mail_types": [
        "Reject_Ltrs",
        "Cheque 1099",
        "Exercise_Converted",
        "SOI_Confirms",
        "Exch_chks",
        "ACH_Debit_Enrollment",
        "Transfer",
        "COA",
        "NOTC_WITHDRAW",
        "Repl_Chks",
    ],
    "forecast_horizons": [1, 3, 7, 14],
    "output_dir": "dist_simple",
    "random_state": 42,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | simple_mail | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOG = logging.getLogger(__name__)
US_HOLIDAYS = holidays.US()


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def _find_file(candidates: List[str]) -> Path:
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError(f"None of {candidates} found")


def _to_datetime(s: pd.Series) -> pd.Series:
    """Parse to pandas datetime (UTC naive)."""
    return pd.to_datetime(s, errors="coerce")


def _load_mail() -> pd.DataFrame:
    p = _find_file(["mail.csv", "data/mail.csv"])
    df = pd.read_csv(p)
    df.columns = [c.lower().strip() for c in df.columns]
    df["mail_date"] = _to_datetime(df["mail_date"])
    df = df.dropna(subset=["mail_date"])
    return df


def _load_calls() -> pd.Series:
    p_vol = _find_file(["callvolumes.csv", "data/callvolumes.csv"])
    p_int = _find_file(
        ["callintent.csv", "data/callintent.csv", "callintetn.csv"]
    )

    # volumes per day
    df_vol = pd.read_csv(p_vol)
    df_vol.columns = [c.lower().strip() for c in df_vol.columns]
    dcol_v = next(c for c in df_vol.columns if "date" in c)
    df_vol[dcol_v] = _to_datetime(df_vol[dcol_v])
    vol_daily = (
        df_vol.groupby(dcol_v)[df_vol.columns.difference([dcol_v])[0]].sum()
    )

    # intent (one row per call)
    df_int = pd.read_csv(p_int)
    df_int.columns = [c.lower().strip() for c in df_int.columns]
    dcol_i = next(
        c for c in df_int.columns if "date" in c or "conversationstart" in c
    )
    df_int[dcol_i] = _to_datetime(df_int[dcol_i])
    int_daily = df_int.groupby(dcol_i).size()

    # scale volumes to intent on overlap
    overlap = vol_daily.index.intersection(int_daily.index)
    if len(overlap) >= 5:
        scale = int_daily.loc[overlap].mean() / vol_daily.loc[overlap].mean()
        LOG.info("Scaling legacy callvolumes by %.3f", scale)
        vol_daily *= scale

    total = vol_daily.combine_first(int_daily).sort_index()

    # smooth obvious under-counts (below 5th pct)
    q_low = total.quantile(0.05)
    med7 = total.rolling(7, center=True, min_periods=1).median()
    total.loc[total < q_low] = med7
    return total.rename("calls_total")


# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
def build_feature_target(
    mail: pd.DataFrame, calls: pd.Series
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    # aggregate mail
    mail_daily = (
        mail.groupby(["mail_date", "mail_type"], as_index=False)["mail_volume"]
        .sum()
        .pivot(index="mail_date", columns="mail_type", values="mail_volume")
        .fillna(0)
    )

    # keep only recognised mail types
    available = [t for t in CFG["top_mail_types"] if t in mail_daily.columns]
    missing = set(CFG["top_mail_types"]) - set(available)
    if missing:
        LOG.warning("Missing mail types skipped: %s", ", ".join(missing))
    mail_daily = mail_daily[available]

    mail_daily.index = pd.to_datetime(mail_daily.index)
    calls.index = pd.to_datetime(calls.index)

    # business-day intersection
    biz = ~mail_daily.index.weekday.isin([5, 6]) & ~mail_daily.index.isin(
        US_HOLIDAYS
    )
    mail_daily = mail_daily.loc[biz]
    calls = calls.loc[calls.index.isin(mail_daily.index)]

    daily = mail_daily.join(calls, how="inner")

    # -------- features --------
    feat = pd.DataFrame(index=daily.index)

    for col in available[:5]:
        feat[f"{col}_lag1"] = daily[col].shift(1)

    total_mail = daily[available].sum(axis=1)
    feat["total_mail_lag1"] = total_mail.shift(1)
    feat["total_mail_lag2"] = total_mail.shift(2)
    feat["log_total_mail_lag1"] = np.log1p(total_mail).shift(1)

    feat["weekday"] = daily.index.dayofweek
    feat["month"] = daily.index.month
    feat["is_month_end"] = (daily.index.day > 25).astype(int)

    # target (log scale)
    target = np.log1p(daily["calls_total"])

    feat = feat.dropna()
    target = target.loc[feat.index]

    LOG.info(
        "Features ready: %d rows × %d cols", feat.shape[0], feat.shape[1]
    )
    return feat, target, daily["calls_total"].loc[feat.index]


def make_multi_targets(y: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=y.index)
    for h in CFG["forecast_horizons"]:
        out[f"calls_{h}d"] = y.shift(-1).rolling(h).sum()
    return out.dropna()


# ---------------------------------------------------------
# TRAINING
# ---------------------------------------------------------
def train_one_horizon(
    X: pd.DataFrame, y: pd.Series
) -> Tuple[Pipeline, dict]:
    cv = TimeSeriesSplit(n_splits=3)
    pipelines = {
        "Ridge": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(alpha=10.0, random_state=CFG["random_state"])),
            ]
        ),
        "RF": Pipeline(
            [
                (
                    "rf",
                    RandomForestRegressor(
                        n_estimators=100,
                        max_depth=5,
                        min_samples_leaf=5,
                        random_state=CFG["random_state"],
                        n_jobs=-1,
                    ),
                )
            ]
        ),
    }

    results = {}
    for name, pipe in pipelines.items():
        r2s, rmses = [], []
        for tr, te in cv.split(X):
            pipe.fit(X.iloc[tr], y.iloc[tr])
            pred = pipe.predict(X.iloc[te])
            r2s.append(r2_score(y.iloc[te], pred))
            rmses.append(
                mean_squared_error(
                    np.expm1(y.iloc[te]), np.expm1(pred), squared=False
                )
            )
        results[name] = {
            "R2": float(np.mean(r2s)),
            "RMSE": float(np.mean(rmses)),
            "model": pipe,
        }
        LOG.info(
            "  %s  R2=%.3f  RMSE=%.0f",
            name,
            results[name]["R2"],
            results[name]["RMSE"],
        )

    best_name = max(results, key=lambda n: results[n]["R2"])
    best_pipe = results[best_name]["model"]
    best_pipe.fit(X, y)
    return best_pipe, results


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main() -> None:
    out_dir = Path(CFG["output_dir"])
    out_dir.mkdir(exist_ok=True)

    # 1. load
    LOG.info("Loading data …")
    mail_df = _load_mail()
    calls_s = _load_calls()

    # 2. features / targets
    X, y_log, y_raw = build_feature_target(mail_df, calls_s)
    targets = make_multi_targets(y_log)
    common = X.index.intersection(targets.index)
    X, targets, y_raw = X.loc[common], targets.loc[common], y_raw.loc[common]

    # 3. train
    LOG.info("Training …")
    metrics_all, models = {}, {}
    for h in CFG["forecast_horizons"]:
        col = f"calls_{h}d"
        if col not in targets.columns:
            continue
        LOG.info("Horizon %dd", h)
        model, res = train_one_horizon(X, targets[col])
        metrics_all[f"{h}d"] = res
        models[h] = model
        joblib.dump(model, out_dir / f"simple_model_{h}d.pkl")

    # 4. quick forecast (repeat last-week mean features)
    fut_dates = pd.date_range(
        start=X.index[-1] + timedelta(days=1), periods=14, freq="D"
    )
    last_week_mean = X.iloc[-7:].mean()
    fut_feat = pd.DataFrame(
        [last_week_mean.values] * 14, columns=X.columns, index=fut_dates
    )

    forecasts = {
        h: float(np.expm1(models[h].predict(fut_feat.iloc[[0]])[0]))
        for h in models
    }

    # 5. save metrics
    with open(out_dir / "simple_results.json", "w", encoding="utf-8") as fp:
        json.dump(metrics_all, fp, indent=2)

    with open(out_dir / "forecast_summary.json", "w", encoding="utf-8") as fp:
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

    # 6. console summary
    LOG.info("Forecasts: %s", forecasts)
    for horizon, res in metrics_all.items():
        best = max(res, key=lambda n: res[n]["R2"])
        LOG.info(
            "%s best = %s  R2=%.3f  RMSE=%.0f",
            horizon,
            best,
            res[best]["R2"],
            res[best]["RMSE"],
        )
    LOG.info("Outputs: %s", out_dir.resolve())


if __name__ == "__main__":
    main()