# mail_calls_forecast_v8.py
# ==========================================================
# Predict daily call-centre volume from incoming mail counts
# ----------------------------------------------------------
# * joins **two** call sources ("callvolumes" + "callintent")
# * scales legacy callvolumes to match intent file on overlap
# * smooths obviously under-reported days with a centred 7-day
#   median (only where count < 5th percentile of combined data)
# * mail-only feature set – no intents used as predictors
# * multi-horizon targets (1, 3, 7, 14 days)
# * fast Bayesian hyper-search for 3 models (Ridge, RF, XGB)
# * outputs: forecast plot + per-horizon best model pickle +
#   CSV/JSON metrics
# ----------------------------------------------------------

from __future__ import annotations

import json, logging, os, sys, warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import holidays

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_percentage_error, r2_score)
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.feature_selection import SelectKBest, f_regression

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RUN_ID = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
CFG: Dict = {
    "data_files": {
        "mail": ["mail.csv", "data/mail.csv"],
        "calls_vol": ["callvolumes.csv", "data/callvolumes.csv"],
        "calls_int": ["callintent.csv", "data/callintent.csv", "callintetn.csv"]
    },
    "output_dir": "dist",
    "forecast_horizons": [1, 3, 7, 14],
    "max_features": 25,
    "ts_splits": 3,
    "hyper_iter": 12,
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s | {RUN_ID} | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOG = logging.getLogger("mail_calls")

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------
US_HOLIDAYS = holidays.US()

def _find_file(candidates: List[str]) -> Optional[Path]:
    for p in candidates:
        path = Path(p)
        if path.exists():
            LOG.info(f"Found file: {path}")
            return path
    return None

def _to_date(s: pd.Series) -> pd.Series:
    s1 = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if s1.isna().mean() > 0.3:
        s1 = pd.to_datetime(s.str[:10], errors="coerce")
    return s1.dt.date

# ---------------------------------------------------------------------------
# DATA LOAD + AUGMENT
# ---------------------------------------------------------------------------

def load_mail() -> pd.DataFrame:
    path = _find_file(CFG["data_files"]["mail"])
    if not path:
        raise FileNotFoundError("mail.csv not found")
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    if {"mail_date", "mail_volume", "mail_type"}.issubset(df.columns) is False:
        raise ValueError("mail file must contain mail_date, mail_volume, mail_type")
    df["mail_date"] = _to_date(df["mail_date"])
    df = df.dropna(subset=["mail_date"])
    return df


def load_calls() -> pd.Series:
    # legacy volumes file (may be incomplete)
    p_vol = _find_file(CFG["data_files"]["calls_vol"])
    # rich intent file (line per call)
    p_int = _find_file(CFG["data_files"]["calls_int"])
    if not (p_vol and p_int):
        raise FileNotFoundError("Both call files are required")

    # volumes file ─ one row per day (date,calls)
    df_vol = pd.read_csv(p_vol)
    df_vol.columns = [c.lower().strip() for c in df_vol.columns]
    dcol_v = next(c for c in df_vol.columns if "date" in c)
    df_vol[dcol_v] = _to_date(df_vol[dcol_v])
    vol_daily = (
        df_vol.groupby(dcol_v)[df_vol.columns.difference([dcol_v])[0]].sum()
        .rename("calls_vol")
    )

    # intent file ─ one row per call
    df_int = pd.read_csv(p_int)
    df_int.columns = [c.lower().strip() for c in df_int.columns]
    dcol_i = next(c for c in df_int.columns if "date" in c or "conversationstart" in c)
    df_int[dcol_i] = _to_date(df_int[dcol_i])
    int_daily = df_int.groupby(dcol_i).size().rename("calls_int")

    # align + scale volumes to intent on overlap
    overlap = vol_daily.index.intersection(int_daily.index)
    if len(overlap) >= 5:
        scale = (int_daily.loc[overlap].mean() / vol_daily.loc[overlap].mean())
        LOG.info(f"Scaling callvolumes by factor {scale:.3f} to match intent data")
        vol_daily *= scale
    total = vol_daily.combine_first(int_daily).sort_index()

    # mark low counts (below 5th percentile) + replace with centred 7-day median
    q_low = total.quantile(0.05)
    median7 = total.rolling(7, center=True, min_periods=1).median()
    low_mask = total < q_low
    total.loc[low_mask] = median7.loc[low_mask]
    LOG.info(f"Applied smoothing to {low_mask.sum()} low-volume days")

    return total.rename("calls_total")

# ---------------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def make_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    mail_cols = [c for c in df.columns if c != "calls_total"]

    # simple lags + rolling summaries
    for col in mail_cols:
        for lag in (1, 2, 3, 7, 14):
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
        for win in (3, 7, 14):
            df[f"{col}_sum{win}"] = df[col].shift(1).rolling(win).sum()

    # calendar flags
    df["weekday"] = df.index.dayofweek
    df["is_monday"] = (df.index.dayofweek == 0).astype(int)
    df["is_friday"] = (df.index.dayofweek == 4).astype(int)
    df["month"] = df.index.month
    df["is_holiday"] = df.index.to_series().apply(lambda x: x in US_HOLIDAYS).astype(int)

    df = df.dropna()
    return df


def select_features(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    sel = SelectKBest(f_regression, k=min(CFG["max_features"], X.shape[1]))
    X_sel = sel.fit_transform(X, y)
    return pd.DataFrame(X_sel, columns=X.columns[sel.get_support()], index=X.index)

# ---------------------------------------------------------------------------
# TARGETS
# ---------------------------------------------------------------------------

def make_targets(y: pd.Series) -> pd.DataFrame:
    t = pd.DataFrame(index=y.index)
    for h in CFG["forecast_horizons"]:
        t[f"calls_{h}d"] = y.rolling(h).sum().shift(-h)
    return t.dropna()

# ---------------------------------------------------------------------------
# MODELS
# ---------------------------------------------------------------------------

def model_space(name: str):
    if name == "Ridge":
        return {
            "reg__alpha": Real(0.1, 50.0, prior="log-uniform")
        }
    if name == "RF":
        return {
            "n_estimators": Integer(120, 300),
            "max_depth": Integer(4, 12),
            "min_samples_leaf": Integer(1, 8)
        }
    if name == "XGB":
        return {
            "n_estimators": Integer(120, 300),
            "max_depth": Integer(3, 8),
            "learning_rate": Real(0.03, 0.3, prior="log-uniform")
        }
    raise ValueError(name)


def build_estimator(name: str):
    if name == "Ridge":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge(random_state=CFG["random_state"]))
        ])
    if name == "RF":
        return RandomForestRegressor(random_state=CFG["random_state"], n_jobs=-1)
    if name == "XGB":
        return XGBRegressor(random_state=CFG["random_state"], n_jobs=-1,
                            objective="reg:squarederror", verbosity=0)

# ---------------------------------------------------------------------------
# TRAIN ONE HORIZON
# ---------------------------------------------------------------------------

def train_horizon(X: pd.DataFrame, y: pd.Series, horizon: int, out: Path) -> Dict:
    cv = TimeSeriesSplit(n_splits=CFG["ts_splits"])
    results = {}
    for name in ("Ridge", "RF", "XGB"):
        est = build_estimator(name)
        search = BayesSearchCV(
            est, model_space(name), n_iter=CFG["hyper_iter"],
            cv=cv, scoring="neg_root_mean_squared_error",
            random_state=CFG["random_state"], n_jobs=-1, verbose=0)
        search.fit(X, y)
        best = search.best_estimator_

        # cross-val metrics
        mses, mapes, r2s = [], [], []
        for tr, te in cv.split(X):
            best.fit(X.iloc[tr], y.iloc[tr])
            p = best.predict(X.iloc[te])
            mses.append(mean_squared_error(y.iloc[te], p, squared=False))
            mapes.append(mean_absolute_percentage_error(y.iloc[te], p))
            r2s.append(r2_score(y.iloc[te], p))
        res = {
            "RMSE": float(np.mean(mses)),
            "MAPE": float(np.mean(mapes)),
            "R2": float(np.mean(r2s)),
            "params": search.best_params_
        }
        results[name] = res
        # save model fitted on full data
        best.fit(X, y)
        joblib.dump(best, out / f"model_{name}_{horizon}d.pkl")
        LOG.info(f"  {name} h{horizon}: R2={res['R2']:.3f} RMSE={res['RMSE']:.0f}")
    return results

# ---------------------------------------------------------------------------
# PLOT
# ---------------------------------------------------------------------------

def plot_forecast(y_hist: pd.Series, preds: Dict[int, pd.Series], out: Path):
    plt.figure(figsize=(14, 6))
    plt.plot(y_hist.index, y_hist, "o-", label="Historical", alpha=0.7)
    for h, s in preds.items():
        plt.plot(s.index, s, "-", label=f"{h}d forecast")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.ylabel("Calls"); plt.title("Mail→Call Volume Forecast")
    plt.tight_layout(); plt.savefig(out / "mail_to_calls_forecast.png", dpi=300)
    plt.close()

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    out = Path(CFG["output_dir"]); out.mkdir(exist_ok=True)

    # 1. LOAD + PREP
    LOG.info("[1/4] load + prep data")
    mail = load_mail()
    calls = load_calls()

    # pivot mail daily
    mail_daily = (mail.groupby(["mail_date", "mail_type"], as_index=False)["mail_volume"].sum()
                   .pivot(index="mail_date", columns="mail_type", values="mail_volume").fillna(0))
    mail_daily.index = pd.to_datetime(mail_daily.index)
    calls.index = pd.to_datetime(calls.index)

    # keep business days only
    biz_mask = ~mail_daily.index.weekday.isin([5,6]) & ~mail_daily.index.isin(US_HOLIDAYS)
    mail_daily = mail_daily[biz_mask]
    calls = calls[biz_mask]

    daily = mail_daily.join(calls, how="inner")  # intersection of dates
    LOG.info(f"Final daily frame: {daily.shape}")

    # 2. FEATURES + TARGETS
    feats = make_features(daily)
    X_raw = feats.drop(columns=["calls_total"])
    y_raw = feats["calls_total"]
    X = select_features(X_raw, y_raw)
    targets = make_targets(y_raw)
    common = X.index.intersection(targets.index)
    X, targets = X.loc[common], targets.loc[common]

    # 3. TRAIN PER HORIZON
    LOG.info("[2/4] training models")
    all_metrics, all_preds = {}, {}
    for h in CFG["forecast_horizons"]:
        col = f"calls_{h}d"
        if col not in targets.columns: continue
        m = train_horizon(X, targets[col], h, out)
        all_metrics[f"{h}d"] = m
        # pick best model by R2 to forecast next 14 days for plot
        best_name = max(m, key=lambda n: m[n]["R2"])
        model = joblib.load(out / f"model_{best_name}_{h}d.pkl")
        fut_dates = pd.date_range(X.index[-1] + timedelta(days=1), periods=14)
        fut_X = X.iloc[-14:].copy()  # naive: use last 14 feature rows
        all_preds[h] = pd.Series(model.predict(fut_X), index=fut_dates)

    # 4. PLOT + SAVE METRICS
    LOG.info("[3/4] creating forecast plot")
    plot_forecast(y_raw, all_preds, out)
    with open(out / "results.json", "w") as f:
        json.dump(all_metrics, f, indent=2)
    LOG.info("[4/4] done – outputs in dist/")

if __name__ == "__main__":
    main()
