# -*- coding: utf-8 -*-
"""
Mail-to-Calls Forecasting Pipeline (v7.1)
========================================
A lean, leakage-free script that predicts upcoming call volumes purely from
mail-volume signals.  Key design points:

1. NO intent / economic features – only mail counts + calendar context
2. Multi-horizon targets (1-, 3-, 7-, 14-, 30-day sums) *excluding* the current day
3. Feature selection **inside** each model pipeline → no look-ahead leakage
4. Supports two forecasting modes:
     • Scenario: supply a "future_mail.csv" with planned mail volumes
     • Naïve  : repeat last observed mail volumes forward
5. Bayesian hyper-parameter search on Ridge, ElasticNet, RF, XGB
6. Clean ASCII code – drop straight into "python mail_calls_forecast.py"

Required pip packages (Python ≥3.9):
  pandas numpy matplotlib seaborn holidays scikit-learn scikit-optimize
  xgboost joblib tqdm
"""

import logging, sys, warnings, json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from tqdm import tqdm

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
import joblib

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
RUN_ID = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
CFG: Dict = {
    "data_files": {
        "mail": ["mail.csv", "data/mail.csv"],
        "calls": ["callvolumes.csv", "data/callvolumes.csv"],
        "future_mail": ["future_mail.csv", "data/future_mail.csv"]
    },
    "output_dir": "dist",
    "forecast_horizons": [1, 3, 7, 14, 30],
    "mail_lags": [1, 2, 3, 7, 14],
    "rolling_windows": [3, 7, 14],
    "max_features": 25,
    "cv_splits": 3,
    "random_state": 42,
    "hyper_iter": 12
}

# ----------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format=f"%(asctime)s | {RUN_ID} | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("mail_calls_forecast")

# ----------------------------------------------------------------------------
# Helpers – loading & date parsing
# ----------------------------------------------------------------------------

def find_file(candidates: List[str]) -> Optional[Path]:
    for p in candidates:
        path = Path(p)
        if path.exists():
            log.info(f"Found file: {path}")
            return path
    return None


def _to_dt(series: pd.Series) -> pd.Series:
    s1 = pd.to_datetime(series, errors="coerce", infer_datetime_format=True)
    if s1.isna().mean() <= 0.2:
        return s1
    s2 = pd.to_datetime(series, errors="coerce", infer_datetime_format=True, dayfirst=True)
    return s1.combine_first(s2)


# ----------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------

def load_mail() -> pd.DataFrame:
    path = find_file(CFG["data_files"]["mail"])
    if path is None:
        raise FileNotFoundError("Mail file not found")
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    if {"mail_date", "mail_volume", "mail_type"}.issubset(df.columns) is False:
        raise ValueError("mail_date, mail_volume, mail_type columns required")
    df["mail_date"] = _to_dt(df["mail_date"]).dt.date
    df = df.dropna(subset=["mail_date"])
    return df


def load_calls() -> pd.DataFrame:
    path = find_file(CFG["data_files"]["calls"])
    if path is None:
        raise FileNotFoundError("Calls file not found")
    df = pd.read_csv(path)
    df.columns = df.columns.str.lower().str.strip()
    date_col = next((c for c in df.columns if c in {"date", "call_date", "conversationstart", "conversation_start"}), None)
    if date_col is None:
        raise ValueError("Date column not found in calls file")
    df[date_col] = _to_dt(df[date_col]).dt.date
    df = df.dropna(subset=[date_col])
    return df.rename(columns={date_col: "date"})


# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------

def create_mail_features(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df = df.sort_index()
    mail_cols = [c for c in df.columns if c != "calls_total"]

    # lags
    for col in mail_cols:
        for lag in CFG["mail_lags"]:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)

    # rolling stats (shift 1 to avoid using same-day mail)
    for col in mail_cols:
        for win in CFG["rolling_windows"]:
            rolled = df[col].shift(1).rolling(win)
            df[f"{col}_sum{win}"] = rolled.sum()
            df[f"{col}_avg{win}"] = rolled.mean()

    # calendar
    df["weekday"] = df.index.dayofweek
    df["month"] = df.index.month
    df["quarter"] = df.index.quarter
    us_hols = holidays.US()
    df["is_holiday"] = df.index.to_series().map(lambda d: int(d in us_hols))
    return df


# ----------------------------------------------------------------------------
# Targets (multi-horizon) – exclude current day
# ----------------------------------------------------------------------------

def make_targets(calls: pd.Series) -> pd.DataFrame:
    tgs = {}
    for h in CFG["forecast_horizons"]:
        tgs[f"calls_{h}d"] = calls.shift(-1).rolling(h).sum()  # tomorrow + next h-1 days
    return pd.DataFrame(tgs).dropna()


# ----------------------------------------------------------------------------
# Model utilities
# ----------------------------------------------------------------------------

def build_pipeline(model_name: str) -> Pipeline:
    if model_name == "Ridge":
        reg = Ridge(random_state=CFG["random_state"])
    elif model_name == "ElasticNet":
        reg = ElasticNet(random_state=CFG["random_state"], max_iter=5000)
    elif model_name == "RandomForest":
        reg = RandomForestRegressor(random_state=CFG["random_state"], n_jobs=-1)
    elif model_name == "XGB":
        reg = XGBRegressor(random_state=CFG["random_state"], n_jobs=-1, objective="reg:squarederror", verbosity=0)
    else:
        raise ValueError("unknown model")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("kbest", SelectKBest(f_regression, k=min(CFG["max_features"], 20))),
        ("reg", reg),
    ])
    return pipe


def search_spaces() -> Dict[str, Dict]:
    return {
        "Ridge": {
            "kbest__k": Integer(5, CFG["max_features"]),
            "reg__alpha": Real(0.1, 100.0, prior="log-uniform"),
        },
        "ElasticNet": {
            "kbest__k": Integer(5, CFG["max_features"]),
            "reg__alpha": Real(0.01, 10.0, prior="log-uniform"),
            "reg__l1_ratio": Real(0.1, 0.9),
        },
        "RandomForest": {
            "kbest__k": Integer(5, CFG["max_features"]),
            "reg__n_estimators": Integer(100, 300),
            "reg__max_depth": Integer(5, 15),
            "reg__min_samples_leaf": Integer(1, 8),
        },
        "XGB": {
            "kbest__k": Integer(5, CFG["max_features"]),
            "reg__n_estimators": Integer(100, 300),
            "reg__max_depth": Integer(3, 8),
            "reg__learning_rate": Real(0.03, 0.3),
            "reg__subsample": Real(0.7, 1.0),
            "reg__colsample_bytree": Real(0.7, 1.0),
        },
    }


def cv_metrics(y_true, y_pred) -> Dict[str, float]:
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


# ----------------------------------------------------------------------------
# Training per horizon
# ----------------------------------------------------------------------------

def train_one_horizon(X: pd.DataFrame, y: pd.Series, horizon: int, out_dir: Path) -> Dict[str, Dict]:
    log.info(f"Training horizon {horizon}d (n={len(y)}) ...")
    results = {}
    cv = TimeSeriesSplit(n_splits=CFG["cv_splits"])

    for mdl in ["Ridge", "ElasticNet", "RandomForest", "XGB"]:
        pipe = build_pipeline(mdl)
        search = BayesSearchCV(pipe, search_spaces()[mdl], n_iter=CFG["hyper_iter"],
                               cv=cv, scoring="neg_root_mean_squared_error",
                               random_state=CFG["random_state"], n_jobs=-1, verbose=0)
        search.fit(X, y)
        best = search.best_estimator_
        # cv scores on refit estimator
        scores = []
        for tr, te in cv.split(X):
            best.fit(X.iloc[tr], y.iloc[tr])
            y_hat = best.predict(X.iloc[te])
            scores.append(cv_metrics(y.iloc[te], y_hat))
        avg = {k: float(np.mean([s[k] for s in scores])) for k in scores[0]}
        avg["best_params"] = search.best_params_
        results[mdl] = avg
        joblib.dump(best, out_dir / f"model_{mdl}_{horizon}d.pkl")
        log.info(f"  {mdl}: R2={avg['R2']:.3f} RMSE={avg['RMSE']:.0f} MAPE={avg['MAPE']:.1%}")
    return results


# ----------------------------------------------------------------------------
# Future mail scenarios
# ----------------------------------------------------------------------------

def make_future_mail(last_obs: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Very simple persistence scenario: repeat last observed mail volumes."""
    last_row = last_obs.iloc[[-1]]
    future_idx = pd.date_range(start=last_row.index[-1] + timedelta(days=1), periods=periods, freq="D")
    future = pd.concat([last_row]*periods, ignore_index=True)
    future.index = future_idx
    return future


def load_future_mail(past_mail_cols: List[str], periods: int) -> pd.DataFrame:
    path = find_file(CFG["data_files"]["future_mail"])
    if path is None:
        log.warning("future_mail.csv not provided – using naive persistence")
        return pd.DataFrame(columns=past_mail_cols)  # empty triggers persistence
    f = pd.read_csv(path)
    f.columns = [c.lower().strip() for c in f.columns]
    if {"date"}.union(set(past_mail_cols)).issubset(f.columns) is False:
        log.warning("future_mail.csv columns mismatch – falling back to persistence")
        return pd.DataFrame(columns=past_mail_cols)
    f["date"] = _to_dt(f["date"]).dt.date
    f = f.set_index("date").sort_index()
    return f[past_mail_cols]


# ----------------------------------------------------------------------------
# Forecast visualisation
# ----------------------------------------------------------------------------

def plot_forecast(history: pd.Series, future_preds: Dict[int, pd.Series], out_dir: Path):
    plt.figure(figsize=(14, 8))
    # history – last 60 days
    history.iloc[-60:].plot(label="Historical Calls", marker="o", linewidth=1.5)
    for h, pred in future_preds.items():
        plt.plot(pred.index, pred.values, marker="s", linewidth=2, label=f"Forecast {h}d horizon")
    plt.legend()
    plt.title("Call-Volume Forecast")
    plt.ylabel("Calls")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "mail_to_calls_forecast.png", dpi=300)
    plt.close()


# ----------------------------------------------------------------------------
# Dataset builder
# ----------------------------------------------------------------------------

def build_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    mail = load_mail()
    calls = load_calls()

    # daily aggregation
    mail_daily = (
        mail.groupby(["mail_date", "mail_type"], as_index=False)["mail_volume"].sum()
            .rename(columns={"mail_date": "date"})
    )
    mail_wide = mail_daily.pivot(index="date", columns="mail_type", values="mail_volume").fillna(0)
    calls_daily = calls.groupby("date").size().rename("calls_total")
    daily = mail_wide.join(calls_daily, how="inner")
    daily.index = pd.to_datetime(daily.index)
    # business days only
    hols = holidays.US()
    daily = daily[(~daily.index.weekday.isin([5, 6])) & (~daily.index.isin(hols))]

    # feature matrix
    feats = create_mail_features(daily)
    X = feats.drop(columns=["calls_total"])
    y = feats["calls_total"]

    # targets
    targets = make_targets(y)
    idx = X.index.intersection(targets.index)
    return X.loc[idx], targets.loc[idx], y.loc[idx]


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    out_dir = Path(CFG["output_dir"])
    out_dir.mkdir(exist_ok=True, parents=True)
    log.info("==== MAIL→CALLS PIPELINE START ====")

    X, T, y = build_dataset()
    X.to_csv(out_dir / "features.csv")
    T.to_csv(out_dir / "targets.csv")

    all_results: Dict[str, Dict] = {}
    for h in CFG["forecast_horizons"]:
        tg_name = f"calls_{h}d"
        if tg_name not in T.columns:
            continue
        res = train_one_horizon(X, T[tg_name], h, out_dir)
        all_results[tg_name] = res

    # --- create future DF for forecasting ---
    mail_cols = [c for c in X.columns if not c.startswith("calls_total") and "_lag" not in c]
    future_mail = load_future_mail(mail_cols, periods=30)
    if future_mail.empty:
        future_mail = make_future_mail(X[mail_cols], periods=30)

    future_base = pd.concat([X[mail_cols], future_mail])
    future_feats = create_mail_features(pd.concat([future_base, y], axis=1, join="inner"))
    X_future = future_feats.drop(columns=["calls_total"]).iloc[-len(future_mail):]

    # generate predictions
    future_preds: Dict[int, pd.Series] = {}
    for h in CFG["forecast_horizons"]:
        mdl_file = out_dir / f"model_XGB_{h}d.pkl"
        if not mdl_file.exists():
            mdl_file = next((out_dir.glob(f"model_*_{h}d.pkl")), None)
        if mdl_file is None:
            continue
        mdl = joblib.load(mdl_file)
        pred = pd.Series(mdl.predict(X_future), index=X_future.index)
        future_preds[h] = pred

    plot_forecast(y, future_preds, out_dir)

    # summary json
    with open(out_dir / "results.json", "w") as fp:
        json.dump({"run_id": RUN_ID, "metrics": all_results}, fp, indent=2)

    log.info("Forecast complete. Outputs in %s", out_dir.resolve())


if __name__ == "__main__":
    main()
