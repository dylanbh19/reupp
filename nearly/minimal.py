#!/usr/bin/env python
# simple_mail_forecast_final.py
# =========================================================
# Ultra-simple mail -> call-volume forecast
# • only top-correlated mail types
# • minimal lag + calendar features  
# • correct target construction (sum first, log later)
# =========================================================

from pathlib import Path
import json
import logging
import sys

import numpy as np
import pandas as pd
import holidays
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
CFG = {
    "top_mail_types": [
        "Reject_Ltrs", "Cheque 1099", "Exercise_Converted",
        "SOI_Confirms", "Exch_chks", "ACH_Debit_Enrollment", 
        "Transfer", "COA", "NOTC_WITHDRAW", "Repl_Chks"
    ],
    "forecast_horizons": [1, 3, 7, 14],
    "output_dir": "dist_simple"
}

LOG = logging.getLogger("simple_mail")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | simple_mail | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

US_HOLIDAYS = holidays.US()

# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------
def _to_date(s):
    return pd.to_datetime(s, errors="coerce").dt.date

def _find_file(candidates):
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError(f"None found: {candidates}")

# ---------------------------------------------------------
# DATA LOAD
# ---------------------------------------------------------
def load_data():
    mail_path = _find_file(["mail.csv", "data/mail.csv"])
    vol_path = _find_file(["callvolumes.csv", "data/callvolumes.csv"])  
    intent_path = _find_file(["callintent.csv", "data/callintent.csv", "callintetn.csv"])

    # mail data
    mail = pd.read_csv(mail_path)
    mail.columns = [c.lower().strip() for c in mail.columns]
    mail["mail_date"] = _to_date(mail["mail_date"])
    mail = mail.dropna(subset=["mail_date"])

    # call volumes (legacy)
    df_vol = pd.read_csv(vol_path)
    df_vol.columns = [c.lower().strip() for c in df_vol.columns]
    dcol_v = next(c for c in df_vol.columns if "date" in c)
    df_vol[dcol_v] = _to_date(df_vol[dcol_v])
    vol_daily = df_vol.groupby(dcol_v)[df_vol.columns.difference([dcol_v])[0]].sum()

    # call intent (detailed)
    df_int = pd.read_csv(intent_path)
    df_int.columns = [c.lower().strip() for c in df_int.columns]
    dcol_i = next(c for c in df_int.columns if "date" in c or "conversationstart" in c)
    df_int[dcol_i] = _to_date(df_int[dcol_i])
    int_daily = df_int.groupby(dcol_i).size()

    # scale and merge
    overlap = vol_daily.index.intersection(int_daily.index)
    if len(overlap) >= 5:
        scale = int_daily.loc[overlap].mean() / vol_daily.loc[overlap].mean()
        vol_daily *= scale
        LOG.info("Scaled legacy data by factor %.3f", scale)
    
    calls_total = vol_daily.combine_first(int_daily).sort_index()
    return mail, calls_total

# ---------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------
def build_features(mail, calls_total):
    # pivot mail by type
    mail_daily = (
        mail.groupby(["mail_date", "mail_type"], as_index=False)["mail_volume"]
        .sum()
        .pivot(index="mail_date", columns="mail_type", values="mail_volume")
        .fillna(0)
    )

    # keep only top types
    available_types = [t for t in CFG["top_mail_types"] if t in mail_daily.columns]
    mail_daily = mail_daily[available_types]
    
    # convert to datetime
    mail_daily.index = pd.to_datetime(mail_daily.index)
    calls_total.index = pd.to_datetime(calls_total.index)

    # business days only
    biz_mask = (~mail_daily.index.weekday.isin([5, 6]) 
                & ~mail_daily.index.isin(US_HOLIDAYS))
    mail_daily = mail_daily.loc[biz_mask]
    calls_total = calls_total.loc[calls_total.index.isin(mail_daily.index)]

    # combine
    daily = mail_daily.join(calls_total.rename("calls_total"), how="inner")

    # create minimal feature set
    X = pd.DataFrame(index=daily.index)
    
    # top 5 mail types with 1-day lag
    for mail_type in available_types[:5]:
        X[f"{mail_type}_lag1"] = daily[mail_type].shift(1)

    # total mail features
    total_mail = daily[available_types].sum(axis=1)
    X["total_mail_lag1"] = total_mail.shift(1)
    X["total_mail_lag2"] = total_mail.shift(2)
    X["log_total_mail_lag1"] = np.log1p(total_mail).shift(1)

    # calendar features
    X["weekday"] = daily.index.dayofweek
    X["month"] = daily.index.month
    X["is_month_end"] = (daily.index.day > 25).astype(int)

    # clean
    X = X.dropna()
    y_raw = daily["calls_total"].loc[X.index]

    LOG.info("Features: %d samples x %d features", X.shape[0], X.shape[1])
    return X, y_raw

# ---------------------------------------------------------
# TARGETS (FIXED: sum raw calls first, then log)
# ---------------------------------------------------------
def create_targets(y_raw):
    targets = pd.DataFrame(index=y_raw.index)
    
    for h in CFG["forecast_horizons"]:
        # sum raw call counts for next h days
        raw_sum = y_raw.shift(-1).rolling(h).sum()
        # THEN log transform the sums
        targets[f"calls_{h}d"] = np.log1p(raw_sum)
    
    return targets.dropna()

# ---------------------------------------------------------
# MODEL TRAINING
# ---------------------------------------------------------
def evaluate_model(model, X, y):
    """Cross-validation evaluation"""
    cv = TimeSeriesSplit(n_splits=3)
    rmses, r2s = [], []
    
    for train_idx, test_idx in cv.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = model.predict(X.iloc[test_idx])
        
        rmses.append(np.sqrt(mean_squared_error(y.iloc[test_idx], pred)))
        r2s.append(r2_score(y.iloc[test_idx], pred))
    
    return float(np.mean(rmses)), float(np.mean(r2s))

def train_all_horizons(X, targets):
    """Train models for all forecast horizons"""
    all_metrics = {}
    best_models = {}
    
    for h in CFG["forecast_horizons"]:
        col = f"calls_{h}d"
        if col not in targets.columns:
            continue
            
        LOG.info("Training horizon %dd...", h)
        y = targets[col]
        
        # define models
        models = {
            "Ridge": Pipeline([
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0))
            ]),
            "RF": RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_leaf=3,
                random_state=42
            )
        }
        
        # evaluate each model
        horizon_metrics = {}
        best_name, best_r2 = None, -1e9
        
        for name, model in models.items():
            rmse, r2 = evaluate_model(model, X, y)
            horizon_metrics[name] = {"RMSE": rmse, "R2": r2}
            
            LOG.info("  %s: R2=%.3f RMSE=%.1f", name, r2, rmse)
            
            if r2 > best_r2:
                best_name, best_r2 = name, r2
        
        all_metrics[f"{h}d"] = horizon_metrics
        
        # train best model on full data
        if best_name:
            best_model = models[best_name]
            best_model.fit(X, y)
            best_models[h] = best_model
    
    return all_metrics, best_models

# ---------------------------------------------------------
# FORECASTING
# ---------------------------------------------------------
def generate_forecasts(X, models):
    """Generate forecasts using trained models"""
    forecasts = {}
    
    if models:
        # use last feature row for prediction
        last_features = X.iloc[[-1]]
        
        for horizon, model in models.items():
            pred_log = model.predict(last_features)[0]
            pred_raw = np.expm1(pred_log)  # convert back from log
            forecasts[horizon] = float(pred_raw)
    
    return forecasts

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    output_dir = Path(CFG["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    LOG.info("=== SIMPLE MAIL FORECAST ===")
    
    # load and process data
    LOG.info("Loading data...")
    mail, calls_total = load_data()
    
    LOG.info("Building features...")
    X, y_raw = build_features(mail, calls_total)
    
    LOG.info("Creating targets...")
    targets = create_targets(y_raw)
    
    # align data
    common_idx = X.index.intersection(targets.index)
    X = X.loc[common_idx]
    targets = targets.loc[common_idx]
    
    LOG.info("Final dataset: %d samples", len(X))
    
    # train models
    LOG.info("Training models...")
    metrics, models = train_all_horizons(X, targets)
    
    # generate forecasts
    LOG.info("Generating forecasts...")
    forecasts = generate_forecasts(X, models)
    
    # save results
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    
    with open(output_dir / "forecasts.json", "w") as f:
        json.dump({str(k): v for k, v in forecasts.items()}, f, indent=2)
    
    # save models
    for horizon, model in models.items():
        joblib.dump(model, output_dir / f"model_{horizon}d.pkl")
    
    # summary
    LOG.info("=== RESULTS ===")
    for horizon, pred in forecasts.items():
        LOG.info("Forecast %dd: %.0f calls", horizon, pred)
    
    LOG.info("\nModel performance:")
    for horizon_key, model_metrics in metrics.items():
        best_model = max(model_metrics.keys(), key=lambda x: model_metrics[x]["R2"])
        best_r2 = model_metrics[best_model]["R2"]
        LOG.info("%s: %s (R2=%.3f)", horizon_key, best_model, best_r2)
    
    # success check
    avg_r2 = np.mean([max(m.values(), key=lambda x: x["R2"])["R2"] for m in metrics.values()])
    
    if avg_r2 > 0.1:
        LOG.info("\nSUCCESS: Positive predictive power!")
    elif avg_r2 > 0:
        LOG.info("\nMARGINAL: Weak positive predictive power")
    else:
        LOG.info("\nFAILED: No predictive relationship found")
    
    LOG.info("Results saved to: %s", output_dir.resolve())

if __name__ == "__main__":
    main()