vPS C:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod> & C:/Users/BhungarD/python.exe "c:/Users/BhungarD/OneDrive - Computershare/Desktop/finprod/debug.py"
2025-07-17 13:43:52,004 | baseline | INFO | === BASELINE TEST ===
2025-07-17 13:44:00,482 | baseline | INFO | Daily data shape: (349, 232)
2025-07-17 13:44:00,685 | baseline | INFO | Same-day correlations:
2025-07-17 13:44:00,685 | baseline | INFO |   Raw: 0.117
2025-07-17 13:44:00,685 | baseline | INFO |   Log: 0.269
2025-07-17 13:44:00,686 | baseline | INFO | Top mail type correlations:
2025-07-17 13:44:00,686 | baseline | INFO |   Reject_Ltrs: 0.342
2025-07-17 13:44:00,686 | baseline | INFO |   Cheque 1099: 0.277
2025-07-17 13:44:00,687 | baseline | INFO |   Exercise_Converted: 0.240
2025-07-17 13:44:00,687 | baseline | INFO |   SOI_Confirms: 0.234
2025-07-17 13:44:00,687 | baseline | INFO |   Exch_chks: 0.223
2025-07-17 13:44:00,691 | baseline | INFO | Features: 348 samples x 9 features
2025-07-17 13:44:00,692 | baseline | INFO |
Testing simple models...
2025-07-17 13:44:00,692 | baseline | INFO | Testing RAW target...
2025-07-17 13:44:00,711 | baseline | INFO |   RAW Ridge R2: -7.058 (scores: ['-20.122', '-0.841', '-0.212'])
2025-07-17 13:44:00,712 | baseline | INFO | Testing LOG target...
2025-07-17 13:44:00,723 | baseline | INFO |   LOG Ridge R2: -30.900 (scores: ['-90.442', '-2.216', '-0.042'])
2025-07-17 13:44:00,724 | baseline | INFO |
=== SUMMARY ===
2025-07-17 13:44:00,724 | baseline | INFO | FAILED: Both targets give negative R2
2025-07-17 13:44:00,724 | baseline | INFO |   -> Mail-calls relationship too weak for ML

```python
#!/usr/bin/env python
# baseline_mail_forecast.py
# =========================================================
# BASELINE: Test the exact same relationship the debug found
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
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger("baseline")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | baseline | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def _to_date(s):
    return pd.to_datetime(s, errors="coerce").dt.date

def _find_file(candidates):
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError(f"None found: {candidates}")

def load_and_prepare():
    """Load data exactly like debug script"""
    
    # Load mail
    mail_path = _find_file(["mail.csv", "data/mail.csv"])
    mail = pd.read_csv(mail_path)
    mail.columns = [c.lower().strip() for c in mail.columns]
    mail["mail_date"] = _to_date(mail["mail_date"])
    mail = mail.dropna(subset=["mail_date"])

    # Load calls  
    vol_path = _find_file(["callvolumes.csv", "data/callvolumes.csv"])
    intent_path = _find_file(["callintent.csv", "data/callintent.csv", "callintetn.csv"])

    # Process volumes
    df_vol = pd.read_csv(vol_path)
    df_vol.columns = [c.lower().strip() for c in df_vol.columns]
    dcol_v = next(c for c in df_vol.columns if "date" in c)
    df_vol[dcol_v] = _to_date(df_vol[dcol_v])
    vol_daily = df_vol.groupby(dcol_v)[df_vol.columns.difference([dcol_v])[0]].sum()

    # Process intent
    df_int = pd.read_csv(intent_path)
    df_int.columns = [c.lower().strip() for c in df_int.columns]
    dcol_i = next(c for c in df_int.columns if "date" in c or "conversationstart" in c)
    df_int[dcol_i] = _to_date(df_int[dcol_i])
    int_daily = df_int.groupby(dcol_i).size()

    # Scale and combine
    overlap = vol_daily.index.intersection(int_daily.index)
    if len(overlap) >= 5:
        scale = int_daily.loc[overlap].mean() / vol_daily.loc[overlap].mean()
        vol_daily *= scale
    calls_total = vol_daily.combine_first(int_daily).sort_index()

    # Aggregate mail daily
    mail_daily = (mail.groupby(["mail_date", "mail_type"], as_index=False)["mail_volume"].sum()
                   .pivot(index="mail_date", columns="mail_type", values="mail_volume").fillna(0))
    
    mail_daily.index = pd.to_datetime(mail_daily.index)
    calls_total.index = pd.to_datetime(calls_total.index)

    # Business days only
    us_holidays = holidays.US()
    biz_mask = (~mail_daily.index.weekday.isin([5, 6])) & (~mail_daily.index.isin(us_holidays))
    mail_daily = mail_daily.loc[biz_mask]
    calls_total = calls_total.loc[calls_total.index.isin(mail_daily.index)]

    daily = mail_daily.join(calls_total.rename("calls_total"), how="inner")
    
    LOG.info("Daily data shape: %s", daily.shape)
    return daily

def test_debug_correlations(daily):
    """Test the exact correlations the debug script found"""
    
    # Total mail (like debug script)
    total_mail = daily.iloc[:, :-1].sum(axis=1)
    calls = daily['calls_total']
    
    # Same-day correlation (what debug found working)
    corr_same = calls.corr(total_mail)
    corr_log = calls.corr(np.log1p(total_mail))
    
    LOG.info("Same-day correlations:")
    LOG.info("  Raw: %.3f", corr_same)
    LOG.info("  Log: %.3f", corr_log)
    
    # Test top mail types from debug
    top_types = ["Reject_Ltrs", "Cheque 1099", "Exercise_Converted", "SOI_Confirms", "Exch_chks"]
    
    LOG.info("Top mail type correlations:")
    for mail_type in top_types:
        if mail_type in daily.columns:
            corr = calls.corr(daily[mail_type])
            LOG.info("  %s: %.3f", mail_type, corr)

def create_simple_features(daily):
    """Create the simplest possible features"""
    
    X = pd.DataFrame(index=daily.index)
    
    # SAME-DAY features (what debug found working)
    total_mail = daily.iloc[:, :-1].sum(axis=1)
    X["total_mail_today"] = total_mail
    X["log_total_mail_today"] = np.log1p(total_mail)
    
    # Top 3 mail types same-day
    top_types = ["Reject_Ltrs", "Cheque 1099", "Exercise_Converted"]
    for mail_type in top_types:
        if mail_type in daily.columns:
            X[f"{mail_type}_today"] = daily[mail_type]
    
    # Also try 1-day lag (best lag from debug)
    X["total_mail_lag1"] = total_mail.shift(1)
    X["log_total_mail_lag1"] = np.log1p(total_mail).shift(1)
    
    # Basic time features
    X["weekday"] = daily.index.dayofweek
    X["month"] = daily.index.month
    
    # Target (try both raw and log)
    y_raw = daily["calls_total"]
    y_log = np.log1p(daily["calls_total"])
    
    # Clean
    X = X.dropna()
    y_raw = y_raw.loc[X.index]
    y_log = y_log.loc[X.index]
    
    LOG.info("Features: %d samples x %d features", X.shape[0], X.shape[1])
    return X, y_raw, y_log

def test_simple_model(X, y, target_name):
    """Test simple model like debug script"""
    
    LOG.info("Testing %s target...", target_name)
    
    # Simple Ridge like debug
    cv = TimeSeriesSplit(n_splits=3)
    r2_scores = []
    
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale and fit
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_scaled, y_train)
        
        y_pred = ridge.predict(X_test_scaled)
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)
    
    avg_r2 = np.mean(r2_scores)
    LOG.info("  %s Ridge R2: %.3f (scores: %s)", target_name, avg_r2, [f"{r:.3f}" for r in r2_scores])
    
    return avg_r2

def main():
    LOG.info("=== BASELINE TEST ===")
    
    # Load data exactly like debug
    daily = load_and_prepare()
    
    # Test correlations like debug
    test_debug_correlations(daily)
    
    # Create simple features
    X, y_raw, y_log = create_simple_features(daily)
    
    # Test both raw and log targets
    LOG.info("\nTesting simple models...")
    r2_raw = test_simple_model(X, y_raw, "RAW")
    r2_log = test_simple_model(X, y_log, "LOG")
    
    # Summary
    LOG.info("\n=== SUMMARY ===")
    if r2_raw > 0 or r2_log > 0:
        LOG.info("SUCCESS: Found positive R2!")
        if r2_log > r2_raw:
            LOG.info("  -> Log target works better (R2=%.3f)", r2_log)
        else:
            LOG.info("  -> Raw target works better (R2=%.3f)", r2_raw)
    else:
        LOG.info("FAILED: Both targets give negative R2")
        LOG.info("  -> Mail-calls relationship too weak for ML")
    
    # Feature importance check
    if r2_log > 0:
        LOG.info("\nQuick feature importance test...")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_scaled, y_log)
        
        feature_importance = dict(zip(X.columns, ridge.coef_))
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        LOG.info("Top 5 features:")
        for feat, coef in sorted_features[:5]:
            LOG.info("  %s: %.3f", feat, coef)

if __name__ == "__main__":
    main()
```

**This baseline will tell us:**

1. **Are the correlations the same** as debug script?
1. **Does same-day prediction work** better than lagged?
1. **Raw vs log target** - which actually works?
1. **What features matter** most?

**Expected result**: If this still fails, then the mail→calls relationship is fundamentally too weak for ML prediction, and you should abandon this approach.

Run this and let’s see if we can at least replicate the debug script’s findings!​​​​​​​​​​​​​​​​
