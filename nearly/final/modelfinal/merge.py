## 🚨 **Two Critical Issues Here:**

### **Issue 1: Performance Got WORSE**

- **Original**: MAE = 5,263, R² = 0.497 ✅
- **Enhanced**: MAE = 8,421, R² = -0.110 ❌
- **122% REGRESSION!** The model is now terrible

### **Issue 2: NaN Values Error**

```
ValueError: Input X contains NaN.
```

The economic data has missing values that aren’t being handled properly.

## 🔍 **Root Cause Analysis:**

### **Why Performance Got Worse:**

1. **Data quality issues** - NaN values in Russell2000 data
1. **Feature misalignment** - Economic data dates don’t match mail dates
1. **Overfitting** - Adding noisy economic data hurt the model

### **Why Economic Correlation Doesn’t Help:**

The correlation analysis was **flawed** - it mixed economic indicators with mail types. The “high correlation” might be misleading.

## 🎯 **Back to Basics - Let’s Fix This:**

```python
#!/usr/bin/env python
# back_to_working_model.py
# =========================================================
# GO BACK TO YOUR WORKING MODEL + Careful economic addition
# =========================================================

from pathlib import Path
import json
import logging
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import holidays
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings('ignore')

LOG = logging.getLogger("back_to_working")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | back_to_working | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

CFG = {
    "top_mail_types": [
        "Reject_Ltrs", "Cheque 1099", "Exercise_Converted",
        "SOI_Confirms", "Exch_chks", "ACH_Debit_Enrollment",
        "Transfer", "COA", "NOTC_WITHDRAW", "Repl_Chks"
    ],
    "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
    "bootstrap_samples": 30,
    "output_dir": "back_to_working_results"
}

def _to_date(s):
    return pd.to_datetime(s, errors="coerce").dt.date

def _find_file(candidates):
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError(f"None found: {candidates}")

def load_mail_call_data_clean():
    """Load ONLY mail and calls data - exactly like your working model"""
    
    LOG.info("Loading mail and calls data (working model approach)...")
    
    # EXACT same code as your working model
    mail_path = _find_file(["mail.csv", "data/mail.csv"])
    mail = pd.read_csv(mail_path)
    mail.columns = [c.lower().strip() for c in mail.columns]
    mail["mail_date"] = _to_date(mail["mail_date"])
    mail = mail.dropna(subset=["mail_date"])

    vol_path = _find_file(["callvolumes.csv", "data/callvolumes.csv"])
    intent_path = _find_file(["callintent.csv", "data/callintent.csv", "callintetn.csv"])

    df_vol = pd.read_csv(vol_path)
    df_vol.columns = [c.lower().strip() for c in df_vol.columns]
    dcol_v = next(c for c in df_vol.columns if "date" in c)
    df_vol[dcol_v] = _to_date(df_vol[dcol_v])
    vol_daily = df_vol.groupby(dcol_v)[df_vol.columns.difference([dcol_v])[0]].sum()

    df_int = pd.read_csv(intent_path)
    df_int.columns = [c.lower().strip() for c in df_int.columns]
    dcol_i = next(c for c in df_int.columns if "date" in c or "conversationstart" in c)
    df_int[dcol_i] = _to_date(df_int[dcol_i])
    int_daily = df_int.groupby(dcol_i).size()

    overlap = vol_daily.index.intersection(int_daily.index)
    if len(overlap) >= 5:
        scale = int_daily.loc[overlap].mean() / vol_daily.loc[overlap].mean()
        vol_daily *= scale
        LOG.info("Scaled call volumes by factor: %.2f", scale)
    calls_total = vol_daily.combine_first(int_daily).sort_index()

    mail_daily = (mail.groupby(["mail_date", "mail_type"], as_index=False)["mail_volume"].sum()
                   .pivot(index="mail_date", columns="mail_type", values="mail_volume").fillna(0))
    
    mail_daily.index = pd.to_datetime(mail_daily.index)
    calls_total.index = pd.to_datetime(calls_total.index)

    us_holidays = holidays.US()
    biz_mask = (~mail_daily.index.weekday.isin([5, 6])) & (~mail_daily.index.isin(us_holidays))
    mail_daily = mail_daily.loc[biz_mask]
    calls_total = calls_total.loc[calls_total.index.isin(mail_daily.index)]

    daily = mail_daily.join(calls_total.rename("calls_total"), how="inner")
    
    LOG.info("Clean data shape: %s", daily.shape)
    return daily

def test_economic_data_quality():
    """Test if economic data is actually usable"""
    
    LOG.info("=== TESTING ECONOMIC DATA QUALITY ===")
    
    try:
        # Load economic data
        econ_path = Path("expanded_economic_data.csv")
        if not econ_path.exists():
            LOG.warning("Economic data file not found")
            return None
        
        econ_df = pd.read_csv(econ_path, parse_dates=['Date'])
        econ_df.set_index('Date', inplace=True)
        
        LOG.info(f"Economic data shape: {econ_df.shape}")
        LOG.info(f"Economic data columns: {list(econ_df.columns)}")
        
        # Check for NaN values
        nan_counts = econ_df.isnull().sum()
        LOG.info("NaN counts per column:")
        for col, nan_count in nan_counts.items():
            LOG.info(f"  {col}: {nan_count} NaNs ({nan_count/len(econ_df)*100:.1f}%)")
        
        # Check date range
        LOG.info(f"Economic data date range: {econ_df.index.min()} to {econ_df.index.max()}")
        
        # Test specific indicators
        test_indicators = ["Russell2000", "SP500", "NASDAQ", "Dollar_Index"]
        available_indicators = []
        
        for indicator in test_indicators:
            if indicator in econ_df.columns:
                non_null_pct = (1 - econ_df[indicator].isnull().sum() / len(econ_df)) * 100
                LOG.info(f"{indicator}: {non_null_pct:.1f}% data available")
                if non_null_pct > 90:  # Only use if >90% data available
                    available_indicators.append(indicator)
        
        LOG.info(f"Usable economic indicators: {available_indicators}")
        return econ_df, available_indicators
        
    except Exception as e:
        LOG.error(f"Error loading economic data: {e}")
        return None

def create_baseline_features(daily):
    """Create features EXACTLY like your working model"""
    
    LOG.info("Creating baseline features (working model approach)...")
    
    features_list = []
    targets_list = []
    
    # EXACT same feature creation as your working model
    for i in range(len(daily) - 1):
        current_day = daily.iloc[i]
        next_day = daily.iloc[i + 1]
        
        feature_row = {}
        
        # Mail volumes
        available_types = [t for t in CFG["top_mail_types"] if t in daily.columns]
        for mail_type in available_types:
            feature_row[f"{mail_type}_volume"] = current_day[mail_type]
        
        # Total mail volume
        total_mail = sum(current_day[t] for t in available_types)
        feature_row["total_mail_volume"] = total_mail
        feature_row["log_total_mail_volume"] = np.log1p(total_mail)
        
        # Mail percentile
        mail_history = daily[available_types].sum(axis=1).iloc[:i+1]
        if len(mail_history) > 10:
            feature_row["mail_percentile"] = (mail_history <= total_mail).mean()
        else:
            feature_row["mail_percentile"] = 0.5
        
        # Date features
        current_date = daily.index[i]
        feature_row["weekday"] = current_date.weekday()
        feature_row["month"] = current_date.month
        feature_row["is_month_end"] = 1 if current_date.day > 25 else 0
        feature_row["is_holiday_week"] = 1 if current_date in holidays.US() else 0
        
        # Recent calls
        recent_calls = daily["calls_total"].iloc[max(0, i-5):i+1]
        feature_row["recent_calls_avg"] = recent_calls.mean()
        feature_row["recent_calls_trend"] = recent_calls.diff().mean()
        
        # Target
        target = next_day["calls_total"]
        
        features_list.append(feature_row)
        targets_list.append(target)
    
    X = pd.DataFrame(features_list)
    y = pd.Series(targets_list)
    X = X.fillna(0)
    
    LOG.info("Baseline features: %d samples x %d features", X.shape[0], X.shape[1])
    return X, y

def test_baseline_model(X, y):
    """Test that we can reproduce your working model results"""
    
    LOG.info("=== TESTING BASELINE MODEL ===")
    
    # EXACT same model as your working model
    rf_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42
    )
    
    # Simple train/test split
    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    LOG.info(f"Baseline Model Results:")
    LOG.info(f"  MAE: {mae:.0f}")
    LOG.info(f"  R²: {r2:.3f}")
    
    # Compare with your original
    LOG.info(f"\nComparison with Your Original Working Model:")
    LOG.info(f"  Expected: MAE = 5,263, R² = 0.497")
    LOG.info(f"  Actual:   MAE = {mae:.0f}, R² = {r2:.3f}")
    
    if abs(mae - 5263) < 1000 and abs(r2 - 0.497) < 0.1:
        LOG.info("  ✓ BASELINE REPRODUCED SUCCESSFULLY")
        return True, rf_model, mae, r2
    else:
        LOG.warning("  ✗ BASELINE NOT REPRODUCED - Something is different")
        return False, rf_model, mae, r2

def main():
    output_dir = Path(CFG["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    LOG.info("=== BACK TO WORKING MODEL ===")
    
    # Step 1: Test economic data quality
    econ_result = test_economic_data_quality()
    
    # Step 2: Load clean mail/calls data
    daily = load_mail_call_data_clean()
    
    # Step 3: Create baseline features
    X, y = create_baseline_features(daily)
    
    # Step 4: Test baseline model
    baseline_ok, rf_model, mae, r2 = test_baseline_model(X, y)
    
    if baseline_ok:
        LOG.info("\n=== BASELINE WORKING - CAN PROCEED ===")
        LOG.info("Your working model has been successfully reproduced.")
        LOG.info("Next steps:")
        LOG.info("1. Fix economic data quality issues")
        LOG.info("2. Add economic indicators ONE AT A TIME")
        LOG.info("3. Test each addition carefully")
    else:
        LOG.error("\n=== BASELINE NOT WORKING ===")
        LOG.error("Cannot reproduce your original working model.")
        LOG.error("Need to debug why baseline is different.")
    
    # Save baseline model
    joblib.dump(rf_model, output_dir / "baseline_model.pkl")
    
    # Save results
    with open(output_dir / "baseline_results.json", "w") as f:
        json.dump({
            "baseline_mae": mae,
            "baseline_r2": r2,
            "baseline_reproduced": baseline_ok,
            "economic_data_available": econ_result is not None
        }, f, indent=2)
    
    LOG.info(f"\nResults saved to: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
```

## 🎯 **What This Script Will Do:**

1. **Reproduce your working model** (should get MAE ≈ 5,263, R² ≈ 0.497)
1. **Diagnose economic data quality** (check for NaN values, date mismatches)
1. **Identify the root cause** of why adding economic data hurt performance

**Run this first - let’s make sure we can get back to your working baseline before trying to enhance it!**