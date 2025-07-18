**Two critical issues here:**

## 🚨 **Issue 1: Still Getting Bad Performance**

- **R² = -0.031** (negative = worse than random)
- **MAE = 7,978** (worse than your original 5,263)

## 🚨 **Issue 2: Feature Name Mismatch Error**

The prediction function creates features in different order than training.

## 🔧 **Quick Fix - Simplified Version**

Let me create a simpler, more robust version that fixes both issues:

```python
#!/usr/bin/env python
# simple_enhanced_model.py
# =========================================================
# SIMPLIFIED enhancement - add economic indicators carefully
# Fix feature ordering and avoid overfitting
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

LOG = logging.getLogger("simple_enhanced")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | simple_enhanced | %(levelname)s | %(message)s",
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
    "output_dir": "simple_enhanced_results"
}

def _to_date(s):
    return pd.to_datetime(s, errors="coerce").dt.date

def _find_file(candidates):
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError(f"None found: {candidates}")

def load_data_simple():
    """Load data the same way as your working model"""
    
    LOG.info("Loading data...")
    
    # Load mail (exactly like your working model)
    mail_path = _find_file(["mail.csv", "data/mail.csv"])
    mail = pd.read_csv(mail_path)
    mail.columns = [c.lower().strip() for c in mail.columns]
    mail["mail_date"] = _to_date(mail["mail_date"])
    mail = mail.dropna(subset=["mail_date"])

    # Load calls (exactly like your working model)
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
    
    # Try to add JUST Russell2000 (best economic indicator)
    try:
        econ_path = Path("expanded_economic_data.csv")
        if econ_path.exists():
            econ_df = pd.read_csv(econ_path, parse_dates=['Date'])
            econ_df.set_index('Date', inplace=True)
            if 'Russell2000' in econ_df.columns:
                daily = daily.join(econ_df[['Russell2000']], how='left')
                daily['Russell2000'] = daily['Russell2000'].fillna(method='ffill').fillna(method='bfill')
                LOG.info("Added Russell2000 economic indicator")
            else:
                LOG.warning("Russell2000 not found in economic data")
        else:
            LOG.warning("Economic data file not found")
    except Exception as e:
        LOG.warning(f"Could not load economic data: {e}")
    
    LOG.info(f"Final data shape: {daily.shape}")
    return daily

def create_simple_features(daily):
    """Create features in EXACT same order every time"""
    
    features_list = []
    targets_list = []
    
    # Get available mail types
    available_types = [t for t in CFG["top_mail_types"] if t in daily.columns]
    has_russell2000 = 'Russell2000' in daily.columns
    
    # CRITICAL: Define feature order once and stick to it
    feature_order = []
    
    # Mail volume features (in exact order)
    for mail_type in available_types:
        feature_order.append(f"{mail_type}_volume")
    
    # Total mail features
    feature_order.extend([
        "total_mail_volume",
        "log_total_mail_volume", 
        "mail_percentile"
    ])
    
    # Date features
    feature_order.extend([
        "weekday",
        "month", 
        "is_month_end",
        "is_holiday_week"
    ])
    
    # Recent calls features
    feature_order.extend([
        "recent_calls_avg",
        "recent_calls_trend"
    ])
    
    # Economic feature (only if available)
    if has_russell2000:
        feature_order.append("Russell2000_lag1")
    
    LOG.info(f"Feature order: {feature_order}")
    
    # Create features in exact order
    for i in range(len(daily) - 1):
        current_day = daily.iloc[i]
        next_day = daily.iloc[i + 1]
        
        feature_row = {}
        
        # Mail volumes
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
        
        # Economic feature (only if available)
        if has_russell2000 and i > 0:
            feature_row["Russell2000_lag1"] = daily.iloc[i-1]["Russell2000"]
        
        # Target
        target = next_day["calls_total"]
        
        features_list.append(feature_row)
        targets_list.append(target)
    
    # Create DataFrame with exact feature order
    X = pd.DataFrame(features_list)
    X = X.reindex(columns=feature_order, fill_value=0)
    y = pd.Series(targets_list)
    
    LOG.info(f"Simple features: {X.shape[0]} samples x {X.shape[1]} features")
    LOG.info(f"Feature columns: {list(X.columns)}")
    
    return X, y, feature_order

def evaluate_simple_model(X, y):
    """Evaluate model with time series cross-validation"""
    
    LOG.info("=== EVALUATING SIMPLE ENHANCED MODEL ===")
    
    # Use same Random Forest as your working model
    rf_model = RandomForestRegressor(
        n_estimators=50,   # Same as your working model
        max_depth=6,       # Same as your working model
        min_samples_leaf=3, # Same as your working model
        random_state=42
    )
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    cv_mae_scores = []
    cv_r2_scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_val)
        
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        cv_mae_scores.append(mae)
        cv_r2_scores.append(r2)
    
    avg_mae = np.mean(cv_mae_scores)
    avg_r2 = np.mean(cv_r2_scores)
    
    LOG.info(f"Cross-Validation Results:")
    LOG.info(f"  Average MAE: {avg_mae:.0f}")
    LOG.info(f"  Average R²: {avg_r2:.3f}")
    
    # Compare with your original model
    LOG.info(f"\nComparison with Original Model:")
    LOG.info(f"  Original: MAE = 5,263, R² = 0.497")
    LOG.info(f"  Enhanced: MAE = {avg_mae:.0f}, R² = {avg_r2:.3f}")
    
    if avg_r2 > 0.497:
        LOG.info(f"  ✓ IMPROVEMENT: R² improved by {((avg_r2-0.497)/0.497)*100:.1f}%")
    else:
        LOG.info(f"  ✗ REGRESSION: R² decreased by {((0.497-avg_r2)/0.497)*100:.1f}%")
    
    # Train final model
    rf_model.fit(X, y)
    
    # Feature importance
    feature_importance = dict(zip(X.columns, rf_model.feature_importances_))
    top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:8]
    
    LOG.info("Top 8 feature importance:")
    for feat, importance in top_features:
        LOG.info(f"  {feat}: {importance:.3f}")
    
    return rf_model, avg_mae, avg_r2

def train_quantile_models(X, y, feature_order):
    """Train quantile models"""
    
    LOG.info("Training quantile models...")
    
    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    models = {}
    
    # Store feature order with models
    models["feature_order"] = feature_order
    
    # Quantile models
    for quantile in CFG["quantiles"]:
        LOG.info("  Training %d%% quantile model...", int(quantile * 100))
        
        model = QuantileRegressor(quantile=quantile, alpha=0.1, solver='highs')
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        models[f"quantile_{quantile}"] = model
        LOG.info("    Validation MAE: %.0f", mae)
    
    # Bootstrap ensemble
    LOG.info("  Training bootstrap ensemble...")
    bootstrap_models = []
    
    for i in range(CFG["bootstrap_samples"]):
        sample_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train.iloc[sample_idx]
        y_boot = y_train.iloc[sample_idx]
        
        model = RandomForestRegressor(
            n_estimators=50,
            max_depth=6,
            min_samples_leaf=3,
            random_state=i
        )
        model.fit(X_boot, y_boot)
        bootstrap_models.append(model)
    
    models["bootstrap_ensemble"] = bootstrap_models
    
    return models

def predict_simple_enhanced(models, mail_inputs, russell2000_value=None, date_str=None):
    """Prediction function with guaranteed feature order"""
    
    if date_str is None:
        predict_date = datetime.now() + timedelta(days=1)
    else:
        predict_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Get feature order from models
    feature_order = models["feature_order"]
    
    # Create feature vector in exact order
    feature_row = {}
    
    # Mail volumes
    total_mail = 0
    for mail_type in CFG["top_mail_types"]:
        volume = mail_inputs.get(mail_type, 0)
        feature_row[f"{mail_type}_volume"] = volume
        total_mail += volume
    
    # Total mail features
    feature_row["total_mail_volume"] = total_mail
    feature_row["log_total_mail_volume"] = np.log1p(total_mail)
    feature_row["mail_percentile"] = 0.5
    
    # Date features
    feature_row["weekday"] = predict_date.weekday()
    feature_row["month"] = predict_date.month
    feature_row["is_month_end"] = 1 if predict_date.day > 25 else 0
    feature_row["is_holiday_week"] = 1 if predict_date.date() in holidays.US() else 0
    
    # Recent calls (defaults)
    feature_row["recent_calls_avg"] = 15000
    feature_row["recent_calls_trend"] = 0
    
    # Economic feature
    if "Russell2000_lag1" in feature_order:
        feature_row["Russell2000_lag1"] = russell2000_value if russell2000_value else 2000
    
    # Create DataFrame in exact feature order
    X_input = pd.DataFrame([feature_row])
    X_input = X_input.reindex(columns=feature_order, fill_value=0)
    
    # Get predictions
    quantile_preds = {}
    for quantile in CFG["quantiles"]:
        model = models[f"quantile_{quantile}"]
        pred = model.predict(X_input)[0]
        quantile_preds[f"q{int(quantile*100)}"] = max(0, pred)
    
    # Bootstrap predictions
    bootstrap_preds = []
    for model in models["bootstrap_ensemble"]:
        pred = model.predict(X_input)[0]
        bootstrap_preds.append(max(0, pred))
    
    bootstrap_stats = {
        "mean": np.mean(bootstrap_preds),
        "std": np.std(bootstrap_preds),
        "min": np.min(bootstrap_preds),
        "max": np.max(bootstrap_preds)
    }
    
    return quantile_preds, bootstrap_stats

def main():
    output_dir = Path(CFG["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    LOG.info("=== SIMPLE ENHANCED MODEL ===")
    
    # Load data
    daily = load_data_simple()
    
    # Create features
    X, y, feature_order = create_simple_features(daily)
    
    # Evaluate model
    rf_model, avg_mae, avg_r2 = evaluate_simple_model(X, y)
    
    # Train quantile models
    models = train_quantile_models(X, y, feature_order)
    
    # Save models
    joblib.dump(models, output_dir / "simple_enhanced_models.pkl")
    
    # Test example
    LOG.info("\n=== TESTING EXAMPLE ===")
    
    example_mail = {"Reject_Ltrs": 1500, "Cheque 1099": 800}
    russell_value = 2050
    
    LOG.info(f"Example input: {example_mail}")
    LOG.info(f"Russell2000: {russell_value}")
    
    quantile_preds, bootstrap_stats = predict_simple_enhanced(models, example_mail, russell_value)
    
    LOG.info("Predictions:")
    LOG.info("  Most likely: %.0f calls", quantile_preds["q50"])
    LOG.info("  Business range (25-75%%): %.0f - %.0f calls", 
            quantile_preds["q25"], quantile_preds["q75"])
    
    LOG.info(f"\nSimple enhanced model complete! Results: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
```

**Key fixes:**

1. **Simplified approach**: Only adds Russell2000 (best economic indicator)
1. **Fixed feature ordering**: Creates features in exact same order every time
1. **Conservative enhancement**: Uses same Random Forest parameters as your working model
1. **Proper feature alignment**: Guarantees prediction features match training features

**This should work without errors and hopefully improve your R² from 0.497 to ~0.52-0.55!**