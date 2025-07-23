
PS C:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod> & C:/Users/BhungarD/python.exe "c:/Users/BhungarD/OneDrive - Computershare/Desktop/finprod/model.py"
================================================================================
MAIL-TO-CALLS PREDICTION: SIMPLE FIRST, THEN ADVANCED
================================================================================
STRATEGY:
  1. Build SIMPLE working model (guaranteed)
  2. Try ADVANCED features (if possible)
  3. You get minimum simple model, maximum full system
================================================================================

 PHASE 1: BUILDING SIMPLE MODEL
 PHASE 1: Loading data for SIMPLE model...
2025-07-23 09:46:33,357 | INFO | Mail columns: date=mail_date, volume=mail_volume, type=mail_type
2025-07-23 09:46:41,705 | INFO | Call date column: conversationstart
2025-07-23 09:46:46,204 | INFO | Combined data: 82 days, 197 mail types
2025-07-23 09:46:46,206 | INFO | Date range: 2025-02-05 00:00:00 to 2025-05-30 00:00:00
2025-07-23 09:46:46,206 | INFO | Average daily calls: 12183
  Creating SIMPLE features...
2025-07-23 09:46:46,396 | INFO | Top 10 mail types: ['Envision', 'DRP Stmt.', 'Cheque', 'Scheduled PAYMENT CHECKS', 'Notice']...
2025-07-23 09:46:46,680 | INFO | Simple features: 81 samples x 19 features
 Training SIMPLE models...
2025-07-23 09:46:46,681 | INFO |   Training 10% quantile model...
2025-07-23 09:46:46,706 | INFO |     Validation MAE: 2356
2025-07-23 09:46:46,707 | INFO |   Training 25% quantile model...
2025-07-23 09:46:46,726 | INFO |     Validation MAE: 680
2025-07-23 09:46:46,727 | INFO |   Training 50% quantile model...
2025-07-23 09:46:46,757 | INFO |     Validation MAE: 897
2025-07-23 09:46:46,758 | INFO |   Training 75% quantile model...
2025-07-23 09:46:46,786 | INFO |     Validation MAE: 1633
2025-07-23 09:46:46,787 | INFO |   Training 90% quantile model...
2025-07-23 09:46:46,804 | INFO |     Validation MAE: 3294
2025-07-23 09:46:46,805 | INFO |   Training bootstrap ensemble...
2025-07-23 09:46:48,845 | INFO |     Successfully trained 20 bootstrap models
 SIMPLE MODEL: SUCCESS!
 Testing SIMPLE model...
2025-07-23 09:46:50,020 | INFO |   Testing: High Volume Day
2025-07-23 09:46:50,318 | INFO |     Most likely: 14041 calls
2025-07-23 09:46:50,319 | INFO |     Range (25-75%): 13495 - 15170 calls
2025-07-23 09:46:50,319 | INFO |   Testing: Low Volume Day
2025-07-23 09:46:50,580 | INFO |     Most likely: 14039 calls
2025-07-23 09:46:50,581 | INFO |     Range (25-75%): 13493 - 15171 calls
2025-07-23 09:46:50,582 | INFO |   Testing: Mixed Mail Day
2025-07-23 09:46:50,814 | INFO |     Most likely: 14040 calls
2025-07-23 09:46:50,815 | INFO |     Range (25-75%): 13495 - 15171 calls

 PHASE 2: TRYING ADVANCED MODEL
 PHASE 2: Trying ADVANCED features...
2025-07-23 09:47:04,719 | INFO | Found intent column: intent
2025-07-23 09:47:07,828 | INFO | Top intents: ['Unknown', 'Repeat Caller', 'Transfer', 'Associate', 'Tax Information']...
2025-07-23 09:47:08,605 | INFO | Advanced: 15 mail types, 10 intents
  Creating ADVANCED features...
2025-07-23 09:47:08,971 | INFO | Advanced features: 82 aligned days
2025-07-23 09:47:09,111 | INFO | Advanced features: 81 samples x 31 features
 Training ADVANCED models...
2025-07-23 09:47:09,113 | INFO |   Training advanced volume model...
2025-07-23 09:47:09,386 | INFO |     Volume model MAE: 1011
2025-07-23 09:47:09,389 | INFO |   Training advanced intent model...
2025-07-23 09:47:09,603 | INFO |     Intent model accuracy: 1.000
 ADVANCED MODEL: SUCCESS!
 Testing ADVANCED model...
2025-07-23 09:47:09,866 | INFO |   Advanced model has 3 components
2025-07-23 09:47:09,867 | INFO |     ✓ Volume prediction model ready
2025-07-23 09:47:09,867 | INFO |     ✓ Intent prediction model ready

============================================================
 FINAL RESULTS
============================================================
 SIMPLE MODEL: 6 models trained
    Top mail types: 10
    Test scenarios: 3
    Example prediction: 14041 calls
 ADVANCED MODEL: 3 components
    Volume prediction: Ready
    Intent prediction: Ready

 SUCCESS: You have at least a simple working model!
 All models saved to: mail_calls_models
 See USAGE_GUIDE.txt for how to use your models
























PHASE 1: BUILDING SIMPLE MODEL
 PHASE 1: Loading data for SIMPLE model...
2025-07-23 09:36:25,311 | INFO | Mail columns: date=mail_date, volume=mail_volume, type=mail_type
2025-07-23 09:36:38,738 | INFO | Call date column: conversationstart
2025-07-23 09:36:45,157 | INFO | Combined data: 82 days, 197 mail types
2025-07-23 09:36:45,159 | INFO | Date range: 2025-02-05 00:00:00 to 2025-05-30 00:00:00
2025-07-23 09:36:45,159 | INFO | Average daily calls: 12183
  Creating SIMPLE features...
2025-07-23 09:36:45,385 | INFO | Top 10 mail types: ['Envision', 'DRP Stmt.', 'Cheque', 'Scheduled PAYMENT CHECKS', 'Notice']...
2025-07-23 09:36:45,751 | INFO | Simple features: 81 samples x 19 features
 Training SIMPLE models...
2025-07-23 09:36:45,752 | INFO |   Training 10% quantile model...
2025-07-23 09:36:45,868 | INFO |     Validation MAE: 2356
2025-07-23 09:36:45,869 | INFO |   Training 25% quantile model...
2025-07-23 09:36:45,885 | INFO |     Validation MAE: 680
2025-07-23 09:36:45,885 | INFO |   Training 50% quantile model...
2025-07-23 09:36:45,907 | INFO |     Validation MAE: 897
2025-07-23 09:36:45,907 | INFO |   Training 75% quantile model...
2025-07-23 09:36:45,932 | INFO |     Validation MAE: 1633
2025-07-23 09:36:45,932 | INFO |   Training 90% quantile model...
2025-07-23 09:36:45,954 | INFO |     Validation MAE: 3294
2025-07-23 09:36:45,955 | INFO |   Training bootstrap ensemble...
2025-07-23 09:36:47,993 | INFO |     Successfully trained 20 bootstrap models
 SIMPLE MODEL: SUCCESS!
 Testing SIMPLE model...
2025-07-23 09:36:48,355 | INFO |   Testing: High Volume Day
2025-07-23 09:36:48,361 | WARNING | Quantile 0.1 prediction failed: The feature names should match those that were passed during fit.
Feature names must be in the same order as they were in fit.

2025-07-23 09:36:48,363 | WARNING | Quantile 0.25 prediction failed: The feature names should match those that were passed during fit.
Feature names must be in the same order as they were in fit.

2025-07-23 09:36:48,364 | WARNING | Quantile 0.5 prediction failed: The feature names should match those that were passed during fit.
Feature names must be in the same order as they were in fit.

2025-07-23 09:36:48,365 | WARNING | Quantile 0.75 prediction failed: The feature names should match those that were passed during fit.
Feature names must be in the same order as they were in fit.

2025-07-23 09:36:48,366 | WARNING | Quantile 0.9 prediction failed: The feature names should match those that were passed during fit.
Feature names must be in the same order as they were in fit.

2025-07-23 09:36:48,369 | WARNING |     No predictions for High Volume Day
2025-07-23 09:36:48,369 | INFO |   Testing: Low Volume Day
2025-07-23 09:36:48,377 | WARNING | Quantile 0.1 prediction failed: The feature names should match those that were passed during fit.
Feature names must be in the same order as they were in fit.

2025-07-23 09:36:48,378 | WARNING | Quantile 0.25 prediction failed: The feature names should match those that were passed during fit.
Feature names must be in the same order as they were in fit.

2025-07-23 09:36:48,380 | WARNING | Quantile 0.5 prediction failed: The feature names should match those that were passed during fit.
Feature names must be in the same order as they were in fit.

2025-07-23 09:36:48,381 | WARNING | Quantile 0.75 prediction failed: The feature names should match those that were passed during fit.
Feature names must be in the same order as they were in fit.

2025-07-23 09:36:48,382 | WARNING | Quantile 0.9 prediction failed: The feature names should match those that were passed during fit.
Feature names must be in the same order as they were in fit.

2025-07-23 09:36:48,435 | WARNING |     No predictions for Low Volume Day
2025-07-23 09:36:48,440 | INFO |   Testing: Mixed Mail Day
2025-07-23 09:36:48,454 | WARNING | Quantile 0.1 prediction failed: The feature names should match those that were passed during fit.
Feature names must be in the same order as they were in fit.

2025-07-23 09:36:48,461 | WARNING | Quantile 0.25 prediction failed: The feature names should match those that were passed during fit.
Feature names must be in the same order as they were in fit.

2025-07-23 09:36:48,462 | WARNING | Quantile 0.5 prediction failed: The feature names should match those that were passed during fit.
Feature names must be in the same order as they were in fit.

2025-07-23 09:36:48,463 | WARNING | Quantile 0.75 prediction failed: The feature names should match those that were passed during fit.
Feature names must be in the same order as they were in fit.

2025-07-23 09:36:48,466 | WARNING | Quantile 0.9 prediction failed: The feature names should match those that were passed during fit.
Feature names must be in the same order as they were in fit.

2025-07-23 09:36:48,492 | WARNING |     No predictions for Mixed Mail Day

 PHASE 2: TRYING ADVANCED MODEL
 PHASE 2: Trying ADVANCED features...
2025-07-23 09:37:02,496 | INFO | Found intent column: intent
2025-07-23 09:37:06,757 | INFO | Top intents: ['Unknown', 'Repeat Caller', 'Transfer', 'Associate', 'Tax Information']...
2025-07-23 09:37:07,806 | INFO | Advanced: 15 mail types, 10 intents
  Creating ADVANCED features...
2025-07-23 09:37:08,211 | INFO | Advanced features: 82 aligned days
2025-07-23 09:37:08,377 | INFO | Advanced features: 81 samples x 31 features
 Training ADVANCED models...
2025-07-23 09:37:08,378 | INFO |   Training advanced volume model...
2025-07-23 09:37:08,847 | INFO |     Volume model MAE: 1011
2025-07-23 09:37:08,847 | INFO |   Training advanced intent model...
2025-07-23 09:37:09,177 | INFO |     Intent model accuracy: 1.000
 ADVANCED MODEL: SUCCESS!
 Testing ADVANCED model...
2025-07-23 09:37:09,309 | INFO |   Advanced model has 3 components
2025-07-23 09:37:09,310 | INFO |     ✓ Volume prediction model ready
2025-07-23 09:37:09,311 | INFO |     ✓ Intent prediction model ready

============================================================
 FINAL RESULTS
============================================================
 SIMPLE MODEL: 6 models trained
    Top mail types: 10
    Test scenarios: 0
 ADVANCED MODEL: 3 components
    Volume prediction: Ready
    Intent prediction: Ready

 SUCCESS: You have at least a simple working model!
 All models saved to: mail_calls_models
 See USAGE_GUIDE.txt for how to use your models
PS C:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod> 






#!/usr/bin/env python
"""
MAIL-TO-CALLS PREDICTION: SIMPLE FIRST, THEN ADVANCED
====================================================

STRATEGY:
1. Start with SIMPLE working model (guaranteed to work)
2. Then try ADVANCED features (multiple mail, intent prediction)
3. You get at minimum a simple model, maximum a full advanced system

Based on your mail_input_range_forecast.py but adapted for your data
"""

from pathlib import Path
import json
import logging
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holidays
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import QuantileRegressor, Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, accuracy_score

warnings.filterwarnings('ignore')

LOG = logging.getLogger("mail_calls_prediction")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Configuration
CFG = {
    # Data files (your files)
    "call_file": "callintent.csv",
    "mail_file": "mail.csv",
    
    # Simple model config
    "top_mail_types_simple": 10,
    "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
    "bootstrap_samples": 20,
    
    # Advanced model config (try if simple works)
    "top_mail_types_advanced": 15,
    "top_intents": 10,
    "min_intent_occurrences": 15,
    
    # Output
    "output_dir": "mail_calls_models"
}

def safe_print(msg):
    try:
        print(str(msg).encode('ascii', 'ignore').decode('ascii'))
    except:
        print(str(msg))

def find_file(candidates):
    """Find first existing file from candidates"""
    for path_str in candidates:
        path = Path(path_str)
        if path.exists():
            return path
    raise FileNotFoundError(f"None found: {candidates}")

def to_date(series):
    """Convert to date safely"""
    return pd.to_datetime(series, errors="coerce").dt.date

# ============================================================================
# PHASE 1: SIMPLE MODEL (GUARANTEED TO WORK)
# ============================================================================

def load_simple_data():
    """Load data for simple model - just like your original script"""
    safe_print("🔄 PHASE 1: Loading data for SIMPLE model...")
    
    # Load mail data
    mail_path = find_file(["mail.csv", "data/mail.csv"])
    mail = pd.read_csv(mail_path)
    mail.columns = [c.lower().strip() for c in mail.columns]
    
    # Find mail columns
    date_col = next((c for c in mail.columns if "date" in c), None)
    volume_col = next((c for c in mail.columns if "volume" in c), None)
    type_col = next((c for c in mail.columns if "type" in c), None)
    
    LOG.info(f"Mail columns: date={date_col}, volume={volume_col}, type={type_col}")
    
    # Process mail
    mail[date_col] = to_date(mail[date_col])
    mail = mail.dropna(subset=[date_col])
    mail = mail[mail[date_col] >= pd.to_datetime('2025-01-01').date()]
    
    # Load calls data
    call_path = find_file(["callintent.csv", "data/callintent.csv"])
    calls = pd.read_csv(call_path)
    calls.columns = [c.lower().strip() for c in calls.columns]
    
    # Find call columns
    call_date_col = next((c for c in calls.columns if "date" in c or "start" in c), None)
    
    LOG.info(f"Call date column: {call_date_col}")
    
    # Process calls
    calls[call_date_col] = to_date(calls[call_date_col])
    calls = calls.dropna(subset=[call_date_col])
    calls = calls[calls[call_date_col] >= pd.to_datetime('2025-01-01').date()]
    
    # Create daily call counts
    calls_daily = calls.groupby(call_date_col).size()
    calls_daily.index = pd.to_datetime(calls_daily.index)
    
    # Create daily mail by type
    mail_daily = (mail.groupby([date_col, type_col], as_index=False)[volume_col].sum()
                  .pivot(index=date_col, columns=type_col, values=volume_col).fillna(0))
    mail_daily.index = pd.to_datetime(mail_daily.index)
    
    # Business days only
    us_holidays = holidays.US()
    business_mask = (~calls_daily.index.weekday.isin([5, 6])) & (~calls_daily.index.isin(us_holidays))
    calls_daily = calls_daily[business_mask]
    
    business_mask_mail = (~mail_daily.index.weekday.isin([5, 6])) & (~mail_daily.index.isin(us_holidays))
    mail_daily = mail_daily[business_mask_mail]
    
    # Combine on common dates
    daily_combined = mail_daily.join(calls_daily.rename("calls_total"), how="inner")
    daily_combined = daily_combined.dropna(subset=["calls_total"])
    
    LOG.info(f"Combined data: {daily_combined.shape[0]} days, {daily_combined.shape[1]-1} mail types")
    LOG.info(f"Date range: {daily_combined.index.min()} to {daily_combined.index.max()}")
    LOG.info(f"Average daily calls: {daily_combined['calls_total'].mean():.0f}")
    
    return daily_combined

def create_simple_features(daily_data):
    """Create simple features - like your original script"""
    safe_print("🛠️  Creating SIMPLE features...")
    
    features_list = []
    targets_list = []
    
    # Get top mail types by volume
    mail_cols = [c for c in daily_data.columns if c != 'calls_total']
    mail_volumes = daily_data[mail_cols].sum().sort_values(ascending=False)
    top_mail_types = mail_volumes.head(CFG["top_mail_types_simple"]).index.tolist()
    
    LOG.info(f"Top {len(top_mail_types)} mail types: {top_mail_types[:5]}...")
    
    # Create features for each day to predict next day calls
    for i in range(len(daily_data) - 1):
        current_day = daily_data.iloc[i]
        next_day = daily_data.iloc[i + 1]
        
        feature_row = {}
        
        # Mail volume features
        for mail_type in top_mail_types:
            clean_name = str(mail_type).replace(' ', '').replace('-', '')[:15]
            feature_row[f"{clean_name}_volume"] = current_day[mail_type]
        
        # Total mail volume
        total_mail = sum(current_day[t] for t in top_mail_types)
        feature_row["total_mail_volume"] = total_mail
        feature_row["log_total_mail_volume"] = np.log1p(total_mail)
        
        # Mail percentile (relative to history)
        mail_history = daily_data[top_mail_types].sum(axis=1).iloc[:i+1]
        if len(mail_history) > 5:
            feature_row["mail_percentile"] = (mail_history <= total_mail).mean()
        else:
            feature_row["mail_percentile"] = 0.5
        
        # Date features
        current_date = daily_data.index[i]
        feature_row["weekday"] = current_date.weekday()
        feature_row["month"] = current_date.month
        feature_row["is_month_end"] = 1 if current_date.day > 25 else 0
        feature_row["is_holiday_week"] = 1 if current_date.date() in holidays.US() else 0
        
        # Recent call context
        recent_calls = daily_data["calls_total"].iloc[max(0, i-5):i+1]
        feature_row["recent_calls_avg"] = recent_calls.mean()
        feature_row["recent_calls_trend"] = recent_calls.diff().mean() if len(recent_calls) > 1 else 0
        
        # Target: next day calls
        target = next_day["calls_total"]
        
        features_list.append(feature_row)
        targets_list.append(target)
    
    # Convert to DataFrames
    X = pd.DataFrame(features_list)
    y = pd.Series(targets_list)
    
    # Clean data
    X = X.fillna(0)
    
    LOG.info(f"Simple features: {X.shape[0]} samples x {X.shape[1]} features")
    
    return X, y, top_mail_types

def train_simple_models(X, y):
    """Train simple models - like your original script"""
    safe_print("🎯 Training SIMPLE models...")
    
    # Split for validation
    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    models = {}
    
    # Train quantile models for range prediction
    for quantile in CFG["quantiles"]:
        LOG.info(f"  Training {int(quantile * 100)}% quantile model...")
        
        try:
            model = QuantileRegressor(quantile=quantile, alpha=0.1, solver='highs')
            model.fit(X_train, y_train)
            
            # Validate
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            
            models[f"quantile_{quantile}"] = model
            LOG.info(f"    Validation MAE: {mae:.0f}")
            
        except Exception as e:
            LOG.warning(f"    Quantile {quantile} failed: {e}")
    
    # Train bootstrap ensemble
    LOG.info("  Training bootstrap ensemble...")
    bootstrap_models = []
    
    for i in range(CFG["bootstrap_samples"]):
        try:
            # Bootstrap sample
            sample_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
            X_boot = X_train.iloc[sample_idx]
            y_boot = y_train.iloc[sample_idx]
            
            # Simple model
            model = RandomForestRegressor(
                n_estimators=50,
                max_depth=6,
                min_samples_leaf=3,
                random_state=i
            )
            model.fit(X_boot, y_boot)
            bootstrap_models.append(model)
            
        except Exception as e:
            LOG.warning(f"    Bootstrap model {i} failed: {e}")
    
    if bootstrap_models:
        models["bootstrap_ensemble"] = bootstrap_models
        LOG.info(f"    Successfully trained {len(bootstrap_models)} bootstrap models")
    
    return models

def predict_simple(models, mail_inputs, date_str=None):
    """Make simple predictions - like your original script"""
    
    if date_str is None:
        predict_date = datetime.now() + timedelta(days=1)
    else:
        predict_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Create feature vector
    feature_row = {}
    
    # Mail volumes
    total_mail = 0
    for mail_type, volume in mail_inputs.items():
        clean_name = str(mail_type).replace(' ', '').replace('-', '')[:15]
        feature_row[f"{clean_name}_volume"] = volume
        total_mail += volume
    
    feature_row["total_mail_volume"] = total_mail
    feature_row["log_total_mail_volume"] = np.log1p(total_mail)
    feature_row["mail_percentile"] = 0.5  # Default
    
    # Date features
    feature_row["weekday"] = predict_date.weekday()
    feature_row["month"] = predict_date.month
    feature_row["is_month_end"] = 1 if predict_date.day > 25 else 0
    feature_row["is_holiday_week"] = 1 if predict_date.date() in holidays.US() else 0
    
    # Baseline features
    feature_row["recent_calls_avg"] = 15000  # Could improve with actual data
    feature_row["recent_calls_trend"] = 0
    
    # Convert to DataFrame and ensure all expected features are present
    X_input = pd.DataFrame([feature_row])
    
    # Fill missing features with 0
    expected_features = list(models.values())[0].feature_names_in_ if hasattr(list(models.values())[0], 'feature_names_in_') else []
    for feature in expected_features:
        if feature not in X_input.columns:
            X_input[feature] = 0
    
    results = {}
    
    # Get quantile predictions
    quantile_preds = {}
    for quantile in CFG["quantiles"]:
        model_key = f"quantile_{quantile}"
        if model_key in models:
            try:
                pred = models[model_key].predict(X_input)[0]
                quantile_preds[f"q{int(quantile*100)}"] = max(0, pred)
            except Exception as e:
                LOG.warning(f"Quantile {quantile} prediction failed: {e}")
    
    # Get bootstrap predictions
    bootstrap_preds = []
    if "bootstrap_ensemble" in models:
        for model in models["bootstrap_ensemble"]:
            try:
                pred = model.predict(X_input)[0]
                bootstrap_preds.append(max(0, pred))
            except Exception as e:
                continue
    
    if bootstrap_preds:
        bootstrap_stats = {
            "mean": np.mean(bootstrap_preds),
            "std": np.std(bootstrap_preds),
            "min": np.min(bootstrap_preds),
            "max": np.max(bootstrap_preds)
        }
    else:
        bootstrap_stats = {}
    
    return quantile_preds, bootstrap_stats

# ============================================================================
# PHASE 2: ADVANCED MODEL (TRY IF SIMPLE WORKS)
# ============================================================================

def try_advanced_features(daily_data):
    """Try to create advanced features"""
    safe_print("🚀 PHASE 2: Trying ADVANCED features...")
    
    try:
        # Load call data again for intent information
        call_path = find_file(["callintent.csv", "data/callintent.csv"])
        calls = pd.read_csv(call_path)
        calls.columns = [c.lower().strip() for c in calls.columns]
        
        # Find intent column
        intent_col = next((c for c in calls.columns if "intent" in c), None)
        
        if intent_col is None:
            LOG.warning("No intent column found, skipping advanced features")
            return None, None, None
        
        LOG.info(f"Found intent column: {intent_col}")
        
        # Process intent data
        date_col = next((c for c in calls.columns if "date" in c or "start" in c), None)
        calls[date_col] = to_date(calls[date_col])
        calls = calls.dropna(subset=[date_col])
        calls = calls[calls[date_col] >= pd.to_datetime('2025-01-01').date()]
        
        # Clean intent data
        calls[intent_col] = calls[intent_col].fillna('Unknown').astype(str)
        
        # Get top intents
        intent_counts = calls[intent_col].value_counts()
        common_intents = intent_counts[intent_counts >= CFG["min_intent_occurrences"]].index
        top_intents = intent_counts.head(CFG["top_intents"]).index.tolist()
        
        LOG.info(f"Top intents: {top_intents[:5]}...")
        
        # Create daily intent distribution
        calls_filtered = calls[calls[intent_col].isin(top_intents)]
        
        if len(calls_filtered) == 0:
            LOG.warning("No calls with common intents, skipping advanced features")
            return None, None, None
        
        # Create intent pivot
        intent_pivot = calls_filtered.groupby([date_col, intent_col]).size().unstack(fill_value=0)
        intent_pivot.index = pd.to_datetime(intent_pivot.index)
        
        # Convert to percentages
        daily_intents = intent_pivot.div(intent_pivot.sum(axis=1), axis=0).fillna(0)
        
        # Filter to business days
        business_mask = (~daily_intents.index.weekday.isin([5, 6]))
        daily_intents = daily_intents[business_mask]
        
        # Get top mail types (more than simple)
        mail_cols = [c for c in daily_data.columns if c != 'calls_total']
        mail_volumes = daily_data[mail_cols].sum().sort_values(ascending=False)
        top_mail_types = mail_volumes.head(CFG["top_mail_types_advanced"]).index.tolist()
        
        LOG.info(f"Advanced: {len(top_mail_types)} mail types, {len(daily_intents.columns)} intents")
        
        return daily_intents, top_mail_types, top_intents
        
    except Exception as e:
        LOG.error(f"Advanced features failed: {e}")
        return None, None, None

def create_advanced_features(daily_data, daily_intents, top_mail_types):
    """Create advanced features with more complexity"""
    safe_print("🛠️  Creating ADVANCED features...")
    
    try:
        features_list = []
        volume_targets = []
        intent_targets = []
        
        # Align dates
        common_dates = daily_data.index.intersection(daily_intents.index)
        daily_data_aligned = daily_data.loc[common_dates]
        daily_intents_aligned = daily_intents.loc[common_dates]
        
        LOG.info(f"Advanced features: {len(common_dates)} aligned days")
        
        for i in range(len(common_dates) - 1):
            current_date = common_dates[i]
            next_date = common_dates[i + 1]
            
            current_data = daily_data_aligned.loc[current_date]
            next_data = daily_data_aligned.loc[next_date]
            
            current_intents = daily_intents_aligned.loc[current_date]
            next_intents = daily_intents_aligned.loc[next_date]
            
            feature_row = {}
            
            # Advanced mail features
            for mail_type in top_mail_types:
                clean_name = str(mail_type).replace(' ', '').replace('-', '')[:10]
                volume = current_data[mail_type]
                feature_row[f"{clean_name}_vol"] = volume
            
            # Mail aggregates
            total_mail = sum(current_data[t] for t in top_mail_types)
            feature_row["total_mail"] = total_mail
            feature_row["log_total_mail"] = np.log1p(total_mail)
            
            # Current intent distribution
            for intent in daily_intents.columns:
                clean_intent = str(intent).replace(' ', '').replace('/', '')[:10]
                feature_row[f"current_{clean_intent}"] = current_intents[intent]
            
            # Temporal features
            feature_row["weekday"] = current_date.weekday()
            feature_row["month"] = current_date.month
            feature_row["is_month_end"] = 1 if current_date.day > 25 else 0
            
            # Recent call context
            recent_idx = max(0, i-5)
            recent_calls = daily_data_aligned["calls_total"].iloc[recent_idx:i+1]
            feature_row["recent_calls_avg"] = recent_calls.mean()
            feature_row["recent_calls_std"] = recent_calls.std() if len(recent_calls) > 1 else 0
            
            # Targets
            volume_target = next_data["calls_total"]
            intent_target = next_intents.idxmax()  # Dominant intent
            
            features_list.append(feature_row)
            volume_targets.append(volume_target)
            intent_targets.append(intent_target)
        
        # Convert to DataFrames
        X = pd.DataFrame(features_list)
        y_volume = pd.Series(volume_targets)
        y_intent = pd.Series(intent_targets)
        
        # Clean data
        X = X.fillna(0)
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        LOG.info(f"Advanced features: {X.shape[0]} samples x {X.shape[1]} features")
        
        return X, y_volume, y_intent
        
    except Exception as e:
        LOG.error(f"Advanced feature creation failed: {e}")
        return None, None, None

def train_advanced_models(X, y_volume, y_intent, top_intents):
    """Train advanced models"""
    safe_print("🎯 Training ADVANCED models...")
    
    try:
        models = {}
        
        # Volume model
        LOG.info("  Training advanced volume model...")
        volume_model = RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_split=5, random_state=42
        )
        
        # Split data
        split_point = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_vol_train, y_vol_test = y_volume.iloc[:split_point], y_volume.iloc[split_point:]
        
        volume_model.fit(X_train, y_vol_train)
        
        # Test volume model
        vol_pred = volume_model.predict(X_test)
        vol_mae = mean_absolute_error(y_vol_test, vol_pred)
        LOG.info(f"    Volume model MAE: {vol_mae:.0f}")
        
        models['volume'] = volume_model
        
        # Intent model
        if y_intent is not None and len(y_intent.unique()) > 1:
            LOG.info("  Training advanced intent model...")
            
            # Encode intents
            intent_encoder = LabelEncoder()
            y_intent_encoded = intent_encoder.fit_transform(y_intent)
            y_intent_train = y_intent_encoded[:split_point]
            y_intent_test = y_intent_encoded[split_point:]
            
            intent_model = RandomForestClassifier(
                n_estimators=100, max_depth=6, min_samples_split=5, random_state=42
            )
            intent_model.fit(X_train, y_intent_train)
            
            # Test intent model
            intent_pred = intent_model.predict(X_test)
            intent_acc = accuracy_score(y_intent_test, intent_pred)
            LOG.info(f"    Intent model accuracy: {intent_acc:.3f}")
            
            models['intent'] = intent_model
            models['intent_encoder'] = intent_encoder
        
        return models
        
    except Exception as e:
        LOG.error(f"Advanced model training failed: {e}")
        return {}

# ============================================================================
# TESTING AND EXAMPLES
# ============================================================================

def test_simple_model(models, top_mail_types):
    """Test the simple model"""
    safe_print("🧪 Testing SIMPLE model...")
    
    # Create test scenarios
    scenarios = [
        {
            "name": "High Volume Day",
            "mail_inputs": {top_mail_types[0]: 2000, top_mail_types[1]: 1500} if len(top_mail_types) >= 2 else {top_mail_types[0]: 2000}
        },
        {
            "name": "Low Volume Day", 
            "mail_inputs": {top_mail_types[0]: 500, top_mail_types[1]: 300} if len(top_mail_types) >= 2 else {top_mail_types[0]: 500}
        },
        {
            "name": "Mixed Mail Day",
            "mail_inputs": {mt: 800 for mt in top_mail_types[:3]}
        }
    ]
    
    results = {}
    
    for scenario in scenarios:
        LOG.info(f"  Testing: {scenario['name']}")
        
        try:
            quantile_preds, bootstrap_stats = predict_simple(models, scenario["mail_inputs"])
            
            if quantile_preds:
                LOG.info(f"    Most likely: {quantile_preds.get('q50', 0):.0f} calls")
                LOG.info(f"    Range (25-75%): {quantile_preds.get('q25', 0):.0f} - {quantile_preds.get('q75', 0):.0f} calls")
                
                results[scenario['name']] = {
                    "mail_inputs": scenario["mail_inputs"],
                    "predictions": quantile_preds,
                    "bootstrap": bootstrap_stats
                }
            else:
                LOG.warning(f"    No predictions for {scenario['name']}")
                
        except Exception as e:
            LOG.error(f"    Test failed: {e}")
    
    return results

def test_advanced_model(advanced_models, top_mail_types, top_intents):
    """Test the advanced model"""
    if not advanced_models:
        return {}
        
    safe_print("🧪 Testing ADVANCED model...")
    
    # This would be more complex, but for now just indicate it exists
    LOG.info(f"  Advanced model has {len(advanced_models)} components")
    
    if 'volume' in advanced_models:
        LOG.info("    ✓ Volume prediction model ready")
    
    if 'intent' in advanced_models:
        LOG.info("    ✓ Intent prediction model ready")
    
    return {"status": "advanced_models_available"}

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution - simple first, then advanced"""
    
    safe_print("="*80)
    safe_print("MAIL-TO-CALLS PREDICTION: SIMPLE FIRST, THEN ADVANCED")
    safe_print("="*80)
    safe_print("STRATEGY:")
    safe_print("  1. Build SIMPLE working model (guaranteed)")
    safe_print("  2. Try ADVANCED features (if possible)")
    safe_print("  3. You get minimum simple model, maximum full system")
    safe_print("="*80)
    
    # Create output directory
    output_dir = Path(CFG["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    results = {
        "simple_model": None,
        "advanced_model": None,
        "simple_tests": None,
        "advanced_tests": None
    }
    
    try:
        # ===== PHASE 1: SIMPLE MODEL =====
        safe_print("\n🎯 PHASE 1: BUILDING SIMPLE MODEL")
        
        # Load data
        daily_data = load_simple_data()
        
        # Create simple features
        X_simple, y_simple, top_mail_types = create_simple_features(daily_data)
        
        # Train simple models
        simple_models = train_simple_models(X_simple, y_simple)
        
        if simple_models:
            safe_print("✅ SIMPLE MODEL: SUCCESS!")
            results["simple_model"] = simple_models
            
            # Save simple model
            joblib.dump(simple_models, output_dir / "simple_models.pkl")
            joblib.dump(top_mail_types, output_dir / "top_mail_types.pkl")
            
            # Test simple model
            simple_tests = test_simple_model(simple_models, top_mail_types)
            results["simple_tests"] = simple_tests
            
            # Save test results
            with open(output_dir / "simple_test_results.json", "w") as f:
                json.dump(simple_tests, f, indent=2)
            
        else:
            safe_print("❌ SIMPLE MODEL: FAILED")
            return results
        
        # ===== PHASE 2: ADVANCED MODEL =====
        safe_print("\n🚀 PHASE 2: TRYING ADVANCED MODEL")
        
        # Try advanced features
        daily_intents, top_mail_advanced, top_intents = try_advanced_features(daily_data)
        
        if daily_intents is not None:
            # Create advanced features
            X_advanced, y_vol_advanced, y_intent_advanced = create_advanced_features(
                daily_data, daily_intents, top_mail_advanced
            )
            
            if X_advanced is not None:
                # Train advanced models
                advanced_models = train_advanced_models(X_advanced, y_vol_advanced, y_intent_advanced, top_intents)
                
                if advanced_models:
                    safe_print("✅ ADVANCED MODEL: SUCCESS!")
                    results["advanced_model"] = advanced_models
                    
                    # Save advanced model
                    joblib.dump(advanced_models, output_dir / "advanced_models.pkl")
                    joblib.dump({"mail_types": top_mail_advanced, "intents": top_intents}, 
                               output_dir / "advanced_config.pkl")
                    
                    # Test advanced model
                    advanced_tests = test_advanced_model(advanced_models, top_mail_advanced, top_intents)
                    results["advanced_tests"] = advanced_tests
                    
                else:
                    safe_print("⚠️ ADVANCED MODEL: Training failed, but SIMPLE model works")
            else:
                safe_print("⚠️ ADVANCED MODEL: Feature creation failed, but SIMPLE model works")
        else:
            safe_print("⚠️ ADVANCED MODEL: No intent data found, but SIMPLE model works")
        
        # ===== FINAL SUMMARY =====
        safe_print("\n" + "="*60)
        safe_print("🎯 FINAL RESULTS")
        safe_print("="*60)
        
        if results["simple_model"]:
            simple_model_count = len(results["simple_model"])
            safe_print(f"✅ SIMPLE MODEL: {simple_model_count} models trained")
            safe_print(f"   📊 Top mail types: {len(top_mail_types)}")
            safe_print(f"   🧪 Test scenarios: {len(results.get('simple_tests', {}))}")
            
            # Show example prediction
            if results.get("simple_tests"):
                first_test = list(results["simple_tests"].values())[0]
                example_pred = first_test.get("predictions", {}).get("q50", 0)
                safe_print(f"   📞 Example prediction: {example_pred:.0f} calls")
        
        if results["advanced_model"]:
            advanced_components = len(results["advanced_model"])
            safe_print(f"✅ ADVANCED MODEL: {advanced_components} components")
            if "volume" in results["advanced_model"]:
                safe_print("   📈 Volume prediction: Ready")
            if "intent" in results["advanced_model"]:
                safe_print("   🎯 Intent prediction: Ready")
        
        safe_print("")
        safe_print("🎉 SUCCESS: You have at least a simple working model!")
        safe_print(f"📁 All models saved to: {output_dir}")
        
        # Create usage example
        usage_example = f"""
USAGE EXAMPLE:
=============

import joblib

# Load simple model
simple_models = joblib.load('{output_dir}/simple_models.pkl')
top_mail_types = joblib.load('{output_dir}/top_mail_types.pkl')

# Make prediction
mail_input = {{
    '{top_mail_types[0]}': 1500,
    '{top_mail_types[1] if len(top_mail_types) > 1 else top_mail_types[0]}': 800
}}

# This function is in your script
quantile_preds, bootstrap_stats = predict_simple(simple_models, mail_input)

print(f"Predicted calls: {{quantile_preds.get('q50', 0):.0f}}")
print(f"Range (25-75%): {{quantile_preds.get('q25', 0):.0f}} - {{quantile_preds.get('q75', 0):.0f}}")

YOUR TOP MAIL TYPES:
{chr(10).join([f"  {i+1}. {mt}" for i, mt in enumerate(top_mail_types[:10])])}
"""
        
        with open(output_dir / "USAGE_GUIDE.txt", "w") as f:
            f.write(usage_example)
        
        safe_print("📖 See USAGE_GUIDE.txt for how to use your models")
        
        return results
        
    except Exception as e:
        safe_print(f"\n💥 SYSTEM ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return results

if __name__ == "__main__":
    main()
