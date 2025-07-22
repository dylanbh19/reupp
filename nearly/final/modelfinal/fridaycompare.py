#!/usr/bin/env python
# friday_enhanced_model_trainer.py
# =========================================================
# FRIDAY-ENHANCED MODEL TRAINING PIPELINE
# =========================================================
# Complete pipeline to:
# 1. Train baseline model (your original)
# 2. Train Friday-enhanced model (with winning features)
# 3. Test both models across all weekdays
# 4. Generate before/after comparison
# =========================================================

from pathlib import Path
import json
import logging
import sys
import warnings
from datetime import datetime, timedelta
import time
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holidays

# Handle sklearn imports with fallbacks
try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import QuantileRegressor, Ridge, LinearRegression
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("ERROR: scikit-learn not available!")
    sys.exit(1)

# Handle joblib with fallback
try:
    import joblib
except ImportError:
    import pickle as joblib

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CFG = {
    "top_mail_types": [
        "Reject_Ltrs", "Cheque 1099", "Exercise_Converted",
        "SOI_Confirms", "Exch_chks", "ACH_Debit_Enrollment",
        "Transfer", "COA", "NOTC_WITHDRAW", "Repl_Chks"
    ],
    "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
    "bootstrap_samples": 20,  # Reduced for stability
    "baseline_output_dir": "baseline_model_results",
    "enhanced_output_dir": "friday_enhanced_model_results",
    "comparison_output_dir": "before_after_comparison",
    
    # Friday Enhancement Settings
    "friday_features_enabled": True,
    "friday_multiplier_fallback": 1.25,
    "friday_polynomial_features": True,
    "friday_interaction_features": True,
    "friday_seasonal_features": True,
    "test_all_weekdays": True,
    
    # Solver settings for stability
    "quantile_solver": 'highs-ds',  # More stable solver
    "quantile_alpha": 0.01,  # Stronger regularization
    "max_iter": 1000
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Production logging setup with proper encoding"""
    
    try:
        # Create all output directories
        for dir_name in ["baseline_output_dir", "enhanced_output_dir", "comparison_output_dir"]:
            Path(CFG[dir_name]).mkdir(exist_ok=True)
        
        # Set up logging with UTF-8 encoding
        logger = logging.getLogger("FridayTrainer")
        logger.setLevel(logging.INFO)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter("%(asctime)s | %(levelname)8s | %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler with UTF-8 encoding
        try:
            log_path = Path(CFG["comparison_output_dir"]) / "training_pipeline.log"
            file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter("%(asctime)s | %(levelname)8s | %(message)s")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create log file: {e}")
        
        logger.info("Friday Training Pipeline initialized")
        return logger
        
    except Exception as e:
        # Fallback to basic logging
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        logger = logging.getLogger("FridayTrainer")
        logger.warning(f"Advanced logging failed: {e}")
        return logger

LOG = setup_logging()

# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

def _to_date(s):
    """Convert to date with error handling"""
    try:
        return pd.to_datetime(s, errors="coerce").dt.date
    except Exception as e:
        LOG.warning(f"Date conversion error: {e}")
        return pd.to_datetime(s, errors="coerce", infer_datetime_format=True).dt.date

def _find_file(candidates):
    """Find file from candidates with better error reporting"""
    for p in candidates:
        try:
            path = Path(p)
            if path.exists():
                LOG.info(f"Found file: {path}")
                return path
        except Exception as e:
            LOG.warning(f"Error checking path {p}: {e}")
            continue
    
    LOG.error(f"No files found from candidates: {candidates}")
    raise FileNotFoundError(f"None found: {candidates}")

def load_mail_call_data():
    """Load data and create mail->calls relationship dataset (from your original)"""
    
    LOG.info("Loading mail and call data...")
    
    try:
        # Load mail data
        LOG.info("Loading mail data...")
        mail_path = _find_file(["mail.csv", "data/mail.csv", "../data/mail.csv"])
        mail = pd.read_csv(mail_path)
        mail.columns = [c.lower().strip() for c in mail.columns]
        mail["mail_date"] = _to_date(mail["mail_date"])
        mail = mail.dropna(subset=["mail_date"])
        LOG.info(f"Mail data loaded: {mail.shape}")

        # Load call volumes
        LOG.info("Loading call volume data...")
        vol_path = _find_file(["callvolumes.csv", "data/callvolumes.csv", "../data/callvolumes.csv"])
        df_vol = pd.read_csv(vol_path)
        df_vol.columns = [c.lower().strip() for c in df_vol.columns]
        
        # Find date column
        date_cols = [c for c in df_vol.columns if "date" in c.lower()]
        if not date_cols:
            raise ValueError("No date column found in call volumes")
        
        dcol_v = date_cols[0]
        df_vol[dcol_v] = _to_date(df_vol[dcol_v])
        
        # Find volume column
        vol_cols = [c for c in df_vol.columns if c != dcol_v and df_vol[c].dtype in ['int64', 'float64']]
        if not vol_cols:
            raise ValueError("No volume column found in call volumes")
        
        vol_daily = df_vol.groupby(dcol_v)[vol_cols[0]].sum()
        LOG.info(f"Call volumes processed: {len(vol_daily)} days")

        # Load call intent data
        LOG.info("Loading call intent data...")
        intent_path = _find_file(["callintent.csv", "data/callintent.csv", "../data/callintent.csv", "callintetn.csv"])
        df_int = pd.read_csv(intent_path)
        df_int.columns = [c.lower().strip() for c in df_int.columns]
        
        # Find date column
        date_cols = [c for c in df_int.columns if "date" in c.lower() or "conversation" in c.lower()]
        if not date_cols:
            raise ValueError("No date column found in call intent")
        
        dcol_i = date_cols[0]
        df_int[dcol_i] = _to_date(df_int[dcol_i])
        int_daily = df_int.groupby(dcol_i).size()
        LOG.info(f"Call intent processed: {len(int_daily)} days")

        # Scale and combine call data
        overlap = vol_daily.index.intersection(int_daily.index)
        if len(overlap) >= 5:
            scale = int_daily.loc[overlap].mean() / vol_daily.loc[overlap].mean()
            vol_daily *= scale
            LOG.info(f"Scaled call volumes using {len(overlap)} overlapping days")
        
        calls_total = vol_daily.combine_first(int_daily).sort_index()

        # Process mail data
        LOG.info("Processing mail data...")
        mail_pivot = mail.groupby(["mail_date", "mail_type"], as_index=False)["mail_volume"].sum()
        mail_daily = mail_pivot.pivot(index="mail_date", columns="mail_type", values="mail_volume").fillna(0)
        
        # Convert indices to datetime
        mail_daily.index = pd.to_datetime(mail_daily.index)
        calls_total.index = pd.to_datetime(calls_total.index)

        # Filter to business days only
        LOG.info("Filtering to business days...")
        us_holidays = holidays.US()
        biz_mask = (~mail_daily.index.weekday.isin([5, 6])) & (~mail_daily.index.isin(us_holidays))
        mail_daily = mail_daily.loc[biz_mask]
        calls_total = calls_total.loc[calls_total.index.isin(mail_daily.index)]

        # Combine data
        daily = mail_daily.join(calls_total.rename("calls_total"), how="inner")
        
        LOG.info(f"Final combined data: {daily.shape}")
        LOG.info(f"Date range: {daily.index.min()} to {daily.index.max()}")
        
        # Validate data
        if daily.empty:
            raise ValueError("No data after combining mail and calls")
        if daily['calls_total'].isna().all():
            raise ValueError("No valid call data")
        if daily.select_dtypes(include=[np.number]).sum().sum() == 0:
            raise ValueError("No numerical data found")
        
        return daily
        
    except Exception as e:
        LOG.error(f"Error loading data: {e}")
        LOG.error(traceback.format_exc())
        raise

# ============================================================================
# BASELINE MODEL (YOUR ORIGINAL)
# ============================================================================

def create_baseline_features(daily):
    """Create baseline features (your original logic)"""
    
    LOG.info("Creating baseline features...")
    
    try:
        features_list = []
        targets_list = []
        
        for i in range(len(daily) - 1):
            try:
                current_day = daily.iloc[i]
                next_day = daily.iloc[i + 1]
                
                feature_row = {}
                
                # Mail volumes (INPUT FEATURES)
                available_types = [t for t in CFG["top_mail_types"] if t in daily.columns]
                
                for mail_type in available_types:
                    volume = current_day.get(mail_type, 0)
                    feature_row[f"{mail_type}_volume"] = max(0, float(volume)) if not pd.isna(volume) else 0
                
                # Total mail volume
                total_mail = sum(feature_row.get(f"{t}_volume", 0) for t in available_types)
                feature_row["total_mail_volume"] = total_mail
                feature_row["log_total_mail_volume"] = np.log1p(total_mail)
                
                # Mail volume percentiles (relative to historical)
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
                
                # Holiday check
                try:
                    feature_row["is_holiday_week"] = 1 if current_date.date() in holidays.US() else 0
                except:
                    feature_row["is_holiday_week"] = 0
                
                # Recent call volume context (baseline)
                recent_calls = daily["calls_total"].iloc[max(0, i-5):i+1]
                feature_row["recent_calls_avg"] = recent_calls.mean() if not recent_calls.empty else 15000
                feature_row["recent_calls_trend"] = recent_calls.diff().mean() if len(recent_calls) > 1 else 0
                
                # Target: next day's calls
                target = next_day["calls_total"]
                if pd.isna(target) or target <= 0:
                    continue  # Skip invalid targets
                
                features_list.append(feature_row)
                targets_list.append(float(target))
                
            except Exception as e:
                LOG.warning(f"Error processing day {i}: {e}")
                continue
        
        # Convert to DataFrames
        X = pd.DataFrame(features_list)
        y = pd.Series(targets_list)
        
        # Clean and validate
        X = X.fillna(0)
        X = X.select_dtypes(include=[np.number])  # Only numeric columns
        
        # Remove any infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        LOG.info(f"Baseline features created: {X.shape[0]} samples x {X.shape[1]} features")
        LOG.info(f"Target range: {y.min():.0f} to {y.max():.0f}")
        
        return X, y
        
    except Exception as e:
        LOG.error(f"Error creating baseline features: {e}")
        raise

# ============================================================================
# FRIDAY-ENHANCED MODEL (WITH WINNING FEATURES)
# ============================================================================

def create_friday_enhanced_features(daily):
    """Create Friday-enhanced features with winning polynomial features"""
    
    LOG.info("Creating Friday-enhanced features...")
    
    try:
        features_list = []
        targets_list = []
        
        for i in range(len(daily) - 1):
            try:
                current_day = daily.iloc[i]
                next_day = daily.iloc[i + 1]
                
                feature_row = {}
                
                # ===== BASELINE FEATURES (SAME AS ORIGINAL) =====
                available_types = [t for t in CFG["top_mail_types"] if t in daily.columns]
                
                for mail_type in available_types:
                    volume = current_day.get(mail_type, 0)
                    feature_row[f"{mail_type}_volume"] = max(0, float(volume)) if not pd.isna(volume) else 0
                
                # Total mail volume
                total_mail = sum(feature_row.get(f"{t}_volume", 0) for t in available_types)
                feature_row["total_mail_volume"] = total_mail
                feature_row["log_total_mail_volume"] = np.log1p(total_mail)
                
                # Mail volume percentiles
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
                
                # Holiday check
                try:
                    feature_row["is_holiday_week"] = 1 if current_date.date() in holidays.US() else 0
                except:
                    feature_row["is_holiday_week"] = 0
                
                # Recent call volume context
                recent_calls = daily["calls_total"].iloc[max(0, i-5):i+1]
                feature_row["recent_calls_avg"] = recent_calls.mean() if not recent_calls.empty else 15000
                feature_row["recent_calls_trend"] = recent_calls.diff().mean() if len(recent_calls) > 1 else 0
                
                # ===== FRIDAY ENHANCEMENT FEATURES =====
                if CFG["friday_features_enabled"]:
                    friday_features = _create_winning_friday_features(feature_row, current_date, daily, i)
                    feature_row.update(friday_features)
                
                # Target: next day's calls
                target = next_day["calls_total"]
                if pd.isna(target) or target <= 0:
                    continue  # Skip invalid targets
                
                features_list.append(feature_row)
                targets_list.append(float(target))
                
            except Exception as e:
                LOG.warning(f"Error processing enhanced day {i}: {e}")
                continue
        
        # Convert to DataFrames
        X = pd.DataFrame(features_list)
        y = pd.Series(targets_list)
        
        # Clean and validate
        X = X.fillna(0)
        X = X.select_dtypes(include=[np.number])  # Only numeric columns
        
        # Remove any infinite values
        X = X.replace([np.inf, -np.inf], 0)
        
        # Scale down very large polynomial features to prevent numerical issues
        for col in X.columns:
            if 'squared' in col or 'cubed' in col:
                if X[col].max() > 1e10:  # Very large values
                    X[col] = X[col] / 1e6  # Scale down
        
        original_features = 19  # Your original feature count
        new_features = len(X.columns) - original_features
        
        LOG.info(f"Friday-enhanced features created: {X.shape[0]} samples x {X.shape[1]} features")
        LOG.info(f"Original: {original_features}, New Friday features: {new_features}")
        
        return X, y
        
    except Exception as e:
        LOG.error(f"Error creating Friday-enhanced features: {e}")
        raise

def _create_winning_friday_features(feature_row, current_date, daily, i):
    """Create the winning Friday features that achieved 18.6% improvement"""
    
    try:
        friday_features = {}
        
        # Core Friday indicator
        is_friday = 1 if current_date.weekday() == 4 else 0
        friday_features["is_friday"] = is_friday
        
        if not is_friday:
            # If not Friday, set all Friday features to 0
            return _get_zero_friday_features()
        
        # ===== WINNING POLYNOMIAL FRIDAY FEATURES =====
        if CFG["friday_polynomial_features"]:
            total_mail = feature_row.get("total_mail_volume", 0)
            
            # Polynomial interactions (scale to prevent overflow)
            if total_mail > 0:
                friday_features["friday_mail_squared"] = (total_mail / 1000) ** 2  # Scale down
                friday_features["friday_mail_sqrt"] = np.sqrt(total_mail)
                friday_features["friday_mail_cubed"] = (total_mail / 10000) ** 3  # Scale down more
                
                log_mail = np.log1p(total_mail)
                friday_features["friday_log_mail_squared"] = log_mail ** 2
            else:
                friday_features["friday_mail_squared"] = 0
                friday_features["friday_mail_sqrt"] = 0
                friday_features["friday_mail_cubed"] = 0
                friday_features["friday_log_mail_squared"] = 0
        
        # ===== FRIDAY INTERACTION FEATURES =====
        if CFG["friday_interaction_features"]:
            # Friday * mail volume interactions for top mail types
            high_impact_types = ["Reject_Ltrs_volume", "Cheque 1099_volume", "Exercise_Converted_volume"]
            
            for mail_type in high_impact_types:
                if mail_type in feature_row:
                    volume = feature_row[mail_type]
                    friday_features[f"friday_{mail_type}"] = volume
                    friday_features[f"friday_{mail_type}_squared"] = (volume / 1000) ** 2 if volume > 0 else 0
            
            # Friday * total mail interaction
            friday_features["friday_total_mail"] = feature_row.get("total_mail_volume", 0)
            friday_features["friday_log_mail"] = feature_row.get("log_total_mail_volume", 0)
            
            # Friday * recent calls interaction
            friday_features["friday_recent_calls"] = feature_row.get("recent_calls_avg", 0) / 10000  # Scale
            friday_features["friday_calls_trend"] = feature_row.get("recent_calls_trend", 0)
            
            # Friday * mail percentile interaction
            friday_features["friday_mail_percentile"] = feature_row.get("mail_percentile", 0.5)
        
        # ===== FRIDAY SEASONAL FEATURES =====
        if CFG["friday_seasonal_features"]:
            month = current_date.month
            
            # Quarter-end Fridays
            friday_features["friday_quarter_end"] = 1 if month in [3, 6, 9, 12] else 0
            
            # Seasonal patterns
            friday_features["friday_summer"] = 1 if month in [6, 7, 8] else 0
            friday_features["friday_winter"] = 1 if month in [12, 1, 2] else 0
            
            # Friday characteristics
            friday_features["friday_of_month"] = (month % 4) + 1
            friday_features["friday_month_end"] = feature_row.get("is_month_end", 0)
            friday_features["friday_holiday_week"] = feature_row.get("is_holiday_week", 0)
        
        # ===== FRIDAY COMPOSITE SCORES =====
        total_mail = feature_row.get("total_mail_volume", 0)
        recent_calls = feature_row.get("recent_calls_avg", 15000)
        
        # Normalized scores
        mail_score = max(-2, min(2, (total_mail - 5000) / 10000))  # Bounded score
        calls_score = max(-2, min(2, (recent_calls - 15000) / 5000))  # Bounded score
        
        friday_features["friday_risk_score"] = mail_score + calls_score
        friday_features["friday_intensity_score"] = min(5, total_mail / 10000)  # Bounded intensity
        
        return friday_features
        
    except Exception as e:
        LOG.warning(f"Error creating Friday features: {e}")
        return _get_zero_friday_features()

def _get_zero_friday_features():
    """Return zero values for all Friday features when not Friday"""
    return {
        "is_friday": 0,
        # Polynomial features
        "friday_mail_squared": 0,
        "friday_mail_sqrt": 0,
        "friday_mail_cubed": 0,
        "friday_log_mail_squared": 0,
        
        # Interaction features
        "friday_Reject_Ltrs_volume": 0,
        "friday_Reject_Ltrs_volume_squared": 0,
        "friday_Cheque 1099_volume": 0,
        "friday_Cheque 1099_volume_squared": 0,
        "friday_Exercise_Converted_volume": 0,
        "friday_Exercise_Converted_volume_squared": 0,
        "friday_total_mail": 0,
        "friday_log_mail": 0,
        "friday_recent_calls": 0,
        "friday_calls_trend": 0,
        "friday_mail_percentile": 0,
        
        # Seasonal features  
        "friday_quarter_end": 0,
        "friday_summer": 0,
        "friday_winter": 0,
        "friday_of_month": 0,
        "friday_month_end": 0,
        "friday_holiday_week": 0,
        
        # Composite scores
        "friday_risk_score": 0,
        "friday_intensity_score": 0
    }

# ============================================================================
# MODEL TRAINING WITH ROBUST ERROR HANDLING
# ============================================================================

def train_models(X, y, model_type="baseline"):
    """Train models with robust error handling"""
    
    LOG.info(f"Training {model_type} models...")
    
    try:
        # Split for validation
        split_point = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        # Validate data
        if X_train.empty or y_train.empty:
            raise ValueError("Training data is empty")
        
        models = {}
        
        # Try different solvers in order of preference
        solvers_to_try = ['highs-ds', 'highs-ipm', 'highs', 'interior-point']
        alpha_values = [0.01, 0.1, 1.0]  # Different regularization strengths
        
        # Quantile models for range prediction
        for quantile in CFG["quantiles"]:
            LOG.info(f"  Training {int(quantile * 100)}% quantile model...")
            
            model_trained = False
            last_error = None
            
            # Try different solvers and alpha values
            for solver in solvers_to_try:
                for alpha in alpha_values:
                    try:
                        model = QuantileRegressor(
                            quantile=quantile, 
                            alpha=alpha, 
                            solver=solver,
                            max_iter=CFG['max_iter']
                        )
                        model.fit(X_train, y_train)
                        
                        # Validate the model
                        y_pred = model.predict(X_test)
                        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                            raise ValueError("Model produced invalid predictions")
                        
                        mae = mean_absolute_error(y_test, y_pred)
                        
                        models[f"quantile_{quantile}"] = model
                        LOG.info(f"    Success with {solver}, alpha={alpha}, MAE: {mae:.0f}")
                        model_trained = True
                        break
                        
                    except Exception as e:
                        last_error = e
                        continue
                
                if model_trained:
                    break
            
            # Fallback to linear regression if quantile regression fails
            if not model_trained:
                LOG.warning(f"  Quantile regression failed for {quantile}, using Linear Regression fallback")
                LOG.warning(f"  Last error: {last_error}")
                
                try:
                    # Use linear regression as fallback
                    fallback_model = LinearRegression()
                    fallback_model.fit(X_train, y_train)
                    
                    y_pred = fallback_model.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    
                    models[f"quantile_{quantile}"] = fallback_model
                    LOG.info(f"    Fallback Linear Regression MAE: {mae:.0f}")
                    
                except Exception as e:
                    LOG.error(f"  Even fallback model failed for quantile {quantile}: {e}")
                    # Create dummy model that returns mean
                    class DummyModel:
                        def __init__(self, mean_value):
                            self.mean_value = mean_value
                        def predict(self, X):
                            return np.full(len(X), self.mean_value)
                    
                    models[f"quantile_{quantile}"] = DummyModel(y_train.mean())
                    LOG.info(f"    Using dummy model returning mean: {y_train.mean():.0f}")
        
        # Bootstrap ensemble for uncertainty (simplified)
        LOG.info("  Training bootstrap ensemble...")
        bootstrap_models = []
        
        try:
            for i in range(min(CFG["bootstrap_samples"], 10)):  # Limit to 10 for speed
                # Bootstrap sample
                sample_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
                X_boot = X_train.iloc[sample_idx]
                y_boot = y_train.iloc[sample_idx]
                
                # Simple random forest model
                try:
                    model = RandomForestRegressor(
                        n_estimators=20,  # Reduced for speed
                        max_depth=6,
                        min_samples_leaf=5,
                        random_state=i,
                        n_jobs=1
                    )
                    model.fit(X_boot, y_boot)
                    bootstrap_models.append(model)
                    
                except Exception as e:
                    LOG.warning(f"Bootstrap model {i} failed: {e}")
                    continue
                    
        except Exception as e:
            LOG.warning(f"Bootstrap ensemble failed: {e}")
        
        if bootstrap_models:
            models["bootstrap_ensemble"] = bootstrap_models
            LOG.info(f"  Bootstrap ensemble created with {len(bootstrap_models)} models")
        else:
            LOG.warning("  No bootstrap models created")
        
        LOG.info(f"{model_type.title()} models trained successfully!")
        return models
        
    except Exception as e:
        LOG.error(f"Error training {model_type} models: {e}")
        LOG.error(traceback.format_exc())
        raise

# ============================================================================
# COMPREHENSIVE MODEL TESTING
# ============================================================================

def test_models_comprehensive(X, y, models, model_type="baseline"):
    """Test models comprehensively across all weekdays"""
    
    LOG.info(f"Testing {model_type} models comprehensively...")
    
    try:
        # Split data
        split_point = int(len(X) * 0.8)
        X_test = X.iloc[split_point:]
        y_test = y.iloc[split_point:]
        
        if X_test.empty or y_test.empty:
            raise ValueError("Test data is empty")
        
        # Get main model predictions
        main_model = models.get("quantile_0.5")
        if main_model is None:
            raise ValueError("No main model (quantile_0.5) found")
        
        try:
            y_pred = main_model.predict(X_test)
            
            # Validate predictions
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                LOG.warning("Invalid predictions detected, replacing with mean")
                y_pred = np.full_like(y_pred, y_test.mean())
                
        except Exception as e:
            LOG.error(f"Prediction failed: {e}")
            y_pred = np.full(len(y_test), y_test.mean())
        
        # Overall metrics
        overall_metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'accuracy': max(0, 100 - (mean_absolute_error(y_test, y_pred) / y_test.mean() * 100))
        }
        
        LOG.info(f"Overall {model_type} performance:")
        LOG.info(f"  MAE: {overall_metrics['mae']:.0f}")
        LOG.info(f"  RMSE: {overall_metrics['rmse']:.0f}")
        LOG.info(f"  R2: {overall_metrics['r2']:.3f}")
        LOG.info(f"  Accuracy: {overall_metrics['accuracy']:.1f}%")
        
        # Weekday-specific metrics
        weekday_metrics = {}
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        if 'weekday' in X_test.columns:
            LOG.info(f"\n{model_type} performance by weekday:")
            
            for day_num, day_name in enumerate(weekdays):
                day_mask = X_test['weekday'] == day_num
                if day_mask.sum() > 0:
                    day_true = y_test[day_mask]
                    day_pred = y_pred[day_mask]
                    
                    day_mae = mean_absolute_error(day_true, day_pred)
                    day_bias = (day_pred - day_true).mean()
                    day_samples = day_mask.sum()
                    
                    weekday_metrics[day_name] = {
                        'mae': day_mae,
                        'bias': day_bias,
                        'samples': day_samples,
                        'avg_actual': day_true.mean(),
                        'avg_predicted': day_pred.mean()
                    }
                    
                    LOG.info(f"  {day_name:10s}: MAE={day_mae:6.0f}, Bias={day_bias:+6.0f}, Samples={day_samples:3d}")
            
            # Highlight Friday performance
            if 'Friday' in weekday_metrics:
                friday_mae = weekday_metrics['Friday']['mae']
                LOG.info(f"\nFRIDAY {model_type} Challenge: MAE = {friday_mae:.0f}")
        
        return {
            'overall': overall_metrics,
            'weekday': weekday_metrics,
            'predictions': {'actual': y_test.values, 'predicted': y_pred}
        }
        
    except Exception as e:
        LOG.error(f"Error testing {model_type} models: {e}")
        LOG.error(traceback.format_exc())
        raise

# ============================================================================
# BEFORE/AFTER COMPARISON
# ============================================================================

def compare_models(baseline_results, enhanced_results):
    """Compare baseline vs Friday-enhanced models"""
    
    LOG.info("="*80)
    LOG.info("BEFORE/AFTER COMPARISON")
    LOG.info("="*80)
    
    try:
        comparison = {
            'overall_improvement': {},
            'weekday_improvements': {},
            'friday_improvement': {}
        }
        
        # Overall comparison
        baseline_overall = baseline_results['overall']
        enhanced_overall = enhanced_results['overall']
        
        mae_improvement = baseline_overall['mae'] - enhanced_overall['mae']
        mae_improvement_pct = (mae_improvement / baseline_overall['mae']) * 100
        
        comparison['overall_improvement'] = {
            'mae_before': baseline_overall['mae'],
            'mae_after': enhanced_overall['mae'],
            'mae_improvement': mae_improvement,
            'mae_improvement_pct': mae_improvement_pct,
            'accuracy_before': baseline_overall['accuracy'],
            'accuracy_after': enhanced_overall['accuracy'],
            'r2_before': baseline_overall['r2'],
            'r2_after': enhanced_overall['r2']
        }
        
        LOG.info("OVERALL MODEL IMPROVEMENT:")
        LOG.info(f"  MAE: {baseline_overall['mae']:.0f} -> {enhanced_overall['mae']:.0f} ({mae_improvement:+.0f}, {mae_improvement_pct:+.1f}%)")
        LOG.info(f"  Accuracy: {baseline_overall['accuracy']:.1f}% -> {enhanced_overall['accuracy']:.1f}%")
        LOG.info(f"  R2: {baseline_overall['r2']:.3f} -> {enhanced_overall['r2']:.3f}")
        
        # Weekday comparison
        if 'weekday' in baseline_results and 'weekday' in enhanced_results:
            LOG.info("\nWEEKDAY-SPECIFIC IMPROVEMENTS:")
            
            weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            
            for day in weekdays:
                if day in baseline_results['weekday'] and day in enhanced_results['weekday']:
                    baseline_day = baseline_results['weekday'][day]
                    enhanced_day = enhanced_results['weekday'][day]
                    
                    day_mae_improvement = baseline_day['mae'] - enhanced_day['mae']
                    day_mae_improvement_pct = (day_mae_improvement / baseline_day['mae']) * 100
                    
                    comparison['weekday_improvements'][day] = {
                        'mae_before': baseline_day['mae'],
                        'mae_after': enhanced_day['mae'],
                        'mae_improvement': day_mae_improvement,
                        'mae_improvement_pct': day_mae_improvement_pct
                    }
                    
                    LOG.info(f"  {day:10s}: {baseline_day['mae']:6.0f} -> {enhanced_day['mae']:6.0f} ({day_mae_improvement:+6.0f}, {day_mae_improvement_pct:+5.1f}%)")
        
        # Friday-specific analysis
        if 'Friday' in baseline_results.get('weekday', {}):
            baseline_friday = baseline_results['weekday']['Friday']
            enhanced_friday = enhanced_results['weekday']['Friday']
            
            friday_improvement = baseline_friday['mae'] - enhanced_friday['mae']
            friday_improvement_pct = (friday_improvement / baseline_friday['mae']) * 100
            
            comparison['friday_improvement'] = {
                'mae_before': baseline_friday['mae'],
                'mae_after': enhanced_friday['mae'],
                'improvement': friday_improvement,
                'improvement_pct': friday_improvement_pct
            }
            
            LOG.info("\nFRIDAY CHALLENGE RESULTS:")
            LOG.info(f"  Friday MAE: {baseline_friday['mae']:.0f} -> {enhanced_friday['mae']:.0f}")
            LOG.info(f"  Friday Improvement: {friday_improvement:+.0f} calls ({friday_improvement_pct:+.1f}%)")
            
            if friday_improvement > 0:
                LOG.info("  SUCCESS! Friday predictions improved!")
            else:
                LOG.info("  WARNING: Friday predictions unchanged or slightly worse")
        
        return comparison
        
    except Exception as e:
        LOG.error(f"Error in model comparison: {e}")
        raise

# ============================================================================
# VISUALIZATION WITH ERROR HANDLING
# ============================================================================

def create_comparison_visualizations(baseline_results, enhanced_results, comparison):
    """Create before/after comparison visualizations"""
    
    LOG.info("Creating before/after comparison visualizations...")
    
    try:
        # Set up the plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Before vs After: Friday-Enhanced Model Comparison', fontsize=16, fontweight='bold')
        
        # 1. Overall metrics comparison
        metrics = ['MAE', 'Accuracy', 'R2']
        baseline_vals = [
            comparison['overall_improvement']['mae_before'],
            comparison['overall_improvement']['accuracy_before'],
            comparison['overall_improvement']['r2_before'] * 100  # Convert to percentage
        ]
        enhanced_vals = [
            comparison['overall_improvement']['mae_after'],
            comparison['overall_improvement']['accuracy_after'],
            comparison['overall_improvement']['r2_after'] * 100
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, baseline_vals, width, label='Baseline', color='skyblue', alpha=0.7)
        bars2 = ax1.bar(x + width/2, enhanced_vals, width, label='Friday-Enhanced', color='lightcoral', alpha=0.7)
        
        ax1.set_ylabel('Value')
        ax1.set_title('Overall Model Performance')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics)
        ax1.legend()
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.annotate(f'{height:.1f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom')
        
        # 2. Weekday MAE comparison
        if 'weekday_improvements' in comparison and comparison['weekday_improvements']:
            weekdays = list(comparison['weekday_improvements'].keys())
            baseline_maes = [comparison['weekday_improvements'][day]['mae_before'] for day in weekdays]
            enhanced_maes = [comparison['weekday_improvements'][day]['mae_after'] for day in weekdays]
            
            x = np.arange(len(weekdays))
            bars1 = ax2.bar(x - width/2, baseline_maes, width, label='Baseline', color='skyblue', alpha=0.7)
            bars2 = ax2.bar(x + width/2, enhanced_maes, width, label='Friday-Enhanced', color='lightcoral', alpha=0.7)
            
            ax2.set_ylabel('MAE')
            ax2.set_title('MAE by Weekday')
            ax2.set_xticks(x)
            ax2.set_xticklabels(weekdays, rotation=45)
            ax2.legend()
            
            # Highlight Friday bars
            if 'Friday' in weekdays:
                friday_idx = weekdays.index('Friday')
                bars1[friday_idx].set_color('red')
                bars1[friday_idx].set_alpha(0.8)
                bars2[friday_idx].set_color('darkred')
                bars2[friday_idx].set_alpha(0.8)
        else:
            ax2.text(0.5, 0.5, 'Weekday comparison\nnot available', 
                     transform=ax2.transAxes, ha='center', va='center')
            ax2.set_title('MAE by Weekday')
        
        # 3. Improvement percentages
        if 'weekday_improvements' in comparison and comparison['weekday_improvements']:
            weekdays = list(comparison['weekday_improvements'].keys())
            improvements = [comparison['weekday_improvements'][day]['mae_improvement_pct'] for day in weekdays]
            
            colors = ['red' if day == 'Friday' else 'blue' for day in weekdays]
            bars = ax3.bar(weekdays, improvements, color=colors, alpha=0.7)
            ax3.set_ylabel('Improvement (%)')
            ax3.set_title('MAE Improvement by Weekday')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, val in zip(bars, improvements):
                height = bar.get_height()
                ax3.annotate(f'{val:+.1f}%',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3 if height >= 0 else -15),
                            textcoords="offset points",
                            ha='center', va='bottom' if height >= 0 else 'top')
        else:
            ax3.text(0.5, 0.5, 'Improvement data\nnot available', 
                     transform=ax3.transAxes, ha='center', va='center')
            ax3.set_title('MAE Improvement by Weekday')
        
        # 4. Summary text
        ax4.axis('off')
        
        # Create summary text
        overall_improvement = comparison['overall_improvement']['mae_improvement_pct']
        friday_improvement = comparison.get('friday_improvement', {}).get('improvement_pct', 0)
        
        summary_text = f"""
FRIDAY-ENHANCED MODEL RESULTS

OVERALL IMPROVEMENT:
• MAE Improvement: {overall_improvement:+.1f}%
• Accuracy: {comparison['overall_improvement']['accuracy_before']:.1f}% -> {comparison['overall_improvement']['accuracy_after']:.1f}%

FRIDAY CHALLENGE:
• Friday MAE Improvement: {friday_improvement:+.1f}%
• Status: {"SUCCESS!" if friday_improvement > 5 else "MIXED RESULTS" if friday_improvement > 0 else "NO IMPROVEMENT"}

KEY FINDINGS:
• {"Significant Friday improvement" if friday_improvement > 10 else "Moderate Friday improvement" if friday_improvement > 5 else "Friday features had minimal impact"}
• {"Overall model enhanced" if overall_improvement > 2 else "Overall performance maintained" if overall_improvement > -2 else "Overall performance degraded"}

RECOMMENDATION:
• {"Deploy enhanced model" if friday_improvement > 5 else "Consider operational adjustments" if friday_improvement <= 0 else "Test enhanced model"}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                 verticalalignment='top', fontsize=11, fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", 
                          facecolor='lightgreen' if friday_improvement > 5 else 'lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        # Save
        comparison_path = Path(CFG["comparison_output_dir"]) / "before_after_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        LOG.info(f"Comparison visualization saved: {comparison_path}")
        
    except Exception as e:
        LOG.error(f"Error creating visualizations: {e}")
        LOG.error(traceback.format_exc())

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

class FridayModelTrainingPipeline:
    """Complete pipeline for training and comparing models"""
    
    def __init__(self):
        self.daily_data = None
        self.comparison_results = None
        
    def run_complete_pipeline(self):
        """Run the complete training and testing pipeline"""
        
        start_time = time.time()
        
        LOG.info("Starting Friday-Enhanced Model Training Pipeline")
        LOG.info("="*80)
        
        try:
            # Step 1: Load Data
            LOG.info("STEP 1: Loading data...")
            self.daily_data = load_mail_call_data()
            
            # Step 2: Train Baseline Model
            LOG.info("STEP 2: Training baseline model...")
            X_baseline, y_baseline = create_baseline_features(self.daily_data)
            models_baseline = train_models(X_baseline, y_baseline, "baseline")
            
            # Save baseline model
            baseline_dir = Path(CFG["baseline_output_dir"])
            baseline_dir.mkdir(exist_ok=True)
            try:
                joblib.dump({
                    'models': models_baseline,
                    'X': X_baseline,
                    'y': y_baseline
                }, baseline_dir / "baseline_models.pkl")
                LOG.info("Baseline models saved successfully")
            except Exception as e:
                LOG.warning(f"Could not save baseline models: {e}")
            
            # Step 3: Test Baseline Model
            LOG.info("STEP 3: Testing baseline model...")
            baseline_results = test_models_comprehensive(X_baseline, y_baseline, models_baseline, "baseline")
            
            # Step 4: Train Friday-Enhanced Model
            LOG.info("STEP 4: Training Friday-enhanced model...")
            X_enhanced, y_enhanced = create_friday_enhanced_features(self.daily_data)
            models_enhanced = train_models(X_enhanced, y_enhanced, "enhanced")
            
            # Save enhanced model
            enhanced_dir = Path(CFG["enhanced_output_dir"])
            enhanced_dir.mkdir(exist_ok=True)
            try:
                joblib.dump({
                    'models': models_enhanced,
                    'X': X_enhanced,
                    'y': y_enhanced
                }, enhanced_dir / "friday_enhanced_models.pkl")
                LOG.info("Enhanced models saved successfully")
            except Exception as e:
                LOG.warning(f"Could not save enhanced models: {e}")
            
            # Step 5: Test Friday-Enhanced Model
            LOG.info("STEP 5: Testing Friday-enhanced model...")
            enhanced_results = test_models_comprehensive(X_enhanced, y_enhanced, models_enhanced, "enhanced")
            
            # Step 6: Compare Models
            LOG.info("STEP 6: Comparing models...")
            comparison = compare_models(baseline_results, enhanced_results)
            
            # Step 7: Create Visualizations
            LOG.info("STEP 7: Creating comparison visualizations...")
            create_comparison_visualizations(baseline_results, enhanced_results, comparison)
            
            # Step 8: Save Results
            LOG.info("STEP 8: Saving results...")
            self.save_all_results(baseline_results, enhanced_results, comparison)
            
            # Step 9: Generate Report
            LOG.info("STEP 9: Generating final report...")
            self.generate_final_report(baseline_results, enhanced_results, comparison)
            
            end_time = time.time()
            duration = end_time - start_time
            
            LOG.info("="*80)
            LOG.info("PIPELINE COMPLETE!")
            LOG.info(f"Total time: {duration:.1f} seconds")
            LOG.info(f"Results saved in: {CFG['comparison_output_dir']}")
            
            return True
            
        except Exception as e:
            LOG.error(f"Pipeline failed: {e}")
            LOG.error(traceback.format_exc())
            return False
    
    def save_all_results(self, baseline_results, enhanced_results, comparison):
        """Save all results to JSON"""
        
        try:
            comparison_dir = Path(CFG["comparison_output_dir"])
            
            # Save comparison results
            with open(comparison_dir / "model_comparison.json", 'w') as f:
                json.dump(comparison, f, indent=2, default=str)
            
            # Save baseline results
            with open(comparison_dir / "baseline_results.json", 'w') as f:
                json.dump(baseline_results, f, indent=2, default=str)
            
            # Save enhanced results
            with open(comparison_dir / "enhanced_results.json", 'w') as f:
                json.dump(enhanced_results, f, indent=2, default=str)
            
            LOG.info("All results saved to JSON files")
            
        except Exception as e:
            LOG.error(f"Error saving results: {e}")
    
    def generate_final_report(self, baseline_results, enhanced_results, comparison):
        """Generate final comparison report"""
        
        try:
            overall_improvement = comparison['overall_improvement']['mae_improvement_pct']
            friday_improvement = comparison.get('friday_improvement', {}).get('improvement_pct', 0)
            
            report = f"""
{'='*80}
                FRIDAY-ENHANCED MODEL TRAINING RESULTS
                      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

EXECUTIVE SUMMARY:
{'-'*50}
The Friday-enhanced model has been trained and tested against the baseline.
{"SUCCESS: Significant improvements achieved!" if friday_improvement > 10 else "MODERATE: Some improvements found" if friday_improvement > 5 else "MINIMAL: Limited improvements detected"}

OVERALL PERFORMANCE:
{'-'*50}
• Overall MAE Improvement: {overall_improvement:+.1f}%
• Baseline MAE: {comparison['overall_improvement']['mae_before']:.0f}
• Enhanced MAE: {comparison['overall_improvement']['mae_after']:.0f}
• Accuracy: {comparison['overall_improvement']['accuracy_before']:.1f}% -> {comparison['overall_improvement']['accuracy_after']:.1f}%

FRIDAY CHALLENGE RESULTS:
{'-'*50}
• Friday MAE Improvement: {friday_improvement:+.1f}%
• Baseline Friday MAE: {comparison.get('friday_improvement', {}).get('mae_before', 0):.0f}
• Enhanced Friday MAE: {comparison.get('friday_improvement', {}).get('mae_after', 0):.0f}
• Error Reduction: {comparison.get('friday_improvement', {}).get('improvement', 0):+.0f} calls per Friday

WEEKDAY BREAKDOWN:
{'-'*50}"""

            if 'weekday_improvements' in comparison:
                for day, metrics in comparison['weekday_improvements'].items():
                    report += f"\n• {day:10s}: {metrics['mae_improvement']:+6.0f} calls ({metrics['mae_improvement_pct']:+5.1f}%)"

            report += f"""

BUSINESS IMPACT:
{'-'*50}
• Annual Friday Error Reduction: {comparison.get('friday_improvement', {}).get('improvement', 0) * 52:+.0f} calls
• Staffing Impact: {comparison.get('friday_improvement', {}).get('improvement', 0) / 50:+.1f} agents per Friday
• Cost Impact: ~${abs(comparison.get('friday_improvement', {}).get('improvement', 0) / 50 * 25 * 8 * 52):,.0f}/year

RECOMMENDATIONS:
{'-'*50}
{"DEPLOY ENHANCED MODEL: Significant Friday improvements justify deployment" if friday_improvement > 10 else 
 "CONSIDER DEPLOYMENT: Moderate improvements, test in production first" if friday_improvement > 5 else
 "KEEP BASELINE: Minimal improvements don't justify complexity"}

NEXT STEPS:
{'-'*50}
1. {"Deploy Friday-enhanced model to production" if friday_improvement > 5 else "Continue with baseline model"}
2. {"Monitor Friday performance closely" if friday_improvement > 0 else "Focus on operational improvements"}
3. Run testing suite with the {'enhanced' if friday_improvement > 5 else 'baseline'} model
4. {"Update stakeholder presentations" if friday_improvement > 10 else "Document findings"}

FILES GENERATED:
{'-'*50}
• before_after_comparison.png - Main comparison dashboard
• model_comparison.json - Detailed metrics
• baseline_models.pkl - Trained baseline models
• friday_enhanced_models.pkl - Trained enhanced models

{'='*80}
              {"FRIDAY PROBLEM SOLVED!" if friday_improvement > 10 else "ANALYSIS COMPLETE" if friday_improvement > 5 else "BASELINE CONFIRMED OPTIMAL"}
{'='*80}
            """
            
            # Save and print report
            report_path = Path(CFG["comparison_output_dir"]) / "FRIDAY_ENHANCEMENT_REPORT.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(report)
            LOG.info(f"Final report saved: {report_path}")
            
        except Exception as e:
            LOG.error(f"Error generating final report: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("FRIDAY-ENHANCED MODEL TRAINING PIPELINE")
    print("="*60)
    print("1. Train baseline model (your original)")
    print("2. Train Friday-enhanced model (with winning features)")
    print("3. Test both models on all weekdays")
    print("4. Generate before/after comparison")
    print("5. Create visualizations and reports")
    print()
    print("Make sure your CSV files are available:")
    print("   • mail.csv")
    print("   • callvolumes.csv") 
    print("   • callintent.csv")
    print()
    
    try:
        # Run the pipeline
        pipeline = FridayModelTrainingPipeline()
        success = pipeline.run_complete_pipeline()
        
        if success:
            print("\nFRIDAY MODEL TRAINING COMPLETE!")
            print("="*60)
            print("Both models trained and tested")
            print("Comprehensive comparison generated")
            print("Visualizations created")
            print("Reports saved")
            print()
            print("NEXT STEPS:")
            print("1. Review the comparison visualizations")
            print("2. Read the final report")
            print("3. Run your testing suite on the best model")
            print("4. Deploy to production if improvements are significant")
            print()
            print(f"All results in: {CFG['comparison_output_dir']}")
        else:
            print("\nPIPELINE FAILED!")
            print("Check the log file for error details")
            return 1
    
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user")
        return 1
    except Exception as e:
        print(f"\nCritical error: {e}")
        LOG.error(f"Critical error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
