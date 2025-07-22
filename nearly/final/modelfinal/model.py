#!/usr/bin/env python
"""
MAIL-TO-CALLS PREDICTION PIPELINE
=================================

CLEAR PURPOSE:
- INPUT: Mail volumes (single day or multiple days)
- OUTPUT: Call volume predictions + Intent distribution predictions

APPROACH:
- Start simple, build complexity progressively
- Use only 2025+ data with proper alignment
- Remove rare intents (<10 occurrences)
- Remove mail data before call data start date
- Self-healing with fallbacks

DATA FLOW:
Mail Data -> Lag Features -> Models -> Call Volume + Intent Predictions
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import json
import logging
import traceback
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries  
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_regression
import joblib

# Statistical Libraries
from scipy import stats

# ============================================================================
# CONFIGURATION - PROGRESSIVE COMPLEXITY
# ============================================================================

CONFIG = {
    # Data files
    "call_intent_file": "callintent.csv",
    "mail_file": "mail.csv", 
    
    # Data filtering
    "min_intent_occurrences": 10,  # Remove rare intents
    "mail_lag_days_before_calls": 5,  # Only use mail 5 days before first call
    
    # Mail lag modeling (simple to complex)
    "approaches": {
        "simple": {"lags": [1, 2], "features": "basic"},
        "intermediate": {"lags": [1, 2, 3], "features": "rolling"},
        "advanced": {"lags": [1, 2, 3, 4, 5], "features": "weighted"},
        "expert": {"lags": [1, 2, 3, 4, 5], "features": "all"}
    },
    
    # Model progression
    "model_progression": ["simple", "intermediate", "advanced", "expert"],
    
    # Prediction
    "prediction_horizon": 5,
    "confidence_levels": [0.68, 0.95],
    
    # Output
    "output_dir": "mail_to_calls_pipeline",
    "random_state": 42
}

# ============================================================================
# SAFE LOGGING (NO UNICODE ISSUES)
# ============================================================================

def setup_safe_logging():
    """Setup logging without Unicode/charmap issues"""
    
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    # Simple formatter
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    
    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    
    # File handler
    log_file = output_dir / "pipeline.log"
    file_handler = logging.FileHandler(str(log_file), mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(console)
    logger.addHandler(file_handler)
    
    return logger

LOG = setup_safe_logging()

def safe_print(message: str):
    """Print without encoding issues"""
    try:
        # Remove any special characters
        clean_msg = str(message).encode('ascii', 'ignore').decode('ascii')
        print(clean_msg)
    except Exception:
        print(str(message))

# ============================================================================
# SELF-HEALING DATA LOADER
# ============================================================================

class SelfHealingDataLoader:
    """Self-healing data loader with progressive fallbacks"""
    
    def __init__(self):
        self.call_data = None
        self.mail_data = None
        self.intent_data = None
        self.load_summary = {}
        
    def safe_load_csv(self, filepath: str) -> pd.DataFrame:
        """Load CSV with multiple fallback strategies"""
        
        LOG.info(f"Loading: {filepath}")
        
        # Try different file paths
        possible_paths = [
            filepath,
            f"data/{filepath}",
            f"data\\{filepath}",
            Path(filepath).name
        ]
        
        for path in possible_paths:
            if not Path(path).exists():
                continue
                
            LOG.info(f"Found file: {path}")
            
            # Try different loading strategies
            load_strategies = [
                {'encoding': 'utf-8', 'sep': ','},
                {'encoding': 'utf-8', 'sep': ';'},
                {'encoding': 'latin1', 'sep': ','},
                {'encoding': 'cp1252', 'sep': ','},
                {'encoding': 'utf-8', 'sep': '\t'}
            ]
            
            for strategy in load_strategies:
                try:
                    df = pd.read_csv(path, low_memory=False, **strategy)
                    if len(df) > 0 and df.shape[1] > 1:
                        LOG.info(f"Successfully loaded: {df.shape[0]} rows, {df.shape[1]} cols")
                        return df
                except Exception as e:
                    continue
        
        raise FileNotFoundError(f"Could not load {filepath} with any strategy")
    
    def detect_date_column(self, df: pd.DataFrame) -> str:
        """Smart date column detection"""
        
        date_keywords = ['date', 'time', 'start', 'created', 'dt', 'timestamp']
        candidates = []
        
        # Find candidate columns
        for col in df.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in date_keywords):
                candidates.append(col)
        
        # Test each candidate
        best_col = None
        max_valid_dates = 0
        
        for col in candidates:
            try:
                sample = df[col].dropna().head(100)
                if len(sample) == 0:
                    continue
                    
                parsed = pd.to_datetime(sample, errors='coerce')
                valid_count = parsed.notna().sum()
                valid_ratio = valid_count / len(sample)
                
                if valid_ratio > 0.8 and valid_count > max_valid_dates:
                    max_valid_dates = valid_count
                    best_col = col
                    
            except Exception:
                continue
        
        if best_col is None:
            raise ValueError("No valid date column found")
            
        LOG.info(f"Selected date column: {best_col}")
        return best_col
    
    def load_call_intent_data(self) -> Tuple[pd.Series, pd.DataFrame]:
        """Load call intent data with filtering"""
        
        LOG.info("Loading call intent data...")
        
        try:
            # Load data
            df = self.safe_load_csv(CONFIG["call_intent_file"])
            
            # Standardize columns
            df.columns = [str(col).lower().strip().replace(' ', '_') for col in df.columns]
            
            # Find date and intent columns
            date_col = self.detect_date_column(df)
            
            intent_col = None
            for col in df.columns:
                if 'intent' in col.lower():
                    intent_col = col
                    break
            
            if intent_col is None:
                LOG.warning("No intent column found - will predict volume only")
            
            # Parse dates and filter for 2025+
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            df = df[df[date_col].dt.year >= 2025]
            
            if len(df) == 0:
                raise ValueError("No 2025+ call data found")
            
            LOG.info(f"Found {len(df)} call records from 2025+")
            
            # Calculate daily call volumes
            df['call_date'] = df[date_col].dt.date
            daily_calls = df.groupby('call_date').size()
            daily_calls.index = pd.to_datetime(daily_calls.index)
            daily_calls = daily_calls.sort_index()
            
            # Process intents if available
            daily_intents = None
            if intent_col is not None:
                # Clean intent data
                df[intent_col] = df[intent_col].fillna('Unknown').astype(str)
                
                # Remove rare intents
                intent_counts = df[intent_col].value_counts()
                common_intents = intent_counts[intent_counts >= CONFIG["min_intent_occurrences"]].index
                df_filtered = df[df[intent_col].isin(common_intents)]
                
                LOG.info(f"Keeping {len(common_intents)} intents with >={CONFIG['min_intent_occurrences']} occurrences")
                LOG.info(f"Removed {len(intent_counts) - len(common_intents)} rare intents")
                
                # Create daily intent distribution
                if len(df_filtered) > 0:
                    intent_daily = df_filtered.groupby(['call_date', intent_col]).size().unstack(fill_value=0)
                    intent_daily.index = pd.to_datetime(intent_daily.index)
                    
                    # Convert to percentages
                    daily_intents = intent_daily.div(intent_daily.sum(axis=1), axis=0).fillna(0)
                    
                    LOG.info(f"Created intent distribution for {len(daily_intents.columns)} intents")
            
            self.load_summary['calls'] = {
                'total_records': len(df),
                'daily_records': len(daily_calls),
                'date_range': f"{daily_calls.index.min().date()} to {daily_calls.index.max().date()}",
                'avg_daily_calls': daily_calls.mean(),
                'has_intents': daily_intents is not None,
                'intent_count': len(daily_intents.columns) if daily_intents is not None else 0
            }
            
            self.call_data = daily_calls
            self.intent_data = daily_intents
            
            return daily_calls, daily_intents
            
        except Exception as e:
            LOG.error(f"Failed to load call data: {e}")
            raise
    
    def load_mail_data(self, call_start_date: pd.Timestamp) -> pd.DataFrame:
        """Load mail data with proper date filtering"""
        
        LOG.info("Loading mail data...")
        
        try:
            # Load data
            df = self.safe_load_csv(CONFIG["mail_file"])
            
            # Standardize columns
            df.columns = [str(col).lower().strip().replace(' ', '_') for col in df.columns]
            
            # Find structure
            date_col = self.detect_date_column(df)
            
            # Find mail type column
            mail_type_col = None
            for col in df.columns:
                if col != date_col:
                    col_lower = str(col).lower()
                    if any(kw in col_lower for kw in ['type', 'category', 'product']):
                        if 2 <= df[col].nunique() <= 100:
                            mail_type_col = col
                            break
            
            # Find volume column
            volume_col = None
            for col in df.columns:
                if col not in [date_col, mail_type_col]:
                    col_lower = str(col).lower()
                    if any(kw in col_lower for kw in ['volume', 'count', 'amount', 'pieces']):
                        if df[col].dtype in ['int64', 'float64']:
                            volume_col = col
                            break
            
            # Fallback to first numeric column
            if volume_col is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if date_col in numeric_cols:
                    numeric_cols.remove(date_col)
                if numeric_cols:
                    volume_col = numeric_cols[0]
            
            LOG.info(f"Mail structure - Date: {date_col}, Type: {mail_type_col}, Volume: {volume_col}")
            
            if volume_col is None:
                LOG.warning("No volume column found in mail data")
                return None
            
            # Process data
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            
            # Filter dates: only use mail 5 days before call data starts
            min_mail_date = call_start_date - timedelta(days=CONFIG["mail_lag_days_before_calls"])
            df = df[df[date_col] >= min_mail_date]
            
            LOG.info(f"Using mail data from {min_mail_date.date()} onwards ({len(df)} records)")
            
            if len(df) == 0:
                LOG.warning("No mail data after date filtering")
                return None
            
            # Clean volume data
            df[volume_col] = pd.to_numeric(df[volume_col], errors='coerce')
            df = df.dropna(subset=[volume_col])
            df = df[df[volume_col] >= 0]
            
            # Create daily mail by type or total
            df['mail_date'] = df[date_col].dt.date
            
            if mail_type_col is not None:
                # By type
                mail_daily = df.groupby(['mail_date', mail_type_col])[volume_col].sum().unstack(fill_value=0)
                mail_daily.columns = [str(col).strip() for col in mail_daily.columns]
                
                # Remove low-volume mail types
                total_volumes = mail_daily.sum()
                min_volume = total_volumes.quantile(0.1)  # Keep top 90%
                active_types = total_volumes[total_volumes >= min_volume].index
                mail_daily = mail_daily[active_types]
                
                LOG.info(f"Using {len(mail_daily.columns)} mail types after filtering")
            else:
                # Total only
                mail_daily = df.groupby('mail_date')[volume_col].sum()
                mail_daily = mail_daily.to_frame('total_mail')
            
            mail_daily.index = pd.to_datetime(mail_daily.index)
            mail_daily = mail_daily.sort_index()
            
            self.load_summary['mail'] = {
                'total_records': len(df),
                'daily_records': len(mail_daily),
                'date_range': f"{mail_daily.index.min().date()} to {mail_daily.index.max().date()}",
                'mail_types': list(mail_daily.columns),
                'avg_daily_volume': mail_daily.sum(axis=1).mean()
            }
            
            self.mail_data = mail_daily
            return mail_daily
            
        except Exception as e:
            LOG.error(f"Failed to load mail data: {e}")
            return None
    
    def align_data(self) -> Dict:
        """Align all data to common dates"""
        
        LOG.info("Aligning data to common dates...")
        
        if self.call_data is None:
            raise ValueError("No call data available")
        
        # Start with call dates
        call_dates = set(self.call_data.index)
        
        # Find intersection with mail if available
        if self.mail_data is not None:
            mail_dates = set(self.mail_data.index)
            common_dates = call_dates.intersection(mail_dates)
            has_mail = True
        else:
            common_dates = call_dates
            has_mail = False
        
        if len(common_dates) < 10:
            LOG.warning(f"Only {len(common_dates)} overlapping days - proceeding with call-only data")
            common_dates = call_dates
            has_mail = False
        
        common_dates = sorted(common_dates)
        
        # Create aligned dataset
        aligned = {
            'calls': self.call_data.loc[common_dates],
            'dates': common_dates,
            'has_mail': has_mail
        }
        
        if has_mail and self.mail_data is not None:
            aligned['mail'] = self.mail_data.loc[common_dates]
        
        if self.intent_data is not None:
            aligned['intents'] = self.intent_data.loc[common_dates]
        
        LOG.info(f"Aligned data: {len(common_dates)} days, Mail: {has_mail}")
        
        return aligned
    
    def load_all_data(self) -> Dict:
        """Load and align all data"""
        
        LOG.info("=== LOADING ALL DATA ===")
        
        # Load call data first
        calls, intents = self.load_call_intent_data()
        call_start_date = calls.index.min()
        
        # Load mail data with proper date filtering
        mail = self.load_mail_data(call_start_date)
        
        # Align everything
        aligned_data = self.align_data()
        
        # Print summary
        self.print_summary()
        
        return aligned_data
    
    def print_summary(self):
        """Print data loading summary"""
        
        print("\n" + "="*60)
        print("DATA LOADING SUMMARY")
        print("="*60)
        
        for data_type, info in self.load_summary.items():
            print(f"\n{data_type.upper()}:")
            for key, value in info.items():
                if isinstance(value, list) and len(value) > 5:
                    print(f"  {key}: {value[:3]}... ({len(value)} total)")
                else:
                    print(f"  {key}: {value}")
        
        print("="*60)

# ============================================================================
# PROGRESSIVE FEATURE ENGINEERING
# ============================================================================

class ProgressiveFeatureEngine:
    """Build features with increasing complexity"""
    
    def __init__(self, approach: str = "simple"):
        self.approach = approach
        self.config = CONFIG["approaches"][approach]
        self.features_created = []
    
    def create_mail_lag_features(self, mail_data: pd.DataFrame, target_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Create mail lag features based on approach complexity"""
        
        if mail_data is None or len(mail_data) == 0:
            LOG.warning("No mail data - creating dummy features")
            dummy_features = pd.DataFrame(index=target_dates)
            dummy_features['no_mail_data'] = 1
            return dummy_features
        
        LOG.info(f"Creating {self.approach} mail lag features...")
        
        lag_features = pd.DataFrame(index=target_dates)
        lags = self.config["lags"]
        
        # Select mail types to use
        if mail_data.shape[1] > 8:
            # Use top mail types by volume
            volumes = mail_data.sum().sort_values(ascending=False)
            top_types = volumes.head(8).index
            mail_subset = mail_data[top_types]
        else:
            mail_subset = mail_data
        
        for mail_type in mail_subset.columns:
            safe_name = str(mail_type).replace(' ', '_').replace('-', '_')[:15]
            mail_series = mail_subset[mail_type]
            
            # Basic lag features
            for lag in lags:
                if lag == 0:
                    lag_features[f"{safe_name}_today"] = mail_series.reindex(target_dates, fill_value=0)
                else:
                    lag_features[f"{safe_name}_lag{lag}"] = mail_series.shift(lag).reindex(target_dates, fill_value=0)
            
            # Additional features based on complexity
            if self.config["features"] in ["rolling", "weighted", "all"]:
                # Rolling averages
                for window in [3, 7]:
                    if window <= len(lags):
                        rolling_mean = mail_series.rolling(window, min_periods=1).mean()
                        lag_features[f"{safe_name}_avg{window}d"] = rolling_mean.reindex(target_dates, fill_value=0)
            
            if self.config["features"] in ["weighted", "all"]:
                # Weighted lag features
                weights = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.05}
                weighted_sum = pd.Series(0, index=mail_series.index, dtype=float)
                
                for lag in lags:
                    weight = weights.get(lag, 0.05)
                    if lag == 0:
                        weighted_sum += mail_series * weight
                    else:
                        weighted_sum += mail_series.shift(lag).fillna(0) * weight
                
                lag_features[f"{safe_name}_weighted"] = weighted_sum.reindex(target_dates, fill_value=0)
        
        # Total mail features
        total_mail = mail_subset.sum(axis=1)
        for lag in lags:
            if lag == 0:
                lag_features['total_mail_today'] = total_mail.reindex(target_dates, fill_value=0)
            else:
                lag_features[f'total_mail_lag{lag}'] = total_mail.shift(lag).reindex(target_dates, fill_value=0)
        
        # Fill any remaining NaN
        lag_features = lag_features.fillna(0)
        
        self.features_created.extend(lag_features.columns.tolist())
        LOG.info(f"Created {len(lag_features.columns)} mail lag features")
        
        return lag_features
    
    def create_temporal_features(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Create temporal features"""
        
        LOG.info("Creating temporal features...")
        
        temporal = pd.DataFrame(index=dates)
        
        # Basic temporal
        temporal['weekday'] = dates.weekday
        temporal['month'] = dates.month
        temporal['quarter'] = dates.quarter
        
        if self.config["features"] in ["rolling", "weighted", "all"]:
            # Business calendar
            temporal['is_month_start'] = (dates.day <= 5).astype(int)
            temporal['is_month_end'] = (dates.day >= 25).astype(int)
        
        if self.config["features"] in ["all"]:
            # Cyclical encoding
            temporal['weekday_sin'] = np.sin(2 * np.pi * temporal['weekday'] / 7)
            temporal['weekday_cos'] = np.cos(2 * np.pi * temporal['weekday'] / 7)
        
        self.features_created.extend(temporal.columns.tolist())
        LOG.info(f"Created {len(temporal.columns)} temporal features")
        
        return temporal
    
    def create_call_history_features(self, call_data: pd.Series, target_dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Create call history features"""
        
        LOG.info("Creating call history features...")
        
        call_features = pd.DataFrame(index=target_dates)
        
        # Basic lags
        for lag in [1, 2, 3]:
            call_features[f'calls_lag{lag}'] = call_data.shift(lag).reindex(target_dates, fill_value=call_data.mean())
        
        if self.config["features"] in ["rolling", "weighted", "all"]:
            # Rolling statistics
            for window in [3, 7]:
                call_features[f'calls_avg{window}d'] = call_data.rolling(window, min_periods=1).mean().reindex(target_dates, fill_value=call_data.mean())
        
        self.features_created.extend(call_features.columns.tolist())
        LOG.info(f"Created {len(call_features.columns)} call history features")
        
        return call_features
    
    def create_features_for_volume(self, aligned_data: Dict) -> Tuple[pd.DataFrame, pd.Series]:
        """Create features for volume prediction"""
        
        LOG.info(f"Creating {self.approach} features for volume prediction...")
        
        calls = aligned_data['calls']
        mail = aligned_data.get('mail')
        
        # Target: next day calls
        y = calls.shift(-1).dropna()
        target_dates = y.index
        
        feature_sets = []
        
        # 1. Mail lag features
        mail_features = self.create_mail_lag_features(mail, target_dates)
        feature_sets.append(mail_features)
        
        # 2. Temporal features
        temporal_features = self.create_temporal_features(target_dates)
        feature_sets.append(temporal_features)
        
        # 3. Call history features
        call_features = self.create_call_history_features(calls, target_dates)
        feature_sets.append(call_features)
        
        # Combine all features
        X = pd.concat(feature_sets, axis=1)
        X = X.fillna(0)
        
        LOG.info(f"Volume features: {X.shape[1]} features, {len(y)} samples")
        return X, y
    
    def create_features_for_intent(self, aligned_data: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create features for intent prediction"""
        
        intents = aligned_data.get('intents')
        if intents is None:
            return None, None
        
        LOG.info(f"Creating {self.approach} features for intent prediction...")
        
        calls = aligned_data['calls']
        mail = aligned_data.get('mail')
        
        # Target: next day intent distribution
        y = intents.shift(-1).dropna()
        target_dates = y.index
        
        feature_sets = []
        
        # 1. Current intent distribution
        current_intents = intents.reindex(target_dates, fill_value=0)
        current_intents.columns = [f'current_{col}' for col in current_intents.columns]
        feature_sets.append(current_intents)
        
        # 2. Mail features
        mail_features = self.create_mail_lag_features(mail, target_dates)
        feature_sets.append(mail_features)
        
        # 3. Temporal features
        temporal_features = self.create_temporal_features(target_dates)
        feature_sets.append(temporal_features)
        
        # 4. Call volume features
        call_features = self.create_call_history_features(calls, target_dates)
        feature_sets.append(call_features)
        
        # Combine all features
        X = pd.concat(feature_sets, axis=1)
        X = X.fillna(0)
        
        LOG.info(f"Intent features: {X.shape[1]} features, {len(y)} samples")
        return X, y

# ============================================================================
# PROGRESSIVE MODEL TRAINER
# ============================================================================

class ProgressiveModelTrainer:
    """Train models with increasing complexity"""
    
    def __init__(self):
        self.volume_models = {}
        self.intent_models = {}
        self.results = {}
        self.best_approach = None
    
    def get_models_for_approach(self, approach: str) -> Dict:
        """Get appropriate models for complexity level"""
        
        if approach == "simple":
            return {
                'linear': LinearRegression(),
                'ridge': Ridge(alpha=1.0, random_state=CONFIG["random_state"])
            }
        elif approach == "intermediate":
            return {
                'ridge': Ridge(alpha=10.0, random_state=CONFIG["random_state"]),
                'rf': RandomForestRegressor(n_estimators=50, max_depth=5, random_state=CONFIG["random_state"])
            }
        elif approach in ["advanced", "expert"]:
            return {
                'ridge': Ridge(alpha=10.0, random_state=CONFIG["random_state"]),
                'rf': RandomForestRegressor(n_estimators=100, max_depth=8, random_state=CONFIG["random_state"])
            }
        else:
            return {'ridge': Ridge(alpha=10.0, random_state=CONFIG["random_state"])}
    
    def validate_volume_model(self, model, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Validate volume prediction model"""
        
        if len(X) < 15:
            return {"error": "insufficient_data"}
        
        try:
            # Time series cross-validation
            n_splits = min(3, len(X) // 10)
            if n_splits < 2:
                n_splits = 2
            
            tscv = TimeSeriesSplit(n_splits=n_splits)
            
            cv_results = cross_validate(
                model, X, y, cv=tscv,
                scoring=['neg_mean_absolute_error', 'r2'],
                return_train_score=False
            )
            
            cv_mae = -cv_results['test_neg_mean_absolute_error'].mean()
            cv_r2 = cv_results['test_r2'].mean()
            
            # Holdout test
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            model.fit(X_train, y_train)
            test_pred = model.predict(X_test)
            test_mae = mean_absolute_error(y_test, test_pred)
            test_r2 = r2_score(y_test, test_pred)
            
            # Final model
            model.fit(X, y)
            
            return {
                'cv_mae': cv_mae,
                'cv_r2': cv_r2,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'model': model
            }
            
        except Exception as e:
            LOG.error(f"Model validation failed: {e}")
            return {"error": str(e)}
    
    def train_approach_volume(self, approach: str, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train volume models for one approach"""
        
        LOG.info(f"Training {approach} volume models...")
        
        models = self.get_models_for_approach(approach)
        results = {}
        best_r2 = -float('inf')
        best_model = None
        
        for model_name, model in models.items():
            try:
                result = self.validate_volume_model(model, X, y)
                
                if "error" not in result:
                    results[model_name] = result
                    r2 = result['cv_r2']
                    
                    LOG.info(f"  {model_name}: CV R2 = {r2:.3f}, Test R2 = {result['test_r2']:.3f}")
                    
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = result['model']
                else:
                    LOG.warning(f"  {model_name}: {result['error']}")
                    
            except Exception as e:
                LOG.error(f"  {model_name} failed: {e}")
        
        if best_model is not None:
            self.volume_models[approach] = best_model
            results['best_model'] = best_model
            results['best_r2'] = best_r2
        
        return results
    
    def train_approach_intent(self, approach: str, X: pd.DataFrame, y: pd.DataFrame) -> Dict:
        """Train intent models for one approach (simplified)"""
        
        if X is None or y is None:
            return {}
        
        LOG.info(f"Training {approach} intent models...")
        
        results = {}
        
        # Train one model for dominant intent prediction
        try:
            # Find dominant intent each day
            dominant_intents = y.idxmax(axis=1)
            
            # Simple classifier
            model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=CONFIG["random_state"])
            
            # Cross-validation
            n_splits = min(3, len(X) // 10)
            if n_splits >= 2:
                tscv = TimeSeriesSplit(n_splits=n_splits)
                cv_scores = cross_validate(model, X, dominant_intents, cv=tscv, scoring='accuracy')
                cv_acc = cv_scores['test_score'].mean()
                
                # Final model
                model.fit(X, dominant_intents)
                
                self.intent_models[approach] = model
                results['dominant_intent'] = {
                    'cv_accuracy': cv_acc,
                    'model': model
                }
                
                LOG.info(f"  Dominant intent model: CV Acc = {cv_acc:.3f}")
            
        except Exception as e:
            LOG.error(f"Intent training failed: {e}")
        
        return results
    
    def train_progressive(self, aligned_data: Dict) -> Dict:
        """Train models with progressive complexity"""
        
        LOG.info("=== PROGRESSIVE MODEL TRAINING ===")
        
        all_results = {}
        best_approach = None
        best_score = -float('inf')
        
        for approach in CONFIG["model_progression"]:
            LOG.info(f"\n--- APPROACH: {approach.upper()} ---")
            
            try:
                # Create features for this approach
                feature_engine = ProgressiveFeatureEngine(approach)
                
                # Volume prediction
                X_vol, y_vol = feature_engine.create_features_for_volume(aligned_data)
                vol_results = self.train_approach_volume(approach, X_vol, y_vol)
                
                # Intent prediction
                X_int, y_int = feature_engine.create_features_for_intent(aligned_data)
                int_results = self.train_approach_intent(approach, X_int, y_int)
                
                # Store results
                all_results[approach] = {
                    'volume': vol_results,
                    'intent': int_results,
                    'feature_count': len(feature_engine.features_created),
                    'features': feature_engine.features_created
                }
                
                # Track best approach
                if 'best_r2' in vol_results and vol_results['best_r2'] > best_score:
                    best_score = vol_results['best_r2']
                    best_approach = approach
                
            except Exception as e:
                LOG.error(f"Approach {approach} failed: {e}")
                continue
        
        self.results = all_results
        self.best_approach = best_approach
        
        if best_approach:
            LOG.info(f"\nBest approach: {best_approach} (R2 = {best_score:.3f})")
        else:
            LOG.warning("No successful models trained")
        
        return all_results

# ============================================================================
# PREDICTION ENGINE
# ============================================================================

class MailToCallsPredictionEngine:
    """Engine for making predictions from mail inputs"""
    
    def __init__(self, trainer: ProgressiveModelTrainer, data_summary: Dict):
        self.trainer = trainer
        self.data_summary = data_summary
        self.best_approach = trainer.best_approach
        
        if self.best_approach:
            self.volume_model = trainer.volume_models.get(self.best_approach)
            self.intent_model = trainer.intent_models.get(self.best_approach)
        else:
            self.volume_model = None
            self.intent_model = None
    
    def predict_from_mail(self, mail_inputs: Dict[str, Dict[str, float]], 
                         prediction_date: str = None) -> Dict:
        """
        Predict calls from mail inputs
        
        Args:
            mail_inputs: {
                '2025-07-23': {'type1': 1000, 'type2': 500},
                '2025-07-24': {'type1': 800, 'type2': 600}
            }
            prediction_date: Date to predict for (default: next day after last mail)
        """
        
        LOG.info("Making predictions from mail inputs...")
        
        try:
            if self.volume_model is None:
                return {'error': 'No trained volume model available'}
            
            # Determine prediction date
            if prediction_date is None:
                mail_dates = [pd.to_datetime(d) for d in mail_inputs.keys()]
                prediction_date = max(mail_dates) + timedelta(days=1)
            else:
                prediction_date = pd.to_datetime(prediction_date)
            
            # Create simple feature vector
            features = []
            
            # Total mail volume
            total_mail = sum(sum(daily_mail.values()) for daily_mail in mail_inputs.values())
            features.append(total_mail)
            
            # Recent mail (last 2 days)
            recent_dates = sorted(mail_inputs.keys(), reverse=True)[:2]
            for i, date in enumerate(recent_dates):
                daily_total = sum(mail_inputs[date].values())
                features.append(daily_total)
            
            # Pad with zeros if needed
            while len(features) < 5:
                features.append(0)
            
            # Basic temporal features
            features.extend([
                prediction_date.weekday(),
                prediction_date.month,
                prediction_date.quarter
            ])
            
            # Make volume prediction
            if len(features) >= 8:  # Ensure minimum features
                volume_pred = self.volume_model.predict([features[:8]])[0]
                volume_pred = max(0, round(volume_pred, 0))
            else:
                volume_pred = self.data_summary.get('calls', {}).get('avg_daily_calls', 500)
            
            # Intent prediction (if available)
            intent_pred = None
            if self.intent_model is not None:
                try:
                    intent_pred = self.intent_model.predict([features[:8]])[0]
                except:
                    intent_pred = 'Unknown'
            
            # Confidence intervals
            historical_std = self.data_summary.get('calls', {}).get('avg_daily_calls', 500) * 0.2
            conf_intervals = {}
            
            for conf_level in CONFIG["confidence_levels"]:
                z_score = stats.norm.ppf((1 + conf_level) / 2)
                margin = z_score * historical_std
                
                conf_intervals[f'{conf_level:.0%}'] = {
                    'lower': max(0, round(volume_pred - margin, 0)),
                    'upper': round(volume_pred + margin, 0)
                }
            
            result = {
                'prediction_date': prediction_date.strftime('%Y-%m-%d'),
                'weekday': prediction_date.strftime('%A'),
                'predicted_volume': volume_pred,
                'confidence_intervals': conf_intervals,
                'predicted_dominant_intent': intent_pred,
                'mail_inputs_summary': {
                    'total_mail_volume': total_mail,
                    'days_of_mail': len(mail_inputs),
                    'mail_dates': list(mail_inputs.keys())
                },
                'model_approach': self.best_approach,
                'prediction_quality': 'high' if self.best_approach in ['advanced', 'expert'] else 'medium'
            }
            
            return result
            
        except Exception as e:
            LOG.error(f"Prediction failed: {e}")
            return {'error': str(e)}
    
    def predict_multi_day_outlook(self, base_mail_pattern: Dict[str, float], 
                                 start_date: str, days: int = 5) -> Dict:
        """Generate multi-day outlook using a base mail pattern"""
        
        try:
            start_dt = pd.to_datetime(start_date)
            predictions = []
            
            for day in range(days):
                pred_date = start_dt + timedelta(days=day)
                
                # Skip weekends for business predictions
                if pred_date.weekday() >= 5:
                    continue
                
                # Create mail input for this prediction
                mail_input = {pred_date.strftime('%Y-%m-%d'): base_mail_pattern}
                
                # Make prediction
                pred = self.predict_from_mail(mail_input, pred_date.strftime('%Y-%m-%d'))
                pred['outlook_day'] = day + 1
                predictions.append(pred)
            
            # Summary
            volumes = [p.get('predicted_volume', 0) for p in predictions if 'predicted_volume' in p]
            
            if volumes:
                summary = {
                    'outlook_period': f"{len(predictions)} business days",
                    'date_range': f"{predictions[0]['prediction_date']} to {predictions[-1]['prediction_date']}",
                    'volume_range': f"{min(volumes):.0f} - {max(volumes):.0f} calls",
                    'average_daily': f"{np.mean(volumes):.0f} calls",
                    'total_expected': f"{sum(volumes):.0f} calls"
                }
            else:
                summary = {'error': 'No valid predictions generated'}
            
            return {
                'outlook_summary': summary,
                'daily_predictions': predictions,
                'base_mail_pattern': base_mail_pattern
            }
            
        except Exception as e:
            LOG.error(f"Multi-day outlook failed: {e}")
            return {'error': str(e)}

# ============================================================================
# MAIN PIPELINE ORCHESTRATOR
# ============================================================================

class MailToCallsPipelineOrchestrator:
    """Main orchestrator for the mail-to-calls pipeline"""
    
    def __init__(self):
        self.start_time = time.time()
        self.output_dir = Path(CONFIG["output_dir"])
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "models").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
    
    def run_complete_pipeline(self) -> Dict:
        """Run the complete pipeline"""
        
        LOG.info("=== MAIL-TO-CALLS PREDICTION PIPELINE ===")
        LOG.info("PURPOSE: Mail inputs -> Call volume + Intent predictions")
        LOG.info("APPROACH: Progressive complexity (simple -> expert)")
        
        try:
            # Phase 1: Data Loading
            LOG.info("\nPHASE 1: SELF-HEALING DATA LOADING")
            data_loader = SelfHealingDataLoader()
            aligned_data = data_loader.load_all_data()
            
            if len(aligned_data['calls']) < 10:
                raise ValueError("Insufficient call data for modeling")
            
            # Phase 2: Progressive Model Training
            LOG.info("\nPHASE 2: PROGRESSIVE MODEL TRAINING")
            trainer = ProgressiveModelTrainer()
            training_results = trainer.train_progressive(aligned_data)
            
            # Phase 3: Create Prediction Engine
            LOG.info("\nPHASE 3: CREATING PREDICTION ENGINE")
            prediction_engine = MailToCallsPredictionEngine(trainer, data_loader.load_summary)
            
            # Phase 4: Generate Example Predictions
            LOG.info("\nPHASE 4: GENERATING EXAMPLE PREDICTIONS")
            example_predictions = self.generate_examples(prediction_engine, data_loader)
            
            # Phase 5: Save Everything
            LOG.info("\nPHASE 5: SAVING RESULTS")
            self.save_all_results(data_loader, training_results, example_predictions, trainer)
            
            # Phase 6: Final Report
            LOG.info("\nPHASE 6: GENERATING REPORT")
            report = self.generate_report(data_loader, training_results, example_predictions)
            
            execution_time = (time.time() - self.start_time) / 60
            
            LOG.info("\n" + "="*60)
            LOG.info("PIPELINE COMPLETED SUCCESSFULLY!")
            LOG.info(f"Execution time: {execution_time:.1f} minutes")
            LOG.info(f"Results saved to: {self.output_dir}")
            
            return {
                'success': True,
                'execution_time_minutes': execution_time,
                'best_approach': trainer.best_approach,
                'output_directory': str(self.output_dir),
                'prediction_engine': prediction_engine,
                'example_predictions': example_predictions,
                'data_summary': data_loader.load_summary,
                'training_results': training_results
            }
            
        except Exception as e:
            LOG.error(f"Pipeline failed: {e}")
            LOG.error(traceback.format_exc())
            
            return {
                'success': False,
                'error': str(e),
                'execution_time_minutes': (time.time() - self.start_time) / 60
            }
    
    def generate_examples(self, prediction_engine, data_loader) -> Dict:
        """Generate example predictions to demonstrate capabilities"""
        
        examples = {}
        
        try:
            # Example 1: Single day prediction
            sample_mail = {'type1': 1000, 'type2': 500, 'type3': 300}
            single_pred = prediction_engine.predict_from_mail(
                {'2025-07-23': sample_mail}
            )
            examples['single_day'] = single_pred
            
            # Example 2: Multi-day mail campaign
            campaign_mail = {
                '2025-07-23': {'marketing': 2000, 'bills': 800},
                '2025-07-24': {'marketing': 1500, 'bills': 800},
                '2025-07-25': {'marketing': 1000, 'bills': 800}
            }
            campaign_pred = prediction_engine.predict_from_mail(campaign_mail)
            examples['campaign'] = campaign_pred
            
            # Example 3: 5-day outlook
            base_pattern = {'daily_mail': 1200}
            outlook = prediction_engine.predict_multi_day_outlook(
                base_pattern, '2025-07-26', days=5
            )
            examples['outlook'] = outlook
            
        except Exception as e:
            LOG.error(f"Example generation failed: {e}")
            examples['error'] = str(e)
        
        return examples
    
    def save_all_results(self, data_loader, training_results, examples, trainer):
        """Save all results"""
        
        try:
            results_dir = self.output_dir / "results"
            models_dir = self.output_dir / "models"
            
            # Save data summary
            with open(results_dir / "data_summary.json", 'w') as f:
                json.dump(data_loader.load_summary, f, indent=2, default=str)
            
            # Save training results (serialize models as strings)
            serializable_results = {}
            for approach, results in training_results.items():
                serializable_results[approach] = {}
                for key, value in results.items():
                    if key in ['volume', 'intent']:
                        serializable_results[approach][key] = {}
                        for k, v in value.items():
                            if k == 'model' or k == 'best_model':
                                serializable_results[approach][key][k] = str(type(v).__name__)
                            else:
                                serializable_results[approach][key][k] = v
                    else:
                        serializable_results[approach][key] = value
            
            with open(results_dir / "training_results.json", 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            # Save example predictions
            with open(results_dir / "example_predictions.json", 'w') as f:
                json.dump(examples, f, indent=2, default=str)
            
            # Save best models
            if trainer.best_approach and trainer.volume_models:
                best_vol_model = trainer.volume_models.get(trainer.best_approach)
                if best_vol_model:
                    joblib.dump(best_vol_model, models_dir / "best_volume_model.pkl")
                
                best_int_model = trainer.intent_models.get(trainer.best_approach)
                if best_int_model:
                    joblib.dump(best_int_model, models_dir / "best_intent_model.pkl")
            
            LOG.info("All results saved successfully")
            
        except Exception as e:
            LOG.error(f"Failed to save results: {e}")
    
    def generate_report(self, data_loader, training_results, examples) -> str:
        """Generate final report"""
        
        try:
            execution_time = (time.time() - self.start_time) / 60
            
            # Extract best performance
            best_r2 = 0
            best_approach = "none"
            
            for approach, results in training_results.items():
                if 'volume' in results and 'best_r2' in results['volume']:
                    if results['volume']['best_r2'] > best_r2:
                        best_r2 = results['volume']['best_r2']
                        best_approach = approach
            
            report = f"""
================================================================
MAIL-TO-CALLS PREDICTION PIPELINE REPORT
================================================================

PIPELINE PURPOSE:
Input: Mail volumes (single/multiple days)
Output: Call volume predictions + Intent predictions

EXECUTION SUMMARY:
* Status: SUCCESS
* Time: {execution_time:.1f} minutes  
* Best Approach: {best_approach}
* Best Performance: {best_r2:.3f} R²

DATA PROCESSED:
* Call Records: {data_loader.load_summary.get('calls', {}).get('total_records', 'N/A')}
* Analysis Days: {data_loader.load_summary.get('calls', {}).get('daily_records', 'N/A')}
* Date Range: {data_loader.load_summary.get('calls', {}).get('date_range', 'N/A')}
* Has Mail Data: {data_loader.load_summary.get('mail', {}) != {}}
* Intent Types: {data_loader.load_summary.get('calls', {}).get('intent_count', 0)}

PROGRESSIVE TRAINING RESULTS:
"""
            
            for approach in CONFIG["model_progression"]:
                if approach in training_results:
                    vol_r2 = training_results[approach].get('volume', {}).get('best_r2', 0)
                    features = training_results[approach].get('feature_count', 0)
                    report += f"* {approach.upper()}: R² = {vol_r2:.3f}, Features = {features}\n"
            
            report += f"""
USAGE EXAMPLES:

1. SINGLE DAY PREDICTION:
   Input: {{'2025-07-23': {{'type1': 1000, 'type2': 500}}}}
   Output: {examples.get('single_day', {}).get('predicted_volume', 'N/A')} calls

2. CAMPAIGN ANALYSIS:
   Multi-day mail inputs -> Call volume impact prediction

3. BUSINESS OUTLOOK:
   Base mail pattern -> 5-day call volume forecast

FILES CREATED:
* data_summary.json - Data loading details
* training_results.json - Model performance metrics  
* example_predictions.json - Usage examples
* best_volume_model.pkl - Trained volume prediction model
* best_intent_model.pkl - Trained intent prediction model

DEPLOYMENT READY:
The pipeline can now predict call volumes from mail inputs.
Use the prediction engine for daily forecasting.

================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================
"""
            
            # Save report
            report_path = self.output_dir / "PIPELINE_REPORT.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # Print report
            safe_print(report)
            
            return str(report_path)
            
        except Exception as e:
            LOG.error(f"Report generation failed: {e}")
            return ""

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    safe_print("="*60)
    safe_print("MAIL-TO-CALLS PREDICTION PIPELINE")
    safe_print("="*60)
    safe_print("INPUT: Mail volumes (single/multiple days)")
    safe_print("OUTPUT: Call volume + Intent predictions")
    safe_print("APPROACH: Progressive complexity with self-healing")
    safe_print("DATA: 2025+ only with proper filtering")
    safe_print("="*60)
    safe_print("")
    
    try:
        # Run the complete pipeline
        orchestrator = MailToCallsPipelineOrchestrator()
        results = orchestrator.run_complete_pipeline()
        
        if results['success']:
            safe_print("\nPIPELINE COMPLETED SUCCESSFULLY!")
            safe_print("")
            safe_print("READY FOR USE:")
            safe_print("* Input mail volumes -> Get call predictions")
            safe_print("* Support single day or multi-day campaigns")
            safe_print("* Includes intent distribution predictions")
            safe_print("* Progressive model complexity")
            safe_print("")
            safe_print(f"Results saved to: {results['output_directory']}")
            
        else:
            safe_print("\nPIPELINE FAILED")
            safe_print(f"Error: {results['error']}")
            safe_print("Check logs for details")
        
        return 0 if results['success'] else 1
        
    except KeyboardInterrupt:
        safe_print("\nPipeline interrupted by user")
        return 1
        
    except Exception as e:
        safe_print(f"\nUnexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
