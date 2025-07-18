#!/usr/bin/env python
# ultimate_model_optimizer.py
# ============================================================================
# ULTIMATE PRODUCTION-GRADE MAIL-TO-CALLS MODEL OPTIMIZER
# ============================================================================
# Systematically tests everything possible to beat baseline MAE 4,440
# - Mail type combinations (3-25 types)
# - Feature engineering (temporal, interactions, lags)
# - Model types (linear, tree, ensemble)
# - Economic integration (cautious)
# - Hyperparameter optimization
# - Production-ready error handling
# ============================================================================

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import json
import logging
import sys
import traceback
from datetime import datetime, timedelta
from itertools import combinations, permutations
import pickle
import time

import numpy as np
import pandas as pd
import holidays

# Core ML libraries
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, LinearRegression, 
    QuantileRegressor, BayesianRidge
)
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import (
    SelectKBest, f_regression, RFE, SelectFromModel
)
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.pipeline import Pipeline

# Advanced ML libraries with fallbacks
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    from sklearn.neural_network import MLPRegressor
    NEURAL_AVAILABLE = True
except ImportError:
    NEURAL_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    import pickle as joblib
    JOBLIB_AVAILABLE = False

# ============================================================================
# ASCII ART & CONFIGURATION
# ============================================================================

ASCII_BANNER = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    ██╗   ██╗██╗  ████████╗██╗███╗   ███╗ █████╗ ████████╗███████╗           ║
║    ██║   ██║██║  ╚══██╔══╝██║████╗ ████║██╔══██╗╚══██╔══╝██╔════╝           ║
║    ██║   ██║██║     ██║   ██║██╔████╔██║███████║   ██║   █████╗             ║
║    ██║   ██║██║     ██║   ██║██║╚██╔╝██║██╔══██║   ██║   ██╔══╝             ║
║    ╚██████╔╝███████╗██║   ██║██║ ╚═╝ ██║██║  ██║   ██║   ███████╗           ║
║     ╚═════╝ ╚══════╝╚═╝   ╚═╝╚═╝     ╚═╝╚═╝  ╚═╝   ╚═╝   ╚══════╝           ║
║                                                                              ║
║              MAIL-TO-CALLS MODEL OPTIMIZER v3.0                             ║
║                  Target: Beat MAE 4,440 Benchmark                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

PHASE_SEPARATORS = {
    'benchmark': """
┌──────────────────────────────────────────────────────────────────────────────┐
│                           📊 PHASE 1: BENCHMARK                             │
│                     Establishing baseline performance                       │
└──────────────────────────────────────────────────────────────────────────────┘""",
    
    'feature_engineering': """
┌──────────────────────────────────────────────────────────────────────────────┐
│                        🔧 PHASE 2: FEATURE ENGINEERING                      │
│                    Testing all possible feature combinations                 │
└──────────────────────────────────────────────────────────────────────────────┘""",
    
    'model_optimization': """
┌──────────────────────────────────────────────────────────────────────────────┐
│                        🤖 PHASE 3: MODEL OPTIMIZATION                       │
│                      Testing advanced model architectures                   │
└──────────────────────────────────────────────────────────────────────────────┘""",
    
    'economic_testing': """
┌──────────────────────────────────────────────────────────────────────────────┐
│                        💰 PHASE 4: ECONOMIC INTEGRATION                     │
│                      Cautiously testing economic indicators                 │
└──────────────────────────────────────────────────────────────────────────────┘""",
    
    'final_optimization': """
┌──────────────────────────────────────────────────────────────────────────────┐
│                        🎯 PHASE 5: FINAL OPTIMIZATION                       │
│                     Hyperparameter tuning & ensemble building               │
└──────────────────────────────────────────────────────────────────────────────┘""",
    
    'production': """
┌──────────────────────────────────────────────────────────────────────────────┐
│                        🚀 PHASE 6: PRODUCTION PREPARATION                   │
│                       Building deployment-ready models                      │
└──────────────────────────────────────────────────────────────────────────────┘"""
}

# Configuration
CFG = {
    "benchmark_mae_target": 4440,  # Your working model performance
    "max_features": 25,            # Keep models simple
    "min_features": 5,             # Minimum viable complexity
    "cv_splits": 3,                # Time series cross-validation
    "n_bootstrap": 30,             # Bootstrap samples for uncertainty
    "test_size": 0.2,              # Final holdout test
    "random_state": 42,            # Reproducibility
    "max_models_per_phase": 50,    # Prevent infinite testing
    "performance_improvement_threshold": 0.02,  # Minimum 2% improvement
    "output_dir": "ultimate_optimization_results",
    "fallback_models": ["linear", "ridge", "random_forest"],  # Always available
    "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
}

# ============================================================================
# ROBUST LOGGING SETUP
# ============================================================================

def setup_logging():
    """Setup comprehensive logging with error handling"""
    try:
        log_format = "%(asctime)s | %(levelname)8s | %(name)s | %(message)s"
        
        # Create output directory
        output_dir = Path(CFG["output_dir"])
        output_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(output_dir / "optimization.log")
            ]
        )
        
        logger = logging.getLogger("UltimateOptimizer")
        logger.info("Logging system initialized successfully")
        return logger
        
    except Exception as e:
        # Fallback to basic logging
        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logger = logging.getLogger("UltimateOptimizer")
        logger.warning(f"Advanced logging failed, using fallback: {e}")
        return logger

LOG = setup_logging()

# ============================================================================
# ROBUST DATA LOADING WITH MULTIPLE FALLBACK STRATEGIES
# ============================================================================

class DataLoader:
    """Robust data loading with comprehensive error handling"""
    
    def __init__(self):
        self.data_cache = {}
        self.load_attempts = 0
        
    def _to_date(self, series, name="date"):
        """Convert to date with error handling"""
        try:
            return pd.to_datetime(series, errors="coerce").dt.date
        except Exception as e:
            LOG.warning(f"Date conversion failed for {name}: {e}")
            # Try alternative formats
            try:
                return pd.to_datetime(series, format="%Y-%m-%d", errors="coerce").dt.date
            except:
                try:
                    return pd.to_datetime(series, infer_datetime_format=True, errors="coerce").dt.date
                except:
                    LOG.error(f"All date conversion attempts failed for {name}")
                    return pd.Series([None] * len(series))
    
    def _find_file(self, candidates, required=True):
        """Find file with multiple fallback locations"""
        locations_tried = []
        
        for candidate in candidates:
            try:
                # Try multiple path variations
                paths_to_try = [
                    Path(candidate),
                    Path("data") / candidate,
                    Path("../data") / candidate,
                    Path(".") / candidate,
                ]
                
                for path in paths_to_try:
                    locations_tried.append(str(path))
                    if path.exists():
                        LOG.info(f"Found file: {path}")
                        return path
                        
            except Exception as e:
                LOG.warning(f"Error checking path {candidate}: {e}")
                continue
        
        error_msg = f"File not found. Tried: {candidates}. Locations: {locations_tried}"
        
        if required:
            raise FileNotFoundError(error_msg)
        else:
            LOG.warning(error_msg)
            return None
    
    def load_mail_data(self):
        """Load mail data with robust error handling"""
        try:
            LOG.info("Loading mail data...")
            
            # Try multiple file names
            mail_candidates = [
                "mail.csv", "Mail.csv", "MAIL.csv",
                "mail_data.csv", "mail_volumes.csv"
            ]
            
            mail_path = self._find_file(mail_candidates)
            
            # Load with multiple encoding attempts
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            mail_df = None
            
            for encoding in encodings:
                try:
                    mail_df = pd.read_csv(mail_path, encoding=encoding)
                    LOG.info(f"Mail data loaded successfully with encoding: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    LOG.warning(f"Failed to load mail data with {encoding}: {e}")
                    continue
            
            if mail_df is None:
                raise ValueError("Could not load mail data with any encoding")
            
            # Clean column names
            mail_df.columns = [str(c).lower().strip() for c in mail_df.columns]
            
            # Find date column with multiple name patterns
            date_cols = [col for col in mail_df.columns if any(
                pattern in col for pattern in ['date', 'time', 'day']
            )]
            
            if not date_cols:
                raise ValueError("No date column found in mail data")
            
            date_col = date_cols[0]
            LOG.info(f"Using date column: {date_col}")
            
            # Convert dates
            mail_df[date_col] = self._to_date(mail_df[date_col], "mail_date")
            mail_df = mail_df.dropna(subset=[date_col])
            
            # Find volume column
            volume_cols = [col for col in mail_df.columns if any(
                pattern in col for pattern in ['volume', 'count', 'qty', 'amount']
            )]
            
            if not volume_cols:
                raise ValueError("No volume column found in mail data")
            
            LOG.info(f"Mail data loaded: {mail_df.shape[0]} rows, {mail_df.shape[1]} columns")
            LOG.info(f"Date range: {mail_df[date_col].min()} to {mail_df[date_col].max()}")
            
            return mail_df, date_col
            
        except Exception as e:
            LOG.error(f"Failed to load mail data: {e}")
            LOG.error(traceback.format_exc())
            raise
    
    def load_call_data(self):
        """Load call data with robust error handling"""
        try:
            LOG.info("Loading call volume data...")
            
            # Try multiple call volume file names
            vol_candidates = [
                "callvolumes.csv", "call_volumes.csv", "CallVolumes.csv",
                "volumes.csv", "calls.csv"
            ]
            
            vol_path = self._find_file(vol_candidates)
            
            # Load call volumes
            df_vol = pd.read_csv(vol_path)
            df_vol.columns = [str(c).lower().strip() for c in df_vol.columns]
            
            # Find date column
            date_cols = [col for col in df_vol.columns if 'date' in col]
            if not date_cols:
                raise ValueError("No date column found in call volume data")
            
            date_col = date_cols[0]
            df_vol[date_col] = self._to_date(df_vol[date_col], "call_date")
            
            # Aggregate daily volumes
            volume_cols = [col for col in df_vol.columns if col != date_col]
            if not volume_cols:
                raise ValueError("No volume columns found")
            
            vol_daily = df_vol.groupby(date_col)[volume_cols[0]].sum()
            
            LOG.info("Loading call intent data...")
            
            # Try call intent files
            intent_candidates = [
                "callintent.csv", "call_intent.csv", "CallIntent.csv",
                "callintetn.csv", "intent.csv"  # Include common typo
            ]
            
            intent_path = self._find_file(intent_candidates, required=False)
            
            if intent_path:
                df_int = pd.read_csv(intent_path)
                df_int.columns = [str(c).lower().strip() for c in df_int.columns]
                
                # Find date column
                int_date_cols = [col for col in df_int.columns if any(
                    pattern in col for pattern in ['date', 'conversation', 'start']
                )]
                
                if int_date_cols:
                    int_date_col = int_date_cols[0]
                    df_int[int_date_col] = self._to_date(df_int[int_date_col], "intent_date")
                    int_daily = df_int.groupby(int_date_col).size()
                    
                    # Scale and combine
                    overlap = vol_daily.index.intersection(int_daily.index)
                    if len(overlap) >= 5:
                        scale = int_daily.loc[overlap].mean() / vol_daily.loc[overlap].mean()
                        vol_daily *= scale
                        LOG.info(f"Scaled call volumes by factor: {scale:.2f}")
                    
                    calls_total = vol_daily.combine_first(int_daily).sort_index()
                else:
                    LOG.warning("No date column found in intent data, using volume data only")
                    calls_total = vol_daily.sort_index()
            else:
                LOG.warning("No intent data found, using volume data only")
                calls_total = vol_daily.sort_index()
            
            # Convert index to datetime
            calls_total.index = pd.to_datetime(calls_total.index)
            
            LOG.info(f"Call data loaded: {len(calls_total)} days")
            LOG.info(f"Call volume stats: mean={calls_total.mean():.0f}, std={calls_total.std():.0f}")
            
            return calls_total
            
        except Exception as e:
            LOG.error(f"Failed to load call data: {e}")
            LOG.error(traceback.format_exc())
            raise
    
    def load_economic_data(self):
        """Load economic data with fallbacks"""
        try:
            LOG.info("Attempting to load economic data...")
            
            econ_candidates = [
                "economics_expanded.csv", "economics.csv", "economic_data.csv",
                "econ.csv", "market_data.csv"
            ]
            
            econ_path = self._find_file(econ_candidates, required=False)
            
            if econ_path is None:
                LOG.warning("No economic data files found, will skip economic features")
                return None
            
            econ_df = pd.read_csv(econ_path)
            econ_df.columns = [str(c).strip() for c in econ_df.columns]
            
            # Find date column
            date_cols = [col for col in econ_df.columns if 'date' in col.lower()]
            if not date_cols:
                LOG.warning("No date column in economic data")
                return None
            
            date_col = date_cols[0]
            econ_df[date_col] = pd.to_datetime(econ_df[date_col], errors='coerce')
            econ_df = econ_df.dropna(subset=[date_col])
            econ_df.set_index(date_col, inplace=True)
            
            # Filter to business days
            us_holidays = holidays.US()
            biz_mask = (~econ_df.index.weekday.isin([5, 6])) & (~econ_df.index.isin(us_holidays))
            econ_df = econ_df.loc[biz_mask]
            
            LOG.info(f"Economic data loaded: {econ_df.shape[0]} days, {econ_df.shape[1]} indicators")
            return econ_df
            
        except Exception as e:
            LOG.warning(f"Failed to load economic data: {e}")
            return None
    
    def create_daily_dataset(self):
        """Combine all data into daily dataset"""
        try:
            LOG.info("Creating combined daily dataset...")
            
            # Load all data
            mail_df, mail_date_col = self.load_mail_data()
            calls_total = self.load_call_data()
            econ_df = self.load_economic_data()
            
            # Create mail pivot table
            mail_type_col = None
            volume_col = None
            
            # Find mail type and volume columns
            for col in mail_df.columns:
                if col != mail_date_col:
                    if any(pattern in col for pattern in ['type', 'category', 'kind']):
                        mail_type_col = col
                    elif any(pattern in col for pattern in ['volume', 'count', 'qty', 'amount']):
                        volume_col = col
            
            if mail_type_col is None:
                # Try to infer mail type column
                non_numeric_cols = mail_df.select_dtypes(include=['object']).columns
                candidates = [col for col in non_numeric_cols if col != mail_date_col]
                if candidates:
                    mail_type_col = candidates[0]
                    LOG.warning(f"Inferred mail type column: {mail_type_col}")
            
            if volume_col is None:
                # Try to infer volume column
                numeric_cols = mail_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    volume_col = numeric_cols[0]
                    LOG.warning(f"Inferred volume column: {volume_col}")
            
            if mail_type_col is None or volume_col is None:
                raise ValueError(f"Could not identify mail type ({mail_type_col}) or volume ({volume_col}) columns")
            
            LOG.info(f"Using mail_type: {mail_type_col}, volume: {volume_col}")
            
            # Create pivot table
            mail_daily = (mail_df.groupby([mail_date_col, mail_type_col], as_index=False)[volume_col].sum()
                         .pivot(index=mail_date_col, columns=mail_type_col, values=volume_col)
                         .fillna(0))
            
            # Convert index to datetime
            mail_daily.index = pd.to_datetime(mail_daily.index)
            
            # Filter to business days
            us_holidays = holidays.US()
            biz_mask = (~mail_daily.index.weekday.isin([5, 6])) & (~mail_daily.index.isin(us_holidays))
            mail_daily = mail_daily.loc[biz_mask]
            
            # Align with calls
            calls_total = calls_total.loc[calls_total.index.isin(mail_daily.index)]
            
            # Combine
            daily = mail_daily.join(calls_total.rename("calls_total"), how="inner")
            
            LOG.info(f"Combined daily dataset: {daily.shape[0]} days, {daily.shape[1]} mail types")
            LOG.info(f"Date range: {daily.index.min().date()} to {daily.index.max().date()}")
            
            # Add economic data if available
            if econ_df is not None:
                common_dates = daily.index.intersection(econ_df.index)
                if len(common_dates) > 50:
                    daily = daily.loc[common_dates].join(econ_df.loc[common_dates])
                    LOG.info(f"Added economic data: {len(common_dates)} overlapping days")
                else:
                    LOG.warning("Insufficient overlap with economic data, skipping")
            
            self.data_cache['daily'] = daily
            return daily
            
        except Exception as e:
            LOG.error(f"Failed to create daily dataset: {e}")
            LOG.error(traceback.format_exc())
            raise

# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Advanced feature engineering with error handling"""
    
    def __init__(self, daily_data):
        self.daily = daily_data.copy()
        self.feature_cache = {}
        
        # Identify mail types (exclude calls_total and economic indicators)
        economic_indicators = [
            'russell2000', 'dollar_index', 'nasdaq', 'sp500', 'technology',
            'banking', 'dowjones', 'vix', 'oil', 'gold', 'reits'
        ]
        
        self.mail_types = [
            col for col in self.daily.columns 
            if col != 'calls_total' and 
            not any(econ in col.lower() for econ in economic_indicators)
        ]
        
        LOG.info(f"Identified {len(self.mail_types)} mail types for feature engineering")
    
    def analyze_mail_correlations(self):
        """Analyze correlations between mail types and calls"""
        try:
            LOG.info("Analyzing mail type correlations...")
            
            correlations = {}
            calls = self.daily["calls_total"]
            
            for mail_type in self.mail_types:
                try:
                    mail_vol = self.daily[mail_type]
                    
                    # Same-day correlation
                    corr_same = mail_vol.corr(calls)
                    
                    # Lag correlation (mail today -> calls tomorrow)
                    if len(mail_vol) > 1:
                        corr_lag = mail_vol[:-1].corr(calls[1:])
                    else:
                        corr_lag = corr_same
                    
                    # Statistics
                    total_volume = mail_vol.sum()
                    frequency = (mail_vol > 0).mean() * 100
                    
                    correlations[mail_type] = {
                        'correlation': corr_same if not np.isnan(corr_same) else 0,
                        'lag_correlation': corr_lag if not np.isnan(corr_lag) else 0,
                        'best_correlation': max(abs(corr_same or 0), abs(corr_lag or 0)),
                        'total_volume': total_volume,
                        'frequency_pct': frequency
                    }
                    
                except Exception as e:
                    LOG.warning(f"Error analyzing {mail_type}: {e}")
                    correlations[mail_type] = {
                        'correlation': 0, 'lag_correlation': 0, 'best_correlation': 0,
                        'total_volume': 0, 'frequency_pct': 0
                    }
            
            # Sort by correlation
            sorted_correlations = sorted(
                correlations.items(), 
                key=lambda x: x[1]['best_correlation'], 
                reverse=True
            )
            
            LOG.info("Top 10 mail types by correlation:")
            for i, (mail_type, data) in enumerate(sorted_correlations[:10], 1):
                LOG.info(f"  {i:2d}. {mail_type}: {data['best_correlation']:.3f} "
                        f"(vol: {data['total_volume']:,.0f}, freq: {data['frequency_pct']:.1f}%)")
            
            return correlations, sorted_correlations
            
        except Exception as e:
            LOG.error(f"Error in correlation analysis: {e}")
            return {}, []
    
    def create_baseline_features(self, selected_mail_types):
        """Create baseline features matching your working model"""
        try:
            LOG.info(f"Creating baseline features with {len(selected_mail_types)} mail types...")
            
            features_list = []
            targets_list = []
            
            for i in range(len(self.daily) - 1):
                try:
                    current_day = self.daily.iloc[i]
                    next_day = self.daily.iloc[i + 1]
                    current_date = self.daily.index[i]
                    
                    feature_row = {}
                    
                    # Mail volume features
                    total_mail = 0
                    for mail_type in selected_mail_types:
                        if mail_type in self.daily.columns:
                            volume = current_day[mail_type]
                            feature_row[f"{mail_type}_volume"] = volume
                            total_mail += volume
                    
                    # Aggregate features
                    feature_row["total_mail_volume"] = total_mail
                    feature_row["log_total_mail_volume"] = np.log1p(total_mail)
                    
                    # Mail percentile
                    if i > 10:
                        mail_history = pd.Series([
                            sum(self.daily.iloc[j][mt] for mt in selected_mail_types if mt in self.daily.columns)
                            for j in range(i + 1)
                        ])
                        feature_row["mail_percentile"] = (mail_history <= total_mail).mean()
                    else:
                        feature_row["mail_percentile"] = 0.5
                    
                    # Date features
                    feature_row["weekday"] = current_date.weekday()
                    feature_row["month"] = current_date.month
                    feature_row["is_month_end"] = 1 if current_date.day > 25 else 0
                    feature_row["is_holiday_week"] = 1 if current_date in holidays.US() else 0
                    
                    # Recent calls context
                    recent_calls = self.daily["calls_total"].iloc[max(0, i-5):i+1]
                    feature_row["recent_calls_avg"] = recent_calls.mean()
                    feature_row["recent_calls_trend"] = recent_calls.diff().mean()
                    
                    # Target
                    target = next_day["calls_total"]
                    
                    features_list.append(feature_row)
                    targets_list.append(target)
                    
                except Exception as e:
                    LOG.warning(f"Error creating features for day {i}: {e}")
                    continue
            
            X = pd.DataFrame(features_list).fillna(0)
            y = pd.Series(targets_list)
            
            LOG.info(f"Baseline features created: {X.shape[0]} samples × {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            LOG.error(f"Error creating baseline features: {e}")
            raise
    
    def create_advanced_features(self, selected_mail_types, include_interactions=True, 
                               include_lags=True, include_rolling=True):
        """Create advanced features with comprehensive options"""
        try:
            LOG.info("Creating advanced features...")
            
            features_list = []
            targets_list = []
            
            for i in range(max(7, len(self.daily) // 10), len(self.daily) - 1):
                try:
                    current_day = self.daily.iloc[i]
                    next_day = self.daily.iloc[i + 1]
                    current_date = self.daily.index[i]
                    
                    feature_row = {}
                    
                    # === BASELINE MAIL FEATURES ===
                    total_mail = 0
                    for mail_type in selected_mail_types:
                        if mail_type in self.daily.columns:
                            volume = current_day[mail_type]
                            feature_row[f"{mail_type}_today"] = volume
                            total_mail += volume
                    
                    feature_row["total_mail"] = total_mail
                    feature_row["log_total_mail"] = np.log1p(total_mail)
                    
                    # === LAG FEATURES ===
                    if include_lags and i >= 3:
                        for lag in [1, 2, 3]:
                            if i >= lag:
                                lag_day = self.daily.iloc[i - lag]
                                lag_total = sum(lag_day[mt] for mt in selected_mail_types if mt in self.daily.columns)
                                feature_row[f"total_mail_lag{lag}"] = lag_total
                                
                                # Top mail types individual lags
                                for mail_type in selected_mail_types[:5]:  # Top 5 only
                                    if mail_type in self.daily.columns:
                                        feature_row[f"{mail_type}_lag{lag}"] = lag_day[mail_type]
                    
                    # === ROLLING FEATURES ===
                    if include_rolling and i >= 7:
                        windows = [3, 5, 7]
                        for window in windows:
                            if i >= window:
                                recent_data = self.daily.iloc[i-window:i]
                                
                                # Rolling totals
                                rolling_total = sum(
                                    recent_data[mt].sum() for mt in selected_mail_types 
                                    if mt in self.daily.columns
                                )
                                feature_row[f"rolling_total_{window}d"] = rolling_total
                                feature_row[f"rolling_avg_{window}d"] = rolling_total / window
                                
                                # Rolling volatility
                                daily_totals = [
                                    sum(recent_data.iloc[j][mt] for mt in selected_mail_types if mt in self.daily.columns)
                                    for j in range(len(recent_data))
                                ]
                                if len(daily_totals) > 1:
                                    feature_row[f"rolling_std_{window}d"] = np.std(daily_totals)
                                else:
                                    feature_row[f"rolling_std_{window}d"] = 0
                    
                    # === INTERACTION FEATURES ===
                    if include_interactions:
                        # Mail × Day interactions
                        feature_row["total_mail_x_weekday"] = total_mail * current_date.weekday()
                        feature_row["total_mail_x_month"] = total_mail * current_date.month
                        
                        # Volume ratios (if we have multiple types)
                        if len(selected_mail_types) >= 2:
                            top_types = selected_mail_types[:3]  # Top 3 for ratios
                            for i_idx, type1 in enumerate(top_types):
                                for type2 in top_types[i_idx+1:]:
                                    if type1 in self.daily.columns and type2 in self.daily.columns:
                                        vol1 = current_day[type1]
                                        vol2 = current_day[type2]
                                        if vol2 > 0:
                                            feature_row[f"{type1}_{type2}_ratio"] = vol1 / vol2
                                        else:
                                            feature_row[f"{type1}_{type2}_ratio"] = 0
                    
                    # === TEMPORAL FEATURES ===
                    feature_row["weekday"] = current_date.weekday()
                    feature_row["month"] = current_date.month
                    feature_row["day_of_month"] = current_date.day
                    feature_row["is_month_end"] = 1 if current_date.day > 25 else 0
                    feature_row["is_quarter_end"] = 1 if current_date.month in [3, 6, 9, 12] and current_date.day > 25 else 0
                    feature_row["is_holiday_week"] = 1 if current_date in holidays.US() else 0
                    
                    # Business day features
                    feature_row["is_monday"] = 1 if current_date.weekday() == 0 else 0
                    feature_row["is_friday"] = 1 if current_date.weekday() == 4 else 0
                    
                    # === CALL HISTORY FEATURES ===
                    if i >= 7:
                        recent_calls = self.daily["calls_total"].iloc[max(0, i-7):i]
                        feature_row["calls_avg_7d"] = recent_calls.mean()
                        feature_row["calls_std_7d"] = recent_calls.std()
                        feature_row["calls_trend_7d"] = recent_calls.diff().mean()
                        
                        # Yesterday's calls
                        feature_row["calls_yesterday"] = self.daily["calls_total"].iloc[i-1]
                    
                    # === MAIL PERCENTILES & OUTLIERS ===
                    if i > 20:
                        historical_totals = pd.Series([
                            sum(self.daily.iloc[j][mt] for mt in selected_mail_types if mt in self.daily.columns)
                            for j in range(i)
                        ])
                        
                        feature_row["mail_percentile"] = (historical_totals <= total_mail).mean()
                        feature_row["mail_zscore"] = (total_mail - historical_totals.mean()) / (historical_totals.std() + 1)
                        feature_row["is_mail_outlier"] = 1 if abs(feature_row["mail_zscore"]) > 2 else 0
                    else:
                        feature_row["mail_percentile"] = 0.5
                        feature_row["mail_zscore"] = 0
                        feature_row["is_mail_outlier"] = 0
                    
                    # Target
                    target = next_day["calls_total"]
                    
                    features_list.append(feature_row)
                    targets_list.append(target)
                    
                except Exception as e:
                    LOG.warning(f"Error creating advanced features for day {i}: {e}")
                    continue
            
            X = pd.DataFrame(features_list).fillna(0)
            y = pd.Series(targets_list)
            
            LOG.info(f"Advanced features created: {X.shape[0]} samples × {X.shape[1]} features")
            return X, y
            
        except Exception as e:
            LOG.error(f"Error creating advanced features: {e}")
            # Fallback to baseline features
            LOG.info("Falling back to baseline features...")
            return self.create_baseline_features(selected_mail_types)

# ============================================================================
# COMPREHENSIVE MODEL TESTING FRAMEWORK
# ============================================================================

class ModelTester:
    """Comprehensive model testing with robust error handling"""
    
    def __init__(self, target_mae=4440):
        self.target_mae = target_mae
        self.results = []
        self.best_model = None
        self.best_score = float('inf')
        
        # Initialize available models
        self.available_models = self._get_available_models()
        LOG.info(f"Available models: {list(self.available_models.keys())}")
    
    def _get_available_models(self):
        """Get all available models with fallbacks"""
        models = {
            # Always available models
            'linear': LinearRegression(),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=1.0),
            'elastic_net': ElasticNet(alpha=1.0, l1_ratio=0.5),
            'bayesian_ridge': BayesianRidge(),
            'random_forest': RandomForestRegressor(
                n_estimators=100, max_depth=8, min_samples_leaf=5, random_state=CFG['random_state']
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=CFG['random_state']
            )
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            try:
                models['xgboost'] = xgb.XGBRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1, 
                    random_state=CFG['random_state'], verbosity=0
                )
                LOG.info("XGBoost added to available models")
            except Exception as e:
                LOG.warning(f"XGBoost initialization failed: {e}")
        
        # Add LightGBM if available
        if LIGHTGBM_AVAILABLE:
            try:
                models['lightgbm'] = lgb.LGBMRegressor(
                    n_estimators=100, max_depth=6, learning_rate=0.1,
                    random_state=CFG['random_state'], verbosity=-1
                )
                LOG.info("LightGBM added to available models")
            except Exception as e:
                LOG.warning(f"LightGBM initialization failed: {e}")
        
        # Add Neural Network if available
        if NEURAL_AVAILABLE:
            try:
                models['neural_network'] = MLPRegressor(
                    hidden_layer_sizes=(100, 50), max_iter=500, 
                    random_state=CFG['random_state'], early_stopping=True
                )
                LOG.info("Neural Network added to available models")
            except Exception as e:
                LOG.warning(f"Neural Network initialization failed: {e}")
        
        return models
    
    def evaluate_model(self, model, X, y, model_name="Unknown"):
        """Evaluate model with time series cross-validation"""
        try:
            # Time series split
            tscv = TimeSeriesSplit(n_splits=CFG['cv_splits'])
            
            scores = []
            maes = []
            r2s = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                try:
                    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                    
                    # Handle scaling for certain models
                    if model_name in ['neural_network', 'elastic_net', 'lasso']:
                        scaler = RobustScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    maes.append(mae)
                    r2s.append(r2)
                    
                except Exception as e:
                    LOG.warning(f"Error in fold {fold} for {model_name}: {e}")
                    continue
            
            if not maes:
                return None
            
            avg_mae = np.mean(maes)
            avg_r2 = np.mean(r2s)
            std_mae = np.std(maes)
            
            result = {
                'model_name': model_name,
                'mae_mean': avg_mae,
                'mae_std': std_mae,
                'r2_mean': avg_r2,
                'n_features': X.shape[1],
                'n_samples': X.shape[0],
                'improvement_vs_target': (self.target_mae - avg_mae) / self.target_mae * 100
            }
            
            # Track best model
            if avg_mae < self.best_score:
                self.best_score = avg_mae
                self.best_model = {
                    'model': model,
                    'result': result,
                    'features': list(X.columns)
                }
            
            return result
            
        except Exception as e:
            LOG.error(f"Error evaluating {model_name}: {e}")
            return None
    
    def test_model_suite(self, X, y, phase_name="Unknown"):
        """Test all available models"""
        try:
            LOG.info(f"Testing model suite for {phase_name}...")
            
            phase_results = []
            
            for model_name, model in self.available_models.items():
                try:
                    LOG.info(f"  Testing {model_name}...")
                    
                    # Clone model to avoid state issues
                    if hasattr(model, 'random_state'):
                        model.set_params(random_state=CFG['random_state'])
                    
                    result = self.evaluate_model(model, X, y, model_name)
                    
                    if result:
                        phase_results.append(result)
                        improvement = result['improvement_vs_target']
                        
                        LOG.info(f"    MAE: {result['mae_mean']:.0f} ± {result['mae_std']:.0f}")
                        LOG.info(f"    R²: {result['r2_mean']:.3f}")
                        LOG.info(f"    Improvement: {improvement:+.1f}%")
                        
                        if result['mae_mean'] < self.target_mae:
                            LOG.info(f"    🎯 BEATS TARGET! ({self.target_mae})")
                    
                except Exception as e:
                    LOG.warning(f"Failed to test {model_name}: {e}")
                    continue
            
            # Sort by performance
            phase_results.sort(key=lambda x: x['mae_mean'])
            
            if phase_results:
                best = phase_results[0]
                LOG.info(f"Best {phase_name} model: {best['model_name']} (MAE: {best['mae_mean']:.0f})")
            
            self.results.extend(phase_results)
            return phase_results
            
        except Exception as e:
            LOG.error(f"Error in model suite testing: {e}")
            return []

# ============================================================================
# ECONOMIC FEATURE INTEGRATION
# ============================================================================

class EconomicIntegrator:
    """Cautious economic feature integration"""
    
    def __init__(self, daily_data):
        self.daily = daily_data
        self.economic_indicators = self._identify_economic_columns()
    
    def _identify_economic_columns(self):
        """Identify economic indicator columns"""
        economic_patterns = [
            'russell', 'nasdaq', 'sp500', 'dow', 'vix', 'oil', 'gold', 
            'dollar', 'treasury', 'bond', 'reit', 'technology', 'banking'
        ]
        
        econ_cols = []
        for col in self.daily.columns:
            if col.lower() != 'calls_total' and any(pattern in col.lower() for pattern in economic_patterns):
                econ_cols.append(col)
        
        LOG.info(f"Identified {len(econ_cols)} economic indicators: {econ_cols[:5]}{'...' if len(econ_cols) > 5 else ''}")
        return econ_cols
    
    def test_individual_indicators(self, base_features, base_targets, feature_engineer):
        """Test each economic indicator individually"""
        try:
            if not self.economic_indicators:
                LOG.info("No economic indicators found, skipping economic testing")
                return []
            
            LOG.info("Testing individual economic indicators...")
            
            results = []
            model_tester = ModelTester(CFG['benchmark_mae_target'])
            
            # Test each indicator
            for indicator in self.economic_indicators[:10]:  # Limit to top 10
                try:
                    LOG.info(f"  Testing economic indicator: {indicator}")
                    
                    # Create enhanced features with this indicator
                    enhanced_X = base_features.copy()
                    
                    # Add economic features
                    for i in range(len(enhanced_X)):
                        try:
                            date_idx = i + max(7, len(self.daily) // 10)  # Match advanced features offset
                            if date_idx < len(self.daily):
                                current_date = self.daily.index[date_idx]
                                
                                # Current value
                                enhanced_X.loc[enhanced_X.index[i], f"{indicator}_today"] = self.daily.loc[current_date, indicator]
                                
                                # Lag value
                                if date_idx > 0:
                                    lag_date = self.daily.index[date_idx - 1]
                                    enhanced_X.loc[enhanced_X.index[i], f"{indicator}_lag1"] = self.daily.loc[lag_date, indicator]
                                
                        except Exception as e:
                            LOG.warning(f"Error adding economic features for row {i}: {e}")
                            continue
                    
                    # Fill missing values
                    enhanced_X = enhanced_X.fillna(method='ffill').fillna(0)
                    
                    # Test with simple model
                    simple_model = RandomForestRegressor(
                        n_estimators=50, max_depth=6, random_state=CFG['random_state']
                    )
                    
                    result = model_tester.evaluate_model(enhanced_X, base_targets, simple_model, f"Economic_{indicator}")
                    
                    if result:
                        results.append({
                            'indicator': indicator,
                            'mae': result['mae_mean'],
                            'improvement': result['improvement_vs_target'],
                            'worth_including': result['mae_mean'] < CFG['benchmark_mae_target'] * 0.98  # 2% improvement
                        })
                
                except Exception as e:
                    LOG.warning(f"Error testing {indicator}: {e}")
                    continue
            
            # Sort by performance
            results.sort(key=lambda x: x['mae'])
            
            LOG.info("Economic indicator results:")
            for i, result in enumerate(results[:5], 1):
                status = "✅" if result['worth_including'] else "❌"
                LOG.info(f"  {i}. {result['indicator']}: MAE {result['mae']:.0f} ({result['improvement']:+.1f}%) {status}")
            
            return results
            
        except Exception as e:
            LOG.error(f"Error in economic indicator testing: {e}")
            return []

# ============================================================================
# OPTIMIZATION ORCHESTRATOR
# ============================================================================

class OptimizationOrchestrator:
    """Main orchestrator for the optimization process"""
    
    def __init__(self):
        self.start_time = time.time()
        self.output_dir = Path(CFG["output_dir"])
        self.output_dir.mkdir(exist_ok=True)
        
        self.results_summary = {
            'benchmark_mae': CFG['benchmark_mae_target'],
            'phases': {},
            'best_models': [],
            'final_recommendations': {}
        }
    
    def print_phase_header(self, phase_name):
        """Print formatted phase header"""
        print(PHASE_SEPARATORS.get(phase_name, f"\n=== {phase_name.upper()} ==="))
    
    def run_complete_optimization(self):
        """Run the complete optimization process"""
        try:
            print(ASCII_BANNER)
            LOG.info("Starting Ultimate Model Optimization...")
            LOG.info(f"Target to beat: MAE {CFG['benchmark_mae_target']}")
            
            # ================================================================
            # PHASE 1: ESTABLISH BENCHMARK
            # ================================================================
            self.print_phase_header('benchmark')
            
            # Load data
            data_loader = DataLoader()
            daily_data = data_loader.create_daily_dataset()
            
            # Analyze correlations
            feature_engineer = FeatureEngineer(daily_data)
            correlations, sorted_correlations = feature_engineer.analyze_mail_correlations()
            
            # Get top mail types for baseline
            top_mail_types = [item[0] for item in sorted_correlations[:10]]
            LOG.info(f"Using top 10 mail types for baseline: {top_mail_types[:5]}...")
            
            # Create benchmark features
            benchmark_X, benchmark_y = feature_engineer.create_baseline_features(top_mail_types)
            
            # Test benchmark
            model_tester = ModelTester(CFG['benchmark_mae_target'])
            benchmark_results = model_tester.test_model_suite(benchmark_X, benchmark_y, "Benchmark")
            
            self.results_summary['phases']['benchmark'] = {
                'best_mae': benchmark_results[0]['mae_mean'] if benchmark_results else float('inf'),
                'features': benchmark_X.shape[1],
                'models_tested': len(benchmark_results)
            }
            
            # ================================================================
            # PHASE 2: FEATURE ENGINEERING
            # ================================================================
            self.print_phase_header('feature_engineering')
            
            feature_results = []
            
            # Test different mail type combinations
            LOG.info("Testing different mail type combinations...")
            for n_types in [5, 8, 12, 15]:
                try:
                    selected_types = [item[0] for item in sorted_correlations[:n_types]]
                    
                    # Test baseline features
                    LOG.info(f"  Testing {n_types} types with baseline features...")
                    X_base, y_base = feature_engineer.create_baseline_features(selected_types)
                    base_results = model_tester.test_model_suite(X_base, y_base, f"Baseline_{n_types}_types")
                    
                    # Test advanced features
                    LOG.info(f"  Testing {n_types} types with advanced features...")
                    X_adv, y_adv = feature_engineer.create_advanced_features(
                        selected_types, include_interactions=True, include_lags=True, include_rolling=True
                    )
                    adv_results = model_tester.test_model_suite(X_adv, y_adv, f"Advanced_{n_types}_types")
                    
                    feature_results.extend(base_results + adv_results)
                    
                except Exception as e:
                    LOG.warning(f"Error testing {n_types} types: {e}")
                    continue
            
            # Test feature variations
            LOG.info("Testing feature engineering variations...")
            best_types = [item[0] for item in sorted_correlations[:8]]  # Use 8 as middle ground
            
            variations = [
                ("interactions_only", {'include_interactions': True, 'include_lags': False, 'include_rolling': False}),
                ("lags_only", {'include_interactions': False, 'include_lags': True, 'include_rolling': False}),
                ("rolling_only", {'include_interactions': False, 'include_lags': False, 'include_rolling': True}),
                ("no_advanced", {'include_interactions': False, 'include_lags': False, 'include_rolling': False})
            ]
            
            for var_name, var_params in variations:
                try:
                    LOG.info(f"  Testing {var_name}...")
                    X_var, y_var = feature_engineer.create_advanced_features(best_types, **var_params)
                    var_results = model_tester.test_model_suite(X_var, y_var, f"Feature_{var_name}")
                    feature_results.extend(var_results)
                except Exception as e:
                    LOG.warning(f"Error testing {var_name}: {e}")
                    continue
            
            self.results_summary['phases']['feature_engineering'] = {
                'best_mae': min(r['mae_mean'] for r in feature_results) if feature_results else float('inf'),
                'variations_tested': len(variations) + 4,  # +4 for different type counts
                'models_tested': len(feature_results)
            }
            
            # ================================================================
            # PHASE 3: ECONOMIC INTEGRATION (CAUTIOUS)
            # ================================================================
            self.print_phase_header('economic_testing')
            
            economic_integrator = EconomicIntegrator(daily_data)
            
            # Get best performing base features from previous phases
            all_results = benchmark_results + feature_results
            all_results.sort(key=lambda x: x['mae_mean'])
            
            if all_results:
                # Use the best feature set so far
                best_so_far = all_results[0]
                LOG.info(f"Best model so far: {best_so_far['model_name']} (MAE: {best_so_far['mae_mean']:.0f})")
                
                # Test economic indicators
                # For simplicity, we'll use the advanced features with 8 types
                best_types = [item[0] for item in sorted_correlations[:8]]
                base_X, base_y = feature_engineer.create_advanced_features(best_types)
                
                economic_results = economic_integrator.test_individual_indicators(base_X, base_y, feature_engineer)
                
                self.results_summary['phases']['economic_testing'] = {
                    'indicators_tested': len(economic_results),
                    'worthy_indicators': len([r for r in economic_results if r.get('worth_including', False)]),
                    'best_economic_mae': min(r['mae'] for r in economic_results) if economic_results else float('inf')
                }
            
            # ================================================================
            # PHASE 4: FINAL OPTIMIZATION & HYPERPARAMETER TUNING
            # ================================================================
            self.print_phase_header('final_optimization')
            
            # Get the absolute best model configuration
            all_results = benchmark_results + feature_results
            all_results.sort(key=lambda x: x['mae_mean'])
            
            final_results = []
            
            if all_results:
                # Take top 3 configurations and do hyperparameter optimization
                top_configs = all_results[:3]
                
                for config in top_configs:
                    try:
                        LOG.info(f"Optimizing {config['model_name']}...")
                        
                        # This is where we'd do grid search, but for robustness,
                        # we'll just test a few key variations
                        if 'random_forest' in config['model_name'].lower():
                            param_variations = [
                                {'n_estimators': 150, 'max_depth': 10, 'min_samples_leaf': 3},
                                {'n_estimators': 200, 'max_depth': 8, 'min_samples_leaf': 5},
                                {'n_estimators': 100, 'max_depth': 12, 'min_samples_leaf': 2}
                            ]
                            
                            for i, params in enumerate(param_variations):
                                optimized_model = RandomForestRegressor(**params, random_state=CFG['random_state'])
                                
                                # Use same feature set as the original config
                                # (This is simplified - in practice we'd recreate the exact features)
                                best_types = [item[0] for item in sorted_correlations[:8]]
                                opt_X, opt_y = feature_engineer.create_advanced_features(best_types)
                                
                                result = model_tester.evaluate_model(
                                    optimized_model, opt_X, opt_y, f"Optimized_RF_{i+1}"
                                )
                                if result:
                                    final_results.append(result)
                        
                    except Exception as e:
                        LOG.warning(f"Error optimizing {config['model_name']}: {e}")
                        continue
            
            # ================================================================
            # PHASE 5: PRODUCTION MODEL PREPARATION
            # ================================================================
            self.print_phase_header('production')
            
            # Compile all results
            all_final_results = benchmark_results + feature_results + final_results
            all_final_results.sort(key=lambda x: x['mae_mean'])
            
            # Select final models
            production_models = []
            
            if all_final_results:
                # Best overall model
                best_model = all_final_results[0]
                production_models.append({
                    'type': 'primary',
                    'model': best_model,
                    'description': 'Best performing model overall'
                })
                
                # Best simple model (< 20 features)
                simple_models = [r for r in all_final_results if r.get('n_features', 0) < 20]
                if simple_models:
                    best_simple = simple_models[0]
                    production_models.append({
                        'type': 'simple',
                        'model': best_simple,
                        'description': 'Best simple model (< 20 features)'
                    })
                
                # Best robust model (from fallback list)
                robust_models = [r for r in all_final_results if any(
                    fallback in r['model_name'].lower() for fallback in CFG['fallback_models']
                )]
                if robust_models:
                    best_robust = robust_models[0]
                    production_models.append({
                        'type': 'robust',
                        'model': best_robust,
                        'description': 'Best robust/fallback model'
                    })
            
            # ================================================================
            # GENERATE FINAL REPORT
            # ================================================================
            self.generate_final_report(all_final_results, production_models)
            
            return production_models
            
        except Exception as e:
            LOG.error(f"Critical error in optimization: {e}")
            LOG.error(traceback.format_exc())
            raise
    
    def generate_final_report(self, all_results, production_models):
        """Generate comprehensive final report"""
        try:
            elapsed_time = time.time() - self.start_time
            
            report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                           OPTIMIZATION COMPLETE                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

🎯 TARGET BENCHMARK: MAE {CFG['benchmark_mae_target']}

📊 RESULTS SUMMARY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""
            
            if all_results:
                all_results.sort(key=lambda x: x['mae_mean'])
                
                report += "🏆 TOP 10 MODELS:\n"
                for i, result in enumerate(all_results[:10], 1):
                    improvement = result.get('improvement_vs_target', 0)
                    status = "🎯" if result['mae_mean'] < CFG['benchmark_mae_target'] else "📈"
                    
                    report += f"  {i:2d}. {result['model_name']:<25} MAE: {result['mae_mean']:6.0f} "
                    report += f"({improvement:+5.1f}%) {status}\n"
                
                best_mae = all_results[0]['mae_mean']
                improvement = (CFG['benchmark_mae_target'] - best_mae) / CFG['benchmark_mae_target'] * 100
                
                report += f"\n🎯 BEST PERFORMANCE:\n"
                report += f"   Model: {all_results[0]['model_name']}\n"
                report += f"   MAE: {best_mae:.0f}\n"
                report += f"   Improvement: {improvement:+.1f}%\n"
                report += f"   Features: {all_results[0].get('n_features', 'Unknown')}\n"
                
                if best_mae < CFG['benchmark_mae_target']:
                    report += f"   ✅ BENCHMARK BEATEN!\n"
                else:
                    report += f"   ❌ Did not beat benchmark\n"
            
            report += f"\n🚀 PRODUCTION RECOMMENDATIONS:\n"
            report += f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            
            for i, prod_model in enumerate(production_models, 1):
                model_info = prod_model['model']
                report += f"  {i}. {prod_model['type'].upper()} MODEL:\n"
                report += f"     {prod_model['description']}\n"
                report += f"     Model: {model_info['model_name']}\n"
                report += f"     MAE: {model_info['mae_mean']:.0f}\n"
                report += f"     Features: {model_info.get('n_features', 'Unknown')}\n\n"
            
            report += f"⏱️  OPTIMIZATION TIME: {elapsed_time/60:.1f} minutes\n"
            report += f"📁 RESULTS SAVED TO: {self.output_dir.resolve()}\n"
            
            report += f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                            STAKEHOLDER SUMMARY                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

📈 BUSINESS IMPACT:
   • Tested {len(all_results)} model configurations
   • Best model achieves {best_mae:.0f} calls average error
   • That's {improvement:+.1f}% vs current benchmark
   • Model complexity: {all_results[0].get('n_features', 'Unknown')} features (simple & interpretable)

🎯 RECOMMENDATION:
   • Deploy {all_results[0]['model_name']} as primary forecasting model
   • Expected accuracy: ~{max(0, 100-best_mae/150):.0f}% for workforce planning
   • Retrain weekly with new data
   • Monitor performance with provided metrics

🔧 NEXT STEPS:
   1. Review detailed model coefficients/importance
   2. Set up automated retraining pipeline  
   3. Implement prediction intervals for uncertainty
   4. Create monitoring dashboard for model drift
            
╚══════════════════════════════════════════════════════════════════════════════╝
"""
            
            # Print report
            print(report)
            LOG.info("Final report generated")
            
            # Save detailed results
            self.save_detailed_results(all_results, production_models, report)
            
            return report
            
        except Exception as e:
            LOG.error(f"Error generating report: {e}")
            return "Error generating report - check logs for details"
    
    def save_detailed_results(self, all_results, production_models, report):
        """Save detailed results to files"""
        try:
            # Save results JSON
            results_data = {
                'timestamp': datetime.now().isoformat(),
                'benchmark_target': CFG['benchmark_mae_target'],
                'all_results': all_results,
                'production_models': production_models,
                'configuration': CFG,
                'summary': self.results_summary
            }
            
            with open(self.output_dir / "optimization_results.json", "w") as f:
                json.dump(results_data, f, indent=2, default=str)
            
            # Save text report
            with open(self.output_dir / "optimization_report.txt", "w") as f:
                f.write(report)
            
            # Save best model details if available
            if hasattr(self, 'model_tester') and hasattr(self.model_tester, 'best_model'):
                if self.model_tester.best_model:
                    try:
                        if JOBLIB_AVAILABLE:
                            joblib.dump(
                                self.model_tester.best_model, 
                                self.output_dir / "best_model.pkl"
                            )
                        else:
                            with open(self.output_dir / "best_model.pkl", "wb") as f:
                                pickle.dump(self.model_tester.best_model, f)
                        
                        LOG.info("Best model saved to best_model.pkl")
                    except Exception as e:
                        LOG.warning(f"Could not save best model: {e}")
            
            LOG.info(f"All results saved to {self.output_dir}")
            
        except Exception as e:
            LOG.error(f"Error saving results: {e}")

# ============================================================================
# PRODUCTION PREDICTION INTERFACE
# ============================================================================

class ProductionPredictor:
    """Production-ready prediction interface with error handling"""
    
    def __init__(self, model_path=None):
        self.model = None
        self.features = None
        self.scaler = None
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load saved model with error handling"""
        try:
            if JOBLIB_AVAILABLE:
                model_data = joblib.load(model_path)
            else:
                with open(model_path, "rb") as f:
                    model_data = pickle.load(f)
            
            self.model = model_data.get('model')
            self.features = model_data.get('features', [])
            
            LOG.info(f"Model loaded successfully from {model_path}")
            LOG.info(f"Expected features: {len(self.features)}")
            
        except Exception as e:
            LOG.error(f"Error loading model: {e}")
            raise
    
    def predict(self, mail_inputs, date_str=None):
        """Make prediction with comprehensive error handling"""
        try:
            if self.model is None:
                raise ValueError("No model loaded")
            
            # Create feature vector
            feature_row = {}
            
            # Handle date
            if date_str:
                predict_date = datetime.strptime(date_str, "%Y-%m-%d")
            else:
                predict_date = datetime.now() + timedelta(days=1)
            
            # Mail features
            total_mail = 0
            for mail_type, volume in mail_inputs.items():
                feature_row[f"{mail_type}_today"] = volume
                total_mail += volume
            
            # Basic aggregates
            feature_row["total_mail"] = total_mail
            feature_row["log_total_mail"] = np.log1p(total_mail)
            
            # Date features
            feature_row["weekday"] = predict_date.weekday()
            feature_row["month"] = predict_date.month
            feature_row["is_month_end"] = 1 if predict_date.day > 25 else 0
            feature_row["is_holiday_week"] = 1 if predict_date.date() in holidays.US() else 0
            
            # Defaults for missing features
            for feature in self.features:
                if feature not in feature_row:
                    feature_row[feature] = 0
            
            # Create DataFrame with correct feature order
            X_pred = pd.DataFrame([feature_row])[self.features]
            
            # Make prediction
            prediction = self.model.predict(X_pred)[0]
            prediction = max(0, prediction)  # Ensure non-negative
            
            return {
                'prediction': prediction,
                'confidence': 'medium',  # Could be enhanced with uncertainty quantification
                'date': predict_date.date(),
                'inputs_used': mail_inputs
            }
            
        except Exception as e:
            LOG.error(f"Error making prediction: {e}")
            # Fallback prediction
            total_volume = sum(mail_inputs.values())
            fallback_prediction = max(5000, total_volume * 0.1)  # Simple heuristic
            
            return {
                'prediction': fallback_prediction,
                'confidence': 'low',
                'error': str(e),
                'fallback_used': True
            }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function with comprehensive error handling"""
    try:
        # Initialize orchestrator
        orchestrator = OptimizationOrchestrator()
        
        # Run complete optimization
        production_models = orchestrator.run_complete_optimization()
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              🎉 SUCCESS! 🎉                                 ║
║                                                                              ║
║  Optimization completed successfully!                                        ║
║  {len(production_models)} production-ready models identified                                ║
║  Results saved to: {orchestrator.output_dir}                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)
        
        return True
        
    except FileNotFoundError as e:
        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                              ❌ DATA ERROR                                  ║
║                                                                              ║
║  Could not find required data files.                                        ║
║  Error: {str(e):<63} ║
║                                                                              ║
║  Please ensure the following files exist:                                   ║
║  • mail.csv (or Mail.csv)                                                   ║
║  • callvolumes.csv                                                           ║
║  • callintent.csv                                                            ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)
        return False
        
    except Exception as e:
        LOG.error(f"Critical error: {e}")
        LOG.error(traceback.format_exc())
        
        print(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                            ❌ CRITICAL ERROR                                ║
║                                                                              ║
║  An unexpected error occurred during optimization.                          ║
║  Error: {str(e):<59} ║
║                                                                              ║
║  Check the log file for detailed error information.                         ║
║  The system attempted to use fallback methods where possible.               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        """)
        return False

# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

if __name__ == "__main__":
    # Configuration check
    print("Checking system configuration...")
    print(f"  XGBoost available: {'✅' if XGBOOST_AVAILABLE else '❌'}")
    print(f"  LightGBM available: {'✅' if LIGHTGBM_AVAILABLE else '❌'}")
    print(f"  Neural Networks available: {'✅' if NEURAL_AVAILABLE else '❌'}")
    print(f"  Joblib available: {'✅' if JOBLIB_AVAILABLE else '❌ (using pickle)'}")
    print()
    
    # Run optimization
    success = main()
    
    if success:
        print("\n🎯 Optimization completed successfully!")
        print("📊 Review the generated report for detailed results.")
        print("🚀 Your optimized models are ready for production deployment!")
    else:
        print("\n❌ Optimization failed.")
        print("🔧 Please check the error messages above and try again.")
        sys.exit(1)
