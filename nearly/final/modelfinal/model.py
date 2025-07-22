#!/usr/bin/env python
"""
OPTIMIZED MAIL-TO-CALLS PREDICTION SYSTEM
=========================================

OPTIMIZATION STRATEGY:
1. Analyze mail types by VOLUME (top 25)
2. Analyze mail types by CORRELATION (top 25) 
3. Find optimal tradeoff between volume and correlation
4. Test different feature combinations systematically
5. Output the highest accuracy model

SYSTEMATIC APPROACH:
- Mail type selection: Volume vs Correlation analysis
- Feature engineering: Comprehensive feature testing
- Model optimization: Find best performing combination
- Production deployment: Bulletproof prediction system
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
from collections import defaultdict
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import joblib

# Statistical Libraries
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# ============================================================================
# OPTIMIZATION CONFIGURATION
# ============================================================================

CONFIG = {
    # Data files
    "call_intent_file": "callintent.csv",
    "mail_file": "mail.csv",
    
    # Mail type analysis
    "top_n_by_volume": 25,
    "top_n_by_correlation": 25,
    "min_correlation_threshold": 0.01,
    "correlation_lags_to_test": [0, 1, 2, 3, 4, 5],
    
    # Feature engineering options
    "feature_combinations": {
        "minimal": ["basic_lags", "temporal_basic"],
        "standard": ["basic_lags", "rolling", "temporal_basic", "temporal_advanced"],
        "comprehensive": ["basic_lags", "rolling", "weighted", "temporal_basic", "temporal_advanced", "interaction"],
        "advanced": ["basic_lags", "rolling", "weighted", "temporal_basic", "temporal_advanced", "interaction", "trend"]
    },
    
    # Lag configurations
    "lag_options": {
        "basic_lags": [1, 2, 3],
        "extended_lags": [1, 2, 3, 4, 5],
        "focused_lags": [1, 2],
        "comprehensive_lags": [1, 2, 3, 4, 5, 6, 7]
    },
    
    # Model options
    "models_to_test": {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=10.0, random_state=42),
        "lasso": Lasso(alpha=1.0, random_state=42, max_iter=2000),
        "elastic": ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42, max_iter=2000),
        "random_forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
        "gradient_boost": GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=42)
    },
    
    # Optimization settings
    "cv_folds": 5,
    "test_split": 0.2,
    "optimization_metric": "cv_r2",  # or "cv_mae"
    "min_samples_required": 30,
    
    # Output
    "output_dir": "optimized_mail_calls_system",
    "save_analysis": True,
    "save_best_model": True,
    "random_state": 42
}

# ============================================================================
# SAFE LOGGING AND UTILITIES
# ============================================================================

def setup_logging():
    """Setup safe logging"""
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "optimization.log", mode='w', encoding='utf-8')
        ]
    )
    return logging.getLogger(__name__)

LOG = setup_logging()

def safe_print(message: str):
    """Print safely without encoding issues"""
    try:
        clean_msg = str(message).encode('ascii', 'ignore').decode('ascii')
        print(clean_msg)
    except:
        print(str(message))

# ============================================================================
# DATA LOADER AND ANALYZER
# ============================================================================

class OptimizationDataLoader:
    """Load data and perform comprehensive mail type analysis"""
    
    def __init__(self):
        self.call_data = None
        self.mail_data = None
        self.intent_data = None
        self.mail_analysis = {}
        self.best_mail_types = []
        
    def load_data_robust(self, filename: str) -> pd.DataFrame:
        """Robust data loading"""
        LOG.info(f"Loading {filename}...")
        
        paths = [filename, f"data/{filename}", f"data\\{filename}"]
        strategies = [
            {'encoding': 'utf-8', 'sep': ','},
            {'encoding': 'utf-8', 'sep': ';'},
            {'encoding': 'latin1', 'sep': ','}
        ]
        
        for path in paths:
            if not Path(path).exists():
                continue
            for strategy in strategies:
                try:
                    df = pd.read_csv(path, low_memory=False, **strategy)
                    if len(df) > 0:
                        LOG.info(f"Loaded {path}: {len(df)} rows, {df.shape[1]} columns")
                        return df
                except:
                    continue
        
        raise FileNotFoundError(f"Could not load {filename}")
    
    def detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """Smart column detection"""
        columns = {}
        df_cols = [str(col).lower() for col in df.columns]
        
        # Date column
        for i, col in enumerate(df_cols):
            if any(kw in col for kw in ['date', 'time', 'start']):
                try:
                    sample = df.iloc[:100, i].dropna()
                    parsed = pd.to_datetime(sample, errors='coerce')
                    if parsed.notna().sum() / len(sample) > 0.7:
                        columns['date'] = df.columns[i]
                        break
                except:
                    continue
        
        # Volume column
        for i, col in enumerate(df_cols):
            if any(kw in col for kw in ['volume', 'count', 'amount']):
                if df.iloc[:, i].dtype in ['int64', 'float64']:
                    columns['volume'] = df.columns[i]
                    break
        
        # Type column
        for i, col in enumerate(df_cols):
            if any(kw in col for kw in ['type', 'category']):
                unique_count = df.iloc[:, i].nunique()
                if 2 <= unique_count <= 500:
                    columns['type'] = df.columns[i]
                    break
        
        # Intent column
        for i, col in enumerate(df_cols):
            if 'intent' in col:
                columns['intent'] = df.columns[i]
                break
        
        return columns
    
    def load_call_data(self) -> Tuple[pd.Series, pd.DataFrame]:
        """Load and process call data"""
        LOG.info("Loading call intent data...")
        
        df = self.load_data_robust(CONFIG["call_intent_file"])
        columns = self.detect_columns(df)
        
        # Process dates
        date_col = columns['date']
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df[df[date_col].dt.year >= 2025]
        
        LOG.info(f"Found {len(df)} call records from 2025+")
        
        # Daily call volumes
        df['call_date'] = df[date_col].dt.date
        daily_calls = df.groupby('call_date').size()
        daily_calls.index = pd.to_datetime(daily_calls.index)
        daily_calls = daily_calls.sort_index()
        
        # Intent data
        daily_intents = None
        if 'intent' in columns:
            intent_col = columns['intent']
            df[intent_col] = df[intent_col].fillna('Unknown').astype(str)
            
            # Keep common intents
            intent_counts = df[intent_col].value_counts()
            common_intents = intent_counts[intent_counts >= 10].index
            df_filtered = df[df[intent_col].isin(common_intents)]
            
            if len(df_filtered) > 0:
                intent_pivot = df_filtered.groupby(['call_date', intent_col]).size().unstack(fill_value=0)
                intent_pivot.index = pd.to_datetime(intent_pivot.index)
                daily_intents = intent_pivot.div(intent_pivot.sum(axis=1), axis=0).fillna(0)
                LOG.info(f"Created intent data for {len(daily_intents.columns)} intents")
        
        self.call_data = daily_calls
        self.intent_data = daily_intents
        return daily_calls, daily_intents
    
    def load_mail_data(self) -> pd.DataFrame:
        """Load and process mail data"""
        LOG.info("Loading mail data...")
        
        df = self.load_data_robust(CONFIG["mail_file"])
        columns = self.detect_columns(df)
        
        if 'date' not in columns or 'volume' not in columns:
            raise ValueError("Required mail columns not found")
        
        date_col = columns['date']
        volume_col = columns['volume']
        type_col = columns.get('type')
        
        # Process data
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        df = df[df[date_col].dt.year >= 2025]
        
        df[volume_col] = pd.to_numeric(df[volume_col], errors='coerce')
        df = df.dropna(subset=[volume_col])
        df = df[df[volume_col] >= 0]
        
        LOG.info(f"Processed {len(df)} mail records from 2025+")
        
        # Create daily mail data
        df['mail_date'] = df[date_col].dt.date
        
        if type_col:
            df[type_col] = df[type_col].astype(str)
            mail_daily = df.groupby(['mail_date', type_col])[volume_col].sum().unstack(fill_value=0)
        else:
            mail_daily = df.groupby('mail_date')[volume_col].sum().to_frame('total_mail')
        
        mail_daily.index = pd.to_datetime(mail_daily.index)
        mail_daily = mail_daily.sort_index()
        
        LOG.info(f"Created daily mail data: {len(mail_daily)} days, {len(mail_daily.columns)} types")
        
        self.mail_data = mail_daily
        return mail_daily
    
    def analyze_mail_types_comprehensive(self) -> Dict:
        """Comprehensive analysis of mail types by volume and correlation"""
        
        LOG.info("Starting comprehensive mail type analysis...")
        
        if self.call_data is None or self.mail_data is None:
            raise ValueError("Call and mail data must be loaded first")
        
        # Find overlapping dates
        common_dates = self.call_data.index.intersection(self.mail_data.index)
        if len(common_dates) < 20:
            raise ValueError(f"Insufficient overlapping data: {len(common_dates)} days")
        
        aligned_calls = self.call_data.loc[common_dates]
        aligned_mail = self.mail_data.loc[common_dates]
        
        LOG.info(f"Analyzing {len(common_dates)} overlapping days with {len(aligned_mail.columns)} mail types")
        
        analysis = {}
        
        # 1. VOLUME ANALYSIS
        LOG.info("Analyzing mail types by volume...")
        volume_analysis = {}
        
        for mail_type in aligned_mail.columns:
            total_volume = aligned_mail[mail_type].sum()
            avg_daily_volume = aligned_mail[mail_type].mean()
            days_active = (aligned_mail[mail_type] > 0).sum()
            volume_std = aligned_mail[mail_type].std()
            
            volume_analysis[mail_type] = {
                'total_volume': total_volume,
                'avg_daily_volume': avg_daily_volume,
                'days_active': days_active,
                'activity_ratio': days_active / len(common_dates),
                'volume_std': volume_std,
                'volume_cv': volume_std / (avg_daily_volume + 1e-10),  # Coefficient of variation
                'volume_score': total_volume * (days_active / len(common_dates))  # Volume weighted by consistency
            }
        
        # Top mail types by volume
        volume_rankings = sorted(volume_analysis.items(), key=lambda x: x[1]['volume_score'], reverse=True)
        top_by_volume = [item[0] for item in volume_rankings[:CONFIG["top_n_by_volume"]]]
        
        analysis['volume_analysis'] = volume_analysis
        analysis['top_by_volume'] = top_by_volume
        
        LOG.info(f"Top 5 by volume: {top_by_volume[:5]}")
        
        # 2. CORRELATION ANALYSIS
        LOG.info("Analyzing mail types by correlation...")
        correlation_analysis = {}
        
        for mail_type in aligned_mail.columns:
            mail_series = aligned_mail[mail_type]
            
            # Test different lags
            best_correlation = 0
            best_lag = 0
            lag_correlations = {}
            
            for lag in CONFIG["correlation_lags_to_test"]:
                try:
                    if lag == 0:
                        corr, p_value = pearsonr(mail_series, aligned_calls)
                    else:
                        # Lag the calls (positive lag means mail leads calls)
                        lagged_calls = aligned_calls.shift(-lag).dropna()
                        if len(lagged_calls) > 10:
                            mail_subset = mail_series.loc[lagged_calls.index]
                            corr, p_value = pearsonr(mail_subset, lagged_calls)
                        else:
                            corr, p_value = 0, 1
                    
                    if not np.isnan(corr):
                        lag_correlations[lag] = {'correlation': corr, 'p_value': p_value}
                        
                        if abs(corr) > abs(best_correlation):
                            best_correlation = corr
                            best_lag = lag
                            
                except Exception as e:
                    lag_correlations[lag] = {'correlation': 0, 'p_value': 1}
            
            correlation_analysis[mail_type] = {
                'best_correlation': best_correlation,
                'best_lag': best_lag,
                'lag_correlations': lag_correlations,
                'abs_correlation': abs(best_correlation),
                'correlation_strength': 'Strong' if abs(best_correlation) > 0.3 else 
                                      'Moderate' if abs(best_correlation) > 0.1 else 'Weak'
            }
        
        # Top mail types by correlation
        correlation_rankings = sorted(
            correlation_analysis.items(), 
            key=lambda x: x[1]['abs_correlation'], 
            reverse=True
        )
        
        # Filter by minimum threshold
        significant_correlations = [
            (mail_type, data) for mail_type, data in correlation_rankings 
            if data['abs_correlation'] >= CONFIG["min_correlation_threshold"]
        ]
        
        top_by_correlation = [item[0] for item in significant_correlations[:CONFIG["top_n_by_correlation"]]]
        
        analysis['correlation_analysis'] = correlation_analysis
        analysis['top_by_correlation'] = top_by_correlation
        
        LOG.info(f"Top 5 by correlation: {top_by_correlation[:5]}")
        LOG.info(f"Best correlations: {[(t, round(correlation_analysis[t]['best_correlation'], 3)) for t in top_by_correlation[:5]]}")
        
        # 3. COMBINED ANALYSIS - FIND OPTIMAL TRADEOFF
        LOG.info("Finding optimal tradeoff between volume and correlation...")
        
        combined_analysis = {}
        
        for mail_type in aligned_mail.columns:
            volume_data = volume_analysis[mail_type]
            corr_data = correlation_analysis[mail_type]
            
            # Normalize scores to 0-1 range
            volume_rank = next(i for i, (name, _) in enumerate(volume_rankings) if name == mail_type) + 1
            corr_rank = next(i for i, (name, _) in enumerate(correlation_rankings) if name == mail_type) + 1
            
            volume_score_norm = 1 - (volume_rank - 1) / len(volume_rankings)
            correlation_score_norm = 1 - (corr_rank - 1) / len(correlation_rankings)
            
            # Test different weighting schemes
            weighting_schemes = {
                'volume_heavy': 0.8 * volume_score_norm + 0.2 * correlation_score_norm,
                'balanced': 0.5 * volume_score_norm + 0.5 * correlation_score_norm,
                'correlation_heavy': 0.2 * volume_score_norm + 0.8 * correlation_score_norm,
                'correlation_only': correlation_score_norm,
                'volume_only': volume_score_norm
            }
            
            combined_analysis[mail_type] = {
                'volume_rank': volume_rank,
                'correlation_rank': corr_rank,
                'volume_score_norm': volume_score_norm,
                'correlation_score_norm': correlation_score_norm,
                'weighting_schemes': weighting_schemes,
                'volume_raw': volume_data['total_volume'],
                'correlation_raw': corr_data['best_correlation'],
                'best_lag': corr_data['best_lag']
            }
        
        analysis['combined_analysis'] = combined_analysis
        
        # Create different mail type selections for testing
        mail_type_selections = {}
        
        for scheme_name, _ in weighting_schemes.items():
            ranked_by_scheme = sorted(
                combined_analysis.items(),
                key=lambda x: x[1]['weighting_schemes'][scheme_name],
                reverse=True
            )
            mail_type_selections[scheme_name] = [item[0] for item in ranked_by_scheme[:15]]  # Top 15
        
        analysis['mail_type_selections'] = mail_type_selections
        
        # Log the different selections
        for scheme, types in mail_type_selections.items():
            LOG.info(f"{scheme} top 5: {types[:5]}")
        
        self.mail_analysis = analysis
        return analysis
    
    def get_aligned_data(self) -> Dict:
        """Get aligned call and mail data"""
        common_dates = self.call_data.index.intersection(self.mail_data.index)
        
        # Filter to business days
        business_dates = [d for d in common_dates if d.weekday() < 5]
        
        return {
            'calls': self.call_data.loc[business_dates],
            'mail': self.mail_data.loc[business_dates],
            'intents': self.intent_data.loc[business_dates] if self.intent_data is not None else None,
            'dates': business_dates
        }

# ============================================================================
# FEATURE ENGINEERING ENGINE
# ============================================================================

class AdvancedFeatureEngine:
    """Advanced feature engineering with systematic testing"""
    
    def __init__(self):
        self.feature_sets = {}
        self.feature_importance = {}
        
    def create_mail_features(self, mail_data: pd.DataFrame, mail_types: List[str], 
                           feature_config: List[str], lag_config: List[int]) -> pd.DataFrame:
        """Create mail-based features"""
        
        features = pd.DataFrame(index=mail_data.index)
        
        # Select mail types
        available_types = [t for t in mail_types if t in mail_data.columns]
        
        for mail_type in available_types:
            clean_name = str(mail_type).replace(' ', '').replace('-', '').replace('_', '')[:10]
            mail_series = mail_data[mail_type]
            
            # Basic lag features
            if "basic_lags" in feature_config:
                for lag in lag_config:
                    if lag == 0:
                        features[f"{clean_name}_today"] = mail_series
                    else:
                        features[f"{clean_name}_lag{lag}"] = mail_series.shift(lag)
            
            # Rolling features
            if "rolling" in feature_config:
                for window in [3, 7]:
                    if window <= len(mail_data):
                        features[f"{clean_name}_avg{window}"] = mail_series.rolling(window, min_periods=1).mean()
                        features[f"{clean_name}_sum{window}"] = mail_series.rolling(window, min_periods=1).sum()
                        if window >= 3:
                            features[f"{clean_name}_std{window}"] = mail_series.rolling(window, min_periods=1).std()
            
            # Weighted features
            if "weighted" in feature_config:
                weights = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1, 5: 0.05}
                weighted_sum = pd.Series(0.0, index=mail_series.index)
                
                for lag in lag_config:
                    weight = weights.get(lag, 0.05)
                    if lag == 0:
                        weighted_sum += mail_series * weight
                    else:
                        weighted_sum += mail_series.shift(lag).fillna(0) * weight
                
                features[f"{clean_name}_weighted"] = weighted_sum
            
            # Trend features
            if "trend" in feature_config and len(mail_data) >= 7:
                trend = mail_series.rolling(7, min_periods=3).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 2 else 0
                )
                features[f"{clean_name}_trend"] = trend
        
        # Aggregate features
        total_mail = mail_data[available_types].sum(axis=1)
        
        for lag in lag_config:
            if lag == 0:
                features['total_mail_today'] = total_mail
            else:
                features[f'total_mail_lag{lag}'] = total_mail.shift(lag)
        
        if "rolling" in feature_config:
            features['total_mail_avg7'] = total_mail.rolling(7, min_periods=1).mean()
            features['total_mail_std7'] = total_mail.rolling(7, min_periods=1).std()
        
        # Interaction features
        if "interaction" in feature_config and len(available_types) >= 2:
            # Top 2 mail types interaction
            top_types = available_types[:2]
            if len(top_types) == 2:
                type1, type2 = top_types
                features['interaction_top2'] = mail_data[type1] * mail_data[type2]
                features['ratio_top2'] = (mail_data[type1] + 1) / (mail_data[type2] + 1)
        
        # Fill NaN values
        features = features.fillna(0)
        
        return features
    
    def create_temporal_features(self, dates: pd.DatetimeIndex, feature_config: List[str]) -> pd.DataFrame:
        """Create temporal features"""
        
        features = pd.DataFrame(index=dates)
        
        # Basic temporal
        if "temporal_basic" in feature_config:
            features['weekday'] = dates.weekday
            features['month'] = dates.month
            features['quarter'] = dates.quarter
            features['day_of_month'] = dates.day
            features['is_weekend'] = (dates.weekday >= 5).astype(int)
        
        # Advanced temporal
        if "temporal_advanced" in feature_config:
            # Business calendar
            features['is_month_start'] = (dates.day <= 5).astype(int)
            features['is_month_end'] = (dates.day >= 25).astype(int)
            features['is_quarter_end'] = dates.to_series().apply(
                lambda x: 1 if x.month in [3, 6, 9, 12] and x.day >= 25 else 0
            ).values
            
            # Cyclical encoding
            features['weekday_sin'] = np.sin(2 * np.pi * features['weekday'] / 7)
            features['weekday_cos'] = np.cos(2 * np.pi * features['weekday'] / 7)
            features['month_sin'] = np.sin(2 * np.pi * features['month'] / 12)
            features['month_cos'] = np.cos(2 * np.pi * features['month'] / 12)
            
            # Week of year
            features['week_of_year'] = dates.week
            features['week_sin'] = np.sin(2 * np.pi * features['week_of_year'] / 52)
            features['week_cos'] = np.cos(2 * np.pi * features['week_of_year'] / 52)
        
        return features
    
    def create_call_history_features(self, call_data: pd.Series) -> pd.DataFrame:
        """Create call history features"""
        
        features = pd.DataFrame(index=call_data.index)
        
        # Basic lags
        for lag in [1, 2, 3, 7]:
            features[f'calls_lag{lag}'] = call_data.shift(lag)
        
        # Rolling statistics
        for window in [3, 7, 14, 30]:
            if len(call_data) >= window:
                features[f'calls_mean{window}'] = call_data.rolling(window, min_periods=1).mean()
                features[f'calls_std{window}'] = call_data.rolling(window, min_periods=1).std()
                features[f'calls_max{window}'] = call_data.rolling(window, min_periods=1).max()
                features[f'calls_min{window}'] = call_data.rolling(window, min_periods=1).min()
        
        # Trend and momentum
        if len(call_data) >= 7:
            features['calls_trend7'] = call_data.rolling(7, min_periods=3).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 2 else 0
            )
            
            # Momentum (rate of change)
            features['calls_momentum3'] = call_data.pct_change(3)
            features['calls_momentum7'] = call_data.pct_change(7)
        
        # Fill NaN with reasonable defaults
        features = features.fillna(method='ffill').fillna(call_data.mean())
        
        return features
    
    def create_feature_set(self, aligned_data: Dict, mail_types: List[str], 
                          feature_config: List[str], lag_config: List[int]) -> Tuple[pd.DataFrame, pd.Series]:
        """Create complete feature set"""
        
        calls = aligned_data['calls']
        mail = aligned_data['mail']
        
        # Target: next day calls
        y = calls.shift(-1).dropna()
        target_dates = y.index
        
        feature_components = []
        
        # 1. Mail features
        if mail is not None and len(mail_types) > 0:
            mail_features = self.create_mail_features(mail, mail_types, feature_config, lag_config)
            mail_features = mail_features.reindex(target_dates, fill_value=0)
            feature_components.append(mail_features)
        
        # 2. Temporal features
        temporal_features = self.create_temporal_features(target_dates, feature_config)
        feature_components.append(temporal_features)
        
        # 3. Call history features
        call_features = self.create_call_history_features(calls)
        call_features = call_features.reindex(target_dates, fill_value=0)
        feature_components.append(call_features)
        
        # Combine all features
        X = pd.concat(feature_components, axis=1)
        X = X.fillna(0)
        
        return X, y

# ============================================================================
# SYSTEMATIC MODEL OPTIMIZER
# ============================================================================

class SystematicModelOptimizer:
    """Systematic optimization engine"""
    
    def __init__(self, data_loader: OptimizationDataLoader):
        self.data_loader = data_loader
        self.feature_engine = AdvancedFeatureEngine()
        self.optimization_results = {}
        self.best_configuration = None
        self.best_model = None
        self.best_score = -float('inf')
        
    def evaluate_model_configuration(self, model, X: pd.DataFrame, y: pd.Series, config_name: str) -> Dict:
        """Evaluate a single model configuration"""
        
        if len(X) < CONFIG["min_samples_required"]:
            return {"error": "insufficient_samples", "samples": len(X)}
        
        try:
            results = {"config_name": config_name, "features": len(X.columns), "samples": len(X)}
            
            # Cross-validation
            n_splits = min(CONFIG["cv_folds"], len(X) // 10, 5)
            tscv = TimeSeriesSplit(n_splits=max(2, n_splits))
            
            cv_results = cross_validate(
                model, X, y, cv=tscv,
                scoring=['neg_mean_absolute_error', 'r2'],
                return_train_score=False
            )
            
            results['cv_mae'] = -cv_results['test_neg_mean_absolute_error'].mean()
            results['cv_mae_std'] = cv_results['test_neg_mean_absolute_error'].std()
            results['cv_r2'] = cv_results['test_r2'].mean()
            results['cv_r2_std'] = cv_results['test_r2'].std()
            
            # Holdout test
            split_idx = int(len(X) * (1 - CONFIG["test_split"]))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            if len(X_test) >= 3:  # Minimum test set
                model.fit(X_train, y_train)
                test_pred = model.predict(X_test)
                
                results['test_mae'] = mean_absolute_error(y_test, test_pred)
                results['test_r2'] = r2_score(y_test, test_pred)
                results['test_mape'] = np.mean(np.abs((y_test - test_pred) / (y_test + 1e-10))) * 100
                
                # Final model with all data
                model.fit(X, y)
                results['model'] = model
                
                # Feature importance if available
                if hasattr(model, 'feature_importances_'):
                    importance = dict(zip(X.columns, model.feature_importances_))
                    results['feature_importance'] = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
                elif hasattr(model, 'coef_'):
                    importance = dict(zip(X.columns, np.abs(model.coef_)))
                    results['feature_importance'] = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
            else:
                results['test_mae'] = float('inf')
                results['test_r2'] = -float('inf')
            
            return results
            
        except Exception as e:
            return {"error": str(e), "config_name": config_name}
    
    def run_systematic_optimization(self) -> Dict:
        """Run systematic optimization across all combinations"""
        
        LOG.info("=== STARTING SYSTEMATIC OPTIMIZATION ===")
        
        # Get aligned data
        aligned_data = self.data_loader.get_aligned_data()
        mail_analysis = self.data_loader.mail_analysis
        
        LOG.info(f"Optimization dataset: {len(aligned_data['calls'])} samples")
        
        # Test configurations
        configurations_to_test = []
        
        # 1. Mail type selection strategies
        mail_type_strategies = mail_analysis['mail_type_selections']
        
        # 2. Feature combinations
        feature_combinations = CONFIG["feature_combinations"]
        
        # 3. Lag configurations
        lag_configurations = CONFIG["lag_options"]
        
        # 4. Models
        models = CONFIG["models_to_test"]
        
        # Create all combinations (but limit to reasonable number)
        test_combinations = []
        
        for mail_strategy, mail_types in mail_type_strategies.items():
            for feature_name, feature_config in feature_combinations.items():
                for lag_name, lag_config in lag_configurations.items():
                    for model_name, model in models.items():
                        
                        config = {
                            'mail_strategy': mail_strategy,
                            'mail_types': mail_types[:10],  # Limit to top 10
                            'feature_config': feature_config,
                            'lag_config': lag_config,
                            'model_name': model_name,
                            'model': model,
                            'config_id': f"{mail_strategy}_{feature_name}_{lag_name}_{model_name}"
                        }
                        test_combinations.append(config)
        
        LOG.info(f"Testing {len(test_combinations)} configurations...")
        
        # Limit combinations to avoid excessive computation
        if len(test_combinations) > 200:
            LOG.info("Limiting to 200 configurations for efficiency")
            
            # Prioritize certain combinations
            priority_configs = []
            other_configs = []
            
            for config in test_combinations:
                if (config['mail_strategy'] in ['balanced', 'correlation_heavy'] and 
                    config['feature_config'] in ['standard', 'comprehensive'] and
                    config['model_name'] in ['ridge', 'random_forest']):
                    priority_configs.append(config)
                else:
                    other_configs.append(config)
            
            # Take top priority + random sample of others
            np.random.shuffle(other_configs)
            test_combinations = priority_configs + other_configs[:200-len(priority_configs)]
        
        # Run optimization
        all_results = {}
        progress_counter = 0
        
        for config in test_combinations:
            progress_counter += 1
            if progress_counter % 20 == 0:
                LOG.info(f"Progress: {progress_counter}/{len(test_combinations)} configurations tested")
            
            try:
                # Create feature set
                X, y = self.feature_engine.create_feature_set(
                    aligned_data,
                    config['mail_types'],
                    config['feature_config'],
                    config['lag_config']
                )
                
                # Evaluate model
                results = self.evaluate_model_configuration(
                    config['model'],
                    X, y,
                    config['config_id']
                )
                
                # Store configuration details
                results['configuration'] = {
                    'mail_strategy': config['mail_strategy'],
                    'mail_types': config['mail_types'],
                    'feature_config': config['feature_config'],
                    'lag_config': config['lag_config'],
                    'model_name': config['model_name']
                }
                
                all_results[config['config_id']] = results
                
                # Track best configuration
                if "error" not in results:
                    optimization_score = results.get(CONFIG["optimization_metric"], -float('inf'))
                    
                    if optimization_score > self.best_score:
                        self.best_score = optimization_score
                        self.best_configuration = config
                        self.best_model = results.get('model')
                        
                        LOG.info(f"New best configuration: {config['config_id']}")
                        LOG.info(f"  Score ({CONFIG['optimization_metric']}): {optimization_score:.3f}")
                        LOG.info(f"  Mail strategy: {config['mail_strategy']}")
                        LOG.info(f"  Features: {results['features']}")
                        LOG.info(f"  Mail types: {config['mail_types'][:3]}...")
                
            except Exception as e:
                LOG.error(f"Configuration {config['config_id']} failed: {str(e)}")
                all_results[config['config_id']] = {"error": str(e), "configuration": config}
        
        self.optimization_results = all_results
        
        # Log final results
        if self.best_configuration:
            LOG.info(f"\n=== OPTIMIZATION COMPLETE ===")
            LOG.info(f"Best configuration: {self.best_configuration['config_id']}")
            LOG.info(f"Best score: {self.best_score:.3f}")
            LOG.info(f"Total configurations tested: {len(test_combinations)}")
            LOG.info(f"Successful configurations: {len([r for r in all_results.values() if 'error' not in r])}")
        else:
            LOG.warning("No successful configurations found!")
        
        return all_results
    
    def get_top_configurations(self, top_n: int = 10) -> List[Dict]:
        """Get top N configurations"""
        
        successful_configs = []
        
        for config_id, results in self.optimization_results.items():
            if "error" not in results and CONFIG["optimization_metric"] in results:
                score = results[CONFIG["optimization_metric"]]
                successful_configs.append({
                    'config_id': config_id,
                    'score': score,
                    'results': results
                })
        
        # Sort by score
        successful_configs.sort(key=lambda x: x['score'], reverse=True)
        
        return successful_configs[:top_n]
    
    def analyze_optimization_insights(self) -> Dict:
        """Analyze insights from optimization results"""
        
        insights = {}
        
        # Get successful results
        successful = [r for r in self.optimization_results.values() if "error" not in r]
        
        if len(successful) == 0:
            return {"error": "No successful configurations to analyze"}
        
        # Mail strategy effectiveness
        strategy_scores = defaultdict(list)
        for result in successful:
            if 'configuration' in result:
                strategy = result['configuration']['mail_strategy']
                score = result.get(CONFIG["optimization_metric"], 0)
                strategy_scores[strategy].append(score)
        
        strategy_analysis = {}
        for strategy, scores in strategy_scores.items():
            strategy_analysis[strategy] = {
                'avg_score': np.mean(scores),
                'best_score': max(scores),
                'count': len(scores)
            }
        
        insights['mail_strategy_analysis'] = strategy_analysis
        
        # Model effectiveness
        model_scores = defaultdict(list)
        for result in successful:
            if 'configuration' in result:
                model = result['configuration']['model_name']
                score = result.get(CONFIG["optimization_metric"], 0)
                model_scores[model].append(score)
        
        model_analysis = {}
        for model, scores in model_scores.items():
            model_analysis[model] = {
                'avg_score': np.mean(scores),
                'best_score': max(scores),
                'count': len(scores)
            }
        
        insights['model_analysis'] = model_analysis
        
        # Feature combination effectiveness
        feature_scores = defaultdict(list)
        for result in successful:
            if 'configuration' in result:
                features = str(result['configuration']['feature_config'])
                score = result.get(CONFIG["optimization_metric"], 0)
                feature_scores[features].append(score)
        
        feature_analysis = {}
        for features, scores in feature_scores.items():
            feature_analysis[features] = {
                'avg_score': np.mean(scores),
                'best_score': max(scores),
                'count': len(scores)
            }
        
        insights['feature_analysis'] = feature_analysis
        
        return insights

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class OptimizationOrchestrator:
    """Main orchestrator for the optimization process"""
    
    def __init__(self):
        self.start_time = time.time()
        self.output_dir = Path(CONFIG["output_dir"])
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        for subdir in ["analysis", "models", "results"]:
            (self.output_dir / subdir).mkdir(exist_ok=True)
    
    def run_complete_optimization(self) -> Dict:
        """Run the complete optimization process"""
        
        LOG.info("="*80)
        LOG.info("OPTIMIZED MAIL-TO-CALLS PREDICTION SYSTEM")
        LOG.info("="*80)
        LOG.info("STRATEGY: Systematic optimization of:")
        LOG.info("  1. Mail types (Volume vs Correlation tradeoff)")
        LOG.info("  2. Feature engineering combinations")
        LOG.info("  3. Model selection and tuning")
        LOG.info("  4. Find highest accuracy configuration")
        LOG.info("="*80)
        
        try:
            # Phase 1: Load and Analyze Data
            LOG.info("\nPHASE 1: DATA LOADING AND MAIL TYPE ANALYSIS")
            data_loader = OptimizationDataLoader()
            
            # Load data
            data_loader.load_call_data()
            data_loader.load_mail_data()
            
            # Comprehensive mail type analysis
            mail_analysis = data_loader.analyze_mail_types_comprehensive()
            
            # Phase 2: Systematic Optimization
            LOG.info("\nPHASE 2: SYSTEMATIC MODEL OPTIMIZATION")
            optimizer = SystematicModelOptimizer(data_loader)
            optimization_results = optimizer.run_systematic_optimization()
            
            # Phase 3: Analysis and Insights
            LOG.info("\nPHASE 3: ANALYZING OPTIMIZATION RESULTS")
            top_configs = optimizer.get_top_configurations(10)
            insights = optimizer.analyze_optimization_insights()
            
            # Phase 4: Save Results
            LOG.info("\nPHASE 4: SAVING OPTIMIZATION RESULTS")
            self.save_optimization_results(data_loader, optimizer, top_configs, insights)
            
            # Phase 5: Generate Report
            LOG.info("\nPHASE 5: GENERATING OPTIMIZATION REPORT")
            report_path = self.generate_optimization_report(data_loader, optimizer, top_configs, insights)
            
            execution_time = (time.time() - self.start_time) / 60
            
            LOG.info(f"\n{'='*80}")
            LOG.info("OPTIMIZATION COMPLETE!")
            LOG.info(f"Execution time: {execution_time:.1f} minutes")
            if optimizer.best_configuration:
                LOG.info(f"Best configuration: {optimizer.best_configuration['config_id']}")
                LOG.info(f"Best {CONFIG['optimization_metric']}: {optimizer.best_score:.3f}")
            LOG.info(f"Results saved to: {self.output_dir}")
            LOG.info(f"{'='*80}")
            
            return {
                'success': True,
                'execution_time_minutes': execution_time,
                'best_score': optimizer.best_score,
                'best_configuration': optimizer.best_configuration,
                'best_model': optimizer.best_model,
                'top_configurations': top_configs,
                'optimization_insights': insights,
                'mail_analysis': mail_analysis,
                'output_directory': str(self.output_dir),
                'report_path': report_path
            }
            
        except Exception as e:
            LOG.error(f"OPTIMIZATION FAILED: {str(e)}")
            LOG.error(traceback.format_exc())
            
            return {
                'success': False,
                'error': str(e),
                'execution_time_minutes': (time.time() - self.start_time) / 60
            }
    
    def save_optimization_results(self, data_loader, optimizer, top_configs, insights):
        """Save all optimization results"""
        
        try:
            # Save mail analysis
            with open(self.output_dir / "analysis" / "mail_type_analysis.json", 'w') as f:
                json.dump(data_loader.mail_analysis, f, indent=2, default=str)
            
            # Save optimization results (simplified for JSON)
            simplified_results = {}
            for config_id, results in optimizer.optimization_results.items():
                simplified_results[config_id] = {}
                for key, value in results.items():
                    if key == 'model':
                        simplified_results[config_id][key] = str(type(value).__name__)
                    elif key == 'feature_importance' and isinstance(value, list):
                        simplified_results[config_id][key] = value[:10]  # Top 10
                    else:
                        simplified_results[config_id][key] = value
            
            with open(self.output_dir / "results" / "optimization_results.json", 'w') as f:
                json.dump(simplified_results, f, indent=2, default=str)
            
            # Save top configurations
            with open(self.output_dir / "results" / "top_configurations.json", 'w') as f:
                json.dump(top_configs, f, indent=2, default=str)
            
            # Save insights
            with open(self.output_dir / "results" / "optimization_insights.json", 'w') as f:
                json.dump(insights, f, indent=2, default=str)
            
            # Save best model if available
            if optimizer.best_model and CONFIG["save_best_model"]:
                joblib.dump(optimizer.best_model, self.output_dir / "models" / "best_optimized_model.pkl")
                
                # Save best configuration details
                best_config_details = {
                    'configuration': optimizer.best_configuration,
                    'performance': {
                        'best_score': optimizer.best_score,
                        'optimization_metric': CONFIG['optimization_metric']
                    }
                }
                
                with open(self.output_dir / "models" / "best_model_config.json", 'w') as f:
                    json.dump(best_config_details, f, indent=2, default=str)
            
            LOG.info("All optimization results saved successfully")
            
        except Exception as e:
            LOG.error(f"Failed to save optimization results: {str(e)}")
    
    def generate_optimization_report(self, data_loader, optimizer, top_configs, insights) -> str:
        """Generate comprehensive optimization report"""
        
        try:
            execution_time = (time.time() - self.start_time) / 60
            
            report = f"""
================================================================
OPTIMIZED MAIL-TO-CALLS PREDICTION SYSTEM
================================================================
OPTIMIZATION REPORT

EXECUTIVE SUMMARY:
-----------------
Optimization Status: {'SUCCESS' if optimizer.best_configuration else 'PARTIAL'}
Execution Time: {execution_time:.1f} minutes
Configurations Tested: {len(optimizer.optimization_results)}
Best {CONFIG['optimization_metric'].upper()}: {optimizer.best_score:.3f}
Best Configuration: {optimizer.best_configuration['config_id'] if optimizer.best_configuration else 'None'}

MAIL TYPE ANALYSIS RESULTS:
--------------------------"""
            
            # Mail type analysis summary
            if 'top_by_volume' in data_loader.mail_analysis:
                report += f"\nTop 5 Mail Types by Volume:"
                for i, mail_type in enumerate(data_loader.mail_analysis['top_by_volume'][:5]):
                    volume = data_loader.mail_analysis['volume_analysis'][mail_type]['total_volume']
                    report += f"\n  {i+1}. {mail_type}: {volume:,.0f} total volume"
            
            if 'top_by_correlation' in data_loader.mail_analysis:
                report += f"\n\nTop 5 Mail Types by Correlation:"
                for i, mail_type in enumerate(data_loader.mail_analysis['top_by_correlation'][:5]):
                    corr = data_loader.mail_analysis['correlation_analysis'][mail_type]['best_correlation']
                    lag = data_loader.mail_analysis['correlation_analysis'][mail_type]['best_lag']
                    report += f"\n  {i+1}. {mail_type}: {corr:.3f} correlation (lag {lag} days)"
            
            report += f"\n\nOPTIMIZATION INSIGHTS:\n{'-'*25}"
            
            # Strategy effectiveness
            if 'mail_strategy_analysis' in insights:
                report += f"\nMail Strategy Effectiveness:"
                for strategy, data in insights['mail_strategy_analysis'].items():
                    report += f"\n  {strategy}: Avg {CONFIG['optimization_metric']} = {data['avg_score']:.3f} ({data['count']} tests)"
            
            # Model effectiveness
            if 'model_analysis' in insights:
                report += f"\n\nModel Effectiveness:"
                for model, data in insights['model_analysis'].items():
                    report += f"\n  {model}: Avg {CONFIG['optimization_metric']} = {data['avg_score']:.3f} ({data['count']} tests)"
            
            # Top configurations
            report += f"\n\nTOP {len(top_configs)} CONFIGURATIONS:\n{'-'*30}"
            
            for i, config in enumerate(top_configs[:10]):
                config_info = config['results']['configuration']
                report += f"""
{i+1}. Configuration: {config['config_id']}
   Score ({CONFIG['optimization_metric']}): {config['score']:.3f}
   Mail Strategy: {config_info['mail_strategy']}
   Model: {config_info['model_name']}
   Features: {config['results']['features']}
   Test R²: {config['results'].get('test_r2', 'N/A')}
   Test MAE: {config['results'].get('test_mae', 'N/A')}"""
            
            # Best configuration details
            if optimizer.best_configuration:
                best_config = optimizer.best_configuration
                best_results = optimizer.optimization_results[best_config['config_id']]
                
                report += f"""

BEST CONFIGURATION DETAILS:
===========================
Configuration ID: {best_config['config_id']}
Mail Strategy: {best_config['mail_strategy']}
Model: {best_config['model_name']}
Feature Configuration: {best_config['feature_config']}
Lag Configuration: {best_config['lag_config']}

Performance Metrics:
- CV {CONFIG['optimization_metric'].upper()}: {optimizer.best_score:.3f}
- CV MAE: {best_results.get('cv_mae', 'N/A')}
- Test R²: {best_results.get('test_r2', 'N/A')}
- Test MAE: {best_results.get('test_mae', 'N/A')}
- Features Used: {best_results.get('features', 'N/A')}

Top Mail Types Used:
{chr(10).join([f"  {i+1}. {mail_type}" for i, mail_type in enumerate(best_config['mail_types'][:10])])}"""
                
                if 'feature_importance' in best_results:
                    report += f"\n\nTop 10 Most Important Features:"
                    for i, (feature, importance) in enumerate(best_results['feature_importance'][:10]):
                        report += f"\n  {i+1}. {feature}: {importance:.4f}"
            
            report += f"""

PRODUCTION DEPLOYMENT:
=====================
Status: {'READY' if optimizer.best_model else 'NOT READY'}
Model File: {'best_optimized_model.pkl' if optimizer.best_model else 'Not available'}
Configuration File: {'best_model_config.json' if optimizer.best_model else 'Not available'}

Usage Instructions:
1. Load the best model: joblib.load('best_optimized_model.pkl')
2. Use the exact mail types and feature configuration from best_model_config.json
3. Ensure feature engineering matches the optimized pipeline
4. Apply to new mail data for call volume predictions

BUSINESS VALUE:
==============
• Optimized mail type selection based on volume-correlation tradeoff
• Systematic testing of {len(optimizer.optimization_results)} configurations
• Best performing model achieves {optimizer.best_score:.1%} {CONFIG['optimization_metric']} score
• Production-ready prediction system
• Data-driven mail campaign impact assessment

NEXT STEPS:
==========
1. Deploy the best optimized model to production
2. Set up automated daily predictions using optimized configuration
3. Monitor performance and retrain with new data
4. Use mail type insights for campaign planning

================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Optimization Method: Systematic Grid Search
Total Runtime: {execution_time:.1f} minutes
================================================================
"""
            
            # Save report
            report_path = self.output_dir / "OPTIMIZATION_REPORT.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # Print key sections
            safe_print(report)
            
            return str(report_path)
            
        except Exception as e:
            LOG.error(f"Report generation failed: {str(e)}")
            return ""

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    safe_print("="*80)
    safe_print("OPTIMIZED MAIL-TO-CALLS PREDICTION SYSTEM")
    safe_print("="*80)
    safe_print("SYSTEMATIC OPTIMIZATION APPROACH:")
    safe_print("  1. Analyze mail types by VOLUME and CORRELATION")
    safe_print("  2. Find optimal tradeoff between volume and correlation")
    safe_print("  3. Test comprehensive feature combinations")
    safe_print("  4. Systematic model testing and selection")
    safe_print("  5. Output the HIGHEST ACCURACY configuration")
    safe_print("")
    safe_print("GOAL: Maximum prediction accuracy through systematic optimization")
    safe_print("="*80)
    safe_print("")
    
    try:
        orchestrator = OptimizationOrchestrator()
        results = orchestrator.run_complete_optimization()
        
        if results['success']:
            safe_print("\n" + "="*60)
            safe_print("OPTIMIZATION COMPLETED SUCCESSFULLY!")
            safe_print("="*60)
            safe_print("")
            safe_print("OPTIMAL CONFIGURATION FOUND:")
            
            if results['best_configuration']:
                best_config = results['best_configuration']
                safe_print(f"  Configuration: {best_config['config_id']}")
                safe_print(f"  Best Score: {results['best_score']:.3f}")
                safe_print(f"  Mail Strategy: {best_config['mail_strategy']}")
                safe_print(f"  Model: {best_config['model_name']}")
                safe_print(f"  Mail Types Used: {len(best_config['mail_types'])}")
                safe_print("")
                safe_print("TOP MAIL TYPES:")
                for i, mail_type in enumerate(best_config['mail_types'][:5]):
                    safe_print(f"    {i+1}. {mail_type}")
                safe_print("")
            
            safe_print("OPTIMIZATION INSIGHTS:")
            safe_print(f"  Total Configurations Tested: {len(results.get('optimization_insights', {}))}")
            safe_print(f"  Execution Time: {results['execution_time_minutes']:.1f} minutes")
            safe_print(f"  Results Saved: {results['output_directory']}")
            safe_print("")
            safe_print("PRODUCTION READY:")
            safe_print("  * Best optimized model saved")
            safe_print("  * Complete configuration documented") 
            safe_print("  * Mail type analysis completed")
            safe_print("  * Feature engineering optimized")
            
        else:
            safe_print("\n" + "="*50)
            safe_print("OPTIMIZATION FAILED")
            safe_print("="*50)
            safe_print(f"Error: {results['error']}")
            safe_print("Check logs for detailed information")
        
        return 0 if results['success'] else 1
        
    except KeyboardInterrupt:
        safe_print("\nOptimization interrupted by user")
        return 1
        
    except Exception as e:
        safe_print(f"\nSystem error: {str(e)}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
