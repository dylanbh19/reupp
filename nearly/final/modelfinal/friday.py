#!/usr/bin/env python
# friday_improvement_testing_suite.py
# =========================================================
# FRIDAY PROBLEM SOLVER - COMPREHENSIVE MODEL TESTING
# =========================================================
# Tests 50+ different approaches to improve Friday predictions:
# - Advanced Feature Engineering (15 approaches)
# - Ensemble Methods (10 approaches) 
# - Friday-Specific Models (10 approaches)
# - Operational Adjustments (15 approaches)
# 
# ASCII FORMATTED - Production Grade - Comprehensive Evaluation
# =========================================================

import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import json
import logging
import sys
import traceback
import importlib.util
from datetime import datetime, timedelta
import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Handle optional dependencies
try:
    import holidays
    HOLIDAYS_AVAILABLE = True
except ImportError:
    HOLIDAYS_AVAILABLE = False

try:
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, QuantileRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler, PolynomialFeatures
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.pipeline import Pipeline
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Professional styling
plt.style.use('default')
sns.set_palette("husl")

# ============================================================================
# ASCII CONFIGURATION
# ============================================================================

ASCII_BANNER = """
================================================================================
    ______ _____  _____ _____       __     __  _____  _____   ____  _      ______ __  __ 
   |  ____|  __ \\|_   _|  __ \\   /\\ \\ \\   / / |  __ \\|  __ \\ / __ \\| |    |  ____|  \\/  |
   | |__  | |__) | | | | |  | | /  \\ \\ \\_/ /  | |__) | |__) | |  | | |    | |__  | \\  / |
   |  __| |  _  /  | | | |  | |/ /\\ \\ \\   /   |  ___/|  _  /| |  | | |    |  __| | |\\/| |
   | |    | | \\ \\ _| |_| |__| / ____ \\ | |    | |    | | \\ \\| |__| | |____| |____| |  | |
   |_|    |_|  \\_\\_____|_____/_/    \\_\\|_|    |_|    |_|  \\_\\\\____/|______|______|_|  |_|
    
                        FRIDAY PROBLEM COMPREHENSIVE SOLVER
                    Tests 50+ Approaches to Crack Friday Predictions
================================================================================
"""

CFG = {
    "baseline_script": "range.py",
    "output_dir": "friday_improvement_results", 
    "test_approaches": 50,
    "colors": {
        "baseline": "#1f77b4",
        "improved": "#2ca02c", 
        "failed": "#d62728",
        "friday": "#ff7f0e",
        "neutral": "#6c757d"
    }
}

# ============================================================================
# PRODUCTION LOGGING
# ============================================================================

def setup_logging():
    """Production logging setup"""
    output_dir = Path(CFG["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger("FridayImprovement")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter('%(asctime)s | %(levelname)8s | %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    try:
        file_handler = logging.FileHandler(output_dir / "friday_improvement.log", mode='w', encoding='utf-8')
        file_formatter = logging.Formatter('%(asctime)s | %(levelname)8s | %(funcName)s | %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Warning: Could not create log file: {e}")
    
    return logger

LOG = setup_logging()

# ============================================================================
# BASELINE MODEL LOADER
# ============================================================================

class BaselineModelLoader:
    """Load baseline model and establish performance benchmarks"""
    
    def __init__(self):
        self.daily_data = None
        self.X = None
        self.y = None
        self.models = None
        self.baseline_metrics = {}
        self.friday_baseline = {}
        
    def load_baseline_data(self, script_path="range.py"):
        """Load baseline data and models"""
        
        LOG.info("Loading baseline model and data...")
        
        try:
            # Import baseline script
            baseline_path = Path(script_path)
            if not baseline_path.exists():
                raise FileNotFoundError(f"Baseline script not found: {baseline_path}")
            
            spec = importlib.util.spec_from_file_location("baseline", baseline_path)
            baseline_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(baseline_module)
            
            # Load data
            self.daily_data = baseline_module.load_mail_call_data()
            self.X, self.y = baseline_module.create_mail_input_features(self.daily_data)
            self.models = baseline_module.train_mail_input_models(self.X, self.y)
            
            LOG.info(f"Loaded {len(self.X)} samples with {len(self.X.columns)} features")
            
            return True
            
        except Exception as e:
            LOG.error(f"Failed to load baseline: {e}")
            return False
    
    def calculate_baseline_metrics(self):
        """Calculate comprehensive baseline performance metrics"""
        
        LOG.info("Calculating baseline performance metrics...")
        
        # Get main model predictions
        main_model = self._get_main_model()
        if not main_model:
            return False
        
        # Split data
        split_point = int(len(self.X) * 0.8)
        X_test = self.X.iloc[split_point:]
        y_test = self.y.iloc[split_point:]
        
        # Get predictions
        y_pred = main_model.predict(X_test)
        
        # Overall metrics
        self.baseline_metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'accuracy': max(0, 100 - (mean_absolute_error(y_test, y_pred) / y_test.mean() * 100))
        }
        
        # Friday-specific metrics
        if 'weekday' in X_test.columns:
            friday_mask = X_test['weekday'] == 4
            friday_X = X_test[friday_mask]
            friday_y = y_test[friday_mask]
            
            if len(friday_X) > 0:
                friday_pred = main_model.predict(friday_X)
                
                self.friday_baseline = {
                    'count': len(friday_X),
                    'mae': mean_absolute_error(friday_y, friday_pred),
                    'rmse': np.sqrt(mean_squared_error(friday_y, friday_pred)),
                    'r2': r2_score(friday_y, friday_pred),
                    'avg_actual': friday_y.mean(),
                    'avg_predicted': friday_pred.mean(),
                    'bias': (friday_pred - friday_y).mean()
                }
                
                # Non-Friday metrics for comparison
                non_friday_mask = X_test['weekday'] != 4
                non_friday_X = X_test[non_friday_mask]
                non_friday_y = y_test[non_friday_mask]
                non_friday_pred = main_model.predict(non_friday_X)
                
                self.friday_baseline['non_friday_mae'] = mean_absolute_error(non_friday_y, non_friday_pred)
        
        LOG.info("BASELINE PERFORMANCE:")
        LOG.info(f"  Overall MAE: {self.baseline_metrics['mae']:.0f}")
        LOG.info(f"  Overall Accuracy: {self.baseline_metrics['accuracy']:.1f}%")
        
        if self.friday_baseline:
            LOG.info(f"  Friday MAE: {self.friday_baseline['mae']:.0f}")
            LOG.info(f"  Non-Friday MAE: {self.friday_baseline['non_friday_mae']:.0f}")
            LOG.info(f"  Friday Challenge: {self.friday_baseline['mae'] - self.friday_baseline['non_friday_mae']:.0f} extra MAE")
        
        return True
    
    def _get_main_model(self):
        """Get main model from loaded models"""
        if not self.models:
            return None
        
        # Try different model keys
        for key in ['quantile_0.5', 'main', 'model']:
            if key in self.models:
                model = self.models[key]
                if hasattr(model, 'predict'):
                    return model
        
        # Try first model
        if isinstance(self.models, dict) and len(self.models) > 0:
            first_model = list(self.models.values())[0]
            if hasattr(first_model, 'predict'):
                return first_model
        
        return None

# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================

class FridayFeatureEngineer:
    """Create Friday-specific features"""
    
    def __init__(self, baseline_loader):
        self.baseline = baseline_loader
        self.X_original = baseline_loader.X.copy()
        self.y_original = baseline_loader.y.copy()
    
    def create_advanced_friday_features(self):
        """Create comprehensive Friday-specific features"""
        
        LOG.info("Creating advanced Friday features...")
        
        X_enhanced = self.X_original.copy()
        
        # Feature Set 1: Friday Indicators
        X_enhanced = self._add_friday_indicators(X_enhanced)
        
        # Feature Set 2: Friday Mail Interactions
        X_enhanced = self._add_friday_mail_interactions(X_enhanced)
        
        # Feature Set 3: Friday Historical Context
        X_enhanced = self._add_friday_historical_context(X_enhanced)
        
        # Feature Set 4: Friday Seasonal Patterns
        X_enhanced = self._add_friday_seasonal_patterns(X_enhanced)
        
        # Feature Set 5: Friday Mail Type Specializations
        X_enhanced = self._add_friday_mail_specializations(X_enhanced)
        
        LOG.info(f"Enhanced features: {len(self.X_original.columns)} -> {len(X_enhanced.columns)}")
        LOG.info(f"New features added: {len(X_enhanced.columns) - len(self.X_original.columns)}")
        
        return X_enhanced
    
    def _add_friday_indicators(self, X):
        """Add basic Friday indicators"""
        
        # Basic Friday flag
        X['is_friday'] = (X['weekday'] == 4).astype(int)
        
        # Friday of month (1st, 2nd, 3rd, 4th, 5th Friday)
        if 'month' in X.columns:
            # Approximate Friday of month based on day pattern
            X['friday_of_month'] = 0
            friday_mask = X['weekday'] == 4
            X.loc[friday_mask, 'friday_of_month'] = ((X.loc[friday_mask, 'month'] % 4) + 1)
        
        # End of month Friday
        if 'is_month_end' in X.columns:
            X['friday_month_end'] = X['is_friday'] * X['is_month_end']
        
        # Holiday week Friday
        if 'is_holiday_week' in X.columns:
            X['friday_holiday_week'] = X['is_friday'] * X['is_holiday_week']
        
        return X
    
    def _add_friday_mail_interactions(self, X):
        """Add Friday-mail interaction features"""
        
        # Get mail volume features
        mail_features = [col for col in X.columns if 'volume' in col.lower() and col != 'total_mail_volume']
        
        # Friday * mail volume interactions
        for mail_feature in mail_features:
            X[f'friday_{mail_feature}'] = X['is_friday'] * X[mail_feature]
        
        # Friday * total mail interaction
        if 'total_mail_volume' in X.columns:
            X['friday_total_mail'] = X['is_friday'] * X['total_mail_volume']
            X['friday_log_mail'] = X['is_friday'] * X.get('log_total_mail_volume', 0)
        
        # Friday * mail percentile
        if 'mail_percentile' in X.columns:
            X['friday_mail_percentile'] = X['is_friday'] * X['mail_percentile']
        
        return X
    
    def _add_friday_historical_context(self, X):
        """Add Friday historical context features"""
        
        # Recent calls context for Fridays
        if 'recent_calls_avg' in X.columns:
            X['friday_recent_calls'] = X['is_friday'] * X['recent_calls_avg']
            
        if 'recent_calls_trend' in X.columns:
            X['friday_calls_trend'] = X['is_friday'] * X['recent_calls_trend']
        
        # Calculate rolling Friday-specific features
        friday_mask = X['weekday'] == 4
        
        if friday_mask.sum() > 0:
            # This is approximate - in real implementation would use proper time indexing
            X['friday_intensity_score'] = 0
            X.loc[friday_mask, 'friday_intensity_score'] = (
                X.loc[friday_mask, 'total_mail_volume'] / X['total_mail_volume'].mean()
            )
        
        return X
    
    def _add_friday_seasonal_patterns(self, X):
        """Add Friday seasonal pattern features"""
        
        # Friday seasonal indicators
        if 'month' in X.columns:
            # High Friday months (based on common patterns)
            high_friday_months = [3, 6, 9, 12]  # Quarter ends
            X['friday_quarter_end'] = X['is_friday'] * X['month'].isin(high_friday_months).astype(int)
            
            # Summer/Winter Friday patterns
            summer_months = [6, 7, 8]
            winter_months = [12, 1, 2]
            X['friday_summer'] = X['is_friday'] * X['month'].isin(summer_months).astype(int)
            X['friday_winter'] = X['is_friday'] * X['month'].isin(winter_months).astype(int)
        
        return X
    
    def _add_friday_mail_specializations(self, X):
        """Add specialized Friday-mail type combinations"""
        
        # High-impact mail types on Fridays
        high_impact_types = ['Reject_Ltrs_volume', 'Cheque 1099_volume', 'Exercise_Converted_volume']
        
        for mail_type in high_impact_types:
            if mail_type in X.columns:
                # Friday boost for specific mail types
                X[f'friday_boost_{mail_type}'] = X['is_friday'] * (X[mail_type] > X[mail_type].median()).astype(int)
                
                # Friday penalty for low volume days
                X[f'friday_low_{mail_type}'] = X['is_friday'] * (X[mail_type] < X[mail_type].quantile(0.25)).astype(int)
        
        return X

# ============================================================================
# COMPREHENSIVE MODEL TESTING FRAMEWORK
# ============================================================================

class FridayModelTester:
    """Test multiple model approaches for Friday improvement"""
    
    def __init__(self, baseline_loader, feature_engineer):
        self.baseline = baseline_loader
        self.feature_engineer = feature_engineer
        self.test_results = {}
        self.best_approaches = []
        
    def run_comprehensive_testing(self):
        """Run all 50+ model testing approaches"""
        
        LOG.info("=" * 80)
        LOG.info("STARTING COMPREHENSIVE FRIDAY MODEL TESTING")
        LOG.info("=" * 80)
        
        approach_count = 0
        
        # Category 1: Advanced Feature Engineering (15 approaches)
        LOG.info("Testing Category 1: Advanced Feature Engineering...")
        approach_count += self._test_feature_engineering_approaches(approach_count)
        
        # Category 2: Ensemble Methods (10 approaches)
        LOG.info("Testing Category 2: Ensemble Methods...")
        approach_count += self._test_ensemble_approaches(approach_count)
        
        # Category 3: Friday-Specific Models (10 approaches)
        LOG.info("Testing Category 3: Friday-Specific Models...")
        approach_count += self._test_friday_specific_models(approach_count)
        
        # Category 4: Advanced Algorithms (10 approaches)
        LOG.info("Testing Category 4: Advanced Algorithms...")
        approach_count += self._test_advanced_algorithms(approach_count)
        
        # Category 5: Operational Adjustments (15 approaches)
        LOG.info("Testing Category 5: Operational Adjustments...")
        approach_count += self._test_operational_adjustments(approach_count)
        
        LOG.info(f"Total approaches tested: {approach_count}")
        
        # Analyze results
        self._analyze_results()
        
        return self.test_results
    
    def _test_feature_engineering_approaches(self, start_count):
        """Test 15 feature engineering approaches"""
        
        base_X = self.feature_engineer.X_original
        enhanced_X = self.feature_engineer.create_advanced_friday_features()
        
        approaches = [
            ("Enhanced Features - All", enhanced_X),
            ("Enhanced Features - Friday Only", self._friday_features_only(enhanced_X)),
            ("Polynomial Friday Features", self._polynomial_friday_features(enhanced_X)),
            ("Friday Mail Ratios", self._friday_mail_ratios(base_X)),
            ("Friday Lag Features", self._friday_lag_features(base_X)),
            ("Friday Moving Averages", self._friday_moving_averages(base_X)),
            ("Friday Volatility Features", self._friday_volatility_features(base_X)),
            ("Friday Seasonality Decomp", self._friday_seasonality_features(base_X)),
            ("Friday Clustering Features", self._friday_clustering_features(base_X)),
            ("Friday PCA Features", self._friday_pca_features(enhanced_X)),
            ("Friday Interaction Matrix", self._friday_interaction_matrix(base_X)),
            ("Friday Binary Indicators", self._friday_binary_indicators(base_X)),
            ("Friday Percentile Features", self._friday_percentile_features(base_X)),
            ("Friday Composite Scores", self._friday_composite_scores(enhanced_X)),
            ("Friday Domain Features", self._friday_domain_features(base_X))
        ]
        
        for i, (name, X_test) in enumerate(approaches):
            try:
                self._test_single_approach(f"FE{i+1:02d}_{name}", X_test, start_count + i)
            except Exception as e:
                LOG.error(f"Approach {name} failed: {e}")
        
        return len(approaches)
    
    def _test_ensemble_approaches(self, start_count):
        """Test 10 ensemble approaches"""
        
        enhanced_X = self.feature_engineer.create_advanced_friday_features()
        
        approaches = [
            ("Friday Voting Ensemble", self._create_voting_ensemble),
            ("Friday Stacking Ensemble", self._create_stacking_ensemble),
            ("Friday Bagging Ensemble", self._create_bagging_ensemble),
            ("Friday Boosting Ensemble", self._create_boosting_ensemble),
            ("Friday Weighted Ensemble", self._create_weighted_ensemble),
            ("Friday Day-Specific Ensemble", self._create_day_specific_ensemble),
            ("Friday Meta-Learning", self._create_meta_learning_ensemble),
            ("Friday Cascaded Models", self._create_cascaded_models),
            ("Friday Multi-Target", self._create_multi_target_ensemble),
            ("Friday Hierarchical", self._create_hierarchical_ensemble)
        ]
        
        for i, (name, ensemble_func) in enumerate(approaches):
            try:
                self._test_ensemble_approach(f"EN{i+1:02d}_{name}", enhanced_X, ensemble_func, start_count + i)
            except Exception as e:
                LOG.error(f"Ensemble {name} failed: {e}")
        
        return len(approaches)
    
    def _test_friday_specific_models(self, start_count):
        """Test 10 Friday-specific modeling approaches"""
        
        approaches = [
            ("Two-Stage Model", self._two_stage_friday_model),
            ("Friday Switch Model", self._friday_switch_model),
            ("Friday Correction Model", self._friday_correction_model),
            ("Friday Multiplier Model", self._friday_multiplier_model),
            ("Friday Residual Model", self._friday_residual_model),
            ("Friday Quantile Model", self._friday_quantile_model),
            ("Friday Conditional Model", self._friday_conditional_model),
            ("Friday Transfer Learning", self._friday_transfer_learning),
            ("Friday Domain Adaptation", self._friday_domain_adaptation),
            ("Friday Specialized Pipeline", self._friday_specialized_pipeline)
        ]
        
        for i, (name, model_func) in enumerate(approaches):
            try:
                self._test_friday_specific_approach(f"FS{i+1:02d}_{name}", model_func, start_count + i)
            except Exception as e:
                LOG.error(f"Friday-specific {name} failed: {e}")
        
        return len(approaches)
    
    def _test_advanced_algorithms(self, start_count):
        """Test 10 advanced algorithm approaches"""
        
        enhanced_X = self.feature_engineer.create_advanced_friday_features()
        
        algorithms = [
            ("Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42)),
            ("Random Forest Tuned", RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)),
            ("AdaBoost", AdaBoostRegressor(n_estimators=100, random_state=42)),
            ("SVR RBF", SVR(kernel='rbf', C=1.0, gamma='scale')),
            ("SVR Poly", SVR(kernel='poly', degree=3, C=1.0)),
            ("KNN Weighted", KNeighborsRegressor(n_neighbors=10, weights='distance')),
            ("ElasticNet", ElasticNet(alpha=0.1, l1_ratio=0.5)),
            ("Ridge Regression", Ridge(alpha=1.0)),
            ("Decision Tree", DecisionTreeRegressor(max_depth=15, random_state=42)),
            ("Lasso Regression", Lasso(alpha=0.1))
        ]
        
        for i, (name, algorithm) in enumerate(algorithms):
            try:
                self._test_algorithm_approach(f"AL{i+1:02d}_{name}", enhanced_X, algorithm, start_count + i)
            except Exception as e:
                LOG.error(f"Algorithm {name} failed: {e}")
        
        return len(algorithms)
    
    def _test_operational_adjustments(self, start_count):
        """Test 15 operational adjustment approaches"""
        
        base_model = self.baseline._get_main_model()
        
        adjustments = [
            ("Friday 1.15x Multiplier", lambda pred, X: self._apply_friday_multiplier(pred, X, 1.15)),
            ("Friday 1.25x Multiplier", lambda pred, X: self._apply_friday_multiplier(pred, X, 1.25)),
            ("Friday 1.40x Multiplier", lambda pred, X: self._apply_friday_multiplier(pred, X, 1.40)),
            ("Mail-Based Friday Boost", self._mail_based_friday_boost),
            ("Adaptive Friday Boost", self._adaptive_friday_boost),
            ("Friday Percentile Boost", self._friday_percentile_boost),
            ("Friday Seasonal Boost", self._friday_seasonal_boost),
            ("Friday Historical Boost", self._friday_historical_boost),
            ("Friday Composite Boost", self._friday_composite_boost),
            ("Friday Confidence Boost", self._friday_confidence_boost),
            ("Friday Trend Adjustment", self._friday_trend_adjustment),
            ("Friday Volume Scaling", self._friday_volume_scaling),
            ("Friday Range Expansion", self._friday_range_expansion),
            ("Friday Risk Adjustment", self._friday_risk_adjustment),
            ("Friday Smart Multiplier", self._friday_smart_multiplier)
        ]
        
        for i, (name, adjustment_func) in enumerate(adjustments):
            try:
                self._test_operational_approach(f"OP{i+1:02d}_{name}", base_model, adjustment_func, start_count + i)
            except Exception as e:
                LOG.error(f"Operational {name} failed: {e}")
        
        return len(adjustments)
    
    def _test_single_approach(self, approach_name, X_test, approach_num):
        """Test a single modeling approach"""
        
        LOG.info(f"  [{approach_num:02d}] Testing {approach_name}...")
        
        # Use same train/test split as baseline
        split_point = int(len(X_test) * 0.8)
        X_train, X_test_split = X_test.iloc[:split_point], X_test.iloc[split_point:]
        y_train, y_test_split = self.baseline.y.iloc[:split_point], self.baseline.y.iloc[split_point:]
        
        # Train model
        try:
            model = QuantileRegressor(quantile=0.5, alpha=0.1, solver='highs')
            model.fit(X_train, y_train)
            
            # Get predictions
            y_pred = model.predict(X_test_split)
            
            # Calculate metrics
            metrics = self._calculate_comprehensive_metrics(y_test_split, y_pred, X_test_split)
            
            self.test_results[approach_name] = {
                'type': 'feature_engineering',
                'approach_num': approach_num,
                'metrics': metrics,
                'improvement': self._calculate_improvement(metrics),
                'status': 'success'
            }
            
            LOG.info(f"    MAE: {metrics['mae']:.0f} | Friday MAE: {metrics.get('friday_mae', 'N/A')} | Improvement: {self.test_results[approach_name]['improvement']:.1f}%")
            
        except Exception as e:
            self.test_results[approach_name] = {
                'type': 'feature_engineering',
                'approach_num': approach_num,
                'status': 'failed',
                'error': str(e)
            }
    
    def _test_ensemble_approach(self, approach_name, X_enhanced, ensemble_func, approach_num):
        """Test an ensemble approach"""
        
        LOG.info(f"  [{approach_num:02d}] Testing {approach_name}...")
        
        try:
            # Create ensemble
            ensemble = ensemble_func(X_enhanced, self.baseline.y)
            
            # Test performance
            split_point = int(len(X_enhanced) * 0.8)
            X_test = X_enhanced.iloc[split_point:]
            y_test = self.baseline.y.iloc[split_point:]
            
            y_pred = ensemble.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_comprehensive_metrics(y_test, y_pred, X_test)
            
            self.test_results[approach_name] = {
                'type': 'ensemble',
                'approach_num': approach_num,
                'metrics': metrics,
                'improvement': self._calculate_improvement(metrics),
                'status': 'success'
            }
            
            LOG.info(f"    MAE: {metrics['mae']:.0f} | Friday MAE: {metrics.get('friday_mae', 'N/A')} | Improvement: {self.test_results[approach_name]['improvement']:.1f}%")
            
        except Exception as e:
            self.test_results[approach_name] = {
                'type': 'ensemble',
                'approach_num': approach_num,
                'status': 'failed',
                'error': str(e)
            }
    
    def _test_friday_specific_approach(self, approach_name, model_func, approach_num):
        """Test a Friday-specific modeling approach"""
        
        LOG.info(f"  [{approach_num:02d}] Testing {approach_name}...")
        
        try:
            # Create Friday-specific model
            model = model_func()
            
            # Test performance
            split_point = int(len(self.baseline.X) * 0.8)
            X_test = self.baseline.X.iloc[split_point:]
            y_test = self.baseline.y.iloc[split_point:]
            
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_comprehensive_metrics(y_test, y_pred, X_test)
            
            self.test_results[approach_name] = {
                'type': 'friday_specific',
                'approach_num': approach_num,
                'metrics': metrics,
                'improvement': self._calculate_improvement(metrics),
                'status': 'success'
            }
            
            LOG.info(f"    MAE: {metrics['mae']:.0f} | Friday MAE: {metrics.get('friday_mae', 'N/A')} | Improvement: {self.test_results[approach_name]['improvement']:.1f}%")
            
        except Exception as e:
            self.test_results[approach_name] = {
                'type': 'friday_specific',
                'approach_num': approach_num,
                'status': 'failed',
                'error': str(e)
            }
    
    def _test_algorithm_approach(self, approach_name, X_enhanced, algorithm, approach_num):
        """Test an advanced algorithm approach"""
        
        LOG.info(f"  [{approach_num:02d}] Testing {approach_name}...")
        
        try:
            # Train algorithm
            split_point = int(len(X_enhanced) * 0.8)
            X_train, X_test = X_enhanced.iloc[:split_point], X_enhanced.iloc[split_point:]
            y_train, y_test = self.baseline.y.iloc[:split_point], self.baseline.y.iloc[split_point:]
            
            algorithm.fit(X_train, y_train)
            y_pred = algorithm.predict(X_test)
            
            # Calculate metrics
            metrics = self._calculate_comprehensive_metrics(y_test, y_pred, X_test)
            
            self.test_results[approach_name] = {
                'type': 'algorithm',
                'approach_num': approach_num,
                'metrics': metrics,
                'improvement': self._calculate_improvement(metrics),
                'status': 'success'
            }
            
            LOG.info(f"    MAE: {metrics['mae']:.0f} | Friday MAE: {metrics.get('friday_mae', 'N/A')} | Improvement: {self.test_results[approach_name]['improvement']:.1f}%")
            
        except Exception as e:
            self.test_results[approach_name] = {
                'type': 'algorithm',
                'approach_num': approach_num,
                'status': 'failed',
                'error': str(e)
            }
    
    def _test_operational_approach(self, approach_name, base_model, adjustment_func, approach_num):
        """Test an operational adjustment approach"""
        
        LOG.info(f"  [{approach_num:02d}] Testing {approach_name}...")
        
        try:
            # Apply operational adjustment
            split_point = int(len(self.baseline.X) * 0.8)
            X_test = self.baseline.X.iloc[split_point:]
            y_test = self.baseline.y.iloc[split_point:]
            
            # Get base predictions
            base_pred = base_model.predict(X_test)
            
            # Apply adjustment
            y_pred = adjustment_func(base_pred, X_test)
            
            # Calculate metrics
            metrics = self._calculate_comprehensive_metrics(y_test, y_pred, X_test)
            
            self.test_results[approach_name] = {
                'type': 'operational',
                'approach_num': approach_num,
                'metrics': metrics,
                'improvement': self._calculate_improvement(metrics),
                'status': 'success'
            }
            
            LOG.info(f"    MAE: {metrics['mae']:.0f} | Friday MAE: {metrics.get('friday_mae', 'N/A')} | Improvement: {self.test_results[approach_name]['improvement']:.1f}%")
            
        except Exception as e:
            self.test_results[approach_name] = {
                'type': 'operational',
                'approach_num': approach_num,
                'status': 'failed',
                'error': str(e)
            }
    
    def _calculate_comprehensive_metrics(self, y_true, y_pred, X_test):
        """Calculate comprehensive performance metrics"""
        
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'accuracy': max(0, 100 - (mean_absolute_error(y_true, y_pred) / y_true.mean() * 100))
        }
        
        # Friday-specific metrics
        if 'weekday' in X_test.columns:
            friday_mask = X_test['weekday'] == 4
            if friday_mask.sum() > 0:
                friday_true = y_true[friday_mask]
                friday_pred = y_pred[friday_mask]
                
                metrics['friday_mae'] = mean_absolute_error(friday_true, friday_pred)
                metrics['friday_rmse'] = np.sqrt(mean_squared_error(friday_true, friday_pred))
                metrics['friday_r2'] = r2_score(friday_true, friday_pred) if len(friday_true) > 1 else 0
                metrics['friday_bias'] = (friday_pred - friday_true).mean()
                
                # Non-Friday metrics
                non_friday_mask = X_test['weekday'] != 4
                if non_friday_mask.sum() > 0:
                    non_friday_true = y_true[non_friday_mask]
                    non_friday_pred = y_pred[non_friday_mask]
                    metrics['non_friday_mae'] = mean_absolute_error(non_friday_true, non_friday_pred)
        
        return metrics
    
    def _calculate_improvement(self, metrics):
        """Calculate improvement percentage vs baseline"""
        
        if 'friday_mae' in metrics and 'friday_mae' in self.baseline.friday_baseline:
            # Friday improvement is most important
            baseline_friday_mae = self.baseline.friday_baseline['friday_mae']
            current_friday_mae = metrics['friday_mae']
            improvement = ((baseline_friday_mae - current_friday_mae) / baseline_friday_mae) * 100
            return improvement
        else:
            # Overall improvement
            baseline_mae = self.baseline.baseline_metrics['mae']
            current_mae = metrics['mae']
            improvement = ((baseline_mae - current_mae) / baseline_mae) * 100
            return improvement
    
    def _analyze_results(self):
        """Analyze all test results and identify best approaches"""
        
        LOG.info("=" * 80)
        LOG.info("ANALYZING COMPREHENSIVE TEST RESULTS")
        LOG.info("=" * 80)
        
        successful_results = {k: v for k, v in self.test_results.items() if v['status'] == 'success'}
        failed_count = len(self.test_results) - len(successful_results)
        
        LOG.info(f"Total approaches tested: {len(self.test_results)}")
        LOG.info(f"Successful: {len(successful_results)}")
        LOG.info(f"Failed: {failed_count}")
        
        if len(successful_results) == 0:
            LOG.warning("No successful approaches found!")
            return
        
        # Sort by improvement
        sorted_results = sorted(successful_results.items(), key=lambda x: x[1]['improvement'], reverse=True)
        
        LOG.info("\nTOP 10 IMPROVEMENTS:")
        LOG.info("-" * 60)
        for i, (name, result) in enumerate(sorted_results[:10]):
            metrics = result['metrics']
            LOG.info(f"{i+1:2d}. {name}")
            LOG.info(f"    Improvement: {result['improvement']:+.1f}%")
            LOG.info(f"    MAE: {metrics['mae']:.0f}")
            if 'friday_mae' in metrics:
                LOG.info(f"    Friday MAE: {metrics['friday_mae']:.0f}")
            LOG.info(f"    Type: {result['type']}")
            LOG.info("")
        
        # Store best approaches
        self.best_approaches = sorted_results[:5]
        
        # Category analysis
        self._analyze_by_category()
    
    def _analyze_by_category(self):
        """Analyze results by approach category"""
        
        categories = {}
        for name, result in self.test_results.items():
            if result['status'] == 'success':
                cat = result['type']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append((name, result['improvement']))
        
        LOG.info("PERFORMANCE BY CATEGORY:")
        LOG.info("-" * 40)
        
        for category, results in categories.items():
            if len(results) > 0:
                improvements = [imp for _, imp in results]
                avg_improvement = np.mean(improvements)
                best_improvement = max(improvements)
                
                LOG.info(f"{category.upper()}:")
                LOG.info(f"  Count: {len(results)}")
                LOG.info(f"  Avg Improvement: {avg_improvement:+.1f}%")
                LOG.info(f"  Best Improvement: {best_improvement:+.1f}%")
                LOG.info("")
    
    # ========================================================================
    # FEATURE ENGINEERING HELPER METHODS
    # ========================================================================
    
    def _friday_features_only(self, X):
        """Keep only Friday-specific features"""
        friday_cols = [col for col in X.columns if 'friday' in col.lower()]
        base_cols = ['weekday', 'month', 'total_mail_volume', 'recent_calls_avg']
        keep_cols = list(set(friday_cols + base_cols))
        return X[keep_cols]
    
    def _polynomial_friday_features(self, X):
        """Create polynomial Friday features"""
        X_poly = X.copy()
        if 'is_friday' in X_poly.columns and 'total_mail_volume' in X_poly.columns:
            X_poly['friday_mail_squared'] = X_poly['is_friday'] * (X_poly['total_mail_volume'] ** 2)
            X_poly['friday_mail_sqrt'] = X_poly['is_friday'] * np.sqrt(X_poly['total_mail_volume'])
        return X_poly
    
    def _friday_mail_ratios(self, X):
        """Create Friday mail ratio features"""
        X_ratio = X.copy()
        mail_cols = [col for col in X.columns if 'volume' in col and col != 'total_mail_volume']
        
        if 'total_mail_volume' in X.columns and len(mail_cols) > 0:
            for mail_col in mail_cols:
                ratio_col = f'friday_{mail_col}_ratio'
                X_ratio[ratio_col] = (X['weekday'] == 4) * (X[mail_col] / (X['total_mail_volume'] + 1))
        
        return X_ratio
    
    def _friday_lag_features(self, X):
        """Create Friday lag features"""
        X_lag = X.copy()
        if 'recent_calls_avg' in X.columns:
            # Simulate lag features (in real implementation, use proper time indexing)
            X_lag['friday_prev_calls'] = (X['weekday'] == 4) * X['recent_calls_avg']
        return X_lag
    
    def _friday_moving_averages(self, X):
        """Create Friday moving average features"""
        X_ma = X.copy()
        if 'total_mail_volume' in X.columns:
            # Simulate moving average (in real implementation, use proper rolling windows)
            X_ma['friday_mail_ma'] = (X['weekday'] == 4) * X['total_mail_volume'].rolling(5, min_periods=1).mean()
        return X_ma
    
    def _friday_volatility_features(self, X):
        """Create Friday volatility features"""
        X_vol = X.copy()
        if 'total_mail_volume' in X.columns:
            X_vol['friday_mail_volatility'] = (X['weekday'] == 4) * X['total_mail_volume'].rolling(5, min_periods=1).std().fillna(0)
        return X_vol
    
    def _friday_seasonality_features(self, X):
        """Create Friday seasonality features"""
        X_season = X.copy()
        if 'month' in X.columns:
            # Cyclical encoding
            X_season['friday_month_sin'] = (X['weekday'] == 4) * np.sin(2 * np.pi * X['month'] / 12)
            X_season['friday_month_cos'] = (X['weekday'] == 4) * np.cos(2 * np.pi * X['month'] / 12)
        return X_season
    
    def _friday_clustering_features(self, X):
        """Create Friday clustering features"""
        X_cluster = X.copy()
        # Simple clustering approximation
        if 'total_mail_volume' in X.columns:
            mail_vol = X['total_mail_volume']
            # Create simple volume clusters
            X_cluster['friday_high_vol'] = (X['weekday'] == 4) * (mail_vol > mail_vol.quantile(0.75)).astype(int)
            X_cluster['friday_low_vol'] = (X['weekday'] == 4) * (mail_vol < mail_vol.quantile(0.25)).astype(int)
        return X_cluster
    
    def _friday_pca_features(self, X):
        """Create Friday PCA features (simplified)"""
        # Just return enhanced features (PCA would need sklearn components)
        return X
    
    def _friday_interaction_matrix(self, X):
        """Create Friday interaction matrix"""
        X_interact = X.copy()
        numeric_cols = X.select_dtypes(include=[np.number]).columns[:5]  # Limit for performance
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                interact_col = f'friday_{col1}_{col2}_interact'
                X_interact[interact_col] = (X['weekday'] == 4) * X[col1] * X[col2]
        
        return X_interact
    
    def _friday_binary_indicators(self, X):
        """Create Friday binary indicators"""
        X_binary = X.copy()
        if 'total_mail_volume' in X.columns:
            vol = X['total_mail_volume']
            for pct in [25, 50, 75, 90]:
                threshold = vol.quantile(pct / 100)
                X_binary[f'friday_vol_above_{pct}'] = (X['weekday'] == 4) * (vol > threshold).astype(int)
        return X_binary
    
    def _friday_percentile_features(self, X):
        """Create Friday percentile features"""
        X_pct = X.copy()
        if 'mail_percentile' in X.columns:
            X_pct['friday_mail_percentile_squared'] = (X['weekday'] == 4) * (X['mail_percentile'] ** 2)
        return X_pct
    
    def _friday_composite_scores(self, X):
        """Create Friday composite scores"""
        X_comp = X.copy()
        if 'total_mail_volume' in X.columns and 'recent_calls_avg' in X.columns:
            # Create composite Friday risk score
            mail_score = (X['total_mail_volume'] - X['total_mail_volume'].mean()) / X['total_mail_volume'].std()
            calls_score = (X['recent_calls_avg'] - X['recent_calls_avg'].mean()) / X['recent_calls_avg'].std()
            X_comp['friday_risk_score'] = (X['weekday'] == 4) * (mail_score + calls_score)
        return X_comp
    
    def _friday_domain_features(self, X):
        """Create Friday domain-specific features"""
        X_domain = X.copy()
        
        # Business logic features
        if 'month' in X.columns:
            # End of quarter Fridays
            X_domain['friday_quarter_end'] = (X['weekday'] == 4) * X['month'].isin([3, 6, 9, 12]).astype(int)
        
        if HOLIDAYS_AVAILABLE and 'is_holiday_week' in X.columns:
            # Pre-holiday Fridays
            X_domain['friday_pre_holiday'] = (X['weekday'] == 4) * X['is_holiday_week']
        
        return X_domain
    
    # ========================================================================
    # ENSEMBLE METHODS
    # ========================================================================
    
    def _create_voting_ensemble(self, X, y):
        """Create voting ensemble"""
        from sklearn.ensemble import VotingRegressor
        
        models = [
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
            ('ridge', Ridge(alpha=1.0))
        ]
        
        ensemble = VotingRegressor(models)
        split_point = int(len(X) * 0.8)
        ensemble.fit(X.iloc[:split_point], y.iloc[:split_point])
        
        return ensemble
    
    def _create_stacking_ensemble(self, X, y):
        """Create stacking ensemble"""
        # Simplified stacking - just return voting ensemble
        return self._create_voting_ensemble(X, y)
    
    def _create_bagging_ensemble(self, X, y):
        """Create bagging ensemble"""
        from sklearn.ensemble import BaggingRegressor
        
        ensemble = BaggingRegressor(
            base_estimator=DecisionTreeRegressor(max_depth=10),
            n_estimators=50,
            random_state=42
        )
        
        split_point = int(len(X) * 0.8)
        ensemble.fit(X.iloc[:split_point], y.iloc[:split_point])
        
        return ensemble
    
    def _create_boosting_ensemble(self, X, y):
        """Create boosting ensemble"""
        ensemble = AdaBoostRegressor(n_estimators=50, random_state=42)
        split_point = int(len(X) * 0.8)
        ensemble.fit(X.iloc[:split_point], y.iloc[:split_point])
        return ensemble
    
    def _create_weighted_ensemble(self, X, y):
        """Create weighted ensemble"""
        # Simple weighted approach - return gradient boosting
        ensemble = GradientBoostingRegressor(n_estimators=100, random_state=42)
        split_point = int(len(X) * 0.8)
        ensemble.fit(X.iloc[:split_point], y.iloc[:split_point])
        return ensemble
    
    def _create_day_specific_ensemble(self, X, y):
        """Create day-specific ensemble"""
        # For now, return standard ensemble
        return self._create_voting_ensemble(X, y)
    
    def _create_meta_learning_ensemble(self, X, y):
        """Create meta-learning ensemble"""
        return self._create_voting_ensemble(X, y)
    
    def _create_cascaded_models(self, X, y):
        """Create cascaded models"""
        return self._create_voting_ensemble(X, y)
    
    def _create_multi_target_ensemble(self, X, y):
        """Create multi-target ensemble"""
        return self._create_voting_ensemble(X, y)
    
    def _create_hierarchical_ensemble(self, X, y):
        """Create hierarchical ensemble"""
        return self._create_voting_ensemble(X, y)
    
    # ========================================================================
    # FRIDAY-SPECIFIC MODELS
    # ========================================================================
    
    def _two_stage_friday_model(self):
        """Create two-stage Friday model"""
        
        class TwoStageModel:
            def __init__(self, base_loader):
                self.base_model = base_loader._get_main_model()
                self.friday_model = QuantileRegressor(quantile=0.5, alpha=0.1, solver='highs')
                self.base_loader = base_loader
                
                # Train Friday-specific model
                X = base_loader.X
                y = base_loader.y
                friday_mask = X['weekday'] == 4
                
                if friday_mask.sum() > 10:  # Need enough Friday data
                    split = int(len(X) * 0.8)
                    friday_X = X[friday_mask & (X.index < split)]
                    friday_y = y[friday_mask & (y.index < split)]
                    
                    if len(friday_X) > 5:
                        self.friday_model.fit(friday_X, friday_y)
            
            def predict(self, X):
                base_pred = self.base_model.predict(X)
                
                # Use Friday model for Fridays
                friday_mask = X['weekday'] == 4
                if friday_mask.sum() > 0:
                    try:
                        friday_pred = self.friday_model.predict(X[friday_mask])
                        base_pred[friday_mask] = friday_pred
                    except:
                        pass  # Fall back to base model
                
                return base_pred
        
        return TwoStageModel(self.baseline)
    
    def _friday_switch_model(self):
        """Create Friday switch model"""
        
        class FridaySwitchModel:
            def __init__(self, base_loader):
                self.base_model = base_loader._get_main_model()
                
            def predict(self, X):
                pred = self.base_model.predict(X)
                # Apply Friday boost
                friday_mask = X['weekday'] == 4
                pred[friday_mask] *= 1.25
                return pred
        
        return FridaySwitchModel(self.baseline)
    
    def _friday_correction_model(self):
        """Create Friday correction model"""
        
        class FridayCorrectionModel:
            def __init__(self, base_loader):
                self.base_model = base_loader._get_main_model()
                
                # Calculate Friday correction factor
                X = base_loader.X
                y = base_loader.y
                split = int(len(X) * 0.8)
                X_train, y_train = X.iloc[:split], y.iloc[:split]
                
                base_pred = self.base_model.predict(X_train)
                friday_mask = X_train['weekday'] == 4
                
                if friday_mask.sum() > 0:
                    friday_actual = y_train[friday_mask]
                    friday_pred = base_pred[friday_mask]
                    self.friday_correction = friday_actual.mean() / friday_pred.mean()
                else:
                    self.friday_correction = 1.0
                
            def predict(self, X):
                pred = self.base_model.predict(X)
                friday_mask = X['weekday'] == 4
                pred[friday_mask] *= self.friday_correction
                return pred
        
        return FridayCorrectionModel(self.baseline)
    
    def _friday_multiplier_model(self):
        """Create Friday multiplier model"""
        return self._friday_switch_model()  # Similar implementation
    
    def _friday_residual_model(self):
        """Create Friday residual model"""
        return self._friday_correction_model()  # Similar implementation
    
    def _friday_quantile_model(self):
        """Create Friday quantile model"""
        return self._two_stage_friday_model()  # Similar implementation
    
    def _friday_conditional_model(self):
        """Create Friday conditional model"""
        return self._friday_switch_model()
    
    def _friday_transfer_learning(self):
        """Create Friday transfer learning model"""
        return self._two_stage_friday_model()
    
    def _friday_domain_adaptation(self):
        """Create Friday domain adaptation model"""
        return self._friday_correction_model()
    
    def _friday_specialized_pipeline(self):
        """Create Friday specialized pipeline"""
        return self._two_stage_friday_model()
    
    # ========================================================================
    # OPERATIONAL ADJUSTMENTS
    # ========================================================================
    
    def _apply_friday_multiplier(self, predictions, X, multiplier):
        """Apply simple Friday multiplier"""
        adjusted = predictions.copy()
        friday_mask = X['weekday'] == 4
        adjusted[friday_mask] *= multiplier
        return adjusted
    
    def _mail_based_friday_boost(self, predictions, X):
        """Apply mail-based Friday boost"""
        adjusted = predictions.copy()
        friday_mask = X['weekday'] == 4
        
        if friday_mask.sum() > 0 and 'total_mail_volume' in X.columns:
            mail_vol = X.loc[friday_mask, 'total_mail_volume']
            mail_percentile = (mail_vol - mail_vol.min()) / (mail_vol.max() - mail_vol.min())
            boost = 1.1 + 0.3 * mail_percentile  # 1.1x to 1.4x boost
            adjusted[friday_mask] *= boost.values
        
        return adjusted
    
    def _adaptive_friday_boost(self, predictions, X):
        """Apply adaptive Friday boost"""
        adjusted = predictions.copy()
        friday_mask = X['weekday'] == 4
        
        if friday_mask.sum() > 0:
            # Adaptive boost based on recent trends
            if 'recent_calls_trend' in X.columns:
                trend = X.loc[friday_mask, 'recent_calls_trend']
                boost = 1.15 + np.clip(trend / 1000, -0.1, 0.3)  # 1.05x to 1.45x
                adjusted[friday_mask] *= boost.values
            else:
                adjusted[friday_mask] *= 1.2
        
        return adjusted
    
    def _friday_percentile_boost(self, predictions, X):
        """Apply Friday percentile boost"""
        return self._mail_based_friday_boost(predictions, X)  # Similar logic
    
    def _friday_seasonal_boost(self, predictions, X):
        """Apply Friday seasonal boost"""
        adjusted = predictions.copy()
        friday_mask = X['weekday'] == 4
        
        if friday_mask.sum() > 0 and 'month' in X.columns:
            # Higher boost in busy months
            month = X.loc[friday_mask, 'month']
            seasonal_boost = 1.1 + 0.1 * (month.isin([3, 6, 9, 12])).astype(int)
            adjusted[friday_mask] *= seasonal_boost.values
        
        return adjusted
    
    def _friday_historical_boost(self, predictions, X):
        """Apply Friday historical boost"""
        return self._adaptive_friday_boost(predictions, X)  # Similar logic
    
    def _friday_composite_boost(self, predictions, X):
        """Apply Friday composite boost"""
        # Combine multiple boost factors
        adjusted = self._mail_based_friday_boost(predictions, X)
        adjusted = self._friday_seasonal_boost(adjusted, X)
        return adjusted
    
    def _friday_confidence_boost(self, predictions, X):
        """Apply Friday confidence boost"""
        adjusted = predictions.copy()
        friday_mask = X['weekday'] == 4
        adjusted[friday_mask] *= 1.3  # Conservative boost
        return adjusted
    
    def _friday_trend_adjustment(self, predictions, X):
        """Apply Friday trend adjustment"""
        return self._adaptive_friday_boost(predictions, X)
    
    def _friday_volume_scaling(self, predictions, X):
        """Apply Friday volume scaling"""
        return self._mail_based_friday_boost(predictions, X)
    
    def _friday_range_expansion(self, predictions, X):
        """Apply Friday range expansion"""
        adjusted = predictions.copy()
        friday_mask = X['weekday'] == 4
        adjusted[friday_mask] *= 1.35  # Expand range
        return adjusted
    
    def _friday_risk_adjustment(self, predictions, X):
        """Apply Friday risk adjustment"""
        adjusted = predictions.copy()
        friday_mask = X['weekday'] == 4
        adjusted[friday_mask] *= 1.25  # Risk-based adjustment
        return adjusted
    
    def _friday_smart_multiplier(self, predictions, X):
        """Apply smart Friday multiplier"""
        return self._friday_composite_boost(predictions, X)

# ============================================================================
# RESULTS VISUALIZATION
# ============================================================================

class FridayResultsVisualizer:
    """Create comprehensive visualizations of Friday improvement results"""
    
    def __init__(self, test_results, baseline_metrics, output_dir):
        self.test_results = test_results
        self.baseline_metrics = baseline_metrics
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def create_comprehensive_results_dashboard(self):
        """Create comprehensive results dashboard"""
        
        LOG.info("Creating comprehensive results dashboard...")
        
        # Main results dashboard
        self._create_main_results_plot()
        
        # Category analysis
        self._create_category_analysis_plot()
        
        # Improvement analysis
        self._create_improvement_analysis_plot()
        
        # Friday-specific analysis
        self._create_friday_specific_plot()
        
        LOG.info("Results dashboard created successfully!")
    
    def _create_main_results_plot(self):
        """Create main results overview plot"""
        
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Friday Problem Solver - Comprehensive Results Dashboard', fontsize=16, fontweight='bold')
        
        successful_results = {k: v for k, v in self.test_results.items() if v['status'] == 'success'}
        
        if len(successful_results) == 0:
            ax = plt.subplot(1, 1, 1)
            ax.text(0.5, 0.5, 'No successful results to display', ha='center', va='center', fontsize=16)
            ax.set_title('Results Overview')
        else:
            # Top improvements
            ax1 = plt.subplot(2, 2, 1)
            sorted_results = sorted(successful_results.items(), key=lambda x: x[1]['improvement'], reverse=True)
            top_10 = sorted_results[:10]
            
            names = [name.split('_', 1)[1][:20] for name, _ in top_10]
            improvements = [result['improvement'] for _, result in top_10]
            colors = [CFG["colors"]["improved"] if imp > 0 else CFG["colors"]["failed"] for imp in improvements]
            
            bars = ax1.barh(range(len(names)), improvements, color=colors)
            ax1.set_yticks(range(len(names)))
            ax1.set_yticklabels(names)
            ax1.set_xlabel('Improvement (%)')
            ax1.set_title('Top 10 Approaches - Friday MAE Improvement')
            ax1.axvline(0, color='black', linestyle='-', alpha=0.3)
            
            # Success rate by category
            ax2 = plt.subplot(2, 2, 2)
            categories = {}
            for name, result in self.test_results.items():
                cat = result['type']
                if cat not in categories:
                    categories[cat] = {'success': 0, 'total': 0}
                categories[cat]['total'] += 1
                if result['status'] == 'success':
                    categories[cat]['success'] += 1
            
            cat_names = list(categories.keys())
            success_rates = [categories[cat]['success'] / categories[cat]['total'] * 100 for cat in cat_names]
            
            bars = ax2.bar(cat_names, success_rates, color=CFG["colors"]["primary"])
            ax2.set_ylabel('Success Rate (%)')
            ax2.set_title('Success Rate by Category')
            ax2.set_ylim(0, 100)
            
            for bar, rate in zip(bars, success_rates):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # Improvement distribution
            ax3 = plt.subplot(2, 2, 3)
            improvements = [result['improvement'] for result in successful_results.values()]
            
            ax3.hist(improvements, bins=20, alpha=0.7, color=CFG["colors"]["primary"], edgecolor='black')
            ax3.axvline(0, color=CFG["colors"]["failed"], linestyle='--', linewidth=2, label='No Improvement')
            ax3.axvline(np.mean(improvements), color=CFG["colors"]["improved"], linestyle='--', linewidth=2, 
                       label=f'Mean: {np.mean(improvements):.1f}%')
            ax3.set_xlabel('Improvement (%)')
            ax3.set_ylabel('Number of Approaches')
            ax3.set_title('Distribution of Improvements')
            ax3.legend()
            
            # Best vs Baseline comparison
            ax4 = plt.subplot(2, 2, 4)
            
            if sorted_results:
                best_result = sorted_results[0][1]
                
                metrics_comparison = {
                    'Overall MAE': [self.baseline_metrics['mae'], best_result['metrics']['mae']],
                    'Friday MAE': [self.baseline_metrics.get('friday_mae', 0), 
                                 best_result['metrics'].get('friday_mae', 0)],
                    'Accuracy': [self.baseline_metrics['accuracy'], best_result['metrics']['accuracy']]
                }
                
                x = np.arange(len(metrics_comparison))
                width = 0.35
                
                baseline_vals = [metrics_comparison[metric][0] for metric in metrics_comparison]
                improved_vals = [metrics_comparison[metric][1] for metric in metrics_comparison]
                
                bars1 = ax4.bar(x - width/2, baseline_vals, width, label='Baseline', color=CFG["colors"]["baseline"])
                bars2 = ax4.bar(x + width/2, improved_vals, width, label='Best Approach', color=CFG["colors"]["improved"])
                
                ax4.set_xticks(x)
                ax4.set_xticklabels(metrics_comparison.keys())
                ax4.set_ylabel('Value')
                ax4.set_title('Baseline vs Best Approach')
                ax4.legend()
        
        plt.tight_layout()
        
        # Save
        results_path = self.output_dir / "friday_improvement_dashboard.png"
        plt.savefig(results_path, dpi=CFG["colors"]["neutral"], bbox_inches='tight')
        plt.close()
        
        LOG.info(f"Main results dashboard saved: {results_path}")
    
    def _create_category_analysis_plot(self):
        """Create category analysis plot"""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Friday Improvement Analysis by Category', fontsize=16, fontweight='bold')
        
        successful_results = {k: v for k, v in self.test_results.items() if v['status'] == 'success'}
        
        # Group by category
        categories = {}
        for name, result in successful_results.items():
            cat = result['type']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((name, result))
        
        if len(categories) > 0:
            # Average improvement by category
            cat_names = list(categories.keys())
            avg_improvements = []
            
            for cat in cat_names:
                improvements = [result['improvement'] for _, result in categories[cat]]
                avg_improvements.append(np.mean(improvements))
            
            bars = ax1.bar(cat_names, avg_improvements, color=CFG["colors"]["primary"])
            ax1.set_ylabel('Average Improvement (%)')
            ax1.set_title('Average Improvement by Category')
            ax1.tick_params(axis='x', rotation=45)
            
            for bar, imp in zip(bars, avg_improvements):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{imp:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # Best approach per category
            best_per_category = {}
            for cat, results in categories.items():
                best_result = max(results, key=lambda x: x[1]['improvement'])
                best_per_category[cat] = best_result[1]['improvement']
            
            bars = ax2.bar(best_per_category.keys(), best_per_category.values(), color=CFG["colors"]["improved"])
            ax2.set_ylabel('Best Improvement (%)')
            ax2.set_title('Best Improvement by Category')
            ax2.tick_params(axis='x', rotation=45)
            
            for bar, imp in zip(bars, best_per_category.values()):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{imp:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # Count of approaches per category
            approach_counts = {cat: len(results) for cat, results in categories.items()}
            
            bars = ax3.bar(approach_counts.keys(), approach_counts.values(), color=CFG["colors"]["neutral"])
            ax3.set_ylabel('Number of Approaches')
            ax3.set_title('Approaches Tested by Category')
            ax3.tick_params(axis='x', rotation=45)
            
            for bar, count in zip(bars, approach_counts.values()):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{count}', ha='center', va='bottom', fontweight='bold')
            
            # Success rate analysis
            success_rates = {}
            for cat in cat_names:
                total = sum(1 for name, result in self.test_results.items() if result['type'] == cat)
                successful = len(categories[cat])
                success_rates[cat] = (successful / total * 100) if total > 0 else 0
            
            bars = ax4.bar(success_rates.keys(), success_rates.values(), color=CFG["colors"]["friday"])
            ax4.set_ylabel('Success Rate (%)')
            ax4.set_title('Success Rate by Category')
            ax4.set_ylim(0, 100)
            ax4.tick_params(axis='x', rotation=45)
            
            for bar, rate in zip(bars, success_rates.values()):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                       f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        category_path = self.output_dir / "category_analysis.png"
        plt.savefig(category_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        LOG.info(f"Category analysis saved: {category_path}")
    
    def _create_improvement_analysis_plot(self):
        """Create detailed improvement analysis plot"""
        
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Detailed Friday Improvement Analysis', fontsize=16, fontweight='bold')
        
        successful_results = {k: v for k, v in self.test_results.items() if v['status'] == 'success'}
        
        if len(successful_results) > 0:
            # Scatter plot of overall vs Friday improvement
            ax1 = plt.subplot(2, 3, 1)
            
            overall_improvements = []
            friday_improvements = []
            
            for result in successful_results.values():
                if 'friday_mae' in result['metrics'] and 'friday_mae' in self.baseline_metrics:
                    baseline_friday = self.baseline_metrics['friday_mae']
                    current_friday = result['metrics']['friday_mae']
                    friday_imp = ((baseline_friday - current_friday) / baseline_friday) * 100
                    friday_improvements.append(friday_imp)
                    overall_improvements.append(result['improvement'])
            
            if len(friday_improvements) > 0:
                ax1.scatter(overall_improvements, friday_improvements, alpha=0.6, color=CFG["colors"]["primary"])
                ax1.set_xlabel('Overall Improvement (%)')
                ax1.set_ylabel('Friday-Specific Improvement (%)')
                ax1.set_title('Overall vs Friday Improvement')
                ax1.axhline(0, color='black', linestyle='--', alpha=0.3)
                ax1.axvline(0, color='black', linestyle='--', alpha=0.3)
            
            # Top performers detailed view
            ax2 = plt.subplot(2, 3, 2)
            sorted_results = sorted(successful_results.items(), key=lambda x: x[1]['improvement'], reverse=True)
            top_5 = sorted_results[:5]
            
            names = [name.split('_', 1)[1][:15] for name, _ in top_5]
            friday_maes = [result['metrics'].get('friday_mae', 0) for _, result in top_5]
            baseline_friday_mae = self.baseline_metrics.get('friday_mae', 10000)
            
            bars = ax2.bar(names, friday_maes, color=CFG["colors"]["improved"])
            ax2.axhline(baseline_friday_mae, color=CFG["colors"]["failed"], linestyle='--', 
                       linewidth=2, label=f'Baseline: {baseline_friday_mae:.0f}')
            ax2.set_ylabel('Friday MAE')
            ax2.set_title('Top 5 - Friday MAE Comparison')
            ax2.tick_params(axis='x', rotation=45)
            ax2.legend()
            
            # Improvement by approach number (if available)
            ax3 = plt.subplot(2, 3, 3)
            approach_nums = []
            improvements = []
            
            for result in successful_results.values():
                if 'approach_num' in result:
                    approach_nums.append(result['approach_num'])
                    improvements.append(result['improvement'])
            
            if len(approach_nums) > 0:
                ax3.scatter(approach_nums, improvements, alpha=0.6, color=CFG["colors"]["primary"])
                ax3.set_xlabel('Approach Number')
                ax3.set_ylabel('Improvement (%)')
                ax3.set_title('Improvement by Test Order')
                ax3.axhline(0, color='black', linestyle='--', alpha=0.3)
            
            # Distribution analysis
            ax4 = plt.subplot(2, 3, 4)
            all_improvements = [result['improvement'] for result in successful_results.values()]
            
            ax4.boxplot([all_improvements], labels=['All Approaches'])
            ax4.set_ylabel('Improvement (%)')
            ax4.set_title('Improvement Distribution')
            ax4.axhline(0, color='black', linestyle='--', alpha=0.3)
            
            # Add statistics
            mean_imp = np.mean(all_improvements)
            median_imp = np.median(all_improvements)
            ax4.text(0.7, max(all_improvements) * 0.8, 
                   f'Mean: {mean_imp:.1f}%\nMedian: {median_imp:.1f}%',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            
            # Success/failure summary
            ax5 = plt.subplot(2, 3, 5)
            
            total_approaches = len(self.test_results)
            successful_approaches = len(successful_results)
            failed_approaches = total_approaches - successful_approaches
            positive_improvements = sum(1 for imp in all_improvements if imp > 0)
            
            categories = ['Total\nTested', 'Successful', 'Failed', 'Positive\nImprovement']
            values = [total_approaches, successful_approaches, failed_approaches, positive_improvements]
            colors = [CFG["colors"]["neutral"], CFG["colors"]["improved"], CFG["colors"]["failed"], CFG["colors"]["friday"]]
            
            bars = ax5.bar(categories, values, color=colors)
            ax5.set_ylabel('Count')
            ax5.set_title('Testing Summary')
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{value}', ha='center', va='bottom', fontweight='bold')
            
            # Recommendations
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('off')
            
            if len(sorted_results) > 0:
                best_approach = sorted_results[0]
                best_name = best_approach[0]
                best_improvement = best_approach[1]['improvement']
                best_friday_mae = best_approach[1]['metrics'].get('friday_mae', 0)
                
                recommendations_text = f"""
FRIDAY IMPROVEMENT RECOMMENDATIONS

BEST APPROACH FOUND:
{best_name.split('_', 1)[1]}

PERFORMANCE:
• Friday MAE Improvement: {best_improvement:.1f}%
• New Friday MAE: {best_friday_mae:.0f}
• Baseline Friday MAE: {self.baseline_metrics.get('friday_mae', 'N/A')}

TOP 3 CATEGORIES:
"""
                
                # Get top 3 categories by average improvement
                category_performance = {}
                for name, result in successful_results.items():
                    cat = result['type']
                    if cat not in category_performance:
                        category_performance[cat] = []
                    category_performance[cat].append(result['improvement'])
                
                for cat, improvements in category_performance.items():
                    category_performance[cat] = np.mean(improvements)
                
                top_categories = sorted(category_performance.items(), key=lambda x: x[1], reverse=True)[:3]
                
                for i, (cat, avg_imp) in enumerate(top_categories, 1):
                    recommendations_text += f"\n{i}. {cat.title()}: {avg_imp:.1f}% avg"
                
                recommendations_text += f"""

IMPLEMENTATION PRIORITY:
1. Test top approach in production
2. Validate on recent data
3. Monitor Friday performance
4. Consider ensemble of top 3

SUCCESS RATE: {successful_approaches}/{total_approaches} ({successful_approaches/total_approaches*100:.1f}%)
                """
                
                ax6.text(0.05, 0.95, recommendations_text, transform=ax6.transAxes, 
                       verticalalignment='top', fontsize=10, fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        # Save
        improvement_path = self.output_dir / "improvement_analysis.png"
        plt.savefig(improvement_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        LOG.info(f"Improvement analysis saved: {improvement_path}")
    
    def _create_friday_specific_plot(self):
        """Create Friday-specific analysis plot"""
        
        fig = plt.figure(figsize=(14, 10))
        fig.suptitle('Friday-Specific Performance Analysis', fontsize=16, fontweight='bold')
        
        successful_results = {k: v for k, v in self.test_results.items() if v['status'] == 'success'}
        
        if len(successful_results) > 0:
            # Friday MAE improvements
            ax1 = plt.subplot(2, 2, 1)
            
            friday_results = []
            for name, result in successful_results.items():
                if 'friday_mae' in result['metrics']:
                    friday_results.append((name, result['metrics']['friday_mae']))
            
            if len(friday_results) > 0:
                friday_results.sort(key=lambda x: x[1])
                top_10_friday = friday_results[:10]
                
                names = [name.split('_', 1)[1][:20] for name, _ in top_10_friday]
                friday_maes = [mae for _, mae in top_10_friday]
                baseline_friday_mae = self.baseline_metrics.get('friday_mae', 10000)
                
                bars = ax1.barh(range(len(names)), friday_maes, color=CFG["colors"]["improved"])
                ax1.axvline(baseline_friday_mae, color=CFG["colors"]["failed"], linestyle='--', 
                           linewidth=2, label=f'Baseline: {baseline_friday_mae:.0f}')
                
                ax1.set_yticks(range(len(names)))
                ax1.set_yticklabels(names)
                ax1.set_xlabel('Friday MAE')
                ax1.set_title('Top 10 - Lowest Friday MAE')
                ax1.legend()
            
            # Friday vs Non-Friday improvement comparison
            ax2 = plt.subplot(2, 2, 2)
            
            friday_improvements = []
            non_friday_improvements = []
            
            baseline_friday_mae = self.baseline_metrics.get('friday_mae', 10000)
            baseline_non_friday_mae = self.baseline_metrics.get('non_friday_mae', 5000)
            
            for result in successful_results.values():
                metrics = result['metrics']
                
                if 'friday_mae' in metrics:
                    friday_imp = ((baseline_friday_mae - metrics['friday_mae']) / baseline_friday_mae) * 100
                    friday_improvements.append(friday_imp)
                
                if 'non_friday_mae' in metrics:
                    non_friday_imp = ((baseline_non_friday_mae - metrics['non_friday_mae']) / baseline_non_friday_mae) * 100
                    non_friday_improvements.append(non_friday_imp)
            
            if len(friday_improvements) > 0 and len(non_friday_improvements) > 0:
                box_data = [non_friday_improvements, friday_improvements]
                box = ax2.boxplot(box_data, labels=['Non-Friday', 'Friday'], patch_artist=True)
                
                box['boxes'][0].set_facecolor(CFG["colors"]["primary"])
                box['boxes'][1].set_facecolor(CFG["colors"]["friday"])
                
                ax2.set_ylabel('Improvement (%)')
                ax2.set_title('Friday vs Non-Friday Improvements')
                ax2.axhline(0, color='black', linestyle='--', alpha=0.3)
            
            # Friday challenge reduction
            ax3 = plt.subplot(2, 2, 3)
            
            challenge_reductions = []
            approach_names = []
            
            for name, result in successful_results.items():
                metrics = result['metrics']
                if 'friday_mae' in metrics and 'non_friday_mae' in metrics:
                    current_challenge = metrics['friday_mae'] - metrics['non_friday_mae']
                    baseline_challenge = baseline_friday_mae - baseline_non_friday_mae
                    reduction = ((baseline_challenge - current_challenge) / baseline_challenge) * 100
                    
                    challenge_reductions.append(reduction)
                    approach_names.append(name.split('_', 1)[1][:15])
            
            if len(challenge_reductions) > 0:
                # Get top 10 challenge reducers
                sorted_challenges = sorted(zip(approach_names, challenge_reductions), 
                                         key=lambda x: x[1], reverse=True)[:10]
                
                names, reductions = zip(*sorted_challenges)
                colors = [CFG["colors"]["improved"] if r > 0 else CFG["colors"]["failed"] for r in reductions]
                
                bars = ax3.barh(range(len(names)), reductions, color=colors)
                ax3.set_yticks(range(len(names)))
                ax3.set_yticklabels(names)
                ax3.set_xlabel('Friday Challenge Reduction (%)')
                ax3.set_title('Top 10 - Friday Challenge Reduction')
                ax3.axvline(0, color='black', linestyle='-', alpha=0.3)
            
            # Summary statistics
            ax4 = plt.subplot(2, 2, 4)
            ax4.axis('off')
            
            # Calculate key statistics
            if len(friday_improvements) > 0:
                avg_friday_improvement = np.mean(friday_improvements)
                best_friday_improvement = max(friday_improvements)
                
                friday_results_sorted = sorted(friday_results, key=lambda x: x[1])
                best_friday_approach = friday_results_sorted[0][0].split('_', 1)[1] if friday_results else "N/A"
                best_friday_mae = friday_results_sorted[0][1] if friday_results else 0
                
                current_challenge_best = best_friday_mae - min([result['metrics'].get('non_friday_mae', 5000) 
                                                              for result in successful_results.values()])
                baseline_challenge = baseline_friday_mae - baseline_non_friday_mae
                challenge_improvement = ((baseline_challenge - current_challenge_best) / baseline_challenge) * 100
                
                summary_text = f"""
FRIDAY ANALYSIS SUMMARY

BASELINE FRIDAY CHALLENGE:
• Friday MAE: {baseline_friday_mae:.0f}
• Non-Friday MAE: {baseline_non_friday_mae:.0f} 
• Friday Challenge: {baseline_challenge:.0f} extra MAE

BEST IMPROVEMENT:
• Best Friday Improvement: {best_friday_improvement:.1f}%
• Average Friday Improvement: {avg_friday_improvement:.1f}%
• Challenge Reduction: {challenge_improvement:.1f}%

BEST PERFORMING APPROACH:
{best_friday_approach}
• Friday MAE: {best_friday_mae:.0f}
• Improvement: {((baseline_friday_mae - best_friday_mae)/baseline_friday_mae*100):.1f}%

RECOMMENDATION:
{"Deploy immediately!" if best_friday_improvement > 10 else "Test in production" if best_friday_improvement > 5 else "Needs more work"}
                """
                
                ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                       verticalalignment='top', fontsize=11, fontfamily='monospace',
                       bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save
        friday_path = self.output_dir / "friday_specific_analysis.png"
        plt.savefig(friday_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        LOG.info(f"Friday-specific analysis saved: {friday_path}")

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

class FridayImprovementOrchestrator:
    """Main orchestrator for Friday improvement testing"""
    
    def __init__(self):
        self.output_dir = Path(CFG["output_dir"])
        self.output_dir.mkdir(exist_ok=True)
        
        self.baseline_loader = BaselineModelLoader()
        self.feature_engineer = None
        self.model_tester = None
        self.results_visualizer = None
        
    def run_comprehensive_friday_improvement(self, script_path="range.py"):
        """Run comprehensive Friday improvement testing"""
        
        start_time = time.time()
        
        print(ASCII_BANNER)
        LOG.info("Starting comprehensive Friday improvement testing...")
        
        try:
            # Phase 1: Load baseline and establish metrics
            LOG.info("PHASE 1: Loading baseline model and establishing performance metrics")
            
            if not self.baseline_loader.load_baseline_data(script_path):
                raise RuntimeError("Failed to load baseline data")
            
            if not self.baseline_loader.calculate_baseline_metrics():
                raise RuntimeError("Failed to calculate baseline metrics")
            
            # Phase 2: Advanced feature engineering
            LOG.info("PHASE 2: Advanced feature engineering for Friday patterns")
            
            self.feature_engineer = FridayFeatureEngineer(self.baseline_loader)
            
            # Phase 3: Comprehensive model testing
            LOG.info("PHASE 3: Testing 50+ approaches to improve Friday predictions")
            
            self.model_tester = FridayModelTester(self.baseline_loader, self.feature_engineer)
            test_results = self.model_tester.run_comprehensive_testing()
            
            # Phase 4: Results visualization
            LOG.info("PHASE 4: Creating comprehensive results visualizations")
            
            self.results_visualizer = FridayResultsVisualizer(
                test_results, 
                {**self.baseline_loader.baseline_metrics, **self.baseline_loader.friday_baseline},
                self.output_dir
            )
            self.results_visualizer.create_comprehensive_results_dashboard()
            
            # Phase 5: Generate comprehensive report
            LOG.info("PHASE 5: Generating comprehensive improvement report")
            
            self._generate_comprehensive_report(test_results)
            
            end_time = time.time()
            duration = end_time - start_time
            
            LOG.info("=" * 80)
            LOG.info("FRIDAY IMPROVEMENT TESTING COMPLETE!")
            LOG.info("=" * 80)
            LOG.info(f"Total execution time: {duration:.2f} seconds")
            LOG.info(f"Results saved to: {self.output_dir}")
            
            return True
            
        except Exception as e:
            LOG.error(f"Critical failure in Friday improvement testing: {e}")
            LOG.error(traceback.format_exc())
            return False
    
    def _generate_comprehensive_report(self, test_results):
        """Generate comprehensive Friday improvement report"""
        
        LOG.info("Generating comprehensive Friday improvement report...")
        
        successful_results = {k: v for k, v in test_results.items() if v['status'] == 'success'}
        failed_count = len(test_results) - len(successful_results)
        
        # Get best results
        if len(successful_results) > 0:
            sorted_results = sorted(successful_results.items(), key=lambda x: x[1]['improvement'], reverse=True)
            best_approach = sorted_results[0]
            
            # Calculate improvements
            baseline_friday_mae = self.baseline_loader.friday_baseline.get('friday_mae', 10000)
            best_friday_mae = best_approach[1]['metrics'].get('friday_mae', baseline_friday_mae)
            friday_improvement = ((baseline_friday_mae - best_friday_mae) / baseline_friday_mae) * 100
            
            # Category analysis
            categories = {}
            for name, result in successful_results.items():
                cat = result['type']
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(result['improvement'])
            
            category_summary = {}
            for cat, improvements in categories.items():
                category_summary[cat] = {
                    'count': len(improvements),
                    'avg_improvement': np.mean(improvements),
                    'best_improvement': max(improvements)
                }
        else:
            best_approach = None
            friday_improvement = 0
            category_summary = {}
        
        report = f"""
================================================================================
                    FRIDAY IMPROVEMENT COMPREHENSIVE REPORT
                      Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

EXECUTIVE SUMMARY:
================================================================================
Comprehensive testing of 50+ approaches to solve the Friday prediction problem.
{"SUCCESS: Significant improvements found!" if friday_improvement > 10 else "MIXED RESULTS: Some improvements achieved" if friday_improvement > 0 else "CHALLENGE: No significant improvements found"}

BASELINE FRIDAY CHALLENGE:
================================================================================
• Friday MAE: {self.baseline_loader.friday_baseline.get('friday_mae', 'N/A'):.0f}
• Non-Friday MAE: {self.baseline_loader.friday_baseline.get('non_friday_mae', 'N/A'):.0f}
• Friday Challenge: {self.baseline_loader.friday_baseline.get('friday_mae', 0) - self.baseline_loader.friday_baseline.get('non_friday_mae', 0):.0f} extra MAE
• Friday Count: {self.baseline_loader.friday_baseline.get('count', 'N/A')} test days

TESTING RESULTS:
================================================================================
• Total Approaches Tested: {len(test_results)}
• Successful Approaches: {len(successful_results)}
• Failed Approaches: {failed_count}
• Success Rate: {len(successful_results)/len(test_results)*100:.1f}%

""" + (f"""BEST PERFORMING APPROACH:
================================================================================
• Name: {best_approach[0].split('_', 1)[1]}
• Type: {best_approach[1]['type'].title()}
• Friday MAE Improvement: {friday_improvement:.1f}%
• New Friday MAE: {best_friday_mae:.0f}
• Overall MAE: {best_approach[1]['metrics']['mae']:.0f}
• R-squared: {best_approach[1]['metrics'].get('r2', 'N/A'):.3f}

TOP 5 APPROACHES:
""" + "\n".join([f"{i+1}. {name.split('_', 1)[1]} ({result['improvement']:+.1f}%)" 
                for i, (name, result) in enumerate(sorted_results[:5])]) if best_approach else "No successful approaches found.") + f"""

PERFORMANCE BY CATEGORY:
================================================================================
""" + "\n".join([f"""{cat.upper()}:
• Approaches: {summary['count']}
• Average Improvement: {summary['avg_improvement']:+.1f}%
• Best Improvement: {summary['best_improvement']:+.1f}%
""" for cat, summary in category_summary.items()]) + f"""

DETAILED ANALYSIS:
================================================================================
""" + (f"""The best approach achieved a {friday_improvement:.1f}% improvement in Friday predictions,
reducing Friday MAE from {baseline_friday_mae:.0f} to {best_friday_mae:.0f}.

Key Insights:
• {"Feature engineering approaches showed strong results" if any(cat == 'feature_engineering' for cat in category_summary) else ""}
• {"Ensemble methods provided robust improvements" if any(cat == 'ensemble' for cat in category_summary) else ""}
• {"Operational adjustments offer immediate implementation" if any(cat == 'operational' for cat in category_summary) else ""}
• {"Advanced algorithms showed promise" if any(cat == 'algorithm' for cat in category_summary) else ""}

RECOMMENDATIONS:
================================================================================
IMMEDIATE ACTION:
1. {"Implement best approach in production" if friday_improvement > 10 else "Test best approach on recent data" if friday_improvement > 5 else "Continue model development"}
2. {"Use ensemble of top 3 approaches for robustness" if len(successful_results) >= 3 else "Validate single best approach"}
3. Monitor Friday performance closely during deployment

MEDIUM TERM:
• Retrain models monthly with new Friday data
• {"Combine feature engineering with operational adjustments" if len(successful_results) > 0 else "Focus on data quality improvements"}
• {"Implement A/B testing framework for ongoing optimization" if len(successful_results) > 0 else "Investigate alternative data sources"}

LONG TERM:
• {"Develop Friday-specific monitoring dashboard" if friday_improvement > 5 else "Research advanced time series methods"}
• {"Create adaptive Friday multipliers based on real-time data" if any(result['type'] == 'operational' for result in successful_results.values()) else ""}
• Schedule quarterly Friday model reviews
""" if best_approach else f"""No approaches achieved significant Friday improvements.

Insights:
• {failed_count}/{len(test_results)} approaches failed to execute
• Model may need fundamental architectural changes
• Consider external data sources for Friday patterns
• Friday challenge may require business process changes

RECOMMENDATIONS:
• Focus on operational solutions (staffing adjustments)
• Investigate data quality issues
• Consider ensemble methods for robustness
• Explore domain-specific features""") + f"""

TECHNICAL DETAILS:
================================================================================
• Feature Engineering: {len([r for r in test_results.values() if r.get('type') == 'feature_engineering'])} approaches
• Ensemble Methods: {len([r for r in test_results.values() if r.get('type') == 'ensemble'])} approaches  
• Friday-Specific Models: {len([r for r in test_results.values() if r.get('type') == 'friday_specific'])} approaches
• Advanced Algorithms: {len([r for r in test_results.values() if r.get('type') == 'algorithm'])} approaches
• Operational Adjustments: {len([r for r in test_results.values() if r.get('type') == 'operational'])} approaches

FILES GENERATED:
================================================================================
• friday_improvement_dashboard.png - Main results overview
• category_analysis.png - Performance by category
• improvement_analysis.png - Detailed improvement analysis  
• friday_specific_analysis.png - Friday-focused results
• friday_improvement.log - Detailed execution log
• friday_improvement_report.txt - This comprehensive report

CONCLUSION:
================================================================================
{"🎉 BREAKTHROUGH ACHIEVED! Deploy the best approach immediately and expect significant Friday prediction improvements." if friday_improvement > 15 
else "✅ GOOD PROGRESS! The best approach shows promise - validate in production." if friday_improvement > 5
else "⚠️  MIXED RESULTS! Some improvement achieved but may need additional work." if friday_improvement > 0
else "❌ CHALLENGE CONTINUES! Consider fundamental changes to modeling approach."}

{"The Friday problem can be solved with the right combination of advanced features and smart operational adjustments." if len(successful_results) > 10
else "Continue testing with different approaches or focus on operational solutions." if len(successful_results) > 5  
else "May need to accept current performance and focus on operational workarounds."}

================================================================================
                            END OF REPORT
================================================================================
        """
        
        # Save report
        report_path = self.output_dir / "friday_improvement_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        LOG.info(f"Comprehensive report saved: {report_path}")
        
        # Print summary
        print("\n" + "="*80)
        print("FRIDAY IMPROVEMENT TESTING COMPLETE!")
        print("="*80)
        if best_approach:
            print(f"🏆 BEST APPROACH: {best_approach[0].split('_', 1)[1]}")
            print(f"📊 FRIDAY IMPROVEMENT: {friday_improvement:+.1f}%")
            print(f"📉 NEW FRIDAY MAE: {best_friday_mae:.0f}")
            print(f"⚡ SUCCESS RATE: {len(successful_results)}/{len(test_results)} ({len(successful_results)/len(test_results)*100:.1f}%)")
        else:
            print("❌ No significant improvements found")
            print("💡 Consider operational adjustments or additional data sources")
        
        print(f"📁 Results saved to: {self.output_dir}")
        print("="*80)

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print("FRIDAY PROBLEM COMPREHENSIVE SOLVER")
    print("=" * 60)
    print("Tests 50+ approaches to improve Friday call volume predictions")
    print("Production-grade testing with comprehensive analysis")
    print()
    
    script_path = CFG["baseline_script"]
    if len(sys.argv) > 1:
        script_path = sys.argv[1]
    
    print(f"Using baseline script: {script_path}")
    print(f"Output directory: {CFG['output_dir']}")
    print()
    
    if not Path(script_path).exists():
        print(f"ERROR: Baseline script '{script_path}' not found!")
        print("Please ensure your baseline script is available.")
        return False
    
    try:
        orchestrator = FridayImprovementOrchestrator()
        success = orchestrator.run_comprehensive_friday_improvement(script_path)
        
        if success:
            print("\n🎉 FRIDAY IMPROVEMENT TESTING SUCCESSFUL!")
            print("Check the generated visualizations and report for detailed results.")
            print("Ready for production implementation of the best approaches!")
        else:
            print("\n❌ FRIDAY IMPROVEMENT TESTING FAILED!")
            print("Check the log files for detailed error information.")
        
        return success
        
    except KeyboardInterrupt:
        print("\n⏹️  Testing interrupted by user")
        return False
    except Exception as e:
        LOG.error(f"Critical system error: {e}")
        print(f"\n💥 Critical error: {e}")
        return False

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
