#!/usr/bin/env python
“””
MAIL-TO-CALLS PREDICTION SYSTEM V2

- Integrates two economic data files
- Predicts call volume ranges (confidence intervals)
- Improved feature engineering consistency
- ASCII-only output for Windows compatibility
  “””

import warnings
warnings.filterwarnings(‘ignore’)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import sys
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import pearsonr
import joblib

# ============================================================================

# CONFIGURATION - CHANGE YOUR FILE PATHS HERE

# ============================================================================

CONFIG = {
# ============ YOUR FILE PATHS ============
“call_file”: “ACDMail.csv”,
“mail_file”: “mail.csv”,
“economic_data_file_1”: “expanded_economic_data.csv”,  # First economic data file
“economic_data_file_2”: “”,  # Second economic data file (leave empty if not using)
“holidays_file”: “us_holidays.csv”,

```
# ============ YOUR COLUMN NAMES ============
"call_date_col": "Date",
"call_volume_col": "ACDCalls",
"mail_date_col": "mail_date",
"mail_volume_col": "mail_volume",
"mail_type_col": "mail_type",

# ============ ANALYSIS SETTINGS ============
"output_dir": "mail_call_prediction_system_v2",
"top_mail_types": 8,
"test_size": 0.25,
"random_state": 42,

# Feature engineering
"max_lag_days": 7,
"rolling_windows": [3, 7],

# Prediction settings
"prediction_confidence": 0.8,  # 80% confidence interval
"n_estimators_for_intervals": 100,  # Trees for uncertainty estimation

# Visualization
"figure_size": (15, 10),
"plot_style": "seaborn-v0_8",
```

}

def safe_print(msg):
“”“Print ASCII-safe messages”””
try:
# Convert to string and replace non-ASCII characters
msg_str = str(msg)
ascii_msg = msg_str.encode(‘ascii’, ‘replace’).decode(‘ascii’)
print(ascii_msg)
except:
print(”[Error printing message]”)

def remove_us_holidays(df, date_col=‘date’):
“”“Remove US holidays from the DataFrame”””
safe_print(”   Removing US holidays from call data…”)

```
try:
    holidays_df = pd.read_csv(CONFIG["holidays_file"])
    holiday_dates = set(holidays_df['holiday_date'])
    
    initial_len = len(df)
    df = df[~df[date_col].dt.strftime('%Y-%m-%d').isin(holiday_dates)].copy()
    removed = initial_len - len(df)
    
    safe_print(f"   Removed {removed} holiday rows.")
    safe_print(f"   Data after holiday removal: {len(df)} rows.")
    return df
    
except FileNotFoundError:
    safe_print("   WARNING: holidays file not found, skipping holiday removal")
    return df
```

# ============================================================================

# STEP 1: DATA LOADING WITH DUAL ECONOMIC FILES

# ============================================================================

class DataManager:
def **init**(self):
self.call_data = None
self.mail_data = None
self.economic_data_1 = None
self.economic_data_2 = None
self.merged_data = None
self.output_dir = Path(CONFIG[“output_dir”])
self.output_dir.mkdir(exist_ok=True)

```
def load_call_data(self):
    """Load call data"""
    safe_print("=" * 80)
    safe_print("STEP 1A: LOADING CALL DATA")
    safe_print("=" * 80)
    
    try:
        df = pd.read_csv(CONFIG["call_file"])
        safe_print(f"   Loaded {len(df)} rows from {CONFIG['call_file']}")
        
        # Clean and process
        df_clean = df[[CONFIG["call_date_col"], CONFIG["call_volume_col"]]].copy()
        df_clean.columns = ['date', 'call_volume']
        
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        df_clean = df_clean.dropna()
        df_clean = df_clean[df_clean['call_volume'] > 5]
        
        # Business days only
        df_clean = df_clean[df_clean['date'].dt.weekday < 5]
        df_clean = remove_us_holidays(df_clean, 'date')
        
        df_clean = df_clean.sort_values('date').reset_index(drop=True)
        self.call_data = df_clean
        
        safe_print(f"   Clean call data: {len(df_clean)} business days")
        safe_print(f"   Date range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")
        
        return df_clean
        
    except Exception as e:
        safe_print(f"   ERROR loading call data: {e}")
        raise
        
def load_mail_data(self):
    """Load mail data"""
    safe_print("\n" + "=" * 80)
    safe_print("STEP 1B: LOADING MAIL DATA")
    safe_print("=" * 80)
    
    try:
        df = pd.read_csv(CONFIG["mail_file"], low_memory=False)
        safe_print(f"   Loaded {len(df)} rows from {CONFIG['mail_file']}")
        
        # Process mail data
        df['mail_date'] = pd.to_datetime(df[CONFIG["mail_date_col"]])
        df = df[df[CONFIG["mail_volume_col"]] > 0]
        
        # Aggregate by date and type
        daily_mail = df.groupby(['mail_date', CONFIG["mail_type_col"]])[CONFIG["mail_volume_col"]].sum().reset_index()
        daily_mail.columns = ['date', 'mail_type', 'volume']
        
        # Business days only
        daily_mail = daily_mail[daily_mail['date'].dt.weekday < 5]
        
        # Pivot to wide format
        mail_pivot = daily_mail.pivot(index='date', columns='mail_type', values='volume').fillna(0)
        mail_pivot = mail_pivot.reset_index()
        
        self.mail_data = mail_pivot
        
        safe_print(f"   Clean mail data: {len(mail_pivot)} business days")
        safe_print(f"   Mail types: {len(mail_pivot.columns) - 1}")
        
        return mail_pivot
        
    except Exception as e:
        safe_print(f"   ERROR loading mail data: {e}")
        raise
        
def load_economic_data(self):
    """Load both economic data files"""
    safe_print("\n" + "=" * 80)
    safe_print("STEP 1C: LOADING ECONOMIC DATA")
    safe_print("=" * 80)
    
    economic_dfs = []
    
    # Load first economic file
    if CONFIG["economic_data_file_1"]:
        try:
            df1 = pd.read_csv(CONFIG["economic_data_file_1"])
            df1['date'] = pd.to_datetime(df1['Date'] if 'Date' in df1.columns else df1.iloc[:, 0])
            df1 = df1.drop(columns=['Date'] if 'Date' in df1.columns else df1.columns[0])
            economic_dfs.append(df1)
            self.economic_data_1 = df1
            safe_print(f"   Loaded economic file 1: {len(df1)} rows, {len(df1.columns)-1} indicators")
        except Exception as e:
            safe_print(f"   WARNING: Could not load economic file 1: {e}")
            
    # Load second economic file
    if CONFIG["economic_data_file_2"]:
        try:
            df2 = pd.read_csv(CONFIG["economic_data_file_2"])
            df2['date'] = pd.to_datetime(df2['Date'] if 'Date' in df2.columns else df2.iloc[:, 0])
            df2 = df2.drop(columns=['Date'] if 'Date' in df2.columns else df2.columns[0])
            economic_dfs.append(df2)
            self.economic_data_2 = df2
            safe_print(f"   Loaded economic file 2: {len(df2)} rows, {len(df2.columns)-1} indicators")
        except Exception as e:
            safe_print(f"   WARNING: Could not load economic file 2: {e}")
            
    return economic_dfs
    
def merge_all_data(self):
    """Merge call, mail, and economic data"""
    safe_print("\n" + "=" * 80)
    safe_print("STEP 1D: MERGING ALL DATA")
    safe_print("=" * 80)
    
    # Start with call and mail merge
    merged = pd.merge(self.call_data, self.mail_data, on='date', how='inner')
    safe_print(f"   Call + Mail merge: {len(merged)} days")
    
    # Add economic data
    if self.economic_data_1 is not None:
        merged = pd.merge(merged, self.economic_data_1, on='date', how='left')
        safe_print(f"   Added economic data 1")
        
    if self.economic_data_2 is not None:
        merged = pd.merge(merged, self.economic_data_2, on='date', how='left')
        safe_print(f"   Added economic data 2")
        
    # Forward fill economic data for non-trading days
    economic_cols = [col for col in merged.columns if col not in self.call_data.columns and col not in self.mail_data.columns]
    if economic_cols:
        merged[economic_cols] = merged[economic_cols].fillna(method='ffill')
        merged = merged.dropna(subset=economic_cols)
        
    merged = merged.sort_values('date').reset_index(drop=True)
    self.merged_data = merged
    
    safe_print(f"   Final merged dataset: {len(merged)} days")
    safe_print(f"   Total features: {len(merged.columns)}")
    
    return merged
```

# ============================================================================

# STEP 2: ENHANCED FEATURE ENGINEERING

# ============================================================================

class FeatureEngineer:
def **init**(self, merged_data):
self.data = merged_data
self.feature_names = []
self.mail_columns = []
self.economic_columns = []

```
def identify_columns(self):
    """Identify mail and economic columns"""
    # All columns except date and call_volume
    all_feature_cols = [col for col in self.data.columns if col not in ['date', 'call_volume']]
    
    # Separate mail and economic columns based on typical patterns
    for col in all_feature_cols:
        # Economic indicators typically have specific names
        if any(indicator in col.lower() for indicator in ['treasury', 'vix', 'oil', 'gold', 'etf', 'index', 'dow', 'nasdaq', 's&p', 'russell']):
            self.economic_columns.append(col)
        else:
            self.mail_columns.append(col)
            
    safe_print(f"   Identified {len(self.mail_columns)} mail types")
    safe_print(f"   Identified {len(self.economic_columns)} economic indicators")
    
def create_features(self, lag_days=1):
    """Create consistent feature set"""
    safe_print("\n" + "=" * 80)
    safe_print("STEP 3: FEATURE ENGINEERING")
    safe_print("=" * 80)
    
    self.identify_columns()
    
    features_list = []
    targets_list = []
    dates_list = []
    
    # Determine lookback period
    max_lookback = max(CONFIG["rolling_windows"] + [3])
    
    # Create features for each valid day
    for i in range(max_lookback, len(self.data) - lag_days):
        feature_row = {}
        current_date = self.data.iloc[i]['date']
        
        # Get top mail types by volume
        mail_volumes = self.data[self.mail_columns].sum()
        top_mail_types = mail_volumes.nlargest(CONFIG["top_mail_types"]).index.tolist()
        
        # Mail features (only top types to reduce dimensionality)
        for mail_type in top_mail_types:
            clean_name = ''.join(c for c in mail_type if c.isalnum())[:20]
            
            # Lag features
            for lag in [1, 2, 3]:
                feature_row[f"mail_{clean_name}_lag{lag}"] = self.data.iloc[i - lag][mail_type]
                
            # Rolling averages
            for window in CONFIG["rolling_windows"]:
                feature_row[f"mail_{clean_name}_avg{window}"] = self.data[mail_type].iloc[i-window+1:i+1].mean()
                
        # Economic features
        for econ_col in self.economic_columns:
            clean_name = ''.join(c for c in econ_col if c.isalnum())[:20]
            
            # Current value
            feature_row[f"econ_{clean_name}"] = self.data.iloc[i][econ_col]
            
            # Change from previous day
            if i > 0:
                prev_val = self.data.iloc[i-1][econ_col]
                if prev_val != 0:
                    feature_row[f"econ_{clean_name}_pct_change"] = (self.data.iloc[i][econ_col] - prev_val) / prev_val * 100
                    
        # Call history features (reduced to prevent overfitting)
        feature_row['calls_lag1'] = self.data.iloc[i - 1]['call_volume']
        feature_row['calls_avg3'] = self.data['call_volume'].iloc[i-3:i].mean()
        feature_row['calls_avg7'] = self.data['call_volume'].iloc[i-7:i].mean()
        
        # Temporal features
        feature_row['weekday'] = current_date.weekday()
        feature_row['month'] = current_date.month
        feature_row['day_of_month'] = current_date.day
        feature_row['quarter'] = (current_date.month - 1) // 3 + 1
        
        # Target
        target = self.data.iloc[i + lag_days]['call_volume']
        
        features_list.append(feature_row)
        targets_list.append(target)
        dates_list.append(current_date)
        
    # Convert to dataframes
    X = pd.DataFrame(features_list).fillna(0)
    y = pd.Series(targets_list, name='call_volume')
    dates = pd.Series(dates_list, name='date')
    
    self.feature_names = X.columns.tolist()
    
    safe_print(f"   Created {len(X.columns)} features from {len(X)} samples")
    safe_print(f"   Feature breakdown:")
    safe_print(f"     - Mail features: {len([f for f in X.columns if f.startswith('mail_')])}")
    safe_print(f"     - Economic features: {len([f for f in X.columns if f.startswith('econ_')])}")
    safe_print(f"     - Call history: {len([f for f in X.columns if f.startswith('calls_')])}")
    safe_print(f"     - Temporal: {len([f for f in X.columns if f in ['weekday', 'month', 'day_of_month', 'quarter']])}")
    
    return X, y, dates, top_mail_types
```

# ============================================================================

# STEP 4: MODEL TRAINING WITH RANGE PREDICTIONS

# ============================================================================

class RangePredictor:
def **init**(self, base_model, confidence=0.8):
self.base_model = base_model
self.confidence = confidence
self.lower_model = None
self.upper_model = None

```
def fit(self, X, y):
    """Train models for point estimate and prediction intervals"""
    # Train base model
    self.base_model.fit(X, y)
    
    # Get residuals from base model
    base_predictions = self.base_model.predict(X)
    residuals = y - base_predictions
    
    # Calculate percentiles for confidence interval
    lower_percentile = (1 - self.confidence) / 2
    upper_percentile = 1 - lower_percentile
    
    # Train models to predict bounds
    lower_bound = base_predictions + np.percentile(residuals, lower_percentile * 100)
    upper_bound = base_predictions + np.percentile(residuals, upper_percentile * 100)
    
    self.lower_model = RandomForestRegressor(n_estimators=50, random_state=42)
    self.upper_model = RandomForestRegressor(n_estimators=50, random_state=42)
    
    self.lower_model.fit(X, lower_bound)
    self.upper_model.fit(X, upper_bound)
    
    return self
    
def predict(self, X):
    """Predict point estimate and range"""
    point_estimate = self.base_model.predict(X)
    lower_bound = self.lower_model.predict(X)
    upper_bound = self.upper_model.predict(X)
    
    # Ensure logical ordering
    lower_bound = np.minimum(lower_bound, point_estimate)
    upper_bound = np.maximum(upper_bound, point_estimate)
    
    return point_estimate, lower_bound, upper_bound
```

class ModelBuilder:
def **init**(self, output_dir):
self.output_dir = output_dir / “models”
self.output_dir.mkdir(exist_ok=True)

```
def train_models(self, X, y, dates):
    """Train models with range prediction capability"""
    safe_print("\n" + "=" * 80)
    safe_print("STEP 4: MODEL TRAINING WITH RANGE PREDICTIONS")
    safe_print("=" * 80)
    
    # Split data
    split_idx = int(len(X) * (1 - CONFIG["test_size"]))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_test = dates.iloc[split_idx:]
    
    safe_print(f"   Train: {len(X_train)} samples")
    safe_print(f"   Test: {len(X_test)} samples")
    
    # Try different models
    models = {
        'linear': LinearRegression(),
        'ridge': Ridge(alpha=1.0, random_state=CONFIG["random_state"]),
        'forest': RandomForestRegressor(
            n_estimators=CONFIG["n_estimators_for_intervals"],
            max_depth=8,
            min_samples_leaf=5,
            random_state=CONFIG["random_state"]
        )
    }
    
    results = {}
    best_score = -float('inf')
    best_name = None
    best_range_model = None
    
    for name, base_model in models.items():
        safe_print(f"\n   Training {name}...")
        
        try:
            # Create range predictor
            range_model = RangePredictor(base_model, confidence=CONFIG["prediction_confidence"])
            range_model.fit(X_train, y_train)
            
            # Get predictions
            point_pred, lower_pred, upper_pred = range_model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, point_pred)
            mae = mean_absolute_error(y_test, point_pred)
            
            # Coverage: how often the true value falls within predicted range
            coverage = np.mean((y_test >= lower_pred) & (y_test <= upper_pred))
            
            # Average interval width
            avg_width = np.mean(upper_pred - lower_pred)
            
            results[name] = {
                'model': range_model,
                'r2': r2,
                'mae': mae,
                'coverage': coverage,
                'avg_width': avg_width,
                'predictions': (point_pred, lower_pred, upper_pred)
            }
            
            safe_print(f"     R-squared: {r2:.3f}")
            safe_print(f"     MAE: {mae:.0f}")
            safe_print(f"     Coverage: {coverage:.1%}")
            safe_print(f"     Avg Range Width: {avg_width:.0f}")
            
            # Select best based on combined score
            score = r2 - 0.1 * (1 - coverage)  # Penalize poor coverage
            if score > best_score:
                best_score = score
                best_name = name
                best_range_model = range_model
                
        except Exception as e:
            safe_print(f"     ERROR: {e}")
            
    if best_range_model:
        safe_print(f"\n   BEST MODEL: {best_name}")
        
        # Save model
        model_info = {
            'model': best_range_model,
            'model_name': best_name,
            'features': X.columns.tolist(),
            'performance': results[best_name],
            'confidence': CONFIG["prediction_confidence"]
        }
        
        joblib.dump(model_info, self.output_dir / "best_model.pkl")
        
        # Create validation plots
        self.create_validation_plots(y_test, results[best_name]['predictions'], 
                                   dates_test, best_name, results)
        
        return best_range_model, best_name, results
    else:
        safe_print("\n   ERROR: No model succeeded")
        return None, None, None
        
def create_validation_plots(self, y_test, predictions, dates_test, best_name, results):
    """Create validation plots with prediction ranges"""
    safe_print("\n   Creating validation plots...")
    
    point_pred, lower_pred, upper_pred = predictions
    
    fig, axes = plt.subplots(2, 2, figsize=CONFIG["figure_size"])
    fig.suptitle(f'Model Validation: {best_name}', fontsize=16)
    
    # Actual vs Predicted with ranges
    axes[0, 0].scatter(y_test, point_pred, alpha=0.6, label='Predictions')
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Calls')
    axes[0, 0].set_ylabel('Predicted Calls')
    axes[0, 0].set_title('Actual vs Predicted')
    axes[0, 0].legend()
    
    # Time series with prediction intervals
    axes[0, 1].plot(dates_test, y_test, 'b-', label='Actual', linewidth=2)
    axes[0, 1].plot(dates_test, point_pred, 'r-', label='Predicted', linewidth=2)
    axes[0, 1].fill_between(dates_test, lower_pred, upper_pred, alpha=0.3, color='red', 
                           label=f'{int(CONFIG["prediction_confidence"]*100)}% Interval')
    axes[0, 1].set_title('Predictions with Confidence Intervals')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Residuals
    residuals = y_test - point_pred
    axes[1, 0].scatter(point_pred, residuals, alpha=0.6)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Predicted Calls')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residual Plot')
    
    # Model comparison
    model_names = list(results.keys())
    r2_scores = [results[name]['r2'] for name in model_names]
    coverages = [results[name]['coverage'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, r2_scores, width, label='R-squared')
    axes[1, 1].bar(x + width/2, coverages, width, label='Coverage')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(model_names)
    axes[1, 1].set_title('Model Performance Comparison')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(self.output_dir / "model_validation.png", dpi=150)
    plt.close()
    
    safe_print("     Validation plots saved")
```

# ============================================================================

# STEP 5: PREDICTION SYSTEM

# ============================================================================

class PredictionSystem:
def **init**(self, model_info, top_mail_types):
self.model = model_info[‘model’]
self.features = model_info[‘features’]
self.top_mail_types = top_mail_types
self.confidence = model_info[‘confidence’]

```
def predict_call_range(self, mail_volumes, economic_data=None, call_history=None):
    """Predict call volume range"""
    try:
        features = {}
        
        # Mail features
        for mail_type in self.top_mail_types:
            clean_name = ''.join(c for c in mail_type if c.isalnum())[:20]
            volume = mail_volumes.get(mail_type, 0)
            
            # Simplified features for prediction
            for lag in [1, 2, 3]:
                features[f"mail_{clean_name}_lag{lag}"] = volume
            for window in CONFIG["rolling_windows"]:
                features[f"mail_{clean_name}_avg{window}"] = volume
                
        # Economic features
        if economic_data:
            for indicator, value in economic_data.items():
                clean_name = ''.join(c for c in indicator if c.isalnum())[:20]
                features[f"econ_{clean_name}"] = value
                features[f"econ_{clean_name}_pct_change"] = 0  # No change for single prediction
                
        # Call history
        if call_history:
            features['calls_lag1'] = call_history.get('yesterday', 10000)
            features['calls_avg3'] = call_history.get('avg_3day', 10000)
            features['calls_avg7'] = call_history.get('avg_7day', 10000)
        else:
            # Use defaults
            features['calls_lag1'] = 10000
            features['calls_avg3'] = 10000
            features['calls_avg7'] = 10000
            
        # Temporal features
        now = datetime.now()
        features['weekday'] = now.weekday()
        features['month'] = now.month
        features['day_of_month'] = now.day
        features['quarter'] = (now.month - 1) // 3 + 1
        
        # Create feature vector
        feature_vector = pd.DataFrame([features])
        
        # Ensure all required features are present
        for feat in self.features:
            if feat not in feature_vector.columns:
                feature_vector[feat] = 0
                
        feature_vector = feature_vector[self.features]
        
        # Get predictions
        point, lower, upper = self.model.predict(feature_vector)
        
        return {
            'status': 'success',
            'predicted_calls': int(point[0]),
            'lower_bound': int(max(0, lower[0])),
            'upper_bound': int(upper[0]),
            'confidence': f"{int(self.confidence * 100)}%"
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }
```

# ============================================================================

# MAIN ORCHESTRATOR

# ============================================================================

def main():
safe_print(“MAIL-TO-CALLS PREDICTION SYSTEM V2”)
safe_print(”=” * 80)

```
try:
    # Load data
    data_manager = DataManager()
    data_manager.load_call_data()
    data_manager.load_mail_data()
    data_manager.load_economic_data()
    merged_data = data_manager.merge_all_data()
    
    # Feature engineering
    feature_engineer = FeatureEngineer(merged_data)
    X, y, dates, top_mail_types = feature_engineer.create_features(lag_days=1)
    
    # Model training
    model_builder = ModelBuilder(data_manager.output_dir)
    best_model, best_name, results = model_builder.train_models(X, y, dates)
    
    if not best_model:
        safe_print("\n   MODELING FAILED")
        return {'success': False}
        
    # Save configuration
    config_info = {
        'features': X.columns.tolist(),
        'top_mail_types': top_mail_types,
        'economic_columns': feature_engineer.economic_columns,
        'mail_columns': feature_engineer.mail_columns
    }
    
    joblib.dump(config_info, data_manager.output_dir / "config_info.pkl")
    
    # Test prediction system
    safe_print("\n" + "=" * 80)
    safe_print("STEP 5: TESTING PREDICTION SYSTEM")
    safe_print("=" * 80)
    
    model_info = joblib.load(data_manager.output_dir / "models" / "best_model.pkl")
    prediction_system = PredictionSystem(model_info, top_mail_types)
    
    # Example prediction
    test_mail = {mail_type: 10000 for mail_type in top_mail_types[:5]}
    test_econ = {col: merged_data[col].mean() for col in feature_engineer.economic_columns[:5]}
    
    result = prediction_system.predict_call_range(test_mail, test_econ)
    
    if result['status'] == 'success':
        safe_print("\n   PREDICTION TEST:")
        safe_print(f"   Predicted calls: {result['predicted_calls']:,}")
        safe_print(f"   Range: {result['lower_bound']:,} - {result['upper_bound']:,}")
        safe_print(f"   Confidence: {result['confidence']}")
    else:
        safe_print(f"\n   PREDICTION ERROR: {result['message']}")
        
    safe_print("\n" + "=" * 80)
    safe_print("SYSTEM READY!")
    safe_print("=" * 80)
    
    return {'success': True}
    
except Exception as e:
    safe_print(f"\n   SYSTEM ERROR: {str(e)}")
    import traceback
    traceback.print_exc()
    return {'success': False, 'error': str(e)}
```

if **name** == “**main**”:
result = main()
if result.get(‘success’):
safe_print(”\n>> MAIL-TO-CALLS PREDICTION SYSTEM V2 READY FOR PRODUCTION!”)
else:
safe_print(”\n>> SYSTEM FAILED”)