#!/usr/bin/env python
“””
RIGOROUS MODEL TESTING SCRIPT V2

- Consistent feature engineering with training
- Proper time series validation
- Range prediction evaluation
- ASCII-only output
  “””

import warnings
warnings.filterwarnings(‘ignore’)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import logging
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# ============================================================================

# CONFIGURATION

# ============================================================================

CONFIG = {
“model_dir”: “mail_call_prediction_system_v2”,
“output_dir”: “mail_call_prediction_system_v2/test_results”,

```
# Data files
"call_file": "ACDMail.csv",
"mail_file": "mail.csv",
"economic_data_file_1": "expanded_economic_data.csv",
"economic_data_file_2": "",
"holidays_file": "us_holidays.csv",

# Test settings
"cv_splits": 5,
"test_on_recent_data": True,  # Test on most recent 20% of data

# Same as training config
"rolling_windows": [3, 7],
"top_mail_types": 8,
```

}

# ============================================================================

# SETUP

# ============================================================================

def setup_logging(output_dir):
“”“Setup ASCII-safe logging”””
log_dir = Path(output_dir)
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f”test_log_{datetime.now().strftime(’%Y%m%d_%H%M%S’)}.log”

```
# Custom formatter to handle ASCII
class ASCIIFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        return msg.encode('ascii', 'replace').decode('ascii')

# Setup handlers
file_handler = logging.FileHandler(log_file)
console_handler = logging.StreamHandler()

formatter = ASCIIFormatter('%(asctime)s [%(levelname)s] - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logging.info(f"Test results will be saved to: {log_dir.resolve()}")
return logger
```

# ============================================================================

# DATA LOADING

# ============================================================================

def load_all_data():
“”“Load and merge all data files”””
logging.info(“Loading all data files…”)

```
try:
    # Load call data
    call_df = pd.read_csv(CONFIG["call_file"])
    call_df = call_df[['Date', 'ACDCalls']].rename(columns={'Date': 'date', 'ACDCalls': 'call_volume'})
    call_df['date'] = pd.to_datetime(call_df['date'])
    call_df = call_df[call_df['call_volume'] > 5]
    
    # Business days only
    call_df = call_df[call_df['date'].dt.weekday < 5]
    
    # Remove holidays if file exists
    try:
        holidays_df = pd.read_csv(CONFIG["holidays_file"])
        holiday_dates = set(holidays_df['holiday_date'])
        call_df = call_df[~call_df['date'].dt.strftime('%Y-%m-%d').isin(holiday_dates)]
    except:
        logging.warning("Could not remove holidays")
        
    # Load mail data
    mail_df = pd.read_csv(CONFIG["mail_file"], low_memory=False)
    mail_df['mail_date'] = pd.to_datetime(mail_df['mail_date'])
    
    # Aggregate by date and type
    mail_pivot = mail_df.pivot_table(
        index='mail_date', 
        columns='mail_type', 
        values='mail_volume', 
        aggfunc='sum'
    ).fillna(0)
    mail_pivot.index.name = 'date'
    mail_pivot = mail_pivot.reset_index()
    
    # Merge call and mail
    merged = pd.merge(call_df, mail_pivot, on='date', how='inner')
    
    # Load economic data
    if CONFIG["economic_data_file_1"]:
        try:
            econ1 = pd.read_csv(CONFIG["economic_data_file_1"])
            econ1['date'] = pd.to_datetime(econ1['Date'] if 'Date' in econ1.columns else econ1.iloc[:, 0])
            econ1 = econ1.drop(columns=['Date'] if 'Date' in econ1.columns else econ1.columns[0])
            merged = pd.merge(merged, econ1, on='date', how='left')
        except Exception as e:
            logging.warning(f"Could not load economic file 1: {e}")
            
    if CONFIG["economic_data_file_2"]:
        try:
            econ2 = pd.read_csv(CONFIG["economic_data_file_2"])
            econ2['date'] = pd.to_datetime(econ2['Date'] if 'Date' in econ2.columns else econ2.iloc[:, 0])
            econ2 = econ2.drop(columns=['Date'] if 'Date' in econ2.columns else econ2.columns[0])
            merged = pd.merge(merged, econ2, on='date', how='left')
        except Exception as e:
            logging.warning(f"Could not load economic file 2: {e}")
            
    # Forward fill economic data
    economic_cols = [col for col in merged.columns if any(ind in col.lower() for ind in ['treasury', 'vix', 'oil', 'gold', 'etf', 'index'])]
    if economic_cols:
        merged[economic_cols] = merged[economic_cols].fillna(method='ffill')
        merged = merged.dropna(subset=economic_cols)
        
    merged = merged.sort_values('date').reset_index(drop=True)
    
    logging.info(f"Loaded {len(merged)} days of merged data")
    logging.info(f"Date range: {merged['date'].min()} to {merged['date'].max()}")
    
    return merged
    
except Exception as e:
    logging.error(f"Error loading data: {e}")
    raise
```

# ============================================================================

# FEATURE RECREATION

# ============================================================================

def recreate_features(data, config_info):
“”“Recreate features exactly as in training”””
logging.info(“Recreating features…”)

```
features_list = []
targets_list = []
dates_list = []

# Get saved configuration
top_mail_types = config_info['top_mail_types']

# Identify columns
all_cols = [col for col in data.columns if col not in ['date', 'call_volume']]
economic_cols = [col for col in all_cols if any(ind in col.lower() for ind in ['treasury', 'vix', 'oil', 'gold', 'etf', 'index', 'dow', 'nasdaq'])]

max_lookback = max(CONFIG["rolling_windows"] + [3])
lag_days = 1

for i in range(max_lookback, len(data) - lag_days):
    feature_row = {}
    current_date = data.iloc[i]['date']
    
    # Mail features
    for mail_type in top_mail_types:
        if mail_type in data.columns:
            clean_name = ''.join(c for c in mail_type if c.isalnum())[:20]
            
            for lag in [1, 2, 3]:
                feature_row[f"mail_{clean_name}_lag{lag}"] = data.iloc[i - lag][mail_type]
                
            for window in CONFIG["rolling_windows"]:
                feature_row[f"mail_{clean_name}_avg{window}"] = data[mail_type].iloc[i-window+1:i+1].mean()
                
    # Economic features
    for econ_col in economic_cols:
        if econ_col in data.columns:
            clean_name = ''.join(c for c in econ_col if c.isalnum())[:20]
            feature_row[f"econ_{clean_name}"] = data.iloc[i][econ_col]
            
            if i > 0:
                prev_val = data.iloc[i-1][econ_col]
                if prev_val != 0:
                    feature_row[f"econ_{clean_name}_pct_change"] = (data.iloc[i][econ_col] - prev_val) / prev_val * 100
                    
    # Call history
    feature_row['calls_lag1'] = data.iloc[i - 1]['call_volume']
    feature_row['calls_avg3'] = data['call_volume'].iloc[i-3:i].mean()
    feature_row['calls_avg7'] = data['call_volume'].iloc[i-7:i].mean()
    
    # Temporal
    feature_row['weekday'] = current_date.weekday()
    feature_row['month'] = current_date.month
    feature_row['day_of_month'] = current_date.day
    feature_row['quarter'] = (current_date.month - 1) // 3 + 1
    
    # Target
    target = data.iloc[i + lag_days]['call_volume']
    
    features_list.append(feature_row)
    targets_list.append(target)
    dates_list.append(current_date)
    
X = pd.DataFrame(features_list).fillna(0)
y = pd.Series(targets_list)
dates = pd.Series(dates_list)

logging.info(f"Created {len(X)} samples with {len(X.columns)} features")

return X, y, dates
```

# ============================================================================

# TESTING FUNCTIONS

# ============================================================================

def evaluate_range_predictions(y_true, predictions):
“”“Evaluate range prediction performance”””
point_pred, lower_pred, upper_pred = predictions

```
# Point prediction metrics
mae = mean_absolute_error(y_true, point_pred)
r2 = r2_score(y_true, point_pred)
mape = np.mean(np.abs((y_true - point_pred) / y_true)) * 100

# Range prediction metrics
coverage = np.mean((y_true >= lower_pred) & (y_true <= upper_pred))
avg_width = np.mean(upper_pred - lower_pred)
width_percentage = avg_width / np.mean(y_true) * 100

return {
    'mae': mae,
    'r2': r2,
    'mape': mape,
    'coverage': coverage,
    'avg_width': avg_width,
    'width_percentage': width_percentage
}
```

def test_on_holdout(model, X, y, dates, features_expected):
“”“Test on most recent data”””
logging.info(”\n— Testing on Recent Holdout Data —”)

```
# Use most recent 20% as test
split_idx = int(len(X) * 0.8)
X_test = X.iloc[split_idx:]
y_test = y.iloc[split_idx:]
dates_test = dates.iloc[split_idx:]

# Align features
for feat in features_expected:
    if feat not in X_test.columns:
        X_test[feat] = 0
X_test = X_test[features_expected]

# Get predictions
predictions = model.predict(X_test)

# Evaluate
metrics = evaluate_range_predictions(y_test, predictions)

logging.info(f"Holdout Test Results ({len(X_test)} samples):")
logging.info(f"  R-squared: {metrics['r2']:.3f}")
logging.info(f"  MAE: {metrics['mae']:,.0f}")
logging.info(f"  MAPE: {metrics['mape']:.1f}%")
logging.info(f"  Coverage: {metrics['coverage']:.1%}")
logging.info(f"  Avg Range Width: {metrics['avg_width']:,.0f} ({metrics['width_percentage']:.1f}%)")

return metrics, predictions, y_test, dates_test
```

def time_series_cv(model, X, y, features_expected):
“”“Perform time series cross-validation”””
logging.info(”\n— Time Series Cross-Validation —”)

```
tscv = TimeSeriesSplit(n_splits=CONFIG['cv_splits'])
cv_results = []

for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Align features
    for feat in features_expected:
        if feat not in X_test.columns:
            X_test[feat] = 0
    X_test = X_test[features_expected]
    
    # Train a new model instance
    from sklearn.base import clone
    fold_model = clone(model.base_model)
    
    # Create range predictor
    from sklearn.ensemble import RandomForestRegressor
    fold_range_model = type(model)(fold_model, confidence=model.confidence)
    fold_range_model.fit(X_train[features_expected], y_train)
    
    # Predict
    predictions = fold_range_model.predict(X_test)
    
    # Evaluate
    metrics = evaluate_range_predictions(y_test, predictions)
    cv_results.append(metrics)
    
    logging.info(f"  Fold {fold+1}: R2={metrics['r2']:.3f}, MAE={metrics['mae']:.0f}, Coverage={metrics['coverage']:.1%}")
    
# Average results
avg_metrics = {}
for key in cv_results[0].keys():
    avg_metrics[key] = np.mean([r[key] for r in cv_results])
    avg_metrics[f"{key}_std"] = np.std([r[key] for r in cv_results])
    
logging.info("\nCV Average Results:")
logging.info(f"  R-squared: {avg_metrics['r2']:.3f} (+/- {avg_metrics['r2_std']:.3f})")
logging.info(f"  MAE: {avg_metrics['mae']:,.0f} (+/- {avg_metrics['mae_std']:,.0f})")
logging.info(f"  Coverage: {avg_metrics['coverage']:.1%} (+/- {avg_metrics['coverage_std']:.1%})")

return avg_metrics
```

def analyze_feature_importance(model, features):
“”“Analyze feature importance”””
logging.info(”\n— Feature Importance Analysis —”)

```
if hasattr(model.base_model, 'feature_importances_'):
    importances = pd.Series(
        model.base_model.feature_importances_, 
        index=features
    ).sort_values(ascending=False)
    
    logging.info("Top 15 Most Important Features:")
    for i, (feat, imp) in enumerate(importances.head(15).items()):
        logging.info(f"  {i+1:2d}. {feat:<40} {imp:.4f}")
        
    # Feature type breakdown
    mail_imp = importances[[f for f in importances.index if f.startswith('mail_')]].sum()
    econ_imp = importances[[f for f in importances.index if f.startswith('econ_')]].sum()
    call_imp = importances[[f for f in importances.index if f.startswith('calls_')]].sum()
    temp_imp = importances[[f for f in importances.index if f in ['weekday', 'month', 'day_of_month', 'quarter']]].sum()
    
    logging.info("\nImportance by Feature Type:")
    logging.info(f"  Mail features: {mail_imp:.3f} ({mail_imp/importances.sum():.1%})")
    logging.info(f"  Economic features: {econ_imp:.3f} ({econ_imp/importances.sum():.1%})")
    logging.info(f"  Call history: {call_imp:.3f} ({call_imp/importances.sum():.1%})")
    logging.info(f"  Temporal: {temp_imp:.3f} ({temp_imp/importances.sum():.1%})")
    
    return importances
else:
    logging.info("  Model does not support feature importances")
    return None
```

def create_diagnostic_plots(output_dir, y_test, predictions, dates_test, importances=None):
“”“Create diagnostic plots”””
logging.info(”\n— Creating Diagnostic Plots —”)

```
point_pred, lower_pred, upper_pred = predictions

# Figure 1: Time series with ranges
plt.figure(figsize=(15, 6))
plt.plot(dates_test, y_test, 'b-', label='Actual', linewidth=2)
plt.plot(dates_test, point_pred, 'r--', label='Predicted', linewidth=2)
plt.fill_between(dates_test, lower_pred, upper_pred, alpha=0.3, color='red', label='Prediction Range')
plt.xlabel('Date')
plt.ylabel('Call Volume')
plt.title('Model Performance: Actual vs Predicted with Confidence Intervals')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_dir / 'time_series_validation.png', dpi=150)
plt.close()

# Figure 2: Residual analysis
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

residuals = y_test - point_pred

# Residuals vs predicted
axes[0, 0].scatter(point_pred, residuals, alpha=0.6)
axes[0, 0].axhline(y=0, color='r', linestyle='--')
axes[0, 0].set_xlabel('Predicted Calls')
axes[0, 0].set_ylabel('Residuals')
axes[0, 0].set_title('Residuals vs Predicted')

# Residual histogram
axes[0, 1].hist(residuals, bins=30, edgecolor='black')
axes[0, 1].set_xlabel('Residuals')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('Residual Distribution')

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot')

# Coverage by prediction magnitude
in_range = (y_test >= lower_pred) & (y_test <= upper_pred)
bins = pd.qcut(point_pred, q=5, duplicates='drop')
coverage_by_bin = pd.Series(in_range).groupby(bins).mean()

axes[1, 1].bar(range(len(coverage_by_bin)), coverage_by_bin.values)
axes[1, 1].set_xlabel('Prediction Quintile')
axes[1, 1].set_ylabel('Coverage Rate')
axes[1, 1].set_title('Coverage by Prediction Magnitude')
axes[1, 1].set_ylim([0, 1])

plt.tight_layout()
plt.savefig(output_dir / 'residual_diagnostics.png', dpi=150)
plt.close()

# Figure 3: Feature importance (if available)
if importances is not None:
    plt.figure(figsize=(10, 8))
    importances.head(20).sort_values().plot(kind='barh')
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig(output_dir / 'feature_importance.png', dpi=150)
    plt.close()
    
logging.info("  Diagnostic plots saved")
```

# ============================================================================

# MAIN TEST FUNCTION

# ============================================================================

def run_comprehensive_test():
“”“Run comprehensive model testing”””
output_dir = Path(CONFIG[“output_dir”])
logger = setup_logging(output_dir)

```
logging.info("Starting Comprehensive Model Testing...")
logging.info("=" * 80)

try:
    # Load model and config
    model_path = Path(CONFIG["model_dir"]) / "models" / "best_model.pkl"
    config_path = Path(CONFIG["model_dir"]) / "config_info.pkl"
    
    if not model_path.exists():
        logging.error(f"Model not found at {model_path}")
        return
        
    model_info = joblib.load(model_path)
    config_info = joblib.load(config_path)
    
    model = model_info['model']
    model_name = model_info['model_name']
    features_expected = model_info['features']
    
    logging.info(f"Loaded model: {model_name}")
    logging.info(f"Expected features: {len(features_expected)}")
    
    # Load and prepare data
    data = load_all_data()
    X, y, dates = recreate_features(data, config_info)
    
    # Align features with model expectations
    missing_features = set(features_expected) - set(X.columns)
    if missing_features:
        logging.warning(f"Adding {len(missing_features)} missing features with value 0")
        for feat in missing_features:
            X[feat] = 0
            
    # Test 1: Holdout test
    holdout_metrics, predictions, y_test, dates_test = test_on_holdout(
        model, X, y, dates, features_expected
    )
    
    # Test 2: Time series CV
    cv_metrics = time_series_cv(model, X, y, features_expected)
    
    # Test 3: Feature importance
    importances = analyze_feature_importance(model, features_expected)
    
    # Create diagnostic plots
    create_diagnostic_plots(output_dir, y_test, predictions, dates_test, importances)
    
    # Generate report
    report = f"""
```

# COMPREHENSIVE MODEL TEST REPORT

Model: {model_name}
Date: {datetime.now().strftime(’%Y-%m-%d %H:%M:%S’)}

## HOLDOUT TEST RESULTS (Recent 20% of data)

R-squared: {holdout_metrics[‘r2’]:.3f}
MAE: {holdout_metrics[‘mae’]:,.0f}
MAPE: {holdout_metrics[‘mape’]:.1f}%
Coverage: {holdout_metrics[‘coverage’]:.1%}
Average Range Width: {holdout_metrics[‘avg_width’]:,.0f} ({holdout_metrics[‘width_percentage’]:.1f}% of mean)

## TIME SERIES CROSS-VALIDATION ({CONFIG[‘cv_splits’]} folds)

R-squared: {cv_metrics[‘r2’]:.3f} (+/- {cv_metrics[‘r2_std’]:.3f})
MAE: {cv_metrics[‘mae’]:,.0f} (+/- {cv_metrics[‘mae_std’]:,.0f})
Coverage: {cv_metrics[‘coverage’]:.1%} (+/- {cv_metrics[‘coverage_std’]:.1%})

## CONCLUSION

The model shows {“good” if holdout_metrics[‘r2’] > 0.5 else “moderate” if holdout_metrics[‘r2’] > 0.3 else “poor”} predictive performance.
The prediction intervals achieve {holdout_metrics[‘coverage’]:.0%} coverage, {“meeting” if holdout_metrics[‘coverage’] >= 0.75 else “below”} the target.
“””

```
    # Save report
    with open(output_dir / 'test_report.txt', 'w') as f:
        f.write(report)
        
    logging.info("\n" + report)
    logging.info("=" * 80)
    logging.info("Testing complete. Results saved to: " + str(output_dir))
    
except Exception as e:
    logging.error(f"Testing failed: {str(e)}")
    import traceback
    traceback.print_exc()
```

if **name** == “**main**”:
run_comprehensive_test()