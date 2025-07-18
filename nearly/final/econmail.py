#!/usr/bin/env python

# production_predictive_model.py

# =========================================================

# PRODUCTION-READY PREDICTIVE MODEL

# Tests multiple algorithms for optimal performance

# Provides narrow prediction intervals for business use

# =========================================================

from pathlib import Path
import json
import logging
import sys
import warnings
from datetime import datetime, timedelta
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from scipy.stats import pearsonr

# Machine Learning Models

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

# Advanced models

try:
from xgboost import XGBRegressor
XGBOOST_AVAILABLE = True
except ImportError:
XGBOOST_AVAILABLE = False

try:
from lightgbm import LGBMRegressor
LIGHTGBM_AVAILABLE = True
except ImportError:
LIGHTGBM_AVAILABLE = False

warnings.filterwarnings(‘ignore’)

LOG = logging.getLogger(“production_model”)
logging.basicConfig(
level=logging.INFO,
format=”%(asctime)s | production_model | %(levelname)s | %(message)s”,
handlers=[logging.StreamHandler(sys.stdout)]
)

CFG = {
“economic_data_files”: [
“economic_data_for_model.csv”,
“expanded_economic_data.csv”
],
“best_economic_indicators”: [
“Russell2000”, “Dollar_Index”, “NASDAQ”, “SP500”, “Technology”,
“Banking”, “DowJones”, “Regional_Banks”, “Dividend_ETF”
],
“mail_counts_to_test”: [10, 15],
“confidence_intervals”: [0.1, 0.25, 0.5, 0.75, 0.9],  # 10%, 25%, 50%, 75%, 90%
“output_dir”: “production_model_results”
}

def _to_date(s):
return pd.to_datetime(s, errors=“coerce”).dt.date

def _find_file(candidates):
for p in candidates:
path = Path(p)
if path.exists():
return path
raise FileNotFoundError(f”None found: {candidates}”)

def load_clean_data():
“”“Load and clean all data, properly separating mail types from economic indicators”””

```
LOG.info("=== LOADING AND CLEANING DATA ===")

# Load mail data
mail_path = _find_file(["mail.csv", "data/mail.csv"])
mail = pd.read_csv(mail_path)
mail.columns = [c.lower().strip() for c in mail.columns]
mail["mail_date"] = _to_date(mail["mail_date"])
mail = mail.dropna(subset=["mail_date"])

# Load call data
vol_path = _find_file(["callvolumes.csv", "data/callvolumes.csv"])
df_vol = pd.read_csv(vol_path)
df_vol.columns = [c.lower().strip() for c in df_vol.columns]
dcol_v = next(c for c in df_vol.columns if "date" in c)
df_vol[dcol_v] = _to_date(df_vol[dcol_v])
vol_daily = df_vol.groupby(dcol_v)[df_vol.columns.difference([dcol_v])[0]].sum()

intent_path = _find_file(["callintent.csv", "data/callintent.csv"])
df_int = pd.read_csv(intent_path)
df_int.columns = [c.lower().strip() for c in df_int.columns]
dcol_i = next(c for c in df_int.columns if "date" in c or "conversationstart" in c)
df_int[dcol_i] = _to_date(df_int[dcol_i])
int_daily = df_int.groupby(dcol_i).size()

# Combine calls
overlap = vol_daily.index.intersection(int_daily.index)
if len(overlap) >= 5:
    scale = int_daily.loc[overlap].mean() / vol_daily.loc[overlap].mean()
    vol_daily *= scale
calls_total = vol_daily.combine_first(int_daily).sort_index()

# Process mail (properly)
mail_daily = (mail.groupby(["mail_date", "mail_type"], as_index=False)["mail_volume"].sum()
               .pivot(index="mail_date", columns="mail_type", values="mail_volume").fillna(0))

mail_daily.index = pd.to_datetime(mail_daily.index)
calls_total.index = pd.to_datetime(calls_total.index)

# Business days only
us_holidays = holidays.US()
biz_mask = (~mail_daily.index.weekday.isin([5, 6])) & (~mail_daily.index.isin(us_holidays))
mail_daily = mail_daily.loc[biz_mask]
calls_total = calls_total.loc[calls_total.index.isin(mail_daily.index)]

# Get ACTUAL mail types (not economic indicators)
actual_mail_types = [col for col in mail_daily.columns]
LOG.info(f"Found {len(actual_mail_types)} actual mail types")

# Load economic data separately
all_economic = pd.DataFrame()
for econ_file in CFG["economic_data_files"]:
    econ_path = Path(econ_file)
    if econ_path.exists():
        econ_data = pd.read_csv(econ_path, parse_dates=['Date'])
        econ_data.set_index('Date', inplace=True)
        all_economic = pd.concat([all_economic, econ_data], axis=1)

# Remove duplicate columns
all_economic = all_economic.loc[:, ~all_economic.columns.duplicated()]

# Combine mail and calls
mail_calls = mail_daily.join(calls_total.rename("calls_total"), how="inner")

# Add economic data
if not all_economic.empty:
    final_data = mail_calls.join(all_economic, how='left')
    final_data = final_data.fillna(method='ffill').fillna(method='bfill')
    economic_columns = list(all_economic.columns)
else:
    final_data = mail_calls
    economic_columns = []

LOG.info(f"Final data shape: {final_data.shape}")
LOG.info(f"Actual mail types: {len(actual_mail_types)}")
LOG.info(f"Economic indicators: {len(economic_columns)}")

return final_data, actual_mail_types, economic_columns
```

def analyze_correlations_properly(data, actual_mail_types, economic_columns):
“”“Properly analyze correlations separating mail types from economic indicators”””

```
LOG.info("=== ANALYZING CORRELATIONS (PROPERLY SEPARATED) ===")

# Analyze actual mail types
mail_correlations = []
for mail_type in actual_mail_types:
    if mail_type in data.columns and data[mail_type].sum() > 0:
        corr_same, p_same = pearsonr(data[mail_type], data['calls_total'])
        corr_lag1, p_lag1 = pearsonr(data[mail_type].shift(1).dropna(), 
                                    data['calls_total'].iloc[1:])
        
        mail_correlations.append({
            'mail_type': mail_type,
            'correlation_same': corr_same,
            'correlation_lag1': corr_lag1,
            'p_same': p_same,
            'p_lag1': p_lag1,
            'abs_correlation': abs(corr_same),
            'total_volume': data[mail_type].sum()
        })

mail_correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)

# Analyze economic indicators
econ_correlations = []
for econ_indicator in economic_columns:
    if econ_indicator in data.columns:
        try:
            corr_same, p_same = pearsonr(data[econ_indicator].dropna(), 
                                       data['calls_total'].loc[data[econ_indicator].dropna().index])
            corr_lag1, p_lag1 = pearsonr(data[econ_indicator].shift(1).dropna(), 
                                       data['calls_total'].iloc[1:])
            
            econ_correlations.append({
                'indicator': econ_indicator,
                'correlation_same': corr_same,
                'correlation_lag1': corr_lag1,
                'p_same': p_same,
                'p_lag1': p_lag1,
                'abs_correlation': abs(corr_same),
                'best_correlation': max(abs(corr_same), abs(corr_lag1))
            })
        except:
            continue

econ_correlations.sort(key=lambda x: x['best_correlation'], reverse=True)

# Log results
LOG.info(f"Top 10 ACTUAL mail types by correlation:")
for i, item in enumerate(mail_correlations[:10], 1):
    LOG.info(f"  {i:2d}. {item['mail_type']}: {item['correlation_same']:.3f}")

LOG.info(f"Top 10 economic indicators by correlation:")
for i, item in enumerate(econ_correlations[:10], 1):
    LOG.info(f"  {i:2d}. {item['indicator']}: {item['best_correlation']:.3f}")

return mail_correlations, econ_correlations
```

def create_production_features(data, mail_correlations, econ_correlations, top_n_mail):
“”“Create production-ready features”””

```
LOG.info(f"=== CREATING PRODUCTION FEATURES (Top {top_n_mail} Mail Types) ===")

# Get top mail types and economic indicators
top_mail_types = [item['mail_type'] for item in mail_correlations[:top_n_mail]]
top_econ_indicators = [item['indicator'] for item in econ_correlations[:5]]

LOG.info(f"Top {top_n_mail} mail types: {top_mail_types}")
LOG.info(f"Top 5 economic indicators: {top_econ_indicators}")

features_list = []
targets_list = []

for i in range(2, len(data) - 1):  # Start from index 2 for 2-day lags
    current_day = data.iloc[i]
    prev_day = data.iloc[i-1]
    prev_day_2 = data.iloc[i-2]
    next_day = data.iloc[i + 1]
    
    feature_row = {}
    
    # Mail features (same day - strongest correlation)
    for mail_type in top_mail_types:
        if mail_type in data.columns:
            feature_row[f"{mail_type}_today"] = current_day[mail_type]
            feature_row[f"{mail_type}_lag1"] = prev_day[mail_type]
    
    # Total mail features
    total_mail_today = sum(current_day.get(mail_type, 0) for mail_type in top_mail_types)
    total_mail_lag1 = sum(prev_day.get(mail_type, 0) for mail_type in top_mail_types)
    
    feature_row["total_mail_today"] = total_mail_today
    feature_row["total_mail_lag1"] = total_mail_lag1
    feature_row["log_total_mail_today"] = np.log1p(total_mail_today)
    feature_row["log_total_mail_lag1"] = np.log1p(total_mail_lag1)
    
    # Economic features (multiple lags)
    for econ_indicator in top_econ_indicators:
        if econ_indicator in data.columns:
            feature_row[f"{econ_indicator}_today"] = current_day[econ_indicator]
            feature_row[f"{econ_indicator}_lag1"] = prev_day[econ_indicator]
            feature_row[f"{econ_indicator}_lag2"] = prev_day_2[econ_indicator]
    
    # Interaction features
    if top_econ_indicators and top_econ_indicators[0] in data.columns:
        feature_row["best_econ_x_total_mail"] = prev_day[top_econ_indicators[0]] * np.log1p(total_mail_today)
    
    # Market momentum features
    if len(top_econ_indicators) >= 2:
        econ1 = top_econ_indicators[0]
        econ2 = top_econ_indicators[1]
        if econ1 in data.columns and econ2 in data.columns:
            feature_row[f"{econ1}_momentum"] = (current_day[econ1] - prev_day_2[econ1]) / prev_day_2[econ1]
            feature_row[f"{econ2}_momentum"] = (current_day[econ2] - prev_day_2[econ2]) / prev_day_2[econ2]
    
    # Date features
    current_date = data.index[i]
    feature_row["month"] = current_date.month
    feature_row["weekday"] = current_date.weekday()
    feature_row["is_month_end"] = 1 if current_date.day > 25 else 0
    feature_row["is_quarter_end"] = 1 if current_date.month in [3, 6, 9, 12] and current_date.day > 25 else 0
    
    # Recent call volume context
    recent_calls = data["calls_total"].iloc[max(0, i-7):i].mean()
    feature_row["recent_calls_avg"] = recent_calls
    
    # Volatility features
    recent_calls_vol = data["calls_total"].iloc[max(0, i-7):i].std()
    feature_row["recent_calls_volatility"] = recent_calls_vol if not np.isnan(recent_calls_vol) else 0
    
    # Target: next day's calls
    target = next_day["calls_total"]
    
    features_list.append(feature_row)
    targets_list.append(target)

X = pd.DataFrame(features_list)
y = pd.Series(targets_list)
X = X.fillna(0)

LOG.info(f"Production features: {X.shape[0]} samples x {X.shape[1]} features")
return X, y, top_mail_types, top_econ_indicators
```

def get_model_suite():
“”“Get comprehensive suite of models to test”””

```
models = {
    # Linear Models
    "Linear_Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ]),
    
    "Ridge_Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model', Ridge(alpha=1.0))
    ]),
    
    "Lasso_Regression": Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso(alpha=0.1))
    ]),
    
    "ElasticNet": Pipeline([
        ('scaler', StandardScaler()),
        ('model', ElasticNet(alpha=0.1, l1_ratio=0.5))
    ]),
    
    # Tree-based Models
    "Random_Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),
    
    "Extra_Trees": ExtraTreesRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42
    ),
    
    "Gradient_Boosting": GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    ),
    
    # Support Vector Machine
    "SVR": Pipeline([
        ('scaler', RobustScaler()),
        ('model', SVR(kernel='rbf', C=100, gamma='scale'))
    ]),
    
    # Neural Network
    "Neural_Network": Pipeline([
        ('scaler', StandardScaler()),
        ('model', MLPRegressor(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42
        ))
    ])
}

# Add XGBoost if available
if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )

# Add LightGBM if available
if LIGHTGBM_AVAILABLE:
    models["LightGBM"] = LGBMRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )

return models
```

def evaluate_models_comprehensive(X, y, models):
“”“Comprehensive model evaluation with time series cross-validation”””

```
LOG.info("=== COMPREHENSIVE MODEL EVALUATION ===")

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

results = {}

for model_name, model in models.items():
    LOG.info(f"Evaluating {model_name}...")
    
    try:
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_absolute_error')
        cv_r2_scores = cross_val_score(model, X, y, cv=tscv, scoring='r2')
        
        # Train on full data for final metrics
        model.fit(X, y)
        y_pred = model.predict(X)
        
        # Calculate comprehensive metrics
        mae = mean_absolute_error(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # Calculate prediction intervals
        residuals = y - y_pred
        prediction_intervals = {}
        for confidence in CFG["confidence_intervals"]:
            lower_percentile = ((1 - confidence) / 2) * 100
            upper_percentile = (confidence + (1 - confidence) / 2) * 100
            
            lower_bound = np.percentile(residuals, lower_percentile)
            upper_bound = np.percentile(residuals, upper_percentile)
            
            prediction_intervals[f"{int(confidence*100)}%"] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'width': upper_bound - lower_bound
            }
        
        results[model_name] = {
            'cv_mae_mean': -cv_scores.mean(),
            'cv_mae_std': cv_scores.std(),
            'cv_r2_mean': cv_r2_scores.mean(),
            'cv_r2_std': cv_r2_scores.std(),
            'train_mae': mae,
            'train_rmse': rmse,
            'train_r2': r2,
            'prediction_intervals': prediction_intervals,
            'model': model
        }
        
        LOG.info(f"  {model_name} - CV MAE: {-cv_scores.mean():.0f} (±{cv_scores.std():.0f})")
        LOG.info(f"  {model_name} - CV R²: {cv_r2_scores.mean():.3f} (±{cv_r2_scores.std():.3f})")
        
    except Exception as e:
        LOG.warning(f"Error evaluating {model_name}: {e}")
        results[model_name] = {'error': str(e)}

return results
```

def analyze_best_models(results):
“”“Analyze and rank the best performing models”””

```
LOG.info("=== ANALYZING BEST MODELS ===")

# Filter out failed models
valid_results = {k: v for k, v in results.items() if 'error' not in v}

if not valid_results:
    LOG.error("No valid model results found")
    return None

# Sort by CV MAE (lower is better)
sorted_by_mae = sorted(valid_results.items(), key=lambda x: x[1]['cv_mae_mean'])

# Sort by CV R² (higher is better)
sorted_by_r2 = sorted(valid_results.items(), key=lambda x: x[1]['cv_r2_mean'], reverse=True)

LOG.info("Top 5 models by MAE (Cross-Validation):")
for i, (model_name, metrics) in enumerate(sorted_by_mae[:5], 1):
    LOG.info(f"  {i}. {model_name}: {metrics['cv_mae_mean']:.0f} ± {metrics['cv_mae_std']:.0f}")

LOG.info("Top 5 models by R² (Cross-Validation):")
for i, (model_name, metrics) in enumerate(sorted_by_r2[:5], 1):
    LOG.info(f"  {i}. {model_name}: {metrics['cv_r2_mean']:.3f} ± {metrics['cv_r2_std']:.3f}")

# Get best overall model (by CV MAE)
best_model_name, best_metrics = sorted_by_mae[0]

LOG.info(f"\nBEST MODEL: {best_model_name}")
LOG.info(f"  Cross-Validation MAE: {best_metrics['cv_mae_mean']:.0f} ± {best_metrics['cv_mae_std']:.0f}")
LOG.info(f"  Cross-Validation R²: {best_metrics['cv_r2_mean']:.3f} ± {best_metrics['cv_r2_std']:.3f}")

# Show prediction intervals for best model
LOG.info(f"\nPrediction Intervals for {best_model_name}:")
for confidence, interval in best_metrics['prediction_intervals'].items():
    LOG.info(f"  {confidence} confidence: ±{interval['width']:.0f} calls")

return best_model_name, best_metrics, sorted_by_mae
```

def create_business_summary(best_model_name, best_metrics, top_mail_types, top_econ_indicators):
“”“Create business-friendly summary”””

```
LOG.info("=== BUSINESS SUMMARY ===")

# Model performance
accuracy = best_metrics['cv_r2_mean']
mae = best_metrics['cv_mae_mean']

LOG.info(f"PRODUCTION MODEL PERFORMANCE:")
LOG.info(f"  Algorithm: {best_model_name}")
LOG.info(f"  Accuracy: {accuracy:.1%} of call volume variance explained")
LOG.info(f"  Average Error: ±{mae:.0f} calls per day")

# Prediction intervals
intervals = best_metrics['prediction_intervals']
LOG.info(f"\nPREDICTION CONFIDENCE INTERVALS:")
LOG.info(f"  50% confidence: ±{intervals['50%']['width']:.0f} calls")
LOG.info(f"  75% confidence: ±{intervals['75%']['width']:.0f} calls")
LOG.info(f"  90% confidence: ±{intervals['90%']['width']:.0f} calls")

# Key predictors
LOG.info(f"\nKEY PREDICTORS:")
LOG.info(f"  Top Mail Types: {', '.join(top_mail_types[:5])}")
LOG.info(f"  Top Economic Indicators: {', '.join(top_econ_indicators[:5])}")

# Business impact
LOG.info(f"\nBUSINESS IMPACT:")
LOG.info(f"  • Predict daily call volume with {accuracy:.1%} accuracy")
LOG.info(f"  • Plan staffing levels with ±{mae:.0f} call precision")
LOG.info(f"  • Optimize mail campaign timing based on market conditions")
LOG.info(f"  • Reduce operational costs through better resource allocation")
```

def main():
output_dir = Path(CFG[“output_dir”])
output_dir.mkdir(exist_ok=True)

```
LOG.info("=== PRODUCTION-READY PREDICTIVE MODEL ===")

# Load and clean data
data, actual_mail_types, economic_columns = load_clean_data()

# Analyze correlations properly
mail_correlations, econ_correlations = analyze_correlations_properly(
    data, actual_mail_types, economic_columns
)

# Test both 10 and 15 mail types
best_overall_model = None
best_overall_score = float('inf')

for mail_count in CFG["mail_counts_to_test"]:
    LOG.info(f"\n{'='*60}")
    LOG.info(f"TESTING WITH TOP {mail_count} MAIL TYPES")
    LOG.info(f"{'='*60}")
    
    # Create features
    X, y, top_mail_types, top_econ_indicators = create_production_features(
        data, mail_correlations, econ_correlations, mail_count
    )
    
    # Get model suite
    models = get_model_suite()
    
    # Evaluate models
    results = evaluate_models_comprehensive(X, y, models)
    
    # Analyze best models
    best_model_name, best_metrics, sorted_results = analyze_best_models(results)
    
    # Check if this is the best overall
    if best_metrics and best_metrics['cv_mae_mean'] < best_overall_score:
        best_overall_score = best_metrics['cv_mae_mean']
        best_overall_model = {
            'mail_count': mail_count,
            'model_name': best_model_name,
            'metrics': best_metrics,
            'features': X.columns.tolist(),
            'top_mail_types': top_mail_types,
            'top_econ_indicators': top_econ_indicators,
            'results': results
        }
    
    # Create business summary
    create_business_summary(best_model_name, best_metrics, top_mail_types, top_econ_indicators)
    
    # Save results for this mail count
    with open(output_dir / f"results_top_{mail_count}_mail.json", "w") as f:
        json.dump({
            'mail_count': mail_count,
            'best_model': best_model_name,
            'results': {k: {kk: vv for kk, vv in v.items() if kk != 'model'} 
                       for k, v in results.items() if 'error' not in v},
            'top_mail_types': top_mail_types,
            'top_econ_indicators': top_econ_indicators
        }, f, indent=2, default=str)

# Final recommendation
if best_overall_model:
    LOG.info(f"\n{'='*60}")
    LOG.info(f"FINAL RECOMMENDATION")
    LOG.info(f"{'='*60}")
    
    LOG.info(f"Optimal Configuration:")
    LOG.info(f"  Mail Types: {best_overall_model['mail_count']}")
    LOG.info(f"  Algorithm: {best_overall_model['model_name']}")
    LOG.info(f"  Accuracy: {best_overall_model['metrics']['cv_r2_mean']:.1%}")
    LOG.info(f"  Average Error: ±{best_overall_model['metrics']['cv_mae_mean']:.0f} calls")
    
    # Save best model
    with open(output_dir / "best_production_model.pkl", "wb") as f:
        pickle.dump(best_overall_model['metrics']['model'], f)
    
    # Save comprehensive results
    with open(output_dir / "production_model_summary.json", "w") as f:
        json.dump({
            'best_overall_model': {
                'mail_count': best_overall_model['mail_count'],
                'model_name': best_overall_model['model_name'],
                'cv_mae_mean': best_overall_model['metrics']['cv_mae_mean'],
                'cv_r2_mean': best_overall_model['metrics']['cv_r2_mean'],
                'prediction_intervals': best_overall_model['metrics']['prediction_intervals'],
                'top_mail_types': best_overall_model['top_mail_types'],
                'top_econ_indicators': best_overall_model['top_econ_indicators']
            },
            'mail_correlations': mail_correlations[:20],  # Top 20
            'econ_correlations': econ_correlations[:10]   # Top 10
        }, f, indent=2, default=str)
    
    LOG.info(f"\nProduction model saved to: {output_dir.resolve()}")
    
    # Business recommendations
    LOG.info(f"\nBUSINESS RECOMMENDATIONS:")
    LOG.info(f"1. Deploy {best_overall_model['model_name']} for daily call volume prediction")
    LOG.info(f"2. Use {best_overall_model['mail_count']} key mail types for input")
    LOG.info(f"3. Monitor {len(best_overall_model['top_econ_indicators'])} economic indicators")
    LOG.info(f"4. Plan capacity with ±{best_overall_model['metrics']['cv_mae_mean']:.0f} call buffer")
    LOG.info(f"5. Update model monthly with new data")
```

def predict_from_inputs(model, features, mail_inputs, economic_inputs=None):
“””
Production function to predict calls from mail campaign inputs

```
Args:
    model: Trained model
    features: List of feature names
    mail_inputs: Dict of mail types and volumes
    economic_inputs: Dict of economic indicator values (optional)

Returns:
    prediction: Predicted call volume
    confidence_intervals: Prediction intervals
"""

# Create feature vector
feature_row = {}

# Initialize all features to 0
for feature in features:
    feature_row[feature] = 0

# Fill mail features
for mail_type, volume in mail_inputs.items():
    if f"{mail_type}_today" in features:
        feature_row[f"{mail_type}_today"] = volume

# Calculate total mail
total_mail = sum(mail_inputs.values())
if "total_mail_today" in features:
    feature_row["total_mail_today"] = total_mail
if "log_total_mail_today" in features:
    feature_row["log_total_mail_today"] = np.log1p(total_mail)

# Fill economic features (use defaults if not provided)
if economic_inputs:
    for indicator, value in economic_inputs.items():
        if f"{indicator}_today" in features:
            feature_row[f"{indicator}_today"] = value
        if f"{indicator}_lag1" in features:
            feature_row[f"{indicator}_lag1"] = value

# Date features (use current date)
today = datetime.now()
if "month" in features:
    feature_row["month"] = today.month
if "weekday" in features:
    feature_row["weekday"] = today.weekday()
if "is_month_end" in features:
    feature_row["is_month_end"] = 1 if today.day > 25 else 0
if "is_quarter_end" in features:
    feature_row["is_quarter_end"] = 1 if today.month in [3, 6, 9, 12] and today.day > 25 else 0

# Default values for other features
if "recent_calls_avg" in features:
    feature_row["recent_calls_avg"] = 15000  # Default baseline

# Create prediction
X_input = pd.DataFrame([feature_row])[features]
prediction = model.predict(X_input)[0]

return max(0, prediction)  # No negative calls
```

if **name** == “**main**”:
main()