#!/usr/bin/env python

# comprehensive_analysis.py

# =========================================================

# COMPREHENSIVE ANALYSIS using YOUR data:

# 1. Test ALL indicators from expanded_economic_data.csv

# 2. Test original economic_data_for_model.csv indicators

# 3. Test mail types 10/15/20 with all economic combinations

# 4. Find optimal indicators and mail type count

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
from scipy.stats import pearsonr

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings(‘ignore’)

LOG = logging.getLogger(“comprehensive_analysis”)
logging.basicConfig(
level=logging.INFO,
format=”%(asctime)s | comprehensive | %(levelname)s | %(message)s”,
handlers=[logging.StreamHandler(sys.stdout)]
)

CFG = {
“original_economic_data”: “economic_data_for_model.csv”,
“expanded_economic_data”: “expanded_economic_data.csv”,
“mail_counts_to_test”: [10, 15, 20],
“output_dir”: “comprehensive_analysis_results”
}

def _to_date(s):
return pd.to_datetime(s, errors=“coerce”).dt.date

def _find_file(candidates):
for p in candidates:
path = Path(p)
if path.exists():
return path
raise FileNotFoundError(f”None found: {candidates}”)

def load_all_data():
“”“Load mail, calls, and both economic datasets”””

```
LOG.info("=== LOADING ALL DATA ===")

# Load mail data
LOG.info("Loading mail data...")
mail_path = _find_file(["mail.csv", "data/mail.csv"])
mail = pd.read_csv(mail_path)
mail.columns = [c.lower().strip() for c in mail.columns]
mail["mail_date"] = _to_date(mail["mail_date"])
mail = mail.dropna(subset=["mail_date"])

# Load call data
LOG.info("Loading call data...")
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
    LOG.info(f"Scaled call volumes by factor: {scale:.2f}")
calls_total = vol_daily.combine_first(int_daily).sort_index()

# Process mail
mail_daily = (mail.groupby(["mail_date", "mail_type"], as_index=False)["mail_volume"].sum()
               .pivot(index="mail_date", columns="mail_type", values="mail_volume").fillna(0))

mail_daily.index = pd.to_datetime(mail_daily.index)
calls_total.index = pd.to_datetime(calls_total.index)

# Business days only
us_holidays = holidays.US()
biz_mask = (~mail_daily.index.weekday.isin([5, 6])) & (~mail_daily.index.isin(us_holidays))
mail_daily = mail_daily.loc[biz_mask]
calls_total = calls_total.loc[calls_total.index.isin(mail_daily.index)]

combined_base = mail_daily.join(calls_total.rename("calls_total"), how="inner")

# Load original economic data
LOG.info("Loading original economic data...")
original_econ = pd.DataFrame()
original_path = Path(CFG["original_economic_data"])
if original_path.exists():
    original_econ = pd.read_csv(original_path, parse_dates=['Date'])
    original_econ.set_index('Date', inplace=True)
    LOG.info(f"Original economic indicators: {list(original_econ.columns)}")
else:
    LOG.warning("Original economic data not found")

# Load expanded economic data
LOG.info("Loading expanded economic data...")
expanded_econ = pd.DataFrame()
expanded_path = Path(CFG["expanded_economic_data"])
if expanded_path.exists():
    expanded_econ = pd.read_csv(expanded_path, parse_dates=['Date'])
    expanded_econ.set_index('Date', inplace=True)
    LOG.info(f"Expanded economic indicators: {list(expanded_econ.columns)}")
else:
    LOG.warning("Expanded economic data not found")

# Combine all economic data
all_economic = pd.DataFrame()
if not original_econ.empty:
    all_economic = pd.concat([all_economic, original_econ], axis=1)
if not expanded_econ.empty:
    all_economic = pd.concat([all_economic, expanded_econ], axis=1)

# Remove duplicate columns
all_economic = all_economic.loc[:, ~all_economic.columns.duplicated()]

if not all_economic.empty:
    LOG.info(f"Combined economic data: {len(all_economic.columns)} indicators")
    LOG.info(f"All economic indicators: {list(all_economic.columns)}")
    
    # Join with mail/calls data
    combined_full = combined_base.join(all_economic, how='left')
    combined_full = combined_full.fillna(method='ffill').fillna(method='bfill')
    
    LOG.info(f"Final combined data shape: {combined_full.shape}")
    return combined_full, list(all_economic.columns)
else:
    LOG.warning("No economic data loaded")
    return combined_base, []
```

def rank_all_mail_types(data):
“”“Rank all mail types by correlation with call volume”””

```
LOG.info("=== RANKING ALL MAIL TYPES ===")

# Get all mail columns
mail_columns = [col for col in data.columns if col not in ['calls_total']]
# Filter out economic indicators (they should be numeric and have different patterns)
mail_columns = [col for col in mail_columns if data[col].sum() > 0]  # Only mail with volume

correlations = []

for mail_type in mail_columns:
    try:
        # Same day correlation
        corr_same, p_same = pearsonr(data[mail_type], data['calls_total'])
        # 1-day lag correlation
        corr_lag1, p_lag1 = pearsonr(data[mail_type].shift(1).dropna(), 
                                    data['calls_total'].iloc[1:])
        
        correlations.append({
            'mail_type': mail_type,
            'correlation_same': corr_same,
            'correlation_lag1': corr_lag1,
            'p_same': p_same,
            'p_lag1': p_lag1,
            'abs_correlation': abs(corr_same),
            'total_volume': data[mail_type].sum(),
            'frequency': (data[mail_type] > 0).sum() / len(data)
        })
    except:
        # Skip if correlation calculation fails
        continue

# Sort by absolute correlation strength
correlations.sort(key=lambda x: x['abs_correlation'], reverse=True)

LOG.info(f"Analyzed {len(correlations)} mail types")
LOG.info("Top 25 mail types by correlation:")
for i, item in enumerate(correlations[:25], 1):
    LOG.info(f"  {i:2d}. {item['mail_type']}: {item['correlation_same']:.3f} "
            f"(vol: {item['total_volume']:,.0f}, freq: {item['frequency']:.1%})")

return correlations
```

def test_all_economic_indicators(data, economic_indicators):
“”“Test all economic indicators for correlation with call volume”””

```
LOG.info("=== TESTING ALL ECONOMIC INDICATORS ===")

econ_correlations = []

for indicator in economic_indicators:
    if indicator in data.columns:
        try:
            # Same day correlation
            clean_data = data[[indicator, 'calls_total']].dropna()
            if len(clean_data) > 10:  # Need sufficient data
                corr_same, p_same = pearsonr(clean_data[indicator], clean_data['calls_total'])
                
                # 1-day lag correlation
                lag_data = data[[indicator, 'calls_total']].dropna()
                if len(lag_data) > 11:
                    corr_lag1, p_lag1 = pearsonr(lag_data[indicator].shift(1).dropna(), 
                                                lag_data['calls_total'].iloc[1:])
                else:
                    corr_lag1, p_lag1 = 0, 1
                
                # 5-day rolling correlation
                rolling_indicator = data[indicator].rolling(5).mean().dropna()
                if len(rolling_indicator) > 5:
                    corr_rolling, p_rolling = pearsonr(rolling_indicator, 
                                                      data['calls_total'].loc[rolling_indicator.index])
                else:
                    corr_rolling, p_rolling = 0, 1
                
                econ_correlations.append({
                    'indicator': indicator,
                    'correlation_same': corr_same,
                    'correlation_lag1': corr_lag1,
                    'correlation_rolling': corr_rolling,
                    'p_same': p_same,
                    'p_lag1': p_lag1,
                    'p_rolling': p_rolling,
                    'abs_correlation': abs(corr_same),
                    'best_correlation': max(abs(corr_same), abs(corr_lag1), abs(corr_rolling))
                })
        except Exception as e:
            LOG.warning(f"Error testing {indicator}: {e}")
            continue

# Sort by best correlation
econ_correlations.sort(key=lambda x: x['best_correlation'], reverse=True)

LOG.info(f"Tested {len(econ_correlations)} economic indicators")
LOG.info("Top 15 economic indicators by correlation:")
for i, item in enumerate(econ_correlations[:15], 1):
    LOG.info(f"  {i:2d}. {item['indicator']}: same={item['correlation_same']:.3f}, "
            f"lag1={item['correlation_lag1']:.3f}, rolling={item['correlation_rolling']:.3f}")

return econ_correlations
```

def create_feature_combinations(data, mail_correlations, econ_correlations, mail_count):
“”“Create features for testing with specified mail count”””

```
# Get top mail types
top_mail_types = [item['mail_type'] for item in mail_correlations[:mail_count]]

# Get top economic indicators
top_econ_indicators = [item['indicator'] for item in econ_correlations[:5]]

features_list = []
targets_list = []

for i in range(1, len(data) - 1):
    current_day = data.iloc[i]
    prev_day = data.iloc[i-1]
    next_day = data.iloc[i + 1]
    
    feature_row = {}
    
    # Mail features (same day - strongest correlation)
    for mail_type in top_mail_types:
        if mail_type in data.columns:
            feature_row[f"{mail_type}_today"] = current_day[mail_type]
    
    # Total mail features
    total_mail = sum(current_day.get(mail_type, 0) for mail_type in top_mail_types)
    feature_row["total_mail_today"] = total_mail
    feature_row["log_total_mail_today"] = np.log1p(total_mail)
    
    # Economic features (test different lags)
    for econ_indicator in top_econ_indicators:
        if econ_indicator in data.columns:
            feature_row[f"{econ_indicator}_today"] = current_day[econ_indicator]
            feature_row[f"{econ_indicator}_lag1"] = prev_day[econ_indicator]
    
    # Interaction features (top economic × total mail)
    if top_econ_indicators and top_econ_indicators[0] in data.columns:
        best_econ = prev_day[top_econ_indicators[0]]
        feature_row["best_econ_x_total_mail"] = best_econ * np.log1p(total_mail)
    
    # Date features
    current_date = data.index[i]
    feature_row["month"] = current_date.month
    feature_row["weekday"] = current_date.weekday()
    feature_row["is_month_end"] = 1 if current_date.day > 25 else 0
    
    # Recent call baseline
    recent_calls = data["calls_total"].iloc[max(0, i-5):i].mean()
    feature_row["recent_calls_avg"] = recent_calls
    
    # Target
    target = next_day["calls_total"]
    
    features_list.append(feature_row)
    targets_list.append(target)

X = pd.DataFrame(features_list)
y = pd.Series(targets_list)
X = X.fillna(0)

return X, y, top_mail_types, top_econ_indicators
```

def test_model_combinations(data, mail_correlations, econ_correlations):
“”“Test different combinations of mail types and economic indicators”””

```
LOG.info("=== TESTING MODEL COMBINATIONS ===")

results = {}

for mail_count in CFG["mail_counts_to_test"]:
    LOG.info(f"\nTesting with Top {mail_count} mail types...")
    
    # Create features
    X, y, top_mail_types, top_econ_indicators = create_feature_combinations(
        data, mail_correlations, econ_correlations, mail_count
    )
    
    LOG.info(f"Features: {X.shape[0]} samples x {X.shape[1]} features")
    LOG.info(f"Top mail types: {top_mail_types[:5]}...")  # Show first 5
    LOG.info(f"Top economic indicators: {top_econ_indicators}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test different models
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
    }
    
    mail_count_results = {}
    
    for model_name, model in models.items():
        try:
            # Scale features for linear regression
            if model_name == "Linear Regression":
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(((y_test - y_pred) ** 2).mean())
            
            mail_count_results[model_name] = {
                'r2': r2,
                'mae': mae,
                'rmse': rmse,
                'feature_count': X.shape[1]
            }
            
            LOG.info(f"  {model_name}:")
            LOG.info(f"    R² Score: {r2:.3f}")
            LOG.info(f"    MAE: {mae:.0f}")
            LOG.info(f"    RMSE: {rmse:.0f}")
            
            # Feature importance for Random Forest
            if model_name == "Random Forest" and hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X.columns, model.feature_importances_))
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                
                LOG.info(f"    Top 5 features:")
                for feat, importance in top_features:
                    LOG.info(f"      {feat}: {importance:.3f}")
                
                mail_count_results[model_name]['top_features'] = top_features
            
        except Exception as e:
            LOG.warning(f"Error with {model_name}: {e}")
            mail_count_results[model_name] = {'error': str(e)}
    
    results[f"mail_count_{mail_count}"] = {
        'results': mail_count_results,
        'top_mail_types': top_mail_types,
        'top_econ_indicators': top_econ_indicators,
        'feature_count': X.shape[1]
    }

return results
```

def analyze_results_and_recommendations(results, mail_correlations, econ_correlations):
“”“Analyze results and provide recommendations”””

```
LOG.info("=== ANALYSIS AND RECOMMENDATIONS ===")

# Compare Random Forest performance across mail counts
comparison_data = []
for mail_count in CFG["mail_counts_to_test"]:
    key = f"mail_count_{mail_count}"
    if key in results:
        rf_results = results[key]['results'].get('Random Forest', {})
        if 'r2' in rf_results:
            comparison_data.append({
                'mail_count': mail_count,
                'r2_score': rf_results['r2'],
                'mae': rf_results['mae'],
                'rmse': rf_results['rmse'],
                'feature_count': rf_results['feature_count']
            })

# Print comparison
LOG.info("\nPerformance Comparison (Random Forest):")
LOG.info("Mail Count | R² Score | MAE   | RMSE  | Features")
LOG.info("-" * 50)
for item in comparison_data:
    LOG.info(f"{item['mail_count']:10d} | {item['r2_score']:8.3f} | {item['mae']:5.0f} | {item['rmse']:5.0f} | {item['feature_count']:8d}")

# Find best performing configuration
if comparison_data:
    best_r2 = max(comparison_data, key=lambda x: x['r2_score'])
    best_mae = min(comparison_data, key=lambda x: x['mae'])
    
    LOG.info(f"\nBest R² Score: {best_r2['mail_count']} mail types (R² = {best_r2['r2_score']:.3f})")
    LOG.info(f"Best MAE: {best_mae['mail_count']} mail types (MAE = {best_mae['mae']:.0f})")
    
    # Overall recommendation
    if best_r2['mail_count'] == best_mae['mail_count']:
        recommended_count = best_r2['mail_count']
        LOG.info(f"\nRECOMMENDATION: Use {recommended_count} mail types (best on both metrics)")
    else:
        recommended_count = best_r2['mail_count']
        LOG.info(f"\nRECOMMENDATION: Use {recommended_count} mail types (best accuracy)")

# Show best economic indicators
LOG.info("\nBest Economic Indicators (Top 10):")
for i, item in enumerate(econ_correlations[:10], 1):
    LOG.info(f"  {i:2d}. {item['indicator']}: {item['best_correlation']:.3f}")

# Show best mail types
LOG.info("\nBest Mail Types (Top 15):")
for i, item in enumerate(mail_correlations[:15], 1):
    LOG.info(f"  {i:2d}. {item['mail_type']}: {item['correlation_same']:.3f}")

return comparison_data
```

def main():
output_dir = Path(CFG[“output_dir”])
output_dir.mkdir(exist_ok=True)

```
LOG.info("=== COMPREHENSIVE ANALYSIS ===")
LOG.info("Testing all economic indicators + mail type combinations")

# Load all data
data, economic_indicators = load_all_data()

if not economic_indicators:
    LOG.error("No economic data found. Please check your CSV files.")
    return

# Rank all mail types
mail_correlations = rank_all_mail_types(data)

# Test all economic indicators
econ_correlations = test_all_economic_indicators(data, economic_indicators)

# Test model combinations
model_results = test_model_combinations(data, mail_correlations, econ_correlations)

# Analyze results and provide recommendations
comparison_data = analyze_results_and_recommendations(model_results, mail_correlations, econ_correlations)

# Save comprehensive results
full_results = {
    'analysis_timestamp': datetime.now().isoformat(),
    'mail_correlations': mail_correlations,
    'econ_correlations': econ_correlations,
    'model_results': model_results,
    'comparison_data': comparison_data
}

with open(output_dir / "comprehensive_analysis.json", "w") as f:
    json.dump(full_results, f, indent=2, default=str)

LOG.info(f"\n=== ANALYSIS COMPLETE ===")
LOG.info(f"Results saved to: {output_dir.resolve()}")
```

if **name** == “**main**”:
main()