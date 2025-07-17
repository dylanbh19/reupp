#!/usr/bin/env python

# test_mail_count_performance.py

# =========================================================

# SIMPLE TEST: Compare model performance with:

# - Top 10 mail types + economic indicators

# - Top 15 mail types + economic indicators

# - Top 20 mail types + economic indicators

# =========================================================

from pathlib import Path
import logging
import sys
import warnings
from datetime import datetime

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

LOG = logging.getLogger(“test_mail_count”)
logging.basicConfig(
level=logging.INFO,
format=”%(asctime)s | test_mail_count | %(levelname)s | %(message)s”,
handlers=[logging.StreamHandler(sys.stdout)]
)

CFG = {
“economic_data_path”: “economic_data_for_model.csv”,
“best_economic”: [“SP500”, “FinancialSector”, “InterestRate_10Y”],  # Skip VIX
“test_counts”: [10, 15, 20]  # Test these many top mail types
}

def _to_date(s):
return pd.to_datetime(s, errors=“coerce”).dt.date

def _find_file(candidates):
for p in candidates:
path = Path(p)
if path.exists():
return path
raise FileNotFoundError(f”None found: {candidates}”)

def load_data():
“”“Load mail, calls, and economic data”””

```
LOG.info("Loading data...")

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

combined = mail_daily.join(calls_total.rename("calls_total"), how="inner")

# Load economic data
econ_path = Path(CFG["economic_data_path"])
if econ_path.exists():
    econ_data = pd.read_csv(econ_path, parse_dates=['Date'])
    econ_data.set_index('Date', inplace=True)
    combined = combined.join(econ_data[CFG["best_economic"]], how='left')
    combined = combined.fillna(method='ffill').fillna(method='bfill')

LOG.info("Data loaded: %s", combined.shape)
return combined
```

def get_top_mail_types(data, top_n):
“”“Get top N mail types by correlation with call volume”””

```
# Get all mail columns (exclude calls_total and economic indicators)
exclude_cols = ['calls_total'] + CFG["best_economic"]
mail_columns = [col for col in data.columns if col not in exclude_cols]

correlations = []

for mail_type in mail_columns:
    if data[mail_type].sum() > 0:  # Only consider mail types with volume
        corr_same, p_same = pearsonr(data[mail_type], data['calls_total'])
        
        correlations.append({
            'mail_type': mail_type,
            'correlation': abs(corr_same),  # Use absolute correlation
            'correlation_raw': corr_same,
            'p_value': p_same,
            'avg_volume': data[mail_type].mean()
        })

# Sort by correlation strength
correlations.sort(key=lambda x: x['correlation'], reverse=True)

# Get top N
top_mail_types = [item['mail_type'] for item in correlations[:top_n]]

return top_mail_types, correlations[:top_n]
```

def create_features(data, mail_types):
“”“Create features for the specified mail types + economic indicators”””

```
features_list = []
targets_list = []

# Get available features
available_mail = [t for t in mail_types if t in data.columns]
available_econ = [t for t in CFG["best_economic"] if t in data.columns]

for i in range(1, len(data) - 1):  # Start from index 1 for 1-day lag
    current_day = data.iloc[i]
    prev_day = data.iloc[i-1]
    next_day = data.iloc[i + 1]
    
    feature_row = {}
    
    # Mail features (same day - strongest correlation)
    for mail_type in available_mail:
        feature_row[f"{mail_type}_today"] = current_day[mail_type]
    
    # Total mail volume
    total_mail = sum(current_day[t] for t in available_mail)
    feature_row["total_mail_today"] = total_mail
    feature_row["log_total_mail_today"] = np.log1p(total_mail)
    
    # Economic features (1-day lag)
    for econ_indicator in available_econ:
        feature_row[f"{econ_indicator}_lag1"] = prev_day[econ_indicator]
    
    # Basic interaction: SP500 × Total Mail
    if "SP500" in available_econ:
        feature_row["sp500_x_total_mail"] = prev_day["SP500"] * np.log1p(total_mail)
    
    # Date features
    current_date = data.index[i]
    feature_row["month"] = current_date.month
    feature_row["weekday"] = current_date.weekday()
    feature_row["is_month_end"] = 1 if current_date.day > 25 else 0
    
    # Target: next day's calls
    target = next_day["calls_total"]
    
    features_list.append(feature_row)
    targets_list.append(target)

X = pd.DataFrame(features_list)
y = pd.Series(targets_list)
X = X.fillna(0)

return X, y
```

def test_model_performance(X, y, model_name):
“”“Test model performance with cross-validation”””

```
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Test different models
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
}

results = {}

for name, model in models.items():
    try:
        # Scale features for linear regression
        if name == "Linear Regression":
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
        
        results[name] = {
            'r2': r2,
            'mae': mae,
            'rmse': rmse
        }
        
        # Get feature importance for Random Forest
        if name == "Random Forest" and hasattr(model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            results[name]['top_features'] = top_features
        
    except Exception as e:
        LOG.warning(f"Error with {name}: {e}")
        results[name] = {'error': str(e)}

return results
```

def main():
LOG.info(”=== TESTING TOP 10 vs 15 vs 20 MAIL TYPES + ECONOMIC ===”)

```
# Load data
data = load_data()

# Test different numbers of mail types
results_summary = {}

for top_n in CFG["test_counts"]:
    LOG.info(f"\n=== TESTING TOP {top_n} MAIL TYPES ===")
    
    # Get top N mail types
    top_mail_types, correlations = get_top_mail_types(data, top_n)
    
    LOG.info(f"Top {top_n} mail types:")
    for i, item in enumerate(correlations, 1):
        LOG.info(f"  {i:2d}. {item['mail_type']}: {item['correlation_raw']:.3f}")
    
    # Create features
    X, y = create_features(data, top_mail_types)
    
    LOG.info(f"Features created: {X.shape[0]} samples x {X.shape[1]} features")
    
    # Test model performance
    model_results = test_model_performance(X, y, f"Top_{top_n}_Mail")
    
    # Log results
    LOG.info(f"Model Performance (Top {top_n} Mail Types):")
    for model_name, metrics in model_results.items():
        if 'error' not in metrics:
            LOG.info(f"  {model_name}:")
            LOG.info(f"    R² Score: {metrics['r2']:.3f}")
            LOG.info(f"    MAE: {metrics['mae']:.0f}")
            LOG.info(f"    RMSE: {metrics['rmse']:.0f}")
            
            if 'top_features' in metrics:
                LOG.info(f"    Top 5 features:")
                for feat, importance in metrics['top_features']:
                    LOG.info(f"      {feat}: {importance:.3f}")
    
    # Store results
    results_summary[f"top_{top_n}"] = {
        'mail_types': top_mail_types,
        'correlations': correlations,
        'model_results': model_results,
        'feature_count': X.shape[1]
    }

# Summary comparison
LOG.info("\n=== SUMMARY COMPARISON ===")

comparison_data = []
for top_n in CFG["test_counts"]:
    key = f"top_{top_n}"
    if key in results_summary:
        rf_results = results_summary[key]['model_results'].get('Random Forest', {})
        if 'r2' in rf_results:
            comparison_data.append({
                'mail_count': top_n,
                'feature_count': results_summary[key]['feature_count'],
                'r2_score': rf_results['r2'],
                'mae': rf_results['mae'],
                'rmse': rf_results['rmse']
            })

# Print comparison table
LOG.info("Random Forest Performance Comparison:")
LOG.info("Mail Types | Features | R² Score | MAE   | RMSE")
LOG.info("-" * 50)
for item in comparison_data:
    LOG.info(f"{item['mail_count']:10d} | {item['feature_count']:8d} | {item['r2_score']:8.3f} | {item['mae']:5.0f} | {item['rmse']:5.0f}")

# Recommendations
LOG.info("\n=== RECOMMENDATIONS ===")

if comparison_data:
    # Find best by R² score
    best_r2 = max(comparison_data, key=lambda x: x['r2_score'])
    # Find best by MAE (lowest)
    best_mae = min(comparison_data, key=lambda x: x['mae'])
    
    LOG.info(f"Best R² Score: Top {best_r2['mail_count']} mail types (R² = {best_r2['r2_score']:.3f})")
    LOG.info(f"Best MAE: Top {best_mae['mail_count']} mail types (MAE = {best_mae['mae']:.0f})")
    
    # Check if there's a clear winner
    r2_scores = [item['r2_score'] for item in comparison_data]
    r2_range = max(r2_scores) - min(r2_scores)
    
    if r2_range < 0.02:  # Very small difference
        LOG.info("RECOMMENDATION: All perform similarly. Choose Top 10 for simplicity.")
    elif best_r2['mail_count'] == best_mae['mail_count']:
        LOG.info(f"RECOMMENDATION: Top {best_r2['mail_count']} mail types (best on both metrics)")
    else:
        LOG.info(f"RECOMMENDATION: Top {best_r2['mail_count']} mail types (best accuracy)")

LOG.info("\n=== ANALYSIS COMPLETE ===")
```

if **name** == “**main**”:
main()