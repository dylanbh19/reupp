#!/usr/bin/env python

# mail_econ_call_prediction_analysis.py

# =========================================================

# FOCUSED analysis: How well do mail types/volumes + economic

# indicators predict CALL VOLUMES?

# 

# Target: Call Volume (what we want to predict)

# Predictors: Mail types, mail volumes, economic indicators

# =========================================================

from pathlib import Path
import json
import logging
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import holidays
from scipy.stats import pearsonr, spearmanr

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

warnings.filterwarnings(‘ignore’)

# Set up logging

LOG = logging.getLogger(“mail_econ_call_analysis”)
logging.basicConfig(
level=logging.INFO,
format=”%(asctime)s | analysis | %(levelname)s | %(message)s”,
handlers=[logging.StreamHandler(sys.stdout)]
)

# Configuration

CFG = {
“economic_data_path”: “economic_data_for_model.csv”,
“top_mail_types”: [
“Reject_Ltrs”, “Cheque 1099”, “Exercise_Converted”, “SOI_Confirms”,
“Exch_chks”, “ACH_Debit_Enrollment”, “Transfer”, “COA”,
“NOTC_WITHDRAW”, “Repl_Chks”
],
“output_dir”: “mail_econ_call_analysis”
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
“”“Load mail, calls, and economic data into one dataset”””

```
LOG.info("=== LOADING ALL DATA ===")

# Load mail data
LOG.info("Loading mail data...")
mail_path = _find_file(["mail.csv", "data/mail.csv"])
mail = pd.read_csv(mail_path)
mail.columns = [c.lower().strip() for c in mail.columns]
mail["mail_date"] = _to_date(mail["mail_date"])
mail = mail.dropna(subset=["mail_date"])

# Load call volumes
LOG.info("Loading call volume data...")
vol_path = _find_file(["callvolumes.csv", "data/callvolumes.csv"])
df_vol = pd.read_csv(vol_path)
df_vol.columns = [c.lower().strip() for c in df_vol.columns]
dcol_v = next(c for c in df_vol.columns if "date" in c)
df_vol[dcol_v] = _to_date(df_vol[dcol_v])
vol_daily = df_vol.groupby(dcol_v)[df_vol.columns.difference([dcol_v])[0]].sum()

# Load call intent
LOG.info("Loading call intent data...")
intent_path = _find_file(["callintent.csv", "data/callintent.csv"])
df_int = pd.read_csv(intent_path)
df_int.columns = [c.lower().strip() for c in df_int.columns]
dcol_i = next(c for c in df_int.columns if "date" in c or "conversationstart" in c)
df_int[dcol_i] = _to_date(df_int[dcol_i])
int_daily = df_int.groupby(dcol_i).size()

# Scale and combine call data
overlap = vol_daily.index.intersection(int_daily.index)
if len(overlap) >= 5:
    scale = int_daily.loc[overlap].mean() / vol_daily.loc[overlap].mean()
    vol_daily *= scale
    LOG.info("Scaled call volumes by factor: %.2f", scale)
calls_total = vol_daily.combine_first(int_daily).sort_index()

# Aggregate mail by type and date
mail_daily = (mail.groupby(["mail_date", "mail_type"], as_index=False)["mail_volume"].sum()
               .pivot(index="mail_date", columns="mail_type", values="mail_volume").fillna(0))

# Convert to datetime
mail_daily.index = pd.to_datetime(mail_daily.index)
calls_total.index = pd.to_datetime(calls_total.index)

# Business days only
us_holidays = holidays.US()
biz_mask = (~mail_daily.index.weekday.isin([5, 6])) & (~mail_daily.index.isin(us_holidays))
mail_daily = mail_daily.loc[biz_mask]
calls_total = calls_total.loc[calls_total.index.isin(mail_daily.index)]

# Combine mail and calls
combined = mail_daily.join(calls_total.rename("calls_total"), how="inner")

# Load economic data
LOG.info("Loading economic data...")
econ_path = Path(CFG["economic_data_path"])
if econ_path.exists():
    econ_data = pd.read_csv(econ_path, parse_dates=['Date'])
    econ_data.set_index('Date', inplace=True)
    LOG.info("Economic data columns: %s", list(econ_data.columns))
    
    # Join with economic data
    combined = combined.join(econ_data, how='left')
    combined = combined.fillna(method='ffill').fillna(method='bfill')
    LOG.info("Successfully loaded economic data")
else:
    LOG.warning("Economic data file not found. Analysis will be mail-only.")
    econ_data = pd.DataFrame()

LOG.info("Final combined dataset shape: %s", combined.shape)
LOG.info("Date range: %s to %s", combined.index.min(), combined.index.max())

return combined, econ_data
```

def analyze_mail_call_correlations(data):
“”“Analyze correlations between mail types/volumes and call volumes”””

```
LOG.info("=== MAIL → CALL CORRELATIONS ===")

# Get available mail types
available_mail_types = [t for t in CFG["top_mail_types"] if t in data.columns]

correlations = {}

# Individual mail type correlations
LOG.info("Individual mail type correlations with call volume:")
for mail_type in available_mail_types:
    if mail_type in data.columns:
        # Same-day correlation
        corr_same, p_same = pearsonr(data[mail_type], data['calls_total'])
        # 1-day lag correlation (mail today → calls tomorrow)
        corr_lag1, p_lag1 = pearsonr(data[mail_type].shift(1).dropna(), 
                                    data['calls_total'].iloc[1:])
        # 2-day lag correlation
        corr_lag2, p_lag2 = pearsonr(data[mail_type].shift(2).dropna(), 
                                    data['calls_total'].iloc[2:])
        
        correlations[mail_type] = {
            'same_day': corr_same,
            'lag_1_day': corr_lag1, 
            'lag_2_day': corr_lag2,
            'p_same': p_same,
            'p_lag1': p_lag1,
            'p_lag2': p_lag2
        }
        
        LOG.info("  %s:", mail_type)
        LOG.info("    Same day: %.3f (p=%.3f)", corr_same, p_same)
        LOG.info("    1-day lag: %.3f (p=%.3f)", corr_lag1, p_lag1)
        LOG.info("    2-day lag: %.3f (p=%.3f)", corr_lag2, p_lag2)

# Total mail volume correlations
total_mail = data[available_mail_types].sum(axis=1)
corr_total_same, p_total_same = pearsonr(total_mail, data['calls_total'])
corr_total_lag1, p_total_lag1 = pearsonr(total_mail.shift(1).dropna(), 
                                        data['calls_total'].iloc[1:])

LOG.info("\nTotal mail volume correlations:")
LOG.info("  Same day: %.3f (p=%.3f)", corr_total_same, p_total_same)
LOG.info("  1-day lag: %.3f (p=%.3f)", corr_total_lag1, p_total_lag1)
LOG.info("  Log total same day: %.3f", np.corrcoef(np.log1p(total_mail), data['calls_total'])[0,1])

correlations['total_mail'] = {
    'same_day': corr_total_same,
    'lag_1_day': corr_total_lag1,
    'log_same_day': np.corrcoef(np.log1p(total_mail), data['calls_total'])[0,1]
}

return correlations
```

def analyze_economic_call_correlations(data, econ_data):
“”“Analyze correlations between economic indicators and call volumes”””

```
if econ_data.empty:
    LOG.info("=== ECONOMIC → CALL CORRELATIONS ===")
    LOG.info("No economic data available for analysis")
    return {}

LOG.info("=== ECONOMIC → CALL CORRELATIONS ===")

# Get economic columns (exclude Date if present)
econ_columns = [col for col in econ_data.columns if col.lower() != 'date']

econ_correlations = {}

LOG.info("Economic indicator correlations with call volume:")
for econ_col in econ_columns:
    if econ_col in data.columns:
        # Same-day correlation
        corr_same, p_same = pearsonr(data[econ_col].dropna(), 
                                    data['calls_total'].loc[data[econ_col].dropna().index])
        
        # 1-day lag correlation (economic indicator today → calls tomorrow)
        shifted_econ = data[econ_col].shift(1).dropna()
        corr_lag1, p_lag1 = pearsonr(shifted_econ, 
                                    data['calls_total'].loc[shifted_econ.index])
        
        # Rolling average correlation (5-day average)
        rolling_econ = data[econ_col].rolling(5).mean().dropna()
        corr_rolling, p_rolling = pearsonr(rolling_econ, 
                                          data['calls_total'].loc[rolling_econ.index])
        
        econ_correlations[econ_col] = {
            'same_day': corr_same,
            'lag_1_day': corr_lag1,
            'rolling_5d': corr_rolling,
            'p_same': p_same,
            'p_lag1': p_lag1,
            'p_rolling': p_rolling
        }
        
        LOG.info("  %s:", econ_col)
        LOG.info("    Same day: %.3f (p=%.3f)", corr_same, p_same)
        LOG.info("    1-day lag: %.3f (p=%.3f)", corr_lag1, p_lag1)
        LOG.info("    5-day rolling: %.3f (p=%.3f)", corr_rolling, p_rolling)

return econ_correlations
```

def test_predictive_models(data, econ_data):
“”“Test different models to predict call volumes”””

```
LOG.info("=== PREDICTIVE MODEL TESTING ===")

# Get available features
available_mail_types = [t for t in CFG["top_mail_types"] if t in data.columns]
econ_columns = [col for col in econ_data.columns if col.lower() != 'date'] if not econ_data.empty else []

# Create feature sets
feature_sets = {}

# 1. Mail-only features
mail_features = []
for mail_type in available_mail_types:
    mail_features.append(mail_type)
    mail_features.append(f"{mail_type}_lag1")
    data[f"{mail_type}_lag1"] = data[mail_type].shift(1)

# Add total mail features
total_mail = data[available_mail_types].sum(axis=1)
data['total_mail'] = total_mail
data['total_mail_lag1'] = total_mail.shift(1)
data['log_total_mail'] = np.log1p(total_mail)
data['log_total_mail_lag1'] = np.log1p(total_mail).shift(1)

mail_features.extend(['total_mail', 'total_mail_lag1', 'log_total_mail', 'log_total_mail_lag1'])

# 2. Economic-only features (if available)
econ_features = []
if econ_columns:
    for econ_col in econ_columns:
        if econ_col in data.columns:
            econ_features.append(econ_col)
            econ_features.append(f"{econ_col}_lag1")
            data[f"{econ_col}_lag1"] = data[econ_col].shift(1)

# 3. Combined features
combined_features = mail_features + econ_features

# Define feature sets to test
feature_sets = {
    'Mail Only': [f for f in mail_features if f in data.columns],
    'Economic Only': [f for f in econ_features if f in data.columns],
    'Mail + Economic': [f for f in combined_features if f in data.columns]
}

# Remove empty feature sets
feature_sets = {k: v for k, v in feature_sets.items() if v}

results = {}

for set_name, features in feature_sets.items():
    LOG.info(f"\nTesting {set_name} features ({len(features)} features)...")
    
    # Prepare data
    X = data[features].dropna()
    y = data['calls_total'].loc[X.index]
    
    if len(X) < 50:
        LOG.warning(f"  Not enough data for {set_name} ({len(X)} samples)")
        continue
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Test different models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    set_results = {}
    
    for model_name, model in models.items():
        try:
            # Scale features for linear regression
            if model_name == 'Linear Regression':
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
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            set_results[model_name] = {
                'r2': r2,
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse)
            }
            
            LOG.info(f"  {model_name}:")
            LOG.info(f"    R² Score: {r2:.3f}")
            LOG.info(f"    RMSE: {np.sqrt(mse):.0f}")
            LOG.info(f"    MAE: {mae:.0f}")
            
            # Feature importance for Random Forest
            if model_name == 'Random Forest' and hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(features, model.feature_importances_))
                top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
                LOG.info(f"    Top 5 features:")
                for feat, importance in top_features:
                    LOG.info(f"      {feat}: {importance:.3f}")
            
        except Exception as e:
            LOG.warning(f"  Error with {model_name}: {e}")
    
    results[set_name] = set_results

return results
```

def create_visualizations(data, correlations, econ_correlations, output_dir):
“”“Create visualizations of the analysis”””

```
LOG.info("=== CREATING VISUALIZATIONS ===")

# Create output directory
output_dir = Path(output_dir)
output_dir.mkdir(exist_ok=True)

# 1. Mail type correlations heatmap
mail_corr_data = []
for mail_type, corr_data in correlations.items():
    if mail_type != 'total_mail':
        mail_corr_data.append({
            'Mail Type': mail_type,
            'Same Day': corr_data['same_day'],
            'Lag 1 Day': corr_data['lag_1_day'],
            'Lag 2 Days': corr_data['lag_2_day']
        })

if mail_corr_data:
    df_mail_corr = pd.DataFrame(mail_corr_data)
    df_mail_corr.set_index('Mail Type', inplace=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_mail_corr, annot=True, cmap='RdBu_r', center=0, 
               fmt='.3f', cbar_kws={'label': 'Correlation'})
    plt.title('Mail Type → Call Volume Correlations')
    plt.tight_layout()
    plt.savefig(output_dir / 'mail_correlations_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    LOG.info("Saved: mail_correlations_heatmap.png")

# 2. Economic indicators correlations
if econ_correlations:
    econ_corr_data = []
    for econ_indicator, corr_data in econ_correlations.items():
        econ_corr_data.append({
            'Economic Indicator': econ_indicator,
            'Same Day': corr_data['same_day'],
            'Lag 1 Day': corr_data['lag_1_day'],
            'Rolling 5D': corr_data['rolling_5d']
        })
    
    df_econ_corr = pd.DataFrame(econ_corr_data)
    df_econ_corr.set_index('Economic Indicator', inplace=True)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_econ_corr, annot=True, cmap='RdBu_r', center=0, 
               fmt='.3f', cbar_kws={'label': 'Correlation'})
    plt.title('Economic Indicator → Call Volume Correlations')
    plt.tight_layout()
    plt.savefig(output_dir / 'economic_correlations_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    LOG.info("Saved: economic_correlations_heatmap.png")

# 3. Time series plot of calls vs top predictors
available_mail_types = [t for t in CFG["top_mail_types"] if t in data.columns]
if available_mail_types:
    # Find best correlating mail type
    best_mail_type = max(correlations.keys(), 
                       key=lambda x: abs(correlations[x]['same_day']) if x != 'total_mail' else 0)
    
    if best_mail_type != 'total_mail':
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot calls
        ax1.plot(data.index, data['calls_total'], label='Daily Calls', color='blue', alpha=0.7)
        ax1.set_ylabel('Call Volume')
        ax1.set_title('Call Volume vs Best Mail Predictor')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot best mail type
        ax2.plot(data.index, data[best_mail_type], label=f'{best_mail_type} Volume', 
                color='red', alpha=0.7)
        ax2.set_ylabel('Mail Volume')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / 'calls_vs_best_mail_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()
        LOG.info("Saved: calls_vs_best_mail_timeseries.png")
```

def generate_summary_report(correlations, econ_correlations, model_results, output_dir):
“”“Generate a comprehensive summary report”””

```
LOG.info("=== GENERATING SUMMARY REPORT ===")

report = {
    'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'mail_correlations': correlations,
    'economic_correlations': econ_correlations,
    'model_results': model_results
}

# Save JSON report
output_dir = Path(output_dir)
with open(output_dir / 'analysis_report.json', 'w') as f:
    json.dump(report, f, indent=2, default=str)

# Generate text summary
summary_lines = []
summary_lines.append("MAIL + ECONOMIC → CALL VOLUME ANALYSIS SUMMARY")
summary_lines.append("=" * 60)
summary_lines.append(f"Analysis Date: {report['analysis_date']}")
summary_lines.append("")

# Mail correlations summary
summary_lines.append("BEST MAIL PREDICTORS:")
if correlations:
    # Sort by absolute correlation strength
    sorted_mail = sorted(correlations.items(), 
                       key=lambda x: abs(x[1]['same_day']) if x[0] != 'total_mail' else 0, 
                       reverse=True)
    
    for mail_type, corr_data in sorted_mail[:5]:  # Top 5
        if mail_type != 'total_mail':
            summary_lines.append(f"  {mail_type}: {corr_data['same_day']:.3f} (same day)")

# Economic correlations summary
summary_lines.append("")
summary_lines.append("BEST ECONOMIC PREDICTORS:")
if econ_correlations:
    sorted_econ = sorted(econ_correlations.items(), 
                       key=lambda x: abs(x[1]['same_day']), 
                       reverse=True)
    
    for econ_indicator, corr_data in sorted_econ:
        summary_lines.append(f"  {econ_indicator}: {corr_data['same_day']:.3f} (same day)")
else:
    summary_lines.append("  No economic data available")

# Model results summary
summary_lines.append("")
summary_lines.append("MODEL PERFORMANCE:")
for feature_set, models in model_results.items():
    summary_lines.append(f"  {feature_set}:")
    for model_name, metrics in models.items():
        summary_lines.append(f"    {model_name}: R²={metrics['r2']:.3f}, MAE={metrics['mae']:.0f}")

# Save text summary
with open(output_dir / 'analysis_summary.txt', 'w') as f:
    f.write('\n'.join(summary_lines))

LOG.info("Saved: analysis_report.json")
LOG.info("Saved: analysis_summary.txt")

# Print key findings
LOG.info("=== KEY FINDINGS ===")
for line in summary_lines:
    if line.startswith("BEST") or line.startswith("MODEL") or line.startswith("  "):
        LOG.info(line)
```

def main():
“”“Main analysis function”””

```
LOG.info("=== MAIL + ECONOMIC → CALL VOLUME ANALYSIS ===")

# Create output directory
output_dir = Path(CFG["output_dir"])
output_dir.mkdir(exist_ok=True)

# Load all data
data, econ_data = load_all_data()

# Analyze correlations
correlations = analyze_mail_call_correlations(data)
econ_correlations = analyze_economic_call_correlations(data, econ_data)

# Test predictive models
model_results = test_predictive_models(data, econ_data)

# Create visualizations
create_visualizations(data, correlations, econ_correlations, output_dir)

# Generate summary report
generate_summary_report(correlations, econ_correlations, model_results, output_dir)

LOG.info("=== ANALYSIS COMPLETE ===")
LOG.info("Results saved to: %s", output_dir.resolve())
```

if **name** == “**main**”:
main()