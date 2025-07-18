#!/usr/bin/env python

# mail_type_optimization.py

# =========================================================

# Find optimal balance between mail type coverage and accuracy

# Test different mail type selection strategies

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
import holidays
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

warnings.filterwarnings(‘ignore’)

LOG = logging.getLogger(“mail_optimization”)
logging.basicConfig(
level=logging.INFO,
format=”%(asctime)s | mail_opt | %(levelname)s | %(message)s”,
handlers=[logging.StreamHandler(sys.stdout)]
)

CFG = {
“quantiles”: [0.1, 0.25, 0.5, 0.75, 0.9],
“bootstrap_samples”: 30,
“output_dir”: “mail_type_optimization”
}

def _to_date(s):
return pd.to_datetime(s, errors=“coerce”).dt.date

def _find_file(candidates):
for p in candidates:
path = Path(p)
if path.exists():
return path
raise FileNotFoundError(f”None found: {candidates}”)

def load_mail_call_data():
“”“Load data and create mail->calls relationship dataset”””

```
# Load mail
mail_path = _find_file(["mail.csv", "data/mail.csv"])
mail = pd.read_csv(mail_path)
mail.columns = [c.lower().strip() for c in mail.columns]
mail["mail_date"] = _to_date(mail["mail_date"])
mail = mail.dropna(subset=["mail_date"])

# Load calls
vol_path = _find_file(["callvolumes.csv", "data/callvolumes.csv"])
intent_path = _find_file(["callintent.csv", "data/callintent.csv", "callintetn.csv"])

# Process volumes
df_vol = pd.read_csv(vol_path)
df_vol.columns = [c.lower().strip() for c in df_vol.columns]
dcol_v = next(c for c in df_vol.columns if "date" in c)
df_vol[dcol_v] = _to_date(df_vol[dcol_v])
vol_daily = df_vol.groupby(dcol_v)[df_vol.columns.difference([dcol_v])[0]].sum()

# Process intent
df_int = pd.read_csv(intent_path)
df_int.columns = [c.lower().strip() for c in df_int.columns]
dcol_i = next(c for c in df_int.columns if "date" in c or "conversationstart" in c)
df_int[dcol_i] = _to_date(df_int[dcol_i])
int_daily = df_int.groupby(dcol_i).size()

# Scale and combine
overlap = vol_daily.index.intersection(int_daily.index)
if len(overlap) >= 5:
    scale = int_daily.loc[overlap].mean() / vol_daily.loc[overlap].mean()
    vol_daily *= scale
calls_total = vol_daily.combine_first(int_daily).sort_index()

# Aggregate mail daily
mail_daily = (mail.groupby(["mail_date", "mail_type"], as_index=False)["mail_volume"].sum()
               .pivot(index="mail_date", columns="mail_type", values="mail_volume").fillna(0))

mail_daily.index = pd.to_datetime(mail_daily.index)
calls_total.index = pd.to_datetime(calls_total.index)

# Business days only
us_holidays = holidays.US()
biz_mask = (~mail_daily.index.weekday.isin([5, 6])) & (~mail_daily.index.isin(us_holidays))
mail_daily = mail_daily.loc[biz_mask]
calls_total = calls_total.loc[calls_total.index.isin(mail_daily.index)]

daily = mail_daily.join(calls_total.rename("calls_total"), how="inner")

LOG.info("Daily mail-calls data: %s", daily.shape)
return daily
```

def analyze_mail_type_correlations(daily):
“”“Analyze correlation and business metrics for each mail type”””

```
LOG.info("=== ANALYZING ALL MAIL TYPES ===")

# Get call volumes for correlation
calls = daily["calls_total"]

# Analyze each mail type
mail_analysis = {}

for mail_type in daily.columns:
    if mail_type == "calls_total":
        continue
    
    mail_volumes = daily[mail_type]
    
    # Basic statistics
    total_volume = mail_volumes.sum()
    avg_daily_volume = mail_volumes.mean()
    max_daily_volume = mail_volumes.max()
    frequency = (mail_volumes > 0).mean() * 100  # % of days with this mail type
    
    # Correlation with calls
    correlation = mail_volumes.corr(calls)
    
    # Lag correlation (mail today -> calls tomorrow)
    if len(mail_volumes) > 1:
        lag_correlation = mail_volumes[:-1].corr(calls[1:])
    else:
        lag_correlation = correlation
    
    # Variability
    cv = mail_volumes.std() / (mail_volumes.mean() + 1)  # Coefficient of variation
    
    # Business impact estimate
    # Days with high mail vs low mail
    high_mail_days = mail_volumes > mail_volumes.quantile(0.75)
    low_mail_days = mail_volumes < mail_volumes.quantile(0.25)
    
    if high_mail_days.sum() > 0 and low_mail_days.sum() > 0:
        high_mail_calls = calls[high_mail_days].mean()
        low_mail_calls = calls[low_mail_days].mean()
        call_impact = high_mail_calls - low_mail_calls
    else:
        call_impact = 0
    
    mail_analysis[mail_type] = {
        "total_volume": total_volume,
        "avg_daily_volume": avg_daily_volume,
        "max_daily_volume": max_daily_volume,
        "frequency_pct": frequency,
        "correlation": correlation,
        "lag_correlation": lag_correlation,
        "best_correlation": max(abs(correlation), abs(lag_correlation)),
        "coefficient_variation": cv,
        "call_impact": call_impact,
        "volume_rank": 0,  # Will be filled later
        "correlation_rank": 0  # Will be filled later
    }

# Rank by volume and correlation
sorted_by_volume = sorted(mail_analysis.items(), key=lambda x: x[1]["total_volume"], reverse=True)
sorted_by_correlation = sorted(mail_analysis.items(), key=lambda x: x[1]["best_correlation"], reverse=True)

# Assign ranks
for rank, (mail_type, _) in enumerate(sorted_by_volume, 1):
    mail_analysis[mail_type]["volume_rank"] = rank

for rank, (mail_type, _) in enumerate(sorted_by_correlation, 1):
    mail_analysis[mail_type]["correlation_rank"] = rank

# Log top mail types by different criteria
LOG.info("\nTop 15 mail types by CORRELATION:")
for i, (mail_type, data) in enumerate(sorted_by_correlation[:15], 1):
    LOG.info("%2d. %s: %.3f (vol: %d, freq: %.1f%%)", 
            i, mail_type, data["best_correlation"], 
            int(data["total_volume"]), data["frequency_pct"])

LOG.info("\nTop 15 mail types by VOLUME:")
for i, (mail_type, data) in enumerate(sorted_by_volume[:15], 1):
    LOG.info("%2d. %s: %d (corr: %.3f, freq: %.1f%%)", 
            i, mail_type, int(data["total_volume"]), 
            data["best_correlation"], data["frequency_pct"])

return mail_analysis, sorted_by_correlation, sorted_by_volume
```

def create_mail_features(daily, selected_mail_types):
“”“Create features using only selected mail types”””

```
features_list = []
targets_list = []

# For each day, create features from THAT day's mail to predict NEXT day's calls
for i in range(len(daily) - 1):
    current_day = daily.iloc[i]
    next_day = daily.iloc[i + 1]
    
    feature_row = {}
    
    # Mail volumes for selected types only
    total_mail = 0
    for mail_type in selected_mail_types:
        if mail_type in daily.columns:
            volume = current_day[mail_type]
            feature_row[f"{mail_type}_volume"] = volume
            total_mail += volume
    
    # Total mail volume from selected types
    feature_row["total_mail_volume"] = total_mail
    feature_row["log_total_mail_volume"] = np.log1p(total_mail)
    
    # Mail volume percentiles (relative to historical)
    if i > 10:
        mail_history = pd.Series([
            sum(daily.iloc[j][mail_type] for mail_type in selected_mail_types if mail_type in daily.columns)
            for j in range(i + 1)
        ])
        feature_row["mail_percentile"] = (mail_history <= total_mail).mean()
    else:
        feature_row["mail_percentile"] = 0.5
    
    # Date features
    current_date = daily.index[i]
    feature_row["weekday"] = current_date.weekday()
    feature_row["month"] = current_date.month
    feature_row["is_month_end"] = 1 if current_date.day > 25 else 0
    feature_row["is_holiday_week"] = 1 if current_date in holidays.US() else 0
    
    # Recent call volume context (baseline)
    recent_calls = daily["calls_total"].iloc[max(0, i-5):i+1]
    feature_row["recent_calls_avg"] = recent_calls.mean()
    feature_row["recent_calls_trend"] = recent_calls.diff().mean()
    
    # Target: next day's calls
    target = next_day["calls_total"]
    
    features_list.append(feature_row)
    targets_list.append(target)

# Convert to DataFrames
X = pd.DataFrame(features_list)
y = pd.Series(targets_list)

# Clean
X = X.fillna(0)

return X, y
```

def evaluate_mail_type_selection(daily, selected_mail_types, selection_name):
“”“Evaluate performance with selected mail types”””

```
# Create features
X, y = create_mail_features(daily, selected_mail_types)

# Time series cross-validation
tscv = TimeSeriesSplit(n_splits=3)

scores = []
maes = []

for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    # Use Random Forest (best performer from your tests)
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    score = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    scores.append(score)
    maes.append(mae)

avg_r2 = np.mean(scores)
avg_mae = np.mean(maes)

# Calculate coverage metrics
total_mail_volume = sum(daily[mt].sum() for mt in selected_mail_types if mt in daily.columns)
all_mail_volume = sum(daily[col].sum() for col in daily.columns if col != "calls_total")
coverage_pct = (total_mail_volume / all_mail_volume) * 100

result = {
    "selection_name": selection_name,
    "mail_types": selected_mail_types,
    "num_types": len(selected_mail_types),
    "r2_score": avg_r2,
    "mae": avg_mae,
    "num_features": X.shape[1],
    "coverage_pct": coverage_pct,
    "total_volume": total_mail_volume
}

LOG.info("%s: %d types, MAE=%.0f, R²=%.3f, Coverage=%.1f%%", 
         selection_name, len(selected_mail_types), avg_mae, avg_r2, coverage_pct)

return result
```

def test_selection_strategies(daily, mail_analysis, sorted_by_correlation, sorted_by_volume):
“”“Test different mail type selection strategies”””

```
LOG.info("\n=== TESTING MAIL TYPE SELECTION STRATEGIES ===")

results = []

# Strategy 1: Top N by correlation
LOG.info("\nTesting: Top N by CORRELATION")
for n in [3, 5, 8, 10, 15, 20, 25]:
    top_n_corr = [item[0] for item in sorted_by_correlation[:n]]
    result = evaluate_mail_type_selection(daily, top_n_corr, f"Top {n} by Correlation")
    results.append(result)

# Strategy 2: Top N by volume
LOG.info("\nTesting: Top N by VOLUME")
for n in [3, 5, 8, 10, 15, 20, 25]:
    top_n_vol = [item[0] for item in sorted_by_volume[:n]]
    result = evaluate_mail_type_selection(daily, top_n_vol, f"Top {n} by Volume")
    results.append(result)

# Strategy 3: Hybrid approach - balance correlation and volume
LOG.info("\nTesting: HYBRID APPROACH (correlation + volume)")
for n in [5, 8, 10, 15, 20]:
    # Take top n/2 by correlation and top n/2 by volume
    n_half = n // 2
    top_corr = [item[0] for item in sorted_by_correlation[:n_half]]
    top_vol = [item[0] for item in sorted_by_volume[:n_half]]
    
    # Combine and deduplicate
    hybrid_selection = list(set(top_corr + top_vol))
    
    result = evaluate_mail_type_selection(daily, hybrid_selection, f"Hybrid {n} (corr+vol)")
    results.append(result)

# Strategy 4: Threshold-based selection
LOG.info("\nTesting: THRESHOLD-BASED selection")

# High correlation threshold
for threshold in [0.1, 0.15, 0.2, 0.25, 0.3]:
    high_corr_types = [
        mail_type for mail_type, data in mail_analysis.items()
        if data["best_correlation"] >= threshold
    ]
    if len(high_corr_types) > 0:
        result = evaluate_mail_type_selection(
            daily, high_corr_types, f"Correlation >= {threshold}"
        )
        results.append(result)

# Strategy 5: Frequency + correlation filter
LOG.info("\nTesting: FREQUENCY + CORRELATION filter")

for min_freq in [20, 30, 40, 50]:  # Minimum % of days with mail
    for min_corr in [0.1, 0.15, 0.2]:
        filtered_types = [
            mail_type for mail_type, data in mail_analysis.items()
            if data["frequency_pct"] >= min_freq and data["best_correlation"] >= min_corr
        ]
        if len(filtered_types) > 0:
            result = evaluate_mail_type_selection(
                daily, filtered_types, f"Freq>={min_freq}% & Corr>={min_corr}"
            )
            results.append(result)

return results
```

def find_optimal_selection(results):
“”“Find the optimal mail type selection strategy”””

```
LOG.info("\n=== FINDING OPTIMAL SELECTION ===")

# Sort by different criteria
by_mae = sorted(results, key=lambda x: x["mae"])
by_r2 = sorted(results, key=lambda x: x["r2_score"], reverse=True)

# Create efficiency metric: (1/MAE) * Coverage
# Rewards both low MAE and high coverage
for result in results:
    if result["mae"] > 0:
        result["efficiency"] = (1 / result["mae"]) * result["coverage_pct"]
    else:
        result["efficiency"] = 0

by_efficiency = sorted(results, key=lambda x: x["efficiency"], reverse=True)

LOG.info("\nTOP 10 BY MAE (Accuracy):")
for i, result in enumerate(by_mae[:10], 1):
    LOG.info("%2d. %s: MAE=%.0f, R²=%.3f, Coverage=%.1f%%, Types=%d", 
            i, result["selection_name"], result["mae"], result["r2_score"], 
            result["coverage_pct"], result["num_types"])

LOG.info("\nTOP 10 BY EFFICIENCY (Accuracy × Coverage):")
for i, result in enumerate(by_efficiency[:10], 1):
    LOG.info("%2d. %s: Efficiency=%.4f, MAE=%.0f, Coverage=%.1f%%, Types=%d", 
            i, result["selection_name"], result["efficiency"], result["mae"], 
            result["coverage_pct"], result["num_types"])

# Find the sweet spot
best_mae = by_mae[0]
best_efficiency = by_efficiency[0]

LOG.info("\n=== RECOMMENDATIONS ===")
LOG.info("🎯 BEST ACCURACY: %s", best_mae["selection_name"])
LOG.info("   MAE: %.0f, Coverage: %.1f%%, Types: %d", 
         best_mae["mae"], best_mae["coverage_pct"], best_mae["num_types"])

LOG.info("⚖️ BEST BALANCE: %s", best_efficiency["selection_name"])
LOG.info("   MAE: %.0f, Coverage: %.1f%%, Types: %d", 
         best_efficiency["mae"], best_efficiency["coverage_pct"], best_efficiency["num_types"])

# Look for the point where adding more types doesn't help much
correlation_results = [r for r in results if "by Correlation" in r["selection_name"]]
correlation_results.sort(key=lambda x: x["num_types"])

if len(correlation_results) > 3:
    LOG.info("\n📊 DIMINISHING RETURNS ANALYSIS (Top N by Correlation):")
    for i, result in enumerate(correlation_results):
        improvement = ""
        if i > 0:
            mae_improvement = correlation_results[i-1]["mae"] - result["mae"]
            improvement = f" (Δ MAE: {mae_improvement:+.0f})"
        
        LOG.info("   %d types: MAE=%.0f%s", result["num_types"], result["mae"], improvement)

return best_mae, best_efficiency, by_mae, by_efficiency
```

def create_final_optimized_model(daily, optimal_mail_types):
“”“Create final model with optimal mail type selection”””

```
LOG.info("\n=== CREATING FINAL OPTIMIZED MODEL ===")
LOG.info("Using mail types: %s", optimal_mail_types)

# Create features
X, y = create_mail_features(daily, optimal_mail_types)

# Split for final training (80/20 like original)
split_point = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]

models = {}

# Train quantile models
for quantile in CFG["quantiles"]:
    LOG.info("  Training %d%% quantile model...", int(quantile * 100))
    
    model = QuantileRegressor(quantile=quantile, alpha=0.1, solver='highs')
    model.fit(X_train, y_train)
    
    # Validate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    models[f"quantile_{quantile}"] = model
    LOG.info("    Validation MAE: %.0f", mae)

# Train bootstrap ensemble
LOG.info("  Training bootstrap ensemble...")
bootstrap_models = []

for i in range(CFG["bootstrap_samples"]):
    sample_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_boot = X_train.iloc[sample_idx]
    y_boot = y_train.iloc[sample_idx]
    
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=6,
        min_samples_leaf=3,
        random_state=i
    )
    model.fit(X_boot, y_boot)
    bootstrap_models.append(model)

models["bootstrap_ensemble"] = bootstrap_models
models["feature_columns"] = list(X.columns)
models["selected_mail_types"] = optimal_mail_types

LOG.info("Final optimized model trained with %d features from %d mail types", 
         X.shape[1], len(optimal_mail_types))

return models
```

def main():
output_dir = Path(CFG[“output_dir”])
output_dir.mkdir(exist_ok=True)

```
LOG.info("=== MAIL TYPE SELECTION OPTIMIZATION ===")

# Load data
daily = load_mail_call_data()

# Analyze all mail types
mail_analysis, sorted_by_correlation, sorted_by_volume = analyze_mail_type_correlations(daily)

# Test different selection strategies
results = test_selection_strategies(daily, mail_analysis, sorted_by_correlation, sorted_by_volume)

# Find optimal selection
best_mae, best_efficiency, by_mae, by_efficiency = find_optimal_selection(results)

# Create final model with optimal selection
optimal_types = best_efficiency["mail_types"]  # Use balanced approach
final_models = create_final_optimized_model(daily, optimal_types)

# Save everything
joblib.dump(final_models, output_dir / "optimized_mail_models.pkl")

# Save analysis results
save_data = {
    "mail_analysis": {k: {**v, "total_volume": float(v["total_volume"])} for k, v in mail_analysis.items()},
    "selection_results": results,
    "recommendations": {
        "best_accuracy": best_mae,
        "best_balance": best_efficiency
    }
}

with open(output_dir / "mail_type_optimization_results.json", "w") as f:
    json.dump(save_data, f, indent=2, default=str)

LOG.info("\n=== OPTIMIZATION COMPLETE ===")
LOG.info("Results saved to: %s", output_dir.resolve())
LOG.info("\n=== FINAL RECOMMENDATIONS ===")
LOG.info("🎯 For MAXIMUM ACCURACY: Use %s", best_mae["selection_name"])
LOG.info("⚖️ For BALANCED APPROACH: Use %s", best_efficiency["selection_name"])
LOG.info("📊 Your original 10 types likely had: %.1f%% coverage", 
         next((r["coverage_pct"] for r in results if "Top 10 by" in r["selection_name"]), 0))
```

if **name** == “**main**”:
main()