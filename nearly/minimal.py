# simple_mail_forecast.py

# ==========================================================

# ULTRA-SIMPLE approach based on debug findings

# Key insight: Use only TOP correlated mail types + simple features

# ==========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import holidays
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

# Config

CFG = {
“top_mail_types”: [
“Reject_Ltrs”, “Cheque 1099”, “Exercise_Converted”,
“SOI_Confirms”, “Exch_chks”, “ACH_Debit_Enrollment”,
“Transfer”, “COA”, “NOTC_WITHDRAW”, “Repl_Chks”
],
“forecast_horizons”: [1, 3, 7, 14],
“output_dir”: “dist_simple”
}

def _to_date(s):
return pd.to_datetime(s, errors=“coerce”).dt.date

def _find_file(candidates):
for p in candidates:
if Path(p).exists():
return Path(p)
return None

def load_data():
“”“Load and prepare data focusing on top mail types only”””

```
# Load mail
mail_path = _find_file(["mail.csv", "data/mail.csv"])
mail = pd.read_csv(mail_path)
mail.columns = [c.lower().strip() for c in mail.columns]
mail["mail_date"] = _to_date(mail["mail_date"])
mail = mail.dropna(subset=["mail_date"])

# Load calls
calls_vol_path = _find_file(["callvolumes.csv", "data/callvolumes.csv"])
calls_int_path = _find_file(["callintent.csv", "data/callintent.csv", "callintetn.csv"])

# Process volumes
df_vol = pd.read_csv(calls_vol_path)
df_vol.columns = [c.lower().strip() for c in df_vol.columns]
dcol_v = next(c for c in df_vol.columns if "date" in c)
df_vol[dcol_v] = _to_date(df_vol[dcol_v])
vol_daily = df_vol.groupby(dcol_v)[df_vol.columns.difference([dcol_v])[0]].sum()

# Process intent
df_int = pd.read_csv(calls_int_path)
df_int.columns = [c.lower().strip() for c in df_int.columns]
dcol_i = next(c for c in df_int.columns if "date" in c or "conversationstart" in c)
df_int[dcol_i] = _to_date(df_int[dcol_i])
int_daily = df_int.groupby(dcol_i).size()

# Scale and combine calls
overlap = vol_daily.index.intersection(int_daily.index)
if len(overlap) >= 5:
    scale = int_daily.loc[overlap].mean() / vol_daily.loc[overlap].mean()
    vol_daily *= scale
calls_total = vol_daily.combine_first(int_daily).sort_index()

return mail, calls_total
```

def create_simple_features(mail, calls_total):
“”“Create ONLY the most important features”””

```
# 1. Aggregate mail by top types only
mail_daily = (mail.groupby(["mail_date", "mail_type"], as_index=False)["mail_volume"].sum()
               .pivot(index="mail_date", columns="mail_type", values="mail_volume").fillna(0))

# 2. Keep only top correlated mail types
available_types = [t for t in CFG["top_mail_types"] if t in mail_daily.columns]
mail_daily = mail_daily[available_types]

# 3. Convert to datetime and align
mail_daily.index = pd.to_datetime(mail_daily.index)
calls_total.index = pd.to_datetime(calls_total.index)

# 4. Business days only
us_holidays = holidays.US()
biz_mask = (~mail_daily.index.weekday.isin([5, 6])) & (~mail_daily.index.isin(us_holidays))
mail_daily = mail_daily.loc[biz_mask]
calls_total = calls_total.loc[calls_total.index.isin(mail_daily.index)]

# 5. Combine
daily = mail_daily.join(calls_total.rename("calls_total"), how="inner")

# 6. Create MINIMAL feature set
features = pd.DataFrame(index=daily.index)

# Top 5 mail types with 1-day lag (best correlation)
for mail_type in available_types[:5]:
    if mail_type in daily.columns:
        features[f"{mail_type}_lag1"] = daily[mail_type].shift(1)

# Total mail features
features["total_mail_lag1"] = daily[available_types].sum(axis=1).shift(1)
features["total_mail_lag2"] = daily[available_types].sum(axis=1).shift(2)

# Log-transformed totals (debug showed this helps correlation)
features["log_total_mail_lag1"] = np.log1p(daily[available_types].sum(axis=1)).shift(1)

# Basic time features
features["weekday"] = daily.index.dayofweek
features["month"] = daily.index.month
features["is_month_end"] = (daily.index.day > 25).astype(int)

# Target (log-transformed since debug showed it helps)
target = np.log1p(daily["calls_total"])

# Clean data
features = features.dropna()
target = target.loc[features.index]

print(f"Final features: {features.shape[1]} features, {features.shape[0]} samples")
print(f"Feature list: {list(features.columns)}")

return features, target, daily["calls_total"]
```

def create_targets(target, horizons):
“”“Create multi-horizon targets”””
targets = pd.DataFrame(index=target.index)

```
for h in horizons:
    # Use shift(-1) to predict tomorrow+next h-1 days
    targets[f"calls_{h}d"] = target.shift(-1).rolling(h).sum()

return targets.dropna()
```

def train_simple_model(X, y, horizon):
“”“Train a simple model with proper validation”””

```
if len(X) < 30:
    print(f"  Not enough data for horizon {horizon}")
    return None, {}

# Time series cross-validation
cv = TimeSeriesSplit(n_splits=3)

# Test both Ridge and Random Forest
models = {
    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=10.0))  # Higher alpha for regularization
    ]),
    "RandomForest": RandomForestRegressor(
        n_estimators=100, 
        max_depth=5,  # Shallow trees
        min_samples_leaf=5,  # Prevent overfitting
        random_state=42
    )
}

results = {}

for name, model in models.items():
    r2_scores = []
    rmse_scores = []
    
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        r2_scores.append(r2_score(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    
    results[name] = {
        "R2": np.mean(r2_scores),
        "R2_std": np.std(r2_scores),
        "RMSE": np.mean(rmse_scores),
        "model": model
    }
    
    print(f"  {name}: R² = {np.mean(r2_scores):.3f}±{np.std(r2_scores):.3f}")

# Return best model
best_name = max(results.keys(), key=lambda x: results[x]["R2"])
best_model = results[best_name]["model"]

# Fit on all data
best_model.fit(X, y)

return best_model, results
```

def main():
print(”=”*60)
print(“ULTRA-SIMPLE MAIL→CALLS FORECAST”)
print(”=”*60)

```
# Setup
output_dir = Path(CFG["output_dir"])
output_dir.mkdir(exist_ok=True)

# 1. Load data
print("\n1. Loading data...")
mail, calls_total = load_data()

# 2. Create simple features
print("\n2. Creating simple features...")
X, y_log, y_raw = create_simple_features(mail, calls_total)

# 3. Create targets
print("\n3. Creating multi-horizon targets...")
targets = create_targets(y_log, CFG["forecast_horizons"])

# Align data
common_idx = X.index.intersection(targets.index)
X = X.loc[common_idx]
targets = targets.loc[common_idx]
y_raw = y_raw.loc[common_idx]

print(f"Final dataset: {X.shape[0]} samples")

# 4. Train models
print("\n4. Training models...")
all_results = {}
trained_models = {}

for horizon in CFG["forecast_horizons"]:
    print(f"\nHorizon {horizon}d:")
    target_col = f"calls_{horizon}d"
    
    if target_col in targets.columns:
        model, results = train_simple_model(X, targets[target_col], horizon)
        if model is not None:
            all_results[f"{horizon}d"] = results
            trained_models[horizon] = model
            
            # Save model
            joblib.dump(model, output_dir / f"simple_model_{horizon}d.pkl")

# 5. Generate forecasts
print("\n5. Generating forecasts...")
forecasts = {}

for horizon, model in trained_models.items():
    # Simple future features: repeat last observation
    last_features = X.iloc[[-1]]
    future_pred_log = model.predict(last_features)[0]
    future_pred_raw = np.expm1(future_pred_log)  # Convert back from log
    
    forecasts[horizon] = future_pred_raw

# 6. Create visualization
print("\n6. Creating visualization...")

plt.figure(figsize=(15, 10))

# Plot 1: Historical data
plt.subplot(2, 2, 1)
plt.plot(y_raw.index[-60:], y_raw.iloc[-60:], 'o-', label='Historical Calls', alpha=0.7)
plt.title('Historical Daily Calls (Last 60 Days)')
plt.ylabel('Calls')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Top mail types
plt.subplot(2, 2, 2)
top_3_mail = X.columns[:3]  # First 3 features should be top mail types
for col in top_3_mail:
    if 'lag1' in col:
        mail_type = col.replace('_lag1', '')
        plt.plot(X.index[-60:], X[col].iloc[-60:], label=mail_type, alpha=0.7)
plt.title('Top Mail Types (Last 60 Days)')
plt.ylabel('Mail Volume')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Forecast bar chart
plt.subplot(2, 2, 3)
horizons = list(forecasts.keys())
predictions = list(forecasts.values())

bars = plt.bar(horizons, predictions, color=['skyblue', 'lightgreen', 'orange', 'pink'])
plt.title('Call Volume Forecasts')
plt.ylabel('Predicted Calls')
plt.xlabel('Forecast Horizon (days)')

# Add value labels on bars
for bar, pred in zip(bars, predictions):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(predictions)*0.01,
            f'{pred:.0f}', ha='center', va='bottom')

plt.grid(True, alpha=0.3)

# Plot 4: Model performance
plt.subplot(2, 2, 4)
if all_results:
    horizons_perf = []
    r2_scores = []
    
    for horizon_key, models in all_results.items():
        horizon_num = int(horizon_key.replace('d', ''))
        best_model = max(models.keys(), key=lambda x: models[x]["R2"])
        r2_score = models[best_model]["R2"]
        
        horizons_perf.append(horizon_num)
        r2_scores.append(r2_score)
    
    bars = plt.bar(horizons_perf, r2_scores, color='lightcoral')
    plt.title('Model Performance (R² Score)')
    plt.ylabel('R² Score')
    plt.xlabel('Forecast Horizon (days)')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar, score in zip(bars, r2_scores):
        plt.text(bar.get_x() + bar.get_width()/2, 
                bar.get_height() + (max(r2_scores) - min(r2_scores))*0.02,
                f'{score:.2f}', ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "simple_forecast_results.png", dpi=300, bbox_inches='tight')
plt.show()

# 7. Save results
print("\n7. Saving results...")

# Save metrics
with open(output_dir / "simple_results.json", "w") as f:
    json.dump(all_results, f, indent=2, default=str)

# Create summary
summary = {
    "forecasts": forecasts,
    "feature_count": X.shape[1],
    "sample_count": X.shape[0],
    "top_mail_types": CFG["top_mail_types"][:5]
}

with open(output_dir / "forecast_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

# 8. Results summary
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)

print(f"Features used: {X.shape[1]}")
print(f"Training samples: {X.shape[0]}")
print(f"Top mail types: {CFG['top_mail_types'][:5]}")

print("\nForecasts:")
for horizon, pred in forecasts.items():
    print(f"  {horizon} days: {pred:.0f} calls")

print("\nModel Performance:")
for horizon_key, models in all_results.items():
    best_model = max(models.keys(), key=lambda x: models[x]["R2"])
    r2_score = models[best_model]["R2"]
    print(f"  {horizon_key}: {best_model} (R² = {r2_score:.3f})")

# Success assessment
avg_r2 = np.mean([max(models.values(), key=lambda x: x["R2"])["R2"] for models in all_results.values()])

if avg_r2 > 0.1:
    print("\n✅ SUCCESS: Models show positive predictive power!")
elif avg_r2 > 0:
    print("\n⚠️  MARGINAL: Weak but positive predictive power")
else:
    print("\n❌ FAILED: No predictive power found")

print(f"\nResults saved to: {output_dir.resolve()}")
```

if **name** == “**main**”:
main()