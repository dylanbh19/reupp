# “””
Mail-to-Calls Forecasting Pipeline - Focused & Accurate

A highly accurate pipeline that predicts call volumes from mail campaigns.

INPUT: Mail type volumes + Date
OUTPUT: Predicted call volumes over multiple time horizons (1d, 3d, 7d, 14d, 30d)

Key Features:

- Mail-focused feature engineering (no intent data)
- Multiple prediction horizons
- High accuracy through proper feature selection
- Beautiful forecast visualizations
  “””

# Force headless backend for production

import matplotlib
matplotlib.use(“Agg”)

import json, logging, os, sys, warnings, tempfile, hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import itertools

import joblib, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm
import holidays, yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline

warnings.filterwarnings(‘ignore’)

# —————————————————————————–

# Configuration - Focused on Mail→Calls Prediction

# —————————————————————————–

RUN_ID = datetime.utcnow().strftime(”%Y%m%dT%H%M%SZ”)
PIPELINE_VERSION = “v7.0-MailFocused”

CONFIG = {
“run_id”: RUN_ID,
“version”: PIPELINE_VERSION,

```
# Data files
"data_files": {
    "mail": ["mail.csv", "data/mail.csv"],
    "calls": ["callvolumes.csv", "data/callvolumes.csv"]
},

"output_dir": "dist",

# Prediction horizons (days)
"forecast_horizons": [1, 3, 7, 14, 30],

# Feature engineering - FOCUSED
"mail_lags": [1, 2, 3, 7, 14],           # Mail volume lags
"rolling_windows": [3, 7, 14],           # Rolling averages
"max_features": 25,                      # Strict feature limit

# Model settings
"ts_splits": 3,                          # Reduced for small data
"random_state": 42,
"hyper_opt_iters": 10,                   # Faster iteration

# Performance thresholds
"thresholds": {
    "min_r2": 0.5,                       # Higher standard
    "max_mape": 0.20,                    # 20% MAPE target
}
```

}

# —————————————————————————–

# Logging setup

# —————————————————————————–

logging.basicConfig(
level=logging.INFO,
format=f”%(asctime)s | {RUN_ID} | %(levelname)s | %(message)s”,
handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(“mail_forecast”)

# —————————————————————————–

# Data Loading & Processing

# —————————————————————————–

def find_data_file(file_patterns: List[str]) -> Optional[Path]:
“”“Auto-detect data file”””
for pattern in file_patterns:
path = Path(pattern)
if path.exists():
logger.info(f”Found: {path}”)
return path
return None

def _smart_to_datetime(s: pd.Series) -> pd.Series:
“”“Robust date parser”””
s1 = pd.to_datetime(s, errors=“coerce”, infer_datetime_format=True, dayfirst=False)
if s1.isna().mean() <= 0.2:
return s1

```
s2 = pd.to_datetime(s, errors="coerce", infer_datetime_format=True, dayfirst=True)
s1 = s1.combine_first(s2)
if s1.isna().mean() <= 0.2:
    return s1

s3 = pd.to_datetime(s.str.slice(0, 10), errors="coerce")
return s1.combine_first(s3)
```

def load_mail(file_patterns: List[str]) -> pd.DataFrame:
“”“Load mail data”””
path = find_data_file(file_patterns)
if path is None:
raise FileNotFoundError(f”Mail file not found: {file_patterns}”)

```
df = pd.read_csv(path)
df.columns = [c.lower().strip() for c in df.columns]

if "mail_date" not in df.columns:
    raise ValueError(f"Required column 'mail_date' missing: {list(df.columns)}")
if "mail_volume" not in df.columns:
    raise ValueError(f"Required column 'mail_volume' missing: {list(df.columns)}")

df["mail_date"] = _smart_to_datetime(df["mail_date"]).dt.date
df = df.dropna(subset=["mail_date"])

logger.info(f"Loaded {len(df)} mail records")
return df
```

def load_calls(file_patterns: List[str]) -> pd.DataFrame:
“”“Load call data”””
path = find_data_file(file_patterns)
if path is None:
raise FileNotFoundError(f”Calls file not found: {file_patterns}”)

```
df = pd.read_csv(path)
df.columns = [c.lower().strip() for c in df.columns]

date_candidates = {"date", "conversationstart", "conversation_start", "call_date"}
dcol = next((c for c in df.columns if c in date_candidates), None)

if dcol is None:
    raise ValueError(f"No date column found: {list(df.columns)}")

df[dcol] = _smart_to_datetime(df[dcol]).dt.date
df = df.dropna(subset=[dcol]).rename(columns={dcol: "date"})

logger.info(f"Loaded {len(df)} call records")
return df
```

# —————————————————————————–

# Feature Engineering - Mail Focused

# —————————————————————————–

def create_mail_features(daily_data: pd.DataFrame, target_col: str) -> pd.DataFrame:
“””
Create focused mail-based features for accurate prediction
“””
df = daily_data.copy()
df = df.sort_index()

```
logger.info("Creating mail-focused features...")

# 1. Mail volume lags (how much mail was sent X days ago)
mail_cols = [c for c in df.columns if c not in [target_col]]

for col in mail_cols:
    for lag in CONFIG["mail_lags"]:
        df[f"{col}_lag{lag}"] = df[col].shift(lag)

# 2. Rolling sums of mail (cumulative effect)
for col in mail_cols:
    for window in CONFIG["rolling_windows"]:
        df[f"{col}_sum{window}d"] = df[col].shift(1).rolling(window).sum()
        df[f"{col}_avg{window}d"] = df[col].shift(1).rolling(window).mean()

# 3. Total mail volume features
df["total_mail"] = df[mail_cols].sum(axis=1)
df["total_mail_lag1"] = df["total_mail"].shift(1)
df["total_mail_lag7"] = df["total_mail"].shift(7)

for window in CONFIG["rolling_windows"]:
    df[f"total_mail_sum{window}d"] = df["total_mail"].shift(1).rolling(window).sum()

# 4. Calendar features (business context)
df["weekday"] = df.index.dayofweek
df["is_monday"] = (df.index.dayofweek == 0).astype(int)
df["is_friday"] = (df.index.dayofweek == 4).astype(int)
df["month"] = df.index.month
df["quarter"] = df.index.quarter

# 5. Holiday effects
us_holidays = holidays.US()
df["is_holiday"] = df.index.to_series().apply(lambda x: x in us_holidays).astype(int)
df["days_to_holiday"] = df.index.to_series().apply(
    lambda x: min([abs((h - x).days) for h in us_holidays.keys() 
                  if abs((h - x).days) <= 5] or [99])
)

logger.info(f"Created {df.shape[1] - daily_data.shape[1]} mail-focused features")
return df
```

def select_best_features(X: pd.DataFrame, y: pd.Series, max_features: int) -> pd.DataFrame:
“””
Select the most predictive features to avoid overfitting
“””
logger.info(f”Selecting top {max_features} features from {X.shape[1]} candidates…”)

```
# Remove features with zero variance
X_clean = X.loc[:, X.var() > 0]

# Select best features using F-statistic
selector = SelectKBest(score_func=f_regression, k=min(max_features, X_clean.shape[1]))
X_selected = selector.fit_transform(X_clean, y)

selected_features = X_clean.columns[selector.get_support()]
X_final = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

# Log top features
scores = pd.Series(selector.scores_[selector.get_support()], index=selected_features)
scores = scores.sort_values(ascending=False)

logger.info("Top 10 features:")
for i, (feature, score) in enumerate(scores.head(10).items()):
    logger.info(f"  {i+1}. {feature}: {score:.2f}")

return X_final
```

# —————————————————————————–

# Multi-Horizon Prediction Setup

# —————————————————————————–

def create_multi_horizon_targets(daily_calls: pd.Series, horizons: List[int]) -> pd.DataFrame:
“””
Create multiple prediction targets for different time horizons
“””
targets = pd.DataFrame(index=daily_calls.index)

```
for horizon in horizons:
    # Sum of calls over next N days
    targets[f"calls_{horizon}d"] = daily_calls.rolling(window=horizon).sum().shift(-horizon)

logger.info(f"Created targets for horizons: {horizons} days")
return targets.dropna()
```

# —————————————————————————–

# Model Training - Accuracy Focused

# —————————————————————————–

def make_search_spaces() -> Dict[str, Dict]:
“”“Focused hyperparameter spaces for accuracy”””
return {
“Ridge”: {
“regressor__alpha”: Real(0.1, 100.0, prior=“log-uniform”)
},
“ElasticNet”: {
“regressor__alpha”: Real(0.1, 10.0, prior=“log-uniform”),
“regressor__l1_ratio”: Real(0.1, 0.9)
},
“RandomForest”: {
“n_estimators”: Integer(100, 300),
“max_depth”: Integer(5, 15),
“min_samples_leaf”: Integer(2, 10),
“min_samples_split”: Integer(5, 20)
},
“XGBoost”: {
“n_estimators”: Integer(100, 300),
“max_depth”: Integer(3, 8),
“learning_rate”: Real(0.05, 0.3),
“subsample”: Real(0.8, 1.0),
“colsample_bytree”: Real(0.8, 1.0)
}
}

def build_model(name: str) -> object:
“”“Build model with preprocessing pipeline”””
if name == “Ridge”:
return Pipeline([
(“scaler”, StandardScaler()),
(“regressor”, Ridge(random_state=CONFIG[“random_state”]))
])
elif name == “ElasticNet”:
return Pipeline([
(“scaler”, StandardScaler()),
(“regressor”, ElasticNet(random_state=CONFIG[“random_state”], max_iter=2000))
])
elif name == “RandomForest”:
return RandomForestRegressor(random_state=CONFIG[“random_state”], n_jobs=-1)
elif name == “XGBoost”:
return XGBRegressor(random_state=CONFIG[“random_state”], n_jobs=-1,
objective=“reg:squarederror”, verbosity=0)
else:
raise ValueError(f”Unknown model: {name}”)

def train_horizon_model(X: pd.DataFrame, y: pd.Series, horizon: int, output_dir: Path) -> Dict:
“””
Train model for specific prediction horizon
“””
logger.info(f”Training models for {horizon}-day horizon…”)

```
search_spaces = make_search_spaces()
cv = TimeSeriesSplit(n_splits=CONFIG["ts_splits"])
results = {}

for model_name, param_space in search_spaces.items():
    try:
        base_model = build_model(model_name)
        
        # Bayesian optimization
        search = BayesSearchCV(
            base_model, 
            param_space,
            cv=cv,
            n_iter=CONFIG["hyper_opt_iters"],
            scoring="neg_root_mean_squared_error",
            random_state=CONFIG["random_state"],
            n_jobs=-1
        )
        
        search.fit(X, y)
        best_model = search.best_estimator_
        
        # Cross-validation metrics
        cv_scores = []
        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            
            cv_scores.append({
                "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                "mape": mean_absolute_percentage_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred)
            })
        
        # Average metrics
        avg_metrics = {
            "RMSE": np.mean([s["rmse"] for s in cv_scores]),
            "MAPE": np.mean([s["mape"] for s in cv_scores]),
            "R2": np.mean([s["r2"] for s in cv_scores]),
            "best_params": search.best_params_
        }
        
        results[model_name] = avg_metrics
        
        # Save best model
        best_model.fit(X, y)
        joblib.dump(best_model, output_dir / f"model_{model_name}_{horizon}d.pkl")
        
        logger.info(f"  {model_name}: R²={avg_metrics['R2']:.3f}, RMSE={avg_metrics['RMSE']:.0f}, MAPE={avg_metrics['MAPE']:.1%}")
        
    except Exception as e:
        logger.error(f"  {model_name} failed: {e}")
        continue

return results
```

# —————————————————————————–

# Prediction & Visualization

# —————————————————————————–

def create_forecast_visualization(X: pd.DataFrame, models: Dict, horizons: List[int],
recent_calls: pd.Series, output_dir: Path) -> None:
“””
Create beautiful forecast visualization
“””
fig, axes = plt.subplots(len(horizons), 1, figsize=(15, 4*len(horizons)))
if len(horizons) == 1:
axes = [axes]

```
for i, horizon in enumerate(horizons):
    ax = axes[i]
    
    # Get best model for this horizon
    best_model_path = output_dir / f"model_XGBoost_{horizon}d.pkl"
    if not best_model_path.exists():
        # Fallback to any available model
        available_models = list(output_dir.glob(f"model_*_{horizon}d.pkl"))
        if available_models:
            best_model_path = available_models[0]
        else:
            continue
    
    model = joblib.load(best_model_path)
    
    # Make predictions
    predictions = model.predict(X)
    
    # Plot historical calls (last 30 days)
    historical_dates = recent_calls.index[-30:]
    ax.plot(historical_dates, recent_calls.iloc[-30:], 
            'o-', color='steelblue', label='Historical Daily Calls', linewidth=2, markersize=4)
    
    # Plot predictions (next 14 days)
    future_dates = pd.date_range(start=X.index[-1] + timedelta(days=1), periods=14)
    future_preds = predictions[-14:]  # Last 14 predictions
    
    ax.plot(future_dates, future_preds, 
            's-', color='crimson', label=f'{horizon}-Day Forecast', linewidth=2, markersize=6)
    
    # Add confidence bands (simple approach)
    pred_std = np.std(predictions)
    ax.fill_between(future_dates, 
                   future_preds - 1.96*pred_std, 
                   future_preds + 1.96*pred_std,
                   alpha=0.3, color='crimson', label='95% Confidence')
    
    ax.set_title(f'{horizon}-Day Forecast Horizon', fontsize=14, fontweight='bold')
    ax.set_ylabel('Call Volume')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Rotate x-axis labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
plt.savefig(output_dir / "mail_to_calls_forecast.png", dpi=300, bbox_inches='tight')
plt.close()

logger.info("Forecast visualization created")
```

def generate_forecast_table(X: pd.DataFrame, models_dir: Path, horizons: List[int]) -> pd.DataFrame:
“””
Generate forecast summary table
“””
forecasts = []

```
for horizon in horizons:
    # Load best available model
    model_files = list(models_dir.glob(f"model_*_{horizon}d.pkl"))
    if not model_files:
        continue
    
    model = joblib.load(model_files[0])
    
    # Latest prediction
    latest_pred = model.predict(X.iloc[[-1]])[0]
    
    forecasts.append({
        "Horizon": f"{horizon} day{'s' if horizon > 1 else ''}",
        "Predicted_Calls": int(latest_pred),
        "Model": model_files[0].stem.replace(f"model_", "").replace(f"_{horizon}d", "")
    })

return pd.DataFrame(forecasts)
```

# —————————————————————————–

# Main Pipeline

# —————————————————————————–

def build_mail_focused_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
“””
Build focused mail→calls dataset
“””
# Load only mail and calls data
mail = load_mail(CONFIG[“data_files”][“mail”])
calls = load_calls(CONFIG[“data_files”][“calls”])

```
# Aggregate mail daily by type
mail_daily = (
    mail.groupby(["mail_date", "mail_type"], as_index=False)["mail_volume"].sum()
        .rename(columns={"mail_date": "date"})
)

# Aggregate calls daily
calls_daily = calls.groupby("date").size().rename("calls_total")

# Create mail pivot (wide format)
mail_wide = mail_daily.pivot(index="date", columns="mail_type", values="mail_volume").fillna(0)

# Combine on daily level
daily = mail_wide.join(calls_daily, how='inner')
daily.index = pd.to_datetime(daily.index)

# Filter business days only
us_holidays = holidays.US()
business_mask = ~daily.index.weekday.isin([5, 6]) & ~daily.index.isin(us_holidays)
daily = daily[business_mask]

logger.info(f"Daily dataset: {daily.shape[0]} days, {daily.shape[1]-1} mail types")

# Create features
daily_featured = create_mail_features(daily, "calls_total")

# Prepare features and target
X = daily_featured.drop(columns=["calls_total"])
y = daily_featured["calls_total"]

# Feature selection for accuracy
X_selected = select_best_features(X, y, CONFIG["max_features"])

# Create multi-horizon targets
targets = create_multi_horizon_targets(y, CONFIG["forecast_horizons"])

# Align data
common_index = X_selected.index.intersection(targets.index)
X_final = X_selected.loc[common_index]
targets_final = targets.loc[common_index]
y_final = y.loc[common_index]

logger.info(f"Final dataset: {X_final.shape[0]} samples, {X_final.shape[1]} features")

return X_final, targets_final, y_final
```

def main():
“””
Main pipeline for accurate mail→calls forecasting
“””
output_dir = Path(CONFIG[“output_dir”])
output_dir.mkdir(parents=True, exist_ok=True)

```
logger.info("=" * 70)
logger.info("MAIL-TO-CALLS FORECASTING PIPELINE v7.0")
logger.info("=" * 70)
logger.info("Focus: High-accuracy prediction of call volumes from mail campaigns")

try:
    # Build focused dataset
    logger.info("\n[1/4] Building mail-focused dataset...")
    X, targets, daily_calls = build_mail_focused_dataset()
    
    # Save processed data
    X.to_csv(output_dir / "mail_features.csv")
    targets.to_csv(output_dir / "forecast_targets.csv")
    
    # Train models for each horizon
    logger.info("\n[2/4] Training models for each forecast horizon...")
    all_results = {}
    
    for horizon in CONFIG["forecast_horizons"]:
        if f"calls_{horizon}d" in targets.columns:
            results = train_horizon_model(X, targets[f"calls_{horizon}d"], horizon, output_dir)
            all_results[f"{horizon}d"] = results
    
    # Create forecasts
    logger.info("\n[3/4] Generating forecasts...")
    create_forecast_visualization(X, all_results, CONFIG["forecast_horizons"], 
                                daily_calls, output_dir)
    
    # Generate forecast summary
    forecast_table = generate_forecast_table(X, output_dir, CONFIG["forecast_horizons"])
    forecast_table.to_csv(output_dir / "forecast_summary.csv", index=False)
    
    logger.info("\n[4/4] Creating summary report...")
    
    # Performance summary
    logger.info("\n" + "="*50)
    logger.info("FORECAST SUMMARY")
    logger.info("="*50)
    
    print(forecast_table.to_string(index=False))
    
    # Best model per horizon
    logger.info("\nBest Model Performance:")
    for horizon_key, models in all_results.items():
        if models:
            best_model = max(models.keys(), key=lambda x: models[x]["R2"])
            metrics = models[best_model]
            logger.info(f"  {horizon_key}: {best_model} (R²={metrics['R2']:.3f}, MAPE={metrics['MAPE']:.1%})")
    
    logger.info(f"\n✅ All outputs saved to: {output_dir.resolve()}")
    logger.info("📊 View forecast: mail_to_calls_forecast.png")
    logger.info("📋 Summary table: forecast_summary.csv")
    
except Exception as e:
    logger.error(f"Pipeline failed: {e}")
    import traceback
    logger.error(traceback.format_exc())
    raise
```

if **name** == “**main**”:
main()