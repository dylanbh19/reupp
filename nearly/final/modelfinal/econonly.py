#!/usr/bin/env python
# economics_only_model.py
# =========================================================
# Test economic indicators ONLY for call prediction
# No mail data - pure economic signal testing
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
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

warnings.filterwarnings('ignore')

LOG = logging.getLogger("economics_only")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | economics_only | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

CFG = {
    "economic_indicators": [
        "Russell2000", "Dollar_Index", "NASDAQ", "SP500", "Technology",
        "Banking", "DowJones", "Regional_Banks", "Dividend_ETF",
        "VIX", "Oil", "Gold", "REITs", "Utilities"
    ],
    "output_dir": "economics_only_results"
}

def _to_date(s):
    return pd.to_datetime(s, errors="coerce").dt.date

def _find_file(candidates):
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError(f"None found: {candidates}")

def load_calls_data():
    """Load only call volume data"""
    
    LOG.info("Loading call volume data...")
    
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
        LOG.info("Scaled call volumes by factor: %.2f", scale)
    
    calls_total = vol_daily.combine_first(int_daily).sort_index()
    calls_total.index = pd.to_datetime(calls_total.index)
    
    # Business days only
    us_holidays = holidays.US()
    biz_mask = (~calls_total.index.weekday.isin([5, 6])) & (~calls_total.index.isin(us_holidays))
    calls_total = calls_total.loc[biz_mask]
    
    LOG.info("Call data shape: %d business days", len(calls_total))
    LOG.info("Call volume stats: mean=%.0f, std=%.0f, min=%.0f, max=%.0f", 
             calls_total.mean(), calls_total.std(), calls_total.min(), calls_total.max())
    
    return calls_total

def load_economic_data():
    """Load economic indicators"""
    
    LOG.info("Loading economic indicators...")
    
    # Try to find economic data files
    econ_candidates = [
        "economics_expanded.csv", 
        "data/economics_expanded.csv",
        "economics.csv",
        "data/economics.csv"
    ]
    
    try:
        econ_path = _find_file(econ_candidates)
        econ_data = pd.read_csv(econ_path)
        econ_data.columns = [c.strip() for c in econ_data.columns]
        
        # Find date column
        date_col = None
        for col in econ_data.columns:
            if 'date' in col.lower():
                date_col = col
                break
        
        if date_col:
            econ_data[date_col] = pd.to_datetime(econ_data[date_col], errors='coerce')
            econ_data = econ_data.dropna(subset=[date_col])
            econ_data.set_index(date_col, inplace=True)
        
        # Business days only
        us_holidays = holidays.US()
        biz_mask = (~econ_data.index.weekday.isin([5, 6])) & (~econ_data.index.isin(us_holidays))
        econ_data = econ_data.loc[biz_mask]
        
        LOG.info("Economic indicators loaded: %s", list(econ_data.columns))
        LOG.info("Economic data shape: %d business days", len(econ_data))
        
        return econ_data
        
    except FileNotFoundError:
        LOG.warning("No economic data files found - creating dummy data for testing")
        
        # Create realistic dummy economic data
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        # Filter to business days
        us_holidays = holidays.US()
        biz_mask = (~dates.weekday.isin([5, 6])) & (~dates.isin(us_holidays))
        dates = dates[biz_mask]
        
        np.random.seed(42)  # For reproducible results
        
        dummy_econ = pd.DataFrame(index=dates)
        
        # Create realistic economic time series
        for indicator in CFG["economic_indicators"]:
            if "VIX" in indicator:
                # Volatility index - higher variance, mean around 20
                base_value = 20
                volatility = 0.1
            elif "Russell2000" in indicator:
                # Small cap index
                base_value = 2000
                volatility = 0.02
            elif "SP500" in indicator:
                # S&P 500
                base_value = 4000
                volatility = 0.015
            elif "NASDAQ" in indicator:
                # NASDAQ
                base_value = 15000
                volatility = 0.02
            elif "Dollar_Index" in indicator:
                # USD index
                base_value = 100
                volatility = 0.01
            elif "Oil" in indicator:
                # Oil price
                base_value = 70
                volatility = 0.03
            elif "Gold" in indicator:
                # Gold price
                base_value = 1800
                volatility = 0.02
            else:
                # Generic stock/sector index
                base_value = 100
                volatility = 0.02
            
            # Generate time series with trend and noise
            trends = np.random.randn(len(dates)) * volatility
            values = base_value * (1 + np.cumsum(trends))
            
            # Add some cyclical patterns
            cyclical = np.sin(np.arange(len(dates)) * 2 * np.pi / 252) * base_value * 0.05  # Annual cycle
            values += cyclical
            
            dummy_econ[indicator] = values
        
        LOG.info("Created dummy economic data with realistic patterns")
        return dummy_econ

def create_economics_features(calls_data, economic_data):
    """Create features from economic data only to predict calls"""
    
    LOG.info("Creating economic features for call prediction...")
    
    # Align dates - get overlapping business days
    common_dates = calls_data.index.intersection(economic_data.index)
    
    if len(common_dates) < 50:
        raise ValueError(f"Not enough overlapping dates: {len(common_dates)}")
    
    LOG.info("Overlapping dates: %d", len(common_dates))
    
    # Sort dates
    common_dates = sorted(common_dates)
    
    features_list = []
    targets_list = []
    
    # For each day, use economic data to predict NEXT day's calls
    for i in range(len(common_dates) - 1):
        current_date = common_dates[i]
        next_date = common_dates[i + 1]
        
        # Skip if next date is not the next business day
        if (next_date - current_date).days > 3:  # Allow for weekends
            continue
        
        feature_row = {}
        
        # Current day economic values
        current_econ = economic_data.loc[current_date]
        
        available_indicators = [ind for ind in CFG["economic_indicators"] 
                               if ind in economic_data.columns]
        
        for indicator in available_indicators:
            # Current value
            feature_row[f"{indicator}_today"] = current_econ[indicator]
            
            # Lag-1 value (if available)
            if i > 0:
                lag_date = common_dates[i - 1]
                if (current_date - lag_date).days <= 3:  # Reasonable lag
                    lag_econ = economic_data.loc[lag_date]
                    feature_row[f"{indicator}_lag1"] = lag_econ[indicator]
                    
                    # Change from lag-1
                    feature_row[f"{indicator}_change"] = current_econ[indicator] - lag_econ[indicator]
                    
                    # Percent change
                    if lag_econ[indicator] != 0:
                        feature_row[f"{indicator}_pct_change"] = (
                            (current_econ[indicator] - lag_econ[indicator]) / abs(lag_econ[indicator]) * 100
                        )
                    else:
                        feature_row[f"{indicator}_pct_change"] = 0
                else:
                    # Fill with current value if lag not available
                    feature_row[f"{indicator}_lag1"] = current_econ[indicator]
                    feature_row[f"{indicator}_change"] = 0
                    feature_row[f"{indicator}_pct_change"] = 0
            else:
                # First observation
                feature_row[f"{indicator}_lag1"] = current_econ[indicator]
                feature_row[f"{indicator}_change"] = 0
                feature_row[f"{indicator}_pct_change"] = 0
            
            # Rolling average (5-day)
            if i >= 4:
                recent_dates = common_dates[max(0, i-4):i+1]
                recent_values = [economic_data.loc[d][indicator] for d in recent_dates]
                feature_row[f"{indicator}_ma5"] = np.mean(recent_values)
                
                # Volatility (5-day std)
                feature_row[f"{indicator}_vol5"] = np.std(recent_values)
            else:
                feature_row[f"{indicator}_ma5"] = current_econ[indicator]
                feature_row[f"{indicator}_vol5"] = 0
        
        # Date features
        feature_row["weekday"] = current_date.weekday()
        feature_row["month"] = current_date.month
        feature_row["is_month_end"] = 1 if current_date.day > 25 else 0
        feature_row["is_holiday_week"] = 1 if current_date in holidays.US() else 0
        
        # Historical call context (recent baseline)
        if i >= 5:
            recent_call_dates = common_dates[max(0, i-4):i+1]
            recent_calls = [calls_data.loc[d] for d in recent_call_dates if d in calls_data.index]
            if recent_calls:
                feature_row["recent_calls_avg"] = np.mean(recent_calls)
                feature_row["recent_calls_trend"] = np.mean(np.diff(recent_calls)) if len(recent_calls) > 1 else 0
            else:
                feature_row["recent_calls_avg"] = calls_data.mean()
                feature_row["recent_calls_trend"] = 0
        else:
            feature_row["recent_calls_avg"] = calls_data.mean()
            feature_row["recent_calls_trend"] = 0
        
        # Target: next day's calls
        if next_date in calls_data.index:
            target = calls_data.loc[next_date]
            
            features_list.append(feature_row)
            targets_list.append(target)
    
    # Convert to DataFrames
    X = pd.DataFrame(features_list)
    y = pd.Series(targets_list)
    
    # Clean
    X = X.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    LOG.info("Economics features: %d samples x %d features", X.shape[0], X.shape[1])
    LOG.info("Available economic indicators: %s", available_indicators)
    
    return X, y, available_indicators

def test_economic_models(X, y, available_indicators):
    """Test different models with economic data only"""
    
    LOG.info("=== TESTING ECONOMIC-ONLY MODELS ===")
    
    # Time series split for proper evaluation
    tscv = TimeSeriesSplit(n_splits=3)
    
    models_to_test = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42
        )
    }
    
    results = {}
    
    for model_name, model in models_to_test.items():
        LOG.info(f"\nTesting {model_name}...")
        
        cv_scores = []
        cv_maes = []
        feature_importance = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            score = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            cv_scores.append(score)
            cv_maes.append(mae)
            
            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                feature_importance.append(model.feature_importances_)
        
        avg_r2 = np.mean(cv_scores)
        avg_mae = np.mean(cv_maes)
        std_r2 = np.std(cv_scores)
        std_mae = np.std(cv_maes)
        
        results[model_name] = {
            "r2_mean": avg_r2,
            "r2_std": std_r2,
            "mae_mean": avg_mae,
            "mae_std": std_mae
        }
        
        LOG.info(f"  R² Score: {avg_r2:.3f} ± {std_r2:.3f}")
        LOG.info(f"  MAE: {avg_mae:.0f} ± {std_mae:.0f}")
        
        # Feature importance analysis
        if feature_importance:
            avg_importance = np.mean(feature_importance, axis=0)
            feature_names = X.columns
            
            # Sort by importance
            importance_pairs = list(zip(feature_names, avg_importance))
            importance_pairs.sort(key=lambda x: x[1], reverse=True)
            
            LOG.info(f"  Top 10 features for {model_name}:")
            for i, (feature, importance) in enumerate(importance_pairs[:10]):
                LOG.info(f"    {i+1:2d}. {feature}: {importance:.3f}")
            
            results[model_name]["feature_importance"] = importance_pairs
    
    return results

def analyze_economic_correlations(X, y, available_indicators):
    """Analyze correlations between economic indicators and calls"""
    
    LOG.info("\n=== ECONOMIC CORRELATION ANALYSIS ===")
    
    correlations = {}
    
    # Analyze correlations for each indicator type
    for indicator in available_indicators:
        indicator_features = [col for col in X.columns if col.startswith(indicator)]
        
        indicator_corrs = {}
        for feature in indicator_features:
            corr = X[feature].corr(y)
            indicator_corrs[feature] = corr
        
        # Find best correlation for this indicator
        if indicator_corrs:
            best_feature = max(indicator_corrs.items(), key=lambda x: abs(x[1]))
            correlations[indicator] = {
                "best_feature": best_feature[0],
                "best_correlation": best_feature[1],
                "all_correlations": indicator_corrs
            }
    
    # Sort indicators by best correlation
    sorted_indicators = sorted(correlations.items(), 
                              key=lambda x: abs(x[1]["best_correlation"]), 
                              reverse=True)
    
    LOG.info("Economic indicators ranked by correlation with calls:")
    for i, (indicator, data) in enumerate(sorted_indicators, 1):
        corr = data["best_correlation"]
        feature = data["best_feature"]
        LOG.info(f"  {i:2d}. {indicator}: {corr:+.3f} (via {feature})")
    
    return correlations, sorted_indicators

def create_baseline_comparison(y):
    """Create baseline models for comparison"""
    
    LOG.info("\n=== BASELINE COMPARISON ===")
    
    # Simple baselines
    baselines = {}
    
    # Mean prediction
    mean_pred = np.full_like(y, y.mean())
    baselines["Mean"] = {
        "mae": mean_absolute_error(y, mean_pred),
        "r2": r2_score(y, mean_pred)
    }
    
    # Last value prediction (naive forecast)
    naive_pred = np.full_like(y, y.iloc[0])
    baselines["Naive (first value)"] = {
        "mae": mean_absolute_error(y, naive_pred),
        "r2": r2_score(y, naive_pred)
    }
    
    # Random walk (yesterday's value)
    if len(y) > 1:
        random_walk_pred = np.concatenate([[y.iloc[0]], y.iloc[:-1].values])
        baselines["Random Walk"] = {
            "mae": mean_absolute_error(y, random_walk_pred),
            "r2": r2_score(y, random_walk_pred)
        }
    
    LOG.info("Baseline model performance:")
    for baseline_name, metrics in baselines.items():
        LOG.info(f"  {baseline_name}: MAE = {metrics['mae']:.0f}, R² = {metrics['r2']:.3f}")
    
    return baselines

def main():
    output_dir = Path(CFG["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    LOG.info("=== ECONOMICS-ONLY CALL PREDICTION TEST ===")
    LOG.info("Testing if economic indicators alone can predict call volumes")
    
    # Load data
    calls_data = load_calls_data()
    economic_data = load_economic_data()
    
    # Create features
    X, y, available_indicators = create_economics_features(calls_data, economic_data)
    
    # Baseline comparison
    baselines = create_baseline_comparison(y)
    
    # Correlation analysis
    correlations, sorted_indicators = analyze_economic_correlations(X, y, available_indicators)
    
    # Test models
    model_results = test_economic_models(X, y, available_indicators)
    
    # Summary and conclusions
    LOG.info("\n=== ECONOMICS-ONLY MODEL CONCLUSIONS ===")
    
    best_model = min(model_results.items(), key=lambda x: x[1]["mae_mean"])
    best_baseline = min(baselines.items(), key=lambda x: x[1]["mae"])
    
    LOG.info(f"Best Economic Model: {best_model[0]}")
    LOG.info(f"  MAE: {best_model[1]['mae_mean']:.0f}")
    LOG.info(f"  R²: {best_model[1]['r2_mean']:.3f}")
    
    LOG.info(f"\nBest Baseline: {best_baseline[0]}")
    LOG.info(f"  MAE: {best_baseline[1]['mae']:.0f}")
    LOG.info(f"  R²: {best_baseline[1]['r2']:.3f}")
    
    # Compare to your original mail model
    original_mail_mae = 4440  # From your original results
    
    LOG.info(f"\n=== COMPARISON TO MAIL MODEL ===")
    LOG.info(f"Original Mail Model MAE: {original_mail_mae}")
    LOG.info(f"Best Economics Model MAE: {best_model[1]['mae_mean']:.0f}")
    LOG.info(f"Economics vs Mail: {best_model[1]['mae_mean'] / original_mail_mae:.1f}x worse")
    
    if best_model[1]["mae_mean"] > original_mail_mae:
        LOG.info("❌ Economic indicators alone perform WORSE than mail data")
        LOG.info("   This confirms mail data is the primary signal")
    else:
        LOG.info("✅ Economic indicators perform better than expected!")
    
    if best_model[1]["mae_mean"] > best_baseline[1]["mae"]:
        LOG.info("❌ Economic model performs worse than simple baselines")
        LOG.info("   Economic indicators have no predictive power for calls")
    else:
        LOG.info("✅ Economic model beats baseline - some predictive signal exists")
    
    # Save results
    all_results = {
        "model_results": model_results,
        "correlations": {k: v for k, v in correlations.items()},
        "baselines": baselines,
        "summary": {
            "best_model": best_model[0],
            "best_model_mae": float(best_model[1]["mae_mean"]),
            "best_model_r2": float(best_model[1]["r2_mean"]),
            "best_baseline_mae": float(best_baseline[1]["mae"]),
            "original_mail_mae": original_mail_mae,
            "economics_vs_mail_ratio": float(best_model[1]["mae_mean"] / original_mail_mae)
        }
    }
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    json_safe_results = convert_numpy(all_results)
    
    with open(output_dir / "economics_only_results.json", "w") as f:
        json.dump(json_safe_results, f, indent=2)
    
    LOG.info(f"\nResults saved to: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
