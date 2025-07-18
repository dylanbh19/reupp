**Perfect!** Let me enhance your working script to systematically test each economic indicator and find the best combinations.

```python
#!/usr/bin/env python
# enhanced_economic_testing.py
# =========================================================
# Test each economic indicator individually, then best combinations
# Based on your working mail model + systematic economic testing
# =========================================================

from pathlib import Path
import json
import logging
import sys
import warnings
from datetime import datetime, timedelta
from itertools import combinations

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

warnings.filterwarnings('ignore')

LOG = logging.getLogger("enhanced_economic_testing")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | econ_test | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

CFG = {
    "top_mail_types": [
        "Reject_Ltrs", "Cheque 1099", "Exercise_Converted",
        "SOI_Confirms", "Exch_chks", "ACH_Debit_Enrollment",
        "Transfer", "COA", "NOTC_WITHDRAW", "Repl_Chks"
    ],
    "economic_data_files": [
        "economic_data_for_model.csv",
        "expanded_economic_data.csv"
    ],
    "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
    "bootstrap_samples": 30,
    "output_dir": "enhanced_economic_testing_results"
}

def _to_date(s):
    return pd.to_datetime(s, errors="coerce").dt.date

def _find_file(candidates):
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError(f"None found: {candidates}")

def load_economic_data():
    """Load and clean all economic data"""
    
    LOG.info("Loading economic data...")
    
    all_economic = pd.DataFrame()
    
    for econ_file in CFG["economic_data_files"]:
        econ_path = Path(econ_file)
        if econ_path.exists():
            LOG.info(f"Loading {econ_file}...")
            try:
                econ_df = pd.read_csv(econ_path, parse_dates=['Date'])
                econ_df.set_index('Date', inplace=True)
                
                # Clean numeric columns only
                numeric_cols = econ_df.select_dtypes(include=[np.number]).columns
                econ_df = econ_df[numeric_cols]
                
                # Combine with existing data
                if all_economic.empty:
                    all_economic = econ_df.copy()
                else:
                    all_economic = pd.concat([all_economic, econ_df], axis=1)
                
                LOG.info(f"  Loaded {len(econ_df.columns)} indicators from {econ_file}")
                
            except Exception as e:
                LOG.warning(f"  Error loading {econ_file}: {e}")
    
    if all_economic.empty:
        LOG.warning("No economic data loaded")
        return pd.DataFrame()
    
    # Remove duplicate columns
    all_economic = all_economic.loc[:, ~all_economic.columns.duplicated()]
    
    # Clean NaN values with forward/backward fill
    all_economic = all_economic.fillna(method='ffill').fillna(method='bfill')
    
    # Check data quality
    nan_counts = all_economic.isnull().sum()
    good_indicators = []
    
    LOG.info("Economic data quality check:")
    for col in all_economic.columns:
        nan_pct = (nan_counts[col] / len(all_economic)) * 100
        if nan_pct < 10:  # Less than 10% NaN
            good_indicators.append(col)
            LOG.info(f"  ✓ {col}: {nan_pct:.1f}% NaN")
        else:
            LOG.warning(f"  ✗ {col}: {nan_pct:.1f}% NaN (excluded)")
    
    # Return only good indicators
    if good_indicators:
        clean_economic = all_economic[good_indicators].copy()
        LOG.info(f"Economic data ready: {len(good_indicators)} indicators, {len(clean_economic)} days")
        return clean_economic
    else:
        LOG.warning("No usable economic indicators found")
        return pd.DataFrame()

def load_mail_call_data():
    """Load mail and calls data (same as your working model)"""
    
    LOG.info("Loading mail and calls data...")
    
    # EXACT same code as your working model
    mail_path = _find_file(["mail.csv", "data/mail.csv"])
    mail = pd.read_csv(mail_path)
    mail.columns = [c.lower().strip() for c in mail.columns]
    mail["mail_date"] = _to_date(mail["mail_date"])
    mail = mail.dropna(subset=["mail_date"])

    vol_path = _find_file(["callvolumes.csv", "data/callvolumes.csv"])
    intent_path = _find_file(["callintent.csv", "data/callintent.csv", "callintetn.csv"])

    df_vol = pd.read_csv(vol_path)
    df_vol.columns = [c.lower().strip() for c in df_vol.columns]
    dcol_v = next(c for c in df_vol.columns if "date" in c)
    df_vol[dcol_v] = _to_date(df_vol[dcol_v])
    vol_daily = df_vol.groupby(dcol_v)[df_vol.columns.difference([dcol_v])[0]].sum()

    df_int = pd.read_csv(intent_path)
    df_int.columns = [c.lower().strip() for c in df_int.columns]
    dcol_i = next(c for c in df_int.columns if "date" in c or "conversationstart" in c)
    df_int[dcol_i] = _to_date(df_int[dcol_i])
    int_daily = df_int.groupby(dcol_i).size()

    overlap = vol_daily.index.intersection(int_daily.index)
    if len(overlap) >= 5:
        scale = int_daily.loc[overlap].mean() / vol_daily.loc[overlap].mean()
        vol_daily *= scale
    calls_total = vol_daily.combine_first(int_daily).sort_index()

    mail_daily = (mail.groupby(["mail_date", "mail_type"], as_index=False)["mail_volume"].sum()
                   .pivot(index="mail_date", columns="mail_type", values="mail_volume").fillna(0))
    
    mail_daily.index = pd.to_datetime(mail_daily.index)
    calls_total.index = pd.to_datetime(calls_total.index)

    us_holidays = holidays.US()
    biz_mask = (~mail_daily.index.weekday.isin([5, 6])) & (~mail_daily.index.isin(us_holidays))
    mail_daily = mail_daily.loc[biz_mask]
    calls_total = calls_total.loc[calls_total.index.isin(mail_daily.index)]

    daily = mail_daily.join(calls_total.rename("calls_total"), how="inner")
    
    LOG.info("Daily mail-calls data: %s", daily.shape)
    return daily

def create_baseline_features(daily):
    """Create baseline features (same as your working model)"""
    
    features_list = []
    targets_list = []
    
    for i in range(len(daily) - 1):
        current_day = daily.iloc[i]
        next_day = daily.iloc[i + 1]
        
        feature_row = {}
        
        # Mail volumes
        available_types = [t for t in CFG["top_mail_types"] if t in daily.columns]
        for mail_type in available_types:
            feature_row[f"{mail_type}_volume"] = current_day[mail_type]
        
        # Total mail volume
        total_mail = sum(current_day[t] for t in available_types)
        feature_row["total_mail_volume"] = total_mail
        feature_row["log_total_mail_volume"] = np.log1p(total_mail)
        
        # Mail percentile
        mail_history = daily[available_types].sum(axis=1).iloc[:i+1]
        if len(mail_history) > 10:
            feature_row["mail_percentile"] = (mail_history <= total_mail).mean()
        else:
            feature_row["mail_percentile"] = 0.5
        
        # Date features
        current_date = daily.index[i]
        feature_row["weekday"] = current_date.weekday()
        feature_row["month"] = current_date.month
        feature_row["is_month_end"] = 1 if current_date.day > 25 else 0
        feature_row["is_holiday_week"] = 1 if current_date in holidays.US() else 0
        
        # Recent calls
        recent_calls = daily["calls_total"].iloc[max(0, i-5):i+1]
        feature_row["recent_calls_avg"] = recent_calls.mean()
        feature_row["recent_calls_trend"] = recent_calls.diff().mean()
        
        # Target
        target = next_day["calls_total"]
        
        features_list.append(feature_row)
        targets_list.append(target)
    
    X = pd.DataFrame(features_list)
    y = pd.Series(targets_list)
    
    # ROBUST NaN handling
    X = X.fillna(0)
    y = y.fillna(y.mean())  # Fill target NaNs with mean
    
    # Remove any remaining infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    LOG.info("Baseline features: %d samples x %d features", X.shape[0], X.shape[1])
    return X, y

def create_enhanced_features(daily, economic_data, economic_indicators):
    """Create features with specified economic indicators"""
    
    # First create baseline features
    features_list = []
    targets_list = []
    
    for i in range(len(daily) - 1):
        current_day = daily.iloc[i]
        next_day = daily.iloc[i + 1]
        
        feature_row = {}
        
        # Mail features (same as baseline)
        available_types = [t for t in CFG["top_mail_types"] if t in daily.columns]
        for mail_type in available_types:
            feature_row[f"{mail_type}_volume"] = current_day[mail_type]
        
        total_mail = sum(current_day[t] for t in available_types)
        feature_row["total_mail_volume"] = total_mail
        feature_row["log_total_mail_volume"] = np.log1p(total_mail)
        
        mail_history = daily[available_types].sum(axis=1).iloc[:i+1]
        if len(mail_history) > 10:
            feature_row["mail_percentile"] = (mail_history <= total_mail).mean()
        else:
            feature_row["mail_percentile"] = 0.5
        
        current_date = daily.index[i]
        feature_row["weekday"] = current_date.weekday()
        feature_row["month"] = current_date.month
        feature_row["is_month_end"] = 1 if current_date.day > 25 else 0
        feature_row["is_holiday_week"] = 1 if current_date in holidays.US() else 0
        
        recent_calls = daily["calls_total"].iloc[max(0, i-5):i+1]
        feature_row["recent_calls_avg"] = recent_calls.mean()
        feature_row["recent_calls_trend"] = recent_calls.diff().mean()
        
        # NEW: Economic features
        current_date_normalized = pd.to_datetime(current_date.strftime('%Y-%m-%d'))
        
        if not economic_data.empty and current_date_normalized in economic_data.index:
            for econ_indicator in economic_indicators:
                if econ_indicator in economic_data.columns:
                    # Same day economic value
                    econ_value = economic_data.loc[current_date_normalized, econ_indicator]
                    feature_row[f"{econ_indicator}_today"] = econ_value if not pd.isna(econ_value) else 0
                    
                    # 1-day lag economic value
                    if i > 0:
                        prev_date = pd.to_datetime(daily.index[i-1].strftime('%Y-%m-%d'))
                        if prev_date in economic_data.index:
                            econ_lag_value = economic_data.loc[prev_date, econ_indicator]
                            feature_row[f"{econ_indicator}_lag1"] = econ_lag_value if not pd.isna(econ_lag_value) else 0
                        else:
                            feature_row[f"{econ_indicator}_lag1"] = 0
                    else:
                        feature_row[f"{econ_indicator}_lag1"] = 0
        else:
            # Fill with zeros if no economic data available
            for econ_indicator in economic_indicators:
                feature_row[f"{econ_indicator}_today"] = 0
                feature_row[f"{econ_indicator}_lag1"] = 0
        
        target = next_day["calls_total"]
        
        features_list.append(feature_row)
        targets_list.append(target)
    
    X = pd.DataFrame(features_list)
    y = pd.Series(targets_list)
    
    # ROBUST NaN handling
    X = X.fillna(0)
    y = y.fillna(y.mean())
    X = X.replace([np.inf, -np.inf], 0)
    
    return X, y

def evaluate_model_performance(X, y, model_name="Model"):
    """Evaluate model with cross-validation"""
    
    # Use same Random Forest as your working model
    rf_model = RandomForestRegressor(
        n_estimators=50,
        max_depth=6,
        min_samples_leaf=3,
        random_state=42
    )
    
    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    cv_mae_scores = []
    cv_r2_scores = []
    
    for train_idx, val_idx in tscv.split(X):
        try:
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            rf_model.fit(X_train, y_train)
            y_pred = rf_model.predict(X_val)
            
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            cv_mae_scores.append(mae)
            cv_r2_scores.append(r2)
            
        except Exception as e:
            LOG.warning(f"Error in CV fold for {model_name}: {e}")
            cv_mae_scores.append(999999)  # Very bad score
            cv_r2_scores.append(-999)
    
    avg_mae = np.mean(cv_mae_scores)
    avg_r2 = np.mean(cv_r2_scores)
    
    return avg_mae, avg_r2, rf_model

def test_individual_economic_indicators(daily, economic_data):
    """Test each economic indicator individually"""
    
    LOG.info("=== TESTING INDIVIDUAL ECONOMIC INDICATORS ===")
    
    if economic_data.empty:
        LOG.warning("No economic data available for testing")
        return []
    
    # First test baseline model
    LOG.info("Testing baseline model (mail only)...")
    X_baseline, y_baseline = create_baseline_features(daily)
    baseline_mae, baseline_r2, _ = evaluate_model_performance(X_baseline, y_baseline, "Baseline")
    
    LOG.info(f"Baseline Model:")
    LOG.info(f"  MAE: {baseline_mae:.0f}")
    LOG.info(f"  R²: {baseline_r2:.3f}")
    
    # Test each economic indicator
    individual_results = []
    
    for econ_indicator in economic_data.columns:
        LOG.info(f"\nTesting {econ_indicator}...")
        
        try:
            X_enhanced, y_enhanced = create_enhanced_features(daily, economic_data, [econ_indicator])
            enhanced_mae, enhanced_r2, _ = evaluate_model_performance(
                X_enhanced, y_enhanced, f"Baseline+{econ_indicator}"
            )
            
            # Calculate improvement
            mae_improvement = ((baseline_mae - enhanced_mae) / baseline_mae) * 100
            r2_improvement = ((enhanced_r2 - baseline_r2) / abs(baseline_r2)) * 100 if baseline_r2 != 0 else 0
            
            individual_results.append({
                'indicator': econ_indicator,
                'mae': enhanced_mae,
                'r2': enhanced_r2,
                'mae_improvement_pct': mae_improvement,
                'r2_improvement_pct': r2_improvement,
                'overall_improvement': (mae_improvement + r2_improvement) / 2
            })
            
            LOG.info(f"  MAE: {enhanced_mae:.0f} ({mae_improvement:+.1f}%)")
            LOG.info(f"  R²: {enhanced_r2:.3f} ({r2_improvement:+.1f}%)")
            
        except Exception as e:
            LOG.warning(f"  Error testing {econ_indicator}: {e}")
    
    # Sort by overall improvement
    individual_results.sort(key=lambda x: x['overall_improvement'], reverse=True)
    
    LOG.info("\n=== INDIVIDUAL INDICATOR RESULTS (RANKED) ===")
    for i, result in enumerate(individual_results[:10], 1):  # Top 10
        LOG.info(f"{i:2d}. {result['indicator']}: "
                f"R²={result['r2']:.3f} ({result['r2_improvement_pct']:+.1f}%), "
                f"MAE={result['mae']:.0f} ({result['mae_improvement_pct']:+.1f}%)")
    
    return individual_results, baseline_mae, baseline_r2

def test_best_combinations(daily, economic_data, individual_results, baseline_mae, baseline_r2):
    """Test combinations of best economic indicators"""
    
    LOG.info("\n=== TESTING BEST COMBINATIONS ===")
    
    if len(individual_results) < 2:
        LOG.warning("Not enough good indicators for combination testing")
        return []
    
    # Get top 5 indicators
    top_indicators = [result['indicator'] for result in individual_results[:5]]
    LOG.info(f"Top 5 indicators for combination testing: {top_indicators}")
    
    combination_results = []
    
    # Test combinations of 2, 3, 4, and 5 indicators
    for combo_size in [2, 3, 4, 5]:
        if combo_size > len(top_indicators):
            continue
            
        LOG.info(f"\nTesting combinations of {combo_size} indicators...")
        
        for combo in combinations(top_indicators[:combo_size+2], combo_size):  # Test a few more than top N
            try:
                combo_name = "+".join(combo)
                LOG.info(f"  Testing: {combo_name}")
                
                X_combo, y_combo = create_enhanced_features(daily, economic_data, list(combo))
                combo_mae, combo_r2, _ = evaluate_model_performance(X_combo, y_combo, combo_name)
                
                # Calculate improvement over baseline
                mae_improvement = ((baseline_mae - combo_mae) / baseline_mae) * 100
                r2_improvement = ((combo_r2 - baseline_r2) / abs(baseline_r2)) * 100 if baseline_r2 != 0 else 0
                
                combination_results.append({
                    'combination': list(combo),
                    'combination_name': combo_name,
                    'size': combo_size,
                    'mae': combo_mae,
                    'r2': combo_r2,
                    'mae_improvement_pct': mae_improvement,
                    'r2_improvement_pct': r2_improvement,
                    'overall_improvement': (mae_improvement + r2_improvement) / 2
                })
                
                LOG.info(f"    MAE: {combo_mae:.0f} ({mae_improvement:+.1f}%)")
                LOG.info(f"    R²: {combo_r2:.3f} ({r2_improvement:+.1f}%)")
                
            except Exception as e:
                LOG.warning(f"    Error testing {combo}: {e}")
    
    # Sort by overall improvement
    combination_results.sort(key=lambda x: x['overall_improvement'], reverse=True)
    
    LOG.info("\n=== BEST COMBINATIONS (RANKED) ===")
    for i, result in enumerate(combination_results[:10], 1):  # Top 10
        LOG.info(f"{i:2d}. {result['combination_name']}: "
                f"R²={result['r2']:.3f} ({result['r2_improvement_pct']:+.1f}%), "
                f"MAE={result['mae']:.0f} ({result['mae_improvement_pct']:+.1f}%)")
    
    return combination_results

def create_enhanced_prediction_function(daily, economic_data, best_indicators):
    """Create enhanced prediction function with best economic indicators"""
    
    LOG.info(f"Creating enhanced prediction function with: {best_indicators}")
    
    # Train final model with best indicators
    X_final, y_final = create_enhanced_features(daily, economic_data, best_indicators)
    
    # Train quantile models
    models = {}
    
    split_point = int(len(X_final) * 0.8)
    X_train, X_test = X_final.iloc[:split_point], X_final.iloc[split_point:]
    y_train, y_test = y_final.iloc[:split_point], y_final.iloc[split_point:]
    
    # Store feature names for prediction
    models["feature_names"] = list(X_final.columns)
    models["economic_indicators"] = best_indicators
    
    # Quantile models
    for quantile in CFG["quantiles"]:
        LOG.info("  Training %d%% quantile model...", int(quantile * 100))
        
        model = QuantileRegressor(quantile=quantile, alpha=0.1, solver='highs')
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        models[f"quantile_{quantile}"] = model
        LOG.info("    Validation MAE: %.0f", mae)
    
    # Bootstrap ensemble
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
    
    return models

def predict_enhanced(models, mail_inputs, economic_inputs=None, date_str=None):
    """Enhanced prediction function"""
    
    if date_str is None:
        predict_date = datetime.now() + timedelta(days=1)
    else:
        predict_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Create feature vector
    feature_row = {}
    
    # Mail features
    total_mail = 0
    for mail_type in CFG["top_mail_types"]:
        volume = mail_inputs.get(mail_type, 0)
        feature_row[f"{mail_type}_volume"] = volume
        total_mail += volume
    
    feature_row["total_mail_volume"] = total_mail
    feature_row["log_total_mail_volume"] = np.log1p(total_mail)
    feature_row["mail_percentile"] = 0.5
    
    # Date features
    feature_row["weekday"] = predict_date.weekday()
    feature_row["month"] = predict_date.month
    feature_row["is_month_end"] = 1 if predict_date.day > 25 else 0
    feature_row["is_holiday_week"] = 1 if predict_date.date() in holidays.US() else 0
    
    feature_row["recent_calls_avg"] = 15000
    feature_row["recent_calls_trend"] = 0
    
    # Economic features
    if economic_inputs and "economic_indicators" in models:
        for econ_indicator in models["economic_indicators"]:
            value = economic_inputs.get(econ_indicator, 0)
            feature_row[f"{econ_indicator}_today"] = value
            feature_row[f"{econ_indicator}_lag1"] = value
    else:
        # Use defaults
        if "economic_indicators" in models:
            for econ_indicator in models["economic_indicators"]:
                feature_row[f"{econ_indicator}_today"] = 0
                feature_row[f"{econ_indicator}_lag1"] = 0
    
    # Create DataFrame with exact feature order
    X_input = pd.DataFrame([feature_row])
    if "feature_names" in models:
        X_input = X_input.reindex(columns=models["feature_names"], fill_value=0)
    
    # Get predictions
    quantile_preds = {}
    for quantile in CFG["quantiles"]:
        model = models[f"quantile_{quantile}"]
        pred = model.predict(X_input)[0]
        quantile_preds[f"q{int(quantile*100)}"] = max(0, pred)
    
    # Bootstrap predictions
    bootstrap_preds = []
    for model in models["bootstrap_ensemble"]:
        pred = model.predict(X_input)[0]
        bootstrap_preds.append(max(0, pred))
    
    bootstrap_stats = {
        "mean": np.mean(bootstrap_preds),
        "std": np.std(bootstrap_preds),
        "min": np.min(bootstrap_preds),
        "max": np.max(bootstrap_preds)
    }
    
    return quantile_preds, bootstrap_stats

def main():
    output_dir = Path(CFG["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    LOG.info("=== ENHANCED ECONOMIC TESTING ===")
    
    # Load data
    daily = load_mail_call_data()
    economic_data = load_economic_data()
    
    # Test individual indicators
    individual_results, baseline_mae, baseline_r2 = test_individual_economic_indicators(daily, economic_data)
    
    # Test best combinations
    combination_results = test_best_combinations(daily, economic_data, individual_results, baseline_mae, baseline_r2)
    
    # Find overall best result
    all_results = individual_results + combination_results
    if all_results:
        best_result = max(all_results, key=lambda x: x['overall_improvement'])
        
        LOG.info(f"\n=== BEST OVERALL RESULT ===")
        if 'combination' in best_result:
            LOG.info(f"Best configuration: {best_result['combination_name']}")
            best_indicators = best_result['combination']
        else:
            LOG.info(f"Best configuration: {best_result['indicator']}")
            best_indicators = [best_result['indicator']]
        
        LOG.info(f"Performance: R²={best_result['r2']:.3f}, MAE={best_result['mae']:.0f}")
        LOG.info(f"Improvement: R²={best_result['r2_improvement_pct']:+.1f}%, MAE={best_result['mae_improvement_pct']:+.1f}%")
        
        # Create enhanced model with best indicators
        LOG.info(f"\nCreating enhanced model with best indicators...")
        enhanced_models = create_enhanced_prediction_function(daily, economic_data, best_indicators)
        
        # Save enhanced model
        joblib.dump(enhanced_models, output_dir / "enhanced_models.pkl")
        
        # Test enhanced prediction
        LOG.info(f"\n=== TESTING ENHANCED PREDICTION ===")
        example_mail = {"Reject_Ltrs": 1500, "Cheque 1099": 800, "Exercise_Converted": 200}
        example_econ = {indicator: 2000 for indicator in best_indicators}  # Default values
        
        quantile_preds, bootstrap_stats = predict_enhanced(enhanced_models, example_mail, example_econ)
        
        LOG.info(f"Example prediction:")
        LOG.info(f"  Mail inputs: {example_mail}")
        LOG.info(f"  Economic inputs: {example_econ}")
        LOG.info(f"  Most likely: {quantile_preds['q50']:.0f} calls")
        LOG.info(f"  Business range (25-75%): {quantile_preds['q25']:.0f} - {quantile_preds['q75']:.0f} calls")
        
    else:
        LOG.warning("No improvements found with economic indicators")
        best_indicators = []
    
    # Save all results
    results_summary = {
        "baseline_performance": {"mae": baseline_mae, "r2": baseline_r2},
        "individual_results": individual_results[:10],  # Top 10
        "combination_results": combination_results[:10],  # Top 10
        "best_indicators": best_indicators,
        "best_performance": best_result if all_results else None
    }
    
    with open(output_dir / "economic_testing_results.json", "w") as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    LOG.info(f"\n=== TESTING COMPLETE ===")
    LOG.info(f"Results saved to: {output_dir.resolve()}")

if __name__ == "__main__":
    main()
```

**This enhanced script will:**

1. **🔧 Fix NaN Issues**: Robust handling of missing values with fallbacks
1. **📊 Test Each Economic Indicator**: Individual performance vs baseline
1. **🎯 Test Best Combinations**: 2, 3, 4, and 5 indicator combinations
1. **📈 Rank All Results**: Shows improvement % for​​​​​​​​​​​​​​​​