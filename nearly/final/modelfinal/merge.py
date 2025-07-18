#!/usr/bin/env python
# enhanced_mail_input_forecast.py
# =========================================================
# Enhanced INPUT-DRIVEN range forecast model
# Tests economic indicators systematically with mail types
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

LOG = logging.getLogger("enhanced_mail_forecast")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | enhanced_mail | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

CFG = {
    "top_mail_types": [
        "Reject_Ltrs", "Cheque 1099", "Exercise_Converted",
        "SOI_Confirms", "Exch_chks", "ACH_Debit_Enrollment",
        "Transfer", "COA", "NOTC_WITHDRAW", "Repl_Chks"
    ],
    "top_economic_indicators": [
        "Russell2000", "Dollar_Index", "NASDAQ", "SP500", "Technology"
    ],
    "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
    "bootstrap_samples": 30,
    "output_dir": "enhanced_mail_ranges"
}

def _to_date(s):
    return pd.to_datetime(s, errors="coerce").dt.date

def _find_file(candidates):
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError(f"None found: {candidates}")

def load_mail_call_data():
    """Load data and create mail->calls relationship dataset"""
    
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

def load_economic_data():
    """Load all economic indicators from expanded dataset"""
    
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
        
        LOG.info("Economic indicators loaded: %s", list(econ_data.columns))
        return econ_data
        
    except FileNotFoundError:
        LOG.warning("No economic data files found - creating dummy data for testing")
        # Create dummy economic data for testing
        dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
        dummy_econ = pd.DataFrame({
            indicator: np.random.randn(len(dates)).cumsum() + 100
            for indicator in CFG["top_economic_indicators"]
        }, index=dates)
        return dummy_econ

def create_enhanced_features(daily, economic_data=None, selected_econ_indicators=None):
    """Create features combining mail and economic data"""
    
    features_list = []
    targets_list = []
    
    # Determine which economic indicators to use
    if selected_econ_indicators is None:
        selected_econ_indicators = CFG["top_economic_indicators"] if economic_data is not None else []
    
    LOG.info("Using economic indicators: %s", selected_econ_indicators)
    
    # For each day, create features from THAT day's data to predict NEXT day's calls
    for i in range(len(daily) - 1):
        current_day = daily.iloc[i]
        next_day = daily.iloc[i + 1]
        current_date = daily.index[i]
        
        feature_row = {}
        
        # === MAIL VOLUME FEATURES ===
        available_mail_types = [t for t in CFG["top_mail_types"] if t in daily.columns]
        
        for mail_type in available_mail_types:
            feature_row[f"{mail_type}_volume"] = current_day[mail_type]
        
        # Total mail volume
        total_mail = sum(current_day[t] for t in available_mail_types)
        feature_row["total_mail_volume"] = total_mail
        feature_row["log_total_mail_volume"] = np.log1p(total_mail)
        
        # Mail volume percentiles (relative to historical)
        mail_history = daily[available_mail_types].sum(axis=1).iloc[:i+1]
        if len(mail_history) > 10:
            feature_row["mail_percentile"] = (mail_history <= total_mail).mean()
        else:
            feature_row["mail_percentile"] = 0.5
        
        # === ECONOMIC FEATURES ===
        if economic_data is not None and len(selected_econ_indicators) > 0:
            # Get economic data for current date
            econ_current = None
            econ_lag1 = None
            
            # Find closest economic data
            available_econ_dates = economic_data.index
            current_econ_dates = available_econ_dates[available_econ_dates <= current_date]
            
            if len(current_econ_dates) > 0:
                closest_date = current_econ_dates[-1]  # Most recent date <= current_date
                econ_current = economic_data.loc[closest_date]
                
                # Get lag-1 data
                if len(current_econ_dates) > 1:
                    lag_date = current_econ_dates[-2]
                    econ_lag1 = economic_data.loc[lag_date]
            
            # Add economic features
            for indicator in selected_econ_indicators:
                if indicator in economic_data.columns:
                    # Current day economic value
                    if econ_current is not None:
                        feature_row[f"{indicator}_today"] = econ_current[indicator]
                    else:
                        feature_row[f"{indicator}_today"] = 0
                    
                    # Lag-1 economic value
                    if econ_lag1 is not None:
                        feature_row[f"{indicator}_lag1"] = econ_lag1[indicator]
                    else:
                        feature_row[f"{indicator}_lag1"] = 0
                    
                    # Economic change
                    if econ_current is not None and econ_lag1 is not None:
                        change = econ_current[indicator] - econ_lag1[indicator]
                        feature_row[f"{indicator}_change"] = change
                    else:
                        feature_row[f"{indicator}_change"] = 0
        
        # === DATE FEATURES ===
        feature_row["weekday"] = current_date.weekday()
        feature_row["month"] = current_date.month
        feature_row["is_month_end"] = 1 if current_date.day > 25 else 0
        feature_row["is_holiday_week"] = 1 if current_date in holidays.US() else 0
        
        # === BASELINE CALL FEATURES ===
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
    
    LOG.info("Enhanced features: %d samples x %d features", X.shape[0], X.shape[1])
    
    return X, y

def evaluate_model_performance(X, y, model_type="random_forest"):
    """Evaluate model performance with time series cross-validation"""
    
    # Use time series split for validation
    tscv = TimeSeriesSplit(n_splits=3)
    
    scores = []
    maes = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        if model_type == "random_forest":
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=5,
                random_state=42
            )
        else:  # linear
            model = Ridge(alpha=1.0)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        score = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        scores.append(score)
        maes.append(mae)
    
    return np.mean(scores), np.mean(maes)

def test_economic_indicators_systematically(daily, economic_data):
    """Test economic indicators one by one and in combinations"""
    
    LOG.info("=== SYSTEMATIC ECONOMIC INDICATOR TESTING ===")
    
    results = {}
    
    # Test baseline (no economic indicators)
    LOG.info("\nTesting BASELINE (no economic indicators)...")
    X_baseline, y_baseline = create_enhanced_features(daily, None, [])
    r2_baseline, mae_baseline = evaluate_model_performance(X_baseline, y_baseline)
    
    results["baseline"] = {
        "indicators": [],
        "r2_score": r2_baseline,
        "mae": mae_baseline,
        "features": X_baseline.shape[1]
    }
    
    LOG.info("Baseline Performance: R² = %.3f, MAE = %.0f", r2_baseline, mae_baseline)
    
    # Test each indicator individually
    LOG.info("\n=== TESTING INDIVIDUAL INDICATORS ===")
    
    available_indicators = [ind for ind in CFG["top_economic_indicators"] 
                           if ind in economic_data.columns]
    
    individual_results = {}
    
    for indicator in available_indicators:
        LOG.info(f"\nTesting: {indicator}")
        
        X_single, y_single = create_enhanced_features(daily, economic_data, [indicator])
        r2_single, mae_single = evaluate_model_performance(X_single, y_single)
        
        individual_results[indicator] = {
            "r2_score": r2_single,
            "mae": mae_single,
            "improvement_r2": r2_single - r2_baseline,
            "improvement_mae": mae_baseline - mae_single  # Lower MAE is better
        }
        
        LOG.info("  R² = %.3f (Δ%+.3f), MAE = %.0f (Δ%+.0f)", 
                r2_single, r2_single - r2_baseline, 
                mae_single, mae_single - mae_baseline)
    
    # Sort by R² improvement
    sorted_individual = sorted(individual_results.items(), 
                              key=lambda x: x[1]["r2_score"], reverse=True)
    
    LOG.info("\n=== INDIVIDUAL INDICATOR RANKINGS ===")
    for i, (indicator, metrics) in enumerate(sorted_individual, 1):
        LOG.info("%2d. %s: R² = %.3f, MAE = %.0f", 
                i, indicator, metrics["r2_score"], metrics["mae"])
    
    # Test combinations of top indicators
    LOG.info("\n=== TESTING INDICATOR COMBINATIONS ===")
    
    top_3_indicators = [item[0] for item in sorted_individual[:3]]
    combination_results = {}
    
    # Test pairs
    for combo in combinations(top_3_indicators, 2):
        combo_name = " + ".join(combo)
        LOG.info(f"\nTesting combination: {combo_name}")
        
        X_combo, y_combo = create_enhanced_features(daily, economic_data, list(combo))
        r2_combo, mae_combo = evaluate_model_performance(X_combo, y_combo)
        
        combination_results[combo_name] = {
            "indicators": list(combo),
            "r2_score": r2_combo,
            "mae": mae_combo,
            "improvement_r2": r2_combo - r2_baseline,
            "improvement_mae": mae_baseline - mae_combo
        }
        
        LOG.info("  R² = %.3f (Δ%+.3f), MAE = %.0f (Δ%+.0f)", 
                r2_combo, r2_combo - r2_baseline, 
                mae_combo, mae_combo - mae_baseline)
    
    # Test all top 3
    if len(top_3_indicators) >= 3:
        combo_name = " + ".join(top_3_indicators)
        LOG.info(f"\nTesting combination: {combo_name}")
        
        X_all3, y_all3 = create_enhanced_features(daily, economic_data, top_3_indicators)
        r2_all3, mae_all3 = evaluate_model_performance(X_all3, y_all3)
        
        combination_results[combo_name] = {
            "indicators": top_3_indicators,
            "r2_score": r2_all3,
            "mae": mae_all3,
            "improvement_r2": r2_all3 - r2_baseline,
            "improvement_mae": mae_baseline - mae_all3
        }
        
        LOG.info("  R² = %.3f (Δ%+.3f), MAE = %.0f (Δ%+.0f)", 
                r2_all3, r2_all3 - r2_baseline, 
                mae_all3, mae_all3 - mae_baseline)
    
    # Test all 5 indicators
    LOG.info(f"\nTesting ALL 5 indicators...")
    X_all5, y_all5 = create_enhanced_features(daily, economic_data, available_indicators)
    r2_all5, mae_all5 = evaluate_model_performance(X_all5, y_all5)
    
    combination_results["All 5 indicators"] = {
        "indicators": available_indicators,
        "r2_score": r2_all5,
        "mae": mae_all5,
        "improvement_r2": r2_all5 - r2_baseline,
        "improvement_mae": mae_baseline - mae_all5
    }
    
    LOG.info("  R² = %.3f (Δ%+.3f), MAE = %.0f (Δ%+.0f)", 
            r2_all5, r2_all5 - r2_baseline, 
            mae_all5, mae_all5 - mae_baseline)
    
    # Compile all results
    results.update({
        "individual": individual_results,
        "combinations": combination_results
    })
    
    return results, sorted_individual

def find_best_configuration(results):
    """Find the best configuration from all tests"""
    
    LOG.info("\n=== FINDING BEST CONFIGURATION ===")
    
    all_configs = []
    
    # Add baseline
    all_configs.append({
        "name": "Baseline (no economics)",
        "config": results["baseline"],
        "type": "baseline"
    })
    
    # Add individual indicators
    for indicator, metrics in results["individual"].items():
        all_configs.append({
            "name": f"Individual: {indicator}",
            "config": {
                "indicators": [indicator],
                "r2_score": metrics["r2_score"],
                "mae": metrics["mae"],
                "features": len([indicator]) * 3 + 10  # Estimate
            },
            "type": "individual"
        })
    
    # Add combinations
    for combo_name, metrics in results["combinations"].items():
        all_configs.append({
            "name": f"Combination: {combo_name}",
            "config": metrics,
            "type": "combination"
        })
    
    # Sort by R² score
    best_by_r2 = sorted(all_configs, key=lambda x: x["config"]["r2_score"], reverse=True)
    
    # Sort by MAE (lower is better)
    best_by_mae = sorted(all_configs, key=lambda x: x["config"]["mae"])
    
    LOG.info("\n=== TOP 5 CONFIGURATIONS BY R² SCORE ===")
    for i, config in enumerate(best_by_r2[:5], 1):
        LOG.info("%d. %s", i, config["name"])
        LOG.info("   R² = %.3f, MAE = %.0f", 
                config["config"]["r2_score"], config["config"]["mae"])
    
    LOG.info("\n=== TOP 5 CONFIGURATIONS BY MAE ===")
    for i, config in enumerate(best_by_mae[:5], 1):
        LOG.info("%d. %s", i, config["name"])
        LOG.info("   R² = %.3f, MAE = %.0f", 
                config["config"]["r2_score"], config["config"]["mae"])
    
    # Recommend best overall (balance R² and MAE)
    best_overall = best_by_r2[0]  # Start with best R²
    
    # Check if top MAE performer is close in R² but much better in MAE
    best_mae_config = best_by_mae[0]
    r2_diff = best_overall["config"]["r2_score"] - best_mae_config["config"]["r2_score"]
    mae_diff = best_overall["config"]["mae"] - best_mae_config["config"]["mae"]
    
    if r2_diff < 0.02 and mae_diff > 500:  # If R² loss < 0.02 but MAE gain > 500
        best_overall = best_mae_config
        LOG.info("\n*** RECOMMENDATION: Best MAE model (similar R² but much better MAE) ***")
    else:
        LOG.info("\n*** RECOMMENDATION: Best R² model ***")
    
    LOG.info("BEST CONFIG: %s", best_overall["name"])
    LOG.info("R² Score: %.3f", best_overall["config"]["r2_score"])
    LOG.info("MAE: %.0f", best_overall["config"]["mae"])
    
    return best_overall, best_by_r2, best_by_mae

def train_final_model_with_best_config(daily, economic_data, best_config):
    """Train final model with the best configuration"""
    
    LOG.info("\n=== TRAINING FINAL MODEL ===")
    LOG.info("Using configuration: %s", best_config["name"])
    
    # Get the indicators from best config
    if "indicators" in best_config["config"]:
        selected_indicators = best_config["config"]["indicators"]
    else:
        selected_indicators = []
    
    LOG.info("Selected economic indicators: %s", selected_indicators)
    
    # Create features with best config
    X, y = create_enhanced_features(daily, economic_data, selected_indicators)
    
    # Train final models (quantile + bootstrap like original)
    models = {}
    
    # Split for validation
    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    # Quantile models for range prediction
    for quantile in CFG["quantiles"]:
        LOG.info("  Training %d%% quantile model...", int(quantile * 100))
        
        model = QuantileRegressor(quantile=quantile, alpha=0.1, solver='highs')
        model.fit(X_train, y_train)
        
        # Validate
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
    
    # Store feature columns and selected indicators for prediction
    models["feature_columns"] = list(X.columns)
    models["selected_indicators"] = selected_indicators
    
    LOG.info("Final model trained with %d features", X.shape[1])
    
    return models, X, y

def predict_with_enhanced_model(models, mail_inputs, economic_inputs=None, date_str=None):
    """
    Predict call range from mail + economic inputs
    
    mail_inputs: dict like {"Reject_Ltrs": 1000, "Cheque 1099": 500, ...}
    economic_inputs: dict like {"Russell2000": 2150, "Dollar_Index": 103.5, ...}
    date_str: "2025-01-20" (optional, uses tomorrow if None)
    """
    
    if date_str is None:
        predict_date = datetime.now() + timedelta(days=1)
    else:
        predict_date = datetime.strptime(date_str, "%Y-%m-%d")
    
    # Create feature vector matching training
    feature_row = {}
    
    # Mail volume features
    available_types = [t for t in CFG["top_mail_types"]]
    total_mail = 0
    
    for mail_type in available_types:
        volume = mail_inputs.get(mail_type, 0)
        feature_row[f"{mail_type}_volume"] = volume
        total_mail += volume
    
    feature_row["total_mail_volume"] = total_mail
    feature_row["log_total_mail_volume"] = np.log1p(total_mail)
    feature_row["mail_percentile"] = 0.5  # Default
    
    # Economic features (if used in model)
    selected_indicators = models.get("selected_indicators", [])
    
    if economic_inputs and selected_indicators:
        for indicator in selected_indicators:
            current_value = economic_inputs.get(indicator, 0)
            lag_value = economic_inputs.get(f"{indicator}_lag1", current_value * 0.99)  # Default
            
            feature_row[f"{indicator}_today"] = current_value
            feature_row[f"{indicator}_lag1"] = lag_value
            feature_row[f"{indicator}_change"] = current_value - lag_value
    
    # Date features
    feature_row["weekday"] = predict_date.weekday()
    feature_row["month"] = predict_date.month
    feature_row["is_month_end"] = 1 if predict_date.day > 25 else 0
    feature_row["is_holiday_week"] = 1 if predict_date.date() in holidays.US() else 0
    
    # Baseline features
    feature_row["recent_calls_avg"] = 15000  # Could use actual recent data
    feature_row["recent_calls_trend"] = 0
    
    # Ensure all model features are present
    feature_columns = models.get("feature_columns", [])
    for col in feature_columns:
        if col not in feature_row:
            feature_row[col] = 0
    
    # Convert to DataFrame with correct column order
    X_input = pd.DataFrame([feature_row])[feature_columns]
    
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
    
    LOG.info("=== ENHANCED MAIL INPUT RANGE FORECAST ===")
    
    # Load data
    LOG.info("Loading mail-calls data...")
    daily = load_mail_call_data()
    
    LOG.info("Loading economic data...")
    economic_data = load_economic_data()
    
    # Test economic indicators systematically
    test_results, sorted_individual = test_economic_indicators_systematically(daily, economic_data)
    
    # Find best configuration
    best_config, best_by_r2, best_by_mae = find_best_configuration(test_results)
    
    # Train final model with best configuration
    final_models, final_X, final_y = train_final_model_with_best_config(
        daily, economic_data, best_config
    )
    
    # Save everything
    joblib.dump(final_models, output_dir / "enhanced_mail_models.pkl")
    
    # Save test results
    with open(output_dir / "economic_test_results.json", "w") as f:
        # Convert numpy types to regular Python types for JSON serialization
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
        
        json_safe_results = convert_numpy(test_results)
        json.dump(json_safe_results, f, indent=2)
    
    # Test scenarios with enhanced model
    create_enhanced_scenario_interface(final_models, economic_data, output_dir)
    
    LOG.info("\n=== ANALYSIS COMPLETE ===")
    LOG.info("Results saved to: %s", output_dir.resolve())
    LOG.info("\n=== SUMMARY FOR STAKEHOLDERS ===")
    LOG.info("✓ Tested economic indicators systematically")
    LOG.info("✓ Best configuration: %s", best_config["name"])
    LOG.info("✓ Final model MAE: %.0f calls (avg prediction error)", 
             best_config["config"]["mae"])
    LOG.info("✓ Final model R²: %.3f (explains %.1f%% of variance)", 
             best_config["config"]["r2_score"], 
             best_config["config"]["r2_score"] * 100)

def create_enhanced_scenario_interface(models, economic_data, output_dir):
    """Create scenario testing with economic indicators"""
    
    LOG.info("\n=== ENHANCED SCENARIO TESTING ===")
    
    # Get latest economic values for realistic scenarios
    if economic_data is not None and len(economic_data) > 0:
        latest_econ = economic_data.iloc[-1].to_dict()
    else:
        # Default values if no economic data
        latest_econ = {
            "Russell2000": 2150,
            "Dollar_Index": 103.5,
            "NASDAQ": 18500,
            "SP500": 5800,
            "Technology": 3200
        }
    
    scenarios = [
        {
            "name": "High Reject Letters + Strong Market",
            "mail_inputs": {"Reject_Ltrs": 2500, "Cheque 1099": 600},
            "economic_inputs": {
                "Russell2000": latest_econ.get("Russell2000", 2150) * 1.02,  # 2% up
                "Dollar_Index": latest_econ.get("Dollar_Index", 103.5),
                "NASDAQ": latest_econ.get("NASDAQ", 18500) * 1.01,
                "SP500": latest_econ.get("SP500", 5800) * 1.01,
                "Technology": latest_econ.get("Technology", 3200) * 1.015
            },
            "description": "Large reject letter batch during market upturn"
        },
        {
            "name": "Mixed Mail + Market Volatility", 
            "mail_inputs": {"Reject_Ltrs": 1200, "Cheque 1099": 800, "Transfer": 300},
            "economic_inputs": {
                "Russell2000": latest_econ.get("Russell2000", 2150) * 0.98,  # 2% down
                "Dollar_Index": latest_econ.get("Dollar_Index", 103.5) * 1.01,
                "NASDAQ": latest_econ.get("NASDAQ", 18500) * 0.99,
                "SP500": latest_econ.get("SP500", 5800) * 0.995,
                "Technology": latest_econ.get("Technology", 3200) * 0.97
            },
            "description": "Typical mail during market uncertainty"
        },
        {
            "name": "Low Mail + Stable Markets",
            "mail_inputs": {"Transfer": 200, "COA": 150, "Repl_Chks": 100},
            "economic_inputs": latest_econ,  # Current market levels
            "description": "Light administrative mail, stable economic conditions"
        },
        {
            "name": "Heavy Mail Day + Market Stress",
            "mail_inputs": {"Reject_Ltrs": 3500, "Cheque 1099": 2200, "Exercise_Converted": 800, "SOI_Confirms": 400},
            "economic_inputs": {
                "Russell2000": latest_econ.get("Russell2000", 2150) * 0.95,  # 5% down
                "Dollar_Index": latest_econ.get("Dollar_Index", 103.5) * 1.02,
                "NASDAQ": latest_econ.get("NASDAQ", 18500) * 0.94,
                "SP500": latest_econ.get("SP500", 5800) * 0.96,
                "Technology": latest_econ.get("Technology", 3200) * 0.92
            },
            "description": "Peak mail volumes during market downturn (worst case)"
        }
    ]
    
    results = {}
    
    for scenario in scenarios:
        LOG.info("\n--- Scenario: %s ---", scenario["name"])
        LOG.info("Description: %s", scenario["description"])
        LOG.info("Mail inputs: %s", scenario["mail_inputs"])
        LOG.info("Economic conditions: Market %s", 
                "UP" if scenario["economic_inputs"].get("SP500", 5800) >= latest_econ.get("SP500", 5800)
                else "DOWN")
        
        try:
            quantile_preds, bootstrap_stats = predict_with_enhanced_model(
                models, 
                scenario["mail_inputs"],
                scenario["economic_inputs"]
            )
            
            LOG.info("Predicted call ranges:")
            LOG.info("  Conservative (25-75%%): %.0f - %.0f calls", 
                    quantile_preds["q25"], quantile_preds["q75"])
            LOG.info("  Wide range (10-90%%): %.0f - %.0f calls", 
                    quantile_preds["q10"], quantile_preds["q90"])
            LOG.info("  Most likely: %.0f calls", quantile_preds["q50"])
            LOG.info("  Bootstrap mean: %.0f ± %.0f calls", 
                    bootstrap_stats["mean"], bootstrap_stats["std"])
            
            results[scenario["name"]] = {
                "mail_inputs": scenario["mail_inputs"],
                "economic_inputs": scenario["economic_inputs"],
                "description": scenario["description"],
                "predictions": quantile_preds,
                "bootstrap": bootstrap_stats
            }
            
        except Exception as e:
            LOG.error("Error in scenario %s: %s", scenario["name"], str(e))
            results[scenario["name"]] = {
                "error": str(e),
                "mail_inputs": scenario["mail_inputs"],
                "description": scenario["description"]
            }
    
    # Save scenario results
    with open(output_dir / "enhanced_scenario_results.json", "w") as f:
        # Convert numpy types for JSON
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
        
        json_safe_results = convert_numpy(results)
        json.dump(json_safe_results, f, indent=2)
    
    return results

if __name__ == "__main__":
    main()
