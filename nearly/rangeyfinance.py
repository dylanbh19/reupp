#!/usr/bin/env python
# mail_input_range_forecast_local_data.py
# =================================================================
# ADVANCED INPUT-DRIVEN range forecast model with expanded economic
# indicators and feature engineering, using a local CSV data source.
# =================================================================

from pathlib import Path
import json
import logging
import sys
import warnings
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import holidays
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

LOG = logging.getLogger("mail_input_forecast_advanced_local")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | mail_input_local | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

CFG = {
    "top_mail_types": [
        "Reject_Ltrs", "Cheque 1099", "Exercise_Converted", "SOI_Confirms",
        "Exch_chks", "ACH_Debit_Enrollment", "Transfer", "COA",
        "NOTC_WITHDRAW", "Repl_Chks"
    ],
    "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
    "bootstrap_samples": 30,
    "output_dir": "dist_input_ranges_advanced_local",
    # --- CHANGED: Define the path to your local economic data CSV ---
    "econ_data_path": "economic_data.csv",
    # --- CHANGED: Define the column names from your CSV file ---
    "econ_columns": ["VIX", "SP500", "InterestRate_10Y", "FinancialSector"]
}

def _to_date(s):
    return pd.to_datetime(s, errors="coerce").dt.date

def _find_file(candidates):
    for p in candidates:
        path = Path(p)
        if path.exists():
            return path
    raise FileNotFoundError(f"None found: {candidates}")

def _create_advanced_economic_features(df):
    """
    Takes a dataframe with basic economic data and engineers advanced features.
    (This function remains the same, as it operates on a DataFrame)
    """
    LOG.info("Engineering advanced economic features (lags, rolling avgs, regimes)...")

    # Ensure all required columns are present
    for col in CFG["econ_columns"]:
        if col not in df.columns:
            raise ValueError(f"Missing required economic column in data: '{col}'")
    
    # Calculate daily percentage change for market indices
    df['SP500_pct_change'] = df['SP500'].pct_change()
    df['FinancialSector_pct_change'] = df['FinancialSector'].pct_change()

    # Feature Engineering
    features_to_engineer = ['VIX', 'SP500_pct_change', 'InterestRate_10Y', 'FinancialSector_pct_change']
    for col in features_to_engineer:
        for i in [1, 2]:
            df[f'{col}_lag{i}'] = df[col].shift(i)
        df[f'{col}_roll_avg5'] = df[col].rolling(window=5, min_periods=1).mean()

    df['VIX_is_high'] = (df['VIX'] > 30).astype(int)
    df['SP500_down_streak'] = (df['SP500_pct_change'] < 0).cumsum()
    t = df['SP500_down_streak'].ne(df['SP500_down_streak'].shift())
    df['SP500_consecutive_down_days'] = df.groupby(t.cumsum()).cumcount()

    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        LOG.info("Found and filled %d missing values in economic features.", nan_count)
        df.ffill(inplace=True)
        df.bfill(inplace=True)

    return df

# --- NEW: Function to load economic data from a local CSV file ---
def load_local_economic_data(file_path):
    """
    Loads historical economic data from a specified CSV file.
    The CSV must contain a 'Date' column and the columns specified in CFG["econ_columns"].
    """
    LOG.info("Loading historical economic data from '%s'...", file_path)
    econ_path = Path(file_path)
    if not econ_path.exists():
        LOG.error("Economic data file not found at '%s'. Please create this file.", file_path)
        LOG.error("See the documentation for the required format.")
        sys.exit(1)
    
    econ_data = pd.read_csv(econ_path, parse_dates=['Date'])
    econ_data.set_index('Date', inplace=True)
    return econ_data

def load_mail_call_data():
    """Load all data and create the main modeling dataset."""
    
    # (Mail and call loading logic is unchanged)
    mail_path = _find_file(["mail.csv", "data/mail.csv"]); mail = pd.read_csv(mail_path); mail.columns = [c.lower().strip() for c in mail.columns]; mail["mail_date"] = _to_date(mail["mail_date"]); mail = mail.dropna(subset=["mail_date"])
    vol_path = _find_file(["callvolumes.csv", "data/callvolumes.csv"]); df_vol = pd.read_csv(vol_path); df_vol.columns = [c.lower().strip() for c in df_vol.columns]; dcol_v = next(c for c in df_vol.columns if "date" in c); df_vol[dcol_v] = _to_date(df_vol[dcol_v]); vol_daily = df_vol.groupby(dcol_v)[df_vol.columns.difference([dcol_v])[0]].sum()
    intent_path = _find_file(["callintent.csv", "data/callintent.csv"]); df_int = pd.read_csv(intent_path); df_int.columns = [c.lower().strip() for c in df_int.columns]; dcol_i = next(c for c in df_int.columns if "date" in c or "conversationstart" in c); df_int[dcol_i] = _to_date(df_int[dcol_i]); int_daily = df_int.groupby(dcol_i).size()
    overlap = vol_daily.index.intersection(int_daily.index); scale = int_daily.loc[overlap].mean() / vol_daily.loc[overlap].mean(); vol_daily *= scale; calls_total = vol_daily.combine_first(int_daily).sort_index()
    mail_daily = (mail.groupby(["mail_date", "mail_type"], as_index=False)["mail_volume"].sum().pivot(index="mail_date", columns="mail_type", values="mail_volume").fillna(0))
    mail_daily.index = pd.to_datetime(mail_daily.index); calls_total.index = pd.to_datetime(calls_total.index)
    us_holidays = holidays.US(); biz_mask = (~mail_daily.index.weekday.isin([5, 6])) & (~mail_daily.index.isin(us_holidays)); mail_daily = mail_daily.loc[biz_mask]; calls_total = calls_total.loc[calls_total.index.isin(mail_daily.index)]
    daily = mail_daily.join(calls_total.rename("calls_total"), how="inner")
    
    # --- CHANGED: Load economic data from the local CSV file ---
    econ_data = load_local_economic_data(CFG["econ_data_path"])
        
    # --- Feature Engineering step (unchanged) ---
    econ_df_advanced = _create_advanced_economic_features(econ_data)
    
    daily = daily.join(econ_df_advanced, how='left').ffill().bfill()
    
    # Calculate and store long-term averages for robust defaults
    econ_defaults = daily[econ_df_advanced.columns].mean().to_dict()

    LOG.info("Daily mail-calls-economic data: %s", daily.shape)
    return daily, econ_defaults

def create_mail_input_features(daily):
    """Create the final X, y feature set for modeling."""
    # (This function is unchanged)
    all_feature_cols = [c for c in daily.columns if c not in ['calls_total'] and c in daily.select_dtypes(include=np.number).columns]
    X = daily.iloc[:-1][all_feature_cols].copy()
    y = daily.iloc[1:]['calls_total'].copy()
    common_index = X.index.intersection(y.index)
    X = X.loc[common_index]
    y = y.loc[common_index]
    LOG.info("Mail input features: %d samples x %d features", X.shape[0], X.shape[1])
    LOG.info("Feature columns sample: %s", list(X.columns[:5]))
    return X.fillna(0), y.fillna(0)

def train_mail_input_models(X, y, econ_defaults):
    """Train models and package with economic defaults."""
    # (This function is largely unchanged)
    LOG.info("Training mail input models...")
    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    models = {}
    for quantile in CFG["quantiles"]:
        LOG.info("  Training %d%% quantile model...", int(quantile * 100))
        model = QuantileRegressor(quantile=quantile, alpha=0.1, solver='highs')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        models[f"quantile_{quantile}"] = model
        LOG.info("    Validation MAE: %.0f", mae)
    
    LOG.info("  Training bootstrap ensemble...")
    bootstrap_models = []
    for i in range(CFG["bootstrap_samples"]):
        sample_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot, y_boot = X_train.iloc[sample_idx], y_train.iloc[sample_idx]
        model = RandomForestRegressor(n_estimators=50, max_depth=8, min_samples_leaf=3, random_state=i)
        model.fit(X_boot, y_boot)
        bootstrap_models.append(model)
    models["bootstrap_ensemble"] = bootstrap_models
    
    # Store defaults with the model for robust prediction
    models["econ_defaults"] = econ_defaults
    return models

def predict_from_mail_input(models, mail_inputs, date_str=None):
    """Predict call range, using local economic data for feature engineering."""
    
    predict_date = datetime.strptime(date_str, "%Y-%m-%d") if date_str else datetime.now()
    predict_date_floor = predict_date.normalize() # Ensure time is 00:00:00
    
    # --- CHANGED: Load local data to find features for the prediction date ---
    econ_defaults = models.get("econ_defaults", {})
    try:
        # Load the whole dataset to build features like lags/rolling avgs correctly
        historical_econ_data = load_local_economic_data(CFG["econ_data_path"])
        features_df = _create_advanced_economic_features(historical_econ_data)
        
        # Find the features for the specific prediction date
        if predict_date_floor in features_df.index:
            live_econ_features = features_df.loc[predict_date_floor].to_dict()
            LOG.info("Found economic data for %s in local file.", predict_date.strftime('%Y-%m-%d'))
        else:
            raise ValueError(f"Date {predict_date.strftime('%Y-%m-%d')} not found in economic data file.")
            
    except Exception as e:
        LOG.warning("Could not load or find economic data for prediction date: %s. Using long-term averages as fallback.", e)
        live_econ_features = econ_defaults

    # Create base feature vector
    feature_row = {}
    total_mail = 0
    for mail_type in CFG["top_mail_types"]:
        volume = mail_inputs.get(mail_type, 0)
        # Using a generic name based on the list, not hardcoded column names
        feature_row[mail_type] = volume # The training features are named after the mail_type directly
        total_mail += volume
    
    # Base features must match those created in `create_mail_input_features`
    # The original script didn't create these extra features in training.
    # To keep it consistent, we only add features that existed during training.
    
    # Combine mail features with live economic features
    feature_row.update(live_econ_features)
    
    # Get feature order from a trained model to ensure consistency
    any_model = models["quantile_0.5"]
    X_input = pd.DataFrame([feature_row])[any_model.feature_names_in_].fillna(0)
    
    # Generate predictions
    quantile_preds = {f"q{int(q*100)}": max(0, models[f"quantile_{q}"].predict(X_input)[0]) for q in CFG["quantiles"]}
    bootstrap_preds = [max(0, m.predict(X_input)[0]) for m in models["bootstrap_ensemble"]]
    bootstrap_stats = {"mean": np.mean(bootstrap_preds), "std": np.std(bootstrap_preds), "min": np.min(bootstrap_preds), "max": np.max(bootstrap_preds)}
    
    return quantile_preds, bootstrap_stats

def create_scenario_interface(models, output_dir):
    # This function remains the same
    scenarios = [
        {"name": "High Reject Letters", "mail_inputs": {"Reject_Ltrs": 2000, "Cheque 1099": 500}, "description": "Large batch of reject letters sent"},
        {"name": "Mixed Mail Day", "mail_inputs": {"Reject_Ltrs": 800, "Cheque 1099": 1200, "Exercise_Converted": 300}, "description": "Typical mixed mail campaign"},
    ]
    results = {}; LOG.info("=== SCENARIO TESTING (Local Data Model) ===")
    for scenario in scenarios:
        LOG.info("\nScenario: %s", scenario["name"])
        LOG.info("Mail inputs: %s", scenario["mail_inputs"])
        quantile_preds, _ = predict_from_mail_input(models, scenario["mail_inputs"])
        LOG.info("Predicted call ranges:")
        LOG.info("  Conservative (25-75%%): %.0f - %.0f calls", quantile_preds["q25"], quantile_preds["q75"])
        LOG.info("  Wide range (10-90%%): %.0f - %.0f calls", quantile_preds["q10"], quantile_preds["q90"])
        LOG.info("  Most likely: %.0f calls", quantile_preds["q50"])
        results[scenario["name"]] = {"mail_inputs": scenario["mail_inputs"], "predictions": quantile_preds}
    with open(output_dir / "scenario_results_advanced_local.json", "w") as f:
        json.dump(results, f, indent=2)
    return results

def main():
    output_dir = Path(CFG["output_dir"]); output_dir.mkdir(exist_ok=True)
    LOG.info("=== ADVANCED MAIL INPUT FORECAST (from Local Data) ===")
    
    daily, econ_defaults = load_mail_call_data()
    X, y = create_mail_input_features(daily)
    models = train_mail_input_models(X, y, econ_defaults)
    
    joblib.dump(models, output_dir / "mail_input_models_advanced_local.pkl")
    
    create_scenario_interface(models, output_dir)
    
    LOG.info("\n=== INTERACTIVE EXAMPLE ===")
    example_input = {"Reject_Ltrs": 1500, "Cheque 1099": 800}
    # Using today's date for the mail dispatch
    example_date = datetime.now().strftime("%Y-%m-%d")
    LOG.info("Input: %s for mail date %s", example_input, example_date)
    
    quantile_preds, _ = predict_from_mail_input(models, example_input, example_date)
    LOG.info("Output ranges for calls tomorrow:")
    LOG.info("  Most likely: %.0f calls", quantile_preds["q50"])
    LOG.info("  Business planning (25-75%%): %.0f - %.0f calls", quantile_preds["q25"], quantile_preds["q75"])
    LOG.info("  Capacity planning (10-90%%): %.0f - %.0f calls", quantile_preds["q10"], quantile_preds["q90"])
    
    LOG.info("\nModel ready for stakeholder use!")
    LOG.info("Results saved to: %s", output_dir.resolve())

if __name__ == "__main__":
    main()
