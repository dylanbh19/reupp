#!/usr/bin/env python
# mail_input_range_forecast_economic.py
# =========================================================
# INPUT-DRIVEN range forecast model with ECONOMIC INDICATORS
# Users input: mail_type, volume, date -> get call range
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
import joblib
import yfinance as yf # <-- NEW: Import yfinance

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

LOG = logging.getLogger("mail_input_forecast_economic")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | mail_input_economic | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

CFG = {
    "top_mail_types": [
        "Reject_Ltrs", "Cheque 1099", "Exercise_Converted",
        "SOI_Confirms", "Exch_chks", "ACH_Debit_Enrollment",
        "Transfer", "COA", "NOTC_WITHDRAW", "Repl_Chks"
    ],
    "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
    "bootstrap_samples": 30,
    "output_dir": "dist_input_ranges_economic"
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
    """Load data and create mail->calls relationship dataset, now with economic data."""
    
    # (Existing mail and call loading logic is the same)
    mail_path = _find_file(["mail.csv", "data/mail.csv"])
    mail = pd.read_csv(mail_path)
    mail.columns = [c.lower().strip() for c in mail.columns]
    mail["mail_date"] = _to_date(mail["mail_date"])
    mail = mail.dropna(subset=["mail_date"])
    
    vol_path = _find_file(["callvolumes.csv", "data/callvolumes.csv"])
    intent_path = _find_file(["callintent.csv", "data/callintent.csv"])
    
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

    # --- NEW: Download and merge economic data ---
    LOG.info("Downloading historical economic data (VIX, S&P 500)...")
    start_date = daily.index.min().strftime('%Y-%m-%d')
    end_date = daily.index.max().strftime('%Y-%m-%d')
    
    econ_data = yf.download(
        ['^VIX', '^GSPC'],
        start=start_date,
        end=end_date,
        progress=False
    )
    
    econ_features = pd.DataFrame(index=econ_data.index)
    econ_features['VIX'] = econ_data['Close']['^VIX']
    econ_features['SP500_pct_change'] = econ_data['Close']['^GSPC'].pct_change()
    
    # Forward-fill and back-fill to handle non-trading days and NaNs
    econ_features = econ_features.ffill().bfill()
    
    # Join with the main daily dataset
    daily = daily.join(econ_features, how='left')
    daily[['VIX', 'SP500_pct_change']] = daily[['VIX', 'SP500_pct_change']].ffill().bfill()
    # --- END NEW ---
    
    LOG.info("Daily mail-calls-economic data: %s", daily.shape)
    return daily

def create_mail_input_features(daily):
    """Create features for mail input -> calls prediction, now with economic features."""
    
    features_list = []
    targets_list = []
    
    for i in range(len(daily) - 1):
        current_day = daily.iloc[i]
        next_day = daily.iloc[i + 1]
        
        feature_row = {}
        
        # Mail volumes (INPUT FEATURES)
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
        
        # --- NEW: Add economic features ---
        feature_row['vix'] = current_day['VIX']
        feature_row['sp500_pct_change'] = current_day['SP500_pct_change']
        # --- END NEW ---
        
        # Target: next day's calls
        target = next_day["calls_total"]
        
        features_list.append(feature_row)
        targets_list.append(target)
    
    X = pd.DataFrame(features_list)
    y = pd.Series(targets_list)
    
    X = X.fillna(0)
    
    LOG.info("Mail input features: %d samples x %d features", X.shape[0], X.shape[1])
    LOG.info("Feature columns: %s", sorted(list(X.columns)))
    
    return X, y

def train_mail_input_models(X, y):
    """Train models that predict calls from mail inputs (no changes here)."""
    
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
        model = RandomForestRegressor(n_estimators=50, max_depth=6, min_samples_leaf=3, random_state=i)
        model.fit(X_boot, y_boot)
        bootstrap_models.append(model)
    
    models["bootstrap_ensemble"] = bootstrap_models
    
    return models

def predict_from_mail_input(models, mail_inputs, date_str=None):
    """Predict call range from mail inputs, now fetching live economic data."""
    
    if date_str is None:
        # Mail is assumed to be sent today, for calls tomorrow.
        predict_date = datetime.now()
    else:
        predict_date = datetime.strptime(date_str, "%Y-%m-%d")

    # --- NEW: Fetch recent economic data for the prediction date ---
    LOG.info("Fetching live economic data for prediction date %s...", predict_date.strftime('%Y-%m-%d'))
    # Fetch a 7-day window to ensure we get the last trading day's data
    start_fetch_date = predict_date - timedelta(days=7)
    end_fetch_date = predict_date + timedelta(days=1)
    
    latest_econ_data = yf.download(
        ['^VIX', '^GSPC'],
        start=start_fetch_date,
        end=end_fetch_date,
        progress=False
    )
    
    if latest_econ_data.empty:
        LOG.warning("Could not fetch live economic data. Using default values (VIX=20, S&P500_change=0).")
        latest_vix = 20.0
        latest_sp500_change = 0.0
    else:
        # Get the very last available row of data
        last_day_data = latest_econ_data.iloc[-1]
        latest_vix = last_day_data[('Close', '^VIX')]
        # Calculate pct change using the last two days
        if len(latest_econ_data) > 1:
            sp_last_two_days = latest_econ_data['Close']['^GSPC'].iloc[-2:]
            latest_sp500_change = sp_last_two_days.pct_change().iloc[-1]
        else:
            latest_sp500_change = 0.0
    # --- END NEW ---

    # Create feature vector
    feature_row = {}
    
    # Mail volumes
    total_mail = 0
    for mail_type in CFG["top_mail_types"]:
        volume = mail_inputs.get(mail_type, 0)
        feature_row[f"{mail_type}_volume"] = volume
        total_mail += volume
    
    feature_row["total_mail_volume"] = total_mail
    feature_row["log_total_mail_volume"] = np.log1p(total_mail)
    feature_row["mail_percentile"] = 0.5  # Default
    
    # Date features
    feature_row["weekday"] = predict_date.weekday()
    feature_row["month"] = predict_date.month
    feature_row["is_month_end"] = 1 if predict_date.day > 25 else 0
    feature_row["is_holiday_week"] = 1 if predict_date.date() in holidays.US() else 0
    
    # Baseline features
    feature_row["recent_calls_avg"] = 15000  # Default
    feature_row["recent_calls_trend"] = 0
    
    # --- NEW: Add live economic data to feature row ---
    feature_row['vix'] = latest_vix
    feature_row['sp500_pct_change'] = latest_sp500_change
    # --- END NEW ---
    
    # Get feature order from a trained model to ensure consistency
    any_model = models["quantile_0.5"]
    X_input = pd.DataFrame([feature_row])[any_model.feature_names_in_]
    X_input = X_input.fillna(0)

    # Get quantile predictions
    quantile_preds = {}
    for quantile in CFG["quantiles"]:
        model = models[f"quantile_{quantile}"]
        pred = model.predict(X_input)[0]
        quantile_preds[f"q{int(quantile*100)}"] = max(0, pred)
    
    # Get bootstrap predictions
    bootstrap_preds = []
    for model in models["bootstrap_ensemble"]:
        pred = model.predict(X_input)[0]
        bootstrap_preds.append(max(0, pred))
    
    bootstrap_stats = {
        "mean": np.mean(bootstrap_preds), "std": np.std(bootstrap_preds),
        "min": np.min(bootstrap_preds), "max": np.max(bootstrap_preds)
    }
    
    return quantile_preds, bootstrap_stats

# The 'create_scenario_interface' and 'main' functions remain largely the same,
# but we'll update the name and directory.

def create_scenario_interface(models, output_dir):
    """Create simple interface for scenario testing."""
    
    scenarios = [
        {"name": "High Reject Letters", "mail_inputs": {"Reject_Ltrs": 2000, "Cheque 1099": 500}, "description": "Large batch of reject letters sent"},
        {"name": "Mixed Mail Day", "mail_inputs": {"Reject_Ltrs": 800, "Cheque 1099": 1200, "Exercise_Converted": 300}, "description": "Typical mixed mail campaign"},
        {"name": "Light Mail Day", "mail_inputs": {"Transfer": 200, "COA": 150}, "description": "Low volume administrative mail"},
        {"name": "Heavy Mail Day", "mail_inputs": {"Reject_Ltrs": 3000, "Cheque 1099": 2000, "Exercise_Converted": 800}, "description": "Major mail campaign day"}
    ]
    
    results = {}
    LOG.info("=== SCENARIO TESTING (with Economic Data) ===")
    
    for scenario in scenarios:
        LOG.info("\nScenario: %s", scenario["name"])
        LOG.info("Description: %s", scenario["description"])
        LOG.info("Mail inputs: %s", scenario["mail_inputs"])
        
        quantile_preds, bootstrap_stats = predict_from_mail_input(models, scenario["mail_inputs"])
        
        LOG.info("Predicted call ranges:")
        LOG.info("  Conservative (25-75%%): %.0f - %.0f calls", quantile_preds["q25"], quantile_preds["q75"])
        LOG.info("  Wide range (10-90%%): %.0f - %.0f calls", quantile_preds["q10"], quantile_preds["q90"])
        LOG.info("  Most likely: %.0f calls", quantile_preds["q50"])
        
        results[scenario["name"]] = {
            "mail_inputs": scenario["mail_inputs"], "description": scenario["description"],
            "predictions": quantile_preds, "bootstrap": bootstrap_stats
        }
    
    with open(output_dir / "scenario_results_economic.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

def main():
    output_dir = Path(CFG["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    LOG.info("=== MAIL INPUT RANGE FORECAST (WITH ECONOMIC DATA) ===")
    
    LOG.info("Loading mail-calls & economic data...")
    daily = load_mail_call_data()
    
    LOG.info("Creating mail input features...")
    X, y = create_mail_input_features(daily)
    
    LOG.info("Training mail input models...")
    models = train_mail_input_models(X, y)
    
    joblib.dump(models, output_dir / "mail_input_models_economic.pkl")
    
    scenario_results = create_scenario_interface(models, output_dir)
    
    LOG.info("\n=== INTERACTIVE EXAMPLE ===")
    LOG.info("Example: Predicting calls from specific mail campaign")
    
    example_input = {"Reject_Ltrs": 1500, "Cheque 1099": 800, "Exercise_Converted": 200}
    example_date = "2025-07-18" # Predict for calls on Monday, July 21st
    
    LOG.info("Input: %s for mail date %s", example_input, example_date)
    
    quantile_preds, bootstrap_stats = predict_from_mail_input(models, example_input, example_date)
    
    LOG.info("Output ranges:")
    LOG.info("  Most likely: %.0f calls", quantile_preds["q50"])
    LOG.info("  Business planning (25-75%%): %.0f - %.0f calls", quantile_preds["q25"], quantile_preds["q75"])
    LOG.info("  Capacity planning (10-90%%): %.0f - %.0f calls", quantile_preds["q10"], quantile_preds["q90"])
    
    LOG.info("\nModel ready for stakeholder use!")
    LOG.info("Results saved to: %s", output_dir.resolve())

if __name__ == "__main__":
    main()
