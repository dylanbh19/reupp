#!/usr/bin/env python
"""
Rigorous Model Testing Script

This script loads a pre-trained model and evaluates it against a full dataset.
It is designed to replicate the exact feature engineering process used during training
to ensure there are no feature mismatches.

Steps:
1.  Load the saved model and its required feature list.
2.  Load and prepare the raw call, mail, and economic data.
3.  Recreate all possible features (lags, rolling averages, temporal).
4.  Align the newly created features with the model's expected features.
5.  Run predictions and evaluate the model's performance.
6.  Save the results and a detailed report.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    "model_path": "mail_call_prediction_system/models/best_model.pkl",
    "call_file": "ACDMail.csv",
    "mail_file": "mail.csv",
    "economic_data_file": "expanded_economic_data.csv",
    "output_dir": "mail_call_prediction_system/rigorous_test_results",
    
    # Feature engineering settings (MUST MATCH TRAINING SCRIPT)
    "max_lag_days": 7,
    "rolling_windows": [3, 7],
}

# ============================================================================
# LOGGING SETUP
# ============================================================================
def setup_logging(output_dir):
    """Sets up a logger to save output to a file and print to console."""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"test_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Results will be saved to: {log_dir.resolve()}")
    return logging.getLogger()

# ============================================================================
# DATA PREPARATION (Mirrors the training script)
# ============================================================================

def remove_us_holidays(df, date_col='date'):
    """Remove US holidays from the DataFrame using a pre-generated CSV file."""
    try:
        holidays_df = pd.read_csv("us_holidays.csv")
        holiday_dates_to_remove = set(holidays_df['holiday_date'])
        holiday_mask = df[date_col].dt.strftime('%Y-%m-%d').isin(holiday_dates_to_remove)
        return df[~holiday_mask].copy()
    except FileNotFoundError:
        logging.warning("'us_holidays.csv' not found. Skipping holiday removal.")
        return df

def load_and_prepare_data():
    """Loads and merges call, mail, and economic data."""
    logging.info("Loading and preparing data...")
    # 1. Load Call Data
    call_df = pd.read_csv(CONFIG["call_file"])
    call_df = call_df[["Date", "ACDCalls"]].copy()
    call_df.columns = ['date', 'call_volume']
    call_df['date'] = pd.to_datetime(call_df['date'], errors='coerce')
    call_df = call_df.dropna(subset=['date', 'call_volume'])
    call_df = call_df[call_df['date'].dt.weekday < 5]
    call_df = remove_us_holidays(call_df, 'date')
    
    # 2. Load Mail Data
    mail_df = pd.read_csv(CONFIG["mail_file"], low_memory=False)
    mail_df.columns = [str(c).lower().strip() for c in mail_df.columns]
    date_col = next(c for c in mail_df.columns if 'date' in c)
    vol_col = next(c for c in mail_df.columns if 'volume' in c)
    type_col = next(c for c in mail_df.columns if 'type' in c)
    mail_df[date_col] = pd.to_datetime(mail_df[date_col], errors='coerce')
    mail_df = mail_df.dropna(subset=[date_col, vol_col])
    mail_df['mail_date'] = mail_df[date_col].dt.date
    daily_mail = mail_df.groupby(['mail_date', type_col])[vol_col].sum().reset_index()
    daily_mail.columns = ['date', 'mail_type', 'volume']
    daily_mail['date'] = pd.to_datetime(daily_mail['date'])
    mail_pivot = daily_mail.pivot_table(index='date', columns='mail_type', values='volume').fillna(0)
    
    # 3. Merge Call and Mail
    merged_data = pd.merge(call_df, mail_pivot, on='date', how='inner')
    
    # 4. Load and Merge Economic Data
    try:
        logging.info("Loading and merging economic data...")
        econ_df = pd.read_csv(CONFIG["economic_data_file"])
        econ_df['Date'] = pd.to_datetime(econ_df['Date'])
        econ_df.rename(columns={'Date': 'date'}, inplace=True)
        economic_cols = [col for col in econ_df.columns if col != 'date']
        
        merged_data = pd.merge(merged_data, econ_df, on='date', how='left')
        
        # Forward-fill missing economic data (e.g., for holidays/weekends)
        merged_data[economic_cols] = merged_data[economic_cols].ffill()
        
        # Drop any rows that still have NaNs after ffill (likely at the beginning)
        initial_rows = len(merged_data)
        merged_data.dropna(inplace=True)
        if initial_rows > len(merged_data):
            logging.warning(f"Dropped {initial_rows - len(merged_data)} rows with missing data at the start of the series.")

    except FileNotFoundError:
        logging.warning(f"'{CONFIG['economic_data_file']}' not found. Skipping economic features.")
    
    merged_data = merged_data.sort_values('date').reset_index(drop=True)
    logging.info("All data loaded and merged successfully.")
    return merged_data

def recreate_features(df):
    """
    Recreates the full feature set from the prepared data, mirroring the training script's logic.
    """
    logging.info("Recreating features to match the trained model...")
    
    # Identify all potential feature source columns (mail types and economic indicators)
    source_cols = [col for col in df.columns if col not in ['date', 'call_volume']]
    
    features_list = []
    targets_list = []
    dates_list = []
    
    # We need a buffer at the start for lags/rolling windows and at the end for the target
    max_lookback = max(CONFIG["rolling_windows"] + [lag for lag in range(1, CONFIG["max_lag_days"] + 1)])
    
    # Use a fixed lag for the target variable (assuming 1 day if not specified, which is common)
    target_lag = 1
    
    for i in range(max_lookback, len(df) - target_lag):
        feature_row = {}
        current_date = df.iloc[i]['date']
        
        # 1. Create features for all source columns (mail and economic)
        for col in source_cols:
            clean_name = ''.join(filter(str.isalnum, col))[:25] # Create a safe name
            for lag in [1, 2, 3]:
                feature_row[f"{clean_name}_lag{lag}"] = df.iloc[i - lag][col]
            for window in CONFIG["rolling_windows"]:
                # Use a centered window to match the likely training logic
                feature_row[f"{clean_name}_avg{window}"] = df[col].iloc[i-window+1:i+1].mean()

        # 2. Create Call History features
        for lag in [1, 2, 3]:
            feature_row[f'calls_lag{lag}'] = df.iloc[i - lag]['call_volume']
        for window in CONFIG["rolling_windows"]:
            feature_row[f'calls_avg{window}'] = df['call_volume'].iloc[i-window+1:i+1].mean()
            
        # 3. Create Temporal features
        feature_row['weekday'] = current_date.weekday()
        feature_row['month'] = current_date.month
        feature_row['day_of_month'] = current_date.day
        
        # 4. Define Target variable
        target = df.iloc[i + target_lag]['call_volume']
        
        features_list.append(feature_row)
        targets_list.append(target)
        dates_list.append(current_date)
        
    X = pd.DataFrame(features_list).fillna(0)
    y = pd.Series(targets_list, name='call_volume_target')
    dates = pd.Series(dates_list, name='date')
    
    logging.info(f"Feature set recreated with {len(X)} samples and {len(X.columns)} potential features.")
    return X, y, dates

# ============================================================================
# MAIN TESTING ORCHESTRATOR
# ============================================================================
def run_rigorous_test():
    """Main function to execute the rigorous testing pipeline."""
    output_dir = Path(CONFIG["output_dir"])
    logger = setup_logging(output_dir)
    logger.info("Starting Rigorous Model Testing...")

    # 1. Load Model
    try:
        model_info = joblib.load(CONFIG["model_path"])
        model = model_info['model']
        model_features_expected = model_info['features']
        model_name = model_info.get('model_name', 'Unknown Model')
        logger.info(f"Successfully loaded model '{model_name}' from {CONFIG['model_path']}")
    except FileNotFoundError:
        logger.error(f"FATAL: Model file not found at '{CONFIG['model_path']}'. Please run the training script first.")
        return
    except Exception as e:
        logger.error(f"FATAL: Error loading model file. It might be corrupted. Error: {e}")
        return

    # 2. Load and Prepare Data
    full_data = load_and_prepare_data()

    # 3. Recreate Feature Set from the full data
    X_test_all, y_test, dates = recreate_features(full_data)

    # 4. CRITICAL STEP: Align test features with model's expected features
    logger.info("Aligning test data features with the model's requirements...")
    test_features_available = X_test_all.columns
    
    missing_features = set(model_features_expected) - set(test_features_available)
    extra_features = set(test_features_available) - set(model_features_expected)

    if missing_features:
        logger.error("FATAL: The testing data is missing features that the model was trained on.")
        logger.error(f"Missing features: {sorted(list(missing_features))}")
        logger.error("This is often caused by a mismatch in the raw data (e.g., different columns in economic_data.csv).")
        return

    if extra_features:
        logger.warning(f"Found {len(extra_features)} extra features in test data. These will be ignored.")

    # Ensure the final DataFrame has the exact columns in the exact order the model expects
    try:
        X_test_final = X_test_all[model_features_expected]
        logger.info("Feature alignment successful.")
    except KeyError:
        logger.error("FATAL: A key error occurred during final feature alignment. This should not happen if missing_features check passed.")
        return

    # 5. Make Predictions
    logger.info(f"Making predictions on {len(X_test_final)} test samples...")
    predictions = model.predict(X_test_final)

    # 6. Evaluate Performance
    logger.info("Evaluating model performance...")
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    report = (
        f"RIGOROUS TEST REPORT for '{model_name}'\n"
        f"{'='*50}\n"
        f"Test Date Range: {dates.min().date()} to {dates.max().date()}\n"
        f"Number of Test Samples: {len(y_test)}\n"
        f"{'-'*50}\n"
        f"PERFORMANCE METRICS:\n"
        f"  - R-squared (R²):     {r2:.4f}\n"
        f"  - Mean Absolute Error (MAE): {mae:,.2f}\n"
        f"  - Root Mean Squared Error (RMSE): {rmse:,.2f}\n"
        f"{'='*50}\n"
    )
    logger.info("\n" + report)

    # 7. Save Results
    results_df = pd.DataFrame({
        'date': dates,
        'actual_calls': y_test,
        'predicted_calls': predictions,
        'error': y_test - predictions
    })
    results_df.to_csv(output_dir / "predictions_vs_actuals.csv", index=False)
    
    with open(output_dir / "performance_report.txt", "w") as f:
        f.write(report)
    
    # 8. Create and Save Plot
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.plot(results_df['date'], results_df['actual_calls'], label='Actual Calls', color='dodgerblue', linewidth=2)
    ax.plot(results_df['date'], results_df['predicted_calls'], label='Predicted Calls', color='red', linestyle='--', alpha=0.8)
    ax.set_title(f'Model Performance: Actual vs. Predicted Calls (R²: {r2:.3f})', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Call Volume')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    fig.savefig(output_dir / "performance_plot.png", dpi=150)
    
    logger.info("✅ Rigorous testing complete. All results saved.")


if __name__ == "__main__":
    run_rigorous_test()

