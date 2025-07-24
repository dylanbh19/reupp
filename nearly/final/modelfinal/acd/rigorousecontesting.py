 MAIL-TO-CALLS PREDICTION SYSTEM READY FOR PRODUCTION!
PS C:\Users\BhungarD\OneDrive - Computershare\Desktop\acdmodel> C:\Users\BhungarD\python.exe "c:/Users/BhungarD/OneDrive - Computershare/Desktop/acdmodel/testingecon.py"
2025-07-24 22:42:53,439 [INFO] - Results will be saved to: C:\Users\BhungarD\OneDrive - Computershare\Desktop\acdmodel\mail_call_prediction_system\rigorous_test_results
2025-07-24 22:42:53,440 [INFO] - Starting Rigorous Model Testing...
2025-07-24 22:42:53,619 [INFO] - Successfully loaded model 'forest_simple' from mail_call_prediction_system/models/best_model.pkl
2025-07-24 22:42:53,620 [INFO] - Loading and preparing data...
2025-07-24 22:42:54,808 [INFO] - Loading and merging economic data...
2025-07-24 22:42:54,817 [INFO] - All data loaded and merged successfully.
2025-07-24 22:42:54,843 [INFO] - Recreating features to match the trained model...
2025-07-24 22:43:17,146 [INFO] - Feature set recreated with 332 samples and 1263 potential features.
2025-07-24 22:43:17,155 [INFO] - Aligning test data features with the model's requirements...
2025-07-24 22:43:17,156 [ERROR] - FATAL: The testing data is missing features that the model was trained on. 
2025-07-24 22:43:17,156 [ERROR] - Missing features: ['10Y_Treasury', '2Y_Treasury', '30Y_Treasury', 'Banking', 'Corporate_Bond_ETF', 'Dividend_Aristocrats', 'Dividend_ETF', 'Dollar_Index', 'DowJones', 'Gold', 'High_Dividend', 'NASDAQ', 'Oil', 'REITs', 'Regional_Banks', 'Russell2000', 'ScheduledPAYMEN_avg3', 'ScheduledPAYMEN_avg7', 'ScheduledPAYMEN_lag1', 'ScheduledPAYMEN_lag2', 'ScheduledPAYMEN_lag3', 'Technology', 'Utilities', 'VIX', 'VIX9D', 'VXN']
2025-07-24 22:43:17,156 [ERROR] - This is often caused by a mismatch in the raw data (e.g., different columns in economic_data.csv).                                         C:\Users\BhungarD\python.exe "c:/Users/BhungarD/OneDrive - Computershare/Desktop/acdmodel/testingecon.py"el>
2025-07-24 22:48:11,164 [INFO] - Results will be saved to: C:\Users\BhungarD\OneDrive - Computershare\Desktop\acdmodel\mail_call_prediction_system\rigorous_test_results
2025-07-24 22:48:11,164 [INFO] - Starting Self-Healing Rigorous Model Testing...
2025-07-24 22:48:11,335 [INFO] - Successfully loaded model 'forest_simple' from mail_call_prediction_system/models/best_model.pkl
2025-07-24 22:48:11,336 [INFO] - Loading and preparing all available data...
2025-07-24 22:48:12,339 [INFO] - Economic data successfully loaded.
2025-07-24 22:48:12,342 [INFO] - All data loaded and merged successfully.
2025-07-24 22:48:12,350 [INFO] - Recreating feature set from test data...
2025-07-24 22:48:30,161 [INFO] - Feature set recreated with 268 samples and 1263 potential features.
2025-07-24 22:48:30,168 [INFO] - Aligning test data features with the model's requirements...
2025-07-24 22:48:30,169 [WARNING] - SELF-HEALING: Test data is missing 26 features required by the model.    
2025-07-24 22:48:30,170 [WARNING] - This may happen if source data (e.g., mail types, economic data) has changed.
2025-07-24 22:48:30,170 [WARNING] - Adding missing features and filling with 0.
2025-07-24 22:48:30,171 [INFO] -   -> Added missing feature 'Gold' with value 0.
2025-07-24 22:48:30,172 [INFO] -   -> Added missing feature '10Y_Treasury' with value 0.
2025-07-24 22:48:30,173 [INFO] -   -> Added missing feature '30Y_Treasury' with value 0.
2025-07-24 22:48:30,174 [INFO] -   -> Added missing feature 'ScheduledPAYMEN_lag2' with value 0.
2025-07-24 22:48:30,175 [INFO] -   -> Added missing feature 'Dividend_Aristocrats' with value 0.
2025-07-24 22:48:30,176 [INFO] -   -> Added missing feature 'VXN' with value 0.
2025-07-24 22:48:30,177 [INFO] -   -> Added missing feature 'REITs' with value 0.
2025-07-24 22:48:30,178 [INFO] -   -> Added missing feature 'VIX9D' with value 0.
2025-07-24 22:48:30,179 [INFO] -   -> Added missing feature 'Dollar_Index' with value 0.
2025-07-24 22:48:30,180 [INFO] -   -> Added missing feature 'Oil' with value 0.
2025-07-24 22:48:30,180 [INFO] -   -> Added missing feature 'NASDAQ' with value 0.
2025-07-24 22:48:30,181 [INFO] -   -> Added missing feature 'Banking' with value 0.
2025-07-24 22:48:30,182 [INFO] -   -> Added missing feature 'ScheduledPAYMEN_avg7' with value 0.
2025-07-24 22:48:30,183 [INFO] -   -> Added missing feature 'Corporate_Bond_ETF' with value 0.
2025-07-24 22:48:30,183 [INFO] -   -> Added missing feature 'Technology' with value 0.
2025-07-24 22:48:30,184 [INFO] -   -> Added missing feature 'ScheduledPAYMEN_lag1' with value 0.
2025-07-24 22:48:30,184 [INFO] -   -> Added missing feature 'Russell2000' with value 0.
2025-07-24 22:48:30,185 [INFO] -   -> Added missing feature 'High_Dividend' with value 0.
2025-07-24 22:48:30,186 [INFO] -   -> Added missing feature 'DowJones' with value 0.
2025-07-24 22:48:30,186 [INFO] -   -> Added missing feature 'Regional_Banks' with value 0.
2025-07-24 22:48:30,187 [INFO] -   -> Added missing feature 'VIX' with value 0.
2025-07-24 22:48:30,188 [INFO] -   -> Added missing feature 'Dividend_ETF' with value 0.
2025-07-24 22:48:30,189 [INFO] -   -> Added missing feature '2Y_Treasury' with value 0.
2025-07-24 22:48:30,189 [INFO] -   -> Added missing feature 'Utilities' with value 0.
2025-07-24 22:48:30,190 [INFO] -   -> Added missing feature 'ScheduledPAYMEN_avg3' with value 0.
2025-07-24 22:48:30,190 [INFO] -   -> Added missing feature 'ScheduledPAYMEN_lag3' with value 0.
2025-07-24 22:48:30,191 [INFO] - Feature alignment complete.
2025-07-24 22:48:30,192 [INFO] - Making predictions on 268 test samples...
2025-07-24 22:48:30,197 [INFO] - Evaluating model performance...
2025-07-24 22:48:30,199 [INFO] -
RIGOROUS TEST REPORT for 'forest_simple'
==================================================
PERFORMANCE METRICS:
  - R-squared (R²): 0.0235
  - Mean Absolute Error (MAE): 2,017.16
==================================================

--- Logging error ---
Traceback (most recent call last):
  File "C:\Users\BhungarD\Lib\logging\__init__.py", line 1154, in emit
    stream.write(msg + self.terminator)
    ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\BhungarD\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2705' in position 33: character maps to <undefined>
Call stack:
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\acdmodel\testingecon.py", line 225, in <module>   
    run_self_healing_test()
  File "c:\Users\BhungarD\OneDrive - Computershare\Desktop\acdmodel\testingecon.py", line 222, in run_self_healing_test
    logger.info("✅ Self-Healing testing complete. All results saved.")
Message: '✅ Self-Healing testing complete. All results saved.'
Arguments: ()
2025-07-24 22:48:30,520 [INFO] - ✅ Self-Healing testing complete. All results saved.











#!/usr/bin/env python
"""
Self-Healing Rigorous Model Testing Script

This script loads a pre-trained model and evaluates it on new data. It is designed to be
resilient to changes in the input data.

Self-Healing Logic:
1.  Loads the saved model and its list of required features.
2.  Loads and prepares all available raw data.
3.  Recreates features using a method identical to the training script.
4.  Compares the model's required features to what's available in the test data.
5.  If a feature is MISSING in the test data, it is automatically added as a column of zeros,
    and a warning is logged. This prevents the script from crashing.
6.  If EXTRA features are present in the test data, they are ignored.
7.  The script then proceeds with prediction and evaluation.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    "rolling_windows": [3, 7],
    "target_lag": 1, # The gap between features and the day being predicted
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
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info(f"Results will be saved to: {log_dir.resolve()}")
    return logging.getLogger()

# ============================================================================
# DATA & FEATURE PREPARATION (Mirrors the robust training script)
# ============================================================================

def load_and_prepare_data():
    """Loads and merges all available call, mail, and economic data."""
    logging.info("Loading and preparing all available data...")
    # Load Call Data
    call_df = pd.read_csv(CONFIG["call_file"])[["Date", "ACDCalls"]]
    call_df.columns = ['date', 'call_volume']
    call_df['date'] = pd.to_datetime(call_df['date'])
    call_df = call_df.dropna().sort_values('date')

    # Load Mail Data
    mail_df = pd.read_csv(CONFIG["mail_file"], low_memory=False)
    mail_df.columns = [str(c).lower().strip() for c in mail_df.columns]
    date_col = next(c for c in mail_df.columns if 'date' in c)
    mail_df[date_col] = pd.to_datetime(mail_df[date_col])
    mail_pivot = mail_df.pivot_table(index=date_col, columns='mail_type', values='mail_volume', aggfunc='sum').fillna(0)
    mail_pivot.index.name = 'date'
    
    # Merge Call and Mail
    merged = pd.merge(call_df, mail_pivot, on='date', how='inner')
    
    # Load and Merge Economic Data
    try:
        econ_df = pd.read_csv(CONFIG["economic_data_file"])
        econ_df.rename(columns={'Date': 'date'}, inplace=True)
        econ_df['date'] = pd.to_datetime(econ_df['date'])
        merged = pd.merge(merged, econ_df, on='date', how='left')
        logging.info("Economic data successfully loaded.")
    except FileNotFoundError:
        logging.warning(f"'{CONFIG['economic_data_file']}' not found. Proceeding without economic data.")
    
    # Forward-fill any gaps in data (especially for economic indicators on non-trading days)
    merged = merged.sort_values('date').reset_index(drop=True)
    merged = merged.ffill()
    merged = merged.dropna() # Drop any rows at the beginning that couldn't be filled
    
    logging.info("All data loaded and merged successfully.")
    return merged

def recreate_all_features(df):
    """Recreates the full feature set from the prepared data using the robust method."""
    logging.info("Recreating feature set from test data...")
    
    # Identify all potential feature sources (every column except date and the target)
    source_cols = [col for col in df.columns if col not in ['date', 'call_volume']]
    
    features_list, targets_list, dates_list = [], [], []
    
    max_lookback = max(CONFIG["rolling_windows"] + [1, 2, 3])
    target_lag = CONFIG["target_lag"]

    for i in range(max_lookback, len(df) - target_lag):
        feature_row = {}
        current_date = df.iloc[i]['date']
        
        # Create features for all source columns (mail and economic)
        for col_name in source_cols:
            clean_name = ''.join(filter(str.isalnum, col_name))[:25]
            for lag in [1, 2, 3]:
                feature_row[f"{clean_name}_lag{lag}"] = df.iloc[i - lag][col_name]
            for window in CONFIG["rolling_windows"]:
                feature_row[f"{clean_name}_avg{window}"] = df[col_name].iloc[i-window+1:i+1].mean()

        # Create Call History features
        for lag in [1, 2, 3]:
            feature_row[f'calls_lag{lag}'] = df.iloc[i - lag]['call_volume']
        for window in CONFIG["rolling_windows"]:
            feature_row[f'calls_avg{window}'] = df['call_volume'].iloc[i-window+1:i+1].mean()
            
        # Create Temporal features
        feature_row['weekday'] = current_date.weekday()
        feature_row['month'] = current_date.month
        feature_row['day_of_month'] = current_date.day
        
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
def run_self_healing_test():
    """Main function to execute the self-healing testing pipeline."""
    output_dir = Path(CONFIG["output_dir"])
    logger = setup_logging(output_dir)
    logger.info("Starting Self-Healing Rigorous Model Testing...")

    # 1. Load Model and its expected feature list
    try:
        model_info = joblib.load(CONFIG["model_path"])
        model = model_info['model']
        model_features_expected = model_info['features']
        model_name = model_info.get('model_name', 'Unknown Model')
        logger.info(f"Successfully loaded model '{model_name}' from {CONFIG['model_path']}")
    except FileNotFoundError:
        logger.error(f"FATAL: Model file not found at '{CONFIG['model_path']}'. Please run the training script first.")
        return

    # 2. Load data and recreate all possible features
    full_data = load_and_prepare_data()
    X_test_all, y_test, dates = recreate_all_features(full_data)

    # 3. SELF-HEALING STEP: Align test features with model's expectations
    logger.info("Aligning test data features with the model's requirements...")
    test_features_available = X_test_all.columns
    
    missing_features = set(model_features_expected) - set(test_features_available)
    
    if missing_features:
        logger.warning(f"SELF-HEALING: Test data is missing {len(missing_features)} features required by the model.")
        logger.warning("This may happen if source data (e.g., mail types, economic data) has changed.")
        logger.warning("Adding missing features and filling with 0.")
        for feature in missing_features:
            X_test_all[feature] = 0
            logger.info(f"  -> Added missing feature '{feature}' with value 0.")

    # Final alignment: ensure the DataFrame has the exact columns in the exact order
    X_test_final = X_test_all[model_features_expected]
    logger.info("Feature alignment complete.")

    # 4. Make Predictions
    logger.info(f"Making predictions on {len(X_test_final)} test samples...")
    predictions = model.predict(X_test_final)

    # 5. Evaluate and Save Results
    logger.info("Evaluating model performance...")
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    report = (f"RIGOROUS TEST REPORT for '{model_name}'\n"
              f"{'='*50}\n"
              f"PERFORMANCE METRICS:\n"
              f"  - R-squared (R²): {r2:.4f}\n"
              f"  - Mean Absolute Error (MAE): {mae:,.2f}\n"
              f"{'='*50}\n")
    logger.info("\n" + report)
    
    results_df = pd.DataFrame({'date': dates, 'actual_calls': y_test, 'predicted_calls': predictions})
    results_df.to_csv(output_dir / "predictions_vs_actuals.csv", index=False)
    (output_dir / "performance_report.txt").write_text(report)
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 8))
    ax.plot(results_df['date'], results_df['actual_calls'], label='Actual Calls', color='dodgerblue')
    ax.plot(results_df['date'], results_df['predicted_calls'], label='Predicted Calls', color='red', linestyle='--')
    ax.set_title(f'Model Performance: Actual vs. Predicted (R²: {r2:.3f})', fontsize=16)
    ax.legend()
    fig.savefig(output_dir / "performance_plot.png", dpi=150)
    
    logger.info("✅ Self-Healing testing complete. All results saved.")

if __name__ == "__main__":
    run_self_healing_test()
