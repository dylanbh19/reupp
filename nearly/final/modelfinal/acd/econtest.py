rD/OneDrive - Computershare/Desktop/acdmodel/testingecon.py"
2025-07-24 22:32:56 [INFO] - Results will be saved to: C:\Users\BhungarD\OneDrive - Computershare\Desktop\acdmodel\mail_call_prediction_system\rigorous_test_results
2025-07-24 22:32:56 [INFO] - Starting Rigorous Model Testing...
2025-07-24 22:32:57 [INFO] - Successfully loaded model 'forest_simple' from mail_call_prediction_system\models\best_model.pkl
2025-07-24 22:32:57 [INFO] - Loading and preparing data...
2025-07-24 22:32:58 [INFO] - Loading and merging economic data...
c:\Users\BhungarD\OneDrive - Computershare\Desktop\acdmodel\testingecon.py:76: FutureWarning: DataFrame.fillna with 'method' is deprecated and will raise in a future version. Use obj.ffill() or obj.bfill() instead.    
  merged_data[economic_cols] = merged_data[economic_cols].fillna(method='ffill')
2025-07-24 22:32:58 [INFO] - All data loaded and merged successfully.
2025-07-24 22:32:58 [INFO] - Recreating features to match the trained model...
2025-07-24 22:32:59 [INFO] - Feature set recreated with 268 samples and 69 features.
2025-07-24 22:32:59 [ERROR] - FATAL: Features created for testing do not match features the model was trained on.
2025-07-24 22:32:59 [ERROR] - Model needs 69 features, but test data has 69.
2025-07-24 22:32:59 [ERROR] - Missing from data: {'NASDAQ_avg3', 'NASDAQ_avg7', 'DowJones_avg3', 'NASDAQ_lag3', 'DowJones_lag2', 'DowJones_avg7', 'DowJones_lag3', 'NASDAQ_lag1', 'DowJones_lag1', 'NASDAQ_lag2'}
2025-07-24 22:32:59 [ERROR] - Extra in data: {'Cheque1099_avg7', 'Cheque1099_avg3', 'Cheque1099_lag1', 'Cheque1099_lag3', 'DRP1099_avg3', 'DRP1099_lag1', 'DRP1099_lag3', 'DRP1099_lag2', 'Cheque1099_lag2', 'DRP1099_avg7'}









# File: testing.py (Corrected Version)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import logging
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.base import clone

# ============================================================================
# 1. CONFIGURATION
# ============================================================================
CONFIG = {
    "model_path": "mail_call_prediction_system/models/best_model.pkl",
    "output_dir": "mail_call_prediction_system/rigorous_test_results",
    # Data files needed to recreate the original feature set
    "call_file": "ACDMail.csv",
    "mail_file": "mail.csv",
    "holidays_file": "us_holidays.csv",
    "economic_data_file": "expanded_economic_data.csv", # <-- ADDED
    # Test Settings
    "cv_splits": 5,
    "top_n_features": 20,
    # Settings from your original training script (must match)
    "top_mail_types_count": 8,
    "rolling_windows": [3, 7],
    "best_lag": 1, # From your EDA output (best lag was 1, not 6)
}

# ============================================================================
# 2. LOGGING AND SETUP
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

output_path = Path(CONFIG["output_dir"])
output_path.mkdir(exist_ok=True)
logging.info(f"Results will be saved to: {output_path.resolve()}")


# ============================================================================
# 3. DATA PREPARATION FUNCTION
# ============================================================================
def load_and_prepare_data():
    """
    Loads data and re-engineers features to EXACTLY match the training environment.
    """
    logging.info("Loading and preparing data...")
    try:
        # --- Load and merge base data ---
        calls = pd.read_csv(CONFIG["call_file"])[['Date', 'ACDCalls']].rename(columns={'Date': 'date', 'ACDCalls': 'call_volume'})
        calls['date'] = pd.to_datetime(calls['date'])

        mail = pd.read_csv(CONFIG["mail_file"])
        mail['mail_date'] = pd.to_datetime(mail['mail_date'])
        mail = mail.rename(columns={'mail_date': 'date', 'mail_volume': 'volume', 'mail_type': 'type'})
        mail_pivot = mail.pivot_table(index='date', columns='type', values='volume', aggfunc='sum').fillna(0).reset_index()

        merged_data = pd.merge(calls, mail_pivot, on='date', how='inner')
        
        # --- (NEW) Load and merge economic data ---
        logging.info("Loading and merging economic data...")
        econ_df = pd.read_csv(CONFIG["economic_data_file"])
        econ_df['Date'] = pd.to_datetime(econ_df['Date'])
        econ_df = econ_df.rename(columns={'Date': 'date'})
        
        merged_data = pd.merge(merged_data, econ_df, on='date', how='left')
        
        # Replicate the forward-fill and dropna logic from training
        economic_cols = [col for col in econ_df.columns if col != 'date']
        merged_data[economic_cols] = merged_data[economic_cols].fillna(method='ffill')
        merged_data.dropna(subset=economic_cols, inplace=True)
        merged_data = merged_data.sort_values('date').reset_index(drop=True)
        logging.info("All data loaded and merged successfully.")

        # --- Recreate Features with Original Logic ---
        logging.info("Recreating features to match the trained model...")
        features_list = []
        targets_list = []
        
        # This logic should exactly mirror the FeatureEngineer class in the training script
        top_mail_types = mail_pivot.drop(columns='date').sum().nlargest(CONFIG["top_mail_types_count"]).index.tolist()
        best_lag = CONFIG["best_lag"]
        max_lookback = max(CONFIG["rolling_windows"] + [best_lag])

        for i in range(max_lookback, len(merged_data) - best_lag):
            feature_row = {}
            current_date = merged_data.iloc[i]['date']

            # (NEW) Economic Features
            for econ_col in economic_cols:
                if econ_col in merged_data.columns:
                    feature_row[econ_col] = merged_data.iloc[i][econ_col]

            # Mail Features (Original Logic)
            for mail_type in top_mail_types:
                clean_name = ''.join(filter(str.isalnum, mail_type))[:15]
                for lag in [1, 2, 3]: 
                    feature_row[f"{clean_name}_lag{lag}"] = merged_data.iloc[i - lag][mail_type]
                for window in CONFIG["rolling_windows"]: 
                    feature_row[f"{clean_name}_avg{window}"] = merged_data[mail_type].iloc[i-window+1:i+1].mean()
            
            # Call History Features (Original Logic)
            for lag in [1, 2, 3]:
                feature_row[f'calls_lag{lag}'] = merged_data.iloc[i - lag]['call_volume']
            for window in CONFIG["rolling_windows"]: 
                feature_row[f'calls_avg{window}'] = merged_data['call_volume'].iloc[i-window+1:i+1].mean()

            # Temporal Features (Original Logic)
            feature_row['weekday'] = current_date.weekday()
            feature_row['month'] = current_date.month
            feature_row['day_of_month'] = current_date.day
            
            features_list.append(feature_row)
            targets_list.append(merged_data.iloc[i + best_lag]['call_volume'])

        X = pd.DataFrame(features_list).fillna(0)
        y = pd.Series(targets_list, name='call_volume')
        
        logging.info(f"Feature set recreated with {len(X)} samples and {len(X.columns)} features.")
        return X, y

    except FileNotFoundError as e:
        logging.error(f"Data file not found: {e}. Cannot proceed with testing.")
        return None, None
    except Exception as e:
        logging.error(f"An error occurred during data preparation: {e}", exc_info=True)
        return None, None

# ============================================================================
# 4. TESTING FUNCTIONS (Unchanged)
# ============================================================================
def test_time_series_cv(model, X, y):
    """Performs Time Series Cross-Validation to get a stable measure of model performance."""
    logging.info("--- Starting Time Series Cross-Validation ---")
    tscv = TimeSeriesSplit(n_splits=CONFIG["cv_splits"])
    r2_scores, mae_scores = [], []

    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        preds = model_clone.predict(X_test)

        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        r2_scores.append(r2)
        mae_scores.append(mae)
        logging.info(f"Fold {i+1}/{CONFIG['cv_splits']} | Test R²: {r2:.3f} | Test MAE: {mae:.2f}")

    logging.info("--- Cross-Validation Summary ---")
    logging.info(f"Average R²: {np.mean(r2_scores):.3f} (Std: {np.std(r2_scores):.3f})")
    logging.info(f"Average MAE: {np.mean(mae_scores):.2f} (Std: {np.std(mae_scores):.2f})")

def analyze_feature_importance(model, features):
    """Extracts, plots, and logs the model's feature importances."""
    logging.info("--- Analyzing Feature Importance ---")
    if not hasattr(model, 'feature_importances_'):
        logging.warning("This model type does not support feature_importances_.")
        return

    importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
    
    logging.info("Top 10 most important features:")
    for i, (name, importance) in enumerate(importances.head(10).items()):
        logging.info(f"{i+1:2d}. {name:<30} | Importance: {importance:.4f}")

    plt.figure(figsize=(12, 8))
    importances.head(CONFIG["top_n_features"]).sort_values().plot(kind='barh', color='skyblue')
    plt.title(f'Top {CONFIG["top_n_features"]} Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    plot_path = output_path / "feature_importance.png"
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Feature importance plot saved to {plot_path}")

def analyze_error_by_day(model, X, y):
    """Analyzes model error broken down by the day of the week."""
    logging.info("--- Analyzing Error by Day of the Week ---")
    
    split_idx = int(len(X) * 0.75)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    model_clone = clone(model)
    model_clone.fit(X_train, y_train)
    preds = model_clone.predict(X_test)

    results_df = pd.DataFrame({'actual': y_test, 'predicted': preds, 'weekday': X_test['weekday']})
    results_df['absolute_error'] = (results_df['actual'] - results_df['predicted']).abs()

    mae_by_day = results_df.groupby('weekday')['absolute_error'].mean()
    day_map = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri'}
    mae_by_day.index = mae_by_day.index.map(day_map)

    logging.info("Mean Absolute Error by Day of Week:")
    for day, mae in mae_by_day.items():
        logging.info(f"   {day}: {mae:.2f}")
    
    plt.figure(figsize=(10, 6))
    mae_by_day.plot(kind='bar', color='coral')
    plt.title('Mean Absolute Error by Day of Week')
    plt.ylabel('Mean Absolute Error (Calls)')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plot_path = output_path / "error_by_day.png"
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Error analysis plot saved to {plot_path}")

# ============================================================================
# 5. MAIN ORCHESTRATOR
# ============================================================================
def main():
    """Main function to run all tests."""
    logging.info("Starting Rigorous Model Testing...")

    try:
        model_path = Path(CONFIG["model_path"])
        model_info = joblib.load(model_path)
        model = model_info['model']
        model_features = model_info['features']
        logging.info(f"Successfully loaded model '{model_info['model_name']}' from {model_path}")
    except FileNotFoundError:
        logging.error(f"Model file not found at {CONFIG['model_path']}. Aborting.")
        return

    X, y = load_and_prepare_data()
    if X is None:
        return

    # This check is critical and should now pass
    if set(X.columns) != set(model_features):
        logging.error("FATAL: Features created for testing do not match features the model was trained on.")
        logging.error(f"Model needs {len(model_features)} features, but test data has {len(X.columns)}.")
        missing_in_data = set(model_features) - set(X.columns)
        extra_in_data = set(X.columns) - set(model_features)
        if missing_in_data: logging.error(f"Missing from data: {missing_in_data}")
        if extra_in_data: logging.error(f"Extra in data: {extra_in_data}")
        return
    else:
        logging.info("Feature sets match between test data and trained model. Proceeding...")

    # Ensure column order is identical
    X = X[model_features]

    test_time_series_cv(model, X, y)
    analyze_feature_importance(model, model_features)
    analyze_error_by_day(model, X, y)
    
    logging.info("Rigorous testing complete.")

if __name__ == "__main__":
    main()
