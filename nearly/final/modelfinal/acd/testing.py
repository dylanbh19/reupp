rD/OneDrive - Computershare/Desktop/acdmodel/testing.py"
2025-07-24 22:03:46 [INFO] - Results will be saved to: C:\Users\BhungarD\OneDrive - Computershare\Desktop\acdmodel\mail_call_prediction_system\rigorous_test_results
2025-07-24 22:03:46 [INFO] - Starting Rigorous Model Testing...
2025-07-24 22:03:46 [INFO] - Successfully loaded model 'forest_simple' from mail_call_prediction_system\models\best_model.pkl
2025-07-24 22:03:46 [INFO] - Loading and preparing data...
2025-07-24 22:03:47 [INFO] - All data loaded and merged successfully.
2025-07-24 22:03:47 [INFO] - Recreating features to match the trained model...
2025-07-24 22:03:50 [INFO] - Feature set recreated with 264 samples and 62 features.
2025-07-24 22:03:50 [INFO] - --- Starting Time Series Cross-Validation ---
2025-07-24 22:03:50 [INFO] - Fold 1/5 | Test R²: -0.484 | Test MAE: 1197.60
2025-07-24 22:03:51 [INFO] - Fold 2/5 | Test R²: -0.053 | Test MAE: 1312.14
2025-07-24 22:03:51 [INFO] - Fold 3/5 | Test R²: -0.105 | Test MAE: 4150.99
2025-07-24 22:03:51 [INFO] - Fold 4/5 | Test R²: -0.025 | Test MAE: 3158.11
2025-07-24 22:03:51 [INFO] - Fold 5/5 | Test R²: -0.336 | Test MAE: 1778.83
2025-07-24 22:03:51 [INFO] - --- Cross-Validation Summary ---
2025-07-24 22:03:51 [INFO] - Average R²: -0.201 (Std: 0.179)
2025-07-24 22:03:51 [INFO] - Average MAE: 2319.53 (Std: 1150.95)
2025-07-24 22:03:51 [INFO] - --- Analyzing Feature Importance ---
2025-07-24 22:03:51 [INFO] - Top 10 most important features:
2025-07-24 22:03:51 [INFO] -  1. month                          | Importance: 0.6558
2025-07-24 22:03:51 [INFO] -  2. calls_yesterday                | Importance: 0.0551
2025-07-24 22:03:51 [INFO] -  3. calls_avg7                     | Importance: 0.0454
2025-07-24 22:03:51 [INFO] -  4. Proxy(US)_avg7                 | Importance: 0.0367
2025-07-24 22:03:51 [INFO] -  5. calls_avg3                     | Importance: 0.0274
2025-07-24 22:03:51 [INFO] -  6. calls_2days_ago                | Importance: 0.0246
2025-07-24 22:03:51 [INFO] -  7. Proxy(US)_lag3                 | Importance: 0.0205
2025-07-24 22:03:51 [INFO] -  8. weekday                        | Importance: 0.0148
2025-07-24 22:03:51 [INFO] -  9. DRPStmt._lag2                  | Importance: 0.0100
2025-07-24 22:03:51 [INFO] - 10. Proxy(US)_lag2                 | Importance: 0.0097
2025-07-24 22:03:51 [INFO] - Feature importance plot saved to mail_call_prediction_system\rigorous_test_results\feature_importance.png
2025-07-24 22:03:51 [INFO] - --- Analyzing Error by Day of the Week ---
2025-07-24 22:03:51 [INFO] - Mean Absolute Error by Day of Week:
2025-07-24 22:03:51 [INFO] -    Mon: 2200.12
2025-07-24 22:03:51 [INFO] -    Tue: 1637.79
2025-07-24 22:03:51 [INFO] -    Wed: 1647.84
2025-07-24 22:03:51 [INFO] -    Thu: 2710.29
2025-07-24 22:03:51 [INFO] -    Fri: 1487.83
2025-07-24 22:03:51 [INFO] -    nan: 5185.61
2025-07-24 22:03:51 [INFO] -    nan: 8072.08
2025-07-24 22:03:52 [INFO] - Error analysis plot saved to mail_call_prediction_system\rigorous_test_results\error_by_day.png
2025-07-24 22:03:52 [INFO] - Rigorous testing complete.
PS C:\Users\BhungarD\OneDrive - Computershare\Desktop\acdm
























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
    # Test Settings
    "cv_splits": 5,
    "top_n_features": 20,
    # Settings from your original training script (must match)
    "top_mail_types_count": 8,
    "rolling_windows": [3, 7],
    "best_lag": 6, # From your EDA output
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
        # --- Load and merge data ---
        calls = pd.read_csv(CONFIG["call_file"])[['Date', 'ACDCalls']].rename(columns={'Date': 'date', 'ACDCalls': 'call_volume'})
        calls['date'] = pd.to_datetime(calls['date'])

        mail = pd.read_csv(CONFIG["mail_file"])
        mail['mail_date'] = pd.to_datetime(mail['mail_date'])
        mail = mail.rename(columns={'mail_date': 'date', 'mail_volume': 'volume', 'mail_type': 'type'})
        mail_pivot = mail.pivot_table(index='date', columns='type', values='volume', aggfunc='sum').fillna(0).reset_index()

        merged_data = pd.merge(calls, mail_pivot, on='date', how='inner')
        merged_data = merged_data.sort_values('date').reset_index(drop=True)
        logging.info("All data loaded and merged successfully.")

        # --- Recreate Features with Original Logic ---
        logging.info("Recreating features to match the trained model...")
        features_list = []
        targets_list = []
        
        top_mail_types = mail_pivot.drop(columns='date').sum().nlargest(CONFIG["top_mail_types_count"]).index.tolist()
        best_lag = CONFIG["best_lag"]
        max_lookback = max(CONFIG["rolling_windows"] + [best_lag])

        for i in range(max_lookback, len(merged_data) - best_lag):
            feature_row = {}
            current_date = merged_data.iloc[i]['date']

            # Mail Features (Original Logic)
            for mail_type in top_mail_types:
                clean_name = mail_type.replace(' ', '').replace('-', '').replace('_', '')[:15]
                feature_row[f"{clean_name}_today"] = merged_data.iloc[i][mail_type]
                for lag in [1, 2, 3]: feature_row[f"{clean_name}_lag{lag}"] = merged_data.iloc[i - lag][mail_type]
                for window in CONFIG["rolling_windows"]: feature_row[f"{clean_name}_avg{window}"] = merged_data[mail_type].iloc[i-window+1:i+1].mean()
            
            # Total Mail Features (Original Logic)
            total_mail_today = sum(merged_data.iloc[i][mt] for mt in top_mail_types)
            feature_row['total_mail_today'] = total_mail_today
            for lag in [1, 2, 3]: feature_row[f'total_mail_lag{lag}'] = sum(merged_data.iloc[i - lag][mt] for mt in top_mail_types)
            for window in CONFIG["rolling_windows"]: feature_row[f'total_mail_avg{window}'] = np.mean([sum(merged_data.iloc[j][mt] for mt in top_mail_types) for j in range(i-window+1, i+1)])

            # Call History Features (Original Logic)
            feature_row['calls_yesterday'] = merged_data.iloc[i - 1]['call_volume']
            feature_row['calls_2days_ago'] = merged_data.iloc[i - 2]['call_volume']
            for window in CONFIG["rolling_windows"]: feature_row[f'calls_avg{window}'] = merged_data['call_volume'].iloc[i-window+1:i+1].mean()

            # Temporal Features (Original Logic)
            feature_row['weekday'] = current_date.weekday()
            feature_row['month'] = current_date.month
            feature_row['day_of_month'] = current_date.day
            feature_row['is_month_end'] = 1 if current_date.day >= 25 else 0

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

    if set(X.columns) != set(model_features):
        logging.error("FATAL: Features created for testing do not match features the model was trained on.")
        logging.error(f"Model needs {len(model_features)} features, but test data has {len(X.columns)}.")
        missing_in_data = set(model_features) - set(X.columns)
        extra_in_data = set(X.columns) - set(model_features)
        if missing_in_data: logging.error(f"Missing from data: {missing_in_data}")
        if extra_in_data: logging.error(f"Extra in data: {extra_in_data}")
        return

    X = X[model_features]

    test_time_series_cv(model, X, y)
    analyze_feature_importance(model, model_features)
    analyze_error_by_day(model, X, y)
    
    logging.info("Rigorous testing complete.")

if __name__ == "__main__":
    main()
