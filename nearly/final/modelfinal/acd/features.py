#!/usr/bin/env python
"""
Feature Investigator & Modeling Pipeline

This script starts from scratch to determine the predictive value of mail and economic data.
It is a complete, self-contained pipeline that:
1. Loads and merges all data sources, including multiple economic files.
2. Investigates raw and lagged correlations to find promising features.
3. Automatically selects the best features for modeling.
4. Trains a RandomForest model and evaluates its performance.
5. Generates plots for correlation, performance, and feature importance.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ============================================================================
# CONFIGURATION - Point to your data files here
# ============================================================================
CONFIG = {
    # --- DATA FILES ---
    "call_file": "ACDMail.csv",
    "mail_file": "mail.csv",
    # --- Add all your economic data files to this list ---
    "economic_data_files": [
        "expanded_economic_data.csv",
        "economic_data_for_model.csv" 
    ],
    
    # --- ANALYSIS & MODELING SETTINGS ---
    "output_dir": "feature_investigation_results",
    "target_column": "call_volume",
    "max_lag_days_for_corr": 14,       # How many days back to check for correlations
    "num_features_to_select": 25,      # How many of the best features to use in the model
    "feature_corr_threshold": 0.1,     # Minimum absolute correlation to be considered
    "model_test_size": 0.25,
    "random_state": 42,
}

# ============================================================================
# SETUP (Logging and Directories)
# ============================================================================
def setup_environment(output_dir_name):
    """Creates output directory and sets up logging."""
    output_dir = Path(output_dir_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / f"investigation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info(f"Results will be saved to: {output_dir.resolve()}")
    return output_dir

# ============================================================================
# STEP 1: LOAD AND MERGE ALL DATA
# ============================================================================
def load_and_merge_data():
    """Loads and merges call, mail, and all specified economic data files."""
    logging.info("STEP 1: Loading and merging all data sources...")
    
    # 1. Load Call Data
    try:
        call_df = pd.read_csv(CONFIG["call_file"])[["Date", "ACDCalls"]]
        call_df.columns = ['date', CONFIG["target_column"]]
        call_df['date'] = pd.to_datetime(call_df['date'])
    except Exception as e:
        logging.error(f"Failed to load call data: {e}")
        return None

    # 2. Load and Pivot Mail Data
    try:
        mail_df = pd.read_csv(CONFIG["mail_file"], low_memory=False)
        mail_df.columns = [str(c).lower().strip().replace(' ', '_') for c in mail_df.columns]
        date_col = next(c for c in mail_df.columns if 'date' in c)
        mail_df[date_col] = pd.to_datetime(mail_df[date_col])
        mail_pivot = mail_df.pivot_table(index=date_col, columns='mail_type', values='mail_volume', aggfunc='sum').fillna(0)
    except Exception as e:
        logging.error(f"Failed to load mail data: {e}")
        return None

    # 3. Merge Call and Mail Data
    merged_df = pd.merge(call_df, mail_pivot, left_on='date', right_index=True, how='inner')

    # 4. Load and Merge all Economic Data Files
    for econ_file in CONFIG["economic_data_files"]:
        try:
            econ_df = pd.read_csv(econ_file)
            econ_df.rename(columns=lambda c: 'date' if 'date' in c.lower() else c, inplace=True)
            econ_df['date'] = pd.to_datetime(econ_df['date'])
            merged_df = pd.merge(merged_df, econ_df, on='date', how='left', suffixes=('', '_dup'))
            # Drop duplicate columns that might arise from merging multiple files
            merged_df = merged_df[[c for c in merged_df.columns if not c.endswith('_dup')]]
            logging.info(f"Successfully merged '{econ_file}'")
        except FileNotFoundError:
            logging.warning(f"Economic data file not found, skipping: '{econ_file}'")
        except Exception as e:
            logging.error(f"Failed to merge '{econ_file}': {e}")
            
    # 5. Final Cleaning
    merged_df = merged_df.sort_values('date').reset_index(drop=True)
    # Forward-fill gaps (e.g., economic data on weekends)
    merged_df = merged_df.ffill().dropna()
    logging.info(f"Final merged dataset has {len(merged_df)} rows and {len(merged_df.columns)} columns.")
    
    return merged_df

# ============================================================================
# STEP 2: INVESTIGATE FEATURE CORRELATIONS
# ============================================================================
def investigate_feature_correlations(df, output_dir):
    """Calculates raw and lagged correlations to find the most promising features."""
    logging.info("STEP 2: Investigating feature correlations...")
    target = CONFIG["target_column"]
    features = [col for col in df.columns if col not in ['date', target]]
    
    lag_correlations = []
    for lag in range(CONFIG["max_lag_days_for_corr"] + 1):
        corrs = df[features].shift(lag).corrwith(df[target])
        corrs = corrs.dropna()
        for feature, corr_val in corrs.items():
            lag_correlations.append({'feature': feature, 'lag': lag, 'correlation': corr_val})
            
    corr_df = pd.DataFrame(lag_correlations)
    corr_df['abs_correlation'] = corr_df['correlation'].abs()
    corr_df = corr_df.sort_values('abs_correlation', ascending=False).reset_index(drop=True)

    # Find the best lag for each feature
    best_lag_df = corr_df.loc[corr_df.groupby('feature')['abs_correlation'].idxmax()]
    
    # Filter by threshold and select top N features
    strong_features = best_lag_df[best_lag_df['abs_correlation'] >= CONFIG["feature_corr_threshold"]]
    top_features = strong_features.head(CONFIG["num_features_to_select"])

    logging.info(f"Found {len(top_features)} features with correlation > {CONFIG['feature_corr_threshold']}.")
    logging.info("Top correlated features (feature, best_lag, correlation):")
    for _, row in top_features.iterrows():
        logging.info(f"  - {row['feature']} (lag {row['lag']}): {row['correlation']:.3f}")
        
    # --- Visualization ---
    plt.figure(figsize=(12, 8))
    sns.barplot(x='correlation', y='feature', data=top_features, palette='viridis')
    plt.title('Top Correlated Features with Call Volume (at Best Lag)', fontsize=16)
    plt.xlabel('Correlation')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig(output_dir / "1_top_feature_correlations.png")
    plt.close()
    
    return top_features

# ============================================================================
# STEP 3: FEATURE ENGINEERING FOR THE MODEL
# ============================================================================
def create_model_features(df, selected_features):
    """Creates lag and moving average features ONLY for the selected best predictors."""
    logging.info("STEP 3: Creating model features from selected predictors...")
    
    df_model = df[['date', CONFIG["target_column"]]].copy()
    
    for _, row in selected_features.iterrows():
        feature_name = row['feature']
        # Create a single lagged feature at its best lag
        df_model[f"{feature_name}_lag{row['lag']}"] = df[feature_name].shift(row['lag'])
        # Create a moving average feature
        df_model[f"{feature_name}_avg7"] = df[feature_name].shift(1).rolling(window=7).mean()

    # Add temporal features
    df_model['weekday'] = df_model['date'].dt.weekday
    df_model['day_of_year'] = df_model['date'].dt.dayofyear
    
    # Drop rows with NaNs created by shifting/rolling
    df_model = df_model.dropna().reset_index(drop=True)
    
    X = df_model.drop(columns=['date', CONFIG["target_column"]])
    y = df_model[CONFIG["target_column"]]
    
    logging.info(f"Created final modeling dataset with {len(X)} samples and {len(X.columns)} features.")
    return X, y

# ============================================================================
# STEP 4: TRAIN, EVALUATE, AND EXPLAIN THE MODEL
# ============================================================================
def train_and_evaluate_model(X, y, output_dir):
    """Trains a RandomForest model, evaluates it, and plots feature importances."""
    logging.info("STEP 4: Training and evaluating the model...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG["model_test_size"], random_state=CONFIG["random_state"], shuffle=False
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=CONFIG["random_state"], n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    logging.info(f"--- MODEL EVALUATION ---")
    logging.info(f"R-squared (R2): {r2:.3f}")
    logging.info(f"Mean Absolute Error (MAE): {mae:,.0f}")
    
    # --- Performance Plot ---
    plt.figure(figsize=(15, 7))
    plt.plot(y_test.index, y_test, label='Actual Calls', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted Calls', color='red', linestyle='--')
    plt.title(f'Model Performance (R2: {r2:.3f})', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "2_model_performance.png")
    plt.close()
    
    # --- Feature Importance Plot ---
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importances, palette='mako')
    plt.title('Top 20 Most Important Features in the Model', fontsize=16)
    plt.tight_layout()
    plt.savefig(output_dir / "3_feature_importances.png")
    plt.close()
    logging.info(f"Model explanation plots saved successfully.")


# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================
def main():
    """Runs the full investigation pipeline."""
    output_dir = setup_environment(CONFIG["output_dir"])
    
    merged_data = load_and_merge_data()
    if merged_data is None:
        logging.error("Pipeline stopped due to data loading failure.")
        return
        
    best_features = investigate_feature_correlations(merged_data, output_dir)
    if best_features.empty:
        logging.error("Pipeline stopped: No features met the correlation threshold.")
        return
        
    X, y = create_model_features(merged_data, best_features)
    
    train_and_evaluate_model(X, y, output_dir)
    
    logging.info("--- Feature Investigation Complete ---")

if __name__ == "__main__":
    main()

