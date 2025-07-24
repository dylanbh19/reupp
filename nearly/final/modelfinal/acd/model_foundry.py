#!/usr/bin/env python
"""
Production-Grade Model Foundry

This script systematically evaluates multiple feature sets and modeling algorithms
to find the optimal, non-overfitting model for predicting call volumes.

Methodology:
1.  Loads and merges all data sources (call, mail, multiple economic files).
2.  Performs feature selection to identify the most correlated mail and economic indicators.
3.  Defines multiple feature "scenarios" (e.g., simple, complex).
4.  Defines a suite of regression models (linear, tree-based, gradient boosting).
5.  Uses Time Series Cross-Validation to robustly evaluate each model on each scenario.
6.  Logs all results and saves the best-performing model and a summary report.
"""
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
import joblib
from pathlib import Path
from datetime import datetime

# --- Model Imports ---
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
# Try to import LightGBM, a powerful gradient boosting model
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # --- File Paths ---
    "call_file": "ACDMail.csv",
    "mail_file": "mail.csv",
    "economic_data_files": ["expanded_economic_data.csv", "economic_data_for_model.csv"],
    
    # --- Core Settings ---
    "target_column": "call_volume",
    "output_dir_base": "model_foundry_runs",
    
    # --- Feature Selection ---
    "feature_corr_threshold": 0.1,
    "top_n_mail_features": 15,
    "top_n_econ_features": 10,
    "max_lag_for_selection": 14,
    
    # --- Modeling ---
    "timeseries_cv_splits": 5,
    "random_state": 42,
    "evaluation_metric": "r2",  # Metric to decide the 'best' model ('r2' or 'neg_mae')
}

# ============================================================================
# SETUP
# ============================================================================
def setup_environment():
    """Creates a timestamped output directory and sets up logging."""
    run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(CONFIG["output_dir_base"]) / run_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "foundry_log.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
    )
    logging.info(f"Run results will be saved to: {output_dir.resolve()}")
    return output_dir

# ============================================================================
# DATA LOADING CLASS
# ============================================================================
class DataLoader:
    """Handles loading, merging, and cleaning all data sources."""
    def load_and_merge(self):
        logging.info("--- Starting Data Loading and Merging ---")
        try:
            # Load and process call, mail, and economic data
            call_df = self._load_calls()
            mail_pivot = self._load_mail()
            merged_df = pd.merge(call_df, mail_pivot, left_on='date', right_index=True, how='inner')
            
            for econ_file in CONFIG["economic_data_files"]:
                econ_df = self._load_econ(econ_file)
                if econ_df is not None:
                    merged_df = pd.merge(merged_df, econ_df, on='date', how='left', suffixes=('', '_dup'))
            
            # Clean up final dataframe
            merged_df = merged_df[[c for c in merged_df.columns if not c.endswith('_dup')]]
            merged_df = merged_df.sort_values('date').reset_index(drop=True)
            merged_df = merged_df.ffill().dropna()
            
            logging.info(f"Final merged dataset ready: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
            return merged_df
        except Exception as e:
            logging.error(f"Critical error during data loading: {e}", exc_info=True)
            return None

    def _load_calls(self):
        df = pd.read_csv(CONFIG["call_file"])[["Date", "ACDCalls"]]
        df.columns = ['date', CONFIG["target_column"]]
        df['date'] = pd.to_datetime(df['date'])
        return df

    def _load_mail(self):
        df = pd.read_csv(CONFIG["mail_file"], low_memory=False)
        df.columns = [str(c).lower().strip().replace(' ', '_') for c in df.columns]
        date_col = next(c for c in df.columns if 'date' in c)
        df[date_col] = pd.to_datetime(df[date_col])
        pivot = df.pivot_table(index=date_col, columns='mail_type', values='mail_volume', aggfunc='sum').fillna(0)
        return pivot

    def _load_econ(self, file_path):
        try:
            df = pd.read_csv(file_path)
            df.rename(columns=lambda c: 'date' if 'date' in c.lower() else c, inplace=True)
            df['date'] = pd.to_datetime(df['date'])
            return df
        except FileNotFoundError:
            logging.warning(f"Economic data file not found, skipping: '{file_path}'")
            return None

# ============================================================================
# FEATURE ENGINEERING & SELECTION CLASS
# ============================================================================
class FeatureSelector:
    """Identifies best features and creates scenarios for modeling."""
    def __init__(self, data):
        self.data = data
        self.mail_cols = [c for c in data.columns if c not in CONFIG["economic_data_files"] and c not in ['date', CONFIG["target_column"]]]
        self.econ_cols = [c for c in data.columns if c not in self.mail_cols and c not in ['date', CONFIG["target_column"]]]

    def create_feature_scenarios(self):
        logging.info("--- Creating Feature Scenarios ---")
        top_mail = self._get_top_correlated_features(self.mail_cols, CONFIG["top_n_mail_features"])
        top_econ = self._get_top_correlated_features(self.econ_cols, CONFIG["top_n_econ_features"])
        
        scenarios = {
            "temporal_only": [],
            "temporal_and_mail": top_mail,
            "temporal_and_econ": top_econ,
            "combined_all": list(set(top_mail + top_econ))
        }
        logging.info(f"Created {len(scenarios)} feature scenarios.")
        return scenarios

    def _get_top_correlated_features(self, feature_list, top_n):
        target = CONFIG["target_column"]
        corrs = []
        for lag in range(CONFIG["max_lag_for_selection"] + 1):
            lagged_corrs = self.data[feature_list].shift(lag).corrwith(self.data[target])
            corrs.append(lagged_corrs)
        
        corr_df = pd.concat(corrs, axis=1)
        abs_max_corr = corr_df.abs().max(axis=1)
        strong_features = abs_max_corr[abs_max_corr >= CONFIG["feature_corr_threshold"]]
        top_features = strong_features.nlargest(top_n).index.tolist()
        logging.info(f"Selected top {len(top_features)} features from {len(feature_list)} candidates.")
        return top_features

    @staticmethod
    def engineer_features_for_model(df, source_features):
        """Creates lag/roll features for a given list of source columns."""
        target = CONFIG["target_column"]
        df_model = df[['date', target]].copy()
        
        # Temporal features are always included
        df_model['weekday'] = df_model['date'].dt.weekday
        df_model['month'] = df_model['date'].dt.month
        df_model['day_of_year'] = df_model['date'].dt.dayofyear
        
        for feature in source_features:
            df_model[f'{feature}_lag1'] = df[feature].shift(1)
            df_model[f'{feature}_lag7'] = df[feature].shift(7)
            df_model[f'{feature}_avg7'] = df[feature].shift(1).rolling(window=7).mean()

        df_model = df_model.dropna()
        X = df_model.drop(columns=['date', target])
        y = df_model[target]
        return X, y

# ============================================================================
# MODEL TRAINING & EVALUATION CLASS
# ============================================================================
class ModelTrainer:
    """Manages the training, evaluation, and comparison of all models."""
    def __init__(self, data, scenarios, output_dir):
        self.data = data
        self.scenarios = scenarios
        self.output_dir = output_dir
        self.results = []
        self.best_score = -np.inf
        self.best_model_info = {}

    def run_experiments(self):
        logging.info("--- Starting Model Experiments ---")
        models_to_test = self._get_models()
        
        for scenario_name, source_features in self.scenarios.items():
            logging.info(f"--- Testing Scenario: {scenario_name.upper()} ---")
            X, y = FeatureSelector.engineer_features_for_model(self.data, source_features)
            
            if X.empty:
                logging.warning("No features generated for this scenario, skipping.")
                continue

            for model_name, model in models_to_test.items():
                self._train_and_evaluate_single_model(model_name, model, scenario_name, X, y)

        self._finalize_run()

    def _get_models(self):
        models = {
            'Ridge': Ridge(random_state=CONFIG["random_state"]),
            'RandomForest': RandomForestRegressor(random_state=CONFIG["random_state"], n_jobs=-1),
        }
        if LGB_AVAILABLE:
            models['LightGBM'] = lgb.LGBMRegressor(random_state=CONFIG["random_state"], n_jobs=-1)
        else:
            logging.warning("LightGBM not found. Skipping. For best results, run 'pip install lightgbm'.")
        return models

    def _train_and_evaluate_single_model(self, model_name, model, scenario_name, X, y):
        logging.info(f"Training {model_name} on {X.shape[1]} features...")
        tscv = TimeSeriesSplit(n_splits=CONFIG["timeseries_cv_splits"])
        scores_r2 = []
        scores_mae = []

        for train_index, val_index in tscv.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            scores_r2.append(r2_score(y_val, preds))
            scores_mae.append(mean_absolute_error(y_val, preds))

        avg_r2 = np.mean(scores_r2)
        avg_mae = np.mean(scores_mae)
        logging.info(f"  -> CV Results for {model_name}: Avg R2 = {avg_r2:.3f}, Avg MAE = {avg_mae:,.0f}")
        
        result = {
            "scenario": scenario_name,
            "model": model_name,
            "num_features": X.shape[1],
            "cv_avg_r2": avg_r2,
            "cv_avg_mae": avg_mae,
        }
        self.results.append(result)

        # Check for new best model
        current_score = avg_r2 if CONFIG["evaluation_metric"] == 'r2' else -avg_mae
        if current_score > self.best_score:
            logging.info(f"  -> *** New Best Model Found! ***")
            self.best_score = current_score
            self.best_model_info = {
                "model_name": model_name,
                "scenario_name": scenario_name,
                "score": current_score,
                "full_model": model.fit(X, y), # Retrain on all data
                "features": X.columns.tolist(),
            }

    def _finalize_run(self):
        logging.info("--- Finalizing Run ---")
        if not self.best_model_info:
            logging.error("No models were successfully trained. Aborting.")
            return

        # Save results summary
        report_path = self.output_dir / "experiment_report.json"
        with open(report_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        logging.info(f"Full experiment report saved to {report_path}")

        # Save the best model
        model_path = self.output_dir / "best_model.joblib"
        joblib.dump(self.best_model_info, model_path)
        logging.info(f"Best model '{self.best_model_info['model_name']}' from scenario '{self.best_model_info['scenario_name']}' saved.")

        # Create summary plot
        results_df = pd.DataFrame(self.results)
        plt.figure(figsize=(14, 8))
        sns.barplot(data=results_df, x="cv_avg_r2", y="scenario", hue="model", palette="viridis")
        plt.title('Model Performance (R2) Across Scenarios', fontsize=16)
        plt.xlabel('Average Cross-Validated R-squared')
        plt.ylabel('Feature Scenario')
        plt.legend(title='Model')
        plt.tight_layout()
        plt.savefig(self.output_dir / "summary_performance_plot.png")
        plt.close()

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================
def main():
    """Main function to run the entire Model Foundry pipeline."""
    output_dir = setup_environment()
    
    data_loader = DataLoader()
    full_data = data_loader.load_and_merge()
    
    if full_data is not None:
        feature_selector = FeatureSelector(full_data)
        scenarios = feature_selector.create_feature_scenarios()
        
        trainer = ModelTrainer(full_data, scenarios, output_dir)
        trainer.run_experiments()
        logging.info("--- Model Foundry Run Complete ---")
    else:
        logging.error("Pipeline terminated due to data loading failure.")

if __name__ == "__main__":
    main()
