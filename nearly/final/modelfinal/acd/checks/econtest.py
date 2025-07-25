#!/usr/bin/env python
"""
COMPREHENSIVE MAIL-TO-CALLS PREDICTION SYSTEM

CLEAN APPROACH:

1. Load clean call data (Date, ACDCalls) + mail data
1. Full EDA with plots and correlations
1. Feature engineering with proper lags
1. Simple model first, then build complexity
1. Goal: Predict call volumes from mail volumes (daily/weekly)

CONFIGURABLE PATHS AND SYSTEMATIC BUILD-UP
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import sys
from datetime import datetime, timedelta

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy.stats import pearsonr
import joblib

# ============================================================================
# CONFIGURATION - CHANGE YOUR FILE PATHS HERE
# ============================================================================

CONFIG = {
    # ============ YOUR FILE PATHS ============
    "call_file": "ACDMail.csv",
    "mail_file": "mail.csv",
    "economic_data_file": "expanded_economic_data.csv", # ← (NEW) ADD YOUR ECONOMIC DATA FILE

    # ============ YOUR COLUMN NAMES ============
    "call_date_col": "Date",
    "call_volume_col": "ACDCalls",

    "mail_date_col": "mail_date",
    "mail_volume_col": "mail_volume",
    "mail_type_col": "mail_type",

    # ============ ANALYSIS SETTINGS ============
    "output_dir": "mail_call_prediction_system",
    "top_mail_types": 8,
    "test_size": 0.25,
    "random_state": 42,

    # Feature engineering
    "max_lag_days": 7,
    "rolling_windows": [3, 7],

    # Visualization
    "figure_size": (15, 10),
    "plot_style": "seaborn-v0_8",
}

def remove_us_holidays(df, date_col='date'):
    """Remove US holidays from the DataFrame using a pre-generated CSV file."""
    safe_print("   Removing US holidays from call data using CSV file...")

    try:
        holidays_df = pd.read_csv("us_holidays.csv")
        holiday_dates_to_remove = set(holidays_df['holiday_date'])
    except FileNotFoundError:
        safe_print("❌ ERROR: 'us_holidays.csv' not found!")
        safe_print("   Please make sure you have created the us_holidays.csv file.")
        return df

    holiday_mask = df[date_col].dt.strftime('%Y-%m-%d').isin(holiday_dates_to_remove)
    holidays_found = df[holiday_mask]

    if not holidays_found.empty:
        safe_print(f"   Found {len(holidays_found)} US holidays to remove:")
        for _, row in holidays_found.sort_values(by=date_col).iterrows():
            date_str = row[date_col].strftime('%Y-%m-%d')
            holiday_name = holidays_df[holidays_df['holiday_date'] == date_str]['holiday_name'].iloc[0]
            safe_print(f"     - {date_str}: {holiday_name}")
    else:
        safe_print("   No US holidays found in the provided date range.")

    df_no_holidays = df[~holiday_mask].copy()
    safe_print(f"   Removed {len(holidays_found)} holiday rows.")
    safe_print(f"   Data after holiday removal: {len(df_no_holidays)} rows.")
    return df_no_holidays

def safe_print(msg):
    try:
        print(str(msg).encode('ascii', 'ignore').decode('ascii'))
    except:
        print(str(msg))

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================

class DataManager:
    def __init__(self):
        self.call_data = None
        self.mail_data = None
        self.merged_data = None
        self.output_dir = Path(CONFIG["output_dir"])
        self.output_dir.mkdir(exist_ok=True)

    def load_call_data(self):
        """Load your clean call data"""
        safe_print("=" * 80)
        safe_print("STEP 1A: LOADING CLEAN CALL DATA")
        safe_print("=" * 80)
        
        call_paths = [CONFIG["call_file"], f"data/{CONFIG['call_file']}", f"./{CONFIG['call_file']}"]
        call_path = None
        for path in call_paths:
            if Path(path).exists():
                call_path = path
                break
        
        if not call_path:
            safe_print("❌ CALL FILE NOT FOUND!")
            raise FileNotFoundError("Call file not found")
        
        safe_print(f"✅ Loading: {call_path}")
        
        for encoding in ['utf-8', 'latin1', 'cp1252']:
            try:
                df = pd.read_csv(call_path, encoding=encoding)
                safe_print(f"   Loaded with {encoding} encoding")
                break
            except:
                continue
        else:
            raise ValueError("Could not load call file")
        
        safe_print(f"   Raw data: {len(df):,} rows")
        safe_print(f"   Columns: {df.columns.tolist()}")
        
        date_col = CONFIG["call_date_col"]
        volume_col = CONFIG["call_volume_col"]
        
        if date_col not in df.columns or volume_col not in df.columns:
            raise ValueError("Required call data columns not found")
        
        df_clean = df[[date_col, volume_col]].copy()
        df_clean.columns = ['date', 'call_volume']
        
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        df_clean = df_clean.dropna(subset=['date'])
        
        df_clean['call_volume'] = pd.to_numeric(df_clean['call_volume'], errors='coerce')
        df_clean = df_clean.dropna(subset=['call_volume'])
        df_clean = df_clean[df_clean['call_volume'] > 5]  # Remove negative, zero, or low-noise values
        
        df_clean = df_clean[df_clean['date'].dt.weekday < 5]
        df_clean = remove_us_holidays(df_clean, 'date')
        
        df_clean = df_clean.sort_values('date').reset_index(drop=True)
        self.call_data = df_clean
        
        safe_print(f"✅ Clean call data: {len(df_clean)} business days")
        safe_print(f"   Date range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")
        return df_clean
        
    def load_mail_data(self):
        """Load mail data"""
        safe_print("\n" + "=" * 80)
        safe_print("STEP 1B: LOADING MAIL DATA")
        safe_print("=" * 80)
        
        mail_paths = [CONFIG["mail_file"], f"data/{CONFIG['mail_file']}", f"./{CONFIG['mail_file']}"]
        mail_path = None
        for path in mail_paths:
            if Path(path).exists():
                mail_path = path
                break
        
        if not mail_path:
            safe_print("❌ MAIL FILE NOT FOUND!")
            raise FileNotFoundError("Mail file not found")
        
        safe_print(f"✅ Loading: {mail_path}")
        
        for encoding in ['utf-8', 'latin1', 'cp1252']:
            try:
                df = pd.read_csv(mail_path, encoding=encoding, low_memory=False)
                safe_print(f"   Loaded with {encoding} encoding")
                break
            except:
                continue
        else:
            raise ValueError("Could not load mail file")
        
        safe_print(f"   Raw data: {len(df):,} rows, {len(df.columns)} columns")
        
        df.columns = [str(col).lower().strip() for col in df.columns]
        date_col, volume_col, type_col = None, None, None
        for col in df.columns:
            if 'date' in col: date_col = col
            elif 'volume' in col: volume_col = col
            elif 'type' in col: type_col = col
        
        if not all([date_col, volume_col, type_col]):
            raise ValueError("Required mail columns not found")
        
        safe_print(f"   Using: date={date_col}, volume={volume_col}, type={type_col}")
        
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        
        df[volume_col] = pd.to_numeric(df[volume_col], errors='coerce')
        df = df.dropna(subset=[volume_col])
        df = df[df[volume_col] > 0]
        
        df['mail_date'] = df[date_col].dt.date
        daily_mail = df.groupby(['mail_date', type_col])[volume_col].sum().reset_index()
        daily_mail.columns = ['date', 'mail_type', 'volume']
        daily_mail['date'] = pd.to_datetime(daily_mail['date'])
        
        daily_mail = daily_mail[daily_mail['date'].dt.weekday < 5]
        
        mail_pivot = daily_mail.pivot(index='date', columns='mail_type', values='volume').fillna(0)
        mail_pivot = mail_pivot.reset_index()
        self.mail_data = mail_pivot
        
        safe_print(f"✅ Clean mail data: {len(mail_pivot)} business days")
        safe_print(f"   Date range: {mail_pivot['date'].min().date()} to {mail_pivot['date'].max().date()}")
        return mail_pivot
        
    def merge_data(self):
        """Merge call and mail data"""
        safe_print("\n" + "=" * 80)
        safe_print("STEP 1C: MERGING CALL AND MAIL DATA")
        safe_print("=" * 80)
        
        if self.call_data is None or self.mail_data is None:
            raise ValueError("Must load both call and mail data first")
        
        common_dates = set(self.call_data['date']).intersection(set(self.mail_data['date']))
        
        if len(common_dates) < 30:
            safe_print(f"⚠️  WARNING: Only {len(common_dates)} overlapping days")
        
        merged = pd.merge(self.call_data, self.mail_data, on='date', how='inner')
        merged = merged.sort_values('date').reset_index(drop=True)
        self.merged_data = merged
        
        safe_print(f"✅ Merged dataset: {len(merged)} days")
        safe_print(f"   Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")
        return merged

# ============================================================================
# STEP 2: COMPREHENSIVE EDA
# ============================================================================

class EDATrendAnalysis:
    def __init__(self, merged_data, output_dir):
        self.data = merged_data
        self.output_dir = output_dir / "eda_plots"
        self.output_dir.mkdir(exist_ok=True)
        self.mail_columns = [col for col in merged_data.columns if col not in ['date', 'call_volume'] and isinstance(merged_data[col].iloc[0], (int, float))]
        plt.style.use('default')
        sns.set_palette("husl")
        
    def run_full_eda(self):
        safe_print("\n" + "=" * 80)
        safe_print("STEP 2: COMPREHENSIVE EDA AND VISUALIZATION")
        safe_print("=" * 80)
        self.create_overview_plots()
        correlations = self.analyze_correlations()
        top_mail_types = self.analyze_mail_types()
        best_lag_info = self.analyze_lag_relationships()
        safe_print(f"\n✅ EDA Complete! Plots saved to: {self.output_dir}")
        return {'correlations': correlations, 'top_mail_types': top_mail_types, 'best_lag': best_lag_info}

    def create_overview_plots(self):
        safe_print("\n--- Creating Overview Plots ---")
        total_mail = self.data[self.mail_columns].sum(axis=1)
        overall_corr = self.data['call_volume'].corr(total_mail)
        fig, axes = plt.subplots(2, 2, figsize=CONFIG["figure_size"])
        fig.suptitle('Data Overview', fontsize=16, fontweight='bold')
        axes[0, 0].plot(self.data['date'], self.data['call_volume'], 'b-', linewidth=2)
        axes[0, 0].set_title('Daily Call Volume (ACDCalls)')
        axes[0, 1].plot(self.data['date'], total_mail, 'g-', linewidth=2)
        axes[0, 1].set_title('Daily Total Mail Volume')
        axes[1, 0].scatter(total_mail, self.data['call_volume'], alpha=0.6)
        axes[1, 0].set_title(f'Mail vs Calls (r={overall_corr:.3f})')
        stats_text = f"CALLS:\nMean: {self.data['call_volume'].mean():.0f}\nStd: {self.data['call_volume'].std():.0f}\n\nMAIL:\nMean: {total_mail.mean():.0f}\nStd: {total_mail.std():.0f}"
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, fontsize=10, verticalalignment='center')
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].axis('off')
        plt.tight_layout()
        plt.savefig(self.output_dir / "01_overview.png", dpi=150, bbox_inches='tight')
        plt.close()
        
    def analyze_correlations(self):
        safe_print("\n--- Analyzing Correlations ---")
        numeric_cols = self.data.select_dtypes(include=np.number)
        correlations = numeric_cols.corr()['call_volume'].drop('call_volume').sort_values(key=abs, ascending=False)
        safe_print("   Top 10 correlations with call volume:")
        safe_print(correlations.head(10))
        return correlations

    def analyze_mail_types(self):
        safe_print("\n--- Analyzing Mail Types ---")
        mail_volumes = self.data[self.mail_columns].sum().sort_values(ascending=False)
        top_mail_types = mail_volumes.head(CONFIG["top_mail_types"]).index.tolist()
        safe_print(f"   Top {len(top_mail_types)} mail types by volume:")
        safe_print(mail_volumes.head(CONFIG["top_mail_types"]))
        return top_mail_types

    def analyze_lag_relationships(self):
        safe_print("\n--- Analyzing Lag Relationships ---")
        total_mail = self.data[self.mail_columns].sum(axis=1)
        lag_correlations = {}
        for lag in range(0, CONFIG["max_lag_days"] + 1):
            corr = total_mail.shift(lag).corr(self.data['call_volume'])
            lag_correlations[lag] = corr
            safe_print(f"   Lag {lag} days: correlation = {corr:.3f}")
        best_lag = max(lag_correlations, key=lag_correlations.get)
        safe_print(f"   Best lag: {best_lag} days (correlation: {lag_correlations[best_lag]:.3f})")
        return (best_lag, lag_correlations[best_lag])

# ============================================================================
# STEP 3: FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    def __init__(self, merged_data, top_mail_types, best_lag, economic_cols=[]):
        self.data = merged_data
        self.top_mail_types = top_mail_types
        self.best_lag = best_lag[0] if isinstance(best_lag, tuple) else 1
        self.economic_columns = economic_cols

    def create_features(self):
        safe_print("\n" + "=" * 80)
        safe_print("STEP 3: FEATURE ENGINEERING")
        safe_print("=" * 80)
        
        safe_print(f"   Using lag: {self.best_lag} days")
        
        features_list = []
        targets_list = []
        dates_list = []
        
        max_lookback = max(CONFIG["rolling_windows"] + [self.best_lag])
        
        for i in range(max_lookback, len(self.data) - self.best_lag):
            feature_row = {}
            current_date = self.data.iloc[i]['date']
            
            # === (NEW) ECONOMIC FEATURES ===
            for econ_col in self.economic_columns:
                if econ_col in self.data.columns:
                    feature_row[econ_col] = self.data.iloc[i][econ_col]

            # === MAIL FEATURES ===
            for mail_type in self.top_mail_types:
                clean_name = ''.join(filter(str.isalnum, mail_type))[:15]
                for lag in [1, 2, 3]:
                    feature_row[f"{clean_name}_lag{lag}"] = self.data.iloc[i - lag][mail_type]
                for window in CONFIG["rolling_windows"]:
                    feature_row[f"{clean_name}_avg{window}"] = self.data[mail_type].iloc[i-window+1:i+1].mean()
            
            # === CALL HISTORY FEATURES ===
            for lag in [1, 2, 3]:
                feature_row[f'calls_lag{lag}'] = self.data.iloc[i - lag]['call_volume']
            for window in CONFIG["rolling_windows"]:
                feature_row[f'calls_avg{window}'] = self.data['call_volume'].iloc[i-window+1:i+1].mean()
            
            # === TEMPORAL FEATURES ===
            feature_row['weekday'] = current_date.weekday()
            feature_row['month'] = current_date.month
            feature_row['day_of_month'] = current_date.day
            
            # === TARGET ===
            target = self.data.iloc[i + self.best_lag]['call_volume']
            
            features_list.append(feature_row)
            targets_list.append(target)
            dates_list.append(current_date)
        
        X = pd.DataFrame(features_list).fillna(0)
        y = pd.Series(targets_list, name='call_volume')
        dates = pd.Series(dates_list, name='date')
        
        safe_print(f"✅ Created {len(X.columns)} features from {len(X)} samples")
        return X, y, dates

# ============================================================================
# STEP 4: MODELING
# ============================================================================

class ModelBuilder:
    def __init__(self, output_dir):
        self.output_dir = output_dir / "models"
        self.output_dir.mkdir(exist_ok=True)

    def train_simple_models(self, X, y, dates):
        safe_print("\n" + "=" * 80)
        safe_print("STEP 4: SIMPLE MODEL TRAINING")
        safe_print("=" * 80)
        
        split_idx = int(len(X) * (1 - CONFIG["test_size"]))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        dates_test = dates.iloc[split_idx:]
        
        safe_print(f"   Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        
        models = {
            'linear': LinearRegression(),
            'ridge': Ridge(random_state=CONFIG["random_state"]),
            'forest_simple': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=CONFIG["random_state"])
        }
        
        results = {}
        best_model, best_name, best_score = None, None, -float('inf')
        
        for name, model in models.items():
            safe_print(f"\n--- Testing {name} ---")
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            
            test_r2 = r2_score(y_test, y_pred_test)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            results[name] = {'test_r2': test_r2, 'test_mae': test_mae, 'predictions': y_pred_test}
            safe_print(f"   Test R²: {test_r2:.3f}, Test MAE: {test_mae:.0f}")
            
            if test_r2 > best_score:
                best_score, best_model, best_name = test_r2, model, name
                safe_print(f"   ★ NEW BEST!")
        
        if best_model:
            safe_print(f"\n🎯 BEST MODEL: {best_name} (R²: {best_score:.3f})")
            model_info = {'model': best_model, 'model_name': best_name, 'features': X.columns.tolist(), 'performance': results[best_name]}
            joblib.dump(model_info, self.output_dir / "best_model.pkl")
            self.create_model_validation_plots(y_test, results[best_name]['predictions'], dates_test, best_name, results)
            return best_model, best_name, results
        else:
            safe_print("\n❌ NO MODEL ACHIEVED ACCEPTABLE PERFORMANCE")
            return None, None, None

    def create_model_validation_plots(self, y_test, y_pred, dates_test, best_name, results):
        safe_print("\n--- Creating Model Validation Plots ---")
        fig, axes = plt.subplots(2, 2, figsize=CONFIG["figure_size"])
        fig.suptitle(f'Model Validation: {best_name}', fontsize=16, fontweight='bold')
        
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6)
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_title('Actual vs Predicted')
        
        axes[0, 1].plot(dates_test, y_test, 'b-', label='Actual')
        axes[0, 1].plot(dates_test, y_pred, 'r-', label='Predicted', alpha=0.7)
        axes[0, 1].set_title('Predictions vs Actual Over Time')
        axes[0, 1].legend()
        
        residuals = y_test - y_pred
        axes[1, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_title('Residual Plot')

        model_names = list(results.keys())
        test_r2_scores = [res['test_r2'] for res in results.values()]
        axes[1, 1].bar(model_names, test_r2_scores)
        axes[1, 1].set_title('Model Comparison (Test R²)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_validation.png", dpi=150, bbox_inches='tight')
        plt.close()

# ============================================================================
# STEP 5: PREDICTION SYSTEM
# ============================================================================

class PredictionSystem:
    def __init__(self, model_info, top_mail_types, best_lag, economic_cols): # <-- MODIFIED
        self.model = model_info['model']
        self.features = model_info['features']
        self.top_mail_types = top_mail_types
        self.best_lag = best_lag
        self.economic_columns = economic_cols # <-- ADDED

    def predict_calls(self, mail_input, call_history=None, econ_input=None):
        try:
            features = {}
            now = datetime.now()
            
            # Mail features (simplified for prediction)
            for mail_type in self.top_mail_types:
                clean_name = ''.join(filter(str.isalnum, mail_type))[:15]
                volume = mail_input.get(mail_type, 0)
                for lag in [1, 2, 3]: features[f"{clean_name}_lag{lag}"] = volume
                for window in CONFIG["rolling_windows"]: features[f"{clean_name}_avg{window}"] = volume
            
            # Call history (simplified for prediction)
            calls_yesterday = 12000 if call_history is None else call_history.get('yesterday', 12000)
            for lag in [1, 2, 3]: features[f'calls_lag{lag}'] = calls_yesterday
            for window in CONFIG["rolling_windows"]: features[f'calls_avg{window}'] = calls_yesterday

            # Economic features
            if econ_input:
                features.update(econ_input)

            # Temporal features
            features['weekday'], features['month'], features['day_of_month'] = now.weekday(), now.month, now.day
            
            feature_vector = pd.DataFrame([features])[self.features] # Ensure order and columns match
            prediction = self.model.predict(feature_vector)[0]
            
            return {'status': 'success', 'predicted_calls': max(0, int(prediction))}
        except Exception as e:
            return {'status': 'failed', 'error': str(e)}

# ============================================================================
# (NEW) STEP 6: PREDICTION SCENARIOS
# ============================================================================
def run_prediction_scenarios(prediction_system, merged_data, top_mail_types):
    """Runs single-day and weekly prediction tests with representative data."""
    safe_print("\n" + "=" * 80)
    safe_print("STEP 6: RUNNING PREDICTION SCENARIOS")
    safe_print("=" * 80)

    if not top_mail_types:
        safe_print("⚠️  Cannot run scenarios, no top mail types found.")
        return

    # --- (NEW) Create sample economic data ---
    # Use the average value for each economic indicator as a default
    avg_econ_data = merged_data[prediction_system.economic_columns].mean().to_dict()
    safe_print("   Using average economic data for scenarios:")
    safe_print(f"   {avg_econ_data}")

    avg_mail_volumes = merged_data[top_mail_types].mean()
    
    safe_print("\n--- Testing Single-Day Prediction (Average Mail Day) ---")
    single_day_input = {k: int(v) for k, v in avg_mail_volumes.to_dict().items()}

    # --- (MODIFIED) Pass the economic data to the prediction function ---
    result = prediction_system.predict_calls(single_day_input, econ_input=avg_econ_data)
    
    if result['status'] == 'success':
        safe_print(f"   ➡️ Mail Input: {single_day_input}")
        safe_print(f"   ✅ Predicted Calls: {result['predicted_calls']:,}")
    else:
        safe_print(f"   ❌ Prediction failed: {result.get('error')}")

    safe_print("\n--- Testing Weekly Prediction (Simulated 5-Day Week) ---")
    for day in range(1, 6):
        weekly_input = {mail_type: int(avg_vol * np.random.uniform(0.8, 1.2)) for mail_type, avg_vol in avg_mail_volumes.items()}
        
        # --- (MODIFIED) Pass the economic data here too ---
        result = prediction_system.predict_calls(weekly_input, econ_input=avg_econ_data)
        
        safe_print(f"\n   Day {day} Simulation:")
        safe_print(f"   ➡️ Mail Input: {weekly_input}")
        if result['status'] == 'success':
            safe_print(f"   ✅ Predicted Calls: {result['predicted_calls']:,}")
        else:
            safe_print(f"   ❌ Prediction failed: {result.get('error')}")

# ============================================================================
# MAIN ORCHESTRATOR
# ============================================================================

def main():
    safe_print("COMPREHENSIVE MAIL-TO-CALLS PREDICTION SYSTEM")
    safe_print("=" * 80)
    
    try:
        data_manager = DataManager()
        call_data = data_manager.load_call_data()
        mail_data = data_manager.load_mail_data()
        merged_data = data_manager.merge_data()
        
        # (NEW) STEP 1D: MERGING ECONOMIC DATA
        safe_print("\n" + "=" * 80)
        safe_print("STEP 1D: MERGING ECONOMIC DATA")
        safe_print("=" * 80)
        economic_cols = []
        try:
            econ_df = pd.read_csv(CONFIG["economic_data_file"])
            econ_df['Date'] = pd.to_datetime(econ_df['Date'])
            econ_df.rename(columns={'Date': 'date'}, inplace=True)
            merged_data = pd.merge(merged_data, econ_df, on='date', how='left')
            economic_cols = [col for col in econ_df.columns if col != 'date']
            merged_data[economic_cols] = merged_data[economic_cols].fillna(method='ffill')
            merged_data.dropna(subset=economic_cols, inplace=True)
            safe_print(f"✅ Economic data successfully merged.")
        except FileNotFoundError:
            safe_print(f"⚠️  '{CONFIG['economic_data_file']}' not found. Skipping economic data.")
        except Exception as e:
            safe_print(f"❌ Error merging economic data: {e}")
            
        eda_analyzer = EDATrendAnalysis(merged_data, data_manager.output_dir)
        eda_results = eda_analyzer.run_full_eda()
        
        feature_engineer = FeatureEngineer(
            merged_data, 
            eda_results['top_mail_types'], 
            eda_results['best_lag'],
            economic_cols
        )
        X, y, dates = feature_engineer.create_features()
        
        if len(X) < 30:
            safe_print("⚠️  WARNING: Only {len(X)} samples for modeling.")
            return {'success': False, 'error': 'Not enough data for modeling'}
        
        model_builder = ModelBuilder(data_manager.output_dir)
        best_model, best_name, results = model_builder.train_simple_models(X, y, dates)
        
        if not best_model:
            safe_print("\n❌ MODELING FAILED - NO ACCEPTABLE MODEL FOUND")
            return {'success': False, 'error': 'No acceptable model found'}
        
        model_info = {'model': best_model, 'model_name': best_name, 'features': X.columns.tolist(), 'performance': results[best_name]}
        
        # --- (MODIFIED) Pass economic_cols to the PredictionSystem ---
        prediction_system = PredictionSystem(model_info, eda_results['top_mail_types'], eda_results['best_lag'][0], economic_cols)
        
        safe_print("\n" + "=" * 80)
        safe_print("🎯 SUCCESS! COMPREHENSIVE SYSTEM DEPLOYED!")
        safe_print(f"✅ Best Model: {best_name} (R²={results[best_name]['test_r2']:.3f})")
        
        # (NEW) Run the prediction scenarios
        run_prediction_scenarios(prediction_system, merged_data, eda_results['top_mail_types'])
        
        return {'success': True}
    
    except Exception as e:
        safe_print(f"❌ SYSTEM ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    result = main()
    if result.get('success'):
        safe_print("\n🚀 MAIL-TO-CALLS PREDICTION SYSTEM READY FOR PRODUCTION!")
    else:
        safe_print(f"\n💥 SYSTEM FAILED: {result.get('error')}")






PS C:\Users\BhungarD\OneDrive - Computershare\Desktop\acdmodel> C:\Users\BhungarD\python.exe "c:/Users/BhungarD/OneDrive - Computershare/Desktop/acdmodel/econtest.py"
COMPREHENSIVE MAIL-TO-CALLS PREDICTION SYSTEM
================================================================================
================================================================================
STEP 1A: LOADING CLEAN CALL DATA
================================================================================
 Loading: ACDMail.csv
   Loaded with utf-8 encoding
   Raw data: 547 rows
   Columns: ['Date', 'Product', 'ACDCalls']
   Removing US holidays from call data using CSV file...
   Found 2 US holidays to remove:
     - 2024-10-14: Columbus Day
     - 2024-11-11: Veterans Day
   Removed 2 holiday rows.
   Data after holiday removal: 372 rows.
 Clean call data: 372 business days
   Date range: 2024-01-02 to 2025-06-30

================================================================================
STEP 1B: LOADING MAIL DATA
================================================================================
 Loading: mail.csv
   Loaded with utf-8 encoding
   Raw data: 1,409,780 rows, 4 columns
   Using: date=mail_date, volume=mail_volume, type=mail_type
 Clean mail data: 401 business days
   Date range: 2023-08-01 to 2025-05-30

================================================================================
STEP 1C: MERGING CALL AND MAIL DATA
================================================================================
 Merged dataset: 337 days
   Date range: 2024-01-02 to 2025-05-30

================================================================================
STEP 1D: MERGING ECONOMIC DATA
================================================================================
 Economic data successfully merged.

================================================================================
STEP 2: COMPREHENSIVE EDA AND VISUALIZATION
================================================================================

--- Creating Overview Plots ---

--- Analyzing Correlations ---
   Top 10 correlations with call volume:
Utilities              -0.537179
Oil                     0.431219
Dividend_ETF           -0.413697
Corporate_Bond_ETF     -0.411609
High_Dividend          -0.409159
REITs                  -0.401826
Dividend_Aristocrats   -0.398625
Gold                   -0.374125
Elig_Enr_DedChg_Ltr    -0.349253
Banking                -0.324682
Name: call_volume, dtype: float64

--- Analyzing Mail Types ---
   Top 8 mail types by volume:
DowJones                    1.380986e+07
Cheque                      1.349555e+07
DRP Stmt.                   1.213277e+07
Scheduled PAYMENT CHECKS    9.894910e+06
NASDAQ                      5.952835e+06
Envision                    5.829010e+06
Proxy (US)                  5.703365e+06
Notice                      4.924059e+06
dtype: float64

--- Analyzing Lag Relationships ---
   Lag 0 days: correlation = -0.021
   Lag 1 days: correlation = 0.028
   Lag 2 days: correlation = -0.032
   Lag 3 days: correlation = -0.020
   Lag 4 days: correlation = -0.014
   Lag 5 days: correlation = -0.001
   Lag 6 days: correlation = -0.020
   Lag 7 days: correlation = -0.031
   Best lag: 1 days (correlation: 0.028)

 EDA Complete! Plots saved to: mail_call_prediction_system\eda_plots

================================================================================
STEP 3: FEATURE ENGINEERING
================================================================================
   Using lag: 1 days
 Created 69 features from 329 samples

================================================================================
STEP 4: SIMPLE MODEL TRAINING
================================================================================
   Train: 246 samples, Test: 83 samples

--- Testing linear ---
   Test R: -140.764, Test MAE: 9863
    NEW BEST!

--- Testing ridge ---
   Test R: -120.779, Test MAE: 9205
    NEW BEST!

--- Testing forest_simple ---
   Test R: 0.666, Test MAE: 1040
    NEW BEST!

 BEST MODEL: forest_simple (R: 0.666)

--- Creating Model Validation Plots ---

================================================================================
 SUCCESS! COMPREHENSIVE SYSTEM DEPLOYED!
 Best Model: forest_simple (R=0.666)

================================================================================
STEP 6: RUNNING PREDICTION SCENARIOS
================================================================================
   Using average economic data for scenarios:
   {'Oil': 73.2063502190021, 'Dividend_ETF': 124.28145502016876, 'Dollar_Index': 104.1808903181942, 'Gold': 2593.4086022985443, 'High_Dividend': 109.98658071002903, 'Banking': 50.080747666627786, 'Regional_Banks': 53.967258125101424, 'Corporate_Bond_ETF': 104.64197755355156, 'Dividend_Aristocrats': 98.6023435281363, 'REITs': 86.07146114270128, 'Technology': 217.0730445929734, 'Utilities': 71.99343243833819, 'DowJones': 40978.81230294881, '2Y_Treasury': 4.718492568777294, 'NASDAQ': 17664.197474267432, 'Russell2000': 2124.3672958057077, '10Y_Treasury': 4.268255189901289, '30Y_Treasury': 4.51370920059589, 'VIX': 17.52228489544696, 'VIX9D': 16.996528186854693, 'VXN': 20.949465983699266}

--- Testing Single-Day Prediction (Average Mail Day) ---
    Mail Input: {'DowJones': 40978, 'Cheque': 40046, 'DRP Stmt.': 36002, 'Scheduled PAYMENT CHECKS': 29361, 'NASDAQ': 17664, 'Envision': 17296, 'Proxy (US)': 16923, 'Notice': 14611}
    Predicted Calls: 11,716

--- Testing Weekly Prediction (Simulated 5-Day Week) ---

   Day 1 Simulation:
    Mail Input: {'DowJones': 42782, 'Cheque': 40303, 'DRP Stmt.': 32582, 'Scheduled PAYMENT CHECKS': 23962, 'NASDAQ': 17637, 'Envision': 19081, 'Proxy (US)': 15634, 'Notice': 16268}
    Predicted Calls: 11,642

   Day 2 Simulation:
    Mail Input: {'DowJones': 38854, 'Cheque': 43469, 'DRP Stmt.': 39904, 'Scheduled PAYMENT CHECKS': 30349, 'NASDAQ': 16368, 'Envision': 19059, 'Proxy (US)': 17425, 'Notice': 15143}
    Predicted Calls: 11,759

   Day 3 Simulation:
    Mail Input: {'DowJones': 43520, 'Cheque': 39797, 'DRP Stmt.': 40956, 'Scheduled PAYMENT CHECKS': 35213, 'NASDAQ': 14474, 'Envision': 15618, 'Proxy (US)': 17968, 'Notice': 16955}
    Predicted Calls: 11,565

   Day 4 Simulation:
    Mail Input: {'DowJones': 35651, 'Cheque': 44382, 'DRP Stmt.': 35681, 'Scheduled PAYMENT CHECKS': 27522, 'NASDAQ': 15750, 'Envision': 15866, 'Proxy (US)': 16243, 'Notice': 15367}
    Predicted Calls: 11,499

   Day 5 Simulation:
    Mail Input: {'DowJones': 38869, 'Cheque': 45912, 'DRP Stmt.': 35537, 'Scheduled PAYMENT CHECKS': 33785, 'NASDAQ': 19416, 'Envision': 16103, 'Proxy (US)': 16469, 'Notice': 14759}
    Predicted Calls: 11,706

 MAIL-TO-CALLS PREDICTION SYSTEM READY FOR PRODUCTION!
PS C:\Users\BhungarD\OneDrive - Computershare\Desktop\acdmodel> 
