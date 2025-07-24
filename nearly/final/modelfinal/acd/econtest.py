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
