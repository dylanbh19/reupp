# #!/usr/bin/env python
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
import holidays


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
"call_file": "ACDMail.csv",  # ← CHANGE THIS
"mail_file": "mail.csv",                    # ← CHANGE IF NEEDED


# ============ YOUR COLUMN NAMES ============
"call_date_col": "Date",        # Your call data date column
"call_volume_col": "ACDCalls",  # Your call data volume column

# Mail columns (keep as is unless different)
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
        # Load the list of holidays from your CSV
        holidays_df = pd.read_csv("us_holidays.csv")
        # Create a set of holiday date strings for fast lookup
        holiday_dates_to_remove = set(holidays_df['holiday_date'])
    except FileNotFoundError:
        safe_print("❌ ERROR: 'us_holidays.csv' not found!")
        safe_print("   Please make sure you have created the us_holidays.csv file.")
        # Return the original dataframe if the holiday file is missing
        return df

    # Create a boolean mask by converting the DataFrame's date column to 'YYYY-MM-DD' strings
    # and checking if they exist in our set of holidays.
    holiday_mask = df[date_col].dt.strftime('%Y-%m-%d').isin(holiday_dates_to_remove)

    holidays_found = df[holiday_mask]

    if not holidays_found.empty:
        safe_print(f"   Found {len(holidays_found)} US holidays to remove:")
        for _, row in holidays_found.sort_values(by=date_col).iterrows():
            # Get the date as a string to look up the name
            date_str = row[date_col].strftime('%Y-%m-%d')
            holiday_name = holidays_df[holidays_df['holiday_date'] == date_str]['holiday_name'].iloc[0]
            safe_print(f"     - {date_str}: {holiday_name}")
    else:
        safe_print("   No US holidays found in the provided date range.")

    # Invert the mask to keep non-holidays and create a copy.
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
        
        # Try multiple paths
        call_paths = [
            CONFIG["call_file"],
            f"data/{CONFIG['call_file']}",
            f"./{CONFIG['call_file']}"
        ]
        
        call_path = None
        for path in call_paths:
            if Path(path).exists():
                call_path = path
                break
        
        if not call_path:
            safe_print("❌ CALL FILE NOT FOUND!")
            safe_print("Please update CONFIG['call_file'] with the correct path")
            safe_print("Tried paths:")
            for path in call_paths:
                safe_print(f"  - {path}")
            raise FileNotFoundError("Call file not found")
        
        safe_print(f"✅ Loading: {call_path}")
        
        # Load with encoding attempts
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
        
        # Check for required columns
        date_col = CONFIG["call_date_col"]
        volume_col = CONFIG["call_volume_col"]
        
        if date_col not in df.columns:
            safe_print(f"❌ Date column '{date_col}' not found!")
            safe_print(f"   Available columns: {df.columns.tolist()}")
            raise ValueError(f"Date column '{date_col}' not found")
        
        if volume_col not in df.columns:
            safe_print(f"❌ Volume column '{volume_col}' not found!")
            safe_print(f"   Available columns: {df.columns.tolist()}")
            raise ValueError(f"Volume column '{volume_col}' not found")
        
        # Clean and process
        df_clean = df[[date_col, volume_col]].copy()
        df_clean.columns = ['date', 'call_volume']
        
        # Process dates
        df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
        df_clean = df_clean.dropna(subset=['date'])
        
        # Process call volumes
        df_clean['call_volume'] = pd.to_numeric(df_clean['call_volume'], errors='coerce')
        df_clean = df_clean.dropna(subset=['call_volume'])
        df_clean = df_clean[df_clean['call_volume'] > 5]  # Remove negative values
        
        # Filter to business days only
        df_clean = df_clean[df_clean['date'].dt.weekday < 5]
        df_clean = remove_us_holidays(df_clean, 'date')
        print("REMOVEDDDDHOLIDAYSSSSSSSSSSS")
        # Sort by date
        df_clean = df_clean.sort_values('date').reset_index(drop=True)
        
        self.call_data = df_clean
        
        safe_print(f"✅ Clean call data: {len(df_clean)} business days")
        safe_print(f"   Date range: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")
        safe_print(f"   Call volume: {df_clean['call_volume'].min():.0f} to {df_clean['call_volume'].max():.0f}")
        safe_print(f"   Daily average: {df_clean['call_volume'].mean():.0f} calls")
        
        return df_clean
        
    def load_mail_data(self):
        """Load mail data"""
        safe_print("\n" + "=" * 80)
        safe_print("STEP 1B: LOADING MAIL DATA")
        safe_print("=" * 80)
        
        # Try multiple paths
        mail_paths = [
            CONFIG["mail_file"],
            f"data/{CONFIG['mail_file']}",
            f"./{CONFIG['mail_file']}"
        ]
        
        mail_path = None
        for path in mail_paths:
            if Path(path).exists():
                mail_path = path
                break
        
        if not mail_path:
            safe_print("❌ MAIL FILE NOT FOUND!")
            safe_print("Please update CONFIG['mail_file'] with the correct path")
            raise FileNotFoundError("Mail file not found")
        
        safe_print(f"✅ Loading: {mail_path}")
        
        # Load with encoding attempts
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
        
        # Clean column names and find required columns
        df.columns = [str(col).lower().strip() for col in df.columns]
        
        date_col = volume_col = type_col = None
        for col in df.columns:
            if 'date' in col:
                date_col = col
            elif 'volume' in col:
                volume_col = col
            elif 'type' in col:
                type_col = col
        
        if not all([date_col, volume_col, type_col]):
            safe_print(f"❌ Required mail columns not found!")
            safe_print(f"   Available: {df.columns.tolist()}")
            safe_print(f"   Looking for: date, volume, type columns")
            raise ValueError("Required mail columns not found")
        
        safe_print(f"   Using: date={date_col}, volume={volume_col}, type={type_col}")
        
        # Process mail data
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])
        
        df[volume_col] = pd.to_numeric(df[volume_col], errors='coerce')
        df = df.dropna(subset=[volume_col])
        df = df[df[volume_col] > 0]
        
        # Create daily mail by type
        df['mail_date'] = df[date_col].dt.date
        daily_mail = df.groupby(['mail_date', type_col])[volume_col].sum().reset_index()
        daily_mail.columns = ['date', 'mail_type', 'volume']
        daily_mail['date'] = pd.to_datetime(daily_mail['date'])
        
        # Filter to business days
        daily_mail = daily_mail[daily_mail['date'].dt.weekday < 5]
        
        # Pivot to mail types as columns
        mail_pivot = daily_mail.pivot(index='date', columns='mail_type', values='volume').fillna(0)
        mail_pivot = mail_pivot.reset_index()
        
        self.mail_data = mail_pivot
        
        safe_print(f"✅ Clean mail data: {len(mail_pivot)} business days")
        safe_print(f"   Date range: {mail_pivot['date'].min().date()} to {mail_pivot['date'].max().date()}")
        safe_print(f"   Mail types: {len(mail_pivot.columns)-1}")
        
        return mail_pivot
        
    def merge_data(self):
        """Merge call and mail data"""
        safe_print("\n" + "=" * 80)
        safe_print("STEP 1C: MERGING CALL AND MAIL DATA")
        safe_print("=" * 80)
        
        if self.call_data is None or self.mail_data is None:
            raise ValueError("Must load both call and mail data first")
        
        # Find overlapping dates
        call_dates = set(self.call_data['date'].dt.date)
        mail_dates = set(self.mail_data['date'].dt.date)
        common_dates = call_dates.intersection(mail_dates)
        
        safe_print(f"   Call data: {len(call_dates)} days")
        safe_print(f"   Mail data: {len(mail_dates)} days")
        safe_print(f"   Common dates: {len(common_dates)} days")
        
        if len(common_dates) < 30:
            safe_print(f"⚠️  WARNING: Only {len(common_dates)} overlapping days")
        
        # Filter to common dates and merge
        common_dates_dt = [pd.to_datetime(d) for d in common_dates]
        
        calls_filtered = self.call_data[self.call_data['date'].isin(common_dates_dt)]
        mail_filtered = self.mail_data[self.mail_data['date'].isin(common_dates_dt)]
        
        merged = pd.merge(calls_filtered, mail_filtered, on='date', how='inner')
        merged = merged.sort_values('date').reset_index(drop=True)
        
        self.merged_data = merged
        
        safe_print(f"✅ Merged dataset: {len(merged)} days")
        safe_print(f"   Columns: {len(merged.columns)} (date + calls + {len(merged.columns)-2} mail types)")
        safe_print(f"   Date range: {merged['date'].min().date()} to {merged['date'].max().date()}")
        
        return merged


# ============================================================================

# STEP 2: COMPREHENSIVE EDA WITH PLOTS

# ============================================================================

class EDATrendAnalysis:
    def __init__(self, merged_data, output_dir):
        self.data = merged_data
        self.output_dir = output_dir / "eda_plots"
        self.output_dir.mkdir(exist_ok=True)
        self.mail_columns = [col for col in merged_data.columns if col not in ['date', 'call_volume']]


        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def run_full_eda(self):
        """Run comprehensive EDA"""
        safe_print("\n" + "=" * 80)
        safe_print("STEP 2: COMPREHENSIVE EDA AND VISUALIZATION")
        safe_print("=" * 80)
        
        # Basic overview
        self.create_overview_plots()
        
        # Time series analysis
        self.create_time_series_plots()
        
        # Correlation analysis
        correlations = self.analyze_correlations()
        
        # Mail type analysis
        top_mail_types = self.analyze_mail_types()
        
        # Lag analysis
        best_lag_info = self.analyze_lag_relationships()
        
        safe_print(f"\n✅ EDA Complete! Plots saved to: {self.output_dir}")
        
        return {
            'correlations': correlations,
            'top_mail_types': top_mail_types,
            'best_lag': best_lag_info
        }

    def create_overview_plots(self):
        """Create overview plots"""
        safe_print("\n--- Creating Overview Plots ---")
        
        total_mail = self.data[self.mail_columns].sum(axis=1)
        overall_corr = self.data['call_volume'].corr(total_mail)
        
        fig, axes = plt.subplots(2, 2, figsize=CONFIG["figure_size"])
        fig.suptitle('Data Overview', fontsize=16, fontweight='bold')
        
        # Call volume over time
        axes[0, 0].plot(self.data['date'], self.data['call_volume'], 'b-', linewidth=2)
        axes[0, 0].set_title('Daily Call Volume (ACDCalls)')
        axes[0, 0].set_ylabel('Call Volume')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Total mail over time
        axes[0, 1].plot(self.data['date'], total_mail, 'g-', linewidth=2)
        axes[0, 1].set_title('Daily Total Mail Volume')
        axes[0, 1].set_ylabel('Mail Volume')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Scatter plot: mail vs calls
        axes[1, 0].scatter(total_mail, self.data['call_volume'], alpha=0.6)
        axes[1, 0].set_xlabel('Total Mail Volume')
        axes[1, 0].set_ylabel('Call Volume')
        axes[1, 0].set_title(f'Mail vs Calls (r={overall_corr:.3f})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Basic statistics
        call_stats = self.data['call_volume'].describe()
        mail_stats = total_mail.describe()
        
        stats_text = f"""
        CALL VOLUME STATS:
        Mean: {call_stats['mean']:.0f}
        Std:  {call_stats['std']:.0f}
        Min:  {call_stats['min']:.0f}
        Max:  {call_stats['max']:.0f}
        
        MAIL VOLUME STATS:
        Mean: {mail_stats['mean']:.0f}
        Std:  {mail_stats['std']:.0f}
        Min:  {mail_stats['min']:.0f}
        Max:  {mail_stats['max']:.0f}
        
        CORRELATION: {overall_corr:.3f}
        DAYS: {len(self.data)}
        """
        
        axes[1, 1].text(0.1, 0.5, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='center', fontfamily='monospace')
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "01_overview.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        safe_print(f"   Overall correlation: {overall_corr:.3f}")
        
    def create_time_series_plots(self):
        """Create detailed time series plots"""
        safe_print("\n--- Creating Time Series Plots ---")
        
        total_mail = self.data[self.mail_columns].sum(axis=1)
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        fig.suptitle('Time Series Analysis', fontsize=16, fontweight='bold')
        
        # Call volume time series
        axes[0].plot(self.data['date'], self.data['call_volume'], 'b-', linewidth=2, label='ACDCalls')
        axes[0].set_title('Daily Call Volume Over Time')
        axes[0].set_ylabel('Call Volume')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Total mail time series
        axes[1].plot(self.data['date'], total_mail, 'g-', linewidth=2, label='Total Mail')
        axes[1].set_title('Daily Total Mail Volume Over Time')
        axes[1].set_ylabel('Mail Volume')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        # Normalized overlay
        call_norm = (self.data['call_volume'] - self.data['call_volume'].min()) / (self.data['call_volume'].max() - self.data['call_volume'].min())
        mail_norm = (total_mail - total_mail.min()) / (total_mail.max() - total_mail.min())
        
        axes[2].plot(self.data['date'], call_norm, 'b-', linewidth=2, label='Calls (normalized)', alpha=0.8)
        axes[2].plot(self.data['date'], mail_norm, 'g-', linewidth=2, label='Mail (normalized)', alpha=0.8)
        axes[2].set_title('Normalized Comparison: Calls vs Mail')
        axes[2].set_ylabel('Normalized Values (0-1)')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "02_time_series.png", dpi=150, bbox_inches='tight')
        plt.close()
        
    def analyze_correlations(self):
        """Comprehensive correlation analysis"""
        safe_print("\n--- Analyzing Correlations ---")
        
        # Calculate correlations for all mail types
        correlations = {}
        for mail_type in self.mail_columns:
            if self.data[mail_type].std() > 0:
                corr, p_value = pearsonr(self.data[mail_type], self.data['call_volume'])
                correlations[mail_type] = {'correlation': corr, 'p_value': p_value}
        
        # Sort by absolute correlation
        sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
        
        safe_print("   Top 10 correlations with call volume:")
        for i, (mail_type, stats) in enumerate(sorted_corr[:10]):
            safe_print(f"   {i+1:2d}. {mail_type[:30]:<30}: r={stats['correlation']:>7.3f}")
        
        # Create correlation heatmap
        top_15_types = [item[0] for item in sorted_corr[:15]]
        corr_data = self.data[['call_volume'] + top_15_types].corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_data), k=1)
        sns.heatmap(corr_data, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                mask=mask, square=True, cbar_kws={'label': 'Correlation'})
        plt.title('Correlation Matrix: Calls vs Top 15 Mail Types')
        plt.tight_layout()
        plt.savefig(self.output_dir / "03_correlations.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        return sorted_corr

    def analyze_mail_types(self):
        """Analyze individual mail types"""
        safe_print("\n--- Analyzing Mail Types ---")
        
        # Calculate stats for each mail type
        mail_stats = {}
        for mail_type in self.mail_columns:
            stats_dict = {
                'total_volume': self.data[mail_type].sum(),
                'daily_average': self.data[mail_type].mean(),
                'max_day': self.data[mail_type].max(),
                'correlation': self.data[mail_type].corr(self.data['call_volume'])
            }
            mail_stats[mail_type] = stats_dict
        
        # Sort by total volume
        sorted_by_volume = sorted(mail_stats.items(), key=lambda x: x[1]['total_volume'], reverse=True)
        top_mail_types = [item[0] for item in sorted_by_volume[:CONFIG["top_mail_types"]]]
        
        safe_print(f"   Top {len(top_mail_types)} mail types by volume:")
        for i, (mail_type, stats) in enumerate(sorted_by_volume[:CONFIG["top_mail_types"]]):
            safe_print(f"   {i+1:2d}. {mail_type[:25]:<25}: {stats['total_volume']:>8,.0f} total, r={stats['correlation']:>6.3f}")
        
        # Create mail type analysis plots
        fig, axes = plt.subplots(2, 2, figsize=CONFIG["figure_size"])
        fig.suptitle('Mail Type Analysis', fontsize=16, fontweight='bold')
        
        # Volume ranking
        volumes = [mail_stats[mt]['total_volume'] for mt in top_mail_types]
        axes[0, 0].barh(range(len(top_mail_types)), volumes, alpha=0.7)
        axes[0, 0].set_yticks(range(len(top_mail_types)))
        axes[0, 0].set_yticklabels([mt[:20] for mt in top_mail_types])
        axes[0, 0].set_xlabel('Total Volume')
        axes[0, 0].set_title('Top Mail Types by Volume')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Correlation ranking
        correlations = [mail_stats[mt]['correlation'] for mt in top_mail_types]
        colors = ['red' if c < 0 else 'green' for c in correlations]
        
        axes[0, 1].barh(range(len(top_mail_types)), correlations, alpha=0.7, color=colors)
        axes[0, 1].set_yticks(range(len(top_mail_types)))
        axes[0, 1].set_yticklabels([mt[:20] for mt in top_mail_types])
        axes[0, 1].set_xlabel('Correlation with Calls')
        axes[0, 1].set_title('Correlation with Call Volume')
        axes[0, 1].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Time series of top 3 mail types
        top_3 = top_mail_types[:3]
        for i, mail_type in enumerate(top_3):
            axes[1, 0].plot(self.data['date'], self.data[mail_type], 
                        label=mail_type[:15], linewidth=2)
        
        axes[1, 0].set_title('Top 3 Mail Types Over Time')
        axes[1, 0].set_ylabel('Daily Volume')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Best correlated mail type vs calls
        best_corr_type = max(mail_stats.items(), key=lambda x: abs(x[1]['correlation']))
        best_type_name = best_corr_type[0]
        best_corr_value = best_corr_type[1]['correlation']
        
        axes[1, 1].scatter(self.data[best_type_name], self.data['call_volume'], alpha=0.6)
        axes[1, 1].set_xlabel(f'{best_type_name[:20]} Volume')
        axes[1, 1].set_ylabel('Call Volume')
        axes[1, 1].set_title(f'Best Correlated: {best_type_name[:15]} (r={best_corr_value:.3f})')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "04_mail_types.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        return top_mail_types

    def analyze_lag_relationships(self):
        """Analyze lag relationships"""
        safe_print("\n--- Analyzing Lag Relationships ---")
        
        total_mail = self.data[self.mail_columns].sum(axis=1)
        
        # Calculate correlations for different lags
        lag_correlations = {}
        
        for lag in range(0, CONFIG["max_lag_days"] + 1):
            if len(self.data) > lag:
                if lag == 0:
                    corr = total_mail.corr(self.data['call_volume'])
                else:
                    # Mail today vs calls N days later
                    mail_today = total_mail[:-lag]
                    calls_later = self.data['call_volume'][lag:]
                    
                    if len(mail_today) > 10:
                        corr = mail_today.corr(calls_later)
                    else:
                        corr = 0
                
                lag_correlations[lag] = corr
                safe_print(f"   Lag {lag} days: correlation = {corr:.3f}")
        
        # Find best lag
        best_lag = max(lag_correlations.items(), key=lambda x: abs(x[1]))
        safe_print(f"   Best lag: {best_lag[0]} days (correlation: {best_lag[1]:.3f})")
        
        # Create lag analysis plot
        fig, axes = plt.subplots(1, 2, figsize=CONFIG["figure_size"])
        fig.suptitle('Lag Relationship Analysis', fontsize=16, fontweight='bold')
        
        # Correlation by lag
        lags = list(lag_correlations.keys())
        correlations = list(lag_correlations.values())
        
        bars = axes[0].bar(lags, correlations, alpha=0.7, color='purple')
        axes[0].set_xlabel('Lag (Days)')
        axes[0].set_ylabel('Correlation')
        axes[0].set_title('Correlation by Lag Days')
        axes[0].grid(True, alpha=0.3)
        
        # Highlight best lag
        best_idx = lags.index(best_lag[0])
        bars[best_idx].set_color('red')
        bars[best_idx].set_alpha(1.0)
        
        # Scatter plot for best lag
        if best_lag[0] == 0:
            x_data = total_mail
            y_data = self.data['call_volume']
            title = f'Same Day: Mail vs Calls (r={best_lag[1]:.3f})'
        else:
            x_data = total_mail[:-best_lag[0]]
            y_data = self.data['call_volume'][best_lag[0]:]
            title = f'Mail Today vs Calls +{best_lag[0]} Days (r={best_lag[1]:.3f})'
        
        axes[1].scatter(x_data, y_data, alpha=0.6)
        axes[1].set_xlabel('Mail Volume')
        axes[1].set_ylabel('Call Volume')
        axes[1].set_title(title)
        axes[1].grid(True, alpha=0.3)
        
        # Add trend line
        if len(x_data) > 1:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            axes[1].plot(x_data, p(x_data), "r--", alpha=0.8)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "05_lag_analysis.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        return best_lag


# ============================================================================

# STEP 3: FEATURE ENGINEERING

# ============================================================================

class FeatureEngineer:
    def __init__(self, merged_data, top_mail_types, best_lag):
        self.data = merged_data
        self.top_mail_types = top_mail_types
        self.best_lag = best_lag[0] if best_lag else 1


    def create_features(self):
        """Create features for modeling"""
        safe_print("\n" + "=" * 80)
        safe_print("STEP 3: FEATURE ENGINEERING")
        safe_print("=" * 80)
        
        safe_print(f"   Using lag: {self.best_lag} days")
        safe_print(f"   Top mail types: {len(self.top_mail_types)}")
        
        features_list = []
        targets_list = []
        dates_list = []
        
        # Create features for each day
        max_lookback = max(7, self.best_lag)  # Need history for features
        
        for i in range(max_lookback, len(self.data) - self.best_lag):
            
            feature_row = {}
            current_date = self.data.iloc[i]['date']
            
            # === MAIL FEATURES ===
            for mail_type in self.top_mail_types:
                if mail_type in self.data.columns:
                    clean_name = mail_type.replace(' ', '').replace('-', '').replace('_', '')[:15]
                    
                    # Current day mail
                    feature_row[f"{clean_name}_today"] = self.data.iloc[i][mail_type]
                    
                    # Lag features (1, 2, 3 days ago)
                    for lag in [1, 2, 3]:
                        if i >= lag:
                            feature_row[f"{clean_name}_lag{lag}"] = self.data.iloc[i - lag][mail_type]
                    
                    # Rolling averages
                    for window in CONFIG["rolling_windows"]:
                        if i >= window - 1:
                            window_data = self.data.iloc[i - window + 1:i + 1][mail_type]
                            feature_row[f"{clean_name}_avg{window}"] = window_data.mean()
            
            # === TOTAL MAIL FEATURES ===
            total_mail = sum(self.data.iloc[i][mt] for mt in self.top_mail_types if mt in self.data.columns)
            feature_row['total_mail_today'] = total_mail
            
            # Total mail lags
            for lag in [1, 2, 3]:
                if i >= lag:
                    total_mail_lag = sum(self.data.iloc[i - lag][mt] for mt in self.top_mail_types if mt in self.data.columns)
                    feature_row[f'total_mail_lag{lag}'] = total_mail_lag
            
            # Total mail rolling averages
            for window in CONFIG["rolling_windows"]:
                if i >= window - 1:
                    window_totals = []
                    for j in range(i - window + 1, i + 1):
                        window_total = sum(self.data.iloc[j][mt] for mt in self.top_mail_types if mt in self.data.columns)
                        window_totals.append(window_total)
                    feature_row[f'total_mail_avg{window}'] = np.mean(window_totals)
            
            # === CALL HISTORY FEATURES ===
            feature_row['calls_yesterday'] = self.data.iloc[i - 1]['call_volume'] if i >= 1 else self.data.iloc[i]['call_volume']
            feature_row['calls_2days_ago'] = self.data.iloc[i - 2]['call_volume'] if i >= 2 else self.data.iloc[i]['call_volume']
            
            # Call rolling averages
            for window in CONFIG["rolling_windows"]:
                if i >= window - 1:
                    window_calls = self.data.iloc[i - window + 1:i + 1]['call_volume']
                    feature_row[f'calls_avg{window}'] = window_calls.mean()
            
            # === TEMPORAL FEATURES ===
            feature_row['weekday'] = current_date.weekday()
            feature_row['month'] = current_date.month
            feature_row['day_of_month'] = current_date.day
            feature_row['is_month_end'] = 1 if current_date.day >= 25 else 0
            
            # === TARGET ===
            target_idx = i + self.best_lag
            if target_idx < len(self.data):
                target = self.data.iloc[target_idx]['call_volume']
                
                features_list.append(feature_row)
                targets_list.append(target)
                dates_list.append(current_date)
        
        # Convert to DataFrames
        X = pd.DataFrame(features_list)
        y = pd.Series(targets_list, name='call_volume')
        dates = pd.Series(dates_list, name='date')
        
        # Clean features
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        safe_print(f"✅ Created {len(X.columns)} features from {len(X)} samples")
        safe_print(f"   Mail features: {len([c for c in X.columns if any(mt.replace(' ', '')[:10] in c for mt in self.top_mail_types)])}")
        safe_print(f"   Call history: {len([c for c in X.columns if 'calls' in c])}")
        safe_print(f"   Temporal: {len([c for c in X.columns if any(t in c for t in ['weekday', 'month', 'day'])])}")
        
        return X, y, dates


# ============================================================================

# STEP 4: MODELING - START SIMPLE, BUILD UP

# ============================================================================

class ModelBuilder:
    def __init__(self, output_dir):
        self.output_dir = output_dir / "models"
        self.output_dir.mkdir(exist_ok=True)
        self.models = {}
        self.results = {}


    def train_simple_models(self, X, y, dates):
        """Start with simple models"""
        safe_print("\n" + "=" * 80)
        safe_print("STEP 4: SIMPLE MODEL TRAINING")
        safe_print("=" * 80)
        
        # Time-aware split
        split_idx = int(len(X) * (1 - CONFIG["test_size"]))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        dates_train, dates_test = dates.iloc[:split_idx], dates.iloc[split_idx:]
        
        safe_print(f"   Train: {len(X_train)} samples")
        safe_print(f"   Test: {len(X_test)} samples")
        
        # Simple models to try
        models = {
            'linear': LinearRegression(),
            'ridge_light': Ridge(alpha=1.0, random_state=CONFIG["random_state"]),
            'ridge_strong': Ridge(alpha=10.0, random_state=CONFIG["random_state"]),
            'forest_simple': RandomForestRegressor(
                n_estimators=50, 
                max_depth=6, 
                min_samples_leaf=5,
                random_state=CONFIG["random_state"]
            )
        }
        
        results = {}
        best_model = None
        best_score = -float('inf')
        best_name = None
        
        for name, model in models.items():
            safe_print(f"\n--- Testing {name} ---")
            
            try:
                # Train
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)
                
                # Evaluate
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                
                # MAPE
                mape = np.mean(np.abs((y_test - y_pred_test) / (y_test + 1e-10))) * 100
                mape = min(mape, 200)  # Cap extreme values
                
                overfitting = train_r2 - test_r2
                
                results[name] = {
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'test_mae': test_mae,
                    'test_rmse': test_rmse,
                    'test_mape': mape,
                    'overfitting': overfitting,
                    'model': model,
                    'predictions': y_pred_test
                }
                
                safe_print(f"   Train R²: {train_r2:.3f}")
                safe_print(f"   Test R²:  {test_r2:.3f}")
                safe_print(f"   Test MAE: {test_mae:.0f}")
                safe_print(f"   Test MAPE: {mape:.1f}%")
                safe_print(f"   Overfitting: {overfitting:.3f}")
                
                # Select best (prioritize test R², penalize overfitting)
                adjusted_score = test_r2 - max(0, overfitting - 0.1) * 0.5
                
                if adjusted_score > best_score and test_r2 > 0.05:  # Must have some predictive power
                    best_score = adjusted_score
                    best_model = model
                    best_name = name
                    safe_print(f"   ★ NEW BEST! (Score: {adjusted_score:.3f})")
                
            except Exception as e:
                safe_print(f"   ✗ Failed: {e}")
                results[name] = {'error': str(e)}
        
        self.models = results
        self.results = results
        
        if best_model:
            safe_print(f"\n🎯 BEST MODEL: {best_name}")
            safe_print(f"   Test R²: {results[best_name]['test_r2']:.3f}")
            safe_print(f"   Test MAE: {results[best_name]['test_mae']:.0f}")
            safe_print(f"   Test MAPE: {results[best_name]['test_mape']:.1f}%")
            
            # Train on full dataset
            best_model.fit(X, y)
            
            # Save best model
            model_info = {
                'model': best_model,
                'model_name': best_name,
                'features': X.columns.tolist(),
                'performance': results[best_name]
            }
            
            joblib.dump(model_info, self.output_dir / "best_model.pkl")
            
            # Create validation plots
            self.create_model_validation_plots(X_test, y_test, results[best_name]['predictions'], 
                                            dates_test, best_name, results)
            
            return best_model, best_name, results
        else:
            safe_print("\n❌ NO MODEL ACHIEVED ACCEPTABLE PERFORMANCE")
            return None, None, results

    def create_model_validation_plots(self, X_test, y_test, y_pred, dates_test, best_name, results):
        """Create comprehensive validation plots"""
        safe_print("\n--- Creating Model Validation Plots ---")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Model Validation: {best_name}', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.6, color='blue')
        axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Calls')
        axes[0, 0].set_ylabel('Predicted Calls')
        axes[0, 0].set_title(f'Actual vs Predicted (R²={results[best_name]["test_r2"]:.3f})')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Time series comparison
        axes[0, 1].plot(dates_test, y_test.values, 'b-', label='Actual', linewidth=2, marker='o')
        axes[0, 1].plot(dates_test, y_pred, 'r-', label='Predicted', linewidth=2, marker='s', alpha=0.7)
        axes[0, 1].set_title('Predictions vs Actual Over Time')
        axes[0, 1].set_ylabel('Call Volume')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Residuals
        residuals = y_test - y_pred
        axes[0, 2].scatter(y_pred, residuals, alpha=0.6, color='green')
        axes[0, 2].axhline(y=0, color='r', linestyle='--')
        axes[0, 2].set_xlabel('Predicted Calls')
        axes[0, 2].set_ylabel('Residuals')
        axes[0, 2].set_title('Residual Plot')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Residual histogram
        axes[1, 0].hist(residuals, bins=15, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Error metrics
        metrics_text = f"""
        MODEL PERFORMANCE:
        
        R² Score: {results[best_name]['test_r2']:.3f}
        MAE: {results[best_name]['test_mae']:.0f}
        RMSE: {results[best_name]['test_rmse']:.0f}
        MAPE: {results[best_name]['test_mape']:.1f}%
        
        Overfitting: {results[best_name]['overfitting']:.3f}
        
        Mean Actual: {y_test.mean():.0f}
        Mean Predicted: {y_pred.mean():.0f}
        """
        
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='center', fontfamily='monospace')
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].axis('off')
        
        # 6. Model comparison
        model_names = [name for name in results.keys() if 'error' not in results[name]]
        test_r2_scores = [results[name]['test_r2'] for name in model_names]
        
        bars = axes[1, 2].bar(range(len(model_names)), test_r2_scores, alpha=0.7, color='orange')
        axes[1, 2].set_xticks(range(len(model_names)))
        axes[1, 2].set_xticklabels(model_names, rotation=45)
        axes[1, 2].set_ylabel('Test R²')
        axes[1, 2].set_title('Model Comparison')
        axes[1, 2].grid(True, alpha=0.3)
        
        # Highlight best model
        best_idx = model_names.index(best_name)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_validation.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        safe_print(f"   Validation plots saved: {self.output_dir}/model_validation.png")


# ============================================================================

# STEP 5: PREDICTION SYSTEM

# ============================================================================

class PredictionSystem:
    def __init__(self, model_info, top_mail_types, best_lag):
        self.model = model_info['model']
        self.model_name = model_info['model_name']
        self.features = model_info['features']
        self.top_mail_types = top_mail_types
        self.best_lag = best_lag


    def predict_calls(self, mail_input, call_history=None):
        """
        Predict call volume from mail input
        
        Args:
            mail_input: dict like {'DRP Stmt.': 2000, 'Cheque': 1500, ...}
            call_history: dict like {'yesterday': 12000, '2_days_ago': 11500} (optional)
        
        Returns:
            dict with prediction and details
        """
        
        try:
            # Create feature vector
            features = {}
            
            # Mail features
            for mail_type in self.top_mail_types:
                clean_name = mail_type.replace(' ', '').replace('-', '').replace('_', '')[:15]
                volume = mail_input.get(mail_type, 0)
                
                # Today's mail
                features[f"{clean_name}_today"] = volume
                
                # Simulate lags (use current volume as approximation)
                for lag in [1, 2, 3]:
                    features[f"{clean_name}_lag{lag}"] = volume * 0.8 ** lag  # Decay approximation
                
                # Simulate averages
                for window in CONFIG["rolling_windows"]:
                    features[f"{clean_name}_avg{window}"] = volume * 0.9  # Approximation
            
            # Total mail features
            total_mail = sum(mail_input.get(mt, 0) for mt in self.top_mail_types)
            features['total_mail_today'] = total_mail
            
            for lag in [1, 2, 3]:
                features[f'total_mail_lag{lag}'] = total_mail * 0.8 ** lag
            
            for window in CONFIG["rolling_windows"]:
                features[f'total_mail_avg{window}'] = total_mail * 0.9
            
            # Call history (use provided or defaults)
            if call_history:
                features['calls_yesterday'] = call_history.get('yesterday', 12000)
                features['calls_2days_ago'] = call_history.get('2_days_ago', 12000)
            else:
                features['calls_yesterday'] = 12000  # Default average
                features['calls_2days_ago'] = 12000
            
            for window in CONFIG["rolling_windows"]:
                features[f'calls_avg{window}'] = features['calls_yesterday']
            
            # Temporal features (use current date)
            now = datetime.now()
            features['weekday'] = now.weekday()
            features['month'] = now.month
            features['day_of_month'] = now.day
            features['is_month_end'] = 1 if now.day >= 25 else 0
            
            # Convert to array (match training feature order)
            feature_vector = []
            for feat_name in self.features:
                feature_vector.append(features.get(feat_name, 0))
            
            # Predict
            prediction = self.model.predict([feature_vector])[0]
            prediction = max(0, round(prediction))
            
            return {
                'predicted_calls': int(prediction),
                'prediction_lag_days': self.best_lag,
                'model_used': self.model_name,
                'mail_input': mail_input,
                'total_mail_volume': int(total_mail),
                'status': 'success'
            }
            
        except Exception as e:
            return {'error': str(e), 'status': 'failed'}


    # ============================================================================

# MAIN ORCHESTRATOR

# ============================================================================

def main():


    safe_print("COMPREHENSIVE MAIL-TO-CALLS PREDICTION SYSTEM")
    safe_print("=" * 80)
    safe_print("APPROACH:")
    safe_print("1. Load clean call data (Date, ACDCalls) + mail data")
    safe_print("2. Comprehensive EDA with plots and correlations") 
    safe_print("3. Feature engineering with proper lag analysis")
    safe_print("4. Simple models first, evaluate thoroughly")
    safe_print("5. Build prediction system for daily/weekly inputs")
    safe_print("=" * 80)

    try:
        # STEP 1: Data Loading
        data_manager = DataManager()
        call_data = data_manager.load_call_data()
        mail_data = data_manager.load_mail_data()
        merged_data = data_manager.merge_data()
        
        if len(merged_data) < 50:
            safe_print(f"⚠️  WARNING: Only {len(merged_data)} days of data")
            safe_print("   Results may be unreliable with limited data")
        
        # STEP 2: Comprehensive EDA
        eda_analyzer = EDATrendAnalysis(merged_data, data_manager.output_dir)
        eda_results = eda_analyzer.run_full_eda()
        
        # STEP 3: Feature Engineering
        feature_engineer = FeatureEngineer(
            merged_data, 
            eda_results['top_mail_types'], 
            eda_results['best_lag']
        )
        X, y, dates = feature_engineer.create_features()
        
        if len(X) < 30:
            safe_print(f"⚠️  WARNING: Only {len(X)} samples for modeling")
            safe_print("   Consider collecting more data for better results")
        
        # STEP 4: Model Training
        model_builder = ModelBuilder(data_manager.output_dir)
        best_model, best_name, results = model_builder.train_simple_models(X, y, dates)
        
        if not best_model:
            safe_print("\n❌ MODELING FAILED - NO ACCEPTABLE MODEL FOUND")
            safe_print("\nPOSSIBLE REASONS:")
            safe_print("1. Insufficient data (need more days)")
            safe_print("2. Weak relationship between mail and calls")
            safe_print("3. Too much noise in the data")
            safe_print("4. Need different feature engineering approach")
            return {'success': False, 'error': 'No acceptable model found'}
        
        # STEP 5: Create Prediction System
        model_info = {
            'model': best_model,
            'model_name': best_name,
            'features': X.columns.tolist(),
            'performance': results[best_name]
        }
        
        prediction_system = PredictionSystem(
            model_info, 
            eda_results['top_mail_types'], 
            eda_results['best_lag'][0]
        )
        
        # Test the prediction system
        safe_print("\n" + "=" * 80)
        safe_print("STEP 5: TESTING PREDICTION SYSTEM")
        safe_print("=" * 80)
        
        # Create realistic test input
        test_mail_input = {}
        for i, mail_type in enumerate(eda_results['top_mail_types'][:5]):
            test_mail_input[mail_type] = [2000, 1500, 1200, 1000, 800][i]
        
        test_result = prediction_system.predict_calls(test_mail_input)
        
        if test_result['status'] == 'success':
            safe_print("✅ PREDICTION TEST SUCCESSFUL!")
            safe_print(f"   Mail Input: {test_mail_input}")
            safe_print(f"   Predicted Calls (+{test_result['prediction_lag_days']} days): {test_result['predicted_calls']:,}")
            safe_print(f"   Total Mail Volume: {test_result['total_mail_volume']:,}")
            safe_print(f"   Model Used: {test_result['model_used']}")
        else:
            safe_print(f"❌ PREDICTION TEST FAILED: {test_result['error']}")
        
        # Create usage guide
        usage_guide = f"""


    # MAIL-TO-CALLS PREDICTION SYSTEM - USAGE GUIDE

    SYSTEM PERFORMANCE:

    - Best Model: {best_name}
    - Test R²: {results[best_name]['test_r2']:.3f}
    - Test MAE: {results[best_name]['test_mae']:.0f} calls
    - Test MAPE: {results[best_name]['test_mape']:.1f}%
    - Prediction Lag: {eda_results['best_lag'][0]} days

    ## USAGE EXAMPLE:

    import joblib

    # Load model

    model_info = joblib.load('{data_manager.output_dir}/models/best_model.pkl')

    # Predict calls from mail volumes

    mail_today = {{
    {chr(10).join([f"    '{mt}': 1500," for mt in eda_results['top_mail_types'][:3]])}
    }}

    # This predicts calls {eda_results['best_lag'][0]} days in the future

    prediction = prediction_system.predict_calls(mail_today)
    print(f"Predicted calls: {{prediction['predicted_calls']}}")

    TOP MAIL TYPES (by volume):
    {chr(10).join([f"{i+1:2d}. {mt}" for i, mt in enumerate(eda_results['top_mail_types'])])}

    FILES GENERATED:

    - models/best_model.pkl: Trained model
    - models/model_validation.png: Performance plots
    - eda_plots/: All EDA visualizations
    - USAGE_GUIDE.txt: This guide
    """
    
    
        with open(data_manager.output_dir / "USAGE_GUIDE.txt", 'w') as f:
            f.write(usage_guide)
        
        # Final summary
        safe_print("\n" + "=" * 80)
        safe_print("🎯 SUCCESS! COMPREHENSIVE SYSTEM DEPLOYED!")
        safe_print("=" * 80)
        safe_print(f"✅ Data: {len(merged_data)} days merged successfully")
        safe_print(f"✅ EDA: Full analysis with {len(eda_results['top_mail_types'])} top mail types")
        safe_print(f"✅ Features: {len(X.columns)} engineered features")
        safe_print(f"✅ Model: {best_name} (R²={results[best_name]['test_r2']:.3f})")
        safe_print(f"✅ Lag: {eda_results['best_lag'][0]} days optimal")
        safe_print(f"✅ Files: Saved to {data_manager.output_dir}/")
        safe_print("")
        safe_print("PREDICTION CAPABILITY:")
        safe_print("- Input: Daily mail volumes by type")
        safe_print(f"- Output: Call volume {eda_results['best_lag'][0]} days ahead")
        safe_print("- Use cases: Workforce planning, capacity management")
        safe_print("")
        safe_print("NEXT STEPS:")
        safe_print("1. Review EDA plots for insights")
        safe_print("2. Test with your own mail volume inputs")
        safe_print("3. Monitor prediction accuracy over time")
        safe_print("4. Retrain with more data as available")
        
        return {
            'success': True,
            'best_model': best_model,
            'model_name': best_name,
            'performance': results[best_name],
            'prediction_system': prediction_system,
            'top_mail_types': eda_results['top_mail_types'],
            'best_lag': eda_results['best_lag'],
            'output_dir': str(data_manager.output_dir)
        }
    
    
    except Exception as e:
        safe_print(f"❌ SYSTEM ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    result = main()
    if result['success']:
        safe_print("🚀 MAIL-TO-CALLS PREDICTION SYSTEM READY FOR PRODUCTION!")
    else:
        safe_print(f"💥 SYSTEM FAILED: {result['error']}")
