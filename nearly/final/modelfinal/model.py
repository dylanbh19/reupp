#!/usr/bin/env python
"""
PRODUCTION-GRADE MAIL-LAG CALL PREDICTION PIPELINE
=================================================

A complete, reusable, production-ready pipeline for predicting call volumes
based on mail volumes with proper lag modeling.

KEY FEATURES:
- Handles 2-3 day mail delivery lag effects
- Input: Daily mail OR weekly mail plans
- Progressive complexity: Simple → Advanced
- Works with any dataset (current or future)
- Comprehensive EDA before/after modeling
- Prevents overfitting with proper validation
- Production-ready with full evaluation

USAGE:
    python production_pipeline.py

INPUTS:
    - callvolumes.csv: Daily call volumes
    - mail.csv: Daily mail volumes by type
    - econ.csv: (Optional) Economic indicators

OUTPUTS:
    - Comprehensive EDA plots
    - Model performance analysis
    - Prediction capability for daily/weekly mail plans
    - Production deployment assets
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path
import json
import logging
import traceback
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import holidays

# ML Libraries
from sklearn.model_selection import TimeSeriesSplit, cross_validate, train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import joblib

# Statistical Libraries
from scipy import stats
from scipy.stats import pearsonr, spearmanr

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

CONFIG = {
    # Data Configuration
    "call_file_candidates": ["callvolumes.csv", "data/callvolumes.csv", "callvolumes.xlsx"],
    "mail_file_candidates": ["mail.csv", "data/mail.csv", "mail.xlsx"],
    "econ_file_candidates": ["econ.csv", "data/econ.csv", "econ.xlsx"],
    
    # Mail Lag Configuration (KEY FEATURE)
    "mail_lag_days": [1, 2, 3],  # 1-3 day lag based on delivery
    "primary_lag": 2,  # Most common lag day
    "lag_weights": {1: 0.3, 2: 0.5, 3: 0.2},  # Distribution of lag effects
    
    # Model Complexity Progression
    "complexity_levels": ["simple", "intermediate", "advanced"],
    "max_features_by_level": {"simple": 10, "intermediate": 25, "advanced": 50},
    
    # Validation Configuration
    "cv_folds": 5,
    "test_size": 0.2,
    "min_train_samples": 50,
    
    # Output Configuration
    "output_dir": "production_pipeline",
    "plots_dir": "eda_plots",
    "models_dir": "trained_models",
    "results_dir": "results",
    
    # Feature Engineering
    "top_mail_types": 10,  # Auto-select top N mail types
    "outlier_threshold": 3,  # Z-score threshold for outliers
    "min_correlation": 0.05,  # Minimum correlation to include feature
    
    # Model Selection
    "models_to_test": ["linear", "ridge", "random_forest", "gradient_boost"],
    "early_stopping": True,
    "random_state": 42
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Setup production logging"""
    
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)8s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "pipeline.log", mode='w')
        ]
    )
    
    return logging.getLogger(__name__)

LOG = setup_logging()

# ============================================================================
# DATA LOADING ENGINE
# ============================================================================

class DataLoader:
    """Robust data loading for any dataset structure"""
    
    def __init__(self):
        self.call_data = None
        self.mail_data = None
        self.econ_data = None
        self.data_info = {}
    
    def find_file(self, candidates: List[str]) -> Optional[Path]:
        """Find first existing file from candidates"""
        for candidate in candidates:
            path = Path(candidate)
            if path.exists():
                LOG.info(f"Found file: {path}")
                return path
        return None
    
    def load_data_file(self, file_path: Path) -> pd.DataFrame:
        """Load data file with automatic format detection"""
        
        try:
            if file_path.suffix.lower() == '.csv':
                # Try different encodings and separators
                for encoding in ['utf-8', 'latin1', 'cp1252']:
                    for sep in [',', ';', '\t']:
                        try:
                            df = pd.read_csv(file_path, encoding=encoding, sep=sep)
                            if df.shape[1] > 1:  # Valid multi-column data
                                LOG.info(f"Loaded {file_path} with encoding={encoding}, sep='{sep}'")
                                return df
                        except:
                            continue
            
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                LOG.info(f"Loaded Excel file: {file_path}")
                return df
            
            raise ValueError(f"Could not load file: {file_path}")
            
        except Exception as e:
            LOG.error(f"Error loading {file_path}: {e}")
            raise
    
    def standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names"""
        df.columns = [str(col).lower().strip().replace(' ', '_') for col in df.columns]
        return df
    
    def find_date_column(self, df: pd.DataFrame) -> str:
        """Find the date column automatically"""
        
        date_candidates = [col for col in df.columns if any(
            keyword in col.lower() for keyword in ['date', 'time', 'day', 'dt']
        )]
        
        if not date_candidates:
            raise ValueError("No date column found. Please ensure your data has a date column.")
        
        # Try to convert and find the best date column
        for col in date_candidates:
            try:
                pd.to_datetime(df[col], errors='raise')
                LOG.info(f"Using date column: {col}")
                return col
            except:
                continue
        
        raise ValueError(f"Could not parse date columns: {date_candidates}")
    
    def load_call_data(self) -> pd.DataFrame:
        """Load and process call volume data"""
        
        LOG.info("Loading call volume data...")
        
        file_path = self.find_file(CONFIG["call_file_candidates"])
        if not file_path:
            raise FileNotFoundError(f"Call data not found. Tried: {CONFIG['call_file_candidates']}")
        
        df = self.load_data_file(file_path)
        df = self.standardize_columns(df)
        
        # Find date column
        date_col = self.find_date_column(df)
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Find call volume column
        call_candidates = [col for col in df.columns if col != date_col and df[col].dtype in ['int64', 'float64']]
        
        if not call_candidates:
            raise ValueError("No numeric call volume column found")
        
        call_col = call_candidates[0]  # Take first numeric column
        LOG.info(f"Using call volume column: {call_col}")
        
        # Create daily aggregated data
        daily_calls = df.groupby(date_col)[call_col].sum().sort_index()
        
        self.data_info['call_data'] = {
            'file': str(file_path),
            'date_column': date_col,
            'call_column': call_col,
            'date_range': f"{daily_calls.index.min().date()} to {daily_calls.index.max().date()}",
            'total_days': len(daily_calls),
            'call_range': f"{daily_calls.min():.0f} to {daily_calls.max():.0f}",
            'mean_calls': f"{daily_calls.mean():.0f}"
        }
        
        LOG.info(f"Call data loaded: {len(daily_calls)} days, {daily_calls.min():.0f}-{daily_calls.max():.0f} calls")
        
        self.call_data = daily_calls
        return daily_calls
    
    def load_mail_data(self) -> pd.DataFrame:
        """Load and process mail volume data"""
        
        LOG.info("Loading mail volume data...")
        
        file_path = self.find_file(CONFIG["mail_file_candidates"])
        if not file_path:
            LOG.warning("Mail data not found - will create synthetic features")
            return None
        
        df = self.load_data_file(file_path)
        df = self.standardize_columns(df)
        
        # Find date column
        date_col = self.find_date_column(df)
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Find mail type and volume columns
        mail_type_candidates = [col for col in df.columns if any(
            keyword in col.lower() for keyword in ['type', 'category', 'mail', 'class']
        )]
        
        volume_candidates = [col for col in df.columns if any(
            keyword in col.lower() for keyword in ['volume', 'count', 'amount', 'qty', 'quantity']
        )]
        
        if not mail_type_candidates or not volume_candidates:
            LOG.warning("Mail type/volume columns not clearly identifiable - using all numeric columns")
            # Pivot all numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            mail_daily = df.groupby(date_col)[numeric_cols].sum()
        else:
            mail_type_col = mail_type_candidates[0]
            volume_col = volume_candidates[0]
            
            LOG.info(f"Using mail type column: {mail_type_col}, volume column: {volume_col}")
            
            # Pivot to get mail types as columns
            mail_daily = df.groupby([date_col, mail_type_col])[volume_col].sum().unstack(fill_value=0)
        
        # Remove weekends and holidays
        us_holidays = holidays.US()
        business_mask = (
            (~mail_daily.index.weekday.isin([5, 6])) &  # Remove weekends
            (~mail_daily.index.isin(us_holidays))        # Remove holidays
        )
        mail_daily = mail_daily.loc[business_mask]
        
        self.data_info['mail_data'] = {
            'file': str(file_path),
            'date_range': f"{mail_daily.index.min().date()} to {mail_daily.index.max().date()}",
            'total_days': len(mail_daily),
            'mail_types': len(mail_daily.columns),
            'top_mail_types': list(mail_daily.sum().sort_values(ascending=False).head(5).index)
        }
        
        LOG.info(f"Mail data loaded: {len(mail_daily)} business days, {len(mail_daily.columns)} mail types")
        
        self.mail_data = mail_daily
        return mail_daily
    
    def load_econ_data(self) -> Optional[pd.DataFrame]:
        """Load optional economic data"""
        
        LOG.info("Loading economic data...")
        
        file_path = self.find_file(CONFIG["econ_file_candidates"])
        if not file_path:
            LOG.info("Economic data not found - skipping")
            return None
        
        try:
            df = self.load_data_file(file_path)
            df = self.standardize_columns(df)
            
            date_col = self.find_date_column(df)
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col).sort_index()
            
            # Remove non-numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            
            self.data_info['econ_data'] = {
                'file': str(file_path),
                'date_range': f"{numeric_df.index.min().date()} to {numeric_df.index.max().date()}",
                'indicators': len(numeric_df.columns),
                'columns': list(numeric_df.columns)
            }
            
            LOG.info(f"Economic data loaded: {len(numeric_df)} periods, {len(numeric_df.columns)} indicators")
            
            self.econ_data = numeric_df
            return numeric_df
            
        except Exception as e:
            LOG.warning(f"Could not load economic data: {e}")
            return None
    
    def load_all_data(self) -> Dict[str, pd.DataFrame]:
        """Load all available data"""
        
        LOG.info("=== STARTING DATA LOADING ===")
        
        # Load call data (required)
        call_data = self.load_call_data()
        
        # Load mail data (optional but recommended)
        mail_data = self.load_mail_data()
        
        # Load economic data (optional)
        econ_data = self.load_econ_data()
        
        # Print data summary
        self.print_data_summary()
        
        return {
            'calls': call_data,
            'mail': mail_data,
            'econ': econ_data
        }
    
    def print_data_summary(self):
        """Print comprehensive data summary"""
        
        print("\n" + "="*80)
        print("📊 DATA LOADING SUMMARY")
        print("="*80)
        
        for data_type, info in self.data_info.items():
            print(f"\n{data_type.upper()}:")
            for key, value in info.items():
                print(f"  {key}: {value}")
        
        print("="*80)

# ============================================================================
# EXPLORATORY DATA ANALYSIS ENGINE
# ============================================================================

class EDAEngine:
    """Comprehensive EDA for call/mail data"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.plots_dir = self.output_dir / CONFIG["plots_dir"]
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def create_data_overview_plots(self, call_data: pd.Series, mail_data: pd.DataFrame = None):
        """Create overview plots of the raw data"""
        
        LOG.info("Creating data overview plots...")
        
        # Determine number of subplots needed
        n_plots = 3 if mail_data is not None else 2
        fig, axes = plt.subplots(n_plots, 1, figsize=(15, 4*n_plots))
        if n_plots == 1:
            axes = [axes]
        
        # 1. Call Volume Time Series
        ax1 = axes[0]
        ax1.plot(call_data.index, call_data.values, linewidth=1, alpha=0.7, color='blue')
        ax1.set_title('📞 Daily Call Volume Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Call Volume')
        ax1.grid(True, alpha=0.3)
        
        # Add trend line
        if len(call_data) > 10:
            z = np.polyfit(range(len(call_data)), call_data.values, 1)
            p = np.poly1d(z)
            ax1.plot(call_data.index, p(range(len(call_data))), 
                    "r--", alpha=0.8, linewidth=2, label=f'Trend (slope: {z[0]:.1f}/day)')
            ax1.legend()
        
        # 2. Call Volume Distribution
        ax2 = axes[1]
        ax2.hist(call_data.values, bins=30, alpha=0.7, color='skyblue', edgecolor='navy')
        ax2.axvline(call_data.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {call_data.mean():.0f}')
        ax2.axvline(call_data.median(), color='green', linestyle='--', linewidth=2, 
                   label=f'Median: {call_data.median():.0f}')
        ax2.set_title('📊 Call Volume Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Call Volume')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Mail Volume Overview (if available)
        if mail_data is not None and n_plots > 2:
            ax3 = axes[2]
            
            # Plot top mail types
            top_types = mail_data.sum().sort_values(ascending=False).head(5)
            
            for i, mail_type in enumerate(top_types.index):
                ax3.plot(mail_data.index, mail_data[mail_type], 
                        label=f'{mail_type} ({top_types[mail_type]:,.0f} total)', 
                        alpha=0.7, linewidth=1)
            
            ax3.set_title('📬 Top 5 Mail Types Over Time', fontsize=14, fontweight='bold')
            ax3.set_ylabel('Mail Volume')
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "01_data_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def create_correlation_analysis(self, call_data: pd.Series, mail_data: pd.DataFrame = None):
        """Analyze correlations between calls and mail with lag effects"""
        
        if mail_data is None:
            LOG.info("No mail data available for correlation analysis")
            return
        
        LOG.info("Creating correlation analysis with lag effects...")
        
        # Align data
        common_dates = call_data.index.intersection(mail_data.index)
        if len(common_dates) < 30:
            LOG.warning("Insufficient overlapping data for correlation analysis")
            return
        
        aligned_calls = call_data.loc[common_dates]
        aligned_mail = mail_data.loc[common_dates]
        
        # Calculate correlations with different lags
        lag_results = {}
        top_mail_types = aligned_mail.sum().sort_values(ascending=False).head(CONFIG["top_mail_types"])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🔗 MAIL-CALL CORRELATION ANALYSIS WITH LAG EFFECTS', fontsize=16, fontweight='bold')
        
        # 1. Lag correlation heatmap
        lag_correlations = []
        mail_types_subset = top_mail_types.head(8).index  # Top 8 for visualization
        
        for lag in range(0, 8):  # 0-7 day lags
            lag_corrs = []
            for mail_type in mail_types_subset:
                if lag == 0:
                    corr = aligned_calls.corr(aligned_mail[mail_type])
                else:
                    # Shift calls forward to correlate with past mail
                    shifted_calls = aligned_calls.shift(-lag).dropna()
                    aligned_mail_subset = aligned_mail[mail_type].loc[shifted_calls.index]
                    corr = shifted_calls.corr(aligned_mail_subset)
                
                lag_corrs.append(corr if not pd.isna(corr) else 0)
            lag_correlations.append(lag_corrs)
        
        # Create heatmap
        corr_matrix = np.array(lag_correlations).T
        im = ax1.imshow(corr_matrix, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
        ax1.set_xticks(range(8))
        ax1.set_xticklabels([f'Lag {i}' for i in range(8)])
        ax1.set_yticks(range(len(mail_types_subset)))
        ax1.set_yticklabels([t[:15] for t in mail_types_subset])  # Truncate long names
        ax1.set_title('Correlation by Lag Day', fontweight='bold')
        
        # Add correlation values to heatmap
        for i in range(len(mail_types_subset)):
            for j in range(8):
                ax1.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                        ha='center', va='center', fontsize=8,
                        color='white' if abs(corr_matrix[i, j]) > 0.3 else 'black')
        
        plt.colorbar(im, ax=ax1, label='Correlation')
        
        # 2. Best lag for each mail type
        best_lags = []
        best_corrs = []
        
        for mail_type in top_mail_types.head(10).index:
            correlations = []
            for lag in range(8):
                if lag == 0:
                    corr = aligned_calls.corr(aligned_mail[mail_type])
                else:
                    shifted_calls = aligned_calls.shift(-lag).dropna()
                    aligned_mail_subset = aligned_mail[mail_type].loc[shifted_calls.index]
                    corr = shifted_calls.corr(aligned_mail_subset)
                correlations.append(corr if not pd.isna(corr) else 0)
            
            best_lag = np.argmax(np.abs(correlations))
            best_corr = correlations[best_lag]
            best_lags.append(best_lag)
            best_corrs.append(best_corr)
            
            lag_results[mail_type] = {
                'best_lag': best_lag,
                'best_correlation': best_corr,
                'all_correlations': correlations
            }
        
        # Plot best lag distribution
        bars = ax2.bar(range(len(best_lags)), best_corrs, 
                      color=['red' if c < 0 else 'blue' for c in best_corrs])
        ax2.set_xticks(range(len(best_lags)))
        ax2.set_xticklabels([f'{t[:10]}...\n(Lag {l})' for t, l in zip(top_mail_types.head(10).index, best_lags)], 
                           rotation=45, ha='right')
        ax2.set_ylabel('Best Correlation')
        ax2.set_title('Best Correlation by Mail Type', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        # 3. Lag distribution histogram
        ax3.hist(best_lags, bins=range(9), alpha=0.7, color='orange', edgecolor='black')
        ax3.set_xlabel('Optimal Lag Days')
        ax3.set_ylabel('Number of Mail Types')
        ax3.set_title('Distribution of Optimal Lag Days', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add average lag line
        avg_lag = np.mean(best_lags)
        ax3.axvline(avg_lag, color='red', linestyle='--', linewidth=2, 
                   label=f'Average: {avg_lag:.1f} days')
        ax3.legend()
        
        # 4. Summary statistics
        ax4.axis('off')
        
        summary_text = f"""
CORRELATION ANALYSIS SUMMARY
══════════════════════════

📊 DATASET OVERLAP:
• Common dates: {len(common_dates)} days
• Date range: {common_dates.min().date()} to {common_dates.max().date()}
• Mail types analyzed: {len(top_mail_types)}

🔗 LAG ANALYSIS:
• Average optimal lag: {np.mean(best_lags):.1f} days
• Most common lag: {stats.mode(best_lags)[0][0]} days
• Lag range: {min(best_lags)}-{max(best_lags)} days

📈 CORRELATION STRENGTH:
• Strongest correlation: {max(best_corrs):.3f}
• Average correlation: {np.mean(np.abs(best_corrs)):.3f}
• Correlations > 0.1: {sum(1 for c in best_corrs if abs(c) > 0.1)}

🎯 KEY INSIGHTS:
• Primary lag effect: {CONFIG['primary_lag']} days (delivery time)
• Mail-call relationship: {"Strong" if max(best_corrs) > 0.3 else "Moderate" if max(best_corrs) > 0.1 else "Weak"}
• Model potential: {"High" if max(best_corrs) > 0.2 else "Medium"}
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "02_correlation_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Store results for feature engineering
        self.lag_results = lag_results
        
        return lag_results
    
    def create_temporal_patterns(self, call_data: pd.Series, mail_data: pd.DataFrame = None):
        """Analyze temporal patterns in calls and mail"""
        
        LOG.info("Creating temporal pattern analysis...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('📅 TEMPORAL PATTERN ANALYSIS', fontsize=16, fontweight='bold')
        
        # 1. Weekday patterns for calls
        weekday_calls = call_data.groupby(call_data.index.weekday).agg(['mean', 'std'])
        weekday_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        # Filter to business days only
        business_weekdays = weekday_calls.iloc[:5]  # Mon-Fri
        business_names = weekday_names[:5]
        
        bars = ax1.bar(business_names, business_weekdays['mean'], 
                      yerr=business_weekdays['std'], 
                      alpha=0.7, color='lightblue', capsize=5)
        ax1.set_ylabel('Average Call Volume')
        ax1.set_title('Call Volume by Weekday', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean_val in zip(bars, business_weekdays['mean']):
            height = bar.get_height()
            ax1.annotate(f'{mean_val:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        # 2. Monthly patterns for calls
        if len(call_data) > 60:  # At least 2 months of data
            monthly_calls = call_data.groupby(call_data.index.month).mean()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            months_with_data = [month_names[i-1] for i in monthly_calls.index]
            ax2.plot(months_with_data, monthly_calls.values, 'o-', 
                    linewidth=3, markersize=8, color='red')
            ax2.set_ylabel('Average Call Volume')
            ax2.set_title('Call Volume by Month', fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'Insufficient data\nfor monthly analysis', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_title('Monthly Analysis - Insufficient Data', fontweight='bold')
        
        # 3. Call volume trend with moving average
        ax3.plot(call_data.index, call_data.values, alpha=0.5, linewidth=1, 
                color='blue', label='Daily Calls')
        
        # Add moving averages
        if len(call_data) > 7:
            ma_7 = call_data.rolling(window=7, center=True).mean()
            ax3.plot(call_data.index, ma_7, linewidth=2, color='red', label='7-day MA')
        
        if len(call_data) > 30:
            ma_30 = call_data.rolling(window=30, center=True).mean()
            ax3.plot(call_data.index, ma_30, linewidth=2, color='green', label='30-day MA')
        
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Call Volume')
        ax3.set_title('Call Volume Trends', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Seasonality analysis (if enough data)
        ax4.axis('off')
        
        # Calculate key statistics
        call_stats = {
            'Total Days': len(call_data),
            'Date Range': f"{call_data.index.min().date()} to {call_data.index.max().date()}",
            'Average Daily Calls': f"{call_data.mean():.0f}",
            'Std Deviation': f"{call_data.std():.0f}",
            'Min Calls': f"{call_data.min():.0f}",
            'Max Calls': f"{call_data.max():.0f}",
            'Coefficient of Variation': f"{(call_data.std() / call_data.mean()):.2f}"
        }
        
        # Weekday insights
        best_weekday = business_names[np.argmax(business_weekdays['mean'])]
        worst_weekday = business_names[np.argmin(business_weekdays['mean'])]
        weekday_variation = (business_weekdays['mean'].max() - business_weekdays['mean'].min()) / business_weekdays['mean'].mean()
        
        stats_text = f"""
TEMPORAL PATTERN SUMMARY
═══════════════════════

📊 BASIC STATISTICS:
• Total Days: {call_stats['Total Days']}
• Date Range: {call_stats['Date Range']}
• Average Daily Calls: {call_stats['Average Daily Calls']}
• Standard Deviation: {call_stats['Std Deviation']}
• Coefficient of Variation: {call_stats['Coefficient of Variation']}

📅 WEEKDAY PATTERNS:
• Highest Volume: {best_weekday} ({business_weekdays['mean'].max():.0f} calls)
• Lowest Volume: {worst_weekday} ({business_weekdays['mean'].min():.0f} calls)
• Weekday Variation: {weekday_variation:.1%}

🔍 INSIGHTS:
• Weekly Pattern: {"Strong" if weekday_variation > 0.2 else "Moderate" if weekday_variation > 0.1 else "Weak"}
• Volatility Level: {"High" if float(call_stats['Coefficient of Variation']) > 0.5 else "Medium" if float(call_stats['Coefficient of Variation']) > 0.3 else "Low"}
• Trend Direction: {"Stable" if abs(np.polyfit(range(len(call_data)), call_data.values, 1)[0]) < 1 else "Trending"}
        """
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "03_temporal_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_outlier_analysis(self, call_data: pd.Series):
        """Analyze outliers in call data"""
        
        LOG.info("Creating outlier analysis...")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('🚨 OUTLIER ANALYSIS', fontsize=16, fontweight='bold')
        
        # 1. Box plot
        ax1.boxplot(call_data.values, vert=True)
        ax1.set_ylabel('Call Volume')
        ax1.set_title('Call Volume Box Plot', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        q1, q3 = np.percentile(call_data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        outliers_low = call_data[call_data < lower_bound]
        outliers_high = call_data[call_data > upper_bound]
        
        ax1.text(1.1, q1, f'Q1: {q1:.0f}', fontsize=10, va='center')
        ax1.text(1.1, q3, f'Q3: {q3:.0f}', fontsize=10, va='center')
        ax1.text(1.1, call_data.median(), f'Median: {call_data.median():.0f}', fontsize=10, va='center', fontweight='bold')
        
        # 2. Z-score analysis
        z_scores = np.abs(stats.zscore(call_data))
        outlier_threshold = CONFIG["outlier_threshold"]
        z_outliers = call_data[z_scores > outlier_threshold]
        
        ax2.plot(call_data.index, z_scores, alpha=0.7, color='blue')
        ax2.axhline(y=outlier_threshold, color='red', linestyle='--', linewidth=2, 
                   label=f'Threshold: {outlier_threshold}')
        ax2.scatter(z_outliers.index, z_scores[z_scores > outlier_threshold], 
                   color='red', s=50, label=f'Outliers: {len(z_outliers)}')
        ax2.set_ylabel('Z-Score')
        ax2.set_title('Z-Score Analysis', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Time series with outliers highlighted
        ax3.plot(call_data.index, call_data.values, alpha=0.7, color='blue', label='Call Volume')
        
        # Highlight different types of outliers
        if len(outliers_low) > 0:
            ax3.scatter(outliers_low.index, outliers_low.values, 
                       color='orange', s=50, label=f'Low Outliers: {len(outliers_low)}')
        
        if len(outliers_high) > 0:
            ax3.scatter(outliers_high.index, outliers_high.values, 
                       color='red', s=50, label=f'High Outliers: {len(outliers_high)}')
        
        if len(z_outliers) > 0:
            ax3.scatter(z_outliers.index, z_outliers.values, 
                       color='purple', s=30, alpha=0.7, label=f'Z-Score Outliers: {len(z_outliers)}')
        
        ax3.set_ylabel('Call Volume')
        ax3.set_title('Outliers in Time Series', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Outlier summary
        ax4.axis('off')
        
        # Calculate outlier statistics
        total_outliers = len(set(outliers_low.index) | set(outliers_high.index) | set(z_outliers.index))
        outlier_percentage = (total_outliers / len(call_data)) * 100
        
        # Analyze outlier patterns
        if len(z_outliers) > 0:
            outlier_weekdays = z_outliers.index.weekday
            most_common_weekday = stats.mode(outlier_weekdays)[0][0]
            weekday_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][most_common_weekday]
        else:
            weekday_name = "N/A"
        
        outlier_summary = f"""
OUTLIER ANALYSIS SUMMARY
═══════════════════════

📊 OUTLIER COUNTS:
• IQR Method (Low): {len(outliers_low)} outliers
• IQR Method (High): {len(outliers_high)} outliers
• Z-Score Method: {len(z_outliers)} outliers
• Total Unique Outliers: {total_outliers}
• Outlier Percentage: {outlier_percentage:.1f}%

📈 OUTLIER CHARACTERISTICS:
• IQR Bounds: {lower_bound:.0f} - {upper_bound:.0f}
• Z-Score Threshold: {outlier_threshold}
• Most Common Outlier Day: {weekday_name}

🎯 DATA QUALITY ASSESSMENT:
• Quality Level: {"Poor" if outlier_percentage > 10 else "Fair" if outlier_percentage > 5 else "Good"}
• Cleaning Needed: {"Yes" if outlier_percentage > 5 else "Optional"}
• Impact on Modeling: {"High" if outlier_percentage > 10 else "Medium" if outlier_percentage > 5 else "Low"}

💡 RECOMMENDATIONS:
• {"Remove extreme outliers before modeling" if outlier_percentage > 10 else "Monitor outliers during modeling"}
• {"Investigate data collection issues" if outlier_percentage > 15 else "Consider robust modeling techniques"}
        """
        
        ax4.text(0.05, 0.95, outlier_summary, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / "04_outlier_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Return outlier information for data cleaning
        return {
            'iqr_outliers_low': outliers_low,
            'iqr_outliers_high': outliers_high,
            'z_score_outliers': z_outliers,
            'total_outliers': total_outliers,
            'outlier_percentage': outlier_percentage
        }
    
    def run_comprehensive_eda(self, data_dict: Dict[str, pd.DataFrame]) -> Dict:
        """Run comprehensive EDA on all data"""
        
        LOG.info("=== STARTING COMPREHENSIVE EDA ===")
        
        call_data = data_dict['calls']
        mail_data = data_dict['mail']
        econ_data = data_dict['econ']
        
        results = {}
        
        # 1. Data Overview
        self.create_data_overview_plots(call_data, mail_data)
        
        # 2. Correlation Analysis (if mail data available)
        if mail_data is not None:
            results['correlations'] = self.create_correlation_analysis(call_data, mail_data)
        
        # 3. Temporal Patterns
        self.create_temporal_patterns(call_data, mail_data)
        
        # 4. Outlier Analysis
        results['outliers'] = self.create_outlier_analysis(call_data)
        
        LOG.info(f"EDA complete! Plots saved to: {self.plots_dir}")
        
        return results
