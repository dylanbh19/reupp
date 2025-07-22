PS C:\Users\BhungarD\OneDrive - Computershare\Desktop\finprod> & C:/Users/BhungarD/python.exe "c:/Users/BhungarD/OneDrive - Computershare/Desktop/finprod/compr.py"

 ╔═══════════════════════════════════════════════════════════════════════════╗
 ║                    CALL DATA ANALYSIS & OUTLIER CLEANER                   ║
 ║                           ASCII Edition v1.0                             ║
 ╠═══════════════════════════════════════════════════════════════════════════╣
 ║  > Visualize both call data sources (overlay plot)                       ║
 ║  > Detect and remove statistical outliers                                ║
 ║  > Retrain baseline/enhanced/hybrid models with clean data               ║
 ║  > Compare performance improvements                                       ║
 ╚═══════════════════════════════════════════════════════════════════════════╝

┌─ ANALYSIS STEPS ─────────────────────────────────────────────────────────┐
│ 1. Visualize both call data sources (overlay)           │
│ 2. Detect and remove outliers                           │
│ 3. Retrain models with clean data                       │
│ 4. Compare baseline vs enhanced vs hybrid               │
└──────────────────────────────────────────────────────────────────────┘

===========================================================================
===================  STEP 1: LOADING CALL DATA SOURCES  ===================
===========================================================================
║ Loading call volume data...
║ Found file: data\callvolumes.csv
║ Loading call intent data...
║ Found file: data\callintent.csv

┌─ CALL VOLUMES STATISTICS ───────────────────────────┐
│ Days                 :                       550 │
│ Date Range           :  2024-01-01 to 2025-07-03 │
│ Volume Range         :                 0 to 2903 │
│ Volume Mean          :                       439 │
└──────────────────────────────────────────────────┘
║ WARNING: No call intent data found!
║ Creating call sources overlay plot...
║ Call sources plot saved: call_data_analysis\call_sources_analysis.png
║ WARNING: Insufficient overlap for scaling

┌─ DATA COMBINATION RESULTS ──────────────────────────┐
│ Overlapping Days     :                         0 │
│ Combined Days        :                       550 │
│ Combined Range       :                 0 to 2903 │
└──────────────────────────────────────────────────┘

===========================================================================
==================  STEP 2: OUTLIER DETECTION & REMOVAL  ==================
===========================================================================

┌─ ORIGINAL DATA STATISTICS ──────────────────────────┐
│ Original Days        :                       550 │
│ Original Range       :                 0 to 2903 │
│ Original Mean        :                       439 │
│ Original Std         :                       480 │
└──────────────────────────────────────────────────┘

┌─ OUTLIER DETECTION METHOD ──────────────────────────┐
│ Method               :                       IQR │
│ Multiplier           :                       2.5 │
│ Q25                  :                        61 │
│ Q75                  :                       655 │
│ IQR                  :                       594 │
│ Lower Bound          :                     -1425 │
│ Upper Bound          :                      2141 │
└──────────────────────────────────────────────────┘

┌─ OUTLIERS DETECTED ──────────────────────────────────────────────────────┐
┌────────────┬──────────┬───────┬───────────┐
│    Date    │  Weekday │ Calls │ vs Median │
├────────────┼──────────┼───────┼───────────┤
│ 2024-11-26 │  Tuesday │ 2,480 │   +2,153  │
│ 2024-12-02 │  Monday  │ 2,812 │   +2,485  │
│ 2024-12-16 │  Monday  │ 2,341 │   +2,014  │
│ 2025-01-02 │ Thursday │ 2,630 │   +2,303  │
│ 2025-01-03 │  Friday  │ 2,903 │   +2,576  │
│ 2025-01-06 │  Monday  │ 2,690 │   +2,363  │
│ 2025-01-13 │  Monday  │ 2,294 │   +1,967  │
│ 2025-03-03 │  Monday  │ 2,475 │   +2,148  │
└────────────┴──────────┴───────┴───────────┘

┌─ OUTLIERS BY WEEKDAY ───────────────────────────────────────────────┐
┌──────────┬───────┬────────────┐
│  Weekday │ Count │ Percentage │
├──────────┼───────┼────────────┤
│  Friday  │   1   │    12.5%   │
│  Monday  │   5   │    62.5%   │
│ Thursday │   1   │    12.5%   │
│  Tuesday │   1   │    12.5%   │
└──────────┴───────┴────────────┘

┌─ CLEANED DATA STATISTICS ───────────────────────────┐
│ Outliers Detected    :                         8 │
│ Outliers Percentage  :                      1.5% │
│ Cleaned Days         :                       542 │
│ Cleaned Range        :                 0 to 2009 │
│ Cleaned Mean         :                       408 │
│ Cleaned Std          :                       405 │
└──────────────────────────────────────────────────┘
║ Outlier analysis plot saved: call_data_analysis\outlier_removal_analysis.png

===========================================================================
===============  STEP 3: RETRAINING MODELS WITH CLEAN DATA  ===============
===========================================================================
║ Found file: data\mail.csv

┌─ CLEAN COMBINED DATA ───────────────────────────────┐
│ Combined Data Shape  :                 341 x 232 │
│ Clean Call Range     :                 3 to 2009 │
│ Clean Call Mean      :                       582 │
│ Clean Call Std       :                       391 │
└──────────────────────────────────────────────────┘
║ Creating baseline features with clean data...
║ Clean baseline features: 340 samples x 19 features
║ Creating enhanced features with clean data...
║ Clean enhanced features: 340 samples x 27 features
║ Training baseline model with clean data...
║   CLEAN BASELINE 10% quantile: MAE=246
║   CLEAN BASELINE 25% quantile: MAE=174
║   CLEAN BASELINE 50% quantile: MAE=152
║   CLEAN BASELINE 75% quantile: MAE=178
║   CLEAN BASELINE 90% quantile: MAE=450
║ Training enhanced model with clean data...
║   CLEAN ENHANCED 10% quantile: MAE=200
║   CLEAN ENHANCED 25% quantile: MAE=147
║   CLEAN ENHANCED 50% quantile: MAE=129
║   CLEAN ENHANCED 75% quantile: MAE=240
║   CLEAN ENHANCED 90% quantile: MAE=284
║ Testing CLEAN HYBRID approach...
║ Clean model comparison saved: call_data_analysis\clean_model_comparison.png

===========================================================================
====================  CLEAN MODEL PERFORMANCE RESULTS  ====================
===========================================================================

┌─ OVERALL PERFORMANCE (CLEAN DATA) ──────────────────────────────────────┐
├──────────┬─────┬──────┬───────┬──────────┤
│   Model  │ MAE │ RMSE │   R²  │ Accuracy │
├──────────┼─────┼──────┼───────┼──────────┤
│ Baseline │ 152 │  224 │ 0.283 │   64.3%  │
│ Enhanced │ 129 │  183 │ 0.520 │   69.7%  │
│  Hybrid  │ 143 │  219 │ 0.314 │   66.5%  │
└──────────┴─────┴──────┴───────┴──────────┘

╔═══════════════════════════════════════════════════════════════════════════╗
║                          BEST MODEL:       ENHANCED                          ║
║                          BEST MAE:  129                              ║
╚═══════════════════════════════════════════════════════════════════════════╝

┌─ WEEKDAY PERFORMANCE (CLEAN DATA) ──────────────────────────────────────┐
├───────────┬──────────┬──────────┬────────┤
│  Weekday  │ Baseline │ Enhanced │ Hybrid │
├───────────┼──────────┼──────────┼────────┤
│   Monday  │    163   │    159   │   159  │
│  Tuesday  │    89    │    93    │   93   │
│ Wednesday │    113   │    102   │   102  │
│  Thursday │    96    │    59    │   59   │
│   Friday  │    302   │    236   │   302  │
└───────────┴──────────┴──────────┴────────┘

┌─ MODEL IMPROVEMENTS ────────────────────────────────────────────────────┐
├──────────────────────┬─────────────┬────────┤
│      Comparison      │ Improvement │ Status │
├──────────────────────┼─────────────┼────────┤
│ Enhanced vs Baseline │    +15.1%   │ Better │
│  Hybrid vs Baseline  │    +5.9%    │ Better │
└──────────────────────┴─────────────┴────────┘

===========================================================================
===========================  ANALYSIS COMPLETE  ===========================
===========================================================================

┌─ COMPLETION SUMMARY ────────────────────────────────┐
│ Total Time           :              64.5 seconds │
│ Results Directory    :        call_data_analysis │
│ Status               :                   SUCCESS │
└──────────────────────────────────────────────────┘

╔═══════════════════════════════════════════════════════════════════════════╗
║                            ANALYSIS COMPLETE!                            ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  ✓ Call data sources visualized                                          ║
║  ✓ Outliers detected and removed                                         ║
║  ✓ Models retrained with clean data                                      ║
║  ✓ Performance comparison generated                                       ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  📁 All results in:             call_data_analysis             ║







#!/usr/bin/env python
# call_data_visualizer.py
# =========================================================
# CALL DATA SOURCES VISUALIZATION & OUTLIER CLEANING
# =========================================================
# 1. Plot both call volume sources (overlay)
# 2. Identify and remove outliers  
# 3. Re-run hybrid/baseline models with clean data
# =========================================================

from pathlib import Path
import json
import logging
import sys
import warnings
from datetime import datetime, timedelta
import traceback

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import holidays

try:
    import joblib
except ImportError:
    import pickle as joblib

# Handle sklearn imports
try:
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import QuantileRegressor, Ridge, LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("ERROR: scikit-learn not available!")
    sys.exit(1)

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CFG = {
    "top_mail_types": [
        "Reject_Ltrs", "Cheque 1099", "Exercise_Converted",
        "SOI_Confirms", "Exch_chks", "ACH_Debit_Enrollment",
        "Transfer", "COA", "NOTC_WITHDRAW", "Repl_Chks"
    ],
    "quantiles": [0.1, 0.25, 0.5, 0.75, 0.9],
    "bootstrap_samples": 15,
    "output_dir": "call_data_analysis",
    
    # Outlier detection settings
    "outlier_method": "iqr",  # 'iqr', 'zscore', or 'percentile'
    "iqr_multiplier": 2.5,
    "zscore_threshold": 3,
    "percentile_lower": 1,
    "percentile_upper": 99
}

# ============================================================================
# ASCII ART & FORMATTING
# ============================================================================

def print_ascii_header():
    """Print ASCII art header"""
    print("""
 ╔═══════════════════════════════════════════════════════════════════════════╗
 ║                    CALL DATA ANALYSIS & OUTLIER CLEANER                   ║
 ║                           ASCII Edition v1.0                             ║
 ╠═══════════════════════════════════════════════════════════════════════════╣
 ║  > Visualize both call data sources (overlay plot)                       ║
 ║  > Detect and remove statistical outliers                                ║ 
 ║  > Retrain baseline/enhanced/hybrid models with clean data               ║
 ║  > Compare performance improvements                                       ║
 ╚═══════════════════════════════════════════════════════════════════════════╝
    """)

def print_ascii_section(title):
    """Print ASCII section header"""
    width = 75
    title_len = len(title)
    padding = (width - title_len - 4) // 2
    
    print(f"\n{'='*width}")
    print(f"{'='*padding}  {title}  {'='*(width - padding - title_len - 4)}")
    print(f"{'='*width}")

def print_ascii_table(headers, rows, title=""):
    """Print ASCII formatted table"""
    if title:
        print(f"\n┌─ {title} " + "─" * (70 - len(title)) + "┐")
    
    # Calculate column widths
    col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 for i in range(len(headers))]
    
    # Header
    header_line = "├" + "┬".join("─" * w for w in col_widths) + "┤"
    if not title:
        header_line = "┌" + "┬".join("─" * w for w in col_widths) + "┐"
    
    print(header_line)
    
    header_row = "│" + "│".join(f" {headers[i]:^{col_widths[i]-1}}" for i in range(len(headers))) + "│"
    print(header_row)
    
    separator = "├" + "┼".join("─" * w for w in col_widths) + "┤"
    print(separator)
    
    # Data rows
    for row in rows:
        data_row = "│" + "│".join(f" {str(row[i]):^{col_widths[i]-1}}" for i in range(len(row))) + "│"
        print(data_row)
    
    # Footer
    footer = "└" + "┴".join("─" * w for w in col_widths) + "┘"
    print(footer)

def print_ascii_stats(title, stats_dict):
    """Print statistics in ASCII box"""
    print(f"\n┌─ {title} " + "─" * (50 - len(title)) + "┐")
    
    for key, value in stats_dict.items():
        if isinstance(value, float):
            if abs(value) >= 1000:
                value_str = f"{value:,.0f}"
            else:
                value_str = f"{value:.2f}"
        else:
            value_str = str(value)
            
        print(f"│ {key:<20} : {value_str:>25} │")
    
    print("└" + "─" * 50 + "┘")

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Setup logging with ASCII formatting"""
    
    try:
        output_dir = Path(CFG["output_dir"])
        output_dir.mkdir(exist_ok=True)
        
        logger = logging.getLogger("CallDataAnalyzer")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        
        # Console handler with ASCII formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter("║ %(asctime)s │ %(levelname)8s │ %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        try:
            file_handler = logging.FileHandler(output_dir / "call_analysis.log", mode='w', encoding='utf-8')
            file_formatter = logging.Formatter("%(asctime)s | %(levelname)8s | %(message)s")
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"║ Warning: Could not create log file: {e}")
        
        return logger
        
    except Exception as e:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        logger = logging.getLogger("CallDataAnalyzer")
        logger.warning(f"Advanced logging failed: {e}")
        return logger

LOG = setup_logging()

# ============================================================================
# DATA LOADING UTILITIES  
# ============================================================================

def _to_date(s):
    """Convert to date with error handling"""
    try:
        return pd.to_datetime(s, errors="coerce").dt.date
    except Exception as e:
        LOG.warning(f"Date conversion error: {e}")
        return pd.to_datetime(s, errors="coerce", infer_datetime_format=True).dt.date

def _find_file(candidates):
    """Find file from candidates"""
    for p in candidates:
        try:
            path = Path(p)
            if path.exists():
                print(f"║ Found file: {path}")
                return path
        except Exception as e:
            print(f"║ Warning: Error checking path {p}: {e}")
            continue
    
    print(f"║ ERROR: No files found from candidates: {candidates}")
    raise FileNotFoundError(f"None found: {candidates}")

# ============================================================================
# CALL DATA VISUALIZATION
# ============================================================================

def load_and_visualize_call_sources():
    """Load both call data sources and visualize them"""
    
    print_ascii_section("STEP 1: LOADING CALL DATA SOURCES")
    
    try:
        # Load call volumes
        print("║ Loading call volume data...")
        vol_path = _find_file(["callvolumes.csv", "data/callvolumes.csv"])
        df_vol = pd.read_csv(vol_path)
        df_vol.columns = [c.lower().strip() for c in df_vol.columns]
        
        # Find date column
        date_cols = [c for c in df_vol.columns if "date" in c.lower()]
        if not date_cols:
            raise ValueError("No date column found in call volumes")
        
        dcol_v = date_cols[0]
        df_vol[dcol_v] = pd.to_datetime(df_vol[dcol_v], errors='coerce')
        df_vol = df_vol.dropna(subset=[dcol_v])
        
        # Get first numeric column
        vol_daily = df_vol.groupby(dcol_v)[df_vol.columns.difference([dcol_v])[0]].sum()
        vol_daily = vol_daily.sort_index()
        
        # Load call intent
        print("║ Loading call intent data...")
        intent_path = _find_file(["callintent.csv", "data/callintent.csv", "callintetn.csv"])
        df_int = pd.read_csv(intent_path)
        df_int.columns = [c.lower().strip() for c in df_int.columns]
        
        # Find date column
        date_cols = [c for c in df_int.columns if "date" in c.lower() or "conversation" in c.lower()]
        if not date_cols:
            raise ValueError("No date column found in call intent")
        
        dcol_i = date_cols[0]
        df_int[dcol_i] = pd.to_datetime(df_int[dcol_i], errors='coerce')
        df_int = df_int.dropna(subset=[dcol_i])
        
        int_daily = df_int.groupby(dcol_i).size()
        int_daily = int_daily.sort_index()
        
        # Print ASCII formatted statistics
        vol_stats = {
            "Days": len(vol_daily),
            "Date Range": f"{vol_daily.index.min().date()} to {vol_daily.index.max().date()}",
            "Volume Range": f"{vol_daily.min():.0f} to {vol_daily.max():.0f}",
            "Volume Mean": f"{vol_daily.mean():.0f}"
        }
        print_ascii_stats("CALL VOLUMES STATISTICS", vol_stats)
        
        if len(int_daily) > 0:
            int_stats = {
                "Days": len(int_daily),
                "Date Range": f"{int_daily.index.min().date()} to {int_daily.index.max().date()}",
                "Intent Range": f"{int_daily.min():.0f} to {int_daily.max():.0f}",
                "Intent Mean": f"{int_daily.mean():.0f}"
            }
            print_ascii_stats("CALL INTENT STATISTICS", int_stats)
        else:
            print("║ WARNING: No call intent data found!")
        
        # Create visualization
        create_call_sources_plot(vol_daily, int_daily)
        
        # Analyze overlaps and scaling
        overlap = vol_daily.index.intersection(int_daily.index)
        overlap_stats = {
            "Overlapping Days": len(overlap)
        }
        
        if len(overlap) >= 5:
            scale = int_daily.loc[overlap].mean() / vol_daily.loc[overlap].mean()
            vol_daily_scaled = vol_daily * scale
            overlap_stats["Scale Factor"] = f"{scale:.2f}"
            overlap_stats["Scaled Vol Range"] = f"{vol_daily_scaled.min():.0f} to {vol_daily_scaled.max():.0f}"
            
            # Combine scaled data
            calls_combined = vol_daily_scaled.combine_first(int_daily).sort_index()
        else:
            print("║ WARNING: Insufficient overlap for scaling")
            calls_combined = vol_daily.combine_first(int_daily).sort_index()
            scale = 1.0
        
        overlap_stats["Combined Days"] = len(calls_combined)
        overlap_stats["Combined Range"] = f"{calls_combined.min():.0f} to {calls_combined.max():.0f}"
        
        print_ascii_stats("DATA COMBINATION RESULTS", overlap_stats)
        
        return {
            'vol_daily': vol_daily,
            'int_daily': int_daily,
            'calls_combined': calls_combined,
            'overlap_days': len(overlap),
            'scale_factor': scale
        }
        
    except Exception as e:
        print(f"║ ERROR: Error loading call data: {e}")
        print(f"║ {traceback.format_exc()}")
        return None

def create_call_sources_plot(vol_daily, int_daily):
    """Create overlay plot of both call data sources"""
    
    print("║ Creating call sources overlay plot...")
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Call Data Sources Analysis', fontsize=16, fontweight='bold')
        
        # 1. Raw data overlay
        if len(vol_daily) > 0:
            ax1.plot(vol_daily.index, vol_daily.values, 'b-', alpha=0.7, linewidth=1, label='Call Volumes')
        if len(int_daily) > 0:
            ax1.plot(int_daily.index, int_daily.values, 'r-', alpha=0.7, linewidth=1, label='Call Intent')
        
        ax1.set_title('Raw Call Data Sources (Overlay)')
        ax1.set_ylabel('Call Count')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Identify potential outliers visually
        if len(vol_daily) > 0:
            vol_q99 = vol_daily.quantile(0.99)
            outliers_vol = vol_daily[vol_daily > vol_q99]
            if not outliers_vol.empty:
                ax1.scatter(outliers_vol.index, outliers_vol.values, color='blue', s=50, marker='x', label='Vol Outliers')
        
        if len(int_daily) > 0:
            int_q99 = int_daily.quantile(0.99)
            outliers_int = int_daily[int_daily > int_q99]
            if not outliers_int.empty:
                ax1.scatter(outliers_int.index, outliers_int.values, color='red', s=50, marker='x', label='Intent Outliers')
        
        # 2. Scaled comparison
        if len(vol_daily) > 0 and len(int_daily) > 0:
            # Find overlapping period
            overlap = vol_daily.index.intersection(int_daily.index)
            if len(overlap) > 0:
                scale = int_daily.loc[overlap].mean() / vol_daily.loc[overlap].mean()
                vol_scaled = vol_daily * scale
                
                ax2.plot(vol_scaled.index, vol_scaled.values, 'b-', alpha=0.7, linewidth=1, label=f'Call Volumes (x{scale:.1f})')
                ax2.plot(int_daily.index, int_daily.values, 'r-', alpha=0.7, linewidth=1, label='Call Intent')
                ax2.set_title(f'Scaled Comparison (Scale Factor: {scale:.2f})')
                ax2.set_ylabel('Call Count')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        # 3. Distribution comparison
        if len(vol_daily) > 0:
            ax3.hist(vol_daily.values, bins=50, alpha=0.7, color='blue', label='Call Volumes')
        if len(int_daily) > 0:
            ax3.hist(int_daily.values, bins=50, alpha=0.7, color='red', label='Call Intent')
        
        ax3.set_title('Distribution Comparison')
        ax3.set_xlabel('Call Count')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.set_yscale('log')  # Log scale to see outliers better
        
        # 4. Statistics summary
        ax4.axis('off')
        
        stats_text = "CALL DATA STATISTICS\n\n"
        
        if len(vol_daily) > 0:
            stats_text += f"CALL VOLUMES:\n"
            stats_text += f"  Days: {len(vol_daily)}\n"
            stats_text += f"  Range: {vol_daily.min():.0f} to {vol_daily.max():.0f}\n"
            stats_text += f"  Mean: {vol_daily.mean():.0f}\n"
            stats_text += f"  Median: {vol_daily.median():.0f}\n"
            stats_text += f"  Std: {vol_daily.std():.0f}\n"
            
            # Identify potential outliers
            q75 = vol_daily.quantile(0.75)
            q25 = vol_daily.quantile(0.25)
            iqr = q75 - q25
            outlier_threshold = q75 + 2.5 * iqr
            outliers = vol_daily[vol_daily > outlier_threshold]
            stats_text += f"  Outliers (>Q3+2.5*IQR): {len(outliers)}\n"
            if not outliers.empty:
                stats_text += f"  Max outlier: {outliers.max():.0f}\n"
        
        stats_text += "\n"
        
        if len(int_daily) > 0:
            stats_text += f"CALL INTENT:\n"
            stats_text += f"  Days: {len(int_daily)}\n"
            stats_text += f"  Range: {int_daily.min():.0f} to {int_daily.max():.0f}\n"
            stats_text += f"  Mean: {int_daily.mean():.0f}\n"
            stats_text += f"  Median: {int_daily.median():.0f}\n"
            stats_text += f"  Std: {int_daily.std():.0f}\n"
            
            # Identify potential outliers
            q75 = int_daily.quantile(0.75)
            q25 = int_daily.quantile(0.25)
            iqr = q75 - q25
            outlier_threshold = q75 + 2.5 * iqr
            outliers = int_daily[int_daily > outlier_threshold]
            stats_text += f"  Outliers (>Q3+2.5*IQR): {len(outliers)}\n"
            if not outliers.empty:
                stats_text += f"  Max outlier: {outliers.max():.0f}\n"
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                 verticalalignment='top', fontsize=11, fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(CFG["output_dir"]) / "call_sources_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"║ Call sources plot saved: {output_path}")
        
    except Exception as e:
        print(f"║ ERROR: Error creating call sources plot: {e}")

# ============================================================================
# OUTLIER DETECTION AND REMOVAL
# ============================================================================

def detect_and_remove_outliers(calls_combined):
    """Detect and remove outliers from combined call data"""
    
    print_ascii_section("STEP 2: OUTLIER DETECTION & REMOVAL")
    
    try:
        original_data = calls_combined.copy()
        
        original_stats = {
            "Original Days": len(original_data),
            "Original Range": f"{original_data.min():.0f} to {original_data.max():.0f}",
            "Original Mean": f"{original_data.mean():.0f}",
            "Original Std": f"{original_data.std():.0f}"
        }
        print_ascii_stats("ORIGINAL DATA STATISTICS", original_stats)
        
        # Method 1: IQR-based outlier detection
        if CFG["outlier_method"] == "iqr":
            q75 = calls_combined.quantile(0.75)
            q25 = calls_combined.quantile(0.25)
            iqr = q75 - q25
            
            lower_bound = q25 - CFG["iqr_multiplier"] * iqr
            upper_bound = q75 + CFG["iqr_multiplier"] * iqr
            
            outlier_mask = (calls_combined < lower_bound) | (calls_combined > upper_bound)
            
            iqr_stats = {
                "Method": "IQR",
                "Multiplier": f"{CFG['iqr_multiplier']}",
                "Q25": f"{q25:.0f}",
                "Q75": f"{q75:.0f}",
                "IQR": f"{iqr:.0f}",
                "Lower Bound": f"{lower_bound:.0f}",
                "Upper Bound": f"{upper_bound:.0f}"
            }
            print_ascii_stats("OUTLIER DETECTION METHOD", iqr_stats)
            
        # Method 2: Z-score based
        elif CFG["outlier_method"] == "zscore":
            z_scores = np.abs((calls_combined - calls_combined.mean()) / calls_combined.std())
            outlier_mask = z_scores > CFG["zscore_threshold"]
            
            zscore_stats = {
                "Method": "Z-Score",
                "Threshold": f"{CFG['zscore_threshold']}",
                "Mean": f"{calls_combined.mean():.0f}",
                "Std": f"{calls_combined.std():.0f}"
            }
            print_ascii_stats("OUTLIER DETECTION METHOD", zscore_stats)
            
        # Method 3: Percentile based
        elif CFG["outlier_method"] == "percentile":
            lower_bound = calls_combined.quantile(CFG["percentile_lower"] / 100)
            upper_bound = calls_combined.quantile(CFG["percentile_upper"] / 100)
            
            outlier_mask = (calls_combined < lower_bound) | (calls_combined > upper_bound)
            
            percentile_stats = {
                "Method": "Percentile",
                "Lower Percentile": f"{CFG['percentile_lower']}%",
                "Upper Percentile": f"{CFG['percentile_upper']}%",
                "Lower Bound": f"{lower_bound:.0f}",
                "Upper Bound": f"{upper_bound:.0f}"
            }
            print_ascii_stats("OUTLIER DETECTION METHOD", percentile_stats)
        
        outliers = calls_combined[outlier_mask]
        cleaned_data = calls_combined[~outlier_mask]
        
        # Print outlier details in ASCII table
        if not outliers.empty:
            print("\n┌─ OUTLIERS DETECTED " + "─" * 54 + "┐")
            
            outlier_rows = []
            for date, value in outliers.items():
                weekday = date.strftime('%A')
                outlier_rows.append([
                    date.strftime('%Y-%m-%d'),
                    weekday,
                    f"{value:,.0f}",
                    f"{value - calls_combined.median():+,.0f}"
                ])
            
            print_ascii_table(
                ["Date", "Weekday", "Calls", "vs Median"],
                outlier_rows
            )
            
            # Count outliers by weekday
            weekday_counts = outliers.groupby(outliers.index.strftime('%A')).size()
            if len(weekday_counts) > 0:
                print("\n┌─ OUTLIERS BY WEEKDAY " + "─" * 47 + "┐")
                weekday_rows = []
                for day, count in weekday_counts.items():
                    pct = count / len(outliers) * 100
                    weekday_rows.append([day, f"{count}", f"{pct:.1f}%"])
                
                print_ascii_table(
                    ["Weekday", "Count", "Percentage"],
                    weekday_rows
                )
        
        cleaned_stats = {
            "Outliers Detected": len(outliers),
            "Outliers Percentage": f"{len(outliers)/len(original_data)*100:.1f}%",
            "Cleaned Days": len(cleaned_data),
            "Cleaned Range": f"{cleaned_data.min():.0f} to {cleaned_data.max():.0f}",
            "Cleaned Mean": f"{cleaned_data.mean():.0f}",
            "Cleaned Std": f"{cleaned_data.std():.0f}"
        }
        print_ascii_stats("CLEANED DATA STATISTICS", cleaned_stats)
        
        # Create before/after visualization
        create_outlier_comparison_plot(original_data, cleaned_data, outliers)
        
        return {
            'original': original_data,
            'cleaned': cleaned_data,
            'outliers': outliers,
            'method': CFG["outlier_method"]
        }
        
    except Exception as e:
        print(f"║ ERROR: Error in outlier detection: {e}")
        return None

def create_outlier_comparison_plot(original, cleaned, outliers):
    """Create before/after outlier removal comparison"""
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle('Outlier Detection and Removal Analysis', fontsize=16, fontweight='bold')
        
        # 1. Time series comparison
        ax1.plot(original.index, original.values, 'b-', alpha=0.5, linewidth=1, label='Original Data')
        ax1.plot(cleaned.index, cleaned.values, 'g-', alpha=0.8, linewidth=1, label='Cleaned Data')
        
        if not outliers.empty:
            ax1.scatter(outliers.index, outliers.values, color='red', s=50, marker='x', 
                       label=f'Outliers ({len(outliers)})', zorder=5)
        
        ax1.set_title('Time Series: Before vs After Cleaning')
        ax1.set_ylabel('Call Volume')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribution comparison
        ax2.hist(original.values, bins=50, alpha=0.5, color='blue', label='Original', density=True)
        ax2.hist(cleaned.values, bins=50, alpha=0.7, color='green', label='Cleaned', density=True)
        
        ax2.set_title('Distribution Comparison')
        ax2.set_xlabel('Call Volume')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.set_yscale('log')
        
        # 3. Box plot comparison
        box_data = [original.values, cleaned.values]
        box_labels = ['Original', 'Cleaned']
        
        ax3.boxplot(box_data, labels=box_labels)
        ax3.set_title('Box Plot Comparison')
        ax3.set_ylabel('Call Volume')
        
        # 4. Statistics summary
        ax4.axis('off')
        
        stats_text = f"OUTLIER REMOVAL SUMMARY\n\n"
        stats_text += f"METHOD: {CFG['outlier_method'].upper()}\n\n"
        
        stats_text += f"ORIGINAL DATA:\n"
        stats_text += f"  Days: {len(original)}\n"
        stats_text += f"  Range: {original.min():.0f} - {original.max():.0f}\n"
        stats_text += f"  Mean: {original.mean():.0f}\n"
        stats_text += f"  Std: {original.std():.0f}\n\n"
        
        stats_text += f"CLEANED DATA:\n"
        stats_text += f"  Days: {len(cleaned)}\n"
        stats_text += f"  Range: {cleaned.min():.0f} - {cleaned.max():.0f}\n"
        stats_text += f"  Mean: {cleaned.mean():.0f}\n"
        stats_text += f"  Std: {cleaned.std():.0f}\n\n"
        
        stats_text += f"OUTLIERS REMOVED:\n"
        stats_text += f"  Count: {len(outliers)}\n"
        stats_text += f"  Percentage: {len(outliers)/len(original)*100:.1f}%\n"
        
        if not outliers.empty:
            weekday_counts = outliers.groupby(outliers.index.strftime('%A')).size()
            stats_text += f"  By weekday:\n"
            for day, count in weekday_counts.items():
                stats_text += f"    {day}: {count}\n"
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                 verticalalignment='top', fontsize=11, fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(CFG["output_dir"]) / "outlier_removal_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"║ Outlier analysis plot saved: {output_path}")
        
    except Exception as e:
        print(f"║ ERROR: Error creating outlier comparison plot: {e}")

# ============================================================================
# CLEAN DATA MODEL TRAINING
# ============================================================================

def retrain_models_with_clean_data(clean_calls_data):
    """Retrain models using cleaned call data"""
    
    print_ascii_section("STEP 3: RETRAINING MODELS WITH CLEAN DATA")
    
    try:
        # Load mail data (same as before)
        mail_path = _find_file(["mail.csv", "data/mail.csv"])
        mail = pd.read_csv(mail_path)
        mail.columns = [c.lower().strip() for c in mail.columns]
        mail["mail_date"] = _to_date(mail["mail_date"])
        mail = mail.dropna(subset=["mail_date"])
        
        # Aggregate mail daily
        mail_daily = (mail.groupby(["mail_date", "mail_type"], as_index=False)["mail_volume"].sum()
                       .pivot(index="mail_date", columns="mail_type", values="mail_volume").fillna(0))
        
        mail_daily.index = pd.to_datetime(mail_daily.index)
        
        # Business days only
        us_holidays = holidays.US()
        biz_mask = (~mail_daily.index.weekday.isin([5, 6])) & (~mail_daily.index.isin(us_holidays))
        mail_daily = mail_daily.loc[biz_mask]
        
        # Combine with CLEAN call data
        clean_calls_data.index = pd.to_datetime(clean_calls_data.index)
        clean_daily = mail_daily.join(clean_calls_data.rename("calls_total"), how="inner")
        
        combined_stats = {
            "Combined Data Shape": f"{clean_daily.shape[0]} x {clean_daily.shape[1]}",
            "Clean Call Range": f"{clean_daily['calls_total'].min():.0f} to {clean_daily['calls_total'].max():.0f}",
            "Clean Call Mean": f"{clean_daily['calls_total'].mean():.0f}",
            "Clean Call Std": f"{clean_daily['calls_total'].std():.0f}"
        }
        print_ascii_stats("CLEAN COMBINED DATA", combined_stats)
        
        # Create baseline features
        print("║ Creating baseline features with clean data...")
        X_baseline, y_baseline = create_baseline_features(clean_daily)
        
        # Create enhanced features
        print("║ Creating enhanced features with clean data...")
        X_enhanced, y_enhanced = create_enhanced_features(clean_daily)
        
        # Train baseline model
        print("║ Training baseline model with clean data...")
        baseline_models = train_quantile_models(X_baseline, y_baseline, "CLEAN BASELINE")
        
        # Train enhanced model
        print("║ Training enhanced model with clean data...")
        enhanced_models = train_quantile_models(X_enhanced, y_enhanced, "CLEAN ENHANCED")
        
        # Test both models
        baseline_results = test_model_performance(X_baseline, y_baseline, baseline_models, "CLEAN BASELINE")
        enhanced_results = test_model_performance(X_enhanced, y_enhanced, enhanced_models, "CLEAN ENHANCED")
        
        # Test hybrid approach
        hybrid_results = test_hybrid_approach(X_baseline, X_enhanced, y_baseline, baseline_models, enhanced_models)
        
        # Create comparison visualization and ASCII results
        create_clean_model_comparison(baseline_results, enhanced_results, hybrid_results)
        display_clean_model_results(baseline_results, enhanced_results, hybrid_results)
        
        return {
            'baseline_results': baseline_results,
            'enhanced_results': enhanced_results,
            'hybrid_results': hybrid_results,
            'clean_data': clean_daily
        }
        
    except Exception as e:
        print(f"║ ERROR: Error retraining models with clean data: {e}")
        print(f"║ {traceback.format_exc()}")
        return None

def create_baseline_features(daily):
    """Create baseline features (same as original)"""
    
    features_list = []
    targets_list = []
    
    for i in range(len(daily) - 1):
        try:
            current_day = daily.iloc[i]
            next_day = daily.iloc[i + 1]
            
            feature_row = {}
            
            # Mail volumes
            available_types = [t for t in CFG["top_mail_types"] if t in daily.columns]
            
            for mail_type in available_types:
                volume = current_day.get(mail_type, 0)
                feature_row[f"{mail_type}_volume"] = max(0, float(volume)) if not pd.isna(volume) else 0
            
            # Total mail volume
            total_mail = sum(feature_row.get(f"{t}_volume", 0) for t in available_types)
            feature_row["total_mail_volume"] = total_mail
            feature_row["log_total_mail_volume"] = np.log1p(total_mail)
            
            # Mail percentiles
            mail_history = daily[available_types].sum(axis=1).iloc[:i+1]
            if len(mail_history) > 10:
                feature_row["mail_percentile"] = (mail_history <= total_mail).mean()
            else:
                feature_row["mail_percentile"] = 0.5
            
            # Date features
            current_date = daily.index[i]
            feature_row["weekday"] = current_date.weekday()
            feature_row["month"] = current_date.month
            feature_row["is_month_end"] = 1 if current_date.day > 25 else 0
            
            try:
                feature_row["is_holiday_week"] = 1 if current_date.date() in holidays.US() else 0
            except:
                feature_row["is_holiday_week"] = 0
            
            # Recent call volume context
            recent_calls = daily["calls_total"].iloc[max(0, i-5):i+1]
            feature_row["recent_calls_avg"] = recent_calls.mean() if not recent_calls.empty else 15000
            feature_row["recent_calls_trend"] = recent_calls.diff().mean() if len(recent_calls) > 1 else 0
            
            # Target: next day's calls
            target = next_day["calls_total"]
            if pd.isna(target) or target <= 0:
                continue
            
            features_list.append(feature_row)
            targets_list.append(float(target))
            
        except Exception as e:
            print(f"║ Warning: Error processing baseline day {i}: {e}")
            continue
    
    X = pd.DataFrame(features_list)
    y = pd.Series(targets_list)
    
    X = X.fillna(0)
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], 0)
    
    print(f"║ Clean baseline features: {X.shape[0]} samples x {X.shape[1]} features")
    return X, y

def create_enhanced_features(daily):
    """Create enhanced features with Friday features"""
    
    features_list = []
    targets_list = []
    
    for i in range(len(daily) - 1):
        try:
            current_day = daily.iloc[i]
            next_day = daily.iloc[i + 1]
            
            feature_row = {}
            
            # BASELINE FEATURES (same as above)
            available_types = [t for t in CFG["top_mail_types"] if t in daily.columns]
            
            for mail_type in available_types:
                volume = current_day.get(mail_type, 0)
                feature_row[f"{mail_type}_volume"] = max(0, float(volume)) if not pd.isna(volume) else 0
            
            total_mail = sum(feature_row.get(f"{t}_volume", 0) for t in available_types)
            feature_row["total_mail_volume"] = total_mail
            feature_row["log_total_mail_volume"] = np.log1p(total_mail)
            
            mail_history = daily[available_types].sum(axis=1).iloc[:i+1]
            if len(mail_history) > 10:
                feature_row["mail_percentile"] = (mail_history <= total_mail).mean()
            else:
                feature_row["mail_percentile"] = 0.5
            
            current_date = daily.index[i]
            feature_row["weekday"] = current_date.weekday()
            feature_row["month"] = current_date.month
            feature_row["is_month_end"] = 1 if current_date.day > 25 else 0
            
            try:
                feature_row["is_holiday_week"] = 1 if current_date.date() in holidays.US() else 0
            except:
                feature_row["is_holiday_week"] = 0
            
            recent_calls = daily["calls_total"].iloc[max(0, i-5):i+1]
            feature_row["recent_calls_avg"] = recent_calls.mean() if not recent_calls.empty else 15000
            feature_row["recent_calls_trend"] = recent_calls.diff().mean() if len(recent_calls) > 1 else 0
            
            # FRIDAY ENHANCED FEATURES
            is_friday = current_date.weekday() == 4
            feature_row["is_friday"] = 1 if is_friday else 0
            
            if is_friday:
                # Scaled polynomial features
                if total_mail > 0:
                    feature_row["friday_mail_squared"] = (total_mail / 1000) ** 2
                    feature_row["friday_mail_sqrt"] = np.sqrt(total_mail)
                    feature_row["friday_mail_cubed"] = (total_mail / 10000) ** 3
                    feature_row["friday_log_mail_squared"] = (np.log1p(total_mail)) ** 2
                else:
                    feature_row["friday_mail_squared"] = 0
                    feature_row["friday_mail_sqrt"] = 0
                    feature_row["friday_mail_cubed"] = 0
                    feature_row["friday_log_mail_squared"] = 0
                
                # Friday interactions
                feature_row["friday_total_mail"] = total_mail
                feature_row["friday_log_mail"] = feature_row["log_total_mail_volume"]
                feature_row["friday_recent_calls"] = feature_row["recent_calls_avg"] / 10000
            else:
                # Zero all Friday features for non-Fridays
                friday_feature_names = [
                    "friday_mail_squared", "friday_mail_sqrt", "friday_mail_cubed",
                    "friday_log_mail_squared", "friday_total_mail", "friday_log_mail",
                    "friday_recent_calls"
                ]
                for fname in friday_feature_names:
                    feature_row[fname] = 0
            
            # Target
            target = next_day["calls_total"]
            if pd.isna(target) or target <= 0:
                continue
            
            features_list.append(feature_row)
            targets_list.append(float(target))
            
        except Exception as e:
            print(f"║ Warning: Error processing enhanced day {i}: {e}")
            continue
    
    X = pd.DataFrame(features_list)
    y = pd.Series(targets_list)
    
    X = X.fillna(0)
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], 0)
    
    # Scale down large polynomial features
    for col in X.columns:
        if 'squared' in col or 'cubed' in col:
            if X[col].max() > 1e10:
                X[col] = X[col] / 1e6
    
    print(f"║ Clean enhanced features: {X.shape[0]} samples x {X.shape[1]} features")
    return X, y

def train_quantile_models(X, y, model_type):
    """Train quantile regression models"""
    
    split_point = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    models = {}
    
    solvers_to_try = ['highs-ds', 'highs-ipm', 'highs', 'interior-point']
    alpha_values = [0.01, 0.1, 1.0]
    
    for quantile in CFG["quantiles"]:
        model_trained = False
        
        for solver in solvers_to_try:
            for alpha in alpha_values:
                try:
                    model = QuantileRegressor(quantile=quantile, alpha=alpha, solver=solver)
                    model.fit(X_train, y_train)
                    
                    y_pred = model.predict(X_test)
                    if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                        raise ValueError("Invalid predictions")
                    
                    mae = mean_absolute_error(y_test, y_pred)
                    models[f"quantile_{quantile}"] = model
                    print(f"║   {model_type} {int(quantile*100)}% quantile: MAE={mae:.0f}")
                    model_trained = True
                    break
                    
                except Exception:
                    continue
            
            if model_trained:
                break
        
        if not model_trained:
            # Fallback to linear regression
            try:
                fallback_model = LinearRegression()
                fallback_model.fit(X_train, y_train)
                y_pred = fallback_model.predict(X_test)
                mae = mean_absolute_error(y_test, y_pred)
                models[f"quantile_{quantile}"] = fallback_model
                print(f"║   {model_type} {int(quantile*100)}% fallback: MAE={mae:.0f}")
            except Exception as e:
                print(f"║   ERROR: Even fallback failed for {quantile}: {e}")
    
    return models

def test_model_performance(X, y, models, model_name):
    """Test model performance comprehensively"""
    
    split_point = int(len(X) * 0.8)
    X_test = X.iloc[split_point:]
    y_test = y.iloc[split_point:]
    
    main_model = models.get("quantile_0.5")
    if not main_model:
        print(f"║ ERROR: No main model for {model_name}")
        return {}
    
    try:
        y_pred = main_model.predict(X_test)
        
        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            y_pred = np.full_like(y_pred, y_test.mean())
    except Exception as e:
        print(f"║ ERROR: Prediction failed for {model_name}: {e}")
        y_pred = np.full(len(y_test), y_test.mean())
    
    # Overall metrics
    overall_metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'accuracy': max(0, 100 - (mean_absolute_error(y_test, y_pred) / y_test.mean() * 100))
    }
    
    # Weekday-specific metrics
    weekday_metrics = {}
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    if 'weekday' in X_test.columns:
        for day_num, day_name in enumerate(weekdays):
            day_mask = X_test['weekday'] == day_num
            if day_mask.sum() > 0:
                day_true = y_test[day_mask]
                day_pred = y_pred[day_mask]
                
                day_mae = mean_absolute_error(day_true, day_pred)
                weekday_metrics[day_name] = day_mae
    
    return {
        'overall': overall_metrics,
        'weekday': weekday_metrics
    }

def test_hybrid_approach(X_baseline, X_enhanced, y, baseline_models, enhanced_models):
    """Test hybrid approach with clean data"""
    
    print("║ Testing CLEAN HYBRID approach...")
    
    try:
        split_point = int(len(X_enhanced) * 0.8)
        X_test_enhanced = X_enhanced.iloc[split_point:]
        y_test = y.iloc[split_point:]
        
        predictions = np.zeros(len(X_test_enhanced))
        
        if 'weekday' in X_test_enhanced.columns:
            # Enhanced for Mon-Thu
            non_friday_mask = X_test_enhanced['weekday'] != 4
            friday_mask = X_test_enhanced['weekday'] == 4
            
            if non_friday_mask.sum() > 0:
                predictions[non_friday_mask] = enhanced_models["quantile_0.5"].predict(X_test_enhanced[non_friday_mask])
            
            if friday_mask.sum() > 0:
                # For Fridays, use baseline model with baseline features
                baseline_feature_names = list(X_baseline.columns)
                X_friday_baseline = X_test_enhanced[friday_mask][baseline_feature_names]
                predictions[friday_mask] = baseline_models["quantile_0.5"].predict(X_friday_baseline)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        
        # Weekday breakdown
        weekday_metrics = {}
        if 'weekday' in X_test_enhanced.columns:
            weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            for day_num, day_name in enumerate(weekdays):
                day_mask = X_test_enhanced['weekday'] == day_num
                if day_mask.sum() > 0:
                    day_mae = mean_absolute_error(y_test[day_mask], predictions[day_mask])
                    weekday_metrics[day_name] = day_mae
        
        hybrid_results = {
            'overall': {
                'mae': mae,
                'rmse': rmse, 
                'r2': r2,
                'accuracy': max(0, 100 - (mae / y_test.mean() * 100))
            },
            'weekday': weekday_metrics
        }
        
        return hybrid_results
        
    except Exception as e:
        print(f"║ ERROR: Error in hybrid testing: {e}")
        return {}

def display_clean_model_results(baseline_results, enhanced_results, hybrid_results):
    """Display clean model results in ASCII format"""
    
    print_ascii_section("CLEAN MODEL PERFORMANCE RESULTS")
    
    # Overall performance table
    performance_rows = []
    
    models = ['Baseline', 'Enhanced', 'Hybrid']
    results = [baseline_results, enhanced_results, hybrid_results]
    
    for model, result in zip(models, results):
        if result and 'overall' in result:
            overall = result['overall']
            performance_rows.append([
                model,
                f"{overall['mae']:.0f}",
                f"{overall['rmse']:.0f}",
                f"{overall['r2']:.3f}",
                f"{overall['accuracy']:.1f}%"
            ])
    
    print_ascii_table(
        ["Model", "MAE", "RMSE", "R²", "Accuracy"],
        performance_rows,
        "OVERALL PERFORMANCE (CLEAN DATA)"
    )
    
    # Find best model
    if performance_rows:
        maes = [float(row[1]) for row in performance_rows]
        best_idx = maes.index(min(maes))
        best_model = models[best_idx]
        best_mae = min(maes)
        
        print(f"\n╔═══════════════════════════════════════════════════════════════════════════╗")
        print(f"║                          BEST MODEL: {best_model.upper():^20}                    ║")
        print(f"║                          BEST MAE: {best_mae:^6.0f}                            ║")
        print(f"╚═══════════════════════════════════════════════════════════════════════════╝")
    
    # Weekday performance comparison
    if all(result and 'weekday' in result for result in results):
        weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        weekday_rows = []
        
        for day in weekdays:
            row = [day]
            for result in results:
                mae = result['weekday'].get(day, 0)
                row.append(f"{mae:.0f}")
            weekday_rows.append(row)
        
        print_ascii_table(
            ["Weekday", "Baseline", "Enhanced", "Hybrid"],
            weekday_rows,
            "WEEKDAY PERFORMANCE (CLEAN DATA)"
        )
    
    # Improvement analysis
    if performance_rows and len(performance_rows) >= 2:
        baseline_mae = float(performance_rows[0][1])  # Baseline MAE
        
        improvements = {}
        for i, model in enumerate(models[1:], 1):  # Skip baseline
            if i < len(performance_rows):
                model_mae = float(performance_rows[i][1])
                improvement = (baseline_mae - model_mae) / baseline_mae * 100
                improvements[model] = improvement
        
        improvement_rows = []
        for model, improvement in improvements.items():
            improvement_rows.append([
                f"{model} vs Baseline",
                f"{improvement:+.1f}%",
                "Better" if improvement > 0 else "Worse"
            ])
        
        print_ascii_table(
            ["Comparison", "Improvement", "Status"],
            improvement_rows,
            "MODEL IMPROVEMENTS"
        )

def create_clean_model_comparison(baseline_results, enhanced_results, hybrid_results):
    """Create comparison visualization for clean models"""
    
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Clean Data: Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. Overall MAE comparison
        models = ['Baseline', 'Enhanced', 'Hybrid']
        maes = [
            baseline_results['overall']['mae'],
            enhanced_results['overall']['mae'],
            hybrid_results['overall']['mae']
        ]
        
        colors = ['skyblue', 'lightcoral', 'lightgreen']
        bars = ax1.bar(models, maes, color=colors, alpha=0.7)
        ax1.set_ylabel('MAE')
        ax1.set_title('Overall Model Performance (Clean Data)')
        
        # Add value labels
        for bar, mae in zip(bars, maes):
            height = bar.get_height()
            ax1.annotate(f'{mae:.0f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')
        
        # Highlight best
        best_idx = maes.index(min(maes))
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(2)
        
        # 2. Weekday performance
        if 'weekday' in baseline_results and 'weekday' in enhanced_results and 'weekday' in hybrid_results:
            weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            
            x = np.arange(len(weekdays))
            width = 0.25
            
            baseline_weekday = [baseline_results['weekday'].get(day, 0) for day in weekdays]
            enhanced_weekday = [enhanced_results['weekday'].get(day, 0) for day in weekdays]
            hybrid_weekday = [hybrid_results['weekday'].get(day, 0) for day in weekdays]
            
            ax2.bar(x - width, baseline_weekday, width, label='Baseline', color='skyblue', alpha=0.7)
            ax2.bar(x, enhanced_weekday, width, label='Enhanced', color='lightcoral', alpha=0.7)
            ax2.bar(x + width, hybrid_weekday, width, label='Hybrid', color='lightgreen', alpha=0.7)
            
            ax2.set_xlabel('Weekday')
            ax2.set_ylabel('MAE')
            ax2.set_title('Weekday Performance (Clean Data)')
            ax2.set_xticks(x)
            ax2.set_xticklabels(weekdays)
            ax2.legend()
        
        # 3. Improvement analysis
        baseline_mae = baseline_results['overall']['mae']
        enhanced_improvement = (baseline_mae - enhanced_results['overall']['mae']) / baseline_mae * 100
        hybrid_improvement = (baseline_mae - hybrid_results['overall']['mae']) / baseline_mae * 100
        
        improvements = ['Enhanced vs Baseline', 'Hybrid vs Baseline']
        improvement_values = [enhanced_improvement, hybrid_improvement]
        
        colors = ['lightcoral', 'lightgreen']
        bars = ax3.bar(improvements, improvement_values, color=colors, alpha=0.7)
        ax3.set_ylabel('Improvement (%)')
        ax3.set_title('Model Improvements vs Baseline')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar, improvement in zip(bars, improvement_values):
            height = bar.get_height()
            ax3.annotate(f'{improvement:+.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        # 4. Summary
        ax4.axis('off')
        
        best_model = models[best_idx]
        best_mae = min(maes)
        
        summary_text = f"""
CLEAN DATA MODEL COMPARISON

RESULTS SUMMARY:
  Best Model: {best_model}
  Best MAE: {best_mae:.0f}
  Enhanced Improvement: {enhanced_improvement:+.1f}%
  Hybrid Improvement: {hybrid_improvement:+.1f}%

MODEL PERFORMANCE:
  Baseline: {baseline_results['overall']['mae']:.0f} MAE
  Enhanced: {enhanced_results['overall']['mae']:.0f} MAE  
  Hybrid: {hybrid_results['overall']['mae']:.0f} MAE

DATA QUALITY IMPACT:
  Outliers removed before training
  Clean data ensures reliable results
  Model comparisons now accurate

RECOMMENDATION:
  Deploy {best_model.upper()} model
  {"Significant improvement achieved" if max(improvement_values) > 5 else "Marginal improvement detected"}
  Clean data pipeline critical
        """
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                 verticalalignment='top', fontsize=11, fontfamily='monospace',
                 bbox=dict(boxstyle="round,pad=0.5", 
                          facecolor='lightgreen' if max(improvement_values) > 5 else 'lightyellow', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(CFG["output_dir"]) / "clean_model_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"║ Clean model comparison saved: {output_path}")
        
    except Exception as e:
        print(f"║ ERROR: Error creating clean model comparison: {e}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    
    print_ascii_header()
    
    print("┌─ ANALYSIS STEPS " + "─" * 57 + "┐")
    print("│ 1. Visualize both call data sources (overlay)           │")
    print("│ 2. Detect and remove outliers                           │")  
    print("│ 3. Retrain models with clean data                       │")
    print("│ 4. Compare baseline vs enhanced vs hybrid               │")
    print("└" + "─" * 70 + "┘")
    
    try:
        start_time = datetime.now()
        
        # Step 1: Load and visualize call sources
        call_data = load_and_visualize_call_sources()
        if not call_data:
            print("║ ERROR: Failed to load call data")
            return 1
        
        # Step 2: Detect and remove outliers
        outlier_results = detect_and_remove_outliers(call_data['calls_combined'])
        if not outlier_results:
            print("║ ERROR: Failed to detect outliers")
            return 1
        
        # Step 3: Retrain models with clean data
        clean_model_results = retrain_models_with_clean_data(outlier_results['cleaned'])
        if not clean_model_results:
            print("║ ERROR: Failed to retrain models")
            return 1
        
        # Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print_ascii_section("ANALYSIS COMPLETE")
        
        completion_stats = {
            "Total Time": f"{duration:.1f} seconds",
            "Results Directory": CFG['output_dir'],
            "Status": "SUCCESS"
        }
        print_ascii_stats("COMPLETION SUMMARY", completion_stats)
        
        print("\n╔═══════════════════════════════════════════════════════════════════════════╗")
        print("║                            ANALYSIS COMPLETE!                            ║")
        print("╠═══════════════════════════════════════════════════════════════════════════╣")
        print("║  ✓ Call data sources visualized                                          ║")
        print("║  ✓ Outliers detected and removed                                         ║")
        print("║  ✓ Models retrained with clean data                                      ║")
        print("║  ✓ Performance comparison generated                                       ║")
        print("╠═══════════════════════════════════════════════════════════════════════════╣")
        print(f"║  📁 All results in: {CFG['output_dir']:^42} ║")
        print("╚═══════════════════════════════════════════════════════════════════════════╝")
        
        return 0
        
    except Exception as e:
        print(f"║ ERROR: Critical error: {e}")
        print(f"║ {traceback.format_exc()}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
