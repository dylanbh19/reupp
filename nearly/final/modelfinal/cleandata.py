#!/usr/bin/env python
"""
MAIL-TO-CALLS PREDICTION: STEP BY STEP APPROACH
===============================================

STEP 1: DATA LOADING, CLEANING, AND ALIGNMENT
- Load call intent data and aggregate to daily volumes
- Load mail data 
- Filter to common date ranges
- Remove outliers
- Create clean aligned dataset

SIMPLE, FOCUSED, NO COMPLEX FEATURES YET
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import sys

# Set up plotting
plt.style.use('default')
sns.set_palette("husl")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    "call_file": "callintent.csv",
    "mail_file": "mail.csv",
    "output_dir": "cleaned_data",
    "plot_outliers": True,
    "outlier_method": "iqr",  # 'iqr' or 'zscore'
    "outlier_threshold": 3.0,
    "min_overlap_days": 30
}

def safe_print(msg):
    """Print safely"""
    try:
        print(str(msg).encode('ascii', 'ignore').decode('ascii'))
    except:
        print(str(msg))

# ============================================================================
# STEP 1: DATA LOADING
# ============================================================================

def load_call_data():
    """Load call intent data and create daily call volumes"""
    safe_print("=" * 60)
    safe_print("STEP 1A: LOADING CALL INTENT DATA")
    safe_print("=" * 60)
    
    # Find and load call file
    call_paths = [CONFIG["call_file"], f"data/{CONFIG['call_file']}"]
    call_path = None
    
    for path in call_paths:
        if Path(path).exists():
            call_path = path
            break
    
    if not call_path:
        raise FileNotFoundError(f"Call file not found: {call_paths}")
    
    safe_print(f"Loading: {call_path}")
    
    # Load with error handling
    try:
        df = pd.read_csv(call_path, encoding='utf-8', low_memory=False)
    except:
        df = pd.read_csv(call_path, encoding='latin1', low_memory=False)
    
    safe_print(f"Raw call data: {len(df):,} rows, {len(df.columns)} columns")
    
    # Clean column names
    df.columns = [str(col).lower().strip() for col in df.columns]
    
    # Find date column
    date_col = None
    for col in df.columns:
        if any(keyword in col for keyword in ['date', 'start', 'time']):
            date_col = col
            break
    
    if not date_col:
        raise ValueError("No date column found in call data")
    
    safe_print(f"Using date column: {date_col}")
    
    # Process dates
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    # Filter to 2025+ data
    df = df[df[date_col].dt.year >= 2025]
    safe_print(f"After date filtering: {len(df):,} rows")
    
    if len(df) == 0:
        raise ValueError("No data found for 2025+")
    
    # Create daily call volumes
    df['call_date'] = df[date_col].dt.date
    daily_calls = df.groupby('call_date').size().reset_index()
    daily_calls.columns = ['date', 'call_volume']
    daily_calls['date'] = pd.to_datetime(daily_calls['date'])
    
    # Filter to business days only
    daily_calls = daily_calls[daily_calls['date'].dt.weekday < 5]
    daily_calls = daily_calls.sort_values('date').reset_index(drop=True)
    
    safe_print(f"Daily call data: {len(daily_calls)} business days")
    safe_print(f"Date range: {daily_calls['date'].min().date()} to {daily_calls['date'].max().date()}")
    safe_print(f"Call volume stats:")
    safe_print(f"  Mean: {daily_calls['call_volume'].mean():.0f}")
    safe_print(f"  Std:  {daily_calls['call_volume'].std():.0f}")
    safe_print(f"  Min:  {daily_calls['call_volume'].min()}")
    safe_print(f"  Max:  {daily_calls['call_volume'].max()}")
    
    return daily_calls

def load_mail_data():
    """Load mail data"""
    safe_print("\n" + "=" * 60)
    safe_print("STEP 1B: LOADING MAIL DATA")
    safe_print("=" * 60)
    
    # Find and load mail file
    mail_paths = [CONFIG["mail_file"], f"data/{CONFIG['mail_file']}"]
    mail_path = None
    
    for path in mail_paths:
        if Path(path).exists():
            mail_path = path
            break
    
    if not mail_path:
        raise FileNotFoundError(f"Mail file not found: {mail_paths}")
    
    safe_print(f"Loading: {mail_path}")
    
    # Load with error handling
    try:
        df = pd.read_csv(mail_path, encoding='utf-8', low_memory=False)
    except:
        df = pd.read_csv(mail_path, encoding='latin1', low_memory=False)
    
    safe_print(f"Raw mail data: {len(df):,} rows, {len(df.columns)} columns")
    
    # Clean column names
    df.columns = [str(col).lower().strip() for col in df.columns]
    
    # Find required columns
    date_col = volume_col = type_col = None
    
    for col in df.columns:
        if 'date' in col and date_col is None:
            date_col = col
        elif 'volume' in col and volume_col is None:
            volume_col = col
        elif 'type' in col and type_col is None:
            type_col = col
    
    if not all([date_col, volume_col, type_col]):
        raise ValueError(f"Required columns not found. Found: date={date_col}, volume={volume_col}, type={type_col}")
    
    safe_print(f"Using columns: date={date_col}, volume={volume_col}, type={type_col}")
    
    # Process data
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    
    # Filter to 2025+ data
    df = df[df[date_col].dt.year >= 2025]
    safe_print(f"After date filtering: {len(df):,} rows")
    
    # Clean volume data
    df[volume_col] = pd.to_numeric(df[volume_col], errors='coerce')
    df = df.dropna(subset=[volume_col])
    df = df[df[volume_col] > 0]  # Remove zero/negative volumes
    
    safe_print(f"After volume cleaning: {len(df):,} rows")
    
    # Create daily mail data
    df['mail_date'] = df[date_col].dt.date
    
    # Aggregate by date and type
    daily_mail = df.groupby(['mail_date', type_col])[volume_col].sum().reset_index()
    daily_mail.columns = ['date', 'mail_type', 'volume']
    daily_mail['date'] = pd.to_datetime(daily_mail['date'])
    
    # Filter to business days
    daily_mail = daily_mail[daily_mail['date'].dt.weekday < 5]
    
    # Pivot to get mail types as columns
    mail_pivot = daily_mail.pivot(index='date', columns='mail_type', values='volume').fillna(0)
    mail_pivot = mail_pivot.reset_index()
    
    safe_print(f"Daily mail data: {len(mail_pivot)} business days")
    safe_print(f"Date range: {mail_pivot['date'].min().date()} to {mail_pivot['date'].max().date()}")
    safe_print(f"Mail types: {len(mail_pivot.columns)-1}")
    
    # Show top mail types by volume
    mail_sums = mail_pivot.drop('date', axis=1).sum().sort_values(ascending=False)
    safe_print(f"Top 10 mail types by volume:")
    for i, (mail_type, volume) in enumerate(mail_sums.head(10).items()):
        safe_print(f"  {i+1:2d}. {mail_type}: {volume:,.0f}")
    
    return mail_pivot

# ============================================================================
# STEP 2: FILTER TO COMMON DATE RANGE
# ============================================================================

def filter_to_common_dates(daily_calls, mail_pivot):
    """Filter both datasets to common date range"""
    safe_print("\n" + "=" * 60)
    safe_print("STEP 2: FILTERING TO COMMON DATE RANGE")
    safe_print("=" * 60)
    
    # Find common dates
    call_dates = set(daily_calls['date'].dt.date)
    mail_dates = set(mail_pivot['date'].dt.date)
    common_dates = call_dates.intersection(mail_dates)
    
    safe_print(f"Call data dates: {len(call_dates)}")
    safe_print(f"Mail data dates: {len(mail_dates)}")
    safe_print(f"Common dates: {len(common_dates)}")
    
    if len(common_dates) < CONFIG["min_overlap_days"]:
        raise ValueError(f"Insufficient overlapping days: {len(common_dates)} < {CONFIG['min_overlap_days']}")
    
    # Filter to common dates
    common_dates_dt = [pd.to_datetime(d) for d in common_dates]
    
    calls_filtered = daily_calls[daily_calls['date'].isin(common_dates_dt)].copy()
    mail_filtered = mail_pivot[mail_pivot['date'].isin(common_dates_dt)].copy()
    
    # Sort by date
    calls_filtered = calls_filtered.sort_values('date').reset_index(drop=True)
    mail_filtered = mail_filtered.sort_values('date').reset_index(drop=True)
    
    safe_print(f"Filtered call data: {len(calls_filtered)} days")
    safe_print(f"Filtered mail data: {len(mail_filtered)} days")
    safe_print(f"Date range: {min(common_dates)} to {max(common_dates)}")
    
    return calls_filtered, mail_filtered, sorted(common_dates)

# ============================================================================
# STEP 3: OUTLIER DETECTION AND REMOVAL
# ============================================================================

def detect_outliers_iqr(data, column):
    """Detect outliers using IQR method"""
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (data[column] < lower_bound) | (data[column] > upper_bound)
    return outliers, lower_bound, upper_bound

def detect_outliers_zscore(data, column, threshold=3.0):
    """Detect outliers using Z-score method"""
    z_scores = np.abs((data[column] - data[column].mean()) / data[column].std())
    outliers = z_scores > threshold
    return outliers, data[column].mean() - threshold * data[column].std(), data[column].mean() + threshold * data[column].std()

def remove_outliers(calls_filtered, mail_filtered):
    """Remove outliers from both datasets"""
    safe_print("\n" + "=" * 60)
    safe_print("STEP 3: OUTLIER DETECTION AND REMOVAL")
    safe_print("=" * 60)
    
    original_len = len(calls_filtered)
    
    # Create output directory for plots
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    # Detect call volume outliers
    if CONFIG["outlier_method"] == "iqr":
        call_outliers, call_lower, call_upper = detect_outliers_iqr(calls_filtered, 'call_volume')
    else:
        call_outliers, call_lower, call_upper = detect_outliers_zscore(calls_filtered, 'call_volume', CONFIG["outlier_threshold"])
    
    safe_print(f"Call volume outliers detected: {call_outliers.sum()}/{len(calls_filtered)}")
    safe_print(f"Call volume bounds: {call_lower:.0f} to {call_upper:.0f}")
    
    if call_outliers.sum() > 0:
        safe_print("Outlier dates:")
        outlier_dates = calls_filtered[call_outliers]['date'].dt.date.tolist()
        outlier_volumes = calls_filtered[call_outliers]['call_volume'].tolist()
        for date, volume in zip(outlier_dates, outlier_volumes):
            safe_print(f"  {date}: {volume} calls")
    
    # Plot call volume outliers if requested
    if CONFIG["plot_outliers"]:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(calls_filtered['date'], calls_filtered['call_volume'], 'b-', alpha=0.7, label='Call Volume')
        if call_outliers.sum() > 0:
            plt.scatter(calls_filtered[call_outliers]['date'], calls_filtered[call_outliers]['call_volume'], 
                       color='red', s=50, label='Outliers', zorder=5)
        plt.axhline(y=call_lower, color='r', linestyle='--', alpha=0.5, label='Bounds')
        plt.axhline(y=call_upper, color='r', linestyle='--', alpha=0.5)
        plt.title('Call Volume with Outliers')
        plt.ylabel('Call Volume')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(calls_filtered['call_volume'])
        plt.title('Call Volume Box Plot')
        plt.ylabel('Call Volume')
        
        plt.tight_layout()
        plt.savefig(output_dir / "call_volume_outliers.png", dpi=150, bbox_inches='tight')
        plt.close()
        safe_print(f"Call outlier plot saved: {output_dir}/call_volume_outliers.png")
    
    # Detect mail volume outliers (total daily mail)
    mail_totals = mail_filtered.drop('date', axis=1).sum(axis=1)
    mail_data_for_outliers = pd.DataFrame({'date': mail_filtered['date'], 'total_mail': mail_totals})
    
    if CONFIG["outlier_method"] == "iqr":
        mail_outliers, mail_lower, mail_upper = detect_outliers_iqr(mail_data_for_outliers, 'total_mail')
    else:
        mail_outliers, mail_lower, mail_upper = detect_outliers_zscore(mail_data_for_outliers, 'total_mail', CONFIG["outlier_threshold"])
    
    safe_print(f"Total mail outliers detected: {mail_outliers.sum()}/{len(mail_filtered)}")
    safe_print(f"Total mail bounds: {mail_lower:.0f} to {mail_upper:.0f}")
    
    if mail_outliers.sum() > 0:
        safe_print("Mail outlier dates:")
        outlier_dates = mail_filtered[mail_outliers]['date'].dt.date.tolist()
        outlier_volumes = mail_totals[mail_outliers].tolist()
        for date, volume in zip(outlier_dates, outlier_volumes):
            safe_print(f"  {date}: {volume:.0f} total mail")
    
    # Plot mail outliers if requested
    if CONFIG["plot_outliers"]:
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.plot(mail_filtered['date'], mail_totals, 'g-', alpha=0.7, label='Total Mail Volume')
        if mail_outliers.sum() > 0:
            plt.scatter(mail_filtered[mail_outliers]['date'], mail_totals[mail_outliers], 
                       color='red', s=50, label='Outliers', zorder=5)
        plt.axhline(y=mail_lower, color='r', linestyle='--', alpha=0.5, label='Bounds')
        plt.axhline(y=mail_upper, color='r', linestyle='--', alpha=0.5)
        plt.title('Total Mail Volume with Outliers')
        plt.ylabel('Total Mail Volume')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(mail_totals)
        plt.title('Total Mail Volume Box Plot')
        plt.ylabel('Total Mail Volume')
        
        plt.tight_layout()
        plt.savefig(output_dir / "mail_volume_outliers.png", dpi=150, bbox_inches='tight')
        plt.close()
        safe_print(f"Mail outlier plot saved: {output_dir}/mail_volume_outliers.png")
    
    # Combine outliers (remove days that are outliers in either dataset)
    combined_outliers = call_outliers | mail_outliers
    
    safe_print(f"Combined outliers: {combined_outliers.sum()}/{len(calls_filtered)}")
    
    # Remove outliers
    calls_clean = calls_filtered[~combined_outliers].reset_index(drop=True)
    mail_clean = mail_filtered[~combined_outliers].reset_index(drop=True)
    
    safe_print(f"After outlier removal:")
    safe_print(f"  Original days: {original_len}")
    safe_print(f"  Clean days: {len(calls_clean)}")
    safe_print(f"  Removed: {original_len - len(calls_clean)}")
    
    # Show clean data stats
    safe_print(f"Clean call volume stats:")
    safe_print(f"  Mean: {calls_clean['call_volume'].mean():.0f}")
    safe_print(f"  Std:  {calls_clean['call_volume'].std():.0f}")
    safe_print(f"  Min:  {calls_clean['call_volume'].min()}")
    safe_print(f"  Max:  {calls_clean['call_volume'].max()}")
    
    clean_mail_totals = mail_clean.drop('date', axis=1).sum(axis=1)
    safe_print(f"Clean total mail stats:")
    safe_print(f"  Mean: {clean_mail_totals.mean():.0f}")
    safe_print(f"  Std:  {clean_mail_totals.std():.0f}")
    safe_print(f"  Min:  {clean_mail_totals.min():.0f}")
    safe_print(f"  Max:  {clean_mail_totals.max():.0f}")
    
    return calls_clean, mail_clean

# ============================================================================
# STEP 4: FINAL ALIGNMENT AND VALIDATION
# ============================================================================

def create_final_dataset(calls_clean, mail_clean):
    """Create final aligned dataset"""
    safe_print("\n" + "=" * 60)
    safe_print("STEP 4: CREATING FINAL ALIGNED DATASET")
    safe_print("=" * 60)
    
    # Merge on date
    final_data = pd.merge(calls_clean, mail_clean, on='date', how='inner')
    
    # Ensure date ordering
    final_data = final_data.sort_values('date').reset_index(drop=True)
    
    safe_print(f"Final dataset: {len(final_data)} days")
    safe_print(f"Date range: {final_data['date'].min().date()} to {final_data['date'].max().date()}")
    safe_print(f"Columns: {len(final_data.columns)} (1 date + 1 call_volume + {len(final_data.columns)-2} mail types)")
    
    # Show correlation between total mail and calls
    total_mail = final_data.drop(['date', 'call_volume'], axis=1).sum(axis=1)
    correlation = final_data['call_volume'].corr(total_mail)
    safe_print(f"Correlation (total mail vs calls): {correlation:.3f}")
    
    # Create summary plot
    output_dir = Path(CONFIG["output_dir"])
    
    plt.figure(figsize=(15, 10))
    
    # Time series plot
    plt.subplot(2, 3, 1)
    plt.plot(final_data['date'], final_data['call_volume'], 'b-', label='Calls', linewidth=2)
    plt.title('Daily Call Volume (Clean)')
    plt.ylabel('Calls')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 2)
    plt.plot(final_data['date'], total_mail, 'g-', label='Total Mail', linewidth=2)
    plt.title('Daily Total Mail Volume (Clean)')
    plt.ylabel('Mail Volume')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Scatter plot
    plt.subplot(2, 3, 3)
    plt.scatter(total_mail, final_data['call_volume'], alpha=0.6)
    plt.xlabel('Total Mail Volume')
    plt.ylabel('Call Volume')
    plt.title(f'Mail vs Calls (r={correlation:.3f})')
    plt.grid(True, alpha=0.3)
    
    # Distributions
    plt.subplot(2, 3, 4)
    plt.hist(final_data['call_volume'], bins=20, alpha=0.7, edgecolor='black')
    plt.xlabel('Call Volume')
    plt.ylabel('Frequency')
    plt.title('Call Volume Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 3, 5)
    plt.hist(total_mail, bins=20, alpha=0.7, edgecolor='black', color='green')
    plt.xlabel('Total Mail Volume')
    plt.ylabel('Frequency')
    plt.title('Total Mail Distribution')
    plt.grid(True, alpha=0.3)
    
    # Top mail types
    plt.subplot(2, 3, 6)
    mail_sums = final_data.drop(['date', 'call_volume'], axis=1).sum().sort_values(ascending=False)
    top_types = mail_sums.head(10)
    plt.bar(range(len(top_types)), top_types.values)
    plt.xlabel('Mail Type Rank')
    plt.ylabel('Total Volume')
    plt.title('Top 10 Mail Types')
    plt.xticks(range(len(top_types)), [f'{i+1}' for i in range(len(top_types))])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "final_clean_dataset.png", dpi=150, bbox_inches='tight')
    plt.close()
    safe_print(f"Final dataset plot saved: {output_dir}/final_clean_dataset.png")
    
    return final_data

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution - step by step data cleaning"""
    
    safe_print("MAIL-TO-CALLS PREDICTION: STEP BY STEP DATA CLEANING")
    safe_print("=" * 80)
    safe_print("STEPS:")
    safe_print("1. Load call intent data and aggregate to daily volumes")
    safe_print("2. Load mail data")
    safe_print("3. Filter to common date ranges")
    safe_print("4. Remove outliers")
    safe_print("5. Create final clean aligned dataset")
    safe_print("=" * 80)
    
    try:
        # Step 1: Load data
        daily_calls = load_call_data()
        mail_pivot = load_mail_data()
        
        # Step 2: Filter to common dates
        calls_filtered, mail_filtered, common_dates = filter_to_common_dates(daily_calls, mail_pivot)
        
        # Step 3: Remove outliers
        calls_clean, mail_clean = remove_outliers(calls_filtered, mail_filtered)
        
        # Step 4: Create final dataset
        final_data = create_final_dataset(calls_clean, mail_clean)
        
        # Save final clean dataset
        output_dir = Path(CONFIG["output_dir"])
        final_data.to_csv(output_dir / "final_clean_data.csv", index=False)
        safe_print(f"Final clean dataset saved: {output_dir}/final_clean_data.csv")
        
        # Summary
        safe_print("\n" + "=" * 80)
        safe_print("DATA CLEANING COMPLETE!")
        safe_print("=" * 80)
        safe_print(f"FINAL DATASET SUMMARY:")
        safe_print(f"  Days: {len(final_data)}")
        safe_print(f"  Date range: {final_data['date'].min().date()} to {final_data['date'].max().date()}")
        safe_print(f"  Mail types: {len(final_data.columns) - 2}")
        safe_print(f"  Call volume range: {final_data['call_volume'].min()} to {final_data['call_volume'].max()}")
        safe_print(f"  Files saved to: {output_dir}")
        safe_print("")
        safe_print("READY FOR NEXT STEP: Feature creation and modeling")
        safe_print("=" * 80)
        
        return {
            'success': True,
            'final_data': final_data,
            'output_dir': str(output_dir),
            'summary': {
                'days': len(final_data),
                'mail_types': len(final_data.columns) - 2,
                'date_range': f"{final_data['date'].min().date()} to {final_data['date'].max().date()}"
            }
        }
        
    except Exception as e:
        safe_print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return {'success': False, 'error': str(e)}

if __name__ == "__main__":
    result = main()
    if result['success']:
        safe_print("SUCCESS: Clean dataset ready for modeling!")
    else:
        safe_print(f"FAILED: {result['error']}")
