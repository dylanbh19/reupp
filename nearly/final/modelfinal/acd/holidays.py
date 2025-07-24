Here’s the corrected code to replace your existing `remove_us_holidays` function. Replace the entire function with this:

```python
def remove_us_holidays(df, date_col='date'):
    """Remove US holidays from call data"""
    safe_print("   Removing US holidays from call data...")
    
    # Debug: Check date format
    safe_print(f"   Date column type: {df[date_col].dtype}")
    safe_print(f"   Sample dates: {df[date_col].head(3).tolist()}")
    
    # Get date range
    start_date = df[date_col].min().date()
    end_date = df[date_col].max().date()
    safe_print(f"   Date range: {start_date} to {end_date}")
    
    # Get US holidays for the years in your data
    years = range(start_date.year, end_date.year + 1)
    us_holidays = holidays.US(years=list(years))
    
    # Debug: Show all holidays in range
    all_holidays_in_range = []
    for holiday_date in us_holidays.keys():
        if start_date <= holiday_date <= end_date:
            all_holidays_in_range.append(holiday_date)
    
    safe_print(f"   Total US holidays in date range: {len(all_holidays_in_range)}")
    
    # Create list of holiday dates as datetime.date objects
    holiday_dates_list = []
    for holiday_date in us_holidays.keys():
        if start_date <= holiday_date <= end_date:
            holiday_dates_list.append(holiday_date)
    
    # Find holidays in your data
    # Convert both to same format for comparison
    df['temp_date'] = df[date_col].dt.date
    holiday_mask = df['temp_date'].isin(holiday_dates_list)
    
    # Get the actual holidays found
    holidays_found = df[holiday_mask]['temp_date'].unique()
    
    if len(holidays_found) > 0:
        safe_print(f"   Found {len(holidays_found)} US holidays to remove:")
        for holiday_date in sorted(holidays_found):
            holiday_name = us_holidays.get(holiday_date, "Unknown Holiday")
            # Also show the call volume on that day
            call_volume = df[df['temp_date'] == holiday_date]['call_volume'].iloc[0]
            safe_print(f"     - {holiday_date}: {holiday_name} (calls: {call_volume})")
    else:
        safe_print("   No US holidays found in date range")
        safe_print("   Double-checking some known holidays...")
        # Check for specific holidays as a debug
        test_dates = [
            datetime(2024, 1, 1).date(),   # New Year's Day
            datetime(2024, 7, 4).date(),   # Independence Day
            datetime(2024, 12, 25).date(), # Christmas
        ]
        for test_date in test_dates:
            if start_date <= test_date <= end_date:
                if test_date in df['temp_date'].values:
                    safe_print(f"     - {test_date} is in data but not detected as holiday!")
    
    # Remove holidays
    df_no_holidays = df[~holiday_mask].copy()
    
    # Clean up temp column
    df_no_holidays = df_no_holidays.drop('temp_date', axis=1)
    df = df.drop('temp_date', axis=1)
    
    safe_print(f"   Removed {len(holidays_found)} holiday rows")
    safe_print(f"   Data after holiday removal: {len(df_no_holidays)} rows")
    
    # Additional check for low volume days that might be holidays
    if 'call_volume' in df_no_holidays.columns:
        low_volume_threshold = df_no_holidays['call_volume'].quantile(0.05)
        low_volume_days = df_no_holidays[df_no_holidays['call_volume'] < low_volume_threshold]
        if len(low_volume_days) > 0:
            safe_print(f"   WARNING: {len(low_volume_days)} days with unusually low volume (<{low_volume_threshold:.0f}) remain")
            safe_print("   These might be holidays or other special days:")
            for idx, row in low_volume_days.head(5).iterrows():
                safe_print(f"     - {row['date'].date()}: {row['call_volume']:.0f} calls")
    
    return df_no_holidays
```

Also, add this alternative function if the above doesn’t work, which removes both holidays AND low-volume outliers:

```python
def remove_holidays_and_outliers(df, date_col='date', call_col='call_volume'):
    """Remove US holidays and call volume outliers"""
    safe_print("   Removing US holidays and outliers...")
    
    original_len = len(df)
    
    # First remove holidays
    df = remove_us_holidays(df, date_col)
    
    # Then remove outliers (very low call volumes)
    if call_col in df.columns:
        # Calculate IQR for outlier detection
        Q1 = df[call_col].quantile(0.25)
        Q3 = df[call_col].quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # But also set a minimum threshold
        min_threshold = max(1000, df[call_col].quantile(0.02))
        lower_bound = max(lower_bound, min_threshold)
        
        safe_print(f"   Outlier bounds: [{lower_bound:.0f}, {upper_bound:.0f}]")
        
        # Remove outliers
        outlier_mask = (df[call_col] < lower_bound) | (df[call_col] > upper_bound)
        outliers = df[outlier_mask]
        
        if len(outliers) > 0:
            safe_print(f"   Found {len(outliers)} outlier days:")
            for idx, row in outliers.head(10).iterrows():
                safe_print(f"     - {row[date_col].date()}: {row[call_col]:.0f} calls")
        
        df = df[~outlier_mask].copy()
    
    safe_print(f"   Total removed: {original_len - len(df)} days")
    safe_print(f"   Remaining data: {len(df)} days")
    
    return df
```

To use this in your DataManager class, replace the holiday removal line with:

```python
# Replace this line:
df_clean = remove_us_holidays(df_clean, 'date')

# With this (for holidays + outliers):
df_clean = remove_holidays_and_outliers(df_clean, 'date', 'call_volume')
```

This improved code:

1. Better handles date format conversion
1. Provides detailed debugging output
1. Shows which holidays were found and their call volumes
1. Checks for known holidays if none are detected
1. Identifies remaining low-volume days that might be missed holidays
1. Optionally removes outliers along with holidays​​​​​​​​​​​​​​​​