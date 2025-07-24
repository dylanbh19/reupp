The error is because of incorrect import/usage of the holidays library. Here’s the fix:

**At the top of your file, make sure you have:**

```python
import holidays
```

**Then use this corrected `remove_us_holidays` function:**

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
    
    # Get US holidays - CORRECTED SYNTAX
    us_holidays = holidays.UnitedStates()  # Or holidays.US()
    
    # Find holidays in your data
    df['temp_date'] = df[date_col].dt.date
    
    # Check each date individually
    holiday_mask = df['temp_date'].apply(lambda x: x in us_holidays)
    
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
    
    # Remove holidays
    df_no_holidays = df[~holiday_mask].copy()
    
    # Clean up temp column
    df_no_holidays = df_no_holidays.drop('temp_date', axis=1)
    df = df.drop('temp_date', axis=1)
    
    safe_print(f"   Removed {len(holidays_found)} holiday rows")
    safe_print(f"   Data after holiday removal: {len(df_no_holidays)} rows")
    
    return df_no_holidays
```

**If the above still doesn’t work, use this simpler version:**

```python
def remove_us_holidays(df, date_col='date'):
    """Remove US holidays from call data - simplified version"""
    safe_print("   Removing US holidays from call data...")
    
    original_len = len(df)
    
    # Create US holidays object
    us_holidays = holidays.US()
    
    # Create a list to store indices to drop
    indices_to_drop = []
    
    # Check each date
    for idx, row in df.iterrows():
        date_value = row[date_col]
        if pd.notna(date_value):
            # Convert to date object if needed
            if hasattr(date_value, 'date'):
                check_date = date_value.date()
            else:
                check_date = date_value
            
            # Check if it's a holiday
            if check_date in us_holidays:
                indices_to_drop.append(idx)
                holiday_name = us_holidays.get(check_date)
                safe_print(f"   Found holiday: {check_date} - {holiday_name}")
    
    # Drop holiday rows
    df_no_holidays = df.drop(indices_to_drop).copy()
    
    safe_print(f"   Removed {len(indices_to_drop)} holiday rows")
    safe_print(f"   Data remaining: {len(df_no_holidays)} rows")
    
    return df_no_holidays
```

**Alternative: If holidays library is problematic, here’s a manual approach:**

```python
def remove_us_holidays_manual(df, date_col='date'):
    """Remove US holidays manually without holidays library"""
    safe_print("   Removing US holidays (manual method)...")
    
    # Define major US holidays that affect call centers
    # Format: (month, day)
    fixed_holidays = [
        (1, 1),   # New Year's Day
        (7, 4),   # Independence Day
        (12, 25), # Christmas Day
        (12, 24), # Christmas Eve (many businesses close early)
        (12, 31), # New Year's Eve
    ]
    
    # Create date column for comparison
    df['temp_month'] = df[date_col].dt.month
    df['temp_day'] = df[date_col].dt.day
    
    # Create mask for holidays
    holiday_mask = df.apply(
        lambda row: (row['temp_month'], row['temp_day']) in fixed_holidays, 
        axis=1
    )
    
    # Also check for very low volume days (likely holidays)
    if 'call_volume' in df.columns:
        threshold = df['call_volume'].quantile(0.02)  # Bottom 2%
        low_volume_mask = df['call_volume'] < threshold
        safe_print(f"   Low volume threshold: {threshold:.0f} calls")
        
        # Combine masks
        remove_mask = holiday_mask | low_volume_mask
    else:
        remove_mask = holiday_mask
    
    # Show what we're removing
    holidays_found = df[remove_mask]
    if len(holidays_found) > 0:
        safe_print(f"   Removing {len(holidays_found)} holiday/low-volume days:")
        for idx, row in holidays_found.head(10).iterrows():
            safe_print(f"     - {row[date_col].date()}: {row.get('call_volume', 'N/A')} calls")
    
    # Remove holidays
    df_clean = df[~remove_mask].copy()
    
    # Clean up temp columns
    df_clean = df_clean.drop(['temp_month', 'temp_day'], axis=1)
    
    safe_print(f"   Data after removal: {len(df_clean)} rows")
    
    return df_clean
```

**To use the manual version, replace this line in your DataManager class:**

```python
df_clean = remove_us_holidays_manual(df_clean, 'date')
```

The manual version doesn’t require the holidays library and will remove both fixed holidays and extremely low volume days that are likely holidays or closures.​​​​​​​​​​​​​​​​