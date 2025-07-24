Perfect! Here’s the code to remove US holidays from your call data. Add this right after loading the call data:

## 📅 **ADD THIS CODE AFTER LINE ~115 (after loading call data):**

```python
# Import holidays at the top of the file (add this near other imports)
import holidays

# Add this function after the imports section:
def remove_us_holidays(df, date_col='date'):
    """Remove US holidays from call data"""
    safe_print("   Removing US holidays from call data...")
    
    # Get US holidays for the date range in your data
    us_holidays = holidays.US()
    
    # Find holidays in your data
    holiday_mask = df[date_col].dt.date.isin([date for date in us_holidays.keys() 
                                             if df[date_col].min().date() <= date <= df[date_col].max().date()])
    
    holidays_found = df[holiday_mask][date_col].dt.date.tolist()
    
    if len(holidays_found) > 0:
        safe_print(f"   Found {len(holidays_found)} US holidays to remove:")
        for holiday_date in sorted(holidays_found):
            holiday_name = us_holidays.get(holiday_date)
            safe_print(f"     - {holiday_date}: {holiday_name}")
    else:
        safe_print("   No US holidays found in date range")
    
    # Remove holidays
    df_no_holidays = df[~holiday_mask].copy()
    
    safe_print(f"   Removed {len(holidays_found)} holiday rows")
    safe_print(f"   Data after holiday removal: {len(df_no_holidays)} rows")
    
    return df_no_holidays
```

## 🔧 **THEN ADD THIS LINE IN `load_call_data()` method:**

**Right after this line:**

```python
df_clean = df_clean[df_clean['date'].dt.weekday < 5]
safe_print(f"   Filtered to business days: {len(df_clean)} rows")
```

**Add this:**

```python
# Remove US holidays
df_clean = remove_us_holidays(df_clean, 'date')
```

## 🎯 **COMPLETE SECTION SHOULD LOOK LIKE:**

```python
# Filter to business days only
df_clean = df_clean[df_clean['date'].dt.weekday < 5]
safe_print(f"   Filtered to business days: {len(df_clean)} rows")

# Remove US holidays
df_clean = remove_us_holidays(df_clean, 'date')

# Sort and clean
df_clean = df_clean[['date', 'call_volume']].sort_values('date').reset_index(drop=True)
```

## 📦 **INSTALL HOLIDAYS PACKAGE:**

If you don’t have it installed:

```bash
pip install holidays
```

That’s it! This will automatically detect and remove all US federal holidays (New Year’s Day, MLK Day, Presidents Day, Memorial Day, Independence Day, Labor Day, Columbus Day, Veterans Day, Thanksgiving, Christmas, etc.) from your call data before training the model.

**Re-run your script and you should see much better results!** 🎉​​​​​​​​​​​​​​​​