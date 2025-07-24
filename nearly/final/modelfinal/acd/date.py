Sure! Here are the exact line replacements:

**Replace this section in the `load_call_data()` method:**

```python
# OLD CODE (around line 130):
# Process dates
df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')
df_clean = df_clean.dropna(subset=['date'])

# NEW CODE:
# Process dates with multiple format handling
df_clean['date'] = None
date_formats = [
    '%d-%m-%Y',    # dd-mm-yyyy (your call data format)
    '%d/%m/%Y',    # dd/mm/yyyy
    '%Y-%m-%d',    # yyyy-mm-dd (ISO format)
    '%Y/%m/%d',    # yyyy/mm/dd
]

original_date_col_data = df_clean[CONFIG["call_date_col"]].copy()

# Try each format
for date_format in date_formats:
    try:
        df_clean['date'] = pd.to_datetime(original_date_col_data, format=date_format, errors='coerce')
        valid_dates = df_clean['date'].notna().sum()
        if valid_dates > len(df_clean) * 0.8:  # 80% success rate
            safe_print(f"   Call dates parsed with format: {date_format}")
            break
    except:
        continue

df_clean = df_clean.dropna(subset=['date'])
```

**Replace this section in the `load_mail_data()` method:**

```python
# OLD CODE (around line 260):
# Process data
df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
df = df.dropna(subset=[date_col])

# NEW CODE:
# Process mail dates with multiple format handling
mail_date_formats = [
    '%Y-%m-%d',    # yyyy-mm-dd (your mail data format)
    '%Y/%m/%d',    # yyyy/mm/dd
    '%d-%m-%Y',    # dd-mm-yyyy
    '%d/%m/%Y',    # dd/mm/yyyy
]

original_mail_date_data = df[date_col].copy()

# Try each format for mail dates
for date_format in mail_date_formats:
    try:
        df[date_col] = pd.to_datetime(original_mail_date_data, format=date_format, errors='coerce')
        valid_dates = df[date_col].notna().sum()
        if valid_dates > len(df) * 0.8:  # 80% success rate
            safe_print(f"   Mail dates parsed with format: {date_format}")
            break
    except:
        continue

df = df.dropna(subset=[date_col])
```

That’s it! These two replacements will handle your date format differences automatically. 🎯​​​​​​​​​​​​​​​​