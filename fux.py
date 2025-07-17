The error is in the `clean_infinite_values` function. The issue is with this line:

```python
if df[col].isnull().any():
```

When you have a Series with all NaN values, `.any()` returns an ambiguous truth value. Here’s the fix:

## 🔧 **Quick Fix:**

Replace the `clean_infinite_values` function with this corrected version:

```python
def clean_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """Clean infinite and NaN values from dataset"""
    logger.info("Cleaning infinite and NaN values...")
    
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with median for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        null_count = df[col].isnull().sum()  # FIXED: Use .sum() instead of .any()
        if null_count > 0:
            median_val = df[col].median()
            if pd.isna(median_val):  # If median is also NaN, use 0
                median_val = 0
            df[col] = df[col].fillna(median_val)
            logger.debug(f"Filled {null_count} NaN values in {col} with {median_val}")
    
    logger.info(f"Cleaned dataset shape: {df.shape}")
    return df
```

## 💡 **What was wrong:**

The original code used `.any()` which returns a boolean, but when pandas can’t determine if it should return `True` or `False` (like when all values are NaN), it throws this “ambiguous truth value” error.

The fix uses `.sum()` to count the actual number of NaN values, which always returns a clear integer that can be compared with `> 0`.

This should resolve the error and let your pipeline continue running!​​​​​​​​​​​​​​​​