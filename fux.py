Below are copy-paste-ready patches.
Search for each 📍BEFORE snippet in your file and overwrite / insert the corresponding ✅ AFTER code.
(If the BEFORE text doesn’t match exactly, just locate the line by its comment and drop the AFTER block in the same spot.)

⸻

1 🔧 Add a helper to make column names unique

(put it with the other small utilities – e.g. right after optimize_dtypes)

# -----------------------------------------------------------------------------
# Utility: make column labels unique AFTER they’ve been cleaned/normalised
# -----------------------------------------------------------------------------
def dedup_columns_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all column names are unique by appending '_1', '_2', … to duplicates.
    Does NOT touch values or ordering.
    """
    counts = {}
    new_cols = []
    for col in df.columns:
        idx = counts.get(col, 0)
        new_name = col if idx == 0 else f"{col}_{idx}"
        counts[col] = idx + 1
        new_cols.append(new_name)
    df.columns = new_cols
    return df


⸻

2 🔧 Harden clean_infinite_values (re-paste the whole function)

📍BEFORE

def clean_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """Clean infinite and NaN values from dataset"""
    logger.info("Cleaning infinite and NaN values...")
    
    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        null_count = df[col].isnull().sum()  # FIXED: Use .sum() instead of .any()
        if null_count > 0:
            ...

✅ AFTER

def clean_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """Clean infinite and NaN values from dataset"""
    logger.info("Cleaning infinite and NaN values...")

    # Replace infinite with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        null_count = df[col].isnull().sum()

        # NEW ── in the rare case a duplicate-name pull returned a DataFrame
        if isinstance(null_count, pd.Series):
            null_count = null_count.sum()
        # ---------------------------------------------------------------

        if null_count > 0:
            median_val = df[col].median()
            if pd.isna(median_val):
                median_val = 0
            df[col] = df[col].fillna(median_val)

    logger.info(f"Cleaned dataset shape: {df.shape}")
    return df


⸻

3 🔧 Fix fetch_economic_data call to yfinance

(find the single line with yf.download)

📍BEFORE

data = yf.download(ticker, start=start_date, end=end_date, progress=False, show_errors=False)

✅ AFTER

data = yf.download(ticker, start=start_date, end=end_date, progress=False)


⸻

4 🔧 Deduplicate columns inside build_dataset()

📍Find this block (just after optimize_dtypes(weekly)):

# Optimize memory usage
weekly = optimize_dtypes(weekly)

# Comprehensive data quality validation
quality_report = validate_data_quality(weekly)

✅ Replace with

# Optimize memory usage
weekly = optimize_dtypes(weekly)

# NEW ── ensure we don’t have duplicate column labels
weekly = dedup_columns_df(weekly)

# Comprehensive data quality validation
quality_report = validate_data_quality(weekly)

(everything else in build_dataset() stays the same – you already call
clean_infinite_values and clean_column_names a few lines later).

⸻

5 🔍 Anything else to expect?

Stage	Common trip-ups	Already covered?
Data size vs TimeSeriesSplit	Need ≥ ts_splits+1 rows	If it fails, lower CONFIG["ts_splits"]
LightGBM “bad feature names”	Special chars / dupes	clean_column_names + dedup fix it
yfinance throttling	network errors	gracefully caught – pipeline continues

After the four edits above, rerun:

python call_inbound_forecast_pipeline.py

and the “truth value of a Series is ambiguous” crash (plus the yfinance
warning) will be gone. 🚀