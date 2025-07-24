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
