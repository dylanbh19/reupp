def remove_us_holidays(df, date_col='date'):
    """Remove US federal holidays from the DataFrame."""
    safe_print("   Removing US holidays from call data...")

    # The holidays library is optimized for this type of lookup.
    # It correctly handles observed holidays (e.g., a holiday on a Saturday observed on Friday).
    us_holidays = holidays.US()

    # Create a boolean mask where True indicates the date is a holiday.
    # This is the most direct and reliable way to check for holidays.
    holiday_mask = df[date_col].dt.date.isin(us_holidays)

    holidays_found = df[holiday_mask]

    if not holidays_found.empty:
        safe_print(f"   Found {len(holidays_found)} US holidays to remove:")
        # Display the found holidays and their names
        for _, row in holidays_found.sort_values(by=date_col).iterrows():
            holiday_date = row[date_col].date()
            holiday_name = us_holidays.get(holiday_date)
            safe_print(f"     - {holiday_date}: {holiday_name}")
    else:
        safe_print("   No US holidays found in the provided date range.")

    # Invert the mask to keep non-holidays and create a copy to avoid SettingWithCopyWarning.
    df_no_holidays = df[~holiday_mask].copy()

    safe_print(f"   Removed {len(holidays_found)} holiday rows.")
    safe_print(f"   Data after holiday removal: {len(df_no_holidays)} rows.")

    return df_no_holidays
