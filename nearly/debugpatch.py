# ------------------------------------------------------------------
# ⬇︎  REPLACE the single 'total = ...' line with this whole snippet
# ------------------------------------------------------------------
total = vol_daily.combine_first(int_daily).sort_index()

# ----–  PATCH MISSING DAYS  –------------------------------------
total = total.asfreq("D")                   # explicit calendar index
wknd   = total.index.weekday >= 5
hol    = total.index.isin(holidays.US())    # skip US holidays
total[wknd | hol] = total[wknd | hol].fillna(0)   # centre closed → 0

# interpolate up to 2-day gaps inside business periods
total = total.interpolate(limit=2, limit_area="inside")

# optional quality flag
aug_df = pd.DataFrame(
    {"calls": total, "is_synthetic": total.isna().astype(int)}
)

# final forward-fill any residual NaNs if you *really* need continuity
aug_df["calls"].ffill(inplace=True)

# WRITE OUT
aug_df.to_csv(
    "augmented_calls.csv",
    index_label="date",
    float_format="%.0f"
)
print("Augmented call-volume series written to augmented_calls.csv")
# ------------------------------------------------------------------
# keep returning a Series exactly like before:
return aug_df["calls"].rename("calls_total")