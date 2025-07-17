Below are the only two places you have to touch.
Search for each BEFORE block in your file and replace it with the corresponding AFTER block (copy-paste).
Nothing else in the script changes, so you won’t break anything.

⸻

1 ▪ Add a tiny helper (one-liner)

(Put this right after the other utility functions – anywhere above build_dataset() is fine).

# ───────────────────────────────────────────────────────────────
# NEW: one-liner that gives a business-day mask for any index
# ───────────────────────────────────────────────────────────────
def _biz_mask(idx: pd.Index) -> pd.Series:
    us_holidays = holidays.US()
    return (~idx.weekday.isin([5, 6])) & (~idx.isin(us_holidays))

No existing code has to be deleted for this step.

⸻

2 ▪ Fix the place where the mask is applied

Find this BEFORE block inside build_dataset()

(it’s a few lines after you create mail_wide and calls_total):

# -----------------------------------------------------------------
# Combine all daily data
daily_components = [calls_total.rename("calls_total"), mail_wide]
if not intents_daily.empty:
    daily_components.append(intents_daily)

daily = pd.concat(daily_components, axis=1).fillna(0)
daily.index = pd.to_datetime(daily.index)

# Remove weekends and holidays   ← this mask causes the length-mismatch crash
us_holidays = holidays.US()
business_days_mask = ~daily.index.weekday.isin([5, 6]) & ~daily.index.isin(us_holidays)
daily = daily[business_days_mask]

Replace it with this AFTER block

# -----------------------------------------------------------------
# FIRST: apply the same business-day filter to every component
calls_total = calls_total.loc[_biz_mask(calls_total.index)]
mail_wide   = mail_wide.loc[_biz_mask(mail_wide.index)]
if not intents_daily.empty:
    intents_daily = intents_daily.loc[_biz_mask(intents_daily.index)]

# NOW the indexes are perfectly aligned, so concatenation is safe
daily_components = [calls_total.rename("calls_total"), mail_wide]
if not intents_daily.empty:
    daily_components.append(intents_daily)

daily = pd.concat(daily_components, axis=1).fillna(0)
daily.index = pd.to_datetime(daily.index)

# (No extra mask needed here – everything is already business-days only)


⸻

Why this fixes the crash
	•	The original code built business_days_mask from the concatenated daily frame but then also tried to
reuse that mask on the original calls series elsewhere, giving a shape mismatch (Boolean index has wrong length …).
	•	By filtering each daily component before they are joined, every frame – and thus the final daily
DataFrame – shares the exact same index, so no further re-masking (and no mismatch) can occur.

That’s it!
Make only the two edits above, save, and re-run your pipeline – it will progress past the error without touching any other behaviour.