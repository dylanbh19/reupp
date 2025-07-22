# ============================================================================
# COLOR ERROR FIXES - BEFORE AND AFTER
# ============================================================================

# FIX 1: Around line 1374 (Success rate by category)
# BEFORE:
bars = ax2.bar(cat_names, success_rates, color=CFG["colors"]["primary"])

# AFTER:
bars = ax2.bar(cat_names, success_rates, color=CFG["colors"]["baseline"])


# FIX 2: Around line 1395 (Improvement distribution)
# BEFORE:
ax3.hist(improvements, bins=20, alpha=0.7, color=CFG["colors"]["primary"], edgecolor='black')

# AFTER:
ax3.hist(improvements, bins=20, alpha=0.7, color=CFG["colors"]["baseline"], edgecolor='black')


# FIX 3: Around line 1450 (Average improvement by category)  
# BEFORE:
bars = ax1.bar(cat_names, avg_improvements, color=CFG["colors"]["primary"])

# AFTER:
bars = ax1.bar(cat_names, avg_improvements, color=CFG["colors"]["baseline"])


# FIX 4: Around line 1518 (Scatter plot)
# BEFORE:
ax1.scatter(overall_improvements, friday_improvements, alpha=0.6, color=CFG["colors"]["primary"])

# AFTER:
ax1.scatter(overall_improvements, friday_improvements, alpha=0.6, color=CFG["colors"]["baseline"])


# FIX 5: Around line 1559 (Improvement by approach number)
# BEFORE:
ax3.scatter(approach_nums, improvements, alpha=0.6, color=CFG["colors"]["primary"])

# AFTER:
ax3.scatter(approach_nums, improvements, alpha=0.6, color=CFG["colors"]["baseline"])


# FIX 6: Around line 1629 (Box plot colors)
# BEFORE:
box['boxes'][0].set_facecolor(CFG["colors"]["primary"])

# AFTER:
box['boxes'][0].set_facecolor(CFG["colors"]["baseline"])


# ============================================================================
# COMPLETE SEARCH AND REPLACE LIST
# ============================================================================

# Simply find and replace ALL instances of:
CFG["colors"]["primary"]

# With:
CFG["colors"]["baseline"]


# ============================================================================
# OR ADD THE PRIMARY COLOR (EASIER FIX)
# ============================================================================

# Just add this line to the CFG colors dict around line 44:
CFG = {
    "baseline_script": "range.py",
    "output_dir": "friday_improvement_results", 
    "test_approaches": 50,
    "colors": {
        "baseline": "#1f77b4",
        "improved": "#2ca02c", 
        "failed": "#d62728",
        "friday": "#ff7f0e",
        "neutral": "#6c757d",
        "primary": "#1f77b4"  # <-- ADD THIS LINE (same as baseline)
    }
}
