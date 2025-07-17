#!/usr/bin/env python3
"""
debug_mail_calls.py
============================================
Light-weight diagnostics for the Mail → Calls
pipeline.

Goals
-----
1. Verify that **mail volume** actually has any
   predictive power for **call volume**.
2. Show same-day and lagged correlations.
3. Train a *tiny* Ridge benchmark (top-5 mail
   streams + simple calendar features) with
   expanding-window CV.
4. Plot time-series + scatter for quick eyeball
   inspection.

Add-ons compared with the original snippet
------------------------------------------
* Replaced all smart quotes with ASCII quotes.
* Fixed indentation / PEP-8 spacing.
* Re-uses the 7-day centred-median smoothing so
  results match the production pipeline.
* Prints BOTH raw-scale and log1p correlations.
* Grid-searches Ridge `alpha` over {0.1, 1, 10}.
* Optional LightGBM sanity check (skipped if the
  library is missing).
* Everything is pure ASCII – drop-in ready on
  Windows PowerShell or *nix.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import holidays
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
US_HOLIDAYS = holidays.US()
SMOOTH_Q = 0.05  # percentile threshold for low counts
WINDOW = 7       # centred-median window

# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _to_date(s: pd.Series) -> pd.Series:
    """Robust date parser used in production code."""
    s1 = pd.to_datetime(s, errors="coerce", infer_datetime_format=True)
    if s1.isna().mean() > 0.3:
        s1 = pd.to_datetime(s.str[:10], errors="coerce")
    return s1.dt.date


def _find_file(candidates: List[str]) -> Optional[Path]:
    for p in candidates:
        path = Path(p)
        if path.exists():
            print(f"Found: {path}")
            return path
    return None


# ---------------------------------------------------------------------------
# DATA LOADERS (identical logic to main pipeline)
# ---------------------------------------------------------------------------

def load_mail() -> pd.DataFrame:
    path = _find_file(["mail.csv", "data/mail.csv"])
    if path is None:
        raise FileNotFoundError("mail.csv not found")
    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    df["mail_date"] = _to_date(df["mail_date"])
    df = df.dropna(subset=["mail_date"])
    return df


def load_calls() -> pd.Series:
    p_vol = _find_file(["callvolumes.csv", "data/callvolumes.csv"])
    p_int = _find_file(["callintent.csv", "data/callintent.csv", "callintetn.csv"])
    if p_vol is None or p_int is None:
        raise FileNotFoundError("callvolumes.csv or callintent.csv missing")

    # Legacy daily totals
    df_vol = pd.read_csv(p_vol)
    df_vol.columns = [c.lower().strip() for c in df_vol.columns]
    dcol_v = next(c for c in df_vol.columns if "date" in c)
    df_vol[dcol_v] = _to_date(df_vol[dcol_v])
    vol_daily = df_vol.groupby(dcol_v)[df_vol.columns.difference([dcol_v])[0]].sum()

    # Intent – one row per call
    df_int = pd.read_csv(p_int)
    df_int.columns = [c.lower().strip() for c in df_int.columns]
    dcol_i = next(c for c in df_int.columns if "date" in c or "conversationstart" in c)
    df_int[dcol_i] = _to_date(df_int[dcol_i])
    int_daily = df_int.groupby(dcol_i).size()

    # Scale volumes to intent on overlap
    overlap = vol_daily.index.intersection(int_daily.index)
    if len(overlap) >= 5:
        scale = int_daily.loc[overlap].mean() / vol_daily.loc[overlap].mean()
        print(f"Scaling factor: {scale:.3f}")
        vol_daily *= scale

    total = vol_daily.combine_first(int_daily).sort_index()

    # Reproduce production smoothing
    q_low = total.quantile(SMOOTH_Q)
    med7 = total.rolling(WINDOW, center=True, min_periods=1).median()
    total.loc[total < q_low] = med7.loc[total < q_low]

    return total.rename("calls_total")


# ---------------------------------------------------------------------------
# MAIN DEBUG ROUTINE
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("DEBUGGING MAIL→CALLS RELATIONSHIP")
    print("=" * 60)

    # 1 ▪ Load data ---------------------------------------------------------
    print("\n1. Loading data…")
    mail = load_mail()
    calls = load_calls()
    print(f"Mail records : {len(mail):,}")
    print(f"Call days    : {len(calls):,}")

    # 2 ▪ Daily frame -------------------------------------------------------
    print("\n2. Preparing daily frame…")
    mail_daily = (
        mail.groupby(["mail_date", "mail_type"], as_index=False)["mail_volume"].sum()
        .pivot(index="mail_date", columns="mail_type", values="mail_volume")
        .fillna(0)
    )
    mail_daily.index = pd.to_datetime(mail_daily.index)
    calls.index = pd.to_datetime(calls.index)

    biz_mask = (~mail_daily.index.weekday.isin([5, 6])) & (~mail_daily.index.isin(US_HOLIDAYS))
    mail_daily = mail_daily.loc[biz_mask]
    calls = calls.loc[calls.index.isin(mail_daily.index)]

    daily = mail_daily.join(calls, how="inner")
    print(f"Combined daily shape : {daily.shape}")

    # 3 ▪ Stats -------------------------------------------------------------
    print("\n3. Basic statistics…")
    print(f"Calls   – mean {daily['calls_total'].mean():.0f}  | sd {daily['calls_total'].std():.0f}")
    total_mail = daily.iloc[:, :-1].sum(axis=1)
    print(f"Mail    – mean {total_mail.mean():.0f} | sd {total_mail.std():.0f}")
    print(f"Mail types count    : {daily.shape[1] - 1}")

    # 4 ▪ Correlations ------------------------------------------------------
    print("\n4. Correlations…")
    corr_same = daily["calls_total"].corr(total_mail)
    corr_same_log = np.log1p(daily["calls_total"]).corr(np.log1p(total_mail))
    print(f"Same-day corr        : {corr_same:.3f} (raw)  |  {corr_same_log:.3f} (log1p)")

    lag_corrs: dict[int, float] = {}
    for lag in (1, 2, 3, 7, 14):
        lagged = total_mail.shift(lag)
        lag_corrs[lag] = daily["calls_total"].corr(lagged)
        print(f"Lag {lag:>2} corr          : {lag_corrs[lag]:.3f}")

    # 5 ▪ Top correlated mail streams --------------------------------------
    print("\n5. Top 10 mail types by abs(corr)…")
    mail_corrs = {
        col: daily["calls_total"].corr(daily[col])
        for col in daily.columns[:-1]
        if daily[col].var() > 0
    }
    top_sorted = sorted(mail_corrs.items(), key=lambda kv: abs(kv[1]), reverse=True)
    for rank, (name, corr) in enumerate(top_sorted[:10], 1):
        print(f"  {rank:>2}. {name:<35} {corr:+.3f}")

    # 6 ▪ Tiny Ridge benchmark ---------------------------------------------
    print("\n6. Tiny Ridge benchmark…")
    top5 = [name for name, _ in top_sorted[:5]]
    X = daily[top5].copy()
    y = daily["calls_total"].copy()

    # Basic calendar + lag feature
    X["weekday"] = daily.index.dayofweek
    X["month"] = daily.index.month
    X["total_mail_lag1"] = total_mail.shift(1)

    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X_clean, y_clean = X[mask], y[mask]
    tscv = TimeSeriesSplit(n_splits=3)

    ridge = Ridge()
    grid = GridSearchCV(
        ridge,
        param_grid={"alpha": [0.1, 1.0, 10.0]},
        cv=tscv,
        scoring="r2",
    )

    scaler = StandardScaler()
    r2_scores: list[float] = []
    for train_idx, test_idx in tscv.split(X_clean):
        X_train = scaler.fit_transform(X_clean.iloc[train_idx])
        X_test = scaler.transform(X_clean.iloc[test_idx])
        y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]
        grid.fit(X_train, y_train)
        y_pred = grid.predict(X_test)
        r2_scores.append(r2_score(y_test, y_pred))
    print(f"CV R² scores          : {r2_scores}")
    print(f"Average R²            : {np.mean(r2_scores):.3f}")

    # Optional LightGBM quick check ----------------------------------------
    try:
        from lightgbm import LGBMRegressor  # pylint: disable=import-error

        print("\nLightGBM sanity check…")
        lgbm = LGBMRegressor(random_state=0, n_estimators=200, verbosity=-1)
        r2_lgbm: list[float] = []
        for train_idx, test_idx in tscv.split(X_clean):
            X_train = X_clean.iloc[train_idx]
            X_test = X_clean.iloc[test_idx]
            y_train, y_test = y_clean.iloc[train_idx], y_clean.iloc[test_idx]
            lgbm.fit(X_train, y_train)
            r2_lgbm.append(r2_score(y_test, lgbm.predict(X_test)))
        print(f"LightGBM R²           : {np.mean(r2_lgbm):.3f}")
    except ModuleNotFoundError:
        print("LightGBM not installed – skipping.")

    # 7 ▪ Plots -------------------------------------------------------------
    print("\n7. Generating plots…")
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    axes[0].plot(daily.index, daily["calls_total"], label="Daily Calls")
    axes[0].set_title("Daily Call Volume")
    axes[0].set_ylabel("Calls")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(daily.index, total_mail, label="Total Mail", color="orange")
    axes[1].set_title("Total Mail Volume")
    axes[1].set_ylabel("Mail")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].scatter(total_mail, daily["calls_total"], alpha=0.6)
    axes[2].set_xlabel("Total Mail")
    axes[2].set_ylabel("Calls")
    axes[2].set_title(f"Mail vs Calls (raw corr {corr_same:+.3f})")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("debug_mail_calls.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 8 ▪ Summary ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("DIAGNOSIS SUMMARY")
    print("=" * 60)

    if abs(corr_same) < 0.1 and np.mean(r2_scores) < 0:
        print("❌  Mail volume shows virtually *no* relationship with calls.")
        print("   ↪ Check data alignment, business logic, other drivers.")
    elif np.mean(r2_scores) < 0.1:
        print("⚠️   Relationship is weak – simplify features, try key lags only.")
    else:
        print("✅  There is usable signal – focus on the top lags and mail types.")

    best_lag = max(lag_corrs, key=lambda k: abs(lag_corrs[k]))
    print(f"Best lag by abs(corr): {best_lag} days ({lag_corrs[best_lag]:+.3f})")
    print("Top-3 mail types      :", [name for name, _ in top_sorted[:3]])


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
