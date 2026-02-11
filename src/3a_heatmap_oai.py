#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
3a_heatmap.py — DHS Heatmap Explorer (Wealth × Education → IPV prevalence)
================================================================================

GOAL (H1):
---------
Explore whether IPV prevalence decreases monotonically with:
  - Household wealth
  - Women’s educational attainment

PLOT:
-----
- X axis: Wealth (typically V190 wealth quintile)
- Y axis: Education (typically V106 highest educational level)
- Cell color: Fraction of "Yes" among respondents with valid IPV response (0/1)

INPUT:
------
- DATA/DHS/women_all_answers_gps.csv  (created by your convert.py)

OUTPUT (all start with "3a"):
----------------------------
- output/3a_heatmap_data.csv   (filtered "original" row-level data used)
- output/3a_heatmap.png
- output/3a_heatmap.pdf

DEPENDENCIES:
-------------
pip install pandas numpy matplotlib
================================================================================
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION (easy to change)
# =============================================================================

# Input file
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_CSV = PROJECT_ROOT / "DATA" / "DHS" / "women_all_answers_gps.csv"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Variables (DHS codes) ---
# Outcome: Any IPV (DV module)
OUTCOME_VAR = "D111"  # easy to change: e.g., "D104", "D106", "D108"

# Axes:
WEALTH_VAR = "V190"   # Wealth index (quintile)
EDU_VAR = "V106"      # Highest educational level

# Plot / analysis options
USE_WEIGHTS = False   # You asked for fraction among valid respondents (unweighted).
                      # If you later want weighted prevalence, set True (uses D005 for D* if present, else V005).

FIGSIZE = (10, 6)
DPI = 300
TITLE = "Any IPV prevalence by Wealth × Education (DHS)"

# =============================================================================
# HELPERS
# =============================================================================

def col_lower(df: pd.DataFrame, code: str) -> str:
    """
    Your CSV uses lowercase DHS codes (e.g. 'd111', 'v190').
    This returns the lowercase column name, raising if missing.
    """
    c = code.lower()
    if c not in df.columns:
        raise KeyError(f"Missing column '{c}' for DHS code '{code}'. "
                       f"Available columns (sample): {list(df.columns)[:40]} ...")
    return c


def weighted_mean_binary(values: pd.Series, weights: pd.Series) -> float:
    """Weighted mean for 0/1 binary, ignoring NaNs and nonpositive weights."""
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    m = v.notna() & w.notna() & (w > 0)
    if not m.any():
        return np.nan
    return float((v[m] * w[m]).sum() / w[m].sum())


def pick_weight_column(df: pd.DataFrame, outcome_code: str) -> str | None:
    """
    DHS weights are stored as integers with 6 implied decimals.
    - For DV outcomes (D*), prefer D005 if present.
    - Else fall back to V005 if present.
    Returns lowercase column name or None if no weight columns exist.
    """
    is_dv = outcome_code.upper().startswith("D")
    if is_dv and "d005" in df.columns:
        return "d005"
    if "v005" in df.columns:
        return "v005"
    return None


def build_category_order(series: pd.Series) -> list:
    """
    Reasonable default order:
    - If values look numeric-coded (e.g. 1..5), sort numerically.
    - Otherwise sort by string.
    """
    s = series.dropna()
    if s.empty:
        return []
    # Try numeric sort
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().mean() > 0.9:
        uniq = sorted(sn.dropna().unique().tolist())
        # return as original type-ish (but numeric is fine for plotting labels)
        return uniq
    # Fallback string sort
    uniq = sorted(s.astype(str).unique().tolist())
    return uniq


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("\n" + "=" * 80)
    print("3a HEATMAP — Wealth × Education → IPV prevalence")
    print("=" * 80)

    if not INPUT_CSV.exists():
        print(f"\nERROR: Input CSV not found: {INPUT_CSV}")
        print("Please generate it first (e.g. your convert.py step).")
        sys.exit(1)

    print(f"\nReading: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df):,} rows.")

    # Resolve columns (lowercase in your CSV)
    y_col = col_lower(df, EDU_VAR)
    x_col = col_lower(df, WEALTH_VAR)
    z_col = col_lower(df, OUTCOME_VAR)

    # -----------------------------------------------------------------------------
    # Filter to valid records
    # -----------------------------------------------------------------------------
    # Outcome must be 0/1 (or at least non-missing). We’ll treat any numeric >0 as "Yes"
    # only if your encoding uses 0/1; if it’s strictly 0/1, this is safe.
    # Here we explicitly keep rows where outcome is 0 or 1.
    z_num = pd.to_numeric(df[z_col], errors="coerce")
    valid_outcome = z_num.isin([0, 1])

    # Need non-missing axis vars too
    filtered = df.loc[valid_outcome & df[x_col].notna() & df[y_col].notna(), :].copy()
    filtered["_outcome_01"] = z_num.loc[filtered.index].astype(int)

    print(f"\nValid rows for analysis: {len(filtered):,}")
    print(f"  - Valid {OUTCOME_VAR} (0/1) and non-missing {WEALTH_VAR}, {EDU_VAR}")

    if len(filtered) == 0:
        print("\nERROR: No valid rows after filtering. Check encodings / missingness.")
        sys.exit(1)

    # -----------------------------------------------------------------------------
    # Weights (optional; default off per your request)
    # -----------------------------------------------------------------------------
    weight_col = None
    if USE_WEIGHTS:
        weight_col = pick_weight_column(filtered, OUTCOME_VAR)
        if weight_col is None:
            print("⚠ USE_WEIGHTS=True but no d005/v005 found. Falling back to unweighted.")
            USE_WEIGHTS_LOCAL = False
        else:
            USE_WEIGHTS_LOCAL = True
            filtered["_weight"] = pd.to_numeric(filtered[weight_col], errors="coerce") / 1_000_000.0
            print(f"Using weights from '{weight_col}' (scaled by 1e6).")
    else:
        USE_WEIGHTS_LOCAL = False

    # -----------------------------------------------------------------------------
    # Build ordered categories for axes
    # -----------------------------------------------------------------------------
    x_order = build_category_order(filtered[x_col])
    y_order = build_category_order(filtered[y_col])

    # Make categorical for stable ordering in pivot
    filtered["_x_cat"] = pd.Categorical(filtered[x_col], categories=x_order, ordered=True)
    filtered["_y_cat"] = pd.Categorical(filtered[y_col], categories=y_order, ordered=True)

    # -----------------------------------------------------------------------------
    # Aggregate: prevalence per (education, wealth)
    # -----------------------------------------------------------------------------
    if USE_WEIGHTS_LOCAL:
        # Weighted prevalence (optional)
        heat = (
            filtered.groupby(["_y_cat", "_x_cat"], observed=True)
            .apply(lambda g: weighted_mean_binary(g["_outcome_01"], g["_weight"]))
            .unstack("_x_cat")
        )
        n_cell = (
            filtered.groupby(["_y_cat", "_x_cat"], observed=True)["_outcome_01"]
            .size()
            .unstack("_x_cat")
        )
    else:
        # Unweighted fraction of yes among valid respondents
        heat = (
            filtered.groupby(["_y_cat", "_x_cat"], observed=True)["_outcome_01"]
            .mean()
            .unstack("_x_cat")
        )
        n_cell = (
            filtered.groupby(["_y_cat", "_x_cat"], observed=True)["_outcome_01"]
            .size()
            .unstack("_x_cat")
        )

    # -----------------------------------------------------------------------------
    # Save "original data used" CSV (row-level filtered)
    # -----------------------------------------------------------------------------
    # Keep the original columns plus a couple of helper columns so it’s reproducible.
    out_data_csv = OUTPUT_DIR / "3a_heatmap_data.csv"
    cols_to_save = [c for c in df.columns if c in filtered.columns]  # original cols
    filtered_to_save = filtered[cols_to_save].copy()
    filtered_to_save["_outcome_01"] = filtered["_outcome_01"]
    filtered_to_save["_wealth_axis"] = filtered["_x_cat"].astype(str)
    filtered_to_save["_edu_axis"] = filtered["_y_cat"].astype(str)
    if USE_WEIGHTS_LOCAL:
        filtered_to_save["_weight_used"] = filtered["_weight"]
        filtered_to_save["_weight_source"] = weight_col
    filtered_to_save.to_csv(out_data_csv, index=False)
    print(f"\nSaved row-level analysis data: {out_data_csv}")

    # -----------------------------------------------------------------------------
    # Plot heatmap
    # -----------------------------------------------------------------------------
    # Convert to numpy; keep NaNs for missing combos.
    mat = heat.to_numpy(dtype=float)
    x_labels = [str(x) for x in heat.columns.tolist()]
    y_labels = [str(y) for y in heat.index.tolist()]

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Show heatmap; we set vmin/vmax to [0,1] since it's a fraction.
    im = ax.imshow(mat, aspect="auto", vmin=0.0, vmax=1.0)

    # Axis labels and ticks
    ax.set_xlabel(f"Wealth ({WEALTH_VAR})")
    ax.set_ylabel(f"Education ({EDU_VAR})")
    ax.set_title(f"{TITLE}\nOutcome: {OUTCOME_VAR} | "
                 f"{'Weighted' if USE_WEIGHTS_LOCAL else 'Unweighted'} prevalence")

    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=0)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Fraction 'Yes' among valid respondents", rotation=90, labelpad=12)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0%}"))

    # Optional: annotate cells with prevalence and N (helps monotonicity checks)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat[i, j]
            n = n_cell.to_numpy()[i, j] if (i < n_cell.shape[0] and j < n_cell.shape[1]) else np.nan
            if np.isfinite(val) and np.isfinite(n):
                ax.text(j, i, f"{val:.0%}\n(n={int(n)})", ha="center", va="center", fontsize=8)

    fig.tight_layout()

    out_png = OUTPUT_DIR / "3a_heatmap.png"
    out_pdf = OUTPUT_DIR / "3a_heatmap.pdf"
    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved heatmap PNG: {out_png}")
    print(f"Saved heatmap PDF: {out_pdf}")

    print("\nDone.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
