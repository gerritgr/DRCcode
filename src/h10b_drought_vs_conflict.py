#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
H10B: Drought Exposure vs IPV Analysis - Woman-Level Bar Charts
===============================================================================

WHAT THIS SCRIPT DOES:
----------------------
This script reproduces the core principle of h9_conflict_vs_ipv.py, but uses
drought exposure classes instead of conflict distance:

1. Read woman-level DHS responses (with GPS coordinates).
2. Read drought time-series data (SPEI-01) and count extreme drought months
   per drought grid location.
3. For each woman, find the nearest drought grid location and assign that
   location's "number of extreme drought months" as her drought exposure class.
4. For each IPV target variable, compute the fraction of "Yes" responses by
   drought class and plot bar charts.
5. Save easy-to-reproduce CSV outputs in output/.

ANALYSIS QUESTION:
------------------
"Is IPV prevalence higher in regions with more drought months than in regions
with fewer drought months?"

OUTPUTS (saved in output/):
---------------------------
- h10b_women_drought_linked.csv
  Woman-level dataset with drought class assignment and recoded IPV variables.

- h10b_all_variables_bar_summary.csv
  Combined long-format summary for all variables and all drought classes.

- h10b_{variable}_bar.csv
  Class-level summary table for one target variable.

- h10b_{variable}_bar.jpg
- h10b_{variable}_bar.pdf
  Bar chart for one target variable.

USAGE:
------
Run from project root:
    python src/h10b_drought_vs_conflict.py
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree

# =============================================================================
# CONFIGURATION (EDIT THIS SECTION FOR CUSTOMIZATION)
# =============================================================================

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "DATA"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_DHS_CSV = DATA_DIR / "DHS" / "women_all_answers_gps.csv"
INPUT_DROUGHT_CSV = DATA_DIR / "Drought" / "DRC_spei01_clean.csv"

# ---------------------------------------------------------------------------
# Main analysis variables: focused on explicit violence in the last 12 months
# ---------------------------------------------------------------------------
# Source reviewed for variable definitions and coding:
# - DATA/DHS/variable_names_map.txt
# - DATA/DHS/variable_names_map.csv
#
# IMPORTANT CODING CHOICE:
# For many DV questions (e.g., D103*, D105*):
#   1 = Often (last 12 months)
#   2 = Sometimes (last 12 months)
#   3 = Yes, but not in the last 12 months
#   4 = Yes, but frequency in last 12 months missing
#
# We code:
# - Yes (1): explicit last-12-month exposure -> [1, 2]
# - No (0): explicit non-last-12-month / never -> usually [0, 3]
# - Missing (NaN): ambiguous timing (e.g., code 4) or missing code.
#
# For D130A/B/C (previous partner violence):
# - Yes (1): code 1 (0-11 months ago)
# - No (0): codes [0, 2, 6] (never, 12+ months ago, never had previous partner)
# - Missing (NaN): ambiguous timing codes [3, 4]
#
# Each indicator can be:
# - type "single": based on one DHS variable
# - type "any_of": any listed variables indicate exposure
TARGET_INDICATORS = [
    {
        "id": "IPV12M_ANY",
        "description": "Any IPV by husband/partner (last 12m, emotional/physical/sexual)",
        "type": "any_of",
        "columns": ["D103A", "D103B", "D103C", "D105A", "D105B", "D105C", "D105D", "D105E", "D105F", "D105J", "D105H", "D105I", "D105K"],
        "yes": [1, 2],
        "no": [0, 3],
    },
    {
        "id": "IPV12M_EMOTIONAL",
        "description": "Emotional IPV by husband/partner (last 12m)",
        "type": "any_of",
        "columns": ["D103A", "D103B", "D103C"],
        "yes": [1, 2],
        "no": [0, 3],
    },
    {
        "id": "IPV12M_PHYSICAL_LESS_SEV",
        "description": "Less severe physical IPV by husband/partner (last 12m)",
        "type": "any_of",
        "columns": ["D105A", "D105B", "D105C", "D105J"],
        "yes": [1, 2],
        "no": [0, 3],
    },
    {
        "id": "IPV12M_PHYSICAL_SEVERE",
        "description": "Severe physical IPV by husband/partner (last 12m)",
        "type": "any_of",
        "columns": ["D105D", "D105E", "D105F"],
        "yes": [1, 2],
        "no": [0, 3],
    },
    {
        "id": "IPV12M_SEXUAL",
        "description": "Sexual IPV by husband/partner (last 12m)",
        "type": "any_of",
        "columns": ["D105H", "D105I", "D105K"],
        "yes": [1, 2],
        "no": [0, 3],
    },
    {
        "id": "NONPARTNER_PHYSICAL_12M",
        "description": "Hit by someone other than husband/partner (last 12m)",
        "type": "single",
        "column": "D117A",
        "yes": [1, 2],
        "no": [0],
    },
    {
        "id": "PREV_PARTNER_PHYSICAL_12M",
        "description": "Physical violence by previous partner (0-11 months ago)",
        "type": "single",
        "column": "D130A",
        "yes": [1],
        "no": [0, 2, 6],
    },
    {
        "id": "PREV_PARTNER_SEXUAL_12M",
        "description": "Sexual violence by previous partner (0-11 months ago)",
        "type": "single",
        "column": "D130B",
        "yes": [1],
        "no": [0, 2, 6],
    },
    {
        "id": "PREV_PARTNER_EMOTIONAL_12M",
        "description": "Emotional violence by previous partner (0-11 months ago)",
        "type": "single",
        "column": "D130C",
        "yes": [1],
        "no": [0, 2, 6],
    },
]

# ---------------------------------------------------------------------------
# Date filtering for drought data (explicit analysis window)
# Default behavior: only drought observations from 2022-01-01 to 2023-12-31.
# Set either bound to None for a one-sided filter:
# - DATE_MIN = None  -> no lower bound
# - DATE_MAX = None  -> no upper bound
# Set both to None for no date filtering at all.
# ---------------------------------------------------------------------------
DATE_MIN = "2022-01-01"
DATE_MAX = "2023-12-31"

# ---------------------------------------------------------------------------
# Column candidates in DHS file
# The script picks the first existing column from each list.
# ---------------------------------------------------------------------------
GPS_LAT_CANDIDATES = ["LATNUM", "latnum", "v023a", "sclustlat", "cluster_lat"]
GPS_LON_CANDIDATES = ["LONGNUM", "longnum", "v023b", "sclustlon", "cluster_lon"]
CLUSTER_ID_CANDIDATES = ["v001", "V001", "hv001", "cluster_id"]
CASE_ID_CANDIDATES = ["caseid", "CASEID"]
WEIGHT_CANDIDATES = ["d005", "D005", "v005", "V005"]

# ---------------------------------------------------------------------------
# Weighting settings
# ---------------------------------------------------------------------------
USE_WEIGHTS = True
WEIGHT_VARIABLE = "d005"  # preferred variable; fallback uses WEIGHT_CANDIDATES

# ---------------------------------------------------------------------------
# Matching settings (woman coordinate -> nearest drought grid location)
# ---------------------------------------------------------------------------
# Optional quality filter:
# - If None: no distance filter; all women get nearest drought class assignment.
# - If numeric: women farther than this degree distance are set to NaN class.
MAX_MATCH_DISTANCE_DEG = None

# Approximate km conversion for reporting only.
DEGREE_TO_KM_APPROX = 111.32

# ---------------------------------------------------------------------------
# Class aggregation settings
# ---------------------------------------------------------------------------
# Minimum women per drought class required to keep that class in plots/tables.
MIN_WOMEN_PER_CLASS = 1

# Bootstrap settings for class-level uncertainty intervals
# These bootstrap intervals are used for error bars in the bar charts.
N_BOOTSTRAP = 1000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95
BOOTSTRAP_RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Output filenames
# ---------------------------------------------------------------------------
WOMEN_LINKED_OUTPUT = OUTPUT_DIR / "h10b_women_drought_linked.csv"
COMBINED_SUMMARY_OUTPUT = OUTPUT_DIR / "h10b_all_variables_bar_summary.csv"

# ---------------------------------------------------------------------------
# Plot style settings
# ---------------------------------------------------------------------------
FIGSIZE = (12, 8)
DPI = 300
BAR_COLOR = sns.color_palette("muted")[1]
BAR_EDGE_COLOR = "black"
BAR_ALPHA = 0.85
GRID_ALPHA = 0.25
Y_MIN = 0.0
Y_MAX = 1.0
X_LABEL = "Extreme drought months at nearest drought grid location"
Y_LABEL = "Fraction of 'Yes' responses"
ERRORBAR_COLOR = "black"
ERRORBAR_CAPSIZE = 4
ERRORBAR_LINEWIDTH = 1.0

# =============================================================================
# INTERNAL CONSTANTS (no need to edit)
# =============================================================================

DROUGHT_CLASS_COL = "drought_months_class"
MATCH_DISTANCE_DEG_COL = "drought_match_distance_deg"
MATCH_DISTANCE_KM_COL = "drought_match_distance_km"
ANALYSIS_WEIGHT_COL = "analysis_weight"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_first_existing_column(df, candidates, purpose_name):
    """
    Return the first column name that exists in DataFrame `df` from candidates.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame where we look for columns.
    candidates : list[str]
        Possible column names to try in order.
    purpose_name : str
        Human-readable label used in error messages.

    Returns
    -------
    str
        Existing column name from `candidates`.
    """
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Could not find {purpose_name} column. Tried: {candidates}. "
        f"Available columns include: {list(df.columns)[:40]}"
    )


def to_boolean_extreme_flag(series):
    """
    Convert drought extreme flag series to boolean robustly.

    Handles common cases:
    - boolean dtype
    - numeric (0/1)
    - strings like 'True'/'False', 'yes'/'no', '1'/'0'
    """
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0) != 0

    normalized = series.astype(str).str.strip().str.lower()
    true_values = {"true", "1", "yes", "y", "t"}
    return normalized.isin(true_values)


def load_drought_data(csv_path, date_min=None, date_max=None):
    """
    Load drought data and optionally filter by date range.

    Expected columns:
    - long
    - lat
    - time_indicator
    - is_extreme_drought
    """
    print("  -> Reading drought CSV...")
    drought = pd.read_csv(csv_path)

    required = ["long", "lat", "time_indicator", "is_extreme_drought"]
    missing = [c for c in required if c not in drought.columns]
    if missing:
        raise ValueError(f"Drought CSV missing required columns: {missing}")

    drought["long"] = pd.to_numeric(drought["long"], errors="coerce")
    drought["lat"] = pd.to_numeric(drought["lat"], errors="coerce")
    drought["time_indicator"] = pd.to_datetime(drought["time_indicator"], errors="coerce")
    drought["is_extreme_drought"] = to_boolean_extreme_flag(drought["is_extreme_drought"])

    drought = drought[
        drought["long"].notna() &
        drought["lat"].notna() &
        drought["time_indicator"].notna()
    ].copy()

    n_before = len(drought)
    if date_min is not None:
        drought = drought[drought["time_indicator"] >= pd.to_datetime(date_min)]
    if date_max is not None:
        drought = drought[drought["time_indicator"] <= pd.to_datetime(date_max)]
    n_after = len(drought)

    print(f"  -> Loaded {n_after:,} valid drought rows")
    if date_min is not None or date_max is not None:
        print(f"  -> Date filter kept {n_after:,}/{n_before:,} rows")

    return drought


def summarize_drought_months(drought_df):
    """
    Summarize drought exposure per drought grid location.

    For each (long, lat), compute:
    - total_measurements
    - extreme_drought_months (count of True in is_extreme_drought)
    """
    print("  -> Aggregating drought months per location...")
    summary = (
        drought_df
        .groupby(["long", "lat"], as_index=False)
        .agg(
            total_measurements=("is_extreme_drought", "size"),
            extreme_drought_months=("is_extreme_drought", "sum"),
        )
    )
    summary["extreme_drought_months"] = summary["extreme_drought_months"].astype(int)

    print(f"  -> Found {len(summary):,} drought locations")
    print(f"  -> Max drought months at one location: {summary['extreme_drought_months'].max()}")
    print(f"  -> Mean drought months across locations: {summary['extreme_drought_months'].mean():.2f}")
    return summary


def recode_series_with_coding(series, yes_values, no_values):
    """
    Recode one DHS variable to binary using explicit yes/no code lists.

    Returns
    -------
    pd.Series
        Float series:
        - 1.0 for yes_values
        - 0.0 for no_values
        - NaN otherwise
    """
    series_num = pd.to_numeric(series, errors="coerce")
    valid_values = list(yes_values) + list(no_values)

    recoded = pd.Series(np.nan, index=series.index, dtype=float)
    valid_mask = series_num.isin(valid_values)
    recoded.loc[valid_mask] = series_num.loc[valid_mask].isin(yes_values).astype(float)
    return recoded


def build_indicator_binary(df, spec, lower_lookup):
    """
    Build one analysis indicator from target specification.

    Parameters
    ----------
    df : pd.DataFrame
        DHS-linked data.
    spec : dict
        One entry from TARGET_INDICATORS.
    lower_lookup : dict
        Mapping from lowercase column name to actual column name in df.

    Returns
    -------
    pd.Series, list[str]
        - Binary indicator series (1/0/NaN)
        - List of source columns actually used
    """
    indicator_type = spec["type"]
    yes_values = spec["yes"]
    no_values = spec["no"]

    if indicator_type == "single":
        src_upper = spec["column"]
        src_key = src_upper.lower()
        if src_key not in lower_lookup:
            raise ValueError(f"Missing required column for indicator {spec['id']}: {src_upper}")
        src_actual = lower_lookup[src_key]
        recoded = recode_series_with_coding(df[src_actual], yes_values=yes_values, no_values=no_values)
        return recoded, [src_actual]

    if indicator_type == "any_of":
        component_cols = spec["columns"]
        missing = [c for c in component_cols if c.lower() not in lower_lookup]
        if missing:
            raise ValueError(f"Missing required columns for indicator {spec['id']}: {missing}")

        component_actual = [lower_lookup[c.lower()] for c in component_cols]
        recoded_components = [
            recode_series_with_coding(df[c], yes_values=yes_values, no_values=no_values)
            for c in component_actual
        ]
        comp_matrix = np.column_stack([s.to_numpy() for s in recoded_components])

        any_yes = np.any(comp_matrix == 1.0, axis=1)
        any_valid = np.any(~np.isnan(comp_matrix), axis=1)

        indicator = np.full(shape=len(df), fill_value=np.nan, dtype=float)
        indicator[any_yes] = 1.0
        indicator[~any_yes & any_valid] = 0.0

        return pd.Series(indicator, index=df.index, dtype=float), component_actual

    raise ValueError(f"Unsupported indicator type for {spec['id']}: {indicator_type}")


def assign_nearest_drought_class(dhs_df, drought_summary_df, lat_col, lon_col):
    """
    Assign drought class to each woman by nearest drought location.

    The drought class is the number of extreme drought months at the nearest
    drought grid point.
    """
    print("  -> Assigning nearest drought class to women...")

    women_coords = dhs_df[[lon_col, lat_col]].to_numpy()
    drought_coords = drought_summary_df[["long", "lat"]].to_numpy()

    # KDTree gives fast nearest-neighbor lookup in lon/lat degree space.
    tree = cKDTree(drought_coords)
    distances_deg, nearest_idx = tree.query(women_coords, k=1)

    enriched = dhs_df.copy()
    enriched["nearest_drought_lon"] = drought_summary_df.iloc[nearest_idx]["long"].to_numpy()
    enriched["nearest_drought_lat"] = drought_summary_df.iloc[nearest_idx]["lat"].to_numpy()
    enriched[DROUGHT_CLASS_COL] = drought_summary_df.iloc[nearest_idx]["extreme_drought_months"].to_numpy()
    enriched[MATCH_DISTANCE_DEG_COL] = distances_deg
    enriched[MATCH_DISTANCE_KM_COL] = distances_deg * DEGREE_TO_KM_APPROX

    # Optional distance filter in case users want to drop poor matches.
    if MAX_MATCH_DISTANCE_DEG is not None:
        far_mask = enriched[MATCH_DISTANCE_DEG_COL] > MAX_MATCH_DISTANCE_DEG
        n_far = int(far_mask.sum())
        if n_far > 0:
            enriched.loc[far_mask, DROUGHT_CLASS_COL] = np.nan
            print(f"  -> Set drought class to NaN for {n_far:,} women beyond distance threshold")

    print(f"  -> Women assigned: {len(enriched):,}")
    print(f"  -> Median match distance: {enriched[MATCH_DISTANCE_DEG_COL].median():.4f} degrees")
    return enriched


def weighted_fraction_yes(y, w):
    """
    Compute weighted fraction of Yes responses for binary outcome y in {0,1}.

    Parameters
    ----------
    y : np.ndarray
        Binary vector (0/1).
    w : np.ndarray
        Non-negative weights.
    """
    w_sum = np.sum(w)
    if w_sum <= 0:
        return np.nan
    return float(np.dot(w, y) / w_sum)


def bootstrap_fraction_ci(y, w, use_weights, n_bootstrap, confidence_level, rng):
    """
    Bootstrap confidence interval for fraction of Yes responses.

    The bootstrap resamples women within a drought class with replacement.
    For each replicate:
    - unweighted estimate: mean(y_boot)
    - weighted estimate: weighted_fraction_yes(y_boot, w_boot)
    """
    n = len(y)
    if n == 0:
        return np.nan, np.nan

    estimates = np.zeros(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        y_boot = y[idx]
        if use_weights:
            w_boot = w[idx]
            estimates[b] = weighted_fraction_yes(y_boot, w_boot)
        else:
            estimates[b] = float(np.mean(y_boot))

    alpha = 1.0 - confidence_level
    lower = float(np.nanpercentile(estimates, 100.0 * (alpha / 2.0)))
    upper = float(np.nanpercentile(estimates, 100.0 * (1.0 - alpha / 2.0)))
    return lower, upper


def compute_class_summary(df, binary_col, weight_col):
    """
    Compute class-level IPV prevalence summary by drought class.

    Output columns:
    - drought_months_class
    - n_total
    - n_yes
    - n_no
    - frac_yes_unweighted
    - weighted_total
    - weighted_yes
    - frac_yes_weighted
    """
    work = df[[DROUGHT_CLASS_COL, binary_col, weight_col]].dropna().copy()
    if work.empty:
        return pd.DataFrame()

    work[DROUGHT_CLASS_COL] = work[DROUGHT_CLASS_COL].astype(int)
    rng = np.random.default_rng(BOOTSTRAP_RANDOM_SEED)
    rows = []

    for drought_class, sub in work.groupby(DROUGHT_CLASS_COL):
        y = sub[binary_col].to_numpy(dtype=float)
        w = sub[weight_col].to_numpy(dtype=float)

        n_total = int(len(y))
        n_yes = int(np.sum(y))
        n_no = n_total - n_yes
        weighted_total = float(np.sum(w))
        weighted_yes = float(np.dot(w, y))

        frac_yes_unweighted = float(np.mean(y)) if n_total > 0 else np.nan
        frac_yes_weighted = weighted_fraction_yes(y, w)

        # Bootstrap confidence intervals for both weighted and unweighted rates.
        unweighted_ci_lower, unweighted_ci_upper = bootstrap_fraction_ci(
            y=y,
            w=w,
            use_weights=False,
            n_bootstrap=N_BOOTSTRAP,
            confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
            rng=rng,
        )
        weighted_ci_lower, weighted_ci_upper = bootstrap_fraction_ci(
            y=y,
            w=w,
            use_weights=True,
            n_bootstrap=N_BOOTSTRAP,
            confidence_level=BOOTSTRAP_CONFIDENCE_LEVEL,
            rng=rng,
        )

        rows.append(
            {
                DROUGHT_CLASS_COL: int(drought_class),
                "n_total": n_total,
                "n_yes": n_yes,
                "n_no": n_no,
                "frac_yes_unweighted": frac_yes_unweighted,
                "frac_yes_unweighted_ci_lower": unweighted_ci_lower,
                "frac_yes_unweighted_ci_upper": unweighted_ci_upper,
                "weighted_total": weighted_total,
                "weighted_yes": weighted_yes,
                "frac_yes_weighted": frac_yes_weighted,
                "frac_yes_weighted_ci_lower": weighted_ci_lower,
                "frac_yes_weighted_ci_upper": weighted_ci_upper,
            }
        )

    grouped = pd.DataFrame(rows)
    if grouped.empty:
        return grouped

    if MIN_WOMEN_PER_CLASS > 1:
        grouped = grouped[grouped["n_total"] >= MIN_WOMEN_PER_CLASS].copy()

    grouped = grouped.sort_values(DROUGHT_CLASS_COL).reset_index(drop=True)
    return grouped


def create_bar_chart(summary_df, var_name, var_desc, output_jpg, output_pdf, use_weights=True):
    """
    Create and save bar chart for one target variable.
    """
    if summary_df.empty:
        print(f"  -> No class summary available for {var_name}, skipping plot.")
        return

    y_col = "frac_yes_weighted" if use_weights else "frac_yes_unweighted"
    lower_col = "frac_yes_weighted_ci_lower" if use_weights else "frac_yes_unweighted_ci_lower"
    upper_col = "frac_yes_weighted_ci_upper" if use_weights else "frac_yes_unweighted_ci_upper"

    y_values = summary_df[y_col].to_numpy()
    y_lower = summary_df[lower_col].to_numpy()
    y_upper = summary_df[upper_col].to_numpy()
    x_values = summary_df[DROUGHT_CLASS_COL].to_numpy()
    yerr = np.vstack([np.maximum(0.0, y_values - y_lower), np.maximum(0.0, y_upper - y_values)])

    fig, ax = plt.subplots(figsize=FIGSIZE)

    bars = ax.bar(
        x_values,
        y_values,
        yerr=yerr,
        capsize=ERRORBAR_CAPSIZE,
        error_kw={"ecolor": ERRORBAR_COLOR, "elinewidth": ERRORBAR_LINEWIDTH},
        color=BAR_COLOR,
        edgecolor=BAR_EDGE_COLOR,
        alpha=BAR_ALPHA,
        width=0.8,
    )

    # Annotate each bar with class sample size for transparency.
    for bar, n in zip(bars, summary_df["n_total"].to_numpy()):
        bar_height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            min(bar_height + 0.015, Y_MAX - 0.01),
            f"n={int(n)}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.set_xlabel(X_LABEL, fontsize=12, fontweight="bold")
    ax.set_ylabel(Y_LABEL, fontsize=12, fontweight="bold")
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.grid(True, axis="y", alpha=GRID_ALPHA, linestyle="--", linewidth=0.5)
    ax.set_axisbelow(True)

    # Use integer ticks to match drought classes clearly.
    ax.set_xticks(x_values)
    if len(x_values) > 12:
        ax.tick_params(axis="x", rotation=45)

    title = f"{var_desc} ({var_name}) vs Drought Exposure Class\n"
    title += f"Bar height = fraction Yes by drought-month class, error bars = bootstrap CI"
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    subtitle = "Weighted prevalence (DHS weights)" if use_weights else "Unweighted prevalence"
    ax.text(
        0.99,
        0.98,
        subtitle,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
    )

    plt.tight_layout()
    plt.savefig(output_jpg, dpi=DPI, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """
    Main execution function.
    """
    print("\n" + "=" * 78)
    print("H10B: DROUGHT EXPOSURE CLASS vs IPV (WOMAN-LEVEL)")
    print("=" * 78)

    # ---------------------------------------------------------------------
    # Step 1: Validate input files
    # ---------------------------------------------------------------------
    print("\n[Step 1/7] Validating input files...")
    if not INPUT_DHS_CSV.exists():
        print(f"ERROR: Missing DHS CSV: {INPUT_DHS_CSV}")
        sys.exit(1)
    if not INPUT_DROUGHT_CSV.exists():
        print(f"ERROR: Missing drought CSV: {INPUT_DROUGHT_CSV}")
        sys.exit(1)
    print(f"  OK: {INPUT_DHS_CSV.name}")
    print(f"  OK: {INPUT_DROUGHT_CSV.name}")

    # ---------------------------------------------------------------------
    # Step 2: Load and summarize drought data
    # ---------------------------------------------------------------------
    print("\n[Step 2/7] Loading drought data...")
    drought_df = load_drought_data(INPUT_DROUGHT_CSV, DATE_MIN, DATE_MAX)
    drought_summary = summarize_drought_months(drought_df)

    # ---------------------------------------------------------------------
    # Step 3: Load DHS data and identify core columns
    # ---------------------------------------------------------------------
    print("\n[Step 3/7] Loading DHS data...")
    dhs = pd.read_csv(INPUT_DHS_CSV, low_memory=False)
    print(f"  -> Loaded {len(dhs):,} DHS rows")

    lat_col = find_first_existing_column(dhs, GPS_LAT_CANDIDATES, "latitude")
    lon_col = find_first_existing_column(dhs, GPS_LON_CANDIDATES, "longitude")
    print(f"  -> Using coordinate columns: {lat_col}, {lon_col}")

    # Optional metadata columns (not required but helpful in output CSV)
    cluster_col = None
    case_col = None
    for c in CLUSTER_ID_CANDIDATES:
        if c in dhs.columns:
            cluster_col = c
            break
    for c in CASE_ID_CANDIDATES:
        if c in dhs.columns:
            case_col = c
            break

    # Resolve weight column (if weighting enabled).
    if USE_WEIGHTS:
        # Try preferred weight variable first.
        preferred = WEIGHT_VARIABLE
        if preferred in dhs.columns:
            weight_col = preferred
        elif preferred.upper() in dhs.columns:
            weight_col = preferred.upper()
        else:
            weight_col = find_first_existing_column(dhs, WEIGHT_CANDIDATES, "weight")
        print(f"  -> Using weight column: {weight_col}")
    else:
        weight_col = None
        print("  -> Weighting disabled (USE_WEIGHTS=False)")

    # Keep only women with valid coordinates.
    dhs[lat_col] = pd.to_numeric(dhs[lat_col], errors="coerce")
    dhs[lon_col] = pd.to_numeric(dhs[lon_col], errors="coerce")
    dhs_with_gps = dhs[dhs[lat_col].notna() & dhs[lon_col].notna()].copy()
    print(f"  -> Women with valid coordinates: {len(dhs_with_gps):,}")

    # ---------------------------------------------------------------------
    # Step 4: Assign drought class to each woman by nearest drought location
    # ---------------------------------------------------------------------
    print("\n[Step 4/7] Linking women to drought classes...")
    linked = assign_nearest_drought_class(dhs_with_gps, drought_summary, lat_col, lon_col)

    # Create normalized analysis weights.
    if USE_WEIGHTS:
        linked[ANALYSIS_WEIGHT_COL] = pd.to_numeric(linked[weight_col], errors="coerce") / 1_000_000.0
        linked[ANALYSIS_WEIGHT_COL] = linked[ANALYSIS_WEIGHT_COL].fillna(1.0)
        # Guard against non-positive or invalid weights.
        linked.loc[linked[ANALYSIS_WEIGHT_COL] <= 0, ANALYSIS_WEIGHT_COL] = 1.0
    else:
        linked[ANALYSIS_WEIGHT_COL] = 1.0

    # ---------------------------------------------------------------------
    # Step 5: Recode variables and export woman-level reproducible CSV
    # ---------------------------------------------------------------------
    print("\n[Step 5/7] Recoding variables and exporting woman-level CSV...")

    # Build a deterministic woman_id for reproducibility.
    linked = linked.reset_index(drop=True)
    linked["woman_id"] = np.arange(1, len(linked) + 1)

    # Keep track of generated indicator columns and metadata for later steps.
    recoded_columns = []
    generated_indicators = []

    # Create a lowercase->actual mapping for robust column lookup.
    dhs_lower_lookup = {col.lower(): col for col in linked.columns}

    for spec in TARGET_INDICATORS:
        indicator_id = spec["id"]
        recoded_col = f"{indicator_id.lower()}_binary"

        try:
            indicator_series, source_cols = build_indicator_binary(linked, spec, dhs_lower_lookup)
        except ValueError as err:
            print(f"  -> WARNING: Skipping {indicator_id}: {err}")
            continue

        linked[recoded_col] = indicator_series
        recoded_columns.append(recoded_col)
        generated_indicators.append(
            {
                "id": indicator_id,
                "description": spec["description"],
                "recoded_col": recoded_col,
                "source_columns": source_cols,
            }
        )

        n_valid = int(indicator_series.notna().sum())
        n_yes = int(indicator_series.fillna(0).sum())
        print(f"  -> Built {indicator_id}: {n_valid:,} valid responses, {n_yes:,} yes")

    # Assemble woman-level output with key tracing fields + recoded targets.
    output_cols = ["woman_id"]
    if case_col is not None:
        output_cols.append(case_col)
    if cluster_col is not None:
        output_cols.append(cluster_col)
    output_cols.extend(
        [
            lat_col,
            lon_col,
            "nearest_drought_lon",
            "nearest_drought_lat",
            DROUGHT_CLASS_COL,
            MATCH_DISTANCE_DEG_COL,
            MATCH_DISTANCE_KM_COL,
            ANALYSIS_WEIGHT_COL,
        ]
    )
    output_cols.extend(recoded_columns)

    linked[output_cols].to_csv(WOMEN_LINKED_OUTPUT, index=False)
    print(f"  -> Saved: {WOMEN_LINKED_OUTPUT.name}")

    # ---------------------------------------------------------------------
    # Step 6: Build per-variable class summaries and bar charts
    # ---------------------------------------------------------------------
    print("\n[Step 6/7] Building summaries and plots...")
    all_summaries = []

    for item in generated_indicators:
        var = item["id"]
        var_lower = var.lower()
        recoded_col = item["recoded_col"]
        var_desc = item["description"]
        var_sources = item["source_columns"]

        if recoded_col not in linked.columns:
            print(f"  -> Skipping {var}: recoded column not available.")
            continue

        # Keep rows where both drought class and recoded response exist.
        valid = linked[linked[DROUGHT_CLASS_COL].notna() & linked[recoded_col].notna()].copy()
        if valid.empty:
            print(f"  -> Skipping {var}: no valid rows after filtering.")
            continue

        summary = compute_class_summary(valid, recoded_col, ANALYSIS_WEIGHT_COL)
        if summary.empty:
            print(f"  -> Skipping {var}: no classes meet MIN_WOMEN_PER_CLASS={MIN_WOMEN_PER_CLASS}.")
            continue

        # Add variable metadata to summary.
        summary["variable"] = var
        summary["variable_description"] = var_desc
        summary["source_columns"] = ";".join(var_sources)
        summary["used_weighting"] = USE_WEIGHTS

        # Reorder columns for readability.
        summary = summary[
            [
                "variable",
                "variable_description",
                "source_columns",
                DROUGHT_CLASS_COL,
                "n_total",
                "n_yes",
                "n_no",
                "frac_yes_unweighted",
                "frac_yes_unweighted_ci_lower",
                "frac_yes_unweighted_ci_upper",
                "weighted_total",
                "weighted_yes",
                "frac_yes_weighted",
                "frac_yes_weighted_ci_lower",
                "frac_yes_weighted_ci_upper",
                "used_weighting",
            ]
        ]

        # Save per-variable CSV.
        per_var_csv = OUTPUT_DIR / f"h10b_{var_lower}_bar.csv"
        summary.to_csv(per_var_csv, index=False)
        print(f"  -> Saved: {per_var_csv.name}")

        # Save per-variable plots.
        per_var_jpg = OUTPUT_DIR / f"h10b_{var_lower}_bar.jpg"
        per_var_pdf = OUTPUT_DIR / f"h10b_{var_lower}_bar.pdf"
        create_bar_chart(summary, var, var_desc, per_var_jpg, per_var_pdf, use_weights=USE_WEIGHTS)
        print(f"  -> Saved: {per_var_jpg.name}")
        print(f"  -> Saved: {per_var_pdf.name}")

        all_summaries.append(summary)

    # ---------------------------------------------------------------------
    # Step 7: Save combined summary and print run summary
    # ---------------------------------------------------------------------
    print("\n[Step 7/7] Writing combined summary...")
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined.to_csv(COMBINED_SUMMARY_OUTPUT, index=False)
        print(f"  -> Saved: {COMBINED_SUMMARY_OUTPUT.name}")
    else:
        print("  -> WARNING: No variable summaries were produced.")

    print("\n" + "=" * 78)
    print("H10B ANALYSIS COMPLETE")
    print("=" * 78)
    print("Outputs created in output/:")
    print(f"  - {WOMEN_LINKED_OUTPUT.name}")
    print(f"  - {COMBINED_SUMMARY_OUTPUT.name} (if at least one variable succeeded)")
    print("  - h10b_<indicator>_bar.csv/.jpg/.pdf for each processed indicator")

    print("\nRun settings:")
    configured_ids = [x["id"] for x in TARGET_INDICATORS]
    generated_ids = [x["id"] for x in generated_indicators]
    print(f"  - Configured indicators: {', '.join(configured_ids)}")
    print(f"  - Generated indicators: {', '.join(generated_ids) if generated_ids else 'none'}")
    print(f"  - Date filter: {DATE_MIN or 'earliest'} to {DATE_MAX or 'latest'}")
    print(f"  - Use weights: {USE_WEIGHTS}")
    if USE_WEIGHTS:
        print(f"  - Weight preference: {WEIGHT_VARIABLE}")
    print(f"  - Min women per class: {MIN_WOMEN_PER_CLASS}")
    print(f"  - Max match distance (deg): {MAX_MATCH_DISTANCE_DEG}")
    print(f"  - Bootstrap samples per class: {N_BOOTSTRAP}")
    print(f"  - Bootstrap confidence level: {BOOTSTRAP_CONFIDENCE_LEVEL:.2f}")
    print("")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(1)
    except Exception as exc:
        print(f"\nERROR: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
