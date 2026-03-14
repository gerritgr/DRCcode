#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DHS Data Visualization Script - DRC 2023-24 Survey STILL USES WRONG WEIGHTS 
================================================================================

WHAT THIS SCRIPT DOES:
----------------------
This script creates MAP visualizations from the processed DHS data:

1. MAP VISUALIZATIONS: Shows prevalence of violence indicators by cluster
   - Circle COLOR = Prevalence rate (% of women reporting "Yes")
   - Circle SIZE = Sample size (number of women interviewed)
   - Creates separate maps for each indicator

INPUTS:
-------
- DATA/DHS/women_all_answers_gps.csv (main DHS dataset with GPS)
- DATA/background.png (optional background map of DRC)

OUTPUTS:
--------
- 1a_sexual_violence_any_map.jpg (map of sexual violence-any prevalence)
- 1a_sexual_violence_recent_map.jpg (map of recent sexual violence prevalence)

All outputs are saved in the output/ directory (same level as src/).

USAGE:
------
Run from the project root directory:
    python src/1a_vis_dhs_maps.py

Or from the src directory:
    python 1a_vis_dhs_maps.py

REQUIREMENTS:
-------------
- pandas (data manipulation)
- numpy (numerical operations)
- matplotlib (plotting)
- Pillow/PIL (image handling)

Install with:
    pip install pandas numpy matplotlib Pillow

================================================================================
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# =============================================================================
# CONFIGURATION
# =============================================================================

# Determine paths relative to this script
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Input files
DATA_DIR = PROJECT_ROOT / "DATA"
DHS_DIR = DATA_DIR / "DHS"
INPUT_CSV = DHS_DIR / "women_all_answers_gps.csv"
BACKGROUND_IMAGE = DATA_DIR / "background.png"

# Required indicator columns in women_all_answers_gps.csv
INDICATOR_ANY = "sexual_violence_any"
INDICATOR_RECENT = "sexual_violence_recent"

# Output directory (output/ at same level as src/)
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Geographic extent for DRC [min_lon, max_lon, min_lat, max_lat]
# These coordinates define the bounding box for the Democratic Republic of Congo
# Must match the extent of the background.png image
EXTENT = [11.893979235367297, 31.616531541802978, -13.981788316118982, 5.811825978871415]

# Visualization settings
USE_WEIGHTS = True          # Use DHS sampling weights (recommended)
WEIGHT_VARIABLE = "v005"    # Match h1_heatmap.py style: use v005 and scale by 1,000,000
DPI = 300                   # Resolution of output images (dots per inch)
BG_ALPHA = 0.9            # Transparency of background map (0=invisible, 1=opaque)
CMAP = "magma"             # Color map for prevalence (dark purple to yellow)
CIRCLE_ALPHA = 0.25        # Transparency of cluster circles

# Circle size range (in points²)
SIZE_MIN = 25.0            # Minimum marker area
SIZE_MAX = 300.0           # Maximum marker area

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def weighted_prop(values, weights):
    """
    Calculate weighted proportion for binary (0/1) data.

    In DHS surveys, different women have different sampling probabilities.
    We weight each response to get nationally representative estimates.

    FORMULA:
    Weighted proportion = Sum(value_i × weight_i) / Sum(weight_i)

    Parameters:
    -----------
    values : array-like
        Binary responses (0 or 1, or NaN for missing)
    weights : array-like
        Sampling weights (already scaled, i.e., v005/1,000,000)

    Returns:
    --------
    float : Weighted proportion between 0 and 1, or NaN if no valid data
    """
    v = pd.to_numeric(pd.Series(values), errors="coerce")
    w = pd.to_numeric(pd.Series(weights), errors="coerce")

    # Keep only rows where both value and weight are valid
    mask = v.notna() & w.notna() & (w > 0)

    if not mask.any():
        return np.nan

    return float((v[mask] * w[mask]).sum() / w[mask].sum())


def resolve_column_name(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """
    Return the first matching column name from candidates (case-insensitive).
    """
    col_lookup = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in col_lookup:
            return col_lookup[candidate.lower()]
    return None


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    """
    Main function that orchestrates the MAP visualization workflow.
    """

    print("\n" + "=" * 70)
    print("DHS DATA VISUALIZATION (MAPS ONLY) - DRC 2023-24 Survey")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # STEP 1: VERIFY INPUT FILES
    # -------------------------------------------------------------------------
    print("\n[STEP 1/3] Verifying input files...")
    print("-" * 70)

    if not INPUT_CSV.exists():
        print(f"ERROR: Input CSV not found at {INPUT_CSV}")
        print("\nPlease run DATA/DHS/convert.py first to generate this file.")
        sys.exit(1)

    print(f"✓ Found input CSV: {INPUT_CSV.name}")
    print(f"  Location: {INPUT_CSV}")

    # Check for background image
    has_background = BACKGROUND_IMAGE.exists()
    if has_background:
        print(f"✓ Found background map: {BACKGROUND_IMAGE.name}")
    else:
        print(f"⚠ Background map not found at {BACKGROUND_IMAGE}")
        print(f"  Maps will be created without background image")

    # -------------------------------------------------------------------------
    # STEP 2: LOAD AND AGGREGATE DATA
    # -------------------------------------------------------------------------
    print("\n[STEP 2/3] Loading and aggregating data...")
    print("-" * 70)

    # Load the processed CSV
    print(f"\nReading: {INPUT_CSV.name}")
    data = pd.read_csv(INPUT_CSV)
    print(f"  → Loaded {len(data):,} women's records")

    # Resolve required columns from the main DHS dataset.
    cluster_col = resolve_column_name(data, ["cluster_id", "v001"])
    weight_col = resolve_column_name(data, [WEIGHT_VARIABLE])
    lat_col = resolve_column_name(data, ["LATNUM"])
    lon_col = resolve_column_name(data, ["LONGNUM"])
    indicator_any_col = resolve_column_name(data, [INDICATOR_ANY])
    indicator_recent_col = resolve_column_name(data, [INDICATOR_RECENT])

    required = {
        "cluster id": cluster_col,
        "latitude": lat_col,
        "longitude": lon_col,
        INDICATOR_ANY: indicator_any_col,
        INDICATOR_RECENT: indicator_recent_col,
    }
    missing_required = [label for label, col in required.items() if col is None]
    if missing_required:
        print("\nERROR: Missing required columns in input CSV:")
        for label in missing_required:
            print(f"  - {label}")
        print("\nHint: run src/0_add_rows_of_interest.py first to create the two indicators.")
        sys.exit(1)

    # Normalize/derive standard working columns so downstream plotting logic stays simple.
    data["cluster_id"] = data[cluster_col]
    data["LATNUM"] = pd.to_numeric(data[lat_col], errors="coerce")
    data["LONGNUM"] = pd.to_numeric(data[lon_col], errors="coerce")
    data["_sv_any_num"] = pd.to_numeric(data[indicator_any_col], errors="coerce")
    data["_sv_recent_num"] = pd.to_numeric(data[indicator_recent_col], errors="coerce")

    if USE_WEIGHTS:
        # Match h1_heatmap.py behavior: use selected DHS weight variable and scale by 1e6.
        if weight_col is not None:
            data["sampling_weight"] = (
                pd.to_numeric(data[weight_col], errors="coerce") / 1_000_000.0
            )
            weight_source_msg = f"{weight_col} / 1,000,000"
        else:
            print(f"\nWARNING: {WEIGHT_VARIABLE} not found, using equal weights.")
            data["sampling_weight"] = 1.0
            weight_source_msg = "equal weights (fallback)"

    # Aggregate to cluster level
    print("\nAggregating women's responses by cluster...")

    if USE_WEIGHTS:
        # WEIGHTED aggregation (nationally representative)
        print("  Using WEIGHTED aggregation (recommended)")
        print(f"  → Accounts for DHS sampling design ({weight_source_msg})")

        agg = (
            data.groupby("cluster_id")
                .apply(lambda g: pd.Series({
                    # Weighted prevalence for sexual violence (any)
                    "fraction_any": weighted_prop(
                        g["_sv_any_num"],
                        g["sampling_weight"]
                    ),
                    # Weighted prevalence for sexual violence (recent)
                    "fraction_recent": weighted_prop(
                        g["_sv_recent_num"],
                        g["sampling_weight"]
                    ),
                    # Sample size (number of women)
                    "n_women": len(g),
                    # GPS coordinates
                    "LATNUM": g["LATNUM"].iloc[0] if len(g) > 0 else np.nan,
                    "LONGNUM": g["LONGNUM"].iloc[0] if len(g) > 0 else np.nan
                }), include_groups=False)
                .reset_index()
        )
        title_suffix = "Weighted"
        subtitle = f"Weighted by {weight_source_msg}"

    else:
        # UNWEIGHTED aggregation (simple percentages)
        print("  Using UNWEIGHTED aggregation")
        print("  → Simple percentages (not nationally representative)")

        agg = (
            data.groupby("cluster_id")
                .agg(
                    fraction_any=("_sv_any_num", "mean"),
                    fraction_recent=("_sv_recent_num", "mean"),
                    n_women=("cluster_id", "size"),
                    LATNUM=("LATNUM", "first"),
                    LONGNUM=("LONGNUM", "first")
                )
                .reset_index()
        )
        title_suffix = "Unweighted"
        subtitle = "Simple percentages (not accounting for sampling design)"

    print(f"  → Created summary for {len(agg):,} clusters")

    # Report statistics
    mean_any = agg["fraction_any"].mean()
    mean_recent = agg["fraction_recent"].mean()
    print(f"  → Mean sexual violence (any) prevalence: {mean_any:.1%}")
    print(f"  → Mean sexual violence (recent) prevalence: {mean_recent:.1%}")

    # Filter to valid data within map extent
    # IMPORTANT: Match original script - only require first indicator here
    print("\nFiltering to clusters within map extent...")
    plot_df = agg[
        (agg["LONGNUM"] >= EXTENT[0]) & (agg["LONGNUM"] <= EXTENT[1]) &
        (agg["LATNUM"] >= EXTENT[2]) & (agg["LATNUM"] <= EXTENT[3]) &
        agg["fraction_any"].notna()  # Only first indicator required
    ].copy()

    print(f"  → {len(plot_df):,} clusters within map extent with valid data")

    if len(plot_df) == 0:
        print("\nERROR: No clusters to plot!")
        print("Check that:")
        print("  1. GPS coordinates are valid")
        print("  2. EXTENT matches your study area")
        print("  3. Women have answered the indicators")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STEP 3: CREATE MAP VISUALIZATIONS
    # -------------------------------------------------------------------------
    print("\n[STEP 3/3] Creating map visualizations...")
    print("-" * 70)

    # Load background image if available
    if has_background:
        bg = Image.open(BACKGROUND_IMAGE).convert("RGBA")
        print(f"✓ Loaded background image: {bg.size[0]}×{bg.size[1]} pixels")
    else:
        bg = None

    # Label used in colorbars to document weighting choice.
    value_note = f"(weighted by {weight_source_msg})" if USE_WEIGHTS else "(unweighted)"

    # Determine shared color scale across both maps
    vmin = 0.0
    vmax_any = float(np.nanmax(plot_df["fraction_any"].to_numpy()))
    vmax_recent = float(np.nanmax(plot_df["fraction_recent"].to_numpy()))
    vmax = max(vmax_any, vmax_recent)

    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0

    print(f"\nColor scale (magma colormap):")
    print(f"  → Dark purple = 0% prevalence")
    print(f"  → Bright yellow = {vmax:.1%} prevalence (max across both indicators)")

    # Scale marker sizes using percentile clipping (matching original script exactly)
    sizes = pd.to_numeric(plot_df["n_women"], errors="coerce").fillna(0).astype(float)

    # Robust scaling using percentiles to avoid one extreme cluster dominating the scale
    p05 = float(np.nanpercentile(sizes.to_numpy(), 5))
    p95 = float(np.nanpercentile(sizes.to_numpy(), 95))
    if not np.isfinite(p05):
        p05 = sizes.min()
    if not np.isfinite(p95):
        p95 = sizes.max()
    if p95 <= p05:
        p05, p95 = sizes.min(), sizes.max()
        if p95 <= p05:
            p05, p95 = 0.0, max(1.0, float(sizes.max()))

    # Map sample sizes to marker AREAS (points^2) with linear scaling
    sizes_clipped = sizes.clip(p05, p95)
    sizes_normalized = (sizes_clipped - p05) / (p95 - p05)
    sizes = SIZE_MIN + (SIZE_MAX - SIZE_MIN) * sizes_normalized

    min_n = plot_df["n_women"].min()
    max_n = plot_df["n_women"].max()

    print(f"\nCircle sizes (area-scaled with percentile clipping):")
    print(f"  → Smallest cluster: {min_n:.0f} women")
    print(f"  → Largest cluster: {max_n:.0f} women")
    print(f"  → Marker area range: {SIZE_MIN:.0f} to {SIZE_MAX:.0f} points²")

    # ─────────────────────────────────────────────────────────────────────────
    # MAP 1: Sexual violence (any)
    # ─────────────────────────────────────────────────────────────────────────
    print("\nCreating Map 1: Sexual violence (any) prevalence...")

    df_sv = plot_df.dropna(subset=["fraction_any", "LATNUM", "LONGNUM"]).copy()

    fig1, ax1 = plt.subplots(figsize=(14, 11))

    # Add background image
    if has_background:
        ax1.imshow(bg, extent=EXTENT, origin="upper", alpha=BG_ALPHA, zorder=0)

    # Plot cluster circles
    sc1 = ax1.scatter(
        df_sv["LONGNUM"],
        df_sv["LATNUM"],
        c=df_sv["fraction_any"],
        s=sizes.loc[df_sv.index],
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        alpha=CIRCLE_ALPHA,
        linewidths=0.5,
        edgecolors="black",
        zorder=2
    )

    ax1.set_xlim(EXTENT[0], EXTENT[1])
    ax1.set_ylim(EXTENT[2], EXTENT[3])
    ax1.set_aspect("equal", "box")
    ax1.set_axis_off()
    ax1.set_title(
        f"Prevalence ({title_suffix})\nSexual Violence (Any)\n{subtitle}",
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    # Add colorbar
    cbar1 = plt.colorbar(sc1, ax=ax1, orientation="vertical",
                         fraction=0.03, pad=0.02)
    cbar1.set_label(
        f"Fraction Reporting Sexual Violence (Any)\n{value_note}",
        rotation=90,
        fontsize=11,
        labelpad=15
    )
    cbar1.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'{x:.0%}')
    )

    # Add size legend (matching original script exactly)
    legend_samples = [int(min_n), int((min_n + max_n) / 2), int(max_n)]
    handles1 = []

    for n in legend_samples:
        # Convert sample count n into the same marker AREA scale used above
        nn = float(n)
        nn = min(max(nn, p05), p95)
        t = 0.0 if p95 <= p05 else (nn - p05) / (p95 - p05)
        size_val = SIZE_MIN + (SIZE_MAX - SIZE_MIN) * t
        handles1.append(
            plt.scatter([], [], s=size_val, color="gray", alpha=0.6,
                        edgecolors="black", linewidths=0.5,
                        label=f"{n} women")
        )

    ax1.legend(
        handles=handles1,
        scatterpoints=1,
        frameon=True,
        labelspacing=1.5,
        title="Number of Women\nInterviewed per Cluster",
        loc="lower right",
        fontsize=10,
        title_fontsize=11
    )

    # Save map 1
    output1 = OUTPUT_DIR / "1a_sexual_violence_any_map.jpg"
    plt.savefig(output1, dpi=DPI, bbox_inches="tight")
    plt.close()

    output1_csv = OUTPUT_DIR / "1a_sexual_violence_any_map.csv"
    pd.DataFrame(
        {
            "LONGNUM": df_sv["LONGNUM"],
            "LATNUM": df_sv["LATNUM"],
            "value": df_sv["fraction_any"],
            "size": sizes.loc[df_sv.index],
            "n_women": df_sv["n_women"],
            "cluster_id": df_sv["cluster_id"],
        }
    ).to_csv(output1_csv, index=False)

    print(f"  ✓ Saved: {output1.name}")
    print(f"  ✓ Saved: {output1_csv.name}")
    print(f"    → {len(df_sv):,} clusters plotted")

    # ─────────────────────────────────────────────────────────────────────────
    # MAP 2: Sexual violence (recent)
    # ─────────────────────────────────────────────────────────────────────────
    print("\nCreating Map 2: Sexual violence (recent) prevalence...")

    df_other = plot_df.dropna(subset=["fraction_recent", "LATNUM", "LONGNUM"]).copy()

    fig2, ax2 = plt.subplots(figsize=(14, 11))

    # Add background image
    if has_background:
        ax2.imshow(bg, extent=EXTENT, origin="upper", alpha=BG_ALPHA, zorder=0)

    # Plot cluster circles
    sc2 = ax2.scatter(
        df_other["LONGNUM"],
        df_other["LATNUM"],
        c=df_other["fraction_recent"],
        s=sizes.loc[df_other.index],
        cmap=CMAP,
        vmin=vmin,
        vmax=vmax,
        alpha=CIRCLE_ALPHA,
        linewidths=0.5,
        edgecolors="black",
        zorder=2
    )

    ax2.set_xlim(EXTENT[0], EXTENT[1])
    ax2.set_ylim(EXTENT[2], EXTENT[3])
    ax2.set_aspect("equal", "box")
    ax2.set_axis_off()
    ax2.set_title(
        f"Prevalence ({title_suffix})\nSexual Violence (Recent)\n{subtitle}",
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    # Add colorbar
    cbar2 = plt.colorbar(sc2, ax=ax2, orientation="vertical",
                         fraction=0.03, pad=0.02)
    cbar2.set_label(
        f"Fraction Reporting Sexual Violence (Recent)\n{value_note}",
        rotation=90,
        fontsize=11,
        labelpad=15
    )
    cbar2.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'{x:.0%}')
    )

    # Add size legend
    handles2 = []
    for n in legend_samples:
        nn = float(n)
        nn = min(max(nn, p05), p95)
        t = 0.0 if p95 <= p05 else (nn - p05) / (p95 - p05)
        size_val = SIZE_MIN + (SIZE_MAX - SIZE_MIN) * t
        handles2.append(
            plt.scatter([], [], s=size_val, color="gray", alpha=0.6,
                        edgecolors="black", linewidths=0.5,
                        label=f"{n} women")
        )

    ax2.legend(
        handles=handles2,
        scatterpoints=1,
        frameon=True,
        labelspacing=1.5,
        title="Number of Women\nInterviewed per Cluster",
        loc="lower right",
        fontsize=10,
        title_fontsize=11
    )

    # Save map 2
    output2 = OUTPUT_DIR / "1a_sexual_violence_recent_map.jpg"
    plt.savefig(output2, dpi=DPI, bbox_inches="tight")
    plt.close()

    output2_csv = OUTPUT_DIR / "1a_sexual_violence_recent_map.csv"
    pd.DataFrame(
        {
            "LONGNUM": df_other["LONGNUM"],
            "LATNUM": df_other["LATNUM"],
            "value": df_other["fraction_recent"],
            "size": sizes.loc[df_other.index],
            "n_women": df_other["n_women"],
            "cluster_id": df_other["cluster_id"],
        }
    ).to_csv(output2_csv, index=False)

    print(f"  ✓ Saved: {output2.name}")
    print(f"  ✓ Saved: {output2_csv.name}")
    print(f"    → {len(df_other):,} clusters plotted")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\n📊 OUTPUT FILES (saved in {OUTPUT_DIR.name}/):")
    print(f"\n  1. 1a_sexual_violence_any_map.jpg")
    print(f"     → Map of sexual violence (any) prevalence")
    print(f"     → {len(df_sv):,} clusters visualized")
    print(f"\n  2. 1a_sexual_violence_any_map.csv")
    print(f"     → Data behind the sexual violence-any map (location/value/size)")

    print(f"\n  3. 1a_sexual_violence_recent_map.jpg")
    print(f"     → Map of sexual violence (recent) prevalence")
    print(f"     → {len(df_other):,} clusters visualized")
    print(f"\n  4. 1a_sexual_violence_recent_map.csv")
    print(f"     → Data behind the sexual violence-recent map (location/value/size)")

    print(f"\n⚙️  Settings:")
    print(f"  → Weighting: {title_suffix}")
    print(f"  → Resolution: {DPI} DPI")
    print(f"  → Background map: {'Yes' if has_background else 'No'}")

    print("\n✅ All visualizations created successfully!\n")


# =============================================================================
# RUN THE SCRIPT
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Visualization interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
