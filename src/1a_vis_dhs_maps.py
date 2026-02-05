#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DHS Data Visualization Script - DRC 2023-24 Survey
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
- DATA/DHS/women_indicators_gps.csv (created by convert.py)
- DATA/background.png (optional background map of DRC)

OUTPUTS:
--------
- indicator1_sv_ever_map.jpg (map of sexual violence prevalence)
- indicator2_severe_ever_map.jpg (map of severe violence prevalence)

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
INPUT_CSV = DHS_DIR / "women_indicators_gps.csv"
BACKGROUND_IMAGE = DATA_DIR / "background.png"

# Output directory (output/ at same level as src/)
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Geographic extent for DRC [min_lon, max_lon, min_lat, max_lat]
# These coordinates define the bounding box for the Democratic Republic of Congo
# Must match the extent of the background.png image
EXTENT = [11.893979235367297, 31.616531541802978, -13.981788316118982, 5.811825978871415]

# Visualization settings
USE_WEIGHTS = True          # Use DHS sampling weights (recommended)
DPI = 300                   # Resolution of output images (dots per inch)
BG_ALPHA = 0.55            # Transparency of background map (0=invisible, 1=opaque)
CMAP = "magma"             # Color map for prevalence (dark purple to yellow)
CIRCLE_ALPHA = 0.45        # Transparency of cluster circles

# Circle size range (in points²)
SIZE_MIN = 25.0            # Minimum marker area
SIZE_MAX = 600.0           # Maximum marker area

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

    # Aggregate to cluster level
    print("\nAggregating women's responses by cluster...")

    if USE_WEIGHTS:
        # WEIGHTED aggregation (nationally representative)
        print("  Using WEIGHTED aggregation (recommended)")
        print("  → Accounts for DHS sampling design")

        agg = (
            data.groupby("cluster_id")
                .apply(lambda g: pd.Series({
                    # Weighted prevalence for sexual violence (ever)
                    "fraction_sv_ever": weighted_prop(
                        g["sv_ever"],
                        g["sampling_weight"]
                    ),
                    # Weighted prevalence for severe violence (ever)
                    "fraction_other": weighted_prop(
                        g["other_indicator"],
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
        subtitle = "Weighted by DHS sampling probability (v005)"

    else:
        # UNWEIGHTED aggregation (simple percentages)
        print("  Using UNWEIGHTED aggregation")
        print("  → Simple percentages (not nationally representative)")

        agg = (
            data.groupby("cluster_id")
                .agg(
                    fraction_sv_ever=("sv_ever", "mean"),
                    fraction_other=("other_indicator", "mean"),
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
    mean_sv = agg["fraction_sv_ever"].mean()
    mean_other = agg["fraction_other"].mean()
    print(f"  → Mean sexual violence (ever) prevalence: {mean_sv:.1%}")
    print(f"  → Mean severe violence (ever) prevalence: {mean_other:.1%}")

    # Filter to valid data within map extent
    # IMPORTANT: Match original script - only require first indicator here
    print("\nFiltering to clusters within map extent...")
    plot_df = agg[
        (agg["LONGNUM"] >= EXTENT[0]) & (agg["LONGNUM"] <= EXTENT[1]) &
        (agg["LATNUM"] >= EXTENT[2]) & (agg["LATNUM"] <= EXTENT[3]) &
        agg["fraction_sv_ever"].notna()  # Only first indicator required
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

    # Determine shared color scale across both maps
    vmin = 0.0
    vmax_sv = float(np.nanmax(plot_df["fraction_sv_ever"].to_numpy()))
    vmax_other = float(np.nanmax(plot_df["fraction_other"].to_numpy()))
    vmax = max(vmax_sv, vmax_other)

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
    # MAP 1: Sexual violence (ever) - d108
    # ─────────────────────────────────────────────────────────────────────────
    print("\nCreating Map 1: Sexual violence (ever) prevalence...")

    df_sv = plot_df.dropna(subset=["fraction_sv_ever", "LATNUM", "LONGNUM"]).copy()

    fig1, ax1 = plt.subplots(figsize=(14, 11))

    # Add background image
    if has_background:
        ax1.imshow(bg, extent=EXTENT, origin="upper", alpha=BG_ALPHA, zorder=0)

    # Plot cluster circles
    sc1 = ax1.scatter(
        df_sv["LONGNUM"],
        df_sv["LATNUM"],
        c=df_sv["fraction_sv_ever"],
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
        f"Prevalence ({title_suffix})\nSexual Violence (Ever) — d108\n{subtitle}",
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    # Add colorbar
    cbar1 = plt.colorbar(sc1, ax=ax1, orientation="vertical",
                         fraction=0.03, pad=0.02)
    cbar1.set_label(
        "Fraction Reporting Sexual Violence (Ever)\n(weighted by sampling probability)",
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
    output1 = OUTPUT_DIR / "indicator1_sv_ever_map.jpg"
    plt.savefig(output1, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: {output1.name}")
    print(f"    → {len(df_sv):,} clusters plotted")

    # ─────────────────────────────────────────────────────────────────────────
    # MAP 2: Severe violence (ever) - d107
    # ─────────────────────────────────────────────────────────────────────────
    print("\nCreating Map 2: Severe violence (ever) prevalence...")

    df_other = plot_df.dropna(subset=["fraction_other", "LATNUM", "LONGNUM"]).copy()

    fig2, ax2 = plt.subplots(figsize=(14, 11))

    # Add background image
    if has_background:
        ax2.imshow(bg, extent=EXTENT, origin="upper", alpha=BG_ALPHA, zorder=0)

    # Plot cluster circles
    sc2 = ax2.scatter(
        df_other["LONGNUM"],
        df_other["LATNUM"],
        c=df_other["fraction_other"],
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
        f"Prevalence ({title_suffix})\nSevere Violence (Ever) — d107\n{subtitle}",
        fontsize=14,
        fontweight="bold",
        pad=20
    )

    # Add colorbar
    cbar2 = plt.colorbar(sc2, ax=ax2, orientation="vertical",
                         fraction=0.03, pad=0.02)
    cbar2.set_label(
        "Fraction Reporting Severe Violence (Ever)\n(weighted by sampling probability)",
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
    output2 = OUTPUT_DIR / "indicator2_severe_ever_map.jpg"
    plt.savefig(output2, dpi=DPI, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Saved: {output2.name}")
    print(f"    → {len(df_other):,} clusters plotted")

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\n📊 OUTPUT FILES (saved in {OUTPUT_DIR.name}/):")
    print(f"\n  1. indicator1_sv_ever_map.jpg")
    print(f"     → Map of sexual violence (ever) prevalence")
    print(f"     → {len(df_sv):,} clusters visualized")

    print(f"\n  2. indicator2_severe_ever_map.jpg")
    print(f"     → Map of severe violence (ever) prevalence")
    print(f"     → {len(df_other):,} clusters visualized")

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
