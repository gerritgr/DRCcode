#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DHS Data Visualization Script - DRC 2023-24 Survey
================================================================================

WHAT THIS SCRIPT DOES:
----------------------
This script creates visualizations from the processed DHS data:

1. MAP VISUALIZATIONS: Shows prevalence of violence indicators by cluster
   - Circle COLOR = Prevalence rate (% of women reporting "Yes")
   - Circle SIZE = Sample size (number of women interviewed)
   - Creates separate maps for each indicator

2. CORRELATION SCATTER PLOT: Shows relationship between the two indicators
   - Each point = one cluster
   - X-axis = Sexual violence (ever) prevalence
   - Y-axis = Severe violence (ever) prevalence
   - Point size = sample size

INPUTS:
-------
- DATA/DHS/women_indicators_gps.csv (created by convert.py)
- DATA/background.png (optional background map of DRC)

OUTPUTS:
--------
- indicator1_sv_ever_map.jpg (map of sexual violence prevalence)
- indicator2_severe_ever_map.jpg (map of severe violence prevalence)
- cluster_correlation_scatter.jpg (scatter plot comparing indicators)

All outputs are saved in the src/ directory (same directory as this script).

USAGE:
------
Run from the project root directory:
    python src/1a_vis_dhs.py

Or from the src directory:
    python 1a_vis_dhs.py

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

# Output directory (same as this script - src/)
OUTPUT_DIR = SCRIPT_DIR

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
    
    Example:
    --------
    3 women: [Yes(1), No(0), Yes(1)] with weights [0.5, 2.0, 1.0]
    Weighted proportion = (1×0.5 + 0×2.0 + 1×1.0) / (0.5 + 2.0 + 1.0)
                        = 1.5 / 3.5 = 0.43 (43%)
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
    Main function that orchestrates the visualization workflow.
    """
    
    print("\n" + "=" * 70)
    print("DHS DATA VISUALIZATION - DRC 2023-24 Survey")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # STEP 1: VERIFY INPUT FILES
    # -------------------------------------------------------------------------
    print("\n[STEP 1/4] Verifying input files...")
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
    print("\n[STEP 2/4] Loading and aggregating data...")
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
    print("\n[STEP 3/4] Creating map visualizations...")
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
    # STEP 4: CREATE CORRELATION SCATTER PLOT
    # -------------------------------------------------------------------------
    print("\n[STEP 4/4] Creating correlation scatter plot...")
    print("-" * 70)
    
    # Filter to clusters with both indicators
    df_corr = plot_df.dropna(
        subset=["fraction_sv_ever", "fraction_other"]
    ).copy()
    
    print(f"\n  → {len(df_corr):,} clusters have both indicators")
    
    if len(df_corr) == 0:
        print("  ⚠ No clusters with both indicators - skipping scatter plot")
    else:
        # Calculate correlation
        correlation = df_corr["fraction_sv_ever"].corr(df_corr["fraction_other"])
        print(f"  → Correlation coefficient: {correlation:.3f}")
        
        # Create scatter plot
        fig3, ax3 = plt.subplots(figsize=(10, 10))
        
        # Plot points
        scatter = ax3.scatter(
            df_corr["fraction_sv_ever"],
            df_corr["fraction_other"],
            s=sizes.loc[df_corr.index],
            c=df_corr["n_women"],
            cmap="viridis",
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5
        )
        
        # Add diagonal reference line (y=x)
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1, 
                label="y = x reference")
        
        # Add best fit line
        from numpy.polynomial import polynomial as P
        x = df_corr["fraction_sv_ever"].values
        y = df_corr["fraction_other"].values
        coefs = P.polyfit(x, y, 1)
        fit_x = np.linspace(0, max(x.max(), y.max()), 100)
        fit_y = P.polyval(fit_x, coefs)
        ax3.plot(fit_x, fit_y, 'r-', alpha=0.5, linewidth=2, 
                label=f"Best fit (r={correlation:.3f})")
        
        # Labels and formatting
        ax3.set_xlabel("Sexual Violence (Ever) - d108\nFraction Reporting Yes", 
                      fontsize=12)
        ax3.set_ylabel("Severe Violence (Ever) - d107\nFraction Reporting Yes", 
                      fontsize=12)
        ax3.set_title(
            f"Cluster-Level Correlation Between Violence Indicators\n({title_suffix})",
            fontsize=14,
            fontweight="bold",
            pad=20
        )
        
        # Format axes as percentages
        ax3.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))
        
        # Add grid
        ax3.grid(True, alpha=0.3, linestyle='--')
        
        # Add colorbar for sample size
        cbar3 = plt.colorbar(scatter, ax=ax3)
        cbar3.set_label("Number of Women per Cluster", rotation=90, labelpad=15)
        
        # Add legend
        ax3.legend(loc="upper left", fontsize=10)
        
        # Set equal aspect ratio and limits
        max_val = max(x.max(), y.max()) * 1.05
        ax3.set_xlim(0, max_val)
        ax3.set_ylim(0, max_val)
        ax3.set_aspect("equal", "box")
        
        # Save scatter plot
        output3 = OUTPUT_DIR / "cluster_correlation_scatter.jpg"
        plt.savefig(output3, dpi=DPI, bbox_inches="tight")
        plt.close()
        
        print(f"  ✓ Saved: {output3.name}")
        print(f"    → {len(df_corr):,} clusters plotted")
    
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
    
    if len(df_corr) > 0:
        print(f"\n  3. cluster_correlation_scatter.jpg")
        print(f"     → Correlation plot between indicators")
        print(f"     → {len(df_corr):,} clusters plotted")
        print(f"     → Correlation: {correlation:.3f}")
    
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
        sys.exit(1)" because for some reason the previous script computed another r2 value for the scatter plot "#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DHS (DRC 2023–24) — Sexual Violence Data Export and Visualization
================================================================================

INTRODUCTION TO DHS DATA (for newcomers):
------------------------------------------
The Demographic and Health Surveys (DHS) are nationally representative household
surveys conducted in low- and middle-income countries. This script analyzes data
from the Democratic Republic of Congo (DRC) 2023-24 survey.

KEY DHS CONCEPTS YOU NEED TO KNOW:

1. CLUSTERS (Primary Sampling Units):
   - DHS uses a two-stage sampling design
   - First, they select "clusters" (villages or neighborhoods) across the country
   - Then, they randomly select households within each cluster
   - GPS coordinates are collected for each cluster (not individual households)
   - For privacy, GPS coordinates are "displaced" (randomly shifted) by up to 5km

2. SAMPLING WEIGHTS (v005):
   - Not all women have equal probability of being selected
   - Urban/rural areas, regions, etc. are sampled differently
   - Each woman gets a "weight" to make the data nationally representative
   - The weight v005 must be divided by 1,000,000 before use (DHS convention)
   - When calculating fractions, we weight each response by this value

3. DATA FILES:
   - IR (Individual Recode): One row per interviewed woman aged 15-49
     Contains all her survey responses including sexual violence questions
   - GE (Geographic): One row per cluster with GPS coordinates
   - We link these files using cluster ID (v001 in IR = DHSCLUST in GE)

4. VARIABLES USED IN THIS SCRIPT (corrected):
   - d108: Experienced any sexual violence (D105H-I,K) by husband/partner (Yes/No)
           → Used as the "sexual violence (ever)" indicator.
   - d107: Experienced any severe violence (D105D-F) by husband/partner (Yes/No)
           → Used as a SECOND indicator (another indicator) instead of a 12-month SV variable.

WHAT THIS SCRIPT DOES:
----------------------
STEP 1: Creates output_gemini/ folder (if it doesn't exist)

STEP 2: Exports a CSV file with INDIVIDUAL-LEVEL data
        - Each row = one woman who answered the indicators
        - Includes her responses, sampling weight, and cluster GPS coordinates

STEP 3: Creates MAP VISUALIZATIONS showing prevalence by cluster
        - CIRCLE COLOR (magma scale) = Fraction of women reporting "Yes"
        - CIRCLE SIZE = Number of women interviewed in that cluster
        - Properly accounts for DHS sampling weights

OUTPUTS:
--------
• output_gemini/women_indicators_gps.csv            (individual-level data)
• output_gemini/women_all_answers_gps.csv           (all women + all variables + GPS)
• output_gemini/indicator1_sv_ever_map.jpg          (map for sexual violence ever)
• output_gemini/indicator2_severe_ever_map.jpg      (map for severe violence ever)

================================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# Toggle for using DHS sampling weights
# TRUE = Use proper statistical weights (recommended for national estimates)
# FALSE = Simple counts (each woman counts equally, ignores sampling design)
USE_WEIGHTS = True

# File paths
ROOT = Path(".")                    # Current directory
DATA_CSV = ROOT / "DATA_CSV"        # Folder containing DHS CSV files
OUTDIR = ROOT / "output_gemini"     # Output folder (created automatically)
OUTDIR.mkdir(parents=True, exist_ok=True)

# Background map image (DRC boundaries)
BACKGROUND = ROOT / "background.png"

# Geographic extent [min_lon, max_lon, min_lat, max_lat]
# These coordinates define the bounding box for DRC
# Must match the extent used when creating background.png
EXTENT = [11.893979235367297, 31.616531541802978, -13.981788316118982, 5.811825978871415]

# Visualization settings
DPI = 300                # Resolution of output image (dots per inch)
BG_ALPHA = 0.55         # Transparency of background map (0=invisible, 1=opaque)
CMAP = "magma"          # Color map for prevalence (magma goes from dark purple to yellow)
CIRCLE_ALPHA = 0.45     # Transparency of cluster circles (0=invisible, 1=opaque)

# =============================================================================
# UTILITY FUNCTIONS (Helper functions for data processing)
# =============================================================================

def find_csv(pattern_parts):
    """
    Find a CSV file in the DATA_CSV directory by searching for pattern matches.
    
    DHS data often comes in nested folders with country/survey codes.
    This function helps locate files even if folder structure varies.
    
    Parameters:
    -----------
    pattern_parts : list of str
        List of substrings to search for (case-insensitive)
        Example: ["cdir81fl"] will find CDIR81FL.csv wherever it is
    
    Returns:
    --------
    Path object to the first matching CSV file, or None if not found
    """
    parts_lower = [p.lower() for p in pattern_parts]
    for p in DATA_CSV.rglob("*.csv"):
        s = str(p).lower()
        if all(part in s for part in parts_lower):
            return p
    return None

def safe_read_csv(path):
    """
    Read a CSV file with settings optimized for large DHS datasets.
    
    The low_memory=False option prevents pandas from guessing data types
    in chunks, which can cause issues with mixed-type columns in DHS data.
    """
    return pd.read_csv(path, low_memory=False)

def coerce_binary_yes_no(series):
    """
    Convert DHS binary variables (Yes/No questions) to numeric 0/1 format.
    
    DHS ENCODING CONVENTIONS:
    - Numeric: 1 = Yes, 0 or 2 = No
    - Text: "Yes"/"No" or variations
    - Missing: blank or special codes
    
    This function standardizes all variations into:
    - 1.0 = Yes (woman experienced the event)
    - 0.0 = No (woman did not experience the event)
    - NaN = Missing/no answer
    
    Parameters:
    -----------
    series : pandas Series
        A column from the DHS dataset (e.g., d105b for sexual violence)
    
    Returns:
    --------
    pandas Series with values as 1.0, 0.0, or NaN
    """
    if series is None:
        return None
    s = series.copy()

    # Handle numeric encoding (most common in DHS data)
    if pd.api.types.is_numeric_dtype(s):
        result = np.where(s == 1, 1.0,
                 np.where((s == 0) | (s == 2), 0.0, np.nan)).astype(float)
        return pd.Series(result, index=s.index)

    # Handle text encoding (sometimes present in downloaded data)
    sl = s.astype("string").str.strip().str.lower()
    is_yes = sl.isin({"1", "yes"}) | sl.str.contains(r"(?:yes|ever)", na=False, regex=True)
    is_no  = sl.isin({"0", "2", "no"}) | sl.str.contains(r"(?:no|non)", na=False, regex=True)
    result = np.where(is_yes, 1.0, np.where(is_no, 0.0, np.nan)).astype(float)
    return pd.Series(result, index=s.index)

def weighted_prop(values, weights):
    """
    Calculate weighted proportion for binary (0/1) data.
    
    WHY WE NEED WEIGHTED PROPORTIONS:
    In DHS surveys, not all women have equal probability of selection.
    For example, if urban areas are oversampled, we need to DOWN-weight
    urban responses to get nationally representative estimates.
    
    FORMULA:
    Weighted proportion = Sum(value_i × weight_i) / Sum(weight_i)
    
    where:
    - value_i is 0 or 1 (No or Yes)
    - weight_i is the sampling weight (v005/1,000,000)
    
    Parameters:
    -----------
    values : array-like
        Binary responses (0 or 1, or NaN for missing)
    weights : array-like
        Sampling weights (already scaled, i.e., v005/1,000,000)
    
    Returns:
    --------
    float : Weighted proportion between 0 and 1, or NaN if no valid data
    
    Example:
    --------
    If 3 women in a cluster answered:
    - Woman 1: Yes (1), weight = 0.5
    - Woman 2: No (0), weight = 2.0
    - Woman 3: Yes (1), weight = 1.0
    
    Weighted proportion = (1×0.5 + 0×2.0 + 1×1.0) / (0.5 + 2.0 + 1.0)
                        = 1.5 / 3.5 = 0.43 (43%)
    """
    v = pd.to_numeric(pd.Series(values), errors="coerce")
    w = pd.to_numeric(pd.Series(weights), errors="coerce")
    
    # Keep only rows where both value and weight are valid
    m = v.notna() & w.notna() & (w > 0)
    
    if not m.any():
        return np.nan
    
    return float((v[m] * w[m]).sum() / w[m].sum())

def scale_sizes(x, smin=5, smax=50):
    """
    Convert sample sizes to circle marker sizes for plotting.
    
    Maps the range of sample sizes to a range of visual circle sizes.
    Uses SQUARE ROOT scaling to create stronger visual differences.
    Larger circles = more women interviewed in that cluster.
    
    Parameters:
    -----------
    x : array-like
        Number of women per cluster
    smin : float
        Minimum circle size in points² (for smallest samples)
    smax : float
        Maximum circle size in points² (for largest samples)
    
    Returns:
    --------
    array-like : Marker sizes scaled between smin and smax
    
    Example:
    --------
    If clusters have 10, 50, and 100 women:
    - 10 women  → small circle (size = smin)
    - 100 women → large circle (size = smax)
    - 50 women  → medium circle (sqrt-scaled for stronger visual difference)
    """
    x = pd.to_numeric(x, errors="coerce").fillna(0).astype(float)
    
    # If no valid data, return default size
    if (x > 0).sum() == 0:
        return np.full_like(x, smin, dtype=float)
    
    # If all values are the same, return middle size
    lo, hi = x.min(), x.max()
    if hi <= lo:
        return np.full_like(x, (smin + smax) / 2.0)
    
    # SQUARE ROOT scaling: creates much stronger visual differences
    # This makes small vs large circles much more distinguishable
    x_normalized = (x - lo) / (hi - lo)  # Normalize to 0-1
    x_sqrt = np.sqrt(x_normalized)        # Apply square root
    return smin + (smax - smin) * x_sqrt

# =============================================================================
# STEP 0: LOCATE INPUT FILES
# =============================================================================

print("[STEP 0/4] Locating DHS data files...")
print("-" * 60)

# Find the IR (Individual Recode) file
# This file contains one row per woman interviewed (age 15-49)
# Filename pattern: CDIR81FL.csv (CD=Congo DR, IR=Individual Recode, 81=survey, FL=flat file)
ir_csv = find_csv(["cdir81fl"])

# Find the GE (Geographic) file  
# This file contains one row per cluster with GPS coordinates
# Filename pattern: CDGE81FL.csv (GE=Geographic data)
ge_csv = find_csv(["cdge81fl"])

# Verify both files were found
if ir_csv is None:
    raise FileNotFoundError(
        "IR CSV not found!\n"
        "Expected: A file matching 'CDIR81FL.csv' in the DATA_CSV/ folder.\n"
        "This file contains individual woman-level survey responses.\n"
        "Please ensure you've downloaded the DHS Individual Recode (IR) dataset."
    )
if ge_csv is None:
    raise FileNotFoundError(
        "GE CSV not found!\n"
        "Expected: A file matching 'CDGE81FL.csv' in the DATA_CSV/ folder.\n"
        "This file contains cluster GPS coordinates.\n"
        "Please ensure you've downloaded the DHS Geographic (GE) dataset."
    )

print(f"✓ Found IR file: {ir_csv.name}")
print(f"  Location: {ir_csv}")
print(f"✓ Found GE file: {ge_csv.name}")
print(f"  Location: {ge_csv}")
print()

# =============================================================================
# STEP 1: CREATE CSV WITH INDIVIDUAL-LEVEL DATA (WOMEN + GPS)
# =============================================================================

print("\n[STEP 1/4] Loading DHS data and creating individual-level CSV...")
print("-" * 60)

# ──────────────────────────────────────────────────────────────────────────
# Load the data files
# ──────────────────────────────────────────────────────────────────────────
print("Loading IR file (individual women's responses)...")
ir = safe_read_csv(ir_csv)
print(f"  → Loaded {len(ir):,} women's records")

print("Loading GE file (cluster GPS coordinates)...")
ge = safe_read_csv(ge_csv)
print(f"  → Loaded {len(ge):,} cluster locations")

# ──────────────────────────────────────────────────────────────────────────
# Standardize column names and data types
# ──────────────────────────────────────────────────────────────────────────
# DHS data sometimes has uppercase or lowercase variable names depending on
# the download format. We standardize everything to lowercase for consistency.

# Cluster ID - this is the key variable that links women to GPS locations
ir["v001"] = pd.to_numeric(ir.get("v001"), errors="coerce").astype("Int64")
ge["DHSCLUST"] = pd.to_numeric(ge.get("DHSCLUST"), errors="coerce").astype("Int64")

# GPS coordinates (WGS84 geographic coordinate system)
# Note: These are intentionally "displaced" by DHS for privacy (up to 5km)
ge["LATNUM"]   = pd.to_numeric(ge.get("LATNUM"), errors="coerce")   # Latitude
ge["LONGNUM"]  = pd.to_numeric(ge.get("LONGNUM"), errors="coerce")  # Longitude

# Handle uppercase variable names if present in the downloaded data
if "D108" in ir.columns: 
    ir.rename(columns={"D108": "d108"}, inplace=True)
if "D107" in ir.columns:
    ir.rename(columns={"D107": "d107"}, inplace=True)
if "V005" in ir.columns: 
    ir.rename(columns={"V005": "v005"}, inplace=True)

# ──────────────────────────────────────────────────────────────────────────
# Extract sexual violence indicators
# ──────────────────────────────────────────────────────────────────────────
# d108: "Experienced any sexual violence (D105H-I,K) by husband/partner"
#        → This captures sexual violence (ever/any)
# d107: "Experienced any severe violence (D105D-F) by husband/partner"
#        → This is used as ANOTHER INDICATOR (not last 12 months sexual violence)
#
# Both are converted from Yes/No to 1/0/NaN format for analysis

print("\nExtracting indicators...")
ever_sv   = coerce_binary_yes_no(ir.get("d108"))    # Sexual violence (ever/any)
other_ind = coerce_binary_yes_no(ir.get("d107"))   # Another indicator (severe violence ever)

# Count how many women answered each question
n_answered_ever = ever_sv.notna().sum()
n_answered_other = other_ind.notna().sum()
print(f"  → {n_answered_ever:,} women answered 'sexual violence (ever)' indicator (d108)")
print(f"  → {n_answered_other:,} women answered 'other indicator' (d107)")

# ──────────────────────────────────────────────────────────────────────────
# Get sampling weights
# ──────────────────────────────────────────────────────────────────────────
# v005 is the woman's sampling weight (must be divided by 1,000,000)
# If weights are not available, we'll use equal weights (1.0) for everyone
if "v005" in ir.columns:
    weights = pd.to_numeric(ir["v005"], errors="coerce") / 1_000_000.0
    print(f"  → Using sampling weights (v005) - mean weight: {weights.mean():.3f}")
else:
    weights = pd.Series(1.0, index=ir.index)
    print("  → WARNING: Sampling weights not found, using equal weights")

# ──────────────────────────────────────────────────────────────────────────
# Create individual-level dataframe
# ──────────────────────────────────────────────────────────────────────────
# This dataframe has ONE ROW PER WOMAN who answered the indicators
print("\nCreating individual-level dataframe...")
women_df = pd.DataFrame({
    "cluster_id": ir["v001"],                      # Which cluster she belongs to
    "sv_ever": ever_sv,                            # Sexual violence (ever/any) (1/0/NaN)
    "other_indicator": other_ind,                  # Another indicator (1/0/NaN)
    "sampling_weight": weights                      # Her sampling weight
})

# Keep only women who answered at least one indicator
women_df = women_df[
    women_df["sv_ever"].notna() | 
    women_df["other_indicator"].notna()
].copy()

print(f"  → Kept {len(women_df):,} women with at least one indicator response")

# ──────────────────────────────────────────────────────────────────────────
# Link women to their cluster's GPS coordinates
# ──────────────────────────────────────────────────────────────────────────
# Each woman gets her cluster's GPS location
# Remember: These GPS points are at the CLUSTER level, not household level
print("\nLinking women to GPS coordinates...")
women_with_gps = women_df.merge(
    ge[["DHSCLUST", "LATNUM", "LONGNUM"]],
    left_on="cluster_id",
    right_on="DHSCLUST",
    how="left"  # Keep all women, even if GPS is missing
)

# Check for missing GPS coordinates
n_missing_gps = women_with_gps["LATNUM"].isna().sum()
if n_missing_gps > 0:
    print(f"  → WARNING: {n_missing_gps:,} women have no GPS coordinates")
print(f"  → {len(women_with_gps) - n_missing_gps:,} women successfully linked to GPS")

# ──────────────────────────────────────────────────────────────────────────
# Save individual-level CSV for analysis
# ──────────────────────────────────────────────────────────────────────────
csv_output = OUTDIR / "women_indicators_gps.csv"
women_with_gps.to_csv(csv_output, index=False)
print(f"\n✓ SAVED: {csv_output}")
print(f"  Total rows: {len(women_with_gps):,} women (only those who answered indicators)")
print(f"  Columns: {', '.join(women_with_gps.columns)}")

# ──────────────────────────────────────────────────────────────────────────
# Create additional CSV with ALL interviewed women and ALL their answers
# ──────────────────────────────────────────────────────────────────────────
# This CSV includes EVERY woman from the IR file, regardless of whether
# she answered the indicators or not.
# All original columns are preserved, and GPS coordinates are added.

print("\nCreating comprehensive dataset with all women and all answers...")

# Merge the full IR dataset with GPS coordinates
all_women_with_gps = ir.merge(
    ge[["DHSCLUST", "LATNUM", "LONGNUM"]],
    left_on="v001",
    right_on="DHSCLUST",
    how="left"  # Keep all women from IR, even if GPS is missing
)

# Save the comprehensive CSV
csv_all_output = OUTDIR / "women_all_answers_gps.csv"
all_women_with_gps.to_csv(csv_all_output, index=False)

print(f"✓ SAVED: {csv_all_output}")
print(f"  Total rows: {len(all_women_with_gps):,} women (all interviewed women)")
print(f"  Columns: {len(all_women_with_gps.columns)} columns (all original survey variables + GPS)")

# Check for missing GPS
n_missing_gps_all = all_women_with_gps["LATNUM"].isna().sum()
if n_missing_gps_all > 0:
    print(f"  → {n_missing_gps_all:,} women have no GPS coordinates")
print()

# =============================================================================
# STEP 2: READ CSV AND AGGREGATE TO CLUSTER LEVEL
# =============================================================================

print("\n[STEP 2/4] Aggregating data by cluster for visualization...")
print("-" * 60)

# ──────────────────────────────────────────────────────────────────────────
# Read the CSV we just created
# ──────────────────────────────────────────────────────────────────────────
# We could continue with the dataframe in memory, but reading from CSV
# demonstrates that the CSV is a standalone product that can be used independently
data = pd.read_csv(csv_output)
print(f"Read {len(data):,} women's records from CSV")

# ──────────────────────────────────────────────────────────────────────────
# Aggregate individual responses to cluster-level statistics
# ──────────────────────────────────────────────────────────────────────────
# We want to create ONE CIRCLE PER CLUSTER on the map
# Each circle will show:
#   - COLOR: What fraction of women in this cluster reported "Yes"?
#   - SIZE:  How many women were interviewed in this cluster?

print("\nAggregating women's responses by cluster...")

if USE_WEIGHTS:
    # ──────────────────────────────────────────────────────────────────────
    # WEIGHTED AGGREGATION (recommended for proper statistical inference)
    # ──────────────────────────────────────────────────────────────────────
    # For each cluster, calculate the WEIGHTED fraction of women who said "Yes"
    # This accounts for the fact that different women have different sampling
    # probabilities and ensures our results are nationally representative
    
    agg = (
        data.groupby("cluster_id")
            .apply(lambda g: pd.Series({
                # Weighted fraction for sexual violence (ever)
                "fraction_sv_ever": weighted_prop(g["sv_ever"], 
                                                 g["sampling_weight"]),
                # Weighted fraction for the other indicator
                "fraction_other": weighted_prop(g["other_indicator"], 
                                               g["sampling_weight"]),
                # Total number of women in this cluster (for circle size)
                "n_women": len(g),
                # GPS coordinates (same for all women in cluster)
                "LATNUM": g["LATNUM"].iloc[0] if len(g) > 0 else np.nan,
                "LONGNUM": g["LONGNUM"].iloc[0] if len(g) > 0 else np.nan
            }), include_groups=False)
            .reset_index()
    )
    title_text = "Prevalence (Weighted)"
    subtitle_text = "Each woman weighted by sampling probability (v005)"
    
else:
    # ──────────────────────────────────────────────────────────────────────
    # UNWEIGHTED AGGREGATION (simple percentages, not nationally representative)
    # ──────────────────────────────────────────────────────────────────────
    # For each cluster, calculate the simple percentage who said "Yes"
    # This treats all women equally, ignoring sampling design
    # Use only for exploratory analysis, not for published estimates
    
    agg = (
        data.groupby("cluster_id")
            .agg(
                # Simple mean of 0/1 values = fraction who said "Yes"
                fraction_sv_ever=("sv_ever", "mean"),
                fraction_other=("other_indicator", "mean"),
                n_women=("cluster_id", "size"),  # Count of women
                LATNUM=("LATNUM", "first"),      # GPS (same for all)
                LONGNUM=("LONGNUM", "first")
            )
            .reset_index()
    )
    title_text = "Prevalence (Unweighted)"
    subtitle_text = "Simple percentages, not accounting for sampling design"

print(f"  → Created summary for {len(agg):,} clusters")

# Report overall statistics
overall_frac = agg["fraction_sv_ever"].mean()
print(f"  → Average sexual violence (ever) prevalence across clusters: {overall_frac:.1%}")

# ──────────────────────────────────────────────────────────────────────────
# Filter to clusters within the map extent and with valid data
# ──────────────────────────────────────────────────────────────────────────
# Only plot clusters that:
#   1. Have GPS coordinates within our map boundaries (EXTENT)
#   2. Have valid data for at least the first indicator

plot_df = agg[
    (agg["LONGNUM"] >= EXTENT[0]) & (agg["LONGNUM"] <= EXTENT[1]) &  # Within longitude bounds
    (agg["LATNUM"] >= EXTENT[2]) & (agg["LATNUM"] <= EXTENT[3]) &    # Within latitude bounds
    agg["fraction_sv_ever"].notna()                                  # Has valid data
].copy()

print(f"\nFiltered to {len(plot_df):,} clusters within map extent with valid data")

# Check if we have data to plot
if len(plot_df) == 0:
    raise ValueError(
        "No clusters to plot! Check that:\n"
        "  1. GPS coordinates are valid\n"
        "  2. EXTENT matches your study area\n"
        "  3. Women have answered the indicators"
    )

# Report the range of prevalence values we'll be plotting
min_prev = plot_df["fraction_sv_ever"].min()
max_prev = plot_df["fraction_sv_ever"].max()
print(f"  → Sexual violence (ever, d108) ranges from {min_prev:.1%} to {max_prev:.1%}")
print()

# =============================================================================
# STEP 3: CREATE MAP VISUALIZATIONS (TWO SEPARATE PLOTS)
# =============================================================================

print("\n[STEP 3/4] Creating map visualizations...")
print("-" * 60)

# ──────────────────────────────────────────────────────────────────────────
# Load background map image
# ──────────────────────────────────────────────────────────────────────────
if BACKGROUND.exists():
    bg = Image.open(BACKGROUND).convert("RGBA")
    has_bg = True
    print(f"✓ Loaded background map: {BACKGROUND.name}")
else:
    has_bg = False
    print(f"⚠ Background map not found at {BACKGROUND}")
    print("  → Will create maps without background")

# ──────────────────────────────────────────────────────────────────────────
# Set up shared color scale for both maps
# ──────────────────────────────────────────────────────────────────────────

vmin = 0.0  # Minimum prevalence (0%)

# Find the maximum across BOTH variables for consistent scale
vmax_sv = float(np.nanmax(plot_df["fraction_sv_ever"].to_numpy()))
vmax_other = float(np.nanmax(plot_df["fraction_other"].to_numpy()))
vmax = max(vmax_sv, vmax_other)

# Safety check: ensure vmax is valid
if not np.isfinite(vmax) or vmax <= 0:
    vmax = 1.0  # Default to 100% if data issues

print(f"\nShared color scale (magma colormap):")
print(f"  → Dark purple = 0% prevalence")
print(f"  → Bright yellow = {vmax:.1%} prevalence (max across both maps)")

# ──────────────────────────────────────────────────────────────────────────
# Set up size scale for sample size (shared across both maps)
# ──────────────────────────────────────────────────────────────────────────
# CHANGE (ONLY): Use raw sample-size proportional marker areas so differences are visible.
# Matplotlib's scatter uses marker AREA in points^2, so we map n_women directly
# into a wide area range for strong visual differences.
sizes = pd.to_numeric(plot_df["n_women"], errors="coerce").fillna(0).astype(float)

# Robust scaling using percentiles to avoid one extreme cluster dominating the scale
p05 = float(np.nanpercentile(sizes.to_numpy(), 5))
p95 = float(np.nanpercentile(sizes.to_numpy(), 95))
if not np.isfinite(p05): p05 = sizes.min()
if not np.isfinite(p95): p95 = sizes.max()
if p95 <= p05:
    p05, p95 = sizes.min(), sizes.max()
    if p95 <= p05:
        p05, p95 = 0.0, max(1.0, float(sizes.max()))

# Map sample sizes to marker AREAS (points^2) with a wide range for clear differences
S_MIN = 25.0
S_MAX = 600.0
sizes = (sizes.clip(p05, p95) - p05) / (p95 - p05)
sizes = S_MIN + (S_MAX - S_MIN) * sizes

min_n = plot_df["n_women"].min()
max_n = plot_df["n_women"].max()
print(f"\nCircle size scale (area proportional to sample size with percentile clipping):")
print(f"  → Smallest clusters = {min_n:.0f} women")
print(f"  → Largest clusters = {max_n:.0f} women")

# ──────────────────────────────────────────────────────────────────────────
# PLOT 1: Sexual violence (ever)
# ──────────────────────────────────────────────────────────────────────────
print("\nCreating Plot 1: Sexual violence (ever) prevalence...")

# Filter to clusters with valid SV data
df_ever = plot_df.dropna(subset=["fraction_sv_ever", "LATNUM", "LONGNUM"]).copy()

fig1, ax1 = plt.subplots(figsize=(14, 11))

if has_bg:
    ax1.imshow(bg, extent=EXTENT, origin="upper", alpha=BG_ALPHA, zorder=0)

# Plot circles for "sexual violence ever"
sc1 = ax1.scatter(
    df_ever["LONGNUM"],             # X position (longitude)
    df_ever["LATNUM"],              # Y position (latitude)
    c=df_ever["fraction_sv_ever"],  # COLOR = prevalence (0 to 1)
    s=sizes.loc[df_ever.index],     # SIZE = number of women
    cmap=CMAP,                      # "magma" color scheme
    vmin=vmin,                      # Color scale minimum
    vmax=vmax,                      # Color scale maximum (shared)
    alpha=CIRCLE_ALPHA,             # Circle transparency
    linewidths=0.5,                 # Black edge around circles
    edgecolors="black",
    zorder=2                        # Draw on top of background
)

ax1.set_xlim(EXTENT[0], EXTENT[1])
ax1.set_ylim(EXTENT[2], EXTENT[3])
ax1.set_aspect("equal", "box")
ax1.set_axis_off()
ax1.set_title(
    f"{title_text}\nSexual Violence (Ever) — d108\n{subtitle_text}",
    fontsize=14,
    fontweight="bold",
    pad=20
)

# Add colorbar
cbar1 = plt.colorbar(sc1, ax=ax1, orientation="vertical", fraction=0.03, pad=0.02)
cbar1.set_label(
    "Fraction Reporting Sexual Violence (Ever)\n(weighted by sampling probability)", 
    rotation=90, 
    fontsize=11,
    labelpad=15
)
cbar1.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))

# Add size legend
legend_samples_1 = [
    int(min_n),                          # Smallest
    int((min_n + max_n) / 2),           # Medium
    int(max_n)                           # Largest
]

handles1 = []
for n in legend_samples_1:
    # Convert sample count n into the same marker AREA scale used above
    nn = float(n)
    nn = min(max(nn, p05), p95)
    t = 0.0 if p95 <= p05 else (nn - p05) / (p95 - p05)
    size_val = S_MIN + (S_MAX - S_MIN) * t
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

# Save first plot
fig1_output = OUTDIR / "indicator1_sv_ever_map.jpg"
plt.savefig(fig1_output, dpi=DPI, bbox_inches="tight")
plt.close()

print(f"✓ SAVED: {fig1_output}")
print(f"  → {len(df_ever)} clusters with sexual violence (ever) data")

# ──────────────────────────────────────────────────────────────────────────
# PLOT 2: Other indicator (not SV last 12 months)
# ──────────────────────────────────────────────────────────────────────────
print("\nCreating Plot 2: Other indicator prevalence...")

# Filter to clusters with valid OTHER indicator data
df_other = plot_df.dropna(subset=["fraction_other", "LATNUM", "LONGNUM"]).copy()

fig2, ax2 = plt.subplots(figsize=(14, 11))

if has_bg:
    ax2.imshow(bg, extent=EXTENT, origin="upper", alpha=BG_ALPHA, zorder=0)

# Plot circles for the OTHER indicator
sc2 = ax2.scatter(
    df_other["LONGNUM"],           # X position (longitude)
    df_other["LATNUM"],            # Y position (latitude)
    c=df_other["fraction_other"],  # COLOR = prevalence (0 to 1)
    s=sizes.loc[df_other.index],   # SIZE = number of women (same scale)
    cmap=CMAP,                     # "magma" color scheme (same)
    vmin=vmin,                     # Color scale minimum (same)
    vmax=vmax,                     # Color scale maximum (same)
    alpha=CIRCLE_ALPHA,            # Circle transparency
    linewidths=0.5,                # Black edge around circles
    edgecolors="black",
    zorder=2                       # Draw on top of background
)

ax2.set_xlim(EXTENT[0], EXTENT[1])
ax2.set_ylim(EXTENT[2], EXTENT[3])
ax2.set_aspect("equal", "box")
ax2.set_axis_off()
ax2.set_title(
    f"{title_text}\nOther Indicator (Severe violence) — d107\n{subtitle_text}",
    fontsize=14,
    fontweight="bold",
    pad=20
)

# Add colorbar
cbar2 = plt.colorbar(sc2, ax=ax2, orientation="vertical", fraction=0.03, pad=0.02)
cbar2.set_label(
    "Fraction Reporting Other Indicator\n(weighted by sampling probability)", 
    rotation=90, 
    fontsize=11,
    labelpad=15
)
cbar2.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.0%}'))

# Add size legend
legend_samples_2 = [
    int(min_n),                          # Smallest
    int((min_n + max_n) / 2),           # Medium
    int(max_n)                           # Largest
]

handles2 = []
for n in legend_samples_2:
    # Convert sample count n into the same marker AREA scale used above
    nn = float(n)
    nn = min(max(nn, p05), p95)
    t = 0.0 if p95 <= p05 else (nn - p05) / (p95 - p05)
    size_val = S_MIN + (S_MAX - S_MIN) * t
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

# Save second plot
fig2_output = OUTDIR / "indicator2_severe_ever_map.jpg"
plt.savefig(fig2_output, dpi=DPI, bbox_inches="tight")
plt.close()

print(f"✓ SAVED: {fig2_output}")
print(f"  → {len(df_other)} clusters with other-indicator data")
print(f"\n✓ Both map visualizations created (resolution: {DPI} DPI)")
print()

# =============================================================================
# STEP 4: SUMMARY AND INTERPRETATION GUIDE
# =============================================================================

print("\n[STEP 4/4] Analysis complete! Summary of outputs:")
print("=" * 60)
print(f"\n📁 OUTPUT FOLDER: {OUTDIR}/")
print()
print(f"📊 CSV FILES (2 datasets):")
print(f"   1) women_indicators_gps.csv")
print(f"      → {len(women_with_gps):,} women (only those who answered indicators)")
print(f"      → Contains: cluster_id, indicators, weights, GPS")
print(f"      → Use for: Indicator-specific analysis")
print(f"   2) women_all_answers_gps.csv")
print(f"      → {len(ir):,} women (ALL interviewed women)")
print(f"      → Contains: All original survey variables + GPS coordinates")
print(f"      → Use for: Comprehensive analysis, linking indicators to other variables")
print()
print(f"🗺️  MAP FILES (2 separate visualizations):")
print(f"   1) indicator1_sv_ever_map.jpg")
print(f"      → Sexual violence (ever) using d108")
print(f"      → {len(plot_df.dropna(subset=['fraction_sv_ever'])):,} clusters visualized")
print(f"   2) indicator2_severe_ever_map.jpg")
print(f"      → Other indicator (severe violence) using d107")
print(f"      → {len(plot_df.dropna(subset=['fraction_other'])):,} clusters visualized")
print(f"   → Each circle = one cluster (village/neighborhood)")
print(f"   → Circle COLOR = % of women reporting 'Yes'")
print(f"   → Circle SIZE = number of women interviewed (area-scaled)")
print(f"   → Resolution: {DPI} DPI (publication quality)")
print()
print("⚙️  METHODOLOGY:")
if USE_WEIGHTS:
    print(f"   → Weighted analysis using DHS sampling weights (v005)")
    print(f"   → Results are nationally representative")
    print(f"   → Appropriate for publication and policy use")
else:
    print(f"   → Unweighted analysis (simple percentages)")
    print(f"   → NOT nationally representative")
    print(f"   → Use only for exploratory analysis")
print()
print("=" * 60)
print("\n📖 HOW TO INTERPRET THE MAPS:")
print("-" * 60)
print()
print("COLOR INTERPRETATION (Magma colormap):")
print("  • Dark purple/black areas = Low prevalence (few women reported 'Yes')")
print("  • Yellow/bright areas = High prevalence (many women reported 'Yes')")
print("  • Each color represents the FRACTION (0-100%) of women in that cluster")
print("  • MAP 1: Sexual violence (ever) using d108")
print("  • MAP 2: Other indicator (severe violence) using d107")
print()
print("SIZE INTERPRETATION:")
print("  • Larger circles = More women interviewed = More reliable estimate")
print("  • Smaller circles = Fewer women = Less reliable (more sampling uncertainty)")
print("  • Sample sizes per cluster typically range from 10-100+ women")
print("  • Marker AREA is scaled (with percentile clipping) so differences are visible")
print()
print("⚠️  IMPORTANT CAVEATS:")
print("-" * 60)
print("  1. GPS DISPLACEMENT: Cluster locations are intentionally shifted")
print("     by up to 5km (10km for rural areas) to protect privacy.")
print("     Do NOT use these coordinates for exact location mapping.")
print()
print("  2. CLUSTER-LEVEL DATA: Each circle represents aggregated data")
print("     from multiple women in a geographic area, not individual households.")
print()
print("  3. UNDERREPORTING: Sexual violence is sensitive and likely")
print("     underreported. These numbers represent MINIMUM prevalence.")
print()
print("  4. SAMPLING UNCERTAINTY: Smaller clusters (smaller circles)")
print("     have more uncertainty. A cluster with 10 women showing 50%")
print("     prevalence is less reliable than one with 100 women.")
print()
print("  5. REPRESENTATIVENESS: This analysis is only as representative")
print("     as the DHS sampling design. Always cite the survey properly.")
print()
print("=" * 60)
print("\n✅ Analysis complete! Check the output_gemini/ folder for results.")
print()
"
print("\n✅ Analysis complete! Check the output_gemini/ folder for results.")
print()
