#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DHS Data Visualization Script - DRC 2023-24 Survey (GENERALIZED)
================================================================================

WHAT THIS SCRIPT DOES:
----------------------
This script creates MAP visualizations from the processed DHS data for ANY
set of variables specified in the configuration.

1. MAP VISUALIZATIONS: Shows prevalence of violence indicators by cluster
   - Circle COLOR = Prevalence rate (% of women reporting "Yes")
   - Circle SIZE = Sample size (number of women interviewed)
   - Creates separate maps for each indicator in the variable list

INPUTS:
-------
- DATA/DHS/women_all_answers_gps.csv (created by convert.py)
- DATA/background.png (optional background map of DRC)

OUTPUTS:
--------
For each variable in VARIABLES_TO_PLOT:
- 02a_{variable_name}_map.jpg (map of prevalence)
- 02a_{variable_name}_map.csv (underlying data)

All outputs are saved in the output/ directory (same level as src/).

USAGE:
------
Run from the project root directory:
    python src/02a_IPV_overview_fig.py

Or from the src directory:
    python 02a_IPV_overview_fig.py

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
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

# COLOR SCHEME OPTIONS
# Choose one of the following color schemes for the map:
# 1. "rocket"   - Rocket (dark blue to orange/red) - RECOMMENDED
# 2. "magma"    - Magma (dark purple to yellow)
# 3. "viridis"  - Viridis (perceptually uniform, colorblind-friendly)
# 4. "plasma"   - Plasma (perceptually uniform, high contrast)
# 5. "RdYlGn_r" - Red-Yellow-Green (reversed: high=red, low=green)
COLOR_SCHEME = "rocket"

# VARIABLES TO VISUALIZE
# Define which DHS variables to create maps for
VARIABLES_TO_PLOT = ["D111", "D104", "D106", "D108"]

# VARIABLE DESCRIPTIONS
# Maps variable codes to their full descriptions for figure titles
VARIABLE_DESCRIPTIONS = {
    "D111": "Any IPV",
    "D104": "Emotional IPV",
    "D106": "Physical IPV",
    "D108": "Sexual IPV",
}

# Determine paths relative to this script
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Input files
DATA_DIR = PROJECT_ROOT / "DATA"
DHS_DIR = DATA_DIR / "DHS"
INPUT_CSV = DHS_DIR / "women_all_answers_gps.csv"
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
BG_ALPHA = 0.9              # Transparency of background map (0=invisible, 1=opaque)
CIRCLE_ALPHA = 0.2         # Transparency of cluster circles

# Circle size range (in points²)
SIZE_MIN = 25.0             # Minimum marker area
SIZE_MAX = 300.0            # Maximum marker area

# Legend position for circle sizes
# Options: "on_map", "outside", "none"
LEGEND_POSITION = "outside"  # "on_map" = inside map (lower right), "outside" = outside map area, "none" = no legend

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
        Sampling weights (already scaled, i.e., d005/1,000,000)

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


def aggregate_variable(data, variable_name, use_weights=True):
    """
    Aggregate a single variable to cluster level.
    
    For DHS data, this function:
    1. Filters to women who provided a valid answer (0 or 1) to the variable
    2. Computes the fraction who answered "Yes" (1)
    3. Uses DHS domestic violence weights (d005/1,000,000) if requested
    
    Parameters:
    -----------
    data : pd.DataFrame
        Individual-level data with v001 (cluster), GPS coords, and variables
    variable_name : str
        Name of the variable column to aggregate (e.g., 'D111', 'D104')
    use_weights : bool
        Whether to use weighted aggregation
    
    Returns:
    --------
    pd.DataFrame : Cluster-level aggregated data
    """
    # Convert variable name to lowercase for DHS data
    var_lower = variable_name.lower()
    
    if var_lower not in data.columns:
        raise ValueError(f"Variable '{variable_name}' (or '{var_lower}') not found in data. "
                         f"Available columns: {list(data.columns)}")
    
    # Filter to women who answered this variable (0 or 1, not NaN)
    # This ensures we only consider those who were asked and answered the question
    data_with_response = data[data[var_lower].notna()].copy()
    
    if len(data_with_response) == 0:
        # No valid responses for this variable
        return pd.DataFrame({
            'cluster_id': [],
            f'fraction_{variable_name}': [],
            'n_women': [],
            'LATNUM': [],
            'LONGNUM': []
        })
    
    if use_weights:
        # Use DHS domestic violence weights (d005/1,000,000)
        # Create sampling_weight column if it doesn't exist
        if 'sampling_weight' not in data_with_response.columns:
            if 'd005' in data_with_response.columns:
                data_with_response['sampling_weight'] = (
                    pd.to_numeric(data_with_response['d005'], errors='coerce') / 1_000_000.0
                )
            else:
                # Fallback to equal weights
                data_with_response['sampling_weight'] = 1.0
        
        agg = (
            data_with_response.groupby("v001")
                .apply(lambda g: pd.Series({
                    f"fraction_{variable_name}": weighted_prop(
                        g[var_lower],
                        g["sampling_weight"]
                    ),
                    "n_women": len(g),
                    "LATNUM": g["LATNUM"].iloc[0] if len(g) > 0 else np.nan,
                    "LONGNUM": g["LONGNUM"].iloc[0] if len(g) > 0 else np.nan
                }), include_groups=False)
                .reset_index()
                .rename(columns={'v001': 'cluster_id'})
        )
    else:
        agg = (
            data_with_response.groupby("v001")
                .agg(**{
                    f"fraction_{variable_name}": (var_lower, "mean"),
                    "n_women": ("v001", "size"),
                    "LATNUM": ("LATNUM", "first"),
                    "LONGNUM": ("LONGNUM", "first")
                })
                .reset_index()
                .rename(columns={'v001': 'cluster_id'})
        )
    
    return agg


def create_map_visualization(plot_df, variable_name, variable_description, 
                              sizes, vmin, vmax, title_suffix, subtitle,
                              output_path, output_csv_path, bg=None):
    """
    Create a single map visualization for a variable.
    
    Parameters:
    -----------
    plot_df : pd.DataFrame
        Cluster-level data to plot
    variable_name : str
        Variable code (e.g., 'D111')
    variable_description : str
        Human-readable description (e.g., 'Any IPV')
    sizes : pd.Series
        Marker sizes for each cluster
    vmin, vmax : float
        Color scale bounds
    title_suffix : str
        'Weighted' or 'Unweighted'
    subtitle : str
        Subtitle explaining the weighting
    output_path : Path
        Where to save the JPG
    output_csv_path : Path
        Where to save the CSV
    bg : PIL.Image or None
        Background image
    """
    fraction_col = f"fraction_{variable_name}"
    
    # Filter to valid data
    df_plot = plot_df.dropna(subset=[fraction_col, "LATNUM", "LONGNUM"]).copy()
    
    if len(df_plot) == 0:
        print(f"  ⚠ No valid data for {variable_name}, skipping...")
        return 0
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 11))
    
    # Add background image
    if bg is not None:
        ax.imshow(bg, extent=EXTENT, origin="upper", alpha=BG_ALPHA, zorder=0)
    
    # Plot cluster circles
    sc = ax.scatter(
        df_plot["LONGNUM"],
        df_plot["LATNUM"],
        c=df_plot[fraction_col],
        s=sizes.loc[df_plot.index],
        cmap=COLOR_SCHEME,
        vmin=vmin,
        vmax=vmax,
        alpha=CIRCLE_ALPHA,
        linewidths=0.5,
        edgecolors="black",
        zorder=2
    )
    
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_aspect("equal", "box")
    ax.set_axis_off()
    ax.set_title(
        f"Prevalence ({title_suffix})\n{variable_description} — {variable_name}\n{subtitle}",
        fontsize=14,
        fontweight="bold",
        pad=20
    )
    
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, orientation="vertical",
                        fraction=0.03, pad=0.02)
    cbar.set_label(
        f"Fraction Reporting {variable_description}\n(weighted by DV sampling probability)",
        rotation=90,
        fontsize=11,
        labelpad=15
    )
    cbar.ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'{x:.0%}')
    )
    
    # Add size legend based on LEGEND_POSITION setting
    legend_obj = None
    if LEGEND_POSITION != "none":
        min_n = plot_df["n_women"].min()
        max_n = plot_df["n_women"].max()
        legend_samples = [int(min_n), int((min_n + max_n) / 2), int(max_n)]
        
        # Get percentiles for size scaling (same as used for sizes calculation)
        sizes_raw = pd.to_numeric(plot_df["n_women"], errors="coerce").fillna(0).astype(float)
        p05 = float(np.nanpercentile(sizes_raw.to_numpy(), 5))
        p95 = float(np.nanpercentile(sizes_raw.to_numpy(), 95))
        if not np.isfinite(p05):
            p05 = sizes_raw.min()
        if not np.isfinite(p95):
            p95 = sizes_raw.max()
        if p95 <= p05:
            p05, p95 = sizes_raw.min(), sizes_raw.max()
            if p95 <= p05:
                p05, p95 = 0.0, max(1.0, float(sizes_raw.max()))
        
        handles = []
        for n in legend_samples:
            nn = float(n)
            nn = min(max(nn, p05), p95)
            t = 0.0 if p95 <= p05 else (nn - p05) / (p95 - p05)
            size_val = SIZE_MIN + (SIZE_MAX - SIZE_MIN) * t
            handles.append(
                plt.scatter([], [], s=size_val, color="gray", alpha=0.6,
                            edgecolors="black", linewidths=0.5,
                            label=f"{n} women")
            )
        
        if LEGEND_POSITION == "on_map":
            # Legend inside the map (original behavior)
            legend_obj = ax.legend(
                handles=handles,
                scatterpoints=1,
                frameon=True,
                labelspacing=1.5,
                title="Number of Women\nInterviewed per Cluster",
                loc="lower right",
                fontsize=10,
                title_fontsize=11
            )
        elif LEGEND_POSITION == "outside":
            # Legend outside the map (positioned lower to avoid colorbar overlap)
            legend_obj = ax.legend(
                handles=handles,
                scatterpoints=1,
                frameon=True,
                labelspacing=1.5,
                title="Number of Women\nInterviewed per Cluster",
                loc="lower left",
                bbox_to_anchor=(1.15, 0.0),
                fontsize=10,
                title_fontsize=11
            )
    
    # Save figure
    if LEGEND_POSITION == "outside" and legend_obj is not None:
        # Include legend in the saved area
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight", 
                    bbox_extra_artists=[legend_obj])
    else:
        plt.savefig(output_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    
    # Save CSV
    pd.DataFrame({
        "LONGNUM": df_plot["LONGNUM"],
        "LATNUM": df_plot["LATNUM"],
        "value": df_plot[fraction_col],
        "size": sizes.loc[df_plot.index],
        "n_women": df_plot["n_women"],
        "cluster_id": df_plot["cluster_id"],
    }).to_csv(output_csv_path, index=False)
    
    return len(df_plot)


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    """
    Main function that orchestrates the MAP visualization workflow.
    """

    print("\n" + "=" * 70)
    print("DHS DATA VISUALIZATION (GENERALIZED) - DRC 2023-24 Survey")
    print("=" * 70)
    print(f"\nVariables to visualize: {', '.join(VARIABLES_TO_PLOT)}")

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
    # STEP 2: LOAD DATA
    # -------------------------------------------------------------------------
    print("\n[STEP 2/4] Loading data...")
    print("-" * 70)

    # Load the processed CSV
    print(f"\nReading: {INPUT_CSV.name}")
    data = pd.read_csv(INPUT_CSV)
    print(f"  → Loaded {len(data):,} women's records (all interviewed women)")
    
    # Verify cluster ID column exists
    if 'v001' not in data.columns:
        print(f"\nERROR: 'v001' (cluster ID) column not found in data")
        print(f"Available columns: {list(data.columns)}")
        sys.exit(1)
    
    # Verify all variables exist (checking lowercase versions)
    missing_vars = []
    for v in VARIABLES_TO_PLOT:
        if v.lower() not in data.columns:
            missing_vars.append(v)
    
    if missing_vars:
        print(f"\nERROR: Variables not found in data: {missing_vars}")
        print(f"Note: Looking for lowercase versions (e.g., 'd111' not 'D111')")
        print(f"Available columns: {list(data.columns)}")
        sys.exit(1)
    
    print(f"  → All {len(VARIABLES_TO_PLOT)} variables found in data")

    # -------------------------------------------------------------------------
    # STEP 3: AGGREGATE DATA FOR ALL VARIABLES
    # -------------------------------------------------------------------------
    print("\n[STEP 3/4] Aggregating data by cluster...")
    print("-" * 70)

    if USE_WEIGHTS:
        print("  Using WEIGHTED aggregation (recommended)")
        print("  → Accounts for DHS domestic violence sampling design")
        title_suffix = "Weighted"
        subtitle = "Weighted by DHS domestic violence sampling probability (d005)"
    else:
        print("  Using UNWEIGHTED aggregation")
        print("  → Simple percentages (not nationally representative)")
        title_suffix = "Unweighted"
        subtitle = "Simple percentages (not accounting for sampling design)"

    # Aggregate all variables at once
    all_agg_data = []
    for var in VARIABLES_TO_PLOT:
        var_lower = var.lower()
        print(f"\n  Processing {var} ({VARIABLE_DESCRIPTIONS.get(var, 'Unknown')})...")
        
        # Count women who answered this variable
        n_answered = data[var_lower].notna().sum()
        n_yes = (data[var_lower] == 1.0).sum()
        pct_yes = (n_yes / n_answered * 100) if n_answered > 0 else 0
        print(f"    → {n_answered:,} women answered this question")
        print(f"    → {n_yes:,} answered 'Yes' ({pct_yes:.1f}%)")
        
        agg = aggregate_variable(data, var, USE_WEIGHTS)
        
        # Keep only the fraction column and merge with other data
        fraction_col = f"fraction_{var}"
        if len(all_agg_data) == 0:
            # First variable - include all columns
            all_agg_data.append(agg)
        else:
            # Subsequent variables - only add the fraction column
            all_agg_data[0] = all_agg_data[0].merge(
                agg[["cluster_id", fraction_col]], 
                on="cluster_id", 
                how="outer"
            )
        
        mean_val = agg[fraction_col].mean()
        print(f"    → Cluster-level mean prevalence: {mean_val:.1%}")
        print(f"    → {len(agg):,} clusters with data for this variable")
    
    # Consolidated aggregated data
    agg_consolidated = all_agg_data[0]
    print(f"\n  → Consolidated data: {len(agg_consolidated):,} total clusters")

    # Filter to valid data within map extent
    print("\n  Filtering to clusters within map extent...")
    
    # Require at least one valid fraction value
    fraction_cols = [f"fraction_{v}" for v in VARIABLES_TO_PLOT]
    has_any_data = agg_consolidated[fraction_cols].notna().any(axis=1)
    
    plot_df = agg_consolidated[
        (agg_consolidated["LONGNUM"] >= EXTENT[0]) & 
        (agg_consolidated["LONGNUM"] <= EXTENT[1]) &
        (agg_consolidated["LATNUM"] >= EXTENT[2]) & 
        (agg_consolidated["LATNUM"] <= EXTENT[3]) &
        has_any_data
    ].copy()

    print(f"    → {len(plot_df):,} clusters within map extent with valid data")

    if len(plot_df) == 0:
        print("\nERROR: No clusters to plot!")
        print("Check that:")
        print("  1. GPS coordinates are valid")
        print("  2. EXTENT matches your study area")
        print("  3. Women have answered the indicators")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # STEP 4: CREATE MAP VISUALIZATIONS
    # -------------------------------------------------------------------------
    print("\n[STEP 4/4] Creating map visualizations...")
    print("-" * 70)

    # Load background image if available
    if has_background:
        bg = Image.open(BACKGROUND_IMAGE).convert("RGBA")
        print(f"✓ Loaded background image: {bg.size[0]}×{bg.size[1]} pixels")
    else:
        bg = None

    # Determine shared color scale across all maps
    vmin = 0.0
    vmax = 0.0
    for var in VARIABLES_TO_PLOT:
        fraction_col = f"fraction_{var}"
        var_max = float(np.nanmax(plot_df[fraction_col].to_numpy()))
        if np.isfinite(var_max):
            vmax = max(vmax, var_max)
    
    if vmax <= 0:
        vmax = 1.0

    print(f"\nColor scale ({COLOR_SCHEME} colormap):")
    print(f"  → Low values = 0% prevalence")
    print(f"  → High values = {vmax:.1%} prevalence (max across all variables)")

    # Scale marker sizes using percentile clipping
    sizes = pd.to_numeric(plot_df["n_women"], errors="coerce").fillna(0).astype(float)

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

    sizes_clipped = sizes.clip(p05, p95)
    sizes_normalized = (sizes_clipped - p05) / (p95 - p05)
    sizes = SIZE_MIN + (SIZE_MAX - SIZE_MIN) * sizes_normalized

    min_n = plot_df["n_women"].min()
    max_n = plot_df["n_women"].max()

    print(f"\nCircle sizes (area-scaled with percentile clipping):")
    print(f"  → Smallest cluster: {min_n:.0f} women")
    print(f"  → Largest cluster: {max_n:.0f} women")
    print(f"  → Marker area range: {SIZE_MIN:.0f} to {SIZE_MAX:.0f} points²")
    print(f"  → Legend position: {LEGEND_POSITION}")

    # Create maps for each variable
    output_files = []
    
    for i, var in enumerate(VARIABLES_TO_PLOT, 1):
        var_desc = VARIABLE_DESCRIPTIONS.get(var, var)
        print(f"\n[{i}/{len(VARIABLES_TO_PLOT)}] Creating map for {var} ({var_desc})...")
        
        output_jpg = OUTPUT_DIR / f"02a_{var}_map.jpg"
        output_csv = OUTPUT_DIR / f"02a_{var}_map.csv"
        
        n_plotted = create_map_visualization(
            plot_df=plot_df,
            variable_name=var,
            variable_description=var_desc,
            sizes=sizes,
            vmin=vmin,
            vmax=vmax,
            title_suffix=title_suffix,
            subtitle=subtitle,
            output_path=output_jpg,
            output_csv_path=output_csv,
            bg=bg
        )
        
        if n_plotted > 0:
            print(f"  ✓ Saved: {output_jpg.name}")
            print(f"  ✓ Saved: {output_csv.name}")
            print(f"    → {n_plotted:,} clusters plotted")
            output_files.append((var, var_desc, output_jpg, output_csv, n_plotted))

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\n📊 OUTPUT FILES (saved in {OUTPUT_DIR.name}/):\n")
    
    for i, (var, desc, jpg, csv, n) in enumerate(output_files, 1):
        print(f"  {2*i-1}. {jpg.name}")
        print(f"     → Map of {desc} ({var}) prevalence")
        print(f"     → {n:,} clusters visualized")
        print(f"\n  {2*i}. {csv.name}")
        print(f"     → Data behind the {desc} map (location/value/size)\n")

    print(f"⚙️  Settings:")
    print(f"  → Variables: {', '.join(VARIABLES_TO_PLOT)}")
    print(f"  → Weighting: {title_suffix}")
    print(f"  → Resolution: {DPI} DPI")
    print(f"  → Color scheme: {COLOR_SCHEME}")
    print(f"  → Background map: {'Yes' if has_background else 'No'}")
    print(f"  → Legend position: {LEGEND_POSITION}")

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
