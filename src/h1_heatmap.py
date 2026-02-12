#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DHS Heatmap Visualization Script - DRC 2023-24 Survey
================================================================================

WHAT THIS SCRIPT DOES:
----------------------
This script creates HEATMAP visualizations to explore the relationship between
intimate partner violence (IPV) prevalence and socioeconomic factors.

PURPOSE:
Tests hypothesis H1: The prevalence of intimate partner violence (IPV) 
decreases monotonically with household wealth and women's educational attainment.

VISUALIZATION:
- X-axis: Household wealth quintile (V190)
- Y-axis: Women's educational attainment (V149)
- Color: Prevalence of IPV (% of women reporting "Yes")
- Uses weighted aggregation with D005 (domestic violence weights)

INPUTS:
-------
- DATA/DHS/women_all_answers_gps.csv (created by convert.py)

OUTPUTS:
--------
For each variable in VARIABLES_TO_PLOT:
- h1_{variable}_heatmap.csv (aggregated data with wealth × education cells)
- h1_{variable}_heatmap.png (heatmap visualization - PNG format)
- h1_{variable}_heatmap.pdf (heatmap visualization - PDF format)

All outputs are saved in the output/ directory (same level as src/).

USAGE:
------
Run from the project root directory:
    python src/h1_heatmap.py

Or from the src directory:
    python h1_heatmap.py

REQUIREMENTS:
-------------
- pandas (data manipulation)
- numpy (numerical operations)
- matplotlib (plotting)
- seaborn (statistical visualization)

Install with:
    pip install pandas numpy matplotlib seaborn

================================================================================
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURATION
# =============================================================================

# COLOR SCHEME OPTIONS
# Choose one of the following color schemes for the heatmap:
# 1. "RdYlGn_r" - Red-Yellow-Green (reversed: high=red, low=green) - RECOMMENDED
# 2. "RdBu_r"   - Red-Blue (reversed: high=red, low=blue)
# 3. "coolwarm" - Cool-Warm (high=red, low=blue)
# 4. "viridis"  - Viridis (perceptually uniform, colorblind-friendly)
# 5. "plasma"   - Plasma (perceptually uniform, high contrast)
COLOR_SCHEME = "rocket"

# VARIABLES TO VISUALIZE
# Define which DHS variables to create heatmaps for
VARIABLES_TO_PLOT = ["D111", "D104", "D106", "D108"]

# VARIABLE DESCRIPTIONS
# Maps variable codes to their full descriptions for figure titles
VARIABLE_DESCRIPTIONS = {
    "D111": "Any IPV",
    "D104": "Emotional IPV",
    "D106": "Physical IPV",
    "D108": "Sexual IPV",
}

# INPUT VARIABLES (socioeconomic factors)
WEALTH_VARIABLE = "v190"      # Wealth index combined (1=poorest, 5=richest)
EDUCATION_VARIABLE = "v106"   # Highest educational level (0=no education, 3=higher)

# Variable labels for plotting
WEALTH_LABELS = {
    1: "Poorest",
    2: "Poorer", 
    3: "Middle",
    4: "Richer",
    5: "Richest"
}

EDUCATION_LABELS = {
    0: "No education",
    1: "Primary",
    2: "Secondary",
    3: "Higher"
}

# Missing value codes to exclude
EDUCATION_MISSING_CODES = [9]  # 9 = Missing for V106

# Determine paths relative to this script
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Input files
DATA_DIR = PROJECT_ROOT / "DATA"
DHS_DIR = DATA_DIR / "DHS"
INPUT_CSV = DHS_DIR / "women_all_answers_gps.csv"

# Output directory (output/ at same level as src/)
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Visualization settings
USE_WEIGHTS = False          # Use DHS sampling weights (recommended)
WEIGHT_VARIABLE = "d005"    # Weight variable to use: "d005" (DV weights) or "v005" (household weights)
                            # Only used if USE_WEIGHTS = True
MIN_SAMPLES_PER_CELL = 20   # Minimum sample size to display a cell (default: 20)
                            # Cells with fewer samples will be hidden (shown as blank)
DPI = 300                   # Resolution of output images (dots per inch)
FIGSIZE = (10, 8)          # Figure size in inches

# Color scale range (prevalence bounds for heatmap)
VMIN = 0.0                 # Minimum prevalence (0 = 0%)
VMAX = 0.5                 # Maximum prevalence (0.5 = 50%)

# Weight variable descriptions (for documentation)
WEIGHT_DESCRIPTIONS = {
    "d005": "Domestic Violence weights (6 decimals) - for DV module subsample",
    "v005": "Women's individual sample weight (6 decimals) - for all women"
}

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
        Sampling weights (already scaled, i.e., weight/1,000,000)

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


def aggregate_by_wealth_education(data, output_var, wealth_var, education_var, 
                                  use_weights=True, weight_var='d005'):
    """
    Aggregate IPV prevalence by wealth quintile and educational attainment.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Individual-level data
    output_var : str
        IPV variable to analyze (e.g., 'd111')
    wealth_var : str
        Wealth quintile variable (e.g., 'v190')
    education_var : str
        Educational attainment variable (e.g., 'v106')
    use_weights : bool
        Whether to use weighted aggregation
    weight_var : str
        Weight variable to use ('d005' or 'v005')
    
    Returns:
    --------
    pd.DataFrame : Aggregated data with columns [wealth, education, prevalence, n_women]
    """
    # Filter to women who:
    # 1. Answered the IPV question
    # 2. Have valid wealth data
    # 3. Have valid education data (excluding missing code 9)
    data_valid = data[
        data[output_var].notna() & 
        data[wealth_var].notna() & 
        data[education_var].notna() &
        (~data[education_var].isin(EDUCATION_MISSING_CODES))  # Exclude missing values
    ].copy()
    
    if len(data_valid) == 0:
        print(f"\nWARNING: No valid data found for {output_var}")
        return pd.DataFrame(columns=['wealth', 'education', 'prevalence', 'n_women'])
    
    if use_weights:
        # Create sampling_weight column from specified weight variable
        if weight_var in data_valid.columns:
            data_valid['sampling_weight'] = (
                pd.to_numeric(data_valid[weight_var], errors='coerce') / 1_000_000.0
            )
            print(f"    → Using weight variable: {weight_var.upper()}")
        else:
            print(f"\nWARNING: {weight_var} not found, using equal weights")
            data_valid['sampling_weight'] = 1.0
        
        # Aggregate with weights
        agg = (
            data_valid.groupby([wealth_var, education_var])
                .apply(lambda g: pd.Series({
                    'prevalence': weighted_prop(g[output_var], g['sampling_weight']),
                    'n_women': len(g)
                }), include_groups=False)
                .reset_index()
                .rename(columns={wealth_var: 'wealth', education_var: 'education'})
        )
    else:
        # Aggregate without weights
        agg = (
            data_valid.groupby([wealth_var, education_var])
                .agg(**{
                    'prevalence': (output_var, 'mean'),
                    'n_women': (output_var, 'size')
                })
                .reset_index()
                .rename(columns={wealth_var: 'wealth', education_var: 'education'})
        )
    
    return agg


def create_heatmap(data, output_var_desc, wealth_labels, education_labels, 
                   output_png, output_pdf, vmin=0.0, vmax=1.0, cmap='RdYlGn_r', 
                   figsize=(10, 8), min_samples=20):
    """
    Create and save heatmap visualization.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Aggregated data with columns [wealth, education, prevalence, n_women]
    output_var_desc : str
        Description of the output variable (e.g., "Any IPV")
    wealth_labels : dict
        Labels for wealth quintiles
    education_labels : dict
        Labels for education levels
    output_png : Path
        Path to save PNG file
    output_pdf : Path
        Path to save PDF file
    vmin : float
        Minimum value for color scale (default 0.0)
    vmax : float
        Maximum value for color scale (default 1.0)
    cmap : str
        Colormap name
    figsize : tuple
        Figure size (width, height) in inches
    min_samples : int
        Minimum number of samples required to display a cell (default 20)
    """
    # Create a copy of the data to avoid modifying the original
    data_filtered = data.copy()
    
    # Filter cells with fewer than min_samples
    cells_hidden = (data_filtered['n_women'] < min_samples).sum()
    if cells_hidden > 0:
        print(f"    → Hiding {cells_hidden} cells with < {min_samples} samples")
    
    # Set prevalence to NaN for cells below threshold (will appear blank)
    data_filtered.loc[data_filtered['n_women'] < min_samples, 'prevalence'] = np.nan
    
    # Pivot data to create matrix for heatmap
    heatmap_data = data_filtered.pivot(index='education', columns='wealth', values='prevalence')
    
    # Create sample size matrix for annotations
    sample_sizes = data_filtered.pivot(index='education', columns='wealth', values='n_women')
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    sns.heatmap(
        heatmap_data,
        annot=True,
        fmt='.1%',
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        cbar_kws={'label': 'Prevalence of ' + output_var_desc},
        linewidths=0.5,
        linecolor='white',
        ax=ax
    )
    
    # Customize axis labels
    if not heatmap_data.empty:
        # X-axis (wealth)
        wealth_tick_labels = [wealth_labels.get(int(col), f"Q{int(col)}") 
                              for col in heatmap_data.columns]
        ax.set_xticklabels(wealth_tick_labels, rotation=45, ha='right')
        
        # Y-axis (education) - reverse order so "No education" is at bottom
        education_tick_labels = [education_labels.get(int(idx), f"Level {int(idx)}") 
                                for idx in heatmap_data.index]
        ax.set_yticklabels(education_tick_labels, rotation=0)
    
    # Set axis labels and title
    ax.set_xlabel('Household Wealth Quintile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Highest Educational Level', fontsize=12, fontweight='bold')
    
    title = f'Prevalence of {output_var_desc} by Wealth and Education\n'
    if USE_WEIGHTS:
        weight_desc = WEIGHT_DESCRIPTIONS.get(WEIGHT_VARIABLE, WEIGHT_VARIABLE)
        title += f'Weighted by {WEIGHT_VARIABLE.upper()} ({weight_desc.split(" - ")[0]})'
    else:
        title += 'Unweighted (Simple Proportions)'
    
    # Add note about minimum sample size if cells are hidden
    if cells_hidden > 0:
        title += f'\n(Cells with < {min_samples} samples hidden)'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add sample size annotations (in smaller font below prevalence)
    for i, education_level in enumerate(heatmap_data.index):
        for j, wealth_level in enumerate(heatmap_data.columns):
            if not pd.isna(sample_sizes.iloc[i, j]):
                n = int(sample_sizes.iloc[i, j])
                # Show sample size in gray if above threshold, red if below
                color = 'gray' if n >= min_samples else 'red'
                ax.text(j + 0.5, i + 0.7, f'n={n}', 
                       ha='center', va='center', fontsize=8, color=color)
    
    plt.tight_layout()
    
    # Save outputs
    plt.savefig(output_png, dpi=DPI, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close()


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    """
    Main function that orchestrates the heatmap visualization workflow.
    """

    print("\n" + "=" * 70)
    print("DHS HEATMAP VISUALIZATION - DRC 2023-24 Survey")
    print("=" * 70)
    print(f"\nVariables to analyze: {', '.join(VARIABLES_TO_PLOT)}")
    print(f"X-axis: Household Wealth Quintile ({WEALTH_VARIABLE.upper()})")
    print(f"Y-axis: Highest Educational Level ({EDUCATION_VARIABLE.upper()})")

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

    # -------------------------------------------------------------------------
    # STEP 2: LOAD DATA
    # -------------------------------------------------------------------------
    print("\n[STEP 2/4] Loading data...")
    print("-" * 70)

    # Load the processed CSV
    print(f"\nReading: {INPUT_CSV.name}")
    data = pd.read_csv(INPUT_CSV)
    print(f"  → Loaded {len(data):,} women's records")
    
    # Convert variable names to lowercase
    wealth_var = WEALTH_VARIABLE.lower()
    education_var = EDUCATION_VARIABLE.lower()
    
    # Verify required variables exist
    missing_vars = []
    
    # Check wealth and education variables
    for var, name in [(wealth_var, WEALTH_VARIABLE),
                      (education_var, EDUCATION_VARIABLE)]:
        if var not in data.columns:
            missing_vars.append(name)
    
    # Check all IPV variables
    for var in VARIABLES_TO_PLOT:
        var_lower = var.lower()
        if var_lower not in data.columns:
            missing_vars.append(var)
    
    if missing_vars:
        print(f"\nERROR: Variables not found in data: {missing_vars}")
        print(f"Available columns: {list(data.columns)[:20]}...")
        sys.exit(1)
    
    print(f"  → All required variables found in data")
    
    # Print summary statistics
    print(f"\nVariable summary:")
    for var in VARIABLES_TO_PLOT:
        var_lower = var.lower()
        n_valid = data[var_lower].notna().sum()
        print(f"  {var}: {n_valid:,} valid responses")
    print(f"  {WEALTH_VARIABLE}: {data[wealth_var].notna().sum():,} valid responses")
    print(f"  {EDUCATION_VARIABLE}: {data[education_var].notna().sum():,} valid responses")
    
    # Check wealth quintile distribution
    print(f"\nWealth quintile distribution:")
    print(f"  (Note: Should be ~20% each in full sample, may differ in DV subsample)")
    wealth_dist = data[wealth_var].value_counts(normalize=True).sort_index() * 100
    for q in [1, 2, 3, 4, 5]:
        if q in wealth_dist.index:
            pct = wealth_dist[q]
            print(f"  Q{q} ({WEALTH_LABELS[q]}): {pct:.1f}%")
    
    # Check education distribution
    print(f"\nEducation level distribution:")
    edu_dist = data[education_var].value_counts(normalize=True).sort_index() * 100
    for level in [0, 1, 2, 3]:
        if level in edu_dist.index:
            pct = edu_dist[level]
            print(f"  Level {level} ({EDUCATION_LABELS[level]}): {pct:.1f}%")

    # -------------------------------------------------------------------------
    # STEP 3: AGGREGATE DATA FOR ALL VARIABLES
    # -------------------------------------------------------------------------
    print("\n[STEP 3/4] Aggregating data...")
    print("-" * 70)

    if USE_WEIGHTS:
        print("  Using WEIGHTED aggregation (recommended)")
        print(f"  → Weight variable: {WEIGHT_VARIABLE.upper()}")
        print(f"  → {WEIGHT_DESCRIPTIONS.get(WEIGHT_VARIABLE, 'Custom weight variable')}")
    else:
        print("  Using UNWEIGHTED aggregation")
        print("  → Simple percentages (not nationally representative)")

    # Store aggregated data for all variables
    all_agg_data = {}
    
    for i, var in enumerate(VARIABLES_TO_PLOT, 1):
        var_desc = VARIABLE_DESCRIPTIONS.get(var, var)
        print(f"\n  [{i}/{len(VARIABLES_TO_PLOT)}] Processing {var} ({var_desc})...")
        
        output_var = var.lower()
        
        agg_data = aggregate_by_wealth_education(
            data=data,
            output_var=output_var,
            wealth_var=wealth_var,
            education_var=education_var,
            use_weights=USE_WEIGHTS,
            weight_var=WEIGHT_VARIABLE
        )
        
        if len(agg_data) == 0:
            print(f"    ⚠ No data for {var}, skipping...")
            continue
        
        print(f"    → Created {len(agg_data)} wealth × education cells")
        print(f"    → Total women: {agg_data['n_women'].sum():,}")
        print(f"    → Prevalence range: {agg_data['prevalence'].min():.1%} to {agg_data['prevalence'].max():.1%}")
        
        all_agg_data[var] = agg_data
    
    if len(all_agg_data) == 0:
        print("\nERROR: No data to plot for any variable!")
        sys.exit(1)
    
    print(f"\n  → Successfully aggregated {len(all_agg_data)} variables")

    # -------------------------------------------------------------------------
    # STEP 4: CREATE HEATMAPS FOR ALL VARIABLES
    # -------------------------------------------------------------------------
    print("\n[STEP 4/4] Creating heatmap visualizations...")
    print("-" * 70)
    print(f"  → Minimum samples per cell: {MIN_SAMPLES_PER_CELL}")

    output_files = []
    
    for i, (var, agg_data) in enumerate(all_agg_data.items(), 1):
        var_desc = VARIABLE_DESCRIPTIONS.get(var, var)
        print(f"\n  [{i}/{len(all_agg_data)}] Creating heatmap for {var} ({var_desc})...")
        
        # Generate output filenames for this variable
        output_csv = OUTPUT_DIR / f"h1_{var.lower()}_heatmap.csv"
        output_png = OUTPUT_DIR / f"h1_{var.lower()}_heatmap.png"
        output_pdf = OUTPUT_DIR / f"h1_{var.lower()}_heatmap.pdf"
        
        # Save aggregated data to CSV
        agg_output = agg_data.copy()
        agg_output['variable'] = var
        agg_output['variable_desc'] = var_desc
        agg_output['wealth_label'] = agg_output['wealth'].map(WEALTH_LABELS)
        agg_output['education_label'] = agg_output['education'].map(EDUCATION_LABELS)
        agg_output['prevalence_pct'] = agg_output['prevalence'] * 100
        
        # Reorder columns for readability
        agg_output = agg_output[[
            'variable', 'variable_desc',
            'wealth', 'wealth_label', 
            'education', 'education_label',
            'prevalence', 'prevalence_pct', 'n_women'
        ]]
        
        agg_output.to_csv(output_csv, index=False)
        
        # Create heatmap
        create_heatmap(
            data=agg_data,
            output_var_desc=var_desc,
            wealth_labels=WEALTH_LABELS,
            education_labels=EDUCATION_LABELS,
            output_png=output_png,
            output_pdf=output_pdf,
            vmin=VMIN,
            vmax=VMAX,
            cmap=COLOR_SCHEME,
            figsize=FIGSIZE,
            min_samples=MIN_SAMPLES_PER_CELL
        )
        
        print(f"    ✓ Saved: {output_csv.name}")
        print(f"    ✓ Saved: {output_png.name}")
        print(f"    ✓ Saved: {output_pdf.name}")
        
        output_files.append((var, var_desc, output_csv, output_png, output_pdf, agg_data))

    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("HEATMAP VISUALIZATION COMPLETE!")
    print("=" * 70)
    print(f"\n📊 OUTPUT FILES (saved in {OUTPUT_DIR.name}/):\n")
    
    file_num = 1
    for var, var_desc, csv_path, png_path, pdf_path, agg_data in output_files:
        print(f"  {file_num}. {csv_path.name}")
        print(f"     → Aggregated data for {var_desc}")
        print(f"     → {len(agg_data)} cells, {agg_data['n_women'].sum():,} women\n")
        file_num += 1
        
        print(f"  {file_num}. {png_path.name}")
        print(f"     → Heatmap for {var_desc} (PNG, {DPI} DPI)\n")
        file_num += 1
        
        print(f"  {file_num}. {pdf_path.name}")
        print(f"     → Heatmap for {var_desc} (PDF, vector)\n")
        file_num += 1

    print(f"⚙️  Settings:")
    print(f"  → Variables analyzed: {', '.join(VARIABLES_TO_PLOT)}")
    print(f"  → Wealth variable: {WEALTH_VARIABLE.upper()} (5 quintiles)")
    print(f"  → Education variable: {EDUCATION_VARIABLE.upper()} (4 levels)")
    if USE_WEIGHTS:
        print(f"  → Weighting: {WEIGHT_VARIABLE.upper()}")
        print(f"    ({WEIGHT_DESCRIPTIONS.get(WEIGHT_VARIABLE, 'Custom weight')})")
    else:
        print(f"  → Weighting: None (unweighted)")
    print(f"  → Minimum samples per cell: {MIN_SAMPLES_PER_CELL}")
    print(f"  → Color scheme: {COLOR_SCHEME}")
    print(f"  → Color scale: {VMIN:.0%} to {VMAX:.0%} (fixed across all heatmaps)")

    # Print interpretation guidance
    print("\n📈 INTERPRETATION:")
    print("  → Red cells = High IPV prevalence")
    print("  → Green cells = Low IPV prevalence")
    print(f"  → Blank cells = Fewer than {MIN_SAMPLES_PER_CELL} samples (unreliable)")
    print("  → H1 hypothesis predicts: Prevalence should decrease")
    print("    from bottom-left (poor, uneducated) to top-right (rich, educated)")

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