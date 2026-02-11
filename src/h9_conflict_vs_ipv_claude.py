#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
Conflict Proximity vs IPV Analysis - DRC 2023-24 Survey
================================================================================

WHAT THIS SCRIPT DOES:
----------------------
This script analyzes the relationship between women's proximity to armed
conflict events and their likelihood of experiencing intimate partner violence.

PURPOSE:
Tests hypothesis H9: Recent exposure to armed conflict is positively associated 
with the probability of experiencing IPV.

VISUALIZATION:
- X-axis: Distance to nearest conflict event (km)
- Y-axis: IPV prevalence (0 = No, 1 = Yes)
- Scatter plot with smooth curve fitted through data
- 95% confidence interval bands via bootstrapping

INPUTS:
-------
- DATA/DHS/women_all_answers_gps.csv (DHS women's responses with GPS)
- DATA/Conflict/Africa_lagged_data_up_to-2024-10-17/Africa_lagged_data_up_to-2024-10-17.xlsx (ACLED conflict data)

OUTPUTS:
--------
- h9_d111_curve.csv (individual-level data: woman_id, distance_km, ipv_response)
- h9_d111_curve.jpg (visualization - JPG format, 300 DPI)
- h9_d111_curve.pdf (visualization - PDF format, vector)

All outputs are saved in the output/ directory (same level as src/).

USAGE:
------
Run from the project root directory:
    python src/h9_conflict_vs_ipv.py

REQUIREMENTS:
-------------
- pandas (data manipulation)
- numpy (numerical operations)
- matplotlib (plotting)
- scipy (smoothing and statistics)
- openpyxl (reading Excel files)

Install with:
    pip install pandas numpy matplotlib scipy openpyxl

================================================================================
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.stats import bootstrap

# =============================================================================
# CONFIGURATION
# =============================================================================

# RANDOM SEED (for deterministic results)
RANDOM_SEED = 42

# IPV VARIABLE TO ANALYZE
IPV_VARIABLE = "d111"  # Any IPV (D111: experienced any form of IPV)
IPV_DESCRIPTION = "Any IPV (D111)"

# WEIGHTING SETTINGS
USE_WEIGHTS = True              # Use DHS sampling weights (recommended)
WEIGHT_VARIABLE = "d005"        # Weight variable: "d005" (DV weights) or "v005" (household weights)

# CONFLICT DATA SETTINGS
# Maximum distance to consider (km) - conflicts beyond this are not used
MAX_CONFLICT_DISTANCE_KM = 500.0

# SMOOTHING SETTINGS
# Smoothing parameter for spline (0 = interpolation, higher = smoother)
# Good range: 0.001 to 0.1. Lower = follows data more closely
SMOOTHING_PARAMETER = 0.01

# Spline degree (3 = cubic spline, recommended)
SPLINE_DEGREE = 3

# BOOTSTRAP SETTINGS
# Number of bootstrap samples for confidence intervals
N_BOOTSTRAP = 1000

# Confidence level (0.95 = 95% CI)
CONFIDENCE_LEVEL = 0.95

# VISUALIZATION SETTINGS
DPI = 300                       # Resolution of output images
FIGSIZE = (12, 8)              # Figure size in inches
MARKER_SIZE = 20               # Size of scatter plot markers
MARKER_ALPHA = 0.3             # Transparency of markers (0-1)
LINE_WIDTH = 2.5               # Width of smooth curve
CI_ALPHA = 0.2                 # Transparency of confidence interval band

# X-axis settings
X_MIN = 0                      # Minimum distance to plot (km)
X_MAX = 200                    # Maximum distance to plot (km)
X_LABEL = "Distance to Nearest Conflict Event (km)"

# Y-axis settings
Y_LABEL = "IPV Response (0 = No, 1 = Yes)"
Y_TICK_LABELS = ["No (0)", "Yes (1)"]

# Determine paths relative to this script
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Input files
DATA_DIR = PROJECT_ROOT / "DATA"
DHS_DIR = DATA_DIR / "DHS"
CONFLICT_DIR = DATA_DIR / "Conflict" / "Africa_lagged_data_up_to-2024-10-17"

INPUT_DHS_CSV = DHS_DIR / "women_all_answers_gps.csv"
INPUT_ACLED_XLSX = CONFLICT_DIR / "Africa_lagged_data_up_to-2024-10-17.xlsx"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Output files
OUTPUT_CSV = OUTPUT_DIR / "h9_d111_curve.csv"
OUTPUT_JPG = OUTPUT_DIR / "h9_d111_curve.jpg"
OUTPUT_PDF = OUTPUT_DIR / "h9_d111_curve.pdf"

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def haversine_distance_km(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two points on Earth using
    Haversine formula. More accurate than Euclidean for larger distances.
    
    Parameters:
    -----------
    lat1, lon1 : float or array
        Latitude and longitude of first point(s) in degrees
    lat2, lon2 : float or array
        Latitude and longitude of second point(s) in degrees
    
    Returns:
    --------
    float or array : Distance in kilometers
    """
    # Earth radius in km
    R = 6371.0
    
    # Convert to radians
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def compute_min_distance_to_conflicts(woman_lat, woman_lon, conflict_lats, conflict_lons):
    """
    Compute minimum distance from a woman's location to any conflict event.
    
    Parameters:
    -----------
    woman_lat, woman_lon : float
        Woman's cluster coordinates
    conflict_lats, conflict_lons : array-like
        Arrays of conflict event coordinates
    
    Returns:
    --------
    float : Minimum distance to any conflict event (km)
    """
    if len(conflict_lats) == 0:
        return np.nan
    
    # Compute distance to all conflict events
    distances = haversine_distance_km(
        woman_lat, woman_lon,
        conflict_lats, conflict_lons
    )
    
    # Return minimum distance
    return np.min(distances)


def fit_weighted_smooth_curve(x, y, weights, s=0.01, k=3, x_eval=None):
    """
    Fit a weighted smoothing spline to binary data.
    
    Parameters:
    -----------
    x : array-like
        X values (distances)
    y : array-like
        Y values (binary: 0 or 1)
    weights : array-like
        Sample weights
    s : float
        Smoothing parameter (0 = interpolation, higher = smoother)
    k : int
        Degree of spline (3 = cubic)
    x_eval : array-like, optional
        X values at which to evaluate the curve. If None, uses x.
    
    Returns:
    --------
    tuple : (x_eval, y_eval) - evaluation points and fitted values
    """
    # Sort by x
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    w_sorted = weights[sort_idx]
    
    # Fit spline
    spline = UnivariateSpline(x_sorted, y_sorted, w=w_sorted, s=s, k=k)
    
    # Evaluate
    if x_eval is None:
        x_eval = np.linspace(x_sorted.min(), x_sorted.max(), 200)
    
    y_eval = spline(x_eval)
    
    # Clip to [0, 1] since we're modeling probabilities
    y_eval = np.clip(y_eval, 0, 1)
    
    return x_eval, y_eval


def bootstrap_confidence_interval(x, y, weights, s=0.01, k=3, x_eval=None,
                                  n_bootstrap=1000, confidence_level=0.95):
    """
    Compute bootstrap confidence interval for the smooth curve.
    
    Parameters:
    -----------
    x, y, weights : array-like
        Data and weights
    s, k : float, int
        Spline parameters
    x_eval : array-like
        X values for evaluation
    n_bootstrap : int
        Number of bootstrap samples
    confidence_level : float
        Confidence level (e.g., 0.95 for 95% CI)
    
    Returns:
    --------
    tuple : (lower_bound, upper_bound) arrays of same length as x_eval
    """
    if x_eval is None:
        x_eval = np.linspace(x.min(), x.max(), 200)
    
    n_samples = len(x)
    y_bootstrap = np.zeros((n_bootstrap, len(x_eval)))
    
    # Set random seed for reproducibility
    rng = np.random.RandomState(RANDOM_SEED)
    
    print(f"  → Computing {n_bootstrap} bootstrap samples...")
    
    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"    Bootstrap sample {i+1}/{n_bootstrap}")
        
        # Resample with replacement
        boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
        x_boot = x[boot_idx]
        y_boot = y[boot_idx]
        w_boot = weights[boot_idx]
        
        # Fit curve to bootstrap sample
        try:
            _, y_fit = fit_weighted_smooth_curve(x_boot, y_boot, w_boot, s=s, k=k, x_eval=x_eval)
            y_bootstrap[i, :] = y_fit
        except:
            # If fitting fails, use NaN (will be excluded from percentile calculation)
            y_bootstrap[i, :] = np.nan
    
    # Compute percentiles (ignoring NaNs)
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    
    lower_bound = np.nanpercentile(y_bootstrap, lower_percentile, axis=0)
    upper_bound = np.nanpercentile(y_bootstrap, upper_percentile, axis=0)
    
    return lower_bound, upper_bound


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    """
    Main function that orchestrates the conflict-IPV analysis.
    """
    
    # Set random seed for reproducibility
    np.random.seed(RANDOM_SEED)
    
    print("\n" + "=" * 70)
    print("CONFLICT PROXIMITY vs IPV ANALYSIS - DRC 2023-24 Survey")
    print("=" * 70)
    print(f"\nHypothesis H9: Recent exposure to armed conflict is positively")
    print(f"               associated with IPV prevalence")
    print(f"\nIPV variable: {IPV_VARIABLE.upper()} ({IPV_DESCRIPTION})")
    print(f"Random seed: {RANDOM_SEED} (for reproducibility)")
    
    # -------------------------------------------------------------------------
    # STEP 1: VERIFY INPUT FILES
    # -------------------------------------------------------------------------
    print("\n[STEP 1/8] Verifying input files...")
    print("-" * 70)
    
    if not INPUT_DHS_CSV.exists():
        print(f"ERROR: DHS CSV not found at {INPUT_DHS_CSV}")
        sys.exit(1)
    
    if not INPUT_ACLED_XLSX.exists():
        print(f"ERROR: ACLED Excel file not found at {INPUT_ACLED_XLSX}")
        sys.exit(1)
    
    print(f"✓ Found DHS CSV: {INPUT_DHS_CSV.name}")
    print(f"  Location: {INPUT_DHS_CSV}")
    print(f"✓ Found ACLED Excel: {INPUT_ACLED_XLSX.name}")
    print(f"  Location: {INPUT_ACLED_XLSX}")
    
    # -------------------------------------------------------------------------
    # STEP 2: LOAD CONFLICT DATA (ACLED)
    # -------------------------------------------------------------------------
    print("\n[STEP 2/8] Loading conflict data...")
    print("-" * 70)
    
    print(f"\nReading ACLED data from Excel...")
    acled = pd.read_excel(INPUT_ACLED_XLSX, engine="openpyxl")
    print(f"  → Loaded {len(acled):,} conflict events")
    
    # Check for required columns
    required_cols = ["LATITUDE", "LONGITUDE"]
    missing_cols = [col for col in required_cols if col not in acled.columns]
    if missing_cols:
        print(f"ERROR: ACLED file missing columns: {missing_cols}")
        sys.exit(1)
    
    # Convert to numeric and filter valid coordinates
    acled["LATITUDE"] = pd.to_numeric(acled["LATITUDE"], errors="coerce")
    acled["LONGITUDE"] = pd.to_numeric(acled["LONGITUDE"], errors="coerce")
    
    acled_valid = acled[acled["LATITUDE"].notna() & acled["LONGITUDE"].notna()].copy()
    print(f"  → {len(acled_valid):,} events with valid coordinates")
    
    # Extract conflict coordinates as arrays for efficient distance computation
    conflict_lats = acled_valid["LATITUDE"].to_numpy()
    conflict_lons = acled_valid["LONGITUDE"].to_numpy()
    
    print(f"  → Latitude range: {conflict_lats.min():.3f} to {conflict_lats.max():.3f}")
    print(f"  → Longitude range: {conflict_lons.min():.3f} to {conflict_lons.max():.3f}")
    
    # -------------------------------------------------------------------------
    # STEP 3: LOAD DHS WOMEN'S DATA
    # -------------------------------------------------------------------------
    print("\n[STEP 3/8] Loading DHS women's data...")
    print("-" * 70)
    
    print(f"\nReading DHS data...")
    dhs = pd.read_csv(INPUT_DHS_CSV)
    print(f"  → Loaded {len(dhs):,} women's records")
    
    # Check for required columns (lowercase)
    ipv_var = IPV_VARIABLE.lower()
    
    # GPS columns in DHS data - try both possible naming conventions (lowercase and uppercase)
    gps_lat_cols = ["latnum", "LATNUM", "v023a", "sclustlat", "cluster_lat"]
    gps_lon_cols = ["longnum", "LONGNUM", "v023b", "sclustlon", "cluster_lon"]
    
    # Find which GPS columns exist
    lat_col = None
    lon_col = None
    for col in gps_lat_cols:
        if col in dhs.columns:
            lat_col = col
            break
    for col in gps_lon_cols:
        if col in dhs.columns:
            lon_col = col
            break
    
    if lat_col is None or lon_col is None:
        print(f"ERROR: Could not find GPS coordinate columns")
        print(f"  Looked for latitude columns: {gps_lat_cols}")
        print(f"  Looked for longitude columns: {gps_lon_cols}")
        print(f"  Available columns: {list(dhs.columns)[:50]}...")
        sys.exit(1)
    
    print(f"  → Found GPS columns: {lat_col}, {lon_col}")
    
    # Check for IPV variable
    required_dhs_cols = [ipv_var]
    
    # Check if weight variable exists (if weighting is enabled)
    if USE_WEIGHTS:
        weight_var = WEIGHT_VARIABLE.lower()
        required_dhs_cols.append(weight_var)
    
    missing_dhs_cols = [col for col in required_dhs_cols if col not in dhs.columns]
    if missing_dhs_cols:
        print(f"ERROR: DHS file missing columns: {missing_dhs_cols}")
        print(f"Available columns: {list(dhs.columns)[:20]}...")
        sys.exit(1)
    
    print(f"  → All required variables found")
    
    # -------------------------------------------------------------------------
    # STEP 4: FILTER TO VALID RESPONSES
    # -------------------------------------------------------------------------
    print("\n[STEP 4/8] Filtering to valid IPV responses...")
    print("-" * 70)
    
    # Filter to women who:
    # 1. Answered the IPV question (0 or 1, not missing)
    # 2. Have valid GPS coordinates
    print(f"\nFiltering criteria:")
    print(f"  1. Valid IPV response ({ipv_var} is 0 or 1)")
    print(f"  2. Valid GPS coordinates ({lat_col}, {lon_col})")
    
    dhs_valid = dhs[
        dhs[ipv_var].notna() &
        dhs[ipv_var].isin([0, 1]) &
        dhs[lat_col].notna() &
        dhs[lon_col].notna()
    ].copy()
    
    print(f"\n  → {len(dhs_valid):,} women with valid data ({len(dhs_valid)/len(dhs)*100:.1f}%)")
    print(f"  → IPV prevalence: {dhs_valid[ipv_var].mean()*100:.1f}%")
    
    # -------------------------------------------------------------------------
    # STEP 5: COMPUTE DISTANCE TO NEAREST CONFLICT
    # -------------------------------------------------------------------------
    print("\n[STEP 5/8] Computing distance to nearest conflict for each woman...")
    print("-" * 70)
    
    print(f"\nComputing distances (this may take a minute)...")
    
    # Compute minimum distance for each woman
    dhs_valid["distance_to_conflict_km"] = dhs_valid.apply(
        lambda row: compute_min_distance_to_conflicts(
            row[lat_col], row[lon_col],
            conflict_lats, conflict_lons
        ),
        axis=1
    )
    
    print(f"  → Distance range: {dhs_valid['distance_to_conflict_km'].min():.1f} to {dhs_valid['distance_to_conflict_km'].max():.1f} km")
    print(f"  → Median distance: {dhs_valid['distance_to_conflict_km'].median():.1f} km")
    print(f"  → Mean distance: {dhs_valid['distance_to_conflict_km'].mean():.1f} km")
    
    # Filter to maximum distance if specified
    if MAX_CONFLICT_DISTANCE_KM is not None:
        n_before = len(dhs_valid)
        dhs_valid = dhs_valid[dhs_valid["distance_to_conflict_km"] <= MAX_CONFLICT_DISTANCE_KM]
        n_after = len(dhs_valid)
        print(f"  → Filtered to distances ≤ {MAX_CONFLICT_DISTANCE_KM} km: {n_after:,} women ({n_after/n_before*100:.1f}%)")
    
    # -------------------------------------------------------------------------
    # STEP 6: PREPARE DATA FOR VISUALIZATION
    # -------------------------------------------------------------------------
    print("\n[STEP 6/8] Preparing data for visualization...")
    print("-" * 70)
    
    # Extract arrays for plotting
    x_data = dhs_valid["distance_to_conflict_km"].to_numpy()
    y_data = dhs_valid[ipv_var].to_numpy()
    
    # Prepare weights
    if USE_WEIGHTS:
        weight_var = WEIGHT_VARIABLE.lower()
        weights = pd.to_numeric(dhs_valid[weight_var], errors="coerce") / 1_000_000.0
        weights = weights.fillna(1.0).to_numpy()
        print(f"  → Using weighted analysis with {weight_var.upper()}")
        print(f"  → Weight range: {weights.min():.4f} to {weights.max():.4f}")
    else:
        weights = np.ones(len(dhs_valid))
        print(f"  → Using unweighted analysis (equal weights)")
    
    # Normalize weights to sum to 1 (for proper probability estimates)
    weights = weights / weights.sum()
    
    print(f"\n  → Final sample size: {len(x_data):,} women")
    print(f"  → Distance range for plotting: {x_data.min():.1f} to {x_data.max():.1f} km")
    print(f"  → IPV responses: {int(y_data.sum())} Yes, {int((1-y_data).sum())} No")
    
    # -------------------------------------------------------------------------
    # STEP 7: FIT SMOOTH CURVE WITH CONFIDENCE INTERVALS
    # -------------------------------------------------------------------------
    print("\n[STEP 7/8] Fitting smooth curve and computing confidence intervals...")
    print("-" * 70)
    
    print(f"\nSmoothing parameters:")
    print(f"  → Smoothing parameter (s): {SMOOTHING_PARAMETER}")
    print(f"  → Spline degree (k): {SPLINE_DEGREE}")
    print(f"  → Bootstrap samples: {N_BOOTSTRAP}")
    print(f"  → Confidence level: {CONFIDENCE_LEVEL*100:.0f}%")
    
    # Define evaluation points for the curve
    x_eval = np.linspace(X_MIN, min(X_MAX, x_data.max()), 200)
    
    # Fit main curve
    print(f"\n  → Fitting main curve...")
    _, y_fit = fit_weighted_smooth_curve(x_data, y_data, weights, 
                                         s=SMOOTHING_PARAMETER, k=SPLINE_DEGREE, 
                                         x_eval=x_eval)
    
    # Compute bootstrap confidence intervals
    print(f"\n  → Computing bootstrap confidence intervals...")
    lower_ci, upper_ci = bootstrap_confidence_interval(
        x_data, y_data, weights,
        s=SMOOTHING_PARAMETER, k=SPLINE_DEGREE,
        x_eval=x_eval,
        n_bootstrap=N_BOOTSTRAP,
        confidence_level=CONFIDENCE_LEVEL
    )
    
    print(f"\n  ✓ Curve fitting complete")
    
    # -------------------------------------------------------------------------
    # STEP 8: CREATE VISUALIZATION AND SAVE OUTPUTS
    # -------------------------------------------------------------------------
    print("\n[STEP 8/8] Creating visualization and saving outputs...")
    print("-" * 70)
    
    # Save raw data to CSV
    print(f"\n  → Saving raw data to CSV...")
    output_data = dhs_valid[[lat_col, lon_col, ipv_var, "distance_to_conflict_km"]].copy()
    output_data["woman_id"] = range(1, len(output_data) + 1)
    output_data = output_data[["woman_id", "distance_to_conflict_km", ipv_var]]
    output_data.columns = ["woman_id", "distance_km", "ipv_response"]
    output_data.to_csv(OUTPUT_CSV, index=False)
    print(f"    ✓ Saved: {OUTPUT_CSV.name}")
    
    # Create visualization
    print(f"\n  → Creating visualization...")
    
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    # Scatter plot of individual responses
    # Separate colors for Yes (1) and No (0)
    mask_no = y_data == 0
    mask_yes = y_data == 1
    
    ax.scatter(x_data[mask_no], y_data[mask_no], 
              s=MARKER_SIZE, alpha=MARKER_ALPHA, 
              c='steelblue', label='No IPV (0)', zorder=2)
    
    ax.scatter(x_data[mask_yes], y_data[mask_yes], 
              s=MARKER_SIZE, alpha=MARKER_ALPHA, 
              c='crimson', label='Yes IPV (1)', zorder=2)
    
    # Confidence interval band
    ax.fill_between(x_eval, lower_ci, upper_ci, 
                    alpha=CI_ALPHA, color='gray', 
                    label=f'{CONFIDENCE_LEVEL*100:.0f}% CI (Bootstrap)', zorder=3)
    
    # Smooth curve
    ax.plot(x_eval, y_fit, 
           linewidth=LINE_WIDTH, color='black', 
           label='Smooth curve (weighted spline)', zorder=4)
    
    # Formatting
    ax.set_xlabel(X_LABEL, fontsize=12, fontweight='bold')
    ax.set_ylabel(Y_LABEL, fontsize=12, fontweight='bold')
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(-0.05, 1.05)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(Y_TICK_LABELS)
    
    # Title
    title = f'IPV Prevalence vs Distance to Nearest Conflict\n'
    title += f'{IPV_DESCRIPTION} | n={len(dhs_valid):,} women'
    if USE_WEIGHTS:
        title += f' | Weighted by {WEIGHT_VARIABLE.upper()}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    ax.legend(loc='best', frameon=True, framealpha=0.9)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save outputs
    plt.savefig(OUTPUT_JPG, dpi=DPI, bbox_inches='tight')
    print(f"    ✓ Saved: {OUTPUT_JPG.name}")
    
    plt.savefig(OUTPUT_PDF, bbox_inches='tight')
    print(f"    ✓ Saved: {OUTPUT_PDF.name}")
    
    plt.close()
    
    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONFLICT-IPV ANALYSIS COMPLETE!")
    print("=" * 70)
    
    print(f"\n📊 OUTPUT FILES (saved in {OUTPUT_DIR.name}/):\n")
    print(f"  1. {OUTPUT_CSV.name}")
    print(f"     → Raw data: woman_id, distance_km, ipv_response")
    print(f"     → {len(output_data):,} women\n")
    
    print(f"  2. {OUTPUT_JPG.name}")
    print(f"     → Visualization (JPG, {DPI} DPI)\n")
    
    print(f"  3. {OUTPUT_PDF.name}")
    print(f"     → Visualization (PDF, vector)\n")
    
    print(f"⚙️  ANALYSIS SETTINGS:")
    print(f"  → IPV variable: {IPV_VARIABLE.upper()} ({IPV_DESCRIPTION})")
    print(f"  → Sample size: {len(dhs_valid):,} women")
    print(f"  → Distance range: {x_data.min():.1f} - {x_data.max():.1f} km")
    print(f"  → Plotting range: {X_MIN} - {X_MAX} km")
    if USE_WEIGHTS:
        print(f"  → Weighting: {WEIGHT_VARIABLE.upper()}")
    else:
        print(f"  → Weighting: None (unweighted)")
    print(f"  → Smoothing parameter: {SMOOTHING_PARAMETER}")
    print(f"  → Bootstrap samples: {N_BOOTSTRAP}")
    print(f"  → Confidence level: {CONFIDENCE_LEVEL*100:.0f}%")
    print(f"  → Random seed: {RANDOM_SEED}")
    
    print(f"\n📈 INTERPRETATION:")
    print(f"  → Each point represents one woman")
    print(f"  → Blue points: Women who did NOT experience IPV")
    print(f"  → Red points: Women who DID experience IPV")
    print(f"  → Black curve: Smooth estimate of IPV probability vs distance")
    print(f"  → Gray band: {CONFIDENCE_LEVEL*100:.0f}% confidence interval (bootstrap)")
    print(f"  → H9 hypothesis predicts: IPV prevalence should DECREASE")
    print(f"    with increasing distance from conflict")
    
    print("\n✅ Analysis complete!\n")
    
    # -------------------------------------------------------------------------
    # DISTANCE-BASED PREVALENCE ANALYSIS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DISTANCE-BASED PREVALENCE ANALYSIS (UNWEIGHTED)")
    print("=" * 70)
    
    print(f"\nIPV prevalence by proximity to conflict:")
    print(f"(Based on {len(dhs_valid):,} women with valid responses)\n")
    
    # Define distance thresholds (in km)
    distance_thresholds = [10, 25, 50, 100]
    
    for threshold in distance_thresholds:
        # Filter to women within this distance
        within_distance = dhs_valid[dhs_valid["distance_to_conflict_km"] <= threshold]
        
        if len(within_distance) > 0:
            # Calculate prevalence (fraction answering yes)
            n_total = len(within_distance)
            n_yes = within_distance[ipv_var].sum()
            prevalence = n_yes / n_total
            
            print(f"  Distance < {threshold:3d} km:")
            print(f"    → {n_total:5,} women ({n_total/len(dhs_valid)*100:5.1f}% of sample)")
            print(f"    → {n_yes:5,} reported IPV")
            print(f"    → Prevalence: {prevalence*100:5.1f}%")
            print()
        else:
            print(f"  Distance < {threshold:3d} km:")
            print(f"    → No women within this distance")
            print()
    
    # Overall prevalence for comparison
    overall_prevalence = dhs_valid[ipv_var].mean()
    print(f"  Overall prevalence (all distances):")
    print(f"    → {len(dhs_valid):5,} women")
    print(f"    → {int(dhs_valid[ipv_var].sum()):5,} reported IPV")
    print(f"    → Prevalence: {overall_prevalence*100:5.1f}%")
    print()


# =============================================================================
# RUN THE SCRIPT
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Analysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)