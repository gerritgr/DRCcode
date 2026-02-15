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
For each variable in VARIABLES_TO_PLOT:
- h9_{variable}_curve.csv (individual-level data: woman_id, distance_km, ipv_response)
- h9_{variable}_curve.jpg (visualization - JPG format, 300 DPI)
- h9_{variable}_curve.pdf (visualization - PDF format, vector)

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
- seaborn (color palettes)
- scipy (smoothing and statistics)
- openpyxl (reading Excel files)

Install with:
    pip install pandas numpy matplotlib seaborn scipy openpyxl

================================================================================
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import UnivariateSpline
from scipy.stats import bootstrap

# =============================================================================
# CONFIGURATION
# =============================================================================

# COLOR SCHEME
# Using seaborn muted palette for consistent, professional colors
COLOR_NO = sns.color_palette("muted")[0]   # Blue for "No" responses
COLOR_YES = sns.color_palette("muted")[3]  # Red for "Yes" responses  
COLOR_LINE = sns.color_palette("muted")[1] # Orange for interpolated curve

# VARIABLES TO VISUALIZE
# Define which DHS variables to create curves for
VARIABLES_TO_PLOT = ["D111", "D104", "D106", "D108", "D117A", "D130A", "V763A"]

# VARIABLE DESCRIPTIONS
# Maps variable codes to their full descriptions for figure titles
VARIABLE_DESCRIPTIONS = {
    "D111": "Any IPV",
    "D104": "Emotional IPV",
    "D106": "Physical IPV",
    "D108": "Sexual IPV",
    "D117A": "Hit by non-partner (last 12m)",
    "D130A": "Previous partner IPV",
    "V763A": "STI in the last 12 month",
}

# VARIABLE CODING
# Defines which response values count as "Yes" (1) and "No" (0) for each variable
# Format: {variable: {"yes": [list of values], "no": [list of values]}}
# All other values are ignored (treated as missing)
VARIABLE_CODING = {
    "D111": {"yes": [1], "no": [0]},           # Standard binary: 0=No, 1=Yes
    "D104": {"yes": [1], "no": [0]},           # Standard binary: 0=No, 1=Yes
    "D106": {"yes": [1], "no": [0]},           # Standard binary: 0=No, 1=Yes
    "D108": {"yes": [1], "no": [0]},           # Standard binary: 0=No, 1=Yes
    "D117A": {"yes": [1, 2], "no": [0, 3, 4, 5, 6]},  # Frequency: 1-2=Yes (often/sometimes), others=No
    "D130A": {"yes": [1], "no": [0]},          # Standard binary: 0=No, 1=Yes
    "V763A": {"yes": [1], "no": [0]},          # Standard binary: 0=No, 1=Yes
}

# RANDOM SEED (for deterministic results)
RANDOM_SEED = 42

# X-AXIS THRESHOLD (for visualization and analysis)
X_AXIS_THRESHOLD_KM = 100      # Only show distances up to this value (default: 100 km)

# CURVE SMOOTHING SETTING
# Higher values = smoother curve, lower values = follows data more closely
# Recommended range: 0.001 to 0.1
CURVE_SMOOTHNESS = 0.01        # Default: 0.01 (moderate smoothing)

# TIME RANGE FOR CONFLICT DATA
# Define which conflict events to include based on their date
# Format: "YYYY-MM-DD" or None (None = no limit)
CONFLICT_DATE_MIN = None       # Lower bound (None = include all early events)
CONFLICT_DATE_MAX = "2023-03-01"  # Upper bound (default: March 1, 2023 - events after this ignored)

# WEIGHTING SETTINGS
USE_WEIGHTS = True              # Use DHS sampling weights (recommended)
WEIGHT_VARIABLE = "d005"        # Weight variable: "d005" (DV weights) or "v005" (household weights)

# CONFLICT DATA SETTINGS
# Maximum distance to consider (km) - conflicts beyond this are not used
MAX_CONFLICT_DISTANCE_KM = 500.0

# SPLINE SETTINGS (used internally - controlled by CURVE_SMOOTHNESS above)
SPLINE_DEGREE = 3              # Degree of spline (3 = cubic, recommended - do not change)

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
X_MAX = X_AXIS_THRESHOLD_KM    # Maximum distance to plot (km) - uses threshold defined above
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

# Note: Output paths will be generated per variable (e.g., h9_d111_curve.csv, h9_d104_curve.csv, etc.)

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
    
    print(f"    → Computing {n_bootstrap} bootstrap samples...")
    
    for i in range(n_bootstrap):
        if (i + 1) % 100 == 0:
            print(f"      Bootstrap sample {i+1}/{n_bootstrap}")
        
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
    print(f"\nVariables to analyze: {', '.join(VARIABLES_TO_PLOT)}")
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
    
    # Filter by date range if specified
    if CONFLICT_DATE_MIN is not None or CONFLICT_DATE_MAX is not None:
        # Try to find date column (common names in ACLED data)
        date_cols = ["EVENT_DATE", "event_date", "date", "DATE", "event_dt", "EVENT_DT"]
        date_col = None
        for col in date_cols:
            if col in acled_valid.columns:
                date_col = col
                break
        
        if date_col is None:
            print(f"  ⚠ WARNING: Could not find date column for filtering")
            print(f"    Looked for: {date_cols}")
            print(f"    Available columns: {list(acled_valid.columns)[:20]}...")
            print(f"    Proceeding without date filtering")
        else:
            print(f"  → Found date column: {date_col}")
            
            # Convert to datetime
            acled_valid[date_col] = pd.to_datetime(acled_valid[date_col], errors='coerce')
            n_before_date = len(acled_valid)
            
            # Apply filters
            if CONFLICT_DATE_MIN is not None:
                date_min = pd.to_datetime(CONFLICT_DATE_MIN)
                acled_valid = acled_valid[acled_valid[date_col] >= date_min]
                print(f"  → Filtered to events on or after {CONFLICT_DATE_MIN}")
            
            if CONFLICT_DATE_MAX is not None:
                date_max = pd.to_datetime(CONFLICT_DATE_MAX)
                acled_valid = acled_valid[acled_valid[date_col] < date_max]
                print(f"  → Filtered to events before {CONFLICT_DATE_MAX}")
            
            n_after_date = len(acled_valid)
            print(f"  → After date filtering: {n_after_date:,} events ({n_after_date/n_before_date*100:.1f}%)")
    
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
    
    # Check all IPV variables exist
    missing_vars = []
    for var in VARIABLES_TO_PLOT:
        if var.lower() not in dhs.columns:
            missing_vars.append(var)
    
    if missing_vars:
        print(f"ERROR: Variables not found in DHS columns: {missing_vars}")
        print(f"Available columns: {list(dhs.columns)[:50]}...")
        sys.exit(1)
    
    # Check if weight variable exists (if weighting is enabled)
    if USE_WEIGHTS:
        weight_var = WEIGHT_VARIABLE.lower()
        if weight_var not in dhs.columns:
            print(f"ERROR: Weight variable '{weight_var}' not found")
            print(f"Available columns: {list(dhs.columns)[:20]}...")
            sys.exit(1)
    
    print(f"  → All required variables found")
    
    # -------------------------------------------------------------------------
    # STEP 4: COMPUTE DISTANCE TO NEAREST CONFLICT FOR ALL WOMEN
    # -------------------------------------------------------------------------
    print("\n[STEP 4/8] Computing distance to nearest conflict for all women...")
    print("-" * 70)
    
    # Filter to women with valid GPS
    dhs_with_gps = dhs[
        dhs[lat_col].notna() & dhs[lon_col].notna()
    ].copy()
    
    print(f"\nWomen with valid GPS: {len(dhs_with_gps):,}")
    print(f"Computing distances (this may take a minute)...")
    
    # Compute minimum distance for each woman
    dhs_with_gps["distance_to_conflict_km"] = dhs_with_gps.apply(
        lambda row: compute_min_distance_to_conflicts(
            row[lat_col], row[lon_col],
            conflict_lats, conflict_lons
        ),
        axis=1
    )
    
    print(f"  → Distance range: {dhs_with_gps['distance_to_conflict_km'].min():.1f} to {dhs_with_gps['distance_to_conflict_km'].max():.1f} km")
    print(f"  → Median distance: {dhs_with_gps['distance_to_conflict_km'].median():.1f} km")
    print(f"  → Mean distance: {dhs_with_gps['distance_to_conflict_km'].mean():.1f} km")
    
    # Filter to maximum distance if specified
    if MAX_CONFLICT_DISTANCE_KM is not None:
        n_before = len(dhs_with_gps)
        dhs_with_gps = dhs_with_gps[dhs_with_gps["distance_to_conflict_km"] <= MAX_CONFLICT_DISTANCE_KM]
        n_after = len(dhs_with_gps)
        print(f"  → Filtered to distances ≤ {MAX_CONFLICT_DISTANCE_KM} km: {n_after:,} women ({n_after/n_before*100:.1f}%)")
    
    # -------------------------------------------------------------------------
    # STEP 5: PROCESS EACH VARIABLE
    # -------------------------------------------------------------------------
    print("\n[STEP 5/8] Processing variables...")
    print("-" * 70)
    
    output_files = []
    
    for var_idx, var in enumerate(VARIABLES_TO_PLOT, 1):
        var_lower = var.lower()
        var_desc = VARIABLE_DESCRIPTIONS.get(var, var)
        
        print(f"\n{'='*70}")
        print(f"[{var_idx}/{len(VARIABLES_TO_PLOT)}] Processing {var} ({var_desc})")
        print(f"{'='*70}")
        
        # Generate output paths for this variable
        OUTPUT_CSV = OUTPUT_DIR / f"h9_{var_lower}_curve.csv"
        OUTPUT_JPG = OUTPUT_DIR / f"h9_{var_lower}_curve.jpg"
        OUTPUT_PDF = OUTPUT_DIR / f"h9_{var_lower}_curve.pdf"
        
        # Filter to valid responses for this variable
        print(f"\nFiltering to valid {var} responses...")
        
        # Get coding rules for this variable
        if var.upper() in VARIABLE_CODING:
            coding = VARIABLE_CODING[var.upper()]
            yes_values = coding["yes"]
            no_values = coding["no"]
            
            # Convert to numeric
            dhs_with_gps[var_lower] = pd.to_numeric(dhs_with_gps[var_lower], errors='coerce')
            
            # Filter to rows with valid responses (either yes or no values)
            valid_values = yes_values + no_values
            dhs_valid = dhs_with_gps[
                dhs_with_gps[var_lower].notna() &
                dhs_with_gps[var_lower].isin(valid_values)
            ].copy()
            
            # Recode to binary: 1 if in yes_values, 0 if in no_values
            dhs_valid[var_lower] = dhs_valid[var_lower].isin(yes_values).astype(int)
            
            print(f"  → Recoded {var}: {yes_values} → Yes (1), {no_values} → No (0)")
            
        else:
            # Fallback: assume standard binary (0 or 1)
            print(f"  ⚠ No coding rules found for {var}, using standard binary (0/1)")
            dhs_valid = dhs_with_gps[
                dhs_with_gps[var_lower].notna() &
                dhs_with_gps[var_lower].isin([0, 1])
            ].copy()
        
        if len(dhs_valid) == 0:
            print(f"  ⚠ No valid data for {var}, skipping...")
            continue
        
        print(f"  → {len(dhs_valid):,} women with valid {var.upper()} response")
        print(f"  → IPV prevalence: {dhs_valid[var_lower].mean()*100:.1f}%")
        
        # Prepare data for visualization
        print(f"\nPreparing data for visualization...")
        
        # Extract arrays for plotting
        x_data = dhs_valid["distance_to_conflict_km"].to_numpy()
        y_data = dhs_valid[var_lower].to_numpy()
        
        # Prepare weights
        if USE_WEIGHTS:
            weight_var = WEIGHT_VARIABLE.lower()
            weights = pd.to_numeric(dhs_valid[weight_var], errors="coerce") / 1_000_000.0
            weights = weights.fillna(1.0).to_numpy()
            print(f"  → Using weighted analysis with {weight_var.upper()}")
        else:
            weights = np.ones(len(dhs_valid))
            print(f"  → Using unweighted analysis (equal weights)")
        
        # Normalize weights to sum to 1 (for proper probability estimates)
        weights = weights / weights.sum()
        
        print(f"  → Final sample size: {len(x_data):,} women")
        print(f"  → IPV responses: {int(y_data.sum())} Yes, {int((1-y_data).sum())} No")
        
        # Fit smooth curve with confidence intervals
        print(f"\nFitting smooth curve and computing confidence intervals...")
        
        # Define evaluation points for the curve
        x_eval = np.linspace(X_MIN, min(X_MAX, x_data.max()), 200)
        
        # Fit main curve
        print(f"  → Fitting main curve...")
        _, y_fit = fit_weighted_smooth_curve(x_data, y_data, weights, 
                                             s=CURVE_SMOOTHNESS, k=SPLINE_DEGREE, 
                                             x_eval=x_eval)
        
        # Compute bootstrap confidence intervals
        print(f"  → Computing bootstrap confidence intervals...")
        lower_ci, upper_ci = bootstrap_confidence_interval(
            x_data, y_data, weights,
            s=CURVE_SMOOTHNESS, k=SPLINE_DEGREE,
            x_eval=x_eval,
            n_bootstrap=N_BOOTSTRAP,
            confidence_level=CONFIDENCE_LEVEL
        )
        
        print(f"  ✓ Curve fitting complete")
        
        # Create visualization and save outputs
        print(f"\nCreating visualization and saving outputs...")
        
        # Save raw data to CSV
        print(f"  → Saving raw data to CSV...")
        output_data = dhs_valid[[lat_col, lon_col, var_lower, "distance_to_conflict_km"]].copy()
        output_data["woman_id"] = range(1, len(output_data) + 1)
        output_data = output_data[["woman_id", "distance_to_conflict_km", var_lower]]
        output_data.columns = ["woman_id", "distance_km", "ipv_response"]
        output_data.to_csv(OUTPUT_CSV, index=False)
        print(f"    ✓ Saved: {OUTPUT_CSV.name}")
        
        # Create visualization
        print(f"  → Creating visualization...")
        
        fig, ax = plt.subplots(figsize=FIGSIZE)
        
        # Scatter plot of individual responses
        # Separate colors for Yes (1) and No (0)
        mask_no = y_data == 0
        mask_yes = y_data == 1
        
        ax.scatter(x_data[mask_no], y_data[mask_no], 
                  s=MARKER_SIZE, alpha=MARKER_ALPHA, 
                  c=COLOR_NO, label=f'No {var_desc} (0)', zorder=2)
        
        ax.scatter(x_data[mask_yes], y_data[mask_yes], 
                  s=MARKER_SIZE, alpha=MARKER_ALPHA, 
                  c=COLOR_YES, label=f'Yes {var_desc} (1)', zorder=2)
        
        # Confidence interval band
        ax.fill_between(x_eval, lower_ci, upper_ci, 
                        alpha=CI_ALPHA, color='gray', 
                        label=f'{CONFIDENCE_LEVEL*100:.0f}% CI (Bootstrap)', zorder=3)
        
        # Smooth curve
        ax.plot(x_eval, y_fit, 
               linewidth=LINE_WIDTH, color=COLOR_LINE, 
               label='Smooth curve (weighted spline)', zorder=4)
        
        # Formatting
        ax.set_xlabel(X_LABEL, fontsize=12, fontweight='bold')
        ax.set_ylabel(Y_LABEL, fontsize=12, fontweight='bold')
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(Y_TICK_LABELS)
        
        # Title
        title = f'{var_desc} ({var}) vs Distance to Nearest Conflict\n'
        title += f'n={len(dhs_valid):,} women'
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
        
        output_files.append((var, var_desc, OUTPUT_CSV, OUTPUT_JPG, OUTPUT_PDF, dhs_valid))
    
    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONFLICT-IPV ANALYSIS COMPLETE!")
    print("=" * 70)
    
    print(f"\n📊 OUTPUT FILES (saved in {OUTPUT_DIR.name}/):\n")
    
    file_num = 1
    for var, var_desc, csv_path, jpg_path, pdf_path, _ in output_files:
        print(f"  {file_num}. {csv_path.name}")
        print(f"     → Data for {var_desc} ({var})\n")
        file_num += 1
        
        print(f"  {file_num}. {jpg_path.name}")
        print(f"     → Plot for {var_desc} ({var}) (JPG)\n")
        file_num += 1
        
        print(f"  {file_num}. {pdf_path.name}")
        print(f"     → Plot for {var_desc} ({var}) (PDF)\n")
        file_num += 1
    
    print(f"⚙️  ANALYSIS SETTINGS:")
    print(f"  → Variables analyzed: {', '.join(VARIABLES_TO_PLOT)}")
    print(f"  → X-axis threshold: {X_AXIS_THRESHOLD_KM} km")
    print(f"  → Curve smoothness: {CURVE_SMOOTHNESS}")
    if CONFLICT_DATE_MIN is not None or CONFLICT_DATE_MAX is not None:
        print(f"  → Conflict date range: {CONFLICT_DATE_MIN or 'earliest'} to {CONFLICT_DATE_MAX or 'latest'}")
    if USE_WEIGHTS:
        print(f"  → Weighting: {WEIGHT_VARIABLE.upper()}")
    else:
        print(f"  → Weighting: None (unweighted)")
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