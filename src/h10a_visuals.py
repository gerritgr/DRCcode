#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
H10: Drought Event Visualization - DRC Region
================================================================================

WHAT THIS SCRIPT DOES:
----------------------
Creates visualizations of extreme drought events in the DRC region by:
1. Loading SPEI-01 drought index data from CSV (1-month timescale)
2. Identifying extreme drought days (SPEI <= -2.33) for each spatial pixel
3. Counting total extreme drought days per pixel using nearest neighbor
4. Visualizing on a map with color intensity showing drought frequency

DROUGHT INDEX:
- SPEI (Standardized Precipitation-Evapotranspiration Index)
- 1-month timescale (SPEI-01)
- Extreme drought threshold: SPEI <= -1.65 (approximately 5% probability)
- Negative values indicate drier than normal conditions

VISUALIZATION:
- Background map of DRC region
- Pixel colors show number of extreme drought days
- Red colormap (rocket): darker = more drought days
- Optional legend showing the color scale

INPUTS:
-------
- DATA/Drought/DRC_spei01_clean.csv (SPEI drought index at grid points over time)
  Format: long,lat,time_indicator,drought_value,is_extreme_drought
  Generated from SPEI-01 NetCDF data, extreme drought = SPEI <= -2.33
- DATA/background.png (optional background map)

OUTPUTS:
--------
- h10_drought_months.png (visualization - PNG format, 300 DPI)
- h10_drought_months.pdf (visualization - PDF format, vector)
- h10_drought_summary.csv (pixel-level summary: lon, lat, extreme_drought_months)

All outputs are saved in the output/ directory.

USAGE:
------
Run from the project root directory:
    python src/h10_visual.py

REQUIREMENTS:
-------------
- pandas (data manipulation)
- numpy (numerical operations)
- matplotlib (plotting)
- seaborn (color palettes - for 'rocket' colormap)
- scipy (nearest neighbor interpolation)
- geopandas (DRC boundary filtering - optional, only if MASK_TO_DRC_BOUNDARY = True)
- requests (to download boundary file if needed)

Install with:
    pip install pandas numpy matplotlib seaborn scipy geopandas requests

================================================================================
"""

from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import cKDTree
from datetime import datetime
import json
import requests

# =============================================================================
# CONFIGURATION
# =============================================================================

# INPUT PATHS
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Input files
DATA_DIR = PROJECT_ROOT / "DATA"
DROUGHT_DIR = DATA_DIR / "Drought"
INPUT_DROUGHT_CSV = DROUGHT_DIR / "DRC_spei01_clean.csv"

# DRC boundary data (Natural Earth GeoJSON)
BOUNDARY_DIR = DROUGHT_DIR / "boundaries"
DRC_BOUNDARY_FILE = BOUNDARY_DIR / "drc_boundary.geojson"
NATURAL_EARTH_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson"

# Background map (optional)
BACKGROUND_MAP = DATA_DIR / "background.png"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# AREA OF INTEREST (AOI) - DRC Region
# Coordinates define the bounding box for the visualization
AOI_MIN_LON = 11.893979235367297
AOI_MAX_LON = 31.616531541802978
AOI_MIN_LAT = -13.981788316118982
AOI_MAX_LAT = 5.811825978871415

# TIME RANGE FOR FILTERING
# Define which drought measurements to include based on their date
# Format: "YYYY-MM-DD" or None (None = no limit)
DATE_MIN = None              # Lower bound (None = include all early dates)
DATE_MAX = None              # Upper bound (None = include all late dates)

# VISUALIZATION SETTINGS

# DRC Boundary Masking
# Options for handling pixels outside DRC borders:
# - "none": Show all pixels in the AOI rectangle (no boundary filtering)
# - "mask": Only show pixels within DRC borders (pixels outside are NaN/white)
# - "fade": Show all pixels, but drastically reduce opacity outside DRC borders
MASK_TO_DRC_BOUNDARY = "fade"  # Options: "none", "mask", "fade"

# Opacity for pixels outside DRC when MASK_TO_DRC_BOUNDARY = "fade"
OUTSIDE_DRC_ALPHA = 0.2         # Very low opacity for non-DRC pixels (0.1 = 10%)

# Raster resolution for the output grid
# Higher = more detailed but slower computation
RASTER_WIDTH = 400           # Number of pixels in longitude direction
RASTER_HEIGHT = 400          # Number of pixels in latitude direction

# Nearest neighbor settings
# How many nearest drought measurement points to consider for each pixel
N_NEIGHBORS = 1              # Use 1 for pure nearest neighbor (recommended)

# Maximum distance to consider a measurement valid for a pixel (in degrees)
# Points farther than this from any measurement will be NaN (appear as white/gaps on map)
# Set to a large value for true nearest neighbor (no white areas)
# If you see white gaps, increase this value
# Typical drought data grid spacing is 0.5° or 0.25°, so 10° ensures full coverage
MAX_DISTANCE_DEG = 10.0      # Large value to ensure all pixels get assigned a value

# Color settings
COLORMAP = "rocket"          # Matplotlib/Seaborn colormap for drought intensity
                            # Seaborn colormaps: "rocket", "mako", "flare", "crest"
                            # Matplotlib colormaps: "Reds", "YlOrRd", "hot", "inferno"
                            # Note: seaborn colormaps require seaborn import
DROUGHT_ALPHA = 0.7         # Transparency of drought overlay (0=invisible, 1=opaque)
BACKGROUND_ALPHA = 0.9      # Transparency of background map

# Legend settings
SHOW_LEGEND = True          # Whether to show the colorbar legend
LEGEND_LABEL = "Extreme Drought Months"  # Label for the colorbar (SPEI-01 is monthly)

# Figure settings
FIGURE_WIDTH = 12           # Width of output figure (inches)
FIGURE_HEIGHT = 10          # Height of output figure (inches)
DPI = 300                   # Resolution of output images

# Grid settings
SHOW_GRID = True            # Whether to show grid lines on the map
GRID_ALPHA = 0.3            # Transparency of grid lines

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def download_drc_boundary(output_path):
    """
    Download DRC boundary from Natural Earth and save as GeoJSON.
    
    Parameters:
    -----------
    output_path : Path
        Where to save the DRC boundary GeoJSON
    
    Returns:
    --------
    bool : True if successful, False otherwise
    """
    try:
        print(f"  → Downloading Natural Earth country boundaries...")
        response = requests.get(NATURAL_EARTH_URL, timeout=30)
        response.raise_for_status()
        
        # Parse the full countries GeoJSON
        countries = json.loads(response.text)
        
        # Find DRC (try multiple possible names)
        drc_names = [
            "Democratic Republic of the Congo",
            "Congo (Kinshasa)",
            "Congo, The Democratic Republic of the"
        ]
        
        drc_feature = None
        for feature in countries.get("features", []):
            name = feature.get("properties", {}).get("NAME", "")
            name_long = feature.get("properties", {}).get("NAME_LONG", "")
            
            if any(drc_name in name or drc_name in name_long for drc_name in drc_names):
                drc_feature = feature
                print(f"    → Found DRC: {name}")
                break
        
        if drc_feature is None:
            print(f"    ✗ Could not find DRC in Natural Earth data")
            return False
        
        # Create a GeoJSON with just DRC
        drc_geojson = {
            "type": "FeatureCollection",
            "features": [drc_feature]
        }
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(drc_geojson, f)
        
        print(f"    ✓ Saved DRC boundary to {output_path.name}")
        return True
        
    except Exception as e:
        print(f"    ✗ Error downloading boundary: {e}")
        return False


def load_drc_boundary():
    """
    Load DRC boundary from GeoJSON file, downloading if necessary.
    
    Returns:
    --------
    geopandas.GeoDataFrame or None : DRC boundary, or None if unavailable
    """
    # Check if boundary file exists
    if not DRC_BOUNDARY_FILE.exists():
        print(f"  → DRC boundary file not found, attempting download...")
        success = download_drc_boundary(DRC_BOUNDARY_FILE)
        if not success:
            print(f"  ⚠ Could not download DRC boundary")
            return None
    
    # Try to load the boundary
    try:
        import geopandas as gpd
        drc_boundary = gpd.read_file(DRC_BOUNDARY_FILE)
        print(f"  ✓ Loaded DRC boundary from {DRC_BOUNDARY_FILE.name}")
        return drc_boundary
    except ImportError:
        print(f"  ⚠ geopandas not installed - cannot use boundary masking")
        print(f"    Install with: pip install geopandas")
        return None
    except Exception as e:
        print(f"  ⚠ Error loading DRC boundary: {e}")
        return None


def get_unified_boundary_geometry(drc_boundary):
    """
    Return one unified geometry for the DRC boundary.
    
    Uses union_all() when available (preferred in recent GeoPandas/Shapely),
    and falls back to unary_union for older versions.
    """
    # Newer GeoPandas API (preferred)
    if hasattr(drc_boundary, "union_all"):
        return drc_boundary.union_all()
    if hasattr(drc_boundary, "geometry") and hasattr(drc_boundary.geometry, "union_all"):
        return drc_boundary.geometry.union_all()
    
    # Backward compatibility for older versions
    if hasattr(drc_boundary, "unary_union"):
        return drc_boundary.unary_union
    if hasattr(drc_boundary, "geometry") and hasattr(drc_boundary.geometry, "unary_union"):
        return drc_boundary.geometry.unary_union
    
    raise AttributeError("Could not compute unified boundary geometry from drc_boundary")


def create_alpha_mask_for_drc(grid_lon, grid_lat, drc_boundary, inside_alpha=1.0, outside_alpha=0.1):
    """
    Create an alpha (transparency) mask for grid, with different opacities inside/outside DRC.
    
    Parameters:
    -----------
    grid_lon, grid_lat : np.ndarray
        2D arrays of longitude and latitude values
    drc_boundary : geopandas.GeoDataFrame
        DRC boundary polygon
    inside_alpha : float
        Opacity for pixels inside DRC (default: 1.0 = fully opaque)
    outside_alpha : float
        Opacity for pixels outside DRC (default: 0.1 = very transparent)
    
    Returns:
    --------
    np.ndarray : 2D array of alpha values (same shape as grid)
    """
    try:
        from shapely.geometry import Point
        
        print(f"  → Creating alpha mask for DRC boundary...")
        print(f"    Inside DRC: {inside_alpha*100:.0f}% opacity")
        print(f"    Outside DRC: {outside_alpha*100:.0f}% opacity")
        
        # Get the unified DRC geometry
        drc_geom = get_unified_boundary_geometry(drc_boundary)
        
        # Create alpha mask (same shape as grid)
        height, width = grid_lon.shape
        alpha_mask = np.zeros((height, width))
        
        n_inside = 0
        
        # Check each pixel
        for i in range(height):
            for j in range(width):
                lon = grid_lon[i, j]
                lat = grid_lat[i, j]
                point = Point(lon, lat)
                
                # Set alpha based on whether point is inside DRC
                if drc_geom.contains(point):
                    alpha_mask[i, j] = inside_alpha
                    n_inside += 1
                else:
                    alpha_mask[i, j] = outside_alpha
        
        total_pixels = height * width
        print(f"    → {n_inside:,} / {total_pixels:,} pixels inside DRC ({n_inside/total_pixels*100:.1f}%)")
        
        return alpha_mask
        
    except Exception as e:
        print(f"  ⚠ Error creating alpha mask: {e}")
        print(f"    Returning uniform alpha mask")
        return np.ones_like(grid_lon)


def mask_grid_to_drc(grid_lon, grid_lat, grid_values, drc_boundary):
    """
    Mask grid values to only show pixels within DRC boundary.
    
    Parameters:
    -----------
    grid_lon, grid_lat : np.ndarray
        2D arrays of longitude and latitude values
    grid_values : np.ndarray
        2D array of values to mask
    drc_boundary : geopandas.GeoDataFrame
        DRC boundary polygon
    
    Returns:
    --------
    np.ndarray : Masked grid values (NaN outside DRC)
    """
    try:
        from shapely.geometry import Point
        
        print(f"  → Masking grid to DRC boundary...")
        
        # Get the unified DRC geometry
        drc_geom = get_unified_boundary_geometry(drc_boundary)
        
        # Create a copy of grid_values to mask
        masked_values = grid_values.copy()
        
        # Check each pixel
        height, width = grid_values.shape
        n_inside = 0
        
        for i in range(height):
            for j in range(width):
                lon = grid_lon[i, j]
                lat = grid_lat[i, j]
                point = Point(lon, lat)
                
                # If point is outside DRC, set to NaN
                if not drc_geom.contains(point):
                    masked_values[i, j] = np.nan
                else:
                    n_inside += 1
        
        total_pixels = height * width
        print(f"    → {n_inside:,} / {total_pixels:,} pixels inside DRC ({n_inside/total_pixels*100:.1f}%)")
        
        return masked_values
        
    except Exception as e:
        print(f"  ⚠ Error masking to DRC boundary: {e}")
        print(f"    Returning unmasked values")
        return grid_values


def load_drought_data(csv_path, date_min=None, date_max=None):
    """
    Load drought data from CSV and filter by date range.
    
    Parameters:
    -----------
    csv_path : Path
        Path to the drought data CSV file
    date_min : str or None
        Minimum date to include (format: "YYYY-MM-DD")
    date_max : str or None
        Maximum date to include (format: "YYYY-MM-DD")
    
    Returns:
    --------
    pd.DataFrame : Drought data with columns: long, lat, time_indicator, 
                   drought_value, is_extreme_drought
    """
    print(f"  → Reading CSV...")
    df = pd.read_csv(csv_path)
    
    # Verify required columns exist
    required_cols = ["long", "lat", "time_indicator", "is_extreme_drought"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    
    print(f"  → Loaded {len(df):,} drought measurements")
    
    # Convert coordinates to numeric
    df["long"] = pd.to_numeric(df["long"], errors="coerce")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    
    # Convert time_indicator to datetime
    df["time_indicator"] = pd.to_datetime(df["time_indicator"], errors="coerce")
    
    # Convert is_extreme_drought to boolean
    if df["is_extreme_drought"].dtype == object:
        df["is_extreme_drought"] = df["is_extreme_drought"].astype(str).str.lower() == "true"
    else:
        df["is_extreme_drought"] = df["is_extreme_drought"].astype(bool)
    
    # Filter to valid data
    df = df[
        df["long"].notna() & 
        df["lat"].notna() & 
        df["time_indicator"].notna()
    ].copy()
    
    print(f"  → {len(df):,} measurements with valid coordinates and dates")
    
    # Filter by date range if specified
    n_before = len(df)
    if date_min is not None:
        date_min_dt = pd.to_datetime(date_min)
        df = df[df["time_indicator"] >= date_min_dt]
        print(f"  → Filtered to dates on or after {date_min}")
    
    if date_max is not None:
        date_max_dt = pd.to_datetime(date_max)
        df = df[df["time_indicator"] <= date_max_dt]
        print(f"  → Filtered to dates on or before {date_max}")
    
    n_after = len(df)
    if n_before != n_after:
        print(f"  → After date filtering: {n_after:,} measurements ({n_after/n_before*100:.1f}%)")
    
    return df


def filter_to_aoi(df, lon_col="long", lat_col="lat"):
    """
    Filter drought data to Area of Interest (AOI).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Drought data
    lon_col, lat_col : str
        Names of longitude and latitude columns
    
    Returns:
    --------
    pd.DataFrame : Filtered data within AOI bounds
    """
    df_aoi = df[
        (df[lon_col] >= AOI_MIN_LON) &
        (df[lon_col] <= AOI_MAX_LON) &
        (df[lat_col] >= AOI_MIN_LAT) &
        (df[lat_col] <= AOI_MAX_LAT)
    ].copy()
    
    return df_aoi


def count_extreme_drought_days(df):
    """
    Count extreme drought occurrences per spatial location.
    
    For each unique (lon, lat) location, counts how many time periods had extreme drought.
    Note: These are monthly measurements (SPEI-01), so output is in months.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Drought data with columns: long, lat, time_indicator, is_extreme_drought
    
    Returns:
    --------
    pd.DataFrame : Summary with columns: long, lat, extreme_drought_months
    """
    print(f"  → Counting extreme drought occurrences per location...")
    
    # Group by location and count extreme drought occurrences
    summary = df.groupby(["long", "lat"]).agg(
        total_measurements=("is_extreme_drought", "size"),
        extreme_drought_months=("is_extreme_drought", "sum")
    ).reset_index()
    
    # Convert boolean sum to integer
    summary["extreme_drought_months"] = summary["extreme_drought_months"].astype(int)
    
    print(f"  → Found {len(summary):,} unique spatial locations")
    print(f"  → Max extreme drought occurrences at any location: {summary['extreme_drought_months'].max()}")
    print(f"  → Mean extreme drought occurrences: {summary['extreme_drought_months'].mean():.1f}")
    
    return summary


def interpolate_to_grid(summary_df, width=RASTER_WIDTH, height=RASTER_HEIGHT, 
                        n_neighbors=N_NEIGHBORS, max_distance=MAX_DISTANCE_DEG):
    """
    Interpolate point drought data to a regular grid using nearest neighbors.
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Summary data with columns: long, lat, extreme_drought_days
    width, height : int
        Grid dimensions in pixels
    n_neighbors : int
        Number of nearest neighbors to use
    max_distance : float
        Maximum distance (in degrees) to consider a measurement valid
    
    Returns:
    --------
    tuple : (grid_lon, grid_lat, grid_values)
        - grid_lon: 2D array of longitude values
        - grid_lat: 2D array of latitude values
        - grid_values: 2D array of extreme drought month counts
    """
    print(f"  → Interpolating to {width}×{height} grid using nearest neighbor...")
    
    # Create regular grid
    lon_lin = np.linspace(AOI_MIN_LON, AOI_MAX_LON, width)
    lat_lin = np.linspace(AOI_MAX_LAT, AOI_MIN_LAT, height)  # North to south
    grid_lon, grid_lat = np.meshgrid(lon_lin, lat_lin)
    
    # Flatten grid for processing
    grid_points = np.column_stack([grid_lon.ravel(), grid_lat.ravel()])
    
    # Build KD-tree for fast nearest neighbor search
    measurement_points = summary_df[["long", "lat"]].to_numpy()
    values = summary_df["extreme_drought_months"].to_numpy()
    
    tree = cKDTree(measurement_points)
    
    # Query nearest neighbors for each grid point
    distances, indices = tree.query(grid_points, k=n_neighbors)
    
    # If using multiple neighbors, average them
    if n_neighbors == 1:
        grid_values_flat = values[indices].astype(float)  # Convert to float to allow NaN
        grid_distances_flat = distances
    else:
        # Weighted average by inverse distance
        weights = 1.0 / (distances + 1e-10)
        weight_sum = weights.sum(axis=1)
        grid_values_flat = (values[indices] * weights).sum(axis=1) / weight_sum
        grid_distances_flat = distances.min(axis=1)
    
    # Mask out points too far from any measurement
    grid_values_flat[grid_distances_flat > max_distance] = np.nan
    
    # Reshape to grid
    grid_values = grid_values_flat.reshape(height, width)
    
    n_valid = (~np.isnan(grid_values)).sum()
    print(f"  → {n_valid:,} / {width*height:,} pixels have valid data ({n_valid/(width*height)*100:.1f}%)")
    
    return grid_lon, grid_lat, grid_values


def create_visualization(grid_lon, grid_lat, grid_values, background_img=None, 
                         output_png=None, output_pdf=None, date_range=None, alpha_mask=None):
    """
    Create drought visualization map.
    
    Parameters:
    -----------
    grid_lon, grid_lat : np.ndarray
        2D arrays of longitude and latitude values
    grid_values : np.ndarray
        2D array of extreme drought day counts
    background_img : np.ndarray or None
        Optional background map image
    output_png, output_pdf : Path or None
        Output file paths
    date_range : tuple or None
        Tuple of (min_date, max_date) for subtitle
    alpha_mask : np.ndarray or None
        2D array of alpha values for variable transparency (for "fade" mode)
    """
    print(f"  → Creating visualization...")
    
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    # Add background map if available
    if background_img is not None:
        ax.imshow(
            background_img,
            extent=[AOI_MIN_LON, AOI_MAX_LON, AOI_MIN_LAT, AOI_MAX_LAT],
            origin="upper",
            zorder=0,
            alpha=BACKGROUND_ALPHA,
        )
    
    # Plot drought data
    # Use pcolormesh for efficient grid plotting
    # If alpha_mask is provided (fade mode), use it; otherwise use constant alpha
    if alpha_mask is not None:
        # Use variable alpha (fade mode)
        drought_map = ax.pcolormesh(
            grid_lon,
            grid_lat,
            grid_values,
            cmap=COLORMAP,
            alpha=alpha_mask,  # Variable alpha based on DRC boundary
            shading='auto',
            zorder=1,
        )
    else:
        # Use constant alpha
        drought_map = ax.pcolormesh(
            grid_lon,
            grid_lat,
            grid_values,
            cmap=COLORMAP,
            alpha=DROUGHT_ALPHA,
            shading='auto',
            zorder=1,
        )
    
    # Set map bounds
    ax.set_xlim(AOI_MIN_LON, AOI_MAX_LON)
    ax.set_ylim(AOI_MIN_LAT, AOI_MAX_LAT)
    ax.set_aspect('equal', adjustable='box')
    
    # Labels and formatting
    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    
    # Title with statistics
    n_valid = (~np.isnan(grid_values)).sum()
    max_months = np.nanmax(grid_values) if n_valid > 0 else 0
    mean_months = np.nanmean(grid_values) if n_valid > 0 else 0
    
    title = f'Extreme Drought Months (SPEI-01 ≤ -1.65) - DRC Region\n'
    title += f'Max: {int(max_months)} months | Mean: {mean_months:.1f} months | '
    title += f'{n_valid:,} pixels with data'
    
    # Add date range if provided
    if date_range is not None:
        min_date, max_date = date_range
        title += f'\nDate Range: {min_date.strftime("%Y-%m-%d")} to {max_date.strftime("%Y-%m-%d")}'
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add colorbar if requested
    if SHOW_LEGEND and n_valid > 0:
        cbar = plt.colorbar(drought_map, ax=ax, orientation="vertical",
                           fraction=0.046, pad=0.04)
        cbar.set_label(LEGEND_LABEL, rotation=90, fontsize=11, labelpad=15)
    
    # Add grid if requested
    if SHOW_GRID:
        ax.grid(True, alpha=GRID_ALPHA, linestyle='--', linewidth=0.5, zorder=2)
    
    plt.tight_layout()
    
    # Save outputs
    if output_png:
        plt.savefig(output_png, dpi=DPI, bbox_inches='tight')
        print(f"    ✓ Saved: {output_png.name}")
    
    if output_pdf:
        plt.savefig(output_pdf, bbox_inches='tight')
        print(f"    ✓ Saved: {output_pdf.name}")
    
    plt.close(fig)


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    """
    Main function that orchestrates the drought visualization.
    """
    
    print("\n" + "=" * 70)
    print("H10: DROUGHT EVENT VISUALIZATION - DRC Region")
    print("=" * 70)
    print(f"\nCreating map of extreme drought month frequency")
    print(f"Using SPEI-01 index (1-month timescale)")
    print(f"Extreme drought threshold: SPEI ≤ -1.65")
    print(f"Area of Interest: DRC and surrounding region")
    
    # -------------------------------------------------------------------------
    # STEP 1: VERIFY INPUT FILES
    # -------------------------------------------------------------------------
    print("\n[STEP 1/6] Verifying input files...")
    print("-" * 70)
    
    if not INPUT_DROUGHT_CSV.exists():
        print(f"ERROR: Drought CSV not found at {INPUT_DROUGHT_CSV}")
        sys.exit(1)
    
    print(f"✓ Found drought CSV: {INPUT_DROUGHT_CSV.name}")
    
    # Check for background map
    background_img = None
    if BACKGROUND_MAP.exists():
        print(f"✓ Found background map: {BACKGROUND_MAP.name}")
        background_img = plt.imread(BACKGROUND_MAP)
        print(f"  → Loaded background image")
    else:
        print(f"⚠ Background map not found at {BACKGROUND_MAP}")
        print(f"  → Map will be created without background image")
    
    # -------------------------------------------------------------------------
    # STEP 2: LOAD DROUGHT DATA
    # -------------------------------------------------------------------------
    print("\n[STEP 2/6] Loading drought data...")
    print("-" * 70)
    
    drought_df = load_drought_data(INPUT_DROUGHT_CSV, DATE_MIN, DATE_MAX)
    
    # Get date range
    min_date = drought_df["time_indicator"].min()
    max_date = drought_df["time_indicator"].max()
    print(f"  → Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
    # Get spatial extent
    min_lon = drought_df["long"].min()
    max_lon = drought_df["long"].max()
    min_lat = drought_df["lat"].min()
    max_lat = drought_df["lat"].max()
    print(f"  → Longitude range: {min_lon:.2f} to {max_lon:.2f}")
    print(f"  → Latitude range: {min_lat:.2f} to {max_lat:.2f}")
    
    # Count extreme drought occurrences
    n_extreme = drought_df["is_extreme_drought"].sum()
    n_total = len(drought_df)
    print(f"  → Extreme drought measurements: {n_extreme:,} / {n_total:,} ({n_extreme/n_total*100:.1f}%)")
    
    # -------------------------------------------------------------------------
    # STEP 3: FILTER TO AOI
    # -------------------------------------------------------------------------
    print("\n[STEP 3/6] Filtering to Area of Interest...")
    print("-" * 70)
    
    drought_aoi = filter_to_aoi(drought_df)
    print(f"  → {len(drought_aoi):,} measurements within AOI ({len(drought_aoi)/len(drought_df)*100:.1f}%)")
    
    if drought_aoi.empty:
        print("\nERROR: No drought data within the specified AOI!")
        print("Check that AOI coordinates match the drought data extent.")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # STEP 4: COUNT EXTREME DROUGHT DAYS
    # -------------------------------------------------------------------------
    print("\n[STEP 4/6] Counting extreme drought days...")
    print("-" * 70)
    
    summary_df = count_extreme_drought_days(drought_aoi)
    
    # Save summary CSV
    output_csv = OUTPUT_DIR / "h10_drought_summary.csv"
    summary_df.to_csv(output_csv, index=False)
    print(f"  ✓ Saved summary: {output_csv.name}")
    
    # -------------------------------------------------------------------------
    # STEP 5: INTERPOLATE TO GRID
    # -------------------------------------------------------------------------
    print("\n[STEP 5/6] Interpolating to regular grid...")
    print("-" * 70)
    
    grid_lon, grid_lat, grid_values = interpolate_to_grid(
        summary_df,
        width=RASTER_WIDTH,
        height=RASTER_HEIGHT,
        n_neighbors=N_NEIGHBORS,
        max_distance=MAX_DISTANCE_DEG
    )
    
    # Apply DRC boundary processing based on mode
    alpha_mask = None  # Will be set in "fade" mode
    
    if MASK_TO_DRC_BOUNDARY == "mask":
        print("\n  → Applying DRC boundary mask (hard mask)...")
        drc_boundary = load_drc_boundary()
        if drc_boundary is not None:
            grid_values = mask_grid_to_drc(grid_lon, grid_lat, grid_values, drc_boundary)
        else:
            print("  ⚠ Skipping boundary masking (boundary data unavailable)")
    
    elif MASK_TO_DRC_BOUNDARY == "fade":
        print("\n  → Applying DRC boundary fade (variable opacity)...")
        drc_boundary = load_drc_boundary()
        if drc_boundary is not None:
            alpha_mask = create_alpha_mask_for_drc(
                grid_lon, grid_lat, drc_boundary,
                inside_alpha=DROUGHT_ALPHA,
                outside_alpha=OUTSIDE_DRC_ALPHA
            )
        else:
            print("  ⚠ Skipping boundary fade (boundary data unavailable)")
    
    elif MASK_TO_DRC_BOUNDARY == "none":
        print("\n  → No DRC boundary filtering applied")
    
    else:
        print(f"\n  ⚠ Unknown MASK_TO_DRC_BOUNDARY value: '{MASK_TO_DRC_BOUNDARY}'")
        print(f"    Valid options: 'none', 'mask', 'fade'")
        print(f"    Proceeding without boundary filtering")
    
    # -------------------------------------------------------------------------
    # STEP 6: CREATE VISUALIZATION
    # -------------------------------------------------------------------------
    print("\n[STEP 6/6] Creating visualization...")
    print("-" * 70)
    
    output_png = OUTPUT_DIR / "h10_drought_months.png"
    output_pdf = OUTPUT_DIR / "h10_drought_months.pdf"
    
    create_visualization(
        grid_lon, grid_lat, grid_values,
        background_img=background_img,
        output_png=output_png,
        output_pdf=output_pdf,
        date_range=(min_date, max_date),
        alpha_mask=alpha_mask
    )
    
    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DROUGHT VISUALIZATION COMPLETE!")
    print("=" * 70)
    
    print(f"\n📊 OUTPUT FILES (saved in {OUTPUT_DIR.name}/):\n")
    
    print(f"  1. h10_drought_summary.csv")
    print(f"     → Summary data: {len(summary_df):,} spatial locations\n")
    
    print(f"  2. h10_drought_months.png")
    print(f"     → Drought map visualization (PNG)\n")
    
    print(f"  3. h10_drought_months.pdf")
    print(f"     → Drought map visualization (PDF, vector)\n")
    
    print(f"⚙️  VISUALIZATION SETTINGS:")
    print(f"  → Area of Interest: {AOI_MIN_LON:.2f} to {AOI_MAX_LON:.2f} lon, "
          f"{AOI_MIN_LAT:.2f} to {AOI_MAX_LAT:.2f} lat")
    if DATE_MIN is not None or DATE_MAX is not None:
        print(f"  → Date range: {DATE_MIN or 'earliest'} to {DATE_MAX or 'latest'}")
    print(f"  → Grid resolution: {RASTER_WIDTH}×{RASTER_HEIGHT} pixels")
    print(f"  → Nearest neighbors: {N_NEIGHBORS}")
    print(f"  → Max distance: {MAX_DISTANCE_DEG}° (~{MAX_DISTANCE_DEG*111:.0f} km)")
    print(f"  → Colormap: {COLORMAP}")
    print(f"  → Drought overlay opacity: {DROUGHT_ALPHA}")
    print(f"  → Show legend: {SHOW_LEGEND}")
    print(f"  → Resolution: {DPI} DPI")
    
    print(f"\n📈 DATA SUMMARY:")
    print(f"  → Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    print(f"  → Unique locations: {len(summary_df):,}")
    print(f"  → Max extreme drought months: {summary_df['extreme_drought_months'].max()}")
    print(f"  → Mean extreme drought months: {summary_df['extreme_drought_months'].mean():.1f}")
    print(f"  → SPEI threshold: ≤ -1.65 (extreme drought)")
    
    print("\n✅ Visualization complete!\n")


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
