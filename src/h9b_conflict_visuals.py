#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
H9B: Conflict Event Visualization - DRC Region
================================================================================

WHAT THIS SCRIPT DOES:
----------------------
Creates visualizations of armed conflict events in the DRC region with:
- Color intensity based on event date (grayish = older, blue/orange = newer)
- Blue for DRC events, Orange for non-DRC events
- Point size based on number of casualties

INPUTS:
-------
- DATA/Conflict/Africa_lagged_data_up_to-2024-10-17/Africa_lagged_data_up_to-2024-10-17.xlsx
- Optional: background map image

OUTPUTS:
--------
- h9b_all_events.png (all events on map)
- h9b_all_events.pdf (all events on map, vector format)
- h9b_drc_only.png (DRC events only)
- h9b_drc_only.pdf (DRC events only, vector format)
- h9b_non_drc_only.png (non-DRC events only)
- h9b_non_drc_only.pdf (non-DRC events only, vector format)
- h9b_filtered_events.csv (filtered event data)

All outputs are saved in the output/ directory.

USAGE:
------
Run from the project root directory:
    python src/h9b_conflict_viz.py

REQUIREMENTS:
-------------
- pandas (data manipulation)
- matplotlib (plotting)
- seaborn (color palettes)
- openpyxl (reading Excel files)

Install with:
    pip install pandas matplotlib seaborn openpyxl

================================================================================
"""

from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

# =============================================================================
# CONFIGURATION
# =============================================================================

# INPUT PATHS
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

# Input files
DATA_DIR = PROJECT_ROOT / "DATA"
DHS_DIR = DATA_DIR / "DHS"
CONFLICT_DIR = DATA_DIR / "Conflict" / "Africa_lagged_data_up_to-2024-10-17"

INPUT_ACLED_XLSX = CONFLICT_DIR / "Africa_lagged_data_up_to-2024-10-17.xlsx"

# Background map (located in DATA directory)
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
# Define which conflict events to include based on their date
# Format: "YYYY-MM-DD" or None (None = no limit)
DATE_MIN = None              # Lower bound (None = include all early events)
DATE_MAX = "2023-03-01"      # Upper bound (default: March 1, 2023)

# DRC COUNTRY NAME
# Name used in the COUNTRY column to identify DRC events
DRC_COUNTRY_NAME = "Democratic Republic of Congo"

# VISUALIZATION SETTINGS

# Point size scaling for casualties
MIN_POINT_SIZE = 20           # Minimum marker size (for 0-1 casualties)
MAX_POINT_SIZE = 100         # Maximum marker size (for high casualties)
SIZE_SCALE_POWER = 0.5       # Power for size scaling (0.5 = square root, 1.0 = linear)

# Point appearance
POINT_ALPHA = 0.15           # Transparency of points (0-1)
POINT_EDGE_WIDTH = 0.5       # Width of point edges

# Legend settings
# Options: "none", "on_map", "outside"
# "none" = no size legend, "on_map" = legend inside map, "outside" = legend outside map area
LEGEND_POSITION = "outside"

# Figure settings
FIGURE_WIDTH = 12            # Width of output figures (inches)
FIGURE_HEIGHT = 10           # Height of output figures (inches)
DPI = 300                    # Resolution of output images

# Color settings
# Using seaborn muted palette colors
COLOR_DRC = sns.color_palette("muted")[0]      # Blue for DRC
COLOR_NON_DRC = sns.color_palette("muted")[1]  # Orange for non-DRC

# Create color maps: gray (older) to blue/orange (more recent)
CMAP_DRC = sns.light_palette(COLOR_DRC, as_cmap=True)
CMAP_NON_DRC = sns.light_palette(COLOR_NON_DRC, as_cmap=True)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def normalize_dates(dates):
    """
    Normalize dates to 0-1 range for color mapping.
    More recent dates get higher values (darker colors).
    
    Parameters:
    -----------
    dates : pd.Series
        Series of datetime objects
    
    Returns:
    --------
    np.array : Normalized values (0-1)
    """
    if len(dates) == 0:
        return np.array([])
    
    # Convert to timestamps
    timestamps = dates.astype(np.int64) / 10**9  # Convert to seconds
    
    # Normalize to 0-1 (most recent = 1)
    min_ts = timestamps.min()
    max_ts = timestamps.max()
    
    if min_ts == max_ts:
        return np.ones(len(dates))
    
    normalized = (timestamps - min_ts) / (max_ts - min_ts)
    return normalized


def scale_casualties(casualties, min_size=MIN_POINT_SIZE, max_size=MAX_POINT_SIZE, power=SIZE_SCALE_POWER):
    """
    Scale casualty numbers to point sizes using power scaling.
    
    Parameters:
    -----------
    casualties : pd.Series or np.array
        Number of casualties per event
    min_size : float
        Minimum point size
    max_size : float
        Maximum point size
    power : float
        Scaling power (0.5 = sqrt, 1.0 = linear)
    
    Returns:
    --------
    np.array : Scaled point sizes
    """
    casualties = np.array(casualties, dtype=float)
    
    # Handle zero/missing casualties
    casualties = np.maximum(casualties, 0)
    
    # Apply power scaling
    scaled = np.power(casualties, power)
    
    # Normalize to size range
    if scaled.max() > 0:
        scaled = min_size + (scaled / scaled.max()) * (max_size - min_size)
    else:
        scaled = np.full_like(scaled, min_size)
    
    return scaled


def plot_events_on_map(
    data,
    output_path,
    title,
    background_img=None,
    show_drc=True,
    show_non_drc=True,
):
    """
    Create a map visualization of conflict events.
    
    Parameters:
    -----------
    data : pd.DataFrame
        Filtered conflict event data
    output_path : Path
        Where to save the output image
    title : str
        Title for the plot
    background_img : np.array or None
        Optional background map image
    show_drc : bool
        Whether to show DRC events
    show_non_drc : bool
        Whether to show non-DRC events
    """
    fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
    
    # Add background map if available
    if background_img is not None:
        ax.imshow(
            background_img,
            extent=[AOI_MIN_LON, AOI_MAX_LON, AOI_MIN_LAT, AOI_MAX_LAT],
            origin="upper",
            zorder=0,
            alpha=0.5,
        )
    
    if not data.empty:
        # Separate DRC and non-DRC events
        # Convert to string type for reliable comparison
        is_drc = data["COUNTRY"].astype("string") == DRC_COUNTRY_NAME
        drc_events = data[is_drc].copy()
        non_drc_events = data[~is_drc].copy()
        
        # Sort by date so older events are plotted first (newer events on top)
        if not drc_events.empty:
            drc_events = drc_events.sort_values("EVENT_DATE")
        if not non_drc_events.empty:
            non_drc_events = non_drc_events.sort_values("EVENT_DATE")
        
        # Track scatter objects for colorbars
        scatter_drc = None
        scatter_other = None
        
        # Plot DRC events (blue)
        if show_drc and not drc_events.empty:
            # Normalize dates for color
            date_colors = normalize_dates(drc_events["EVENT_DATE"])
            
            # Scale casualties for size
            sizes = scale_casualties(drc_events["FATALITIES"])
            
            scatter_drc = ax.scatter(
                drc_events["LONGITUDE"],
                drc_events["LATITUDE"],
                s=sizes,
                c=date_colors,
                cmap=CMAP_DRC,
                alpha=POINT_ALPHA,
                linewidths=POINT_EDGE_WIDTH,
                edgecolors='white',
                vmin=0,
                vmax=1,
                label=f'DRC Events (n={len(drc_events):,})',
                zorder=2,
            )
        
        # Plot non-DRC events (orange)
        if show_non_drc and not non_drc_events.empty:
            # Normalize dates for color
            date_colors = normalize_dates(non_drc_events["EVENT_DATE"])
            
            # Scale casualties for size
            sizes = scale_casualties(non_drc_events["FATALITIES"])
            
            scatter_other = ax.scatter(
                non_drc_events["LONGITUDE"],
                non_drc_events["LATITUDE"],
                s=sizes,
                c=date_colors,
                cmap=CMAP_NON_DRC,
                alpha=POINT_ALPHA,
                linewidths=POINT_EDGE_WIDTH,
                edgecolors='white',
                vmin=0,
                vmax=1,
                label=f'Non-DRC Events (n={len(non_drc_events):,})',
                zorder=3,
            )
        
        # Add colorbars for time-based coloring
        # Position them vertically stacked on the right side
        # For all_events plot, only show DRC colorbar
        show_both_colorbars = show_drc and show_non_drc and (len(drc_events) > 0 and len(non_drc_events) > 0)
        
        if scatter_drc is not None:
            # Get date range for DRC events
            min_date_drc = drc_events["EVENT_DATE"].min()
            max_date_drc = drc_events["EVENT_DATE"].max()
            
            # Create colorbar for DRC (blue)
            cbar_drc = plt.colorbar(
                scatter_drc, 
                ax=ax, 
                orientation="vertical",
                fraction=0.046,
                pad=0.04,
                aspect=20,
            )
            cbar_drc.set_label(
                f"DRC Events - Time\n{min_date_drc.strftime('%Y-%m-%d')} to {max_date_drc.strftime('%Y-%m-%d')}\n(gray = older, blue = newer)",
                rotation=90,
                fontsize=9,
                labelpad=10
            )
            cbar_drc.ax.yaxis.set_ticks([0, 0.5, 1])
            cbar_drc.ax.yaxis.set_ticklabels(['Older', 'Mid', 'Recent'])
        
        # Only add non-DRC colorbar if not showing both types (i.e., for single-type plots)
        if scatter_other is not None and not show_both_colorbars:
            # Get date range for non-DRC events
            min_date_other = non_drc_events["EVENT_DATE"].min()
            max_date_other = non_drc_events["EVENT_DATE"].max()
            
            # Create colorbar for non-DRC (orange)
            # Position it below the DRC colorbar if both exist
            if scatter_drc is not None:
                # Create a new axis for the second colorbar
                from mpl_toolkits.axes_grid1 import make_axes_locatable
                divider = make_axes_locatable(ax)
                cax2 = divider.append_axes("right", size="5%", pad=0.8)
                
                cbar_other = plt.colorbar(
                    scatter_other,
                    cax=cax2,
                    orientation="vertical",
                )
            else:
                cbar_other = plt.colorbar(
                    scatter_other, 
                    ax=ax, 
                    orientation="vertical",
                    fraction=0.046,
                    pad=0.04,
                    aspect=20,
                )
            
            cbar_other.set_label(
                f"Non-DRC Events - Time\n{min_date_other.strftime('%Y-%m-%d')} to {max_date_other.strftime('%Y-%m-%d')}\n(gray = older, orange = newer)",
                rotation=90,
                fontsize=9,
                labelpad=10
            )
            cbar_other.ax.yaxis.set_ticks([0, 0.5, 1])
            cbar_other.ax.yaxis.set_ticklabels(['Older', 'Mid', 'Recent'])
    
    # Set map bounds
    ax.set_xlim(AOI_MIN_LON, AOI_MAX_LON)
    ax.set_ylim(AOI_MIN_LAT, AOI_MAX_LAT)
    ax.set_aspect('equal', adjustable='box')
    
    # Labels and formatting
    ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
    ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Add size legend based on LEGEND_POSITION setting
    legend_obj = None
    if not data.empty and LEGEND_POSITION != "none":
        # Create size legend for fatalities
        # Sample 3 representative sizes
        min_fat = data["FATALITIES"].min()
        max_fat = data["FATALITIES"].max()
        
        if max_fat > min_fat:
            sample_fatalities = [
                int(min_fat) if min_fat > 0 else 1,
                int((min_fat + max_fat) / 2),
                int(max_fat)
            ]
        else:
            sample_fatalities = [int(max_fat)] if max_fat > 0 else [1]
        
        # Create sample scatter points for legend
        handles = []
        for fat in sample_fatalities:
            size = scale_casualties([fat], MIN_POINT_SIZE, MAX_POINT_SIZE, SIZE_SCALE_POWER)[0]
            handles.append(
                plt.scatter([], [], s=size, color='gray', alpha=0.6,
                            edgecolors='black', linewidths=0.5,
                            label=f'{fat:,} fatalities')
            )
        
        if LEGEND_POSITION == "on_map":
            # Legend inside the map (lower right)
            legend_obj = ax.legend(
                handles=handles,
                scatterpoints=1,
                frameon=True,
                framealpha=0.9,
                labelspacing=1.5,
                title="Fatalities per Event",
                loc="lower right",
                fontsize=9,
                title_fontsize=10
            )
        elif LEGEND_POSITION == "outside":
            # Legend outside the map (right side, positioned lower to avoid colorbar overlap)
            legend_obj = ax.legend(
                handles=handles,
                scatterpoints=1,
                frameon=True,
                framealpha=0.9,
                labelspacing=1.5,
                title="Fatalities per Event",
                loc="lower left",
                bbox_to_anchor=(1.15, 0.0),
                fontsize=9,
                title_fontsize=10
            )
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    # Save with legend if outside
    if LEGEND_POSITION == "outside" and legend_obj is not None:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight',
                    bbox_extra_artists=[legend_obj])
        output_pdf = output_path.with_suffix('.pdf')
        plt.savefig(output_pdf, bbox_inches='tight',
                    bbox_extra_artists=[legend_obj])
    else:
        plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
        output_pdf = output_path.with_suffix('.pdf')
        plt.savefig(output_pdf, bbox_inches='tight')
    
    plt.close(fig)
    
    print(f"  ✓ Saved: {output_path.name}")
    print(f"  ✓ Saved: {output_pdf.name}")


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def main():
    """
    Main function that orchestrates the conflict visualization.
    """
    
    print("\n" + "=" * 70)
    print("H9B: CONFLICT EVENT VISUALIZATION - DRC Region")
    print("=" * 70)
    print(f"\nCreating maps of armed conflict events")
    print(f"Area of Interest: DRC and surrounding region")
    
    # -------------------------------------------------------------------------
    # STEP 1: VERIFY INPUT FILES
    # -------------------------------------------------------------------------
    print("\n[STEP 1/5] Verifying input files...")
    print("-" * 70)
    
    if not INPUT_ACLED_XLSX.exists():
        print(f"ERROR: ACLED Excel file not found at {INPUT_ACLED_XLSX}")
        sys.exit(1)
    
    print(f"✓ Found ACLED Excel: {INPUT_ACLED_XLSX.name}")
    
    # Check for background map
    background_img = None
    if BACKGROUND_MAP.exists():
        print(f"✓ Found background map: {BACKGROUND_MAP.name}")
        background_img = plt.imread(BACKGROUND_MAP)
        print(f"  → Loaded background image")
    else:
        print(f"⚠ Background map not found at {BACKGROUND_MAP}")
        print(f"  → Maps will be created without background image")
    
    # -------------------------------------------------------------------------
    # STEP 2: LOAD CONFLICT DATA
    # -------------------------------------------------------------------------
    print("\n[STEP 2/5] Loading conflict data...")
    print("-" * 70)
    
    print(f"\nReading ACLED data from Excel...")
    acled = pd.read_excel(INPUT_ACLED_XLSX, engine="openpyxl")
    print(f"  → Loaded {len(acled):,} conflict events")
    
    # Check for required columns
    required_cols = ["LATITUDE", "LONGITUDE", "COUNTRY"]
    missing_cols = [col for col in required_cols if col not in acled.columns]
    if missing_cols:
        print(f"ERROR: ACLED file missing columns: {missing_cols}")
        sys.exit(1)
    
    # -------------------------------------------------------------------------
    # STEP 3: PROCESS AND FILTER DATA
    # -------------------------------------------------------------------------
    print("\n[STEP 3/5] Processing and filtering data...")
    print("-" * 70)
    
    # Convert coordinates to numeric
    acled["LATITUDE"] = pd.to_numeric(acled["LATITUDE"], errors="coerce")
    acled["LONGITUDE"] = pd.to_numeric(acled["LONGITUDE"], errors="coerce")
    
    # Convert COUNTRY to string type for reliable comparison
    acled["COUNTRY"] = acled["COUNTRY"].astype("string")
    
    # Handle date column (try multiple possible names)
    date_cols = ["EVENT_DATE", "event_date", "date", "DATE"]
    date_col = None
    for col in date_cols:
        if col in acled.columns:
            date_col = col
            break
    
    if date_col is None:
        print(f"ERROR: Could not find date column. Tried: {date_cols}")
        sys.exit(1)
    
    acled["EVENT_DATE"] = pd.to_datetime(acled[date_col], errors="coerce")
    print(f"  → Found date column: {date_col}")
    
    # Handle casualties (try multiple possible names)
    fatality_cols = ["FATALITIES", "fatalities", "deaths", "DEATHS"]
    fatality_col = None
    for col in fatality_cols:
        if col in acled.columns:
            fatality_col = col
            break
    
    if fatality_col is not None:
        acled["FATALITIES"] = pd.to_numeric(acled[fatality_col], errors="coerce").fillna(0)
        print(f"  → Found fatalities column: {fatality_col}")
    else:
        print(f"  ⚠ No fatalities column found, using size=1 for all events")
        acled["FATALITIES"] = 1
    
    # Filter to valid coordinates
    acled_valid = acled[
        acled["LATITUDE"].notna() & 
        acled["LONGITUDE"].notna() &
        acled["EVENT_DATE"].notna()
    ].copy()
    print(f"  → {len(acled_valid):,} events with valid coordinates and dates")
    
    # Filter by date range if specified
    n_before_date = len(acled_valid)
    if DATE_MIN is not None:
        date_min = pd.to_datetime(DATE_MIN)
        acled_valid = acled_valid[acled_valid["EVENT_DATE"] >= date_min]
        print(f"  → Filtered to events on or after {DATE_MIN}")
    
    if DATE_MAX is not None:
        date_max = pd.to_datetime(DATE_MAX)
        acled_valid = acled_valid[acled_valid["EVENT_DATE"] < date_max]
        print(f"  → Filtered to events before {DATE_MAX}")
    
    n_after_date = len(acled_valid)
    if n_before_date != n_after_date:
        print(f"  → After date filtering: {n_after_date:,} events ({n_after_date/n_before_date*100:.1f}%)")
    
    # Filter to Area of Interest (AOI)
    aoi_events = acled_valid[
        (acled_valid["LONGITUDE"] >= AOI_MIN_LON) &
        (acled_valid["LONGITUDE"] <= AOI_MAX_LON) &
        (acled_valid["LATITUDE"] >= AOI_MIN_LAT) &
        (acled_valid["LATITUDE"] <= AOI_MAX_LAT)
    ].copy()
    
    print(f"  → {len(aoi_events):,} events within AOI ({len(aoi_events)/len(acled_valid)*100:.1f}%)")
    
    # Separate DRC and non-DRC
    is_drc = aoi_events["COUNTRY"].astype("string") == DRC_COUNTRY_NAME
    n_drc = is_drc.sum()
    n_non_drc = (~is_drc).sum()
    
    print(f"  → DRC events: {n_drc:,}")
    print(f"  → Non-DRC events: {n_non_drc:,}")
    
    # Date range
    if not aoi_events.empty:
        min_date = aoi_events["EVENT_DATE"].min()
        max_date = aoi_events["EVENT_DATE"].max()
        print(f"  → Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
    # Casualty statistics
    total_fatalities = aoi_events["FATALITIES"].sum()
    mean_fatalities = aoi_events["FATALITIES"].mean()
    max_fatalities = aoi_events["FATALITIES"].max()
    print(f"  → Total fatalities: {int(total_fatalities):,}")
    print(f"  → Mean fatalities per event: {mean_fatalities:.2f}")
    print(f"  → Max fatalities in single event: {int(max_fatalities):,}")
    
    # -------------------------------------------------------------------------
    # STEP 4: SAVE FILTERED DATA
    # -------------------------------------------------------------------------
    print("\n[STEP 4/5] Saving filtered data...")
    print("-" * 70)
    
    # Save filtered events to CSV
    output_csv = OUTPUT_DIR / "h9b_filtered_events.csv"
    aoi_events.to_csv(output_csv, index=False)
    print(f"  ✓ Saved: {output_csv.name} ({len(aoi_events):,} events)")
    
    # -------------------------------------------------------------------------
    # STEP 5: CREATE VISUALIZATIONS
    # -------------------------------------------------------------------------
    print("\n[STEP 5/5] Creating visualizations...")
    print("-" * 70)
    
    if aoi_events.empty:
        print("  ⚠ No events to visualize!")
        return
    
    # Create visualizations
    print("\nGenerating maps...")
    
    # All events
    plot_events_on_map(
        aoi_events,
        OUTPUT_DIR / "h9b_all_events.png",
        f"Armed Conflict Events in DRC Region\n"
        f"Total Events: {len(aoi_events):,} | "
        f"DRC: {n_drc:,} | Non-DRC: {n_non_drc:,}\n"
        f"Date Range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}",
        background_img,
        show_drc=True,
        show_non_drc=True,
    )
    
    # DRC only
    if n_drc > 0:
        plot_events_on_map(
            aoi_events[is_drc],
            OUTPUT_DIR / "h9b_drc_only.png",
            f"Armed Conflict Events in DRC\n"
            f"Total Events: {n_drc:,} | Total Fatalities: {int(aoi_events[is_drc]['FATALITIES'].sum()):,}",
            background_img,
            show_drc=True,
            show_non_drc=False,
        )
    
    # Non-DRC only
    if n_non_drc > 0:
        plot_events_on_map(
            aoi_events[~is_drc],
            OUTPUT_DIR / "h9b_non_drc_only.png",
            f"Armed Conflict Events in Neighboring Countries\n"
            f"Total Events: {n_non_drc:,} | Total Fatalities: {int(aoi_events[~is_drc]['FATALITIES'].sum()):,}",
            background_img,
            show_drc=False,
            show_non_drc=True,
        )
    
    # -------------------------------------------------------------------------
    # SUMMARY
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE!")
    print("=" * 70)
    
    print(f"\n📊 OUTPUT FILES (saved in {OUTPUT_DIR.name}/):\n")
    
    print(f"  1. h9b_filtered_events.csv")
    print(f"     → Filtered event data ({len(aoi_events):,} events)\n")
    
    print(f"  2. h9b_all_events.png")
    print(f"     → Map showing all events (DRC + neighboring countries)\n")
    
    print(f"  3. h9b_all_events.pdf")
    print(f"     → Map showing all events (vector format)\n")
    
    if n_drc > 0:
        print(f"  4. h9b_drc_only.png")
        print(f"     → Map showing DRC events only\n")
        
        print(f"  5. h9b_drc_only.pdf")
        print(f"     → Map showing DRC events only (vector format)\n")
    
    if n_non_drc > 0:
        print(f"  6. h9b_non_drc_only.png")
        print(f"     → Map showing non-DRC events only\n")
        
        print(f"  7. h9b_non_drc_only.pdf")
        print(f"     → Map showing non-DRC events only (vector format)\n")
    
    print(f"⚙️  VISUALIZATION SETTINGS:")
    print(f"  → Area of Interest: {AOI_MIN_LON:.2f} to {AOI_MAX_LON:.2f} lon, {AOI_MIN_LAT:.2f} to {AOI_MAX_LAT:.2f} lat")
    if DATE_MIN is not None or DATE_MAX is not None:
        print(f"  → Date range: {DATE_MIN or 'earliest'} to {DATE_MAX or 'latest'}")
    print(f"  → Point size: Based on fatalities ({MIN_POINT_SIZE}-{MAX_POINT_SIZE})")
    print(f"  → Color: Blue (DRC) / Orange (Non-DRC), grayish = older, colored = newer")
    print(f"  → Resolution: {DPI} DPI")
    
    print(f"\n📈 LEGEND:")
    print(f"  → Point color intensity: Time of event (grayish = older, blue/orange = more recent)")
    print(f"  → Point size: Number of fatalities (larger = more casualties)")
    print(f"  → Blue points: Events in DRC")
    print(f"  → Orange points: Events in neighboring countries")
    
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