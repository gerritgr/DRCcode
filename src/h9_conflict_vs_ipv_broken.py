#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
H9: Conflict exposure vs IPV (D111) — DRC 2023–24 DHS
================================================================================

GOAL:
Explore hypothesis H9:
"Recent exposure to armed conflict is positively associated with the probability
of experiencing IPV."

WHAT THIS SCRIPT DOES:
1) Reads ACLED-style conflict events from the provided Excel file
2) Reads DHS women's responses with GPS cluster coordinates
3) Keeps only women who answered D111 as 0/1 (No/Yes)
4) For each woman, computes distance to the closest conflict event (simple
   Euclidean approximation in km using lat/lon)
5) Scatter plot: x = distance to closest conflict, y = D111 (0/1)
6) Fits a smooth curve (binned weighted mean + moving average smoothing)
7) Bootstraps to compute a 95% confidence interval around the curve
8) Saves:
   - DRCcode/output/h9_d111_curve.csv (raw points + curve values)
   - DRCcode/output/h9_d111_curve.jpg
   - DRCcode/output/h9_d111_curve.pdf

NOTES / ASSUMPTIONS:
- DHS cluster coordinates are expected in columns like gps_longitude/gps_latitude
  (common in merged DHS+GPS outputs). The code includes reasonable fallbacks.
- ACLED Excel is expected to contain LONGITUDE and LATITUDE columns (as in your
  context script). If named differently, adjust the column list in CONFIG.

DETERMINISM:
- Uses a fixed RNG seed for bootstrap resampling and (optional) subsampling.

================================================================================
"""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION (flags + hyperparameters)
# =============================================================================

# --- Color scheme for plot ---
# Colors for scatter points and interpolated line
# Using seaborn color palette defaults: red for "Yes", blue for "No", green for line
COLOR_YES = "#d62728"      # Seaborn red (for D111=1, "Yes" responses)
COLOR_NO = "#1f77b4"       # Seaborn blue (for D111=0, "No" responses)
COLOR_LINE = "#2ca02c"     # Seaborn green (for interpolated curve line)

# --- Paths (relative to this script location) ---
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

DHS_CSV = PROJECT_ROOT / "DATA" / "DHS" / "women_all_answers_gps.csv"
ACLED_XLSX = (
    PROJECT_ROOT
    / "DATA"
    / "Conflict"
    / "Africa_lagged_data_up_to-2024-10-17"
    / "Africa_lagged_data_up_to-2024-10-17.xlsx"
)

OUTDIR = PROJECT_ROOT / "output"
OUTDIR.mkdir(parents=True, exist_ok=True)

OUT_CSV = OUTDIR / "h9_d111_curve.csv"
OUT_JPG = OUTDIR / "h9_d111_curve.jpg"
OUT_PDF = OUTDIR / "h9_d111_curve.pdf"

# --- Analysis variable ---
IPV_VAR = "d111"  # DHS D111 (already lowercased in your pipeline)

# --- Weighting ---
USE_WEIGHTS = True          # If True, weight by WEIGHT_VARIABLE for curve + bootstrap
WEIGHT_VARIABLE = "d005"     # Recommended for DV outcomes (d005) OR "v005" if desired

# --- Distance computation ---
# Simple Euclidean approximation: convert degrees to km.
# km_per_degree_lat ~ 111.32
# km_per_degree_lon ~ 111.32 * cos(lat)
KM_PER_DEG_LAT = 111.32

# --- Plot / curve settings ---
FIGSIZE = (10, 6)
DPI = 300
SCATTER_ALPHA = 0.20
SCATTER_SIZE = 8

# Binning for smooth curve (distance axis)
N_BINS = 40
MAX_DISTANCE_KM = None  # set e.g. 300 to cap x-range; None keeps all

# Smoothing: moving average window over binned means
SMOOTH_WINDOW = 5       # must be >= 1; odd numbers usually nicer

# --- Bootstrap CI settings ---
RNG_SEED = 12345
N_BOOT = 500            # increase for smoother CI (e.g., 1000+), but slower
CI_LOW = 2.5
CI_HIGH = 97.5

# Optional: subsample points for scatter visibility / speed (curve uses full data)
MAX_SCATTER_POINTS = None  # e.g. 20000, or None for all

# --- Expected column names (with fallbacks) ---
# DHS cluster coordinates:
DHS_LON_CANDIDATES = [
    "gps_longitude", "longitude", "lon", "cluster_lon", "dhs_lon",
    # --- ADDED: common DHS GPS/merge names ---
    "cluster_longitude", "shlongitude", "sh_lon", "long", "lng", "x",
    "cl_long", "cl_lon", "gpx_longitude", "g_long", "coord_lon", "coord_longitude",
]
DHS_LAT_CANDIDATES = [
    "gps_latitude", "latitude", "lat", "cluster_lat", "dhs_lat",
    # --- ADDED: common DHS GPS/merge names ---
    "cluster_latitude", "shlatitude", "sh_lat", "y",
    "cl_lat", "gpx_latitude", "g_lat", "coord_lat", "coord_latitude",
]

# ACLED event coordinates:
ACLED_LON_CANDIDATES = ["LONGITUDE", "longitude", "lon", "event_longitude"]
ACLED_LAT_CANDIDATES = ["LATITUDE", "latitude", "lat", "event_latitude"]

# =============================================================================
# HELPERS
# =============================================================================

def pick_first_existing_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    """Return the first column name in candidates that exists in df; else raise."""
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find {label} column. Tried: {candidates}. Available: {list(df.columns)[:50]}...")


def to_numeric_series(s: pd.Series) -> pd.Series:
    """Convert a pandas series to numeric with coercion."""
    return pd.to_numeric(s, errors="coerce")


def distance_km_euclidean(lon1, lat1, lon2, lat2) -> np.ndarray:
    """
    Approximate distance in km between (lon1,lat1) and (lon2,lat2) using a simple
    Euclidean approximation in geographic degrees scaled to km.

    Works well for relative distances at this spatial scale.
    """
    lat1 = np.asarray(lat1, dtype=float)
    lon1 = np.asarray(lon1, dtype=float)
    lat2 = np.asarray(lat2, dtype=float)
    lon2 = np.asarray(lon2, dtype=float)

    # Scale longitude degrees by cos(latitude) (use lat1 as local reference)
    km_per_deg_lon = KM_PER_DEG_LAT * np.cos(np.deg2rad(lat1))
    dx = (lon2 - lon1) * km_per_deg_lon
    dy = (lat2 - lat1) * KM_PER_DEG_LAT
    return np.sqrt(dx * dx + dy * dy)


def compute_min_distance_to_events(
    women_lon: np.ndarray,
    women_lat: np.ndarray,
    events_lon: np.ndarray,
    events_lat: np.ndarray,
    chunk_size: int = 2000,
) -> np.ndarray:
    """
    For each woman point, compute distance to the closest event point.
    Uses chunking for memory safety.

    Returns:
        min_dist_km: array of shape (n_women,)
    """
    n_w = len(women_lon)
    if len(events_lon) == 0:
        return np.full(n_w, np.nan, dtype=float)

    wlon = women_lon.astype(float)
    wlat = women_lat.astype(float)
    elon = events_lon.astype(float)
    elat = events_lat.astype(float)

    out = np.empty(n_w, dtype=float)

    # For each chunk of women, compute distances to all events and take min
    for i0 in range(0, n_w, chunk_size):
        i1 = min(n_w, i0 + chunk_size)

        # Broadcast to (chunk, n_events)
        lat_chunk = wlat[i0:i1][:, None]
        lon_chunk = wlon[i0:i1][:, None]

        # Use local lat for lon scaling (lat_chunk)
        km_per_deg_lon = KM_PER_DEG_LAT * np.cos(np.deg2rad(lat_chunk))
        dx = (elon[None, :] - lon_chunk) * km_per_deg_lon
        dy = (elat[None, :] - lat_chunk) * KM_PER_DEG_LAT
        dist = np.sqrt(dx * dx + dy * dy)

        out[i0:i1] = np.nanmin(dist, axis=1)

    return out


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted mean with safety checks."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    m = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan
    return float(np.sum(v[m] * w[m]) / np.sum(w[m]))


def binned_curve(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None,
    n_bins: int,
    x_max: float | None,
    smooth_window: int,
):
    """
    Build a smooth curve by:
      1) binning x into n_bins
      2) computing (weighted) mean y per bin
      3) applying moving-average smoothing over bin means

    Returns:
      bin_centers, bin_means, smooth_means
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    w = None
    if weights is not None:
        w = np.asarray(weights, dtype=float)[m]

    if x_max is None:
        x_max = float(np.nanmax(x)) if len(x) else 1.0

    x_min = float(np.nanmin(x)) if len(x) else 0.0
    if x_max <= x_min:
        x_max = x_min + 1e-6

    edges = np.linspace(x_min, x_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    means = np.full(n_bins, np.nan, dtype=float)
    counts = np.zeros(n_bins, dtype=int)

    # Assign bins
    bin_idx = np.digitize(x, edges, right=False) - 1
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    for b in range(n_bins):
        mb = bin_idx == b
        counts[b] = int(np.sum(mb))
        if counts[b] == 0:
            continue
        if w is None:
            means[b] = float(np.mean(y[mb]))
        else:
            means[b] = weighted_mean(y[mb], w[mb])

    # Moving average smoothing (ignore NaNs by local averaging)
    sw = max(int(smooth_window), 1)
    smooth = np.full_like(means, np.nan)
    half = sw // 2

    for i in range(n_bins):
        j0 = max(0, i - half)
        j1 = min(n_bins, i + half + 1)
        window_vals = means[j0:j1]
        if np.all(~np.isfinite(window_vals)):
            continue
        smooth[i] = float(np.nanmean(window_vals))

    return centers, means, smooth, counts, edges


def bootstrap_curve_ci(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray | None,
    n_bins: int,
    x_max: float | None,
    smooth_window: int,
    n_boot: int,
    seed: int,
    ci_low: float,
    ci_high: float,
):
    """
    Nonparametric bootstrap over women (rows).
    For each bootstrap replicate, compute the smoothed binned curve.
    Then take percentile bands across replicates for each bin center.
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    if n == 0:
        return None

    # Fix binning edges using original data (important for deterministic comparison)
    centers, means, smooth, counts, edges = binned_curve(
        x=x, y=y, weights=weights, n_bins=n_bins, x_max=x_max, smooth_window=smooth_window
    )

    # Store curves (n_boot, n_bins)
    curves = np.full((n_boot, n_bins), np.nan, dtype=float)

    idx = np.arange(n)

    for b in range(n_boot):
        # Sample indices with replacement
        bs_idx = rng.choice(idx, size=n, replace=True)
        xb = x[bs_idx]
        yb = y[bs_idx]
        wb = weights[bs_idx] if weights is not None else None

        # Compute binned means using the SAME edges
        # (re-implement quickly here to reuse edges exactly)
        bin_idx = np.digitize(xb, edges, right=False) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        means_b = np.full(n_bins, np.nan, dtype=float)
        for k in range(n_bins):
            mk = bin_idx == k
            if not np.any(mk):
                continue
            if wb is None:
                means_b[k] = float(np.mean(yb[mk]))
            else:
                means_b[k] = weighted_mean(yb[mk], wb[mk])

        # Smooth
        sw = max(int(smooth_window), 1)
        half = sw // 2
        smooth_b = np.full(n_bins, np.nan, dtype=float)
        for i in range(n_bins):
            j0 = max(0, i - half)
            j1 = min(n_bins, i + half + 1)
            window_vals = means_b[j0:j1]
            if np.all(~np.isfinite(window_vals)):
                continue
            smooth_b[i] = float(np.nanmean(window_vals))

        curves[b, :] = smooth_b

    # Percentile CI per bin (ignore NaNs)
    lo = np.nanpercentile(curves, ci_low, axis=0)
    hi = np.nanpercentile(curves, ci_high, axis=0)

    return centers, smooth, lo, hi, counts


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("\n" + "=" * 70)
    print("H9: CONFLICT EXPOSURE vs IPV (D111)")
    print("=" * 70)
    print(f"DHS CSV:   {DHS_CSV}")
    print(f"ACLED XLSX:{ACLED_XLSX}")
    print(f"Outcome:   {IPV_VAR.upper()} (0/1)")
    print(f"Weighting: {'ON' if USE_WEIGHTS else 'OFF'}"
          + (f" ({WEIGHT_VARIABLE.upper()})" if USE_WEIGHTS else ""))
    print(f"Bootstrap: n={N_BOOT}, seed={RNG_SEED}")
    print("-" * 70)

    # -------------------------------------------------------------------------
    # 1) Read ACLED conflict events
    # -------------------------------------------------------------------------
    if not ACLED_XLSX.exists():
        print(f"ERROR: Missing conflict file: {ACLED_XLSX}")
        sys.exit(1)

    conflict = pd.read_excel(ACLED_XLSX, engine="openpyxl")

    acled_lon_col = pick_first_existing_column(conflict, ACLED_LON_CANDIDATES, "ACLED longitude")
    acled_lat_col = pick_first_existing_column(conflict, ACLED_LAT_CANDIDATES, "ACLED latitude")

    conflict[acled_lon_col] = to_numeric_series(conflict[acled_lon_col])
    conflict[acled_lat_col] = to_numeric_series(conflict[acled_lat_col])

    conflict = conflict[conflict[acled_lon_col].notna() & conflict[acled_lat_col].notna()].copy()

    print(f"[ok] Loaded conflict events: {len(conflict):,}")

    # -------------------------------------------------------------------------
    # 2) Read DHS women's responses
    # -------------------------------------------------------------------------
    if not DHS_CSV.exists():
        print(f"ERROR: Missing DHS file: {DHS_CSV}")
        sys.exit(1)

    women = pd.read_csv(DHS_CSV)

    # Standardize to lowercase columns (your pipeline seems to do this already,
    # but this makes it robust)
    women.columns = [c.lower() for c in women.columns]
    conflict.columns = [c if c.isupper() else c for c in conflict.columns]  # keep ACLED original casing

    if IPV_VAR.lower() not in women.columns:
        print(f"ERROR: '{IPV_VAR.lower()}' not found in DHS columns.")
        print(f"Available columns: {list(women.columns)[:50]}...")
        sys.exit(1)

    # --- CHANGED: robust DHS GPS detection (exact match first, then substring fallback) ---
    try:
        dhs_lon_col = pick_first_existing_column(
            women, [c.lower() for c in DHS_LON_CANDIDATES], "DHS cluster longitude"
        )
        dhs_lat_col = pick_first_existing_column(
            women, [c.lower() for c in DHS_LAT_CANDIDATES], "DHS cluster latitude"
        )
    except ValueError:
        cols = list(women.columns)

        lon_like = [c for c in cols if ("lon" in c) or ("long" in c)]
        lat_like = [c for c in cols if ("lat" in c)]

        lon_pref = [c for c in lon_like if ("cluster" in c) or ("gps" in c) or ("sh" in c)] or lon_like
        lat_pref = [c for c in lat_like if ("cluster" in c) or ("gps" in c) or ("sh" in c)] or lat_like

        if not lon_pref or not lat_pref:
            raise ValueError(
                "Could not infer DHS GPS columns. "
                f"lon-like candidates: {lon_like[:20]} ; lat-like candidates: {lat_like[:20]}"
            )

        dhs_lon_col = lon_pref[0]
        dhs_lat_col = lat_pref[0]

    print(f"[ok] Using DHS GPS columns: lon='{dhs_lon_col}', lat='{dhs_lat_col}'")

    women[dhs_lon_col] = to_numeric_series(women[dhs_lon_col])
    women[dhs_lat_col] = to_numeric_series(women[dhs_lat_col])
    women[IPV_VAR.lower()] = to_numeric_series(women[IPV_VAR.lower()])

    # -------------------------------------------------------------------------
    # 3) Keep only women who answered yes/no for D111 (0/1)
    # -------------------------------------------------------------------------
    women_valid = women[
        women[IPV_VAR.lower()].isin([0, 1])
        & women[dhs_lon_col].notna()
        & women[dhs_lat_col].notna()
    ].copy()

    print(f"[ok] Women with valid {IPV_VAR.upper()} and GPS: {len(women_valid):,}")

    # -------------------------------------------------------------------------
    # 4) Compute distance to closest conflict event (using cluster coordinates)
    # -------------------------------------------------------------------------
    events_lon = conflict[acled_lon_col].to_numpy(float)
    events_lat = conflict[acled_lat_col].to_numpy(float)

    wlon = women_valid[dhs_lon_col].to_numpy(float)
    wlat = women_valid[dhs_lat_col].to_numpy(float)

    min_dist_km = compute_min_distance_to_events(wlon, wlat, events_lon, events_lat, chunk_size=2000)
    women_valid["min_conflict_dist_km"] = min_dist_km

    # Optional: cap distances for analysis / plot (keeps x-axis readable)
    if MAX_DISTANCE_KM is not None:
        women_valid = women_valid[women_valid["min_conflict_dist_km"] <= float(MAX_DISTANCE_KM)].copy()

    # Prepare weights
    weights = None
    if USE_WEIGHTS:
        wcol = WEIGHT_VARIABLE.lower()
        if wcol in women_valid.columns:
            weights = to_numeric_series(women_valid[wcol]).to_numpy(float) / 1_000_000.0
        else:
            print(f"[warn] Weight variable '{wcol}' not found; falling back to equal weights.")
            weights = np.ones(len(women_valid), dtype=float)

    # -------------------------------------------------------------------------
    # 5–7) Curve + bootstrap CI
    # -------------------------------------------------------------------------
    x = women_valid["min_conflict_dist_km"].to_numpy(float)
    y = women_valid[IPV_VAR.lower()].to_numpy(float)

    ci_res = bootstrap_curve_ci(
        x=x,
        y=y,
        weights=weights,
        n_bins=N_BINS,
        x_max=MAX_DISTANCE_KM,
        smooth_window=SMOOTH_WINDOW,
        n_boot=N_BOOT,
        seed=RNG_SEED,
        ci_low=CI_LOW,
        ci_high=CI_HIGH,
    )

    if ci_res is None:
        print("ERROR: No data available for curve fitting.")
        sys.exit(1)

    centers, smooth, lo, hi, counts = ci_res

    # -------------------------------------------------------------------------
    # 8) Save CSV with raw points + curve/CI values
    # -------------------------------------------------------------------------
    # Raw scatter points
    out_points = pd.DataFrame({
        "min_conflict_dist_km": x,
        "d111": y.astype(int),
    })

    if weights is not None:
        out_points["weight"] = weights

    # Curve data (binned)
    out_curve = pd.DataFrame({
        "bin_center_km": centers,
        "curve_mean": smooth,
        "ci_low": lo,
        "ci_high": hi,
        "n_in_bin": counts,
    })

    # Store both in one CSV by writing curve rows after points with a 'row_type' tag
    out_points2 = out_points.copy()
    out_points2["row_type"] = "point"
    out_curve2 = out_curve.copy()
    out_curve2["row_type"] = "curve"

    # Align columns (union)
    all_cols = sorted(set(out_points2.columns).union(set(out_curve2.columns)))
    out_points2 = out_points2.reindex(columns=all_cols)
    out_curve2 = out_curve2.reindex(columns=all_cols)

    out_all = pd.concat([out_points2, out_curve2], ignore_index=True)
    out_all.to_csv(OUT_CSV, index=False)
    print(f"[ok] Saved CSV: {OUT_CSV.name}")

    # -------------------------------------------------------------------------
    # 5) Plot: scatter + smooth curve + CI band
    # -------------------------------------------------------------------------
    # Optional deterministic subsample for scatter visibility
    scatter_df = women_valid
    if MAX_SCATTER_POINTS is not None and len(women_valid) > int(MAX_SCATTER_POINTS):
        rng = np.random.default_rng(RNG_SEED)
        keep_idx = rng.choice(women_valid.index.to_numpy(), size=int(MAX_SCATTER_POINTS), replace=False)
        scatter_df = women_valid.loc[keep_idx].copy()

    fig, ax = plt.subplots(figsize=FIGSIZE)

    # Scatter: y is 0/1 so jitter helps a bit
    rng = np.random.default_rng(RNG_SEED)
    jitter = rng.normal(loc=0.0, scale=0.02, size=len(scatter_df))
    y_scatter = scatter_df[IPV_VAR.lower()].to_numpy(float) + jitter
    x_scatter = scatter_df["min_conflict_dist_km"].to_numpy(float)
    
    # Separate scatter points by response (0=No, 1=Yes) to use different colors
    mask_no = scatter_df[IPV_VAR.lower()] == 0
    mask_yes = scatter_df[IPV_VAR.lower()] == 1
    
    # Plot "No" responses in blue
    ax.scatter(
        x_scatter[mask_no],
        y_scatter[mask_no],
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        linewidths=0,
        color=COLOR_NO,
        label='No IPV (D111=0)'
    )
    
    # Plot "Yes" responses in red
    ax.scatter(
        x_scatter[mask_yes],
        y_scatter[mask_yes],
        s=SCATTER_SIZE,
        alpha=SCATTER_ALPHA,
        linewidths=0,
        color=COLOR_YES,
        label='Yes IPV (D111=1)'
    )

    # CI band + curve (in green)
    ax.fill_between(centers, lo, hi, alpha=0.25, linewidth=0, color=COLOR_LINE)
    ax.plot(centers, smooth, linewidth=2.5, color=COLOR_LINE, label='Smoothed curve')

    ax.set_xlabel("Distance to closest conflict event (km)")
    ax.set_ylabel(f"{IPV_VAR.upper()} (0=No, 1=Yes)")
    ax.set_ylim(-0.15, 1.15)

    title = "H9: IPV (D111) vs distance to closest conflict event"
    subtitle = "Unweighted" if not USE_WEIGHTS else f"Weighted by {WEIGHT_VARIABLE.upper()}"
    ax.set_title(f"{title}\n{subtitle} • Bootstrap 95% CI (n={N_BOOT})", fontsize=12)

    ax.grid(True, alpha=0.2)
    ax.legend(loc='best', framealpha=0.9)

    plt.tight_layout()

    # Save plot
    plt.savefig(OUT_JPG, dpi=DPI, bbox_inches="tight")
    plt.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"[ok] Saved plot: {OUT_JPG.name}")
    print(f"[ok] Saved plot: {OUT_PDF.name}")
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)