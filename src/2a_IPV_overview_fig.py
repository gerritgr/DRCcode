#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DHS Data Visualization Script - DRC 2023-24 Survey (GENERALIZED)
================================================================================

Creates one MAP + one CSV per variable in VARIABLES.

- Circle COLOR = prevalence rate (% "Yes" among *valid respondents* for that item)
- Circle SIZE  = item-specific sample size (number of valid respondents in cluster)
- Weighted by v005 (optional)

INPUT:
- DATA/DHS/women_all_answers_gps.csv
- DATA/background.png (optional)

OUTPUT (for each variable X):
- output/2a_X_map.jpg
- output/2a_X_map.csv
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

# ✅ Put your variables here
VARIABLES = ["D111", "D104", "D106", "D108"]

# Meanings (from your snippet)
VAR_MEANING = {
    "D111": "Any IPV",
    "D104": "Emotional IPV",
    "D106": "Physical IPV",
    "D108": "Sexual IPV",
}

# Determine paths relative to this script
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent

DATA_DIR = PROJECT_ROOT / "DATA"
DHS_DIR = DATA_DIR / "DHS"

# Input files
INPUT_CSV = DHS_DIR / "women_all_answers_gps.csv"
BACKGROUND_IMAGE = DATA_DIR / "background.png"

# Output directory
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Geographic extent for DRC [min_lon, max_lon, min_lat, max_lat]
EXTENT = [
    11.893979235367297, 31.616531541802978,
    -13.981788316118982, 5.811825978871415
]

# Visualization settings
USE_WEIGHTS = True
DPI = 300
BG_ALPHA = 0.9
CMAP = "magma"
CIRCLE_ALPHA = 0.25

SIZE_MIN = 25.0
SIZE_MAX = 300.0

# DHS-style assumptions for binary items:
#   1 = Yes, 0 = No, everything else (8/9/etc) treated as invalid/missing.
YES_VALUE = 1
VALID_VALUES = {0, 1}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def weighted_prop_binary(values, weights, yes_value=1, valid_values={0, 1}):
    """
    Weighted proportion for a binary indicator, restricted to valid_values.
    Returns NaN if no valid data.
    """
    v = pd.to_numeric(pd.Series(values), errors="coerce")
    w = pd.to_numeric(pd.Series(weights), errors="coerce")

    valid_mask = v.isin(valid_values)
    mask = valid_mask & w.notna() & (w > 0)

    if not mask.any():
        return np.nan

    y = (v[mask] == yes_value).astype(float)
    return float((y * w[mask]).sum() / w[mask].sum())


def unweighted_prop_binary(values, yes_value=1, valid_values={0, 1}):
    v = pd.to_numeric(pd.Series(values), errors="coerce")
    v = v[v.isin(valid_values)]
    if len(v) == 0:
        return np.nan
    return float((v == yes_value).mean())


def compute_marker_sizes(n_series, size_min=25.0, size_max=300.0):
    """
    Percentile-clipped linear scaling of marker AREAS.
    Returns (sizes_scaled, p05, p95).
    """
    sizes = pd.to_numeric(n_series, errors="coerce").fillna(0).astype(float)

    p05 = float(np.nanpercentile(sizes.to_numpy(), 5))
    p95 = float(np.nanpercentile(sizes.to_numpy(), 95))
    if not np.isfinite(p05):
        p05 = float(sizes.min())
    if not np.isfinite(p95):
        p95 = float(sizes.max())
    if p95 <= p05:
        p05, p95 = float(sizes.min()), float(sizes.max())
        if p95 <= p05:
            p05, p95 = 0.0, max(1.0, float(sizes.max()))

    sizes_clipped = sizes.clip(p05, p95)
    sizes_normalized = (sizes_clipped - p05) / (p95 - p05) if p95 > p05 else 0.0
    sizes_scaled = size_min + (size_max - size_min) * sizes_normalized
    return sizes_scaled, p05, p95


def main():
    print("\n" + "=" * 70)
    print("DHS DATA VISUALIZATION (GENERALIZED MAPS) - DRC 2023-24 Survey")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # STEP 1: VERIFY INPUT FILES
    # -------------------------------------------------------------------------
    print("\n[STEP 1/3] Verifying input files...")
    print("-" * 70)

    if not INPUT_CSV.exists():
        print(f"ERROR: Input CSV not found at {INPUT_CSV}")
        sys.exit(1)

    print(f"✓ Found input CSV: {INPUT_CSV.name}")
    print(f"  Location: {INPUT_CSV}")

    has_background = BACKGROUND_IMAGE.exists()
    if has_background:
        print(f"✓ Found background map: {BACKGROUND_IMAGE.name}")
    else:
        print(f"⚠ Background map not found at {BACKGROUND_IMAGE}")
        print("  Maps will be created without background image")

    # -------------------------------------------------------------------------
    # STEP 2: LOAD DATA + BASIC PREP
    # -------------------------------------------------------------------------
    print("\n[STEP 2/3] Loading data...")
    print("-" * 70)

    data = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"  → Loaded {len(data):,} women's records")

    # Normalize column names to handle case mismatches (your header shows lowercase d104 etc.)
    # We build a mapping from uppercase -> actual column name.
    col_lookup = {c.upper(): c for c in data.columns}

    # Cluster id: in DHS, v001 is typically the cluster number
    if "V001" not in col_lookup:
        raise KeyError("Expected a v001/V001 column for cluster id.")
    cluster_col = col_lookup["V001"]

    # GPS
    if "LATNUM" not in col_lookup or "LONGNUM" not in col_lookup:
        raise KeyError("Expected LATNUM and LONGNUM columns in the input CSV.")
    lat_col = col_lookup["LATNUM"]
    lon_col = col_lookup["LONGNUM"]

    # Weights
    weights = None
    if USE_WEIGHTS:
        if "V005" not in col_lookup:
            raise KeyError("USE_WEIGHTS=True but v005/V005 not found.")
        w_col = col_lookup["V005"]
        # DHS v005 is typically scaled by 1,000,000
        data["_w"] = pd.to_numeric(data[w_col], errors="coerce") / 1_000_000.0
        weights = "_w"

    # Background image (once)
    bg = None
    if has_background:
        bg = Image.open(BACKGROUND_IMAGE).convert("RGBA")

    # -------------------------------------------------------------------------
    # STEP 3: LOOP VARIABLES AND CREATE OUTPUTS
    # -------------------------------------------------------------------------
    print("\n[STEP 3/3] Creating outputs...")
    print("-" * 70)

    for var in VARIABLES:
        var_u = var.upper()
        if var_u not in col_lookup:
            print(f"\n⚠ Skipping {var}: column not found in CSV.")
            continue

        var_col = col_lookup[var_u]
        meaning = VAR_MEANING.get(var_u, "Unknown meaning")

        print(f"\n--- Variable {var_u}: {meaning} ---")

        # Restrict to rows that answered this question (valid values only)
        v_num = pd.to_numeric(data[var_col], errors="coerce")
        answered_mask = v_num.isin(VALID_VALUES)

        # If nothing answered, skip
        n_answered = int(answered_mask.sum())
        if n_answered == 0:
            print(f"  ⚠ No valid answers (values in {VALID_VALUES}) found for {var_u}. Skipping.")
            continue

        df = data.loc[answered_mask, [cluster_col, lat_col, lon_col, var_col] + ([weights] if USE_WEIGHTS else [])].copy()
        df.rename(columns={cluster_col: "cluster_id", lat_col: "LATNUM", lon_col: "LONGNUM"}, inplace=True)

        # Aggregate to cluster level (item-specific!)
        if USE_WEIGHTS:
            agg = (
                df.groupby("cluster_id")
                  .apply(lambda g: pd.Series({
                      "fraction": weighted_prop_binary(g[var_col], g[weights], yes_value=YES_VALUE, valid_values=VALID_VALUES),
                      "n_women": len(g),
                      "LATNUM": g["LATNUM"].iloc[0] if len(g) > 0 else np.nan,
                      "LONGNUM": g["LONGNUM"].iloc[0] if len(g) > 0 else np.nan,
                  }), include_groups=False)
                  .reset_index()
            )
            title_suffix = "Weighted"
            subtitle = "Weighted by DHS sampling probability (v005)"
        else:
            agg = (
                df.groupby("cluster_id")
                  .agg(
                      fraction=(var_col, lambda s: unweighted_prop_binary(s, yes_value=YES_VALUE, valid_values=VALID_VALUES)),
                      n_women=(var_col, "size"),
                      LATNUM=("LATNUM", "first"),
                      LONGNUM=("LONGNUM", "first"),
                  )
                  .reset_index()
            )
            title_suffix = "Unweighted"
            subtitle = "Simple percentages (not accounting for sampling design)"

        # Filter to map extent + valid fraction
        plot_df = agg[
            (agg["LONGNUM"] >= EXTENT[0]) & (agg["LONGNUM"] <= EXTENT[1]) &
            (agg["LATNUM"] >= EXTENT[2]) & (agg["LATNUM"] <= EXTENT[3]) &
            agg["fraction"].notna()
        ].copy()

        print(f"  → Valid respondents (overall): {n_answered:,}")
        print(f"  → Clusters with data in extent: {len(plot_df):,}")
        if len(plot_df) == 0:
            print("  ⚠ No clusters to plot after filtering. Skipping.")
            continue

        # Marker sizes based on item-specific n_women
        sizes_scaled, p05, p95 = compute_marker_sizes(plot_df["n_women"], SIZE_MIN, SIZE_MAX)

        # Color scale (per variable)
        vmin = 0.0
        vmax = float(np.nanmax(plot_df["fraction"].to_numpy()))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 11))
        if bg is not None:
            ax.imshow(bg, extent=EXTENT, origin="upper", alpha=BG_ALPHA, zorder=0)

        sc = ax.scatter(
            plot_df["LONGNUM"],
            plot_df["LATNUM"],
            c=plot_df["fraction"],
            s=sizes_scaled,
            cmap=CMAP,
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
            f"Prevalence ({title_suffix})\n{meaning} — {var_u}\n{subtitle}",
            fontsize=14,
            fontweight="bold",
            pad=20
        )

        cbar = plt.colorbar(sc, ax=ax, orientation="vertical", fraction=0.03, pad=0.02)
        cbar.set_label(
            f"Fraction Reporting 'Yes' ({meaning})\n"
            + ("(weighted by sampling probability)" if USE_WEIGHTS else "(unweighted)"),
            rotation=90,
            fontsize=11,
            labelpad=15
        )
        cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0%}"))

        # Size legend
        min_n = int(plot_df["n_women"].min())
        max_n = int(plot_df["n_women"].max())
        legend_samples = [min_n, int((min_n + max_n) / 2), max_n]
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

        ax.legend(
            handles=handles,
            scatterpoints=1,
            frameon=True,
            labelspacing=1.5,
            title="Valid Respondents\nper Cluster",
            loc="lower right",
            fontsize=10,
            title_fontsize=11
        )

        # Save outputs (must start with 2a)
        out_img = OUTPUT_DIR / f"2a_{var_u}_map.jpg"
        plt.savefig(out_img, dpi=DPI, bbox_inches="tight")
        plt.close(fig)

        out_csv = OUTPUT_DIR / f"2a_{var_u}_map.csv"
        pd.DataFrame({
            "LONGNUM": plot_df["LONGNUM"],
            "LATNUM": plot_df["LATNUM"],
            "value": plot_df["fraction"],
            "size": sizes_scaled,
            "n_women": plot_df["n_women"],
            "cluster_id": plot_df["cluster_id"],
            "variable": var_u,
            "meaning": meaning,
        }).to_csv(out_csv, index=False)

        print(f"  ✓ Saved: {out_img.name}")
        print(f"  ✓ Saved: {out_csv.name}")
        print(f"  → Mean prevalence across plotted clusters: {plot_df['fraction'].mean():.1%}")

    print("\n" + "=" * 70)
    print("DONE.")
    print("=" * 70)
    print(f"Outputs saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
