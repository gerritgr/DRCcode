#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
DHS (DRC 2023–24) — Cluster-Level Correlation Analysis
================================================================================

Reads:  DATA/DHS/women_all_answers_gps.csv
Writes: output/1b_cluster_correlation_scatter.jpg
        output/1b_cluster_correlation_data.csv
================================================================================
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# =============================================================================
# USER CONFIGURATION
# =============================================================================

VARIABLE_1 = "sexual_violence_any"     # X-axis
VARIABLE_2 = "sexual_violence_recent"  # Y-axis

VARIABLE_1_LABEL = "Sexual Violence (Any)"
VARIABLE_2_LABEL = "Sexual Violence (Recent)"

USE_WEIGHTS = True  # two-stage v005 weighting
WEIGHT_VARIABLE = "v005"

DPI = 300
FIGSIZE = (10, 10)
POINT_SIZE = 80
POINT_ALPHA = 0.6
POINT_COLOR = "#2E86AB"
LINE_COLOR = "#A23B72"
LINE_WIDTH = 2.5

# =============================================================================
# PATHS (robust when running from src/)
# =============================================================================

SRC_DIR = Path(__file__).resolve().parent              # .../src
ROOT = SRC_DIR.parent                                 # project root
DATA_DIR = ROOT / "DATA" / "DHS"

INPUT_CSV = DATA_DIR / "women_all_answers_gps.csv"

# Output should be written to output/ (same level as src/)
OUTDIR = ROOT / "output"
OUTDIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def coerce_binary_yes_no(series: pd.Series) -> pd.Series:
    """Convert DHS binary Yes/No variables to numeric 0/1 (float), else NaN."""
    if series is None:
        return pd.Series(dtype=float)

    s = series.copy()

    # Numeric encoding (common DHS patterns: 1=yes, 0/2=no)
    if pd.api.types.is_numeric_dtype(s):
        arr = np.where(s == 1, 1.0, np.where((s == 0) | (s == 2), 0.0, np.nan)).astype(float)
        return pd.Series(arr, index=s.index)

    # Text encoding
    sl = s.astype("string").str.strip().str.lower()
    is_yes = sl.isin({"1", "yes"}) | sl.str.contains(r"(?:yes|ever)", na=False, regex=True)
    is_no  = sl.isin({"0", "2", "no"}) | sl.str.contains(r"(?:no|non)", na=False, regex=True)
    arr = np.where(is_yes, 1.0, np.where(is_no, 0.0, np.nan)).astype(float)
    return pd.Series(arr, index=s.index)

def weighted_prop(values, weights) -> float:
    """Weighted proportion for 0/1 data. Returns NaN if no valid data."""
    v = pd.to_numeric(pd.Series(values), errors="coerce")
    w = pd.to_numeric(pd.Series(weights), errors="coerce")
    m = v.notna() & w.notna() & (w > 0)
    if not m.any():
        return np.nan
    return float((v[m] * w[m]).sum() / w[m].sum())

def weighted_correlation(x, y, weights) -> float:
    """Weighted Pearson correlation coefficient."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[mask], y[mask], w[mask]
    if x.size < 2:
        return np.nan

    w = w / w.sum()
    mx = np.sum(w * x)
    my = np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    sx = np.sqrt(np.sum(w * (x - mx) ** 2))
    sy = np.sqrt(np.sum(w * (y - my) ** 2))
    if sx == 0 or sy == 0:
        return np.nan
    return float(cov / (sx * sy))

def weighted_linregress(x, y, weights):
    """Weighted linear regression y = slope*x + intercept."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    x, y, w = x[mask], y[mask], w[mask]
    if x.size < 2:
        return np.nan, np.nan

    w_sum = w.sum()
    mx = np.sum(w * x) / w_sum
    my = np.sum(w * y) / w_sum

    num = np.sum(w * (x - mx) * (y - my))
    den = np.sum(w * (x - mx) ** 2)
    if den == 0:
        return np.nan, np.nan

    slope = num / den
    intercept = my - slope * mx
    return float(slope), float(intercept)

# =============================================================================
# LOAD AND PROCESS
# =============================================================================

print("=" * 70)
print("CLUSTER-LEVEL CORRELATION ANALYSIS (TWO-STAGE WEIGHTING)")
print("=" * 70)
print(f"\nVariable 1 (X-axis): {VARIABLE_1} - {VARIABLE_1_LABEL}")
print(f"Variable 2 (Y-axis): {VARIABLE_2} - {VARIABLE_2_LABEL}")
print(
    f"Weighting: {'Enabled (two-stage ' + WEIGHT_VARIABLE + ')' if USE_WEIGHTS else 'Disabled'}"
)
print(f"Input: {INPUT_CSV}")
print(f"Output dir: {OUTDIR}")
print()

print("[1/4] Loading data...")
if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_CSV}\n"
        f"Expected at: DATA/DHS/women_all_answers_gps.csv"
    )

data = pd.read_csv(INPUT_CSV)
print(f"  ✓ Loaded {len(data):,} women's records")

# Standardize variable names (accept uppercase)
for v in (VARIABLE_1, VARIABLE_2, "v001", WEIGHT_VARIABLE):
    if v.upper() in data.columns and v not in data.columns:
        data.rename(columns={v.upper(): v}, inplace=True)

# Validate required columns
missing = [c for c in (VARIABLE_1, VARIABLE_2, "v001") if c not in data.columns]
if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns include: {list(data.columns)[:50]}{' ...' if len(data.columns) > 50 else ''}"
    )

print("\n[2/4] Processing variables...")
var1_binary = coerce_binary_yes_no(data[VARIABLE_1])
var2_binary = coerce_binary_yes_no(data[VARIABLE_2])

print(f"  → {var1_binary.notna().sum():,} women answered {VARIABLE_1}")
print(f"  → {var2_binary.notna().sum():,} women answered {VARIABLE_2}")

# Weights
if USE_WEIGHTS:
    if WEIGHT_VARIABLE in data.columns:
        weights = pd.to_numeric(data[WEIGHT_VARIABLE], errors="coerce") / 1_000_000.0
    else:
        print(f"  ⚠ Warning: {WEIGHT_VARIABLE} not found, using equal weights")
        weights = pd.Series(1.0, index=data.index)
else:
    weights = pd.Series(1.0, index=data.index)

print("\n[3/4] Aggregating by cluster (Stage 1)...")
temp_df = pd.DataFrame(
    {
        "cluster_id": data["v001"],
        "var1": var1_binary,
        "var2": var2_binary,
        "weight": weights,
    }
)

def cluster_summary(g: pd.DataFrame) -> pd.Series:
    return pd.Series(
        {
            "fraction_var1": weighted_prop(g["var1"], g["weight"]) if USE_WEIGHTS else g["var1"].mean(skipna=True),
            "fraction_var2": weighted_prop(g["var2"], g["weight"]) if USE_WEIGHTS else g["var2"].mean(skipna=True),
            "n_women": int(len(g)),
            "total_weight": float(g["weight"].sum()) if USE_WEIGHTS else float(len(g)),
            "n_answered_both": int((g["var1"].notna() & g["var2"].notna()).sum()),
        }
    )

cluster_stats = (
    temp_df.groupby("cluster_id", dropna=False)
    .apply(cluster_summary, include_groups=False)
    .reset_index()
)

# Keep only clusters where both fractions are defined
cluster_stats = cluster_stats.dropna(subset=["fraction_var1", "fraction_var2"])

print(f"  ✓ Aggregated to {len(cluster_stats):,} clusters with data for both variables")

output_csv = OUTDIR / "1b_cluster_correlation_data.csv"
cluster_stats.to_csv(output_csv, index=False)
print(f"  ✓ Saved cluster data: {output_csv}")

print("\n[4/4] Creating scatter plot (Stage 2)...")

if USE_WEIGHTS:
    pearson_r = weighted_correlation(
        cluster_stats["fraction_var1"],
        cluster_stats["fraction_var2"],
        cluster_stats["total_weight"],
    )
    slope, intercept = weighted_linregress(
        cluster_stats["fraction_var1"],
        cluster_stats["fraction_var2"],
        cluster_stats["total_weight"],
    )

    # Weighted R^2
    y_pred = slope * cluster_stats["fraction_var1"] + intercept
    y_actual = cluster_stats["fraction_var2"]
    w = cluster_stats["total_weight"].to_numpy(dtype=float)
    w_norm = w / w.sum()
    y_bar = np.average(y_actual, weights=w_norm)
    ss_tot = np.sum(w_norm * (y_actual - y_bar) ** 2)
    ss_res = np.sum(w_norm * (y_actual - y_pred) ** 2)
    r_squared = np.nan if ss_tot == 0 else float(1 - (ss_res / ss_tot))

    # p-value: unweighted fallback (common pragmatic choice)
    _, pearson_p = stats.pearsonr(cluster_stats["fraction_var1"], cluster_stats["fraction_var2"])
    weighting_note = f"Two-stage weighted ({WEIGHT_VARIABLE})"
else:
    pearson_r, pearson_p = stats.pearsonr(cluster_stats["fraction_var1"], cluster_stats["fraction_var2"])
    slope, intercept, r_value, p_value, std_err = stats.linregress(
        cluster_stats["fraction_var1"],
        cluster_stats["fraction_var2"],
    )
    r_squared = float(r_value**2)
    weighting_note = "Unweighted"

print(f"\n  Correlation Statistics:")
print(f"  → Pearson r = {pearson_r:.3f}")
print(f"  → p-value   = {pearson_p:.4f}")
print(f"  → R²        = {r_squared:.3f}")
print(f"  → Regression: y = {slope:.3f}x + {intercept:.3f}")

# =============================================================================
# PLOT
# =============================================================================

fig, ax = plt.subplots(figsize=FIGSIZE)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

sc = ax.scatter(
    cluster_stats["fraction_var1"],
    cluster_stats["fraction_var2"],
    s=POINT_SIZE,
    alpha=POINT_ALPHA,
    c=cluster_stats["total_weight"],
    cmap="viridis",
    edgecolors="black",
    linewidths=0.5,
    zorder=2,
)

cbar = plt.colorbar(sc, ax=ax)
cbar.set_label(f"Cluster total weight (sum {WEIGHT_VARIABLE})", fontsize=11)

x_line = np.array([0.0, 1.0])
y_line = slope * x_line + intercept

# Perfect-diagonal reference (y=x): thin, gray, low-opacity.
ax.plot(
    x_line,
    x_line,
    color="gray",
    linewidth=0.8,
    alpha=0.2,
    linestyle="-",
    zorder=0,
)

ax.plot(
    x_line,
    y_line,
    color=LINE_COLOR,
    linewidth=LINE_WIDTH,
    linestyle="--",
    alpha=0.8,
    label=f"Linear fit: y = {slope:.2f}x + {intercept:.2f}\n{weighting_note}",
    zorder=1,
)

ax.set_xlabel(f"Fraction - {VARIABLE_1_LABEL}", fontsize=13, fontweight="bold")
ax.set_ylabel(f"Fraction - {VARIABLE_2_LABEL}", fontsize=13, fontweight="bold")

title_suffix = (
    f"\n(Two-Stage Weighted: Stage 1 = within-cluster {WEIGHT_VARIABLE}, Stage 2 = across-cluster total {WEIGHT_VARIABLE})"
    if USE_WEIGHTS
    else "\n(Unweighted)"
)
ax.set_title(
    f"Cluster-Level Correlation Analysis\n{VARIABLE_1_LABEL} vs {VARIABLE_2_LABEL}{title_suffix}",
    fontsize=14,
    fontweight="bold",
    pad=20,
)

ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_aspect("equal", adjustable="box")

ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, p: f"{y:.0%}"))

ax.grid(True, alpha=0.3, linestyle=":", linewidth=0.5)

stats_text = (
    f"{'Weighted ' if USE_WEIGHTS else ''}Pearson r = {pearson_r:.3f}\n"
    f"p-value = {pearson_p:.4f}\n"
    f"R² = {r_squared:.3f}\n"
    f"n = {len(cluster_stats)} clusters"
)
if USE_WEIGHTS:
    stats_text += (
        f"\n\nStage 1: Within-cluster {WEIGHT_VARIABLE}"
        f"\nStage 2: Across-cluster total {WEIGHT_VARIABLE}"
    )

ax.text(
    0.05,
    0.95,
    stats_text,
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
)

ax.legend(loc="lower right", fontsize=10, framealpha=0.9)

output_fig = OUTDIR / "1b_cluster_correlation_scatter.jpg"
plt.savefig(output_fig, dpi=DPI, bbox_inches="tight")
plt.close()

print(f"\n  ✓ Saved scatter plot: {output_fig}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)
print(f"\nOUTPUTS:")
print(f"  1. {output_csv}")
print(f"  2. {output_fig}")
print("\n✅ Done!")
