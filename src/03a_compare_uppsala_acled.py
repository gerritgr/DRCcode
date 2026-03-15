#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare ACLED vs Uppsala GED event maps for the same years.

Run from project root:
    python src/03_compare_uppsala_acled.py

What this script does:
1) Ensures `background.jpg` exists in this folder:
   - If it exists: reuse it.
   - If missing: download Natural Earth layers and build it.
2) Loads ACLED and Uppsala event data.
3) Creates yearly comparison maps (one ACLED + one Uppsala per year).
4) Creates one aggregate map per dataset across selected years.
5) Saves high-resolution JPG and PDF outputs in `output/`.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import zipfile

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from shapely.geometry import box


# =============================================================================
# CONFIGURATION (EDIT HERE)
# =============================================================================

# Robust project-relative paths
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Input files (DATA/Conflict/...)
INPUT_ACLED_XLSX = (
    PROJECT_ROOT
    / "DATA"
    / "Conflict"
    / "Africa_lagged_data_up_to-2024-10-17"
    / "Africa_lagged_data_up_to-2024-10-17.xlsx"
)
INPUT_UPPSALA_CSV = (
    PROJECT_ROOT
    / "DATA"
    / "Conflict"
    / "Uppsala"
    / "GEDEvent_v25_1.csv"
)

# Output paths (output/03_...)
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Background image cache (also under output/03_...)
BACKGROUND_JPG = OUTPUT_DIR / "03_background.jpg"

# Years to render.
# Use explicit list, or set to None for automatic overlap between ACLED/Uppsala.
YEARS_OF_INTEREST: list[int] | None = [2020, 2021, 2022, 2023, 2024]

# Geographic extent [lon/lat] for DRC-focused map canvas.
MIN_LON = 11.893979235367297
MAX_LON = 31.616531541802978
MIN_LAT = -13.981788316118982
MAX_LAT = 5.811825978871415

# Output quality / size.
FIGSIZE = (10, 10)
DPI = 450
JPG_QUALITY = 95

# Event style (easy to change).
EVENT_COLORS = {
    "inside_drc": "green",   # requested: inside DRC should be green
    "outside_drc": "orange",
}
POINT_SIZE = 10
POINT_ALPHA = 0.72

# Background (Natural Earth) style.
DRC_BORDER_LINEWIDTH = 2.6
ADMIN1_LINEWIDTH = 0.9
OTHER_BORDER_LINEWIDTH = 1.4
LAKES_ALPHA = 0.28
HTTP_TIMEOUT_SECONDS = 120

# Background labeling (to match reference style)
SHOW_NEIGHBOR_LABELS = True
SHOW_CAPITAL_MARKER = True
SHOW_CAPITAL_LABEL = True
NEIGHBOR_LABEL_FONTSIZE = 13
CAPITAL_LABEL_FONTSIZE = 25
CAPITAL_LON = 15.266   # Kinshasa approx
CAPITAL_LAT = -4.441
CAPITAL_MARKER_SIZE = 800
CAPITAL_LABEL_DX = 0.50
CAPITAL_LABEL_DY = -0.15
CAPITAL_EXTENT_PAD_DEG = 0.25

# Natural Earth sources.
NE_URL_COUNTRIES = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip"
NE_URL_ADMIN1 = "https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_1_states_provinces.zip"
NE_URL_LAKES = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_lakes.zip"


# =============================================================================
# BACKGROUND MAP BUILDING (DOWNLOAD ONLY IF NEEDED)
# =============================================================================

def download_zip_bytes(url: str) -> bytes:
    """Download one ZIP file as bytes."""
    response = requests.get(url, timeout=HTTP_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.content


def extract_first_shp_from_zip(zip_bytes: bytes, dest_dir: Path) -> Path:
    """Extract ZIP bytes and return first .shp path found."""
    with zipfile.ZipFile(BytesIO(zip_bytes), "r") as zf:
        zf.extractall(dest_dir)
    shp_paths = sorted(dest_dir.rglob("*.shp"))
    if not shp_paths:
        raise FileNotFoundError(f"No .shp found after extracting to {dest_dir}")
    return shp_paths[0]


def find_drc_polygons(countries_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Find DRC polygons robustly from Natural Earth country layer."""
    for col in ["ADM0_A3", "ISO_A3", "SOV_A3", "GU_A3", "WB_A3"]:
        if col in countries_gdf.columns:
            mask = countries_gdf[col].astype(str).str.upper().eq("COD")
            if mask.any():
                return countries_gdf[mask]

    name_cols = [c for c in ["NAME", "NAME_EN", "ADMIN", "FORMAL_EN"] if c in countries_gdf.columns]
    if name_cols:
        s = countries_gdf[name_cols[0]].astype(str).str.lower()
        mask = (
            s.str.contains("democratic republic of the congo", na=False)
            | s.str.contains("congo, dem. rep", na=False)
            | s.str.contains("dr congo", na=False)
        )
        if mask.any():
            return countries_gdf[mask]

    raise RuntimeError("Could not identify DRC polygons in Natural Earth country layer.")


def build_background_image(background_path: Path) -> None:
    """
    Build and save `background.jpg` from Natural Earth layers.
    Called only if background image is missing.
    """
    print("  -> background.jpg not found, downloading Natural Earth data...")

    bbox_geom = box(MIN_LON, MIN_LAT, MAX_LON, MAX_LAT)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326")

    with tempfile.TemporaryDirectory(prefix="compare_ne_") as tmp:
        tmp_dir = Path(tmp)
        countries_shp = extract_first_shp_from_zip(download_zip_bytes(NE_URL_COUNTRIES), tmp_dir / "countries")
        admin1_shp = extract_first_shp_from_zip(download_zip_bytes(NE_URL_ADMIN1), tmp_dir / "admin1")
        lakes_shp = extract_first_shp_from_zip(download_zip_bytes(NE_URL_LAKES), tmp_dir / "lakes")

        countries = gpd.read_file(countries_shp).to_crs("EPSG:4326")
        admin1 = gpd.read_file(admin1_shp).to_crs("EPSG:4326")
        lakes = gpd.read_file(lakes_shp).to_crs("EPSG:4326")

    countries_clip = gpd.clip(countries, bbox_gdf)
    admin1_clip = gpd.clip(admin1, bbox_gdf)
    lakes_clip = gpd.clip(lakes, bbox_gdf)

    drc = find_drc_polygons(countries_clip)
    others = countries_clip.drop(index=drc.index, errors="ignore")

    if "adm0_a3" in admin1_clip.columns:
        admin1_drc = admin1_clip[admin1_clip["adm0_a3"].astype(str).str.upper().eq("COD")]
    else:
        admin1_drc = admin1_clip

    # Keep your original extent unless Kinshasa would fall outside.
    min_lon2, max_lon2 = MIN_LON, MAX_LON
    min_lat2, max_lat2 = MIN_LAT, MAX_LAT
    if not (MIN_LON <= CAPITAL_LON <= MAX_LON):
        min_lon2 = min(MIN_LON, CAPITAL_LON - CAPITAL_EXTENT_PAD_DEG)
        max_lon2 = max(MAX_LON, CAPITAL_LON + CAPITAL_EXTENT_PAD_DEG)
    if not (MIN_LAT <= CAPITAL_LAT <= MAX_LAT):
        min_lat2 = min(MIN_LAT, CAPITAL_LAT - CAPITAL_EXTENT_PAD_DEG)
        max_lat2 = max(MAX_LAT, CAPITAL_LAT + CAPITAL_EXTENT_PAD_DEG)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    # 1) White base
    countries_clip.plot(ax=ax, facecolor="white", linewidth=0)

    # 2) Lakes
    if len(lakes_clip) > 0:
        lakes_clip.plot(ax=ax, linewidth=0, alpha=LAKES_ALPHA)

    # 3) DRC admin-1 borders
    if len(admin1_drc) > 0:
        admin1_drc.boundary.plot(ax=ax, linewidth=ADMIN1_LINEWIDTH, alpha=0.95)

    # 4) Neighbor country borders
    if len(others) > 0:
        others.boundary.plot(ax=ax, linewidth=OTHER_BORDER_LINEWIDTH, alpha=0.15)

    # 5) DRC border emphasized
    if len(drc) > 0:
        drc.boundary.plot(ax=ax, linewidth=DRC_BORDER_LINEWIDTH, alpha=1.0)

    # 6) Neighbor labels
    if SHOW_NEIGHBOR_LABELS and len(others) > 0:
        label_col = None
        for candidate in ["NAME_EN", "NAME", "ADMIN", "FORMAL_EN"]:
            if candidate in others.columns:
                label_col = candidate
                break
        if label_col is not None:
            pts = others.geometry.representative_point()
            for (x, y), name in zip(zip(pts.x, pts.y), others[label_col].astype(str)):
                ax.text(
                    x,
                    y,
                    name,
                    fontsize=NEIGHBOR_LABEL_FONTSIZE,
                    alpha=0.5,
                    ha="center",
                    va="center",
                )

    # 7) Kinshasa marker + label
    if SHOW_CAPITAL_MARKER:
        ax.scatter(
            [CAPITAL_LON],
            [CAPITAL_LAT],
            marker="*",
            s=CAPITAL_MARKER_SIZE,
            c="black",
            edgecolors="white",
            alpha=0.9,
        )
    if SHOW_CAPITAL_LABEL:
        ax.text(
            CAPITAL_LON + CAPITAL_LABEL_DX,
            CAPITAL_LAT + CAPITAL_LABEL_DY,
            "Kinshasa",
            fontsize=CAPITAL_LABEL_FONTSIZE,
            c="black",
        )

    ax.set_xlim(min_lon2, max_lon2)
    ax.set_ylim(min_lat2, max_lat2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()

    plt.tight_layout(pad=0)
    plt.savefig(
        background_path,
        dpi=DPI,
        bbox_inches="tight",
        pad_inches=0,
        pil_kwargs={"quality": JPG_QUALITY},
    )
    plt.close(fig)
    print(f"  -> Saved: {background_path.resolve()}")


def ensure_background_image(background_path: Path) -> None:
    """Create background image once; reuse it afterward."""
    if background_path.exists():
        print(f"  -> Reusing existing background: {background_path.resolve()}")
        print("  -> Delete this file to force a rebuild with latest style.")
        return
    build_background_image(background_path)


# =============================================================================
# DATA LOAD / TRANSFORM
# =============================================================================

def resolve_column_name(columns: list[str], candidates: list[str]) -> str | None:
    """Case-insensitive column resolver."""
    lookup = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def load_acled_events(path: Path) -> pd.DataFrame:
    """Load ACLED into canonical columns: year, lon, lat, country."""
    if not path.exists():
        raise FileNotFoundError(f"Missing ACLED file: {path}")

    header = pd.read_excel(path, nrows=0, engine="openpyxl").columns.tolist()
    lat_col = resolve_column_name(header, ["LATITUDE"])
    lon_col = resolve_column_name(header, ["LONGITUDE"])
    year_col = resolve_column_name(header, ["YEAR"])
    date_col = resolve_column_name(header, ["EVENT_DATE"])
    country_col = resolve_column_name(header, ["COUNTRY"])

    if lat_col is None or lon_col is None:
        raise ValueError("ACLED file needs LATITUDE and LONGITUDE.")

    usecols = [lat_col, lon_col]
    if year_col is not None:
        usecols.append(year_col)
    if date_col is not None and date_col not in usecols:
        usecols.append(date_col)
    if country_col is not None:
        usecols.append(country_col)

    raw = pd.read_excel(path, usecols=usecols, engine="openpyxl")

    out = pd.DataFrame()
    out["lat"] = pd.to_numeric(raw[lat_col], errors="coerce")
    out["lon"] = pd.to_numeric(raw[lon_col], errors="coerce")
    if year_col is not None:
        out["year"] = pd.to_numeric(raw[year_col], errors="coerce").astype("Int64")
    elif date_col is not None:
        out["year"] = pd.to_datetime(raw[date_col], errors="coerce").dt.year.astype("Int64")
    else:
        raise ValueError("ACLED file needs YEAR or EVENT_DATE.")

    out["country"] = raw[country_col].astype(str) if country_col is not None else ""
    out = out[out["lat"].notna() & out["lon"].notna() & out["year"].notna()].copy()
    out["year"] = out["year"].astype(int)
    return out


def load_uppsala_events(path: Path) -> pd.DataFrame:
    """Load Uppsala GED into canonical columns: year, lon, lat, country."""
    if not path.exists():
        raise FileNotFoundError(f"Missing Uppsala file: {path}")

    header = pd.read_csv(path, nrows=0).columns.tolist()
    lat_col = resolve_column_name(header, ["latitude", "LATITUDE"])
    lon_col = resolve_column_name(header, ["longitude", "LONGITUDE"])
    year_col = resolve_column_name(header, ["year", "YEAR"])
    date_col = resolve_column_name(header, ["date_start", "DATE_START"])
    country_col = resolve_column_name(header, ["country", "COUNTRY"])

    if lat_col is None or lon_col is None:
        raise ValueError("Uppsala file needs latitude and longitude.")

    usecols = [lat_col, lon_col]
    if year_col is not None:
        usecols.append(year_col)
    if date_col is not None and date_col not in usecols:
        usecols.append(date_col)
    if country_col is not None:
        usecols.append(country_col)

    raw = pd.read_csv(path, usecols=usecols, low_memory=False)

    out = pd.DataFrame()
    out["lat"] = pd.to_numeric(raw[lat_col], errors="coerce")
    out["lon"] = pd.to_numeric(raw[lon_col], errors="coerce")
    if year_col is not None:
        out["year"] = pd.to_numeric(raw[year_col], errors="coerce").astype("Int64")
    elif date_col is not None:
        out["year"] = pd.to_datetime(raw[date_col], errors="coerce").dt.year.astype("Int64")
    else:
        raise ValueError("Uppsala file needs year or date_start.")

    out["country"] = raw[country_col].astype(str) if country_col is not None else ""
    out = out[out["lat"].notna() & out["lon"].notna() & out["year"].notna()].copy()
    out["year"] = out["year"].astype(int)
    return out


def filter_to_extent(df: pd.DataFrame) -> pd.DataFrame:
    """Filter events to map extent."""
    return df[
        (df["lon"] >= MIN_LON) & (df["lon"] <= MAX_LON)
        & (df["lat"] >= MIN_LAT) & (df["lat"] <= MAX_LAT)
    ].copy()


def classify_inside_drc_from_country(country_series: pd.Series) -> pd.Series:
    """
    Classify event location as inside DRC using country names.
    """
    s = country_series.astype(str).str.lower().str.strip()
    return (
        s.str.contains("democratic republic of congo", na=False)
        | s.str.contains("democratic republic of the congo", na=False)
        | s.str.contains("dr congo", na=False)
        | s.str.contains("congo, dem", na=False)
        | s.str.contains("dr congo \\(zaire\\)", na=False)
        | s.eq("drc")
    )


def add_inside_drc_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add inside/outside DRC flag."""
    out = df.copy()
    out["inside_drc"] = classify_inside_drc_from_country(out["country"])
    return out


def determine_years(acled_df: pd.DataFrame, uppsala_df: pd.DataFrame) -> list[int]:
    """Resolve year list from config or overlap."""
    if YEARS_OF_INTEREST is not None:
        return sorted(int(y) for y in YEARS_OF_INTEREST)

    acled_years = set(acled_df["year"].dropna().astype(int).tolist())
    uppsala_years = set(uppsala_df["year"].dropna().astype(int).tolist())
    years = sorted(acled_years.intersection(uppsala_years))
    if not years:
        raise ValueError("No overlapping years between ACLED and Uppsala.")
    return years


# =============================================================================
# PLOTTING
# =============================================================================

def save_event_map(
    events_df: pd.DataFrame,
    dataset_label: str,
    title_suffix: str,
    out_stem: Path,
    background_image: np.ndarray,
) -> dict:
    """
    Save one event map as high-res JPG and PDF.
    Returns metadata row for output manifest.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    ax.imshow(
        background_image,
        extent=[MIN_LON, MAX_LON, MIN_LAT, MAX_LAT],
        origin="upper",
        zorder=0,
    )

    inside = events_df[events_df["inside_drc"]]
    outside = events_df[~events_df["inside_drc"]]

    if not inside.empty:
        ax.scatter(
            inside["lon"],
            inside["lat"],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            linewidths=0,
            color=EVENT_COLORS["inside_drc"],
            label=f"Within DRC (n={len(inside):,})",
            zorder=2,
        )
    if not outside.empty:
        ax.scatter(
            outside["lon"],
            outside["lat"],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            linewidths=0,
            color=EVENT_COLORS["outside_drc"],
            label=f"Outside DRC (n={len(outside):,})",
            zorder=3,
        )

    ax.set_xlim(MIN_LON, MAX_LON)
    ax.set_ylim(MIN_LAT, MAX_LAT)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.set_title(
        f"{dataset_label} Events - {title_suffix}\nTotal events in extent: {len(events_df):,}",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )

    if len(events_df) > 0:
        ax.legend(loc="lower right", frameon=True, framealpha=0.9, fontsize=9)

    out_jpg = out_stem.with_suffix(".jpg")
    out_pdf = out_stem.with_suffix(".pdf")
    plt.tight_layout()
    plt.savefig(out_jpg, dpi=DPI, bbox_inches="tight", pad_inches=0.03, pil_kwargs={"quality": JPG_QUALITY})
    plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    return {
        "dataset": dataset_label,
        "title_suffix": title_suffix,
        "n_total": int(len(events_df)),
        "n_inside_drc": int(len(inside)),
        "n_outside_drc": int(len(outside)),
        "jpg_path": str(out_jpg.resolve()),
        "pdf_path": str(out_pdf.resolve()),
    }


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    print("\n" + "=" * 78)
    print("COMPARE ACLED VS UPPSALA EVENT MAPS")
    print("=" * 78)

    if not INPUT_ACLED_XLSX.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_ACLED_XLSX}")
    if not INPUT_UPPSALA_CSV.exists():
        raise FileNotFoundError(f"Missing input: {INPUT_UPPSALA_CSV}")

    print("\n[1/6] Ensuring background image...")
    ensure_background_image(BACKGROUND_JPG)
    bg = plt.imread(BACKGROUND_JPG)

    print("\n[2/6] Loading datasets...")
    acled_raw = load_acled_events(INPUT_ACLED_XLSX)
    uppsala_raw = load_uppsala_events(INPUT_UPPSALA_CSV)
    print(f"  -> ACLED rows loaded: {len(acled_raw):,}")
    print(f"  -> Uppsala rows loaded: {len(uppsala_raw):,}")

    print("\n[3/6] Filtering to map extent...")
    acled = filter_to_extent(acled_raw)
    uppsala = filter_to_extent(uppsala_raw)
    print(f"  -> ACLED rows in extent: {len(acled):,}")
    print(f"  -> Uppsala rows in extent: {len(uppsala):,}")

    print("\n[4/6] Classifying place (within DRC vs outside)...")
    acled = add_inside_drc_flag(acled)
    uppsala = add_inside_drc_flag(uppsala)

    years = determine_years(acled, uppsala)
    print(f"\n[5/6] Years to render: {years}")

    print("\n[6/6] Generating yearly + aggregate maps...")
    manifest_rows: list[dict] = []

    # Yearly maps: two images per year (ACLED + Uppsala)
    for year in years:
        acled_year = acled[acled["year"] == year].copy()
        uppsala_year = uppsala[uppsala["year"] == year].copy()

        meta = save_event_map(
            events_df=acled_year,
            dataset_label="ACLED",
            title_suffix=f"Year {year}",
            out_stem=OUTPUT_DIR / f"03_compare_acled_{year}",
            background_image=bg,
        )
        meta["year"] = year
        manifest_rows.append(meta)

        meta = save_event_map(
            events_df=uppsala_year,
            dataset_label="Uppsala GED",
            title_suffix=f"Year {year}",
            out_stem=OUTPUT_DIR / f"03_compare_uppsala_{year}",
            background_image=bg,
        )
        meta["year"] = year
        manifest_rows.append(meta)

    # Aggregate maps
    acled_all = acled[acled["year"].isin(years)].copy()
    uppsala_all = uppsala[uppsala["year"].isin(years)].copy()
    year_span = f"{min(years)}-{max(years)}" if years else "all"

    meta = save_event_map(
        events_df=acled_all,
        dataset_label="ACLED",
        title_suffix=f"Aggregate ({year_span})",
        out_stem=OUTPUT_DIR / "03_compare_acled_all_years",
        background_image=bg,
    )
    meta["year"] = "all"
    manifest_rows.append(meta)

    meta = save_event_map(
        events_df=uppsala_all,
        dataset_label="Uppsala GED",
        title_suffix=f"Aggregate ({year_span})",
        out_stem=OUTPUT_DIR / "03_compare_uppsala_all_years",
        background_image=bg,
    )
    meta["year"] = "all"
    manifest_rows.append(meta)

    manifest = pd.DataFrame(manifest_rows)[
        ["dataset", "year", "title_suffix", "n_total", "n_inside_drc", "n_outside_drc", "jpg_path", "pdf_path"]
    ]
    manifest_path = OUTPUT_DIR / "03_compare_image_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    print("\nGenerated files:")
    for _, row in manifest.iterrows():
        print(f"  - {Path(row['jpg_path']).name}")
        print(f"  - {Path(row['pdf_path']).name}")
    print(f"\nManifest: {manifest_path.resolve()}")
    print(f"Output dir: {OUTPUT_DIR.resolve()}")
    print("\nDone.")


if __name__ == "__main__":
    main()
