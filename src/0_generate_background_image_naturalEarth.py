import os
import shutil
import zipfile
from io import BytesIO

import requests
import geopandas as gpd
from shapely.geometry import box, Point
import matplotlib.pyplot as plt


# ------------------ EXACT SQUARE (your coords; [lon, lat]) ------------------
min_lon = 11.893979235367297
max_lon = 31.616531541802978
min_lat = -13.981788316118982
max_lat = 5.811825978871415

OUTPATH = "../DATA/background_ne.png"  # final image saved here

# temporary working directory (must not already exist)
out_dir = "_TEMP_aoi_square_outputs"
if os.path.exists(out_dir):
    raise RuntimeError(f"Temporary folder already exists: {out_dir}")
os.makedirs(out_dir, exist_ok=False)

try:
    # ------------------ Natural Earth downloads ------------------
    NE_BASE = "https://naciscdn.org/naturalearth/10m/cultural/"
    NE_ZIPS = {
        "countries": "ne_10m_admin_0_countries.zip",
        "admin1": "ne_10m_admin_1_states_provinces.zip",
    }

    def download_to_bytes(url: str) -> bytes:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        return r.content

    def extract_zip_bytes(zip_bytes: bytes, dest_dir: str) -> str:
        """Extract zip bytes into dest_dir. Returns path to first .shp found."""
        with zipfile.ZipFile(BytesIO(zip_bytes), "r") as z:
            z.extractall(dest_dir)
        for root, _, files in os.walk(dest_dir):
            for f in files:
                if f.lower().endswith(".shp"):
                    return os.path.join(root, f)
        raise FileNotFoundError(f"No .shp found after extracting to {dest_dir}")

    countries_shp = extract_zip_bytes(
        download_to_bytes(NE_BASE + NE_ZIPS["countries"]),
        os.path.join(out_dir, "ne_countries"),
    )
    admin1_shp = extract_zip_bytes(
        download_to_bytes(NE_BASE + NE_ZIPS["admin1"]),
        os.path.join(out_dir, "ne_admin1"),
    )

    countries = gpd.read_file(countries_shp).to_crs("EPSG:4326")
    admin1 = gpd.read_file(admin1_shp).to_crs("EPSG:4326")

    # ------------------ Clip to bbox ------------------
    bbox_geom = box(min_lon, min_lat, max_lon, max_lat)
    bbox_gdf = gpd.GeoDataFrame(geometry=[bbox_geom], crs="EPSG:4326")

    countries_clip = gpd.clip(countries, bbox_gdf)
    admin1_clip = gpd.clip(admin1, bbox_gdf)

    # ------------------ Identify DRC ------------------
    # Natural Earth typically uses ISO3 in ADM0_A3 (DRC = "COD")
    # Fallbacks included in case a field differs.
    drc_mask = None
    for col in ["ADM0_A3", "ISO_A3", "SOV_A3", "GU_A3", "WB_A3"]:
        if col in countries_clip.columns:
            drc_mask = (countries_clip[col] == "COD")
            if drc_mask.any():
                break
    if drc_mask is None or not drc_mask.any():
        # fallback by name
        name_cols = [c for c in ["NAME", "ADMIN", "NAME_EN", "FORMAL_EN"] if c in countries_clip.columns]
        if name_cols:
            nm = countries_clip[name_cols[0]].astype(str).str.lower()
            drc_mask = nm.str.contains("democratic republic of the congo") | nm.str.contains("congo, dem. rep.")
        else:
            # last resort: nothing found -> treat all as non-DRC
            drc_mask = countries_clip.index.to_series().map(lambda _: False)

    drc = countries_clip[drc_mask]
    others = countries_clip[~drc_mask]

    # ------------------ Capital: Kinshasa ------------------
    # Kinshasa approx coordinates
    kin_lon, kin_lat = 15.266, -4.441

    # Ensure capital is inside the rendered extent (expand bbox minimally if needed)
    min_lon2, max_lon2 = min_lon, max_lon
    min_lat2, max_lat2 = min_lat, max_lat

    pad = 0.25  # degrees padding if Kinshasa is just outside
    if not (min_lon <= kin_lon <= max_lon):
        min_lon2 = min(min_lon, kin_lon - pad)
        max_lon2 = max(max_lon, kin_lon + pad)
    if not (min_lat <= kin_lat <= max_lat):
        min_lat2 = min(min_lat, kin_lat - pad)
        max_lat2 = max(max_lat, kin_lat + pad)

    # ------------------ Stylized render ------------------
    fig, ax = plt.subplots(figsize=(10, 10), dpi=450)

    # Background (white)
    countries_clip.plot(ax=ax, facecolor="white", linewidth=0)

    # Provinces inside DRC (thin)
    # Natural Earth admin1 has "adm0_a3" for country code in most versions
    if "adm0_a3" in admin1_clip.columns:
        admin1_drc = admin1_clip[admin1_clip["adm0_a3"] == "COD"]
    else:
        # fallback by name if needed
        admin1_drc = admin1_clip
    admin1_drc.boundary.plot(ax=ax, linewidth=0.9, alpha=0.95)

    # Other countries' borders (lower opacity)
    if len(others) > 0:
        others.boundary.plot(ax=ax, linewidth=1.4, alpha=0.15)

    # DRC border (high emphasis)
    if len(drc) > 0:
        drc.boundary.plot(ax=ax, linewidth=2.6, alpha=1.0)

    # Capital marker + label
    ax.scatter([kin_lon], [kin_lat], s=100, c="black", edgecolors="white")  # marker
    ax.text(kin_lon + 0.15, kin_lat - 0.15, "Kinshasa", fontsize=25, c="black")

    # Extent
    ax.set_xlim(min_lon2, max_lon2)
    ax.set_ylim(min_lat2, max_lat2)
    ax.set_axis_off()
    plt.tight_layout(pad=0)

    # Save final image
    os.makedirs(os.path.dirname(OUTPATH), exist_ok=True)
    plt.savefig(OUTPATH, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    print("Saved:")
    print(" - Final background:", os.path.abspath(OUTPATH))

finally:
    shutil.rmtree(out_dir, ignore_errors=True)
