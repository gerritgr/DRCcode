import math
import os
import requests
import shutil
from io import BytesIO
from PIL import Image  # ImageDraw removed (no red border)

# ------------------ EXACT SQUARE (your coords; [lon, lat]) ------------------
min_lon = 11.893979235367297
max_lon = 31.616531541802978
min_lat = -13.981788316118982
max_lat = 5.811825978871415

OUTPATH = "../DATA/background.png"  # final copy saved to ../DATA with this name

# Zoom/tile config
zoom = 7
tile_size = 256
headers = {'User-Agent': 'MapAssembler/1.0 (personal, non-commercial)'}
tile_url = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

# temporary working directory (must not already exist)
out_dir = "_TEMP_aoi_square_outputs"
if os.path.exists(out_dir):
    raise RuntimeError(f"Temporary folder already exists: {out_dir}")

os.makedirs(out_dir, exist_ok=False)

stitched_path = os.path.join(out_dir, "tiles_stitched.png")
cropped_path  = os.path.join(out_dir, "aoiSquare_exact.png")

# ------------------ helpers ------------------
def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = (lon_deg + 180.0) / 360.0 * n
    ytile = (1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n
    return xtile, ytile

try:
    # Compute tile range that COVERS the bbox (may include margins)
    n = 2.0 ** zoom
    x_min_frac = (min_lon + 180.0) / 360.0 * n
    x_max_frac = (max_lon + 180.0) / 360.0 * n
    y_top_frac = (1.0 - math.asinh(math.tan(math.radians(max_lat))) / math.pi) / 2.0 * n
    y_bot_frac = (1.0 - math.asinh(math.tan(math.radians(min_lat))) / math.pi) / 2.0 * n

    min_xtile = math.floor(x_min_frac)
    max_xtile = math.floor(x_max_frac)
    min_ytile = math.floor(y_top_frac)     # north
    max_ytile = math.floor(y_bot_frac)     # south

    img_width  = (max_xtile - min_xtile + 1) * tile_size
    img_height = (max_ytile - min_ytile + 1) * tile_size

    # Download & stitch tiles
    if not os.path.exists(stitched_path):
        mosaic = Image.new('RGB', (img_width, img_height))
        for dx in range(max_xtile - min_xtile + 1):
            for dy in range(max_ytile - min_ytile + 1):
                x = min_xtile + dx
                y = min_ytile + dy
                url = tile_url.format(z=zoom, x=x, y=y)
                r = requests.get(url, headers=headers, timeout=30)
                if r.status_code == 200:
                    tile = Image.open(BytesIO(r.content)).convert('RGB')
                else:
                    tile = Image.new('RGB', (tile_size, tile_size), (200, 200, 200))
                mosaic.paste(tile, (dx * tile_size, dy * tile_size))
        mosaic.save(stitched_path)
    else:
        mosaic = Image.open(stitched_path)

    # Exact crop in pixel space using bbox corners (no red border)
    xtl, ytl = deg2num(max_lat, min_lon, zoom)   # top-left
    xbr, ybr = deg2num(min_lat, max_lon, zoom)  # bottom-right

    left   = int(round((xtl - min_xtile) * tile_size))
    top    = int(round((ytl - min_ytile) * tile_size))
    right  = int(round((xbr - min_xtile) * tile_size))
    bottom = int(round((ybr - min_ytile) * tile_size))

    cropped = mosaic.crop((left, top, right, bottom))

    # Optional fixed-size resample
    # target = 4096
    # cropped = cropped.resize((target, target), Image.BICUBIC)

    # Save inside outputs folder AND copy to ../DATA as OUTPATH
    cropped.save(cropped_path)
    cropped.save(OUTPATH)  # <- requested: final image in ../DATA/background.png

    print("Saved:")
    print(" - Stitched tiles:", stitched_path)
    print(" - EXACT cropped square:", cropped_path)
    print(" - Final background:", os.path.abspath(OUTPATH))
    print("Pixel size:", cropped.width, "x", cropped.height)

finally:
    # remove temporary working directory
    shutil.rmtree(out_dir, ignore_errors=True)
