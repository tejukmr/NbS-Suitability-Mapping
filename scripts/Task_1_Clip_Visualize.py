# Environmental programming using Python

# Assignment topic: Suitability Mapping of Nature-Based Solutions Locations to Tackle Hydroclimatic Extremes and Water
# Quality Degradation Using Machine Learning 

# Group-4:
# Elias Zgheib
# Ndra Malky
# Rashmi Krishnamurthy
# Teju Kumar Nagaraju

# This script checks the input raster extent, visualizes the rasters and
# clips to the country boundary shapefile and saves it to the individual folder.
# Eg. LULC folder is saved to LULC_clipped 

import os
import geopandas as gpd
import rasterio
from rasterio.mask import mask

# NEW imports for plotting/stats
import numpy as np
import matplotlib.pyplot as plt
from rasterio.enums import Resampling

# User Inputs

country_boundary = r"EU_Country_New.shp"  # Provide the path to boundary shapefile

folders = [                                                       # Provide folder path to raster files
    r"AET",
    r"LULC",
    r"Precipitation",
    r"RootZoneSoilMoisture",
    r"Soil",
    r"Temperature_Mean"
]

SUFFIX = "_clipped"
OVERWRITE = False


# NEW: PLOT + STATS (BEFORE CLIPPING)


PLOT_DIR = os.path.join(os.path.dirname(folders[0]), "Layer_Plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# how many rasters to plot per folder (here 1 is plotted to verify)
PLOT_MAX_PER_FOLDER = 1

def raster_quicklook_and_stats(raster_path: str, out_png: str):
    with rasterio.open(raster_path) as src:
        if src.crs is None:
            raise ValueError(f"Raster has no CRS defined: {raster_path}")

        # resolution (pixel size)
        # transform.a = pixel width, transform.e = -pixel height (often negative)
        px_w = float(src.transform.a)
        px_h = float(abs(src.transform.e))

        # downsample for plotting
        target = 800
        scale = max(src.width / target, src.height / target, 1.0)
        out_w = int(src.width / scale)
        out_h = int(src.height / scale)

        arr = src.read(
            1,
            masked=True,
            out_shape=(out_h, out_w),
            resampling=Resampling.nearest
        )

        # stats ignoring masked/nodata
        data = arr.compressed()  # 1D np array of valid pixels
        if data.size == 0:
            vmin = vmax = p2 = p98 = np.nan
        else:
            vmin = float(np.min(data))
            vmax = float(np.max(data))
            p2 = float(np.percentile(data, 2))
            p98 = float(np.percentile(data, 98))

        print("\n--- INPUT LAYER CHECK ---")
        print("File      :", raster_path)
        print("CRS       :", src.crs)
        print("Size      :", src.width, "x", src.height)
        print("PixelSize :", px_w, "x", px_h)
        print("NoData    :", src.nodata)
        print("Min/Max   :", vmin, "/", vmax)
        print("P2/P98    :", p2, "/", p98)

        # plot (stretch using p2-p98 for continuous layers)
        plt.figure(figsize=(7, 5))
        if np.isfinite(p2) and np.isfinite(p98) and p2 != p98:
            plt.imshow(arr, vmin=p2, vmax=p98)
        else:
            plt.imshow(arr)

        plt.title(os.path.basename(raster_path))
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()

# Loop folders and plot at least one tif per layer
print("\n================= PRE-CLIP INPUT CHECK =================")
for folder in folders:
    if not os.path.isdir(folder):
        print(f"Folder not found, skipping: {folder}")
        continue

    tifs = [f for f in os.listdir(folder) if f.lower().endswith(".tif")]
    if not tifs:
        print(f"No tif found in: {folder}")
        continue

    tifs = sorted(tifs)[:PLOT_MAX_PER_FOLDER]

    for f in tifs:
        in_raster = os.path.join(folder, f)
        out_png = os.path.join(
            PLOT_DIR,
            f"{os.path.basename(os.path.normpath(folder))}__{os.path.splitext(f)[0]}.png"
        )
        raster_quicklook_and_stats(in_raster, out_png)
        print("Saved plot:", out_png)

print("\nPre-clip plotting done. Plots saved at:", PLOT_DIR)
print("========================================================\n")

# Load Country Boundary

gdf = gpd.read_file(country_boundary)
if gdf.crs is None:                                                 # To check whether Country boundary has valid CRS
    raise ValueError("Country boundary shapefile has no CRS defined")

gdf = gdf.dissolve()

# CLIP ALL TIFFs
# Output: E:\VUB\Final\<Param>_Clipped\<all clipped files>

for folder in folders:
    if not os.path.isdir(folder):
        print(f"Folder not found, skipping: {folder}")
        continue

    param_name = os.path.basename(os.path.normpath(folder))   # e.g., AET
    out_dir = os.path.join(os.path.dirname(folder), f"{param_name}_Clipped")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nProcessing folder: {folder}")
    print(f"Output folder     : {out_dir}")

    for fname in os.listdir(folder):
        if not fname.lower().endswith(".tif"):
            continue

        in_raster = os.path.join(folder, fname)
        name, ext = os.path.splitext(fname)
        out_raster = os.path.join(out_dir, f"{name}{SUFFIX}{ext}")

        if (not OVERWRITE) and os.path.exists(out_raster):              # To check whether clipped data already exists
            print(f"Exists, skipping: {out_raster}")
            continue

        with rasterio.open(in_raster) as src:
            if src.crs is None:
                raise ValueError(f"Raster has no CRS defined: {in_raster}")

            # Reproject boundary to raster CRS if required
            if gdf.crs != src.crs:
                gdf_proj = gdf.to_crs(src.crs)
                geom = [gdf_proj.geometry.iloc[0]]
            else:
                geom = [gdf.geometry.iloc[0]]

            clipped, transform = mask(
                src,
                geom,
                crop=True,
                nodata=src.nodata
            )

            profile = src.profile.copy()
            profile.update(
                height=clipped.shape[1],
                width=clipped.shape[2],
                transform=transform,
                compress="lzw"
            )

            with rasterio.open(out_raster, "w", **profile) as dst:
                dst.write(clipped)

        print(f"Clipped -> {out_raster}")

print("\nAll specified folders processed successfully.")

