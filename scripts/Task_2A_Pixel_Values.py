# Environmental programming using Python

# Assignment topic: Suitability Mapping of Nature-Based Solutions Locations to Tackle Hydroclimatic Extremes and Water
# Quality Degradation Using Machine Learning 

# Group-4:
# Elias Zgheib
# Ndra Malky
# Rashmi Krishnamurthy
# Teju Kumar Nagaraju

# This script utilizes the previously clipped raster to extract the pixel values for all data parameters and 
# save it to the dataframe (csv). Pixel values are extracted for pixels lying within the boundary


import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import reproject, Resampling

# User inputs

folders = {
    "AET":  r"AET_Clipped",
    "LULC": r"LULC_Clipped",
    "P":    r"Precipitation_Clipped",
    "RZSM": r"RootZoneSoilMoisture_Clipped",
    "TEMP": r"Temperature_Mean_Clipped",
}
SOIL_FILE = r"Soil_HSG_10km_clipped.tif"   # constant soil raster

YEARS = range(2014, 2025)                                           # Adjust range based on data availability
OUT_CSV = r"PixelDataFrames\pixels_2014_2024_all_inside.csv" # You may change the output location

# categorical layers
CATEGORICAL = {"LULC", "SOIL"}

# OPTIONAL: treat value 0 as nodata for these layers (common for LULC/Soil background)
ZERO_AS_NODATA = {"LULC", "SOIL"}

# -----------------------------------------------------------
os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)

def find_year_tif(folder: str, year: int) -> str:
    y = str(year)
    for f in os.listdir(folder):
        if f.lower().endswith(".tif") and y in f:
            return os.path.join(folder, f)
    return None

def window_coords_pixelid(ref_ds, win):
    W = ref_ds.width
    rows = np.arange(win.row_off, win.row_off + win.height, dtype=np.int32)
    cols = np.arange(win.col_off, win.col_off + win.width, dtype=np.int32)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")

    pixel_id = (rr.astype(np.int64) * np.int64(W) + cc.astype(np.int64)).ravel()
    xs, ys = rasterio.transform.xy(ref_ds.transform, rr, cc, offset="center")
    lon = np.asarray(xs, dtype=np.float64).ravel()
    lat = np.asarray(ys, dtype=np.float64).ravel()
    return rr.ravel(), cc.ravel(), lon, lat, pixel_id

def read_aligned_window(src_ds, ref_ds, win, layer_name: str, is_categorical: bool):
    """
    Read data aligned to ref window. If grids match, read directly.
    Else, reproject/resample into ref window.
    Returns float32 with NaN for nodata.
    """
    # Fast path: already aligned to ref grid
    if (src_ds.crs == ref_ds.crs and
        src_ds.transform == ref_ds.transform and
        src_ds.width == ref_ds.width and
        src_ds.height == ref_ds.height):
        arr = src_ds.read(1, window=win).astype(np.float32)
        nodata = src_ds.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        if layer_name in ZERO_AS_NODATA:
            arr = np.where(arr == 0, np.nan, arr)
        return arr

    # Reproject to ref window
    dst_transform = rasterio.windows.transform(win, ref_ds.transform)
    dst_h, dst_w = win.height, win.width

    dst = np.full((dst_h, dst_w), np.nan, dtype=np.float32)

    resamp = Resampling.nearest if is_categorical else Resampling.bilinear

    reproject(
        source=rasterio.band(src_ds, 1),
        destination=dst,
        src_transform=src_ds.transform,
        src_crs=src_ds.crs,
        dst_transform=dst_transform,
        dst_crs=ref_ds.crs,
        src_nodata=src_ds.nodata,
        dst_nodata=np.nan,
        resampling=resamp
    )

    if layer_name in ZERO_AS_NODATA:
        dst = np.where(dst == 0, np.nan, dst)

    return dst

# -------------------- PRE-OPEN SOIL ONCE --------------------
if not os.path.exists(SOIL_FILE):
    raise FileNotFoundError(f"SOIL_FILE not found: {SOIL_FILE}")

soil_ds = rasterio.open(SOIL_FILE)

# -------------------- BUILD ONE BIG CSV (INSIDE ONLY) --------------------
if os.path.exists(OUT_CSV):
    os.remove(OUT_CSV)

first_write = True

for year in YEARS:
    print(f"\n=== Year {year} ===")

    year_files = {}
    for var, folder in folders.items():
        fp = find_year_tif(folder, year)
        if fp is None:
            raise FileNotFoundError(f"Missing {var} tif for year {year} in: {folder}")
        year_files[var] = fp

    with rasterio.open(year_files["AET"]) as ref:
        open_year = {v: rasterio.open(p) for v, p in year_files.items()}

        try:
            chunks = []
            for _, win in ref.block_windows(1):
                # --- Create INSIDE mask from AET itself ---
                # Use masked read so outside polygon (nodata) becomes masked
                aet_masked = ref.read(1, window=win, masked=True)  # masked array
                inside = (~aet_masked.mask).ravel()               # True only inside polygon

                # If nothing inside in this window, skip
                if not inside.any():
                    continue

                # coords/id
                row, col, lon, lat, pixel_id = window_coords_pixelid(ref, win)

                # Keep only inside pixels
                data = {
                    "year": np.full(inside.sum(), year, dtype=np.int16),
                    "pixel_id": pixel_id[inside],
                    "row": row[inside],
                    "col": col[inside],
                    "lon": lon[inside],
                    "lat": lat[inside],
                }

                # predictors (aligned) -> subset by inside mask
                for var, ds in open_year.items():
                    arr = read_aligned_window(ds, ref, win, layer_name=var, is_categorical=(var in CATEGORICAL))
                    data[var] = arr.ravel()[inside]

                soil_arr = read_aligned_window(soil_ds, ref, win, layer_name="SOIL", is_categorical=True)
                data["SOIL"] = soil_arr.ravel()[inside]

                chunks.append(pd.DataFrame(data))

            if not chunks:
                print("No inside pixels found for year:", year)
                continue

            df_y = pd.concat(chunks, ignore_index=True)

            # (Optional) drop rows where ALL predictors are NaN (rare after inside mask)
            predictor_cols = list(year_files.keys()) + ["SOIL"]
            df_y = df_y.dropna(subset=predictor_cols, how="all")

            df_y.to_csv(OUT_CSV, index=False, mode="w" if first_write else "a", header=first_write)
            first_write = False

            print("Appended inside rows:", len(df_y))

        finally:
            for ds in open_year.values():
                ds.close()

soil_ds.close()

print("\nDONE.")
print("Saved (inside polygon only):", OUT_CSV)

