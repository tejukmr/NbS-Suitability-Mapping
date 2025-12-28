# Environmental programming using Python

# Assignment topic: Suitability Mapping of Nature-Based Solutions Locations to Tackle Hydroclimatic Extremes and Water
# Quality Degradation Using Machine Learning 

# Group-4:
# Elias Zgheib
# Ndra Malky
# Rashmi Krishnamurthy
# Teju Kumar Nagaraju


# This script utilizes the previously extracted pixel values and map the pixel values of Managed Aquifer Recharge (MAR)
# points. It also adds the suitability column to the dataframe, existing MAR pixels is given 1 and non MAR pixels is given as 0
# If more than one MAR points are located in the pixel, this scripts automatically selects one MAR point in the pixel

import os
import pandas as pd
import geopandas as gpd
import rasterio


# User Inputs

# Existing pixel table (from Task 2A_Pixel_Values)
PIXEL_CSV = r"PixelDataFrames\pixels_2014_2024_all_inside.csv"

# Managed Aquifer Recharge (MAR) shapefile 
MAR_SHP   = r"MAR_EU.shp"            # Already clipped to study area
MAR_FIELD = "main_mar_t"             # field containing MAR type/category in attribute table

# Reference raster for point → pixel_id conversion
REF_RASTER = r"AET_Clipped\AET_2014_clipped.tif"

# Outputs
OUT_MAR_SAMPLES = r"PixelDataFrames\mar_samples_pixels.csv"   # You may change the path to save output
OUT_BINARY_CSV  = r"PixelDataFrames\mar_binary_dataset_all_years.csv"  #You may change the path to save output

# User inputs ends here

# If MAR locations are static across all years
STATIC_MAR = True
MAR_YEAR = 2014   # used only if STATIC_MAR = False


# Helpers

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def norm(s):
    """Normalize category string (case-insensitive, trim spaces)."""
    return " ".join(str(s).strip().lower().split())

def mode_category(s: pd.Series):
    """Stable mode (alphabetical tie-break)."""
    vc = s.value_counts()
    top = vc[vc == vc.max()].index.tolist()
    return sorted(top)[0]


# STEP 2A: SAMPLE MAR SHAPEFILE → PIXELS

def sample_mar_points_to_pixels():

    print("\n=== TASK 2A: MAR SAMPLING ===")

    ensure_dir(OUT_MAR_SAMPLES)

    if not os.path.exists(PIXEL_CSV):
        raise FileNotFoundError(f"Pixel CSV not found: {PIXEL_CSV}")
    if not os.path.exists(MAR_SHP):
        raise FileNotFoundError(f"MAR shapefile not found: {MAR_SHP}")

    df_pixels = pd.read_csv(PIXEL_CSV)
    valid_pixels = set(df_pixels["pixel_id"].unique())

    gmar = gpd.read_file(MAR_SHP)

    if MAR_FIELD not in gmar.columns:
        raise ValueError(f"Field '{MAR_FIELD}' not found in MAR shapefile")

    print("Total MAR points:", len(gmar))

    # ---- Normalize MAR categories ----
    CATEGORIES_STD = [
        "In-Channel Modification",
        "Induced Bank Filtration",
        "Rainwater and Run-off Harvesting",
        "Spreading Methods",
        "Well, Shaft and Borehole Recharge",
    ]

    norm_to_std = {norm(c): c for c in CATEGORIES_STD}
    cat_to_code = {c: i for i, c in enumerate(CATEGORIES_STD)}

    gmar = gmar.dropna(subset=[MAR_FIELD]).copy()
    gmar["_cat_norm"] = gmar[MAR_FIELD].astype(str).apply(norm)

    gmar = gmar[gmar["_cat_norm"].isin(norm_to_std)].copy()
    gmar["MAR_label"] = gmar["_cat_norm"].map(norm_to_std)
    gmar.drop(columns="_cat_norm", inplace=True)

    print("After category filtering:", len(gmar))

    # ---- Convert MAR points to pixel_id ----
    with rasterio.open(REF_RASTER) as ref:
        if gmar.crs != ref.crs:
            gmar = gmar.to_crs(ref.crs)

        xs = gmar.geometry.x.to_numpy()
        ys = gmar.geometry.y.to_numpy()
        rows, cols = rasterio.transform.rowcol(ref.transform, xs, ys)
        W = ref.width

    gmar["row"] = rows
    gmar["col"] = cols
    gmar["pixel_id"] = gmar["row"].astype("int64") * W + gmar["col"].astype("int64")

    # ---- Guard: keep only pixels inside raster domain ----
    before = len(gmar)
    gmar = gmar[gmar["pixel_id"].isin(valid_pixels)].copy()
    print(f"After pixel guard: {len(gmar)} (removed {before - len(gmar)})")

    # ---- One MAR label per pixel (mode) ----
    gmar_mode = (
        gmar.groupby("pixel_id")["MAR_label"]
            .apply(mode_category)
            .reset_index()
    )
    gmar_mode["MAR_code"] = gmar_mode["MAR_label"].map(cat_to_code).astype(int)

    # ---- Merge with pixel table ----
    if STATIC_MAR:
        samples = df_pixels.merge(gmar_mode, on="pixel_id", how="inner")
    else:
        df_year = df_pixels[df_pixels["year"] == MAR_YEAR].copy()
        samples = df_year.merge(gmar_mode, on="pixel_id", how="inner")

    samples["pixel_year_id"] = (
        samples["pixel_id"].astype(str) + "_" + samples["year"].astype(str)
    )

    samples.to_csv(OUT_MAR_SAMPLES, index=False)

    print("Unique MAR pixels:", len(gmar_mode))
    print("MAR samples saved:", OUT_MAR_SAMPLES)


# STEP 2B: CREATE BINARY MAR DATASET (ALL YEARS)


def create_binary_dataset():

    print("\n=== TASK 2B: BINARY DATASET ===")

    ensure_dir(OUT_BINARY_CSV)

    df_all = pd.read_csv(PIXEL_CSV)
    df_mar = pd.read_csv(OUT_MAR_SAMPLES)

    print("Total pixel records:", len(df_all))
    print("Total MAR sample records:", len(df_mar))

    # Default: unsuitable
    df_all["MAR_suitable"] = 0

    mar_pixels = set(df_mar["pixel_id"].unique())
    df_all.loc[df_all["pixel_id"].isin(mar_pixels), "MAR_suitable"] = 1

    # Auto-detect feature columns
    EXCLUDE = {
        "year", "pixel_id", "row", "col", "lon", "lat",
        "MAR_label", "MAR_code", "pixel_year_id", "MAR_suitable"
    }

    feature_cols = [c for c in df_all.columns if c not in EXCLUDE]

    final_cols = ["year", "pixel_id", "lon", "lat"] + feature_cols + ["MAR_suitable"]
    df_final = df_all[final_cols].dropna(subset=feature_cols, how="any")

    df_final.to_csv(OUT_BINARY_CSV, index=False)

    print("Binary dataset saved:", OUT_BINARY_CSV)
    print("Class distribution:")
    print(df_final["MAR_suitable"].value_counts())


# MAIN

if __name__ == "__main__":
    sample_mar_points_to_pixels()
    create_binary_dataset()

