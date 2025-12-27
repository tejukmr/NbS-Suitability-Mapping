# Environmental programming using Python

# Assignment topic: Suitability Mapping of Nature-Based Solutions Locations to Tackle Hydroclimatic Extremes and Water
# Quality Degradation Using Machine Learning 

# Group-4:
# Elias Zgheib
# Ndra Malky
# Rashmi Krishnamurthy
# Teju Kumar Nagaraju


# This script utilizes the MAR pixel values csv, selects MAR unsuitable pixels at 5 times more than MAR pixels for training
# and applies the learning to other pixels to identify the suitable points. GRADIENT BOOST Algorithm is used, 70% points is used for training and 30% is used for testing
# Script also generated probability and suitability maps for MAR. Suitability being defined as Low, Medium and High
# Script also exports MAR suitable points that were identified suitable for more than 8 and above years 
# Script also provides Accuracy, precision, Recall and F1 score in CSV
# Script Predicts probability for ALL pixels (all years)
# Script Reclassifies probability to Low/Medium/High
# Exports:
#     * metrics CSV
#     * feature importance CSV
#     * prediction table CSV
#     * common new sites CSV
#     * common new sites SHP (EPSG:4326)
#     * yearly probability + Suitability Low-1, Middle-2, High-3 (LMH) class rasters


import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import rasterio

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

from sklearn.ensemble import HistGradientBoostingClassifier


# -------------------- USER INPUTS --------------------
DATASET = r"E:\VUB\Final\PixelDataFrames\mar_binary_dataset_all_years.csv"
REF_RASTER = r"E:\VUB\Final\AET_Clipped\AET_2014_clipped.tif"

OUT_DIR = r"E:\VUB\Final\PixelDataFrames\GB_MAR_Outputs_COMMON"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_RASTER_DIR = os.path.join(OUT_DIR, "rasters")
os.makedirs(OUT_RASTER_DIR, exist_ok=True)

# Features and target
FEATURES = ["AET", "LULC", "P", "RZSM", "TEMP", "SOIL"]
TARGET = "MAR_suitable"
GROUP_COL = "pixel_id"

# Train/test split
TRAIN_FRAC = 0.70
RANDOM_STATE = 42

# Sample negatives per year (unsuitable pixels): NEG_POS_RATIO * positives
NEG_POS_RATIO = 5

# HGB hyperparameters (reasonable defaults; tune later)
MAX_ITER = 400           # boosting rounds
LEARNING_RATE = 0.05
MAX_LEAF_NODES = 31
MAX_DEPTH = None         # or an int like 6
MIN_SAMPLES_LEAF = 20
L2_REG = 0.0

# Classification thresholds
PROB_THRESHOLD = 0.50
LOW_TH = 0.33
HIGH_TH = 0.66

# Persistence threshold for "common new sites" across years
PERSIST_THRESHOLD = 0.70

# Outputs
OUT_METRICS    = os.path.join(OUT_DIR, "hgb_metrics.csv")
OUT_FEATIMP    = os.path.join(OUT_DIR, "hgb_feature_importance_perm.csv")
OUT_PRED_TABLE = os.path.join(OUT_DIR, "hgb_predictions_all_records.csv")
OUT_COMMON_CSV = os.path.join(OUT_DIR, "hgb_common_new_suitable_pixels.csv")
OUT_COMMON_SHP = os.path.join(OUT_DIR, "hgb_common_new_suitable_pixels.shp")
# -----------------------------------------------------

def export_points_shp(df_in, out_shp):
    """Export lon/lat points to shapefile (EPSG:4326) with attributes."""
    if df_in.empty:
        print(f"WARNING: Empty output; not writing: {out_shp}")
        return
    gdf = gpd.GeoDataFrame(
        df_in.copy(),
        geometry=[Point(xy) for xy in zip(df_in["lon"], df_in["lat"])],
        crs="EPSG:4326"
    )
    gdf.to_file(out_shp)
    print("Saved:", out_shp)

def prob_to_lmh_cols(prob_series, low=0.33, high=0.66):
    """Vectorized LMH class code + label from probability."""
    codes = np.full(prob_series.shape, np.nan, dtype="float32")
    labels = np.full(prob_series.shape, None, dtype=object)

    p = prob_series.to_numpy()

    m0 = np.isnan(p)
    m1 = (~m0) & (p < low)
    m2 = (~m0) & (p >= low) & (p < high)
    m3 = (~m0) & (p >= high)

    codes[m1] = 1; labels[m1] = "Low"
    codes[m2] = 2; labels[m2] = "Medium"
    codes[m3] = 3; labels[m3] = "High"

    return codes, labels

# -----------------------------------------------------
# Permutation Importance for HGB (works for any model)
# -----------------------------------------------------
def permutation_importance_auc(model, X_val, y_val, random_state=42, n_repeats=3):
    """
    Simple permutation importance based on drop in ROC-AUC.
    Returns dataframe with mean_importance over repeats.
    """
    rng = np.random.RandomState(random_state)
    base_prob = model.predict_proba(X_val)[:, 1]
    base_auc = roc_auc_score(y_val, base_prob)

    importances = {c: [] for c in X_val.columns}

    Xp = X_val.copy()
    for _ in range(n_repeats):
        for col in X_val.columns:
            saved = Xp[col].to_numpy().copy()
            rng.shuffle(Xp[col].values)  # permute in-place
            prob = model.predict_proba(Xp)[:, 1]
            auc = roc_auc_score(y_val, prob)
            importances[col].append(base_auc - auc)
            Xp[col] = saved  # restore

    rows = []
    for col, vals in importances.items():
        rows.append({"feature": col,
                     "importance_perm_auc_mean": float(np.mean(vals)),
                     "importance_perm_auc_std": float(np.std(vals))})
    return pd.DataFrame(rows).sort_values("importance_perm_auc_mean", ascending=False)

# ============================================================
# 1) LOAD & CLEAN DATA
# ============================================================
df = pd.read_csv(DATASET)

required = set(["year", "pixel_id", "lon", "lat"] + FEATURES + [TARGET])
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in dataset: {missing}")

df = df.dropna(subset=FEATURES + [TARGET, "year", "pixel_id", "lon", "lat"]).copy()
df["year"] = df["year"].astype(int)
df[TARGET] = df[TARGET].astype(int)

years = sorted(df["year"].unique())
n_years = len(years)

print("Years:", years, "| N years:", n_years)
print("All records:", len(df))
print("Overall class counts:\n", df[TARGET].value_counts())

# ============================================================
# 2) TRAINING TABLE WITH RANDOM NEGATIVE SAMPLING (Task 2)
# ============================================================
train_parts = []
for y in years:
    dyy = df[df["year"] == y].copy()
    pos = dyy[dyy[TARGET] == 1]
    neg = dyy[dyy[TARGET] == 0]

    n_pos = len(pos)
    if n_pos == 0:
        continue

    n_neg_need = min(len(neg), NEG_POS_RATIO * n_pos)
    if n_neg_need > 0:
        neg_sample = neg.sample(n=n_neg_need, random_state=RANDOM_STATE)
        train_parts.append(pos)
        train_parts.append(neg_sample)
    else:
        train_parts.append(pos)

train_df = pd.concat(train_parts, ignore_index=True)

print("\nTraining dataset after negative sampling:")
print("Records:", len(train_df))
print("Class counts:\n", train_df[TARGET].value_counts())

# ============================================================
# 3) TRAIN/TEST SPLIT (70/30), GROUPED BY PIXEL_ID (Task 3)
# ============================================================
X = train_df[FEATURES]
y = train_df[TARGET]
groups = train_df[GROUP_COL]

gss = GroupShuffleSplit(n_splits=1, train_size=TRAIN_FRAC, random_state=RANDOM_STATE)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

# ============================================================
# 4) TRAIN HGB + EVALUATE (Task 3)
# ============================================================
# NOTE: HistGradientBoostingClassifier supports class_weight from sklearn >= 1.4-ish.
# If your sklearn is older and errors on class_weight, remove it and rely on sampling.
hgb = HistGradientBoostingClassifier(
    loss="log_loss",
    learning_rate=LEARNING_RATE,
    max_iter=MAX_ITER,
    max_leaf_nodes=MAX_LEAF_NODES,
    max_depth=MAX_DEPTH,
    min_samples_leaf=MIN_SAMPLES_LEAF,
    l2_regularization=L2_REG,
    random_state=RANDOM_STATE,
    # comment this line if your sklearn doesn't support it:
    class_weight="balanced"
)

hgb.fit(X_train, y_train)

y_pred = hgb.predict(X_test)
y_prob = hgb.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
roc = roc_auc_score(y_test, y_prob)

print("\n=== TEST METRICS ===")
print(f"Accuracy   : {acc:.3f}")
print(f"Precision  : {prec:.3f}")
print(f"Recall     : {rec:.3f}")
print(f"F1-score   : {f1:.3f}")
print(f"ROC-AUC    : {roc:.3f}")
print("\nConfusion Matrix [ [TN FP] [FN TP] ]:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(
    y_test, y_pred,
    target_names=["Unsuitable(0)", "Suitable(1)"],
    digits=3,
    zero_division=0
))

pd.DataFrame([{
    "Accuracy": acc,
    "Precision": prec,
    "Recall": rec,
    "F1_score": f1,
    "ROC_AUC": roc,
    "Train_frac": TRAIN_FRAC,
    "NEG_POS_RATIO": NEG_POS_RATIO,
    "model": "HistGradientBoostingClassifier",
    "max_iter": MAX_ITER,
    "learning_rate": LEARNING_RATE,
    "max_leaf_nodes": MAX_LEAF_NODES,
    "max_depth": MAX_DEPTH,
    "min_samples_leaf": MIN_SAMPLES_LEAF,
    "l2_regularization": L2_REG,
    "prob_threshold": PROB_THRESHOLD,
    "persist_threshold": PERSIST_THRESHOLD,
    "random_state": RANDOM_STATE
}]).to_csv(OUT_METRICS, index=False)

# Permutation importance (AUC drop)
fi = permutation_importance_auc(hgb, X_test, y_test, random_state=RANDOM_STATE, n_repeats=3)
fi.to_csv(OUT_FEATIMP, index=False)

print("\nSaved metrics:", OUT_METRICS)
print("Saved feature importance:", OUT_FEATIMP)

# ============================================================
# 5) PREDICT FOR ALL PIXELS (ALL YEARS) (Task 4)
# ============================================================
df["MAR_probability"] = hgb.predict_proba(df[FEATURES])[:, 1]
df["MAR_predicted"] = (df["MAR_probability"] >= PROB_THRESHOLD).astype(int)

# Reclassify to LMH
df["suit_class_code"], df["suit_class"] = prob_to_lmh_cols(df["MAR_probability"], LOW_TH, HIGH_TH)

df.to_csv(OUT_PRED_TABLE, index=False)
print("\nSaved prediction table:", OUT_PRED_TABLE)

# ============================================================
# 6) COMMON/PERSISTENT NEW SUITABLE SITES ONLY (across years)
#    NEW = originally 0 but predicted 1
# ============================================================
new_all = df[(df[TARGET] == 0) & (df["MAR_predicted"] == 1)].copy()

if new_all.empty:
    print("\nWARNING: No new suitable records found. Common output will be empty.")
    pd.DataFrame().to_csv(OUT_COMMON_CSV, index=False)
else:
    common_summary = (
        new_all.groupby("pixel_id")
              .agg(
                  years_new_suitable=("MAR_predicted", "sum"),
                  mean_probability=("MAR_probability", "mean"),
                  lon=("lon", "first"),
                  lat=("lat", "first"),
                  AET=("AET", "mean"),
                  P=("P", "mean"),
                  RZSM=("RZSM", "mean"),
                  TEMP=("TEMP", "mean"),
                  LULC=("LULC", "first"),
                  SOIL=("SOIL", "first"),
              )
              .reset_index()
    )
    common_summary["years_total"] = n_years
    common_summary["new_suitable_ratio"] = common_summary["years_new_suitable"] / n_years

    common = common_summary[common_summary["new_suitable_ratio"] >= PERSIST_THRESHOLD].copy()
    common = common.sort_values(["new_suitable_ratio", "mean_probability"], ascending=False)

    common["suit_class_code"], common["suit_class"] = prob_to_lmh_cols(common["mean_probability"], LOW_TH, HIGH_TH)

    common.to_csv(OUT_COMMON_CSV, index=False)
    print("\nSaved common new suitable CSV:", OUT_COMMON_CSV)
    print("Common new suitable pixels:", len(common))

    export_points_shp(common, OUT_COMMON_SHP)

# ============================================================
# 7) EXPORT YEAR-WISE PROBABILITY + LMH RASTERS (Task 4)
# ============================================================
with rasterio.open(REF_RASTER) as ref:
    profile = ref.profile.copy()
    width, height = ref.width, ref.height

# Derive row/col from pixel_id (works because pixel_id = row*width + col)
df["row"] = (df["pixel_id"].astype("int64") // width).astype("int32")
df["col"] = (df["pixel_id"].astype("int64") % width).astype("int32")

# Probability raster profile (float32)
prob_profile = profile.copy()
prob_profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")

# Class raster profile (uint8: 0 nodata, 1 low, 2 medium, 3 high)
cls_profile = profile.copy()
cls_profile.update(dtype="uint8", count=1, nodata=0, compress="lzw")

for y in years:
    dyy = df[df["year"] == y]

    prob_arr = np.full((height, width), np.nan, dtype="float32")
    cls_arr = np.zeros((height, width), dtype="uint8")  # 0 = nodata

    rr = dyy["row"].to_numpy()
    cc = dyy["col"].to_numpy()
    pp = dyy["MAR_probability"].to_numpy(dtype="float32")

    m = (rr >= 0) & (rr < height) & (cc >= 0) & (cc < width)
    rr, cc, pp = rr[m], cc[m], pp[m]

    prob_arr[rr, cc] = pp

    valid = ~np.isnan(prob_arr)
    cls_arr[valid & (prob_arr < LOW_TH)] = 1
    cls_arr[valid & (prob_arr >= LOW_TH) & (prob_arr < HIGH_TH)] = 2
    cls_arr[valid & (prob_arr >= HIGH_TH)] = 3

    prob_tif = os.path.join(OUT_RASTER_DIR, f"suitability_probability_{y}.tif")
    cls_tif  = os.path.join(OUT_RASTER_DIR, f"suitability_class_LMH_{y}.tif")

    with rasterio.open(prob_tif, "w", **prob_profile) as dst:
        dst.write(prob_arr, 1)

    with rasterio.open(cls_tif, "w", **cls_profile) as dst:
        dst.write(cls_arr, 1)

    print(f"Saved rasters for {y}: {os.path.basename(prob_tif)} , {os.path.basename(cls_tif)}")

print("\nDONE. All outputs saved in:", OUT_DIR)
