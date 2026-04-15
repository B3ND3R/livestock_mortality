"""
inference_ward_feature_pipeline.py
===================================
Ward-level satellite feature extraction and engineering for inference.

Produces pre-aggregated ward+season parquets that feed directly into
inference_preprocess.py (step 1 of the inference pipeline).

How it differs from marsabit_feature_pipeline.py (training)
------------------------------------------------------------
Training:
  1. Extract 20km-window means per household GPS coordinate → monthly HH table
  2. Engineer features (indices, lags, drought composites, fire)
  3. Aggregate engineered features to ward+season by averaging across HHs

Inference (this file):
  1. Sample N points within each ward polygon on a regular grid
  2. Extract 20km-window means at each sample point → monthly per-point table
  3. Apply identical feature engineering tiers to each point's time series
  4. Average across sample points within each ward → monthly ward table
  5. Aggregate to biannual / quadseasonal / monthly using the same season
     assignment logic as the training script

Spatial sampling rationale
---------------------------
The centroid of a large or irregular ward is a poor representative point —
it may fall in a non-pastoral zone, near the boundary, or outside the polygon
entirely. Sampling N points on a regular grid within the ward boundary and
averaging their 20km-window values more closely approximates the spatial
mean that training computed by averaging over household GPS points within
each ward.

Longer-term improvement (pixel-in-polygon)
-------------------------------------------
For the most accurate ward representation, aggregate all pixels whose centres
fall within the ward boundary directly (geopandas spatial join on the pixel
lat/lon columns). This eliminates both the window radius and the grid
approximation. See README.md for a full description of this approach.

Outputs (S3)
------------
  <OUTPUT_PREFIX>/ward_features_biannual.parquet
  <OUTPUT_PREFIX>/ward_features_quadseasonal.parquet
  <OUTPUT_PREFIX>/ward_features_monthly.parquet

Columns per output
------------------
  ward_name, season, season_year, <feature_cols...>    (biannual / quadseasonal)
  ward_name, year, month, <feature_cols...>             (monthly)

Usage
-----
  python inference_ward_feature_pipeline.py                              # all schemes
  python inference_ward_feature_pipeline.py --scheme biannual            # single scheme
  python inference_ward_feature_pipeline.py --time-start 2020-01 --time-end 2024-12
  python inference_ward_feature_pipeline.py --n-sample-points 25        # denser grid
"""

import argparse
import io
import gc
import logging
import os
import warnings
from typing import Optional

import boto3
import numpy as np
import pandas as pd
from pyproj import Transformer

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── S3 config ─────────────────────────────────────────────────────────────────
SM_BUCKET  = "amazon-sagemaker-575108933641-us-east-1-c422b90ce861"
SM_BASE    = "dzd-ayr06tncl712p3/5t7l23o0xvt99j/dev/data/training"
PC_PREFIX  = f"{SM_BASE}/planetary_computer"
SHARED_BASE = "dzd-ayr06tncl712p3/5t7l23o0xvt99j/shared"

WARD_BOUNDARIES_KEY = f"{SHARED_BASE}/geoBoundaries-KEN-ADM3.geojson"

# Default temporal range for inference — override via CLI args
DEFAULT_TIME_START = "2020-01"
DEFAULT_TIME_END   = "2024-12"

# Climatology years — same as training for VCI/TCI/VHI calculation
CLIM_YEARS = list(range(2008, 2021))

# Window radius around each sample point (same as training)
WINDOW_KM = 20

# Default number of sample points per ward (grid of ~3x3).
# Increase for larger or more heterogeneous wards at the cost of runtime.
N_SAMPLE_POINTS = 9

# ── Collection registry (identical to marsabit_feature_pipeline.py) ───────────
# (s3_stem, crs, resolution_m, collection_type)
COLLECTIONS = {
    "ndvi_250m":       ("ndvi_250m",       "WGS84",   250, "temporal"),
    "evi_250m":        ("evi_250m",        "WGS84",   250, "temporal"),
    "lai":             ("lai",             "WGS84",   500, "temporal"),
    "fpar":            ("fpar",            "WGS84",   500, "temporal"),
    "lst_day":         ("lst_day",         "WGS84",  1000, "temporal"),
    "lst_night":       ("lst_night",       "WGS84",  1000, "temporal"),
    "et":              ("et",              "WGS84",   500, "temporal"),
    "pet":             ("pet",             "WGS84",   500, "temporal"),
    "sr_red":          ("sr_red",          "WGS84",   500, "temporal"),
    "sr_nir":          ("sr_nir",          "WGS84",   500, "temporal"),
    "sr_swir1":        ("sr_swir1",        "WGS84",   500, "temporal"),
    "sr_swir2":        ("sr_swir2",        "WGS84",   500, "temporal"),
    "fire_mask":       ("fire_mask",       "WGS84",  1000, "temporal"),
    "fire_frp_max":    ("fire_frp_max",    "WGS84",  1000, "temporal"),
    "fire_frp_sum":    ("fire_frp_sum",    "WGS84",  1000, "temporal"),
    "s1_vv":           ("s1_vv",           "UTM37N",  100, "temporal"),
    "s1_vh":           ("s1_vh",           "UTM37N",  100, "temporal"),
    "s2_red":          ("s2_red",          "UTM37N",  100, "temporal"),
    "s2_nir":          ("s2_nir",          "UTM37N",  100, "temporal"),
    "s2_swir1":        ("s2_swir1",        "UTM37N",  100, "temporal"),
    "gpp":             ("gpp",             "WGS84",   500, "temporal"),
    "chirps":          ("chirps",          "WGS84",  5000, "temporal"),
    "soil_moisture":   ("soil_moisture",   "WGS84",  9000, "temporal"),
    "dem":             ("dem",             "WGS84",    30, "static"),
    "jrc_occurrence":  ("jrc_occurrence",  "WGS84",    30, "static"),
    "jrc_seasonality": ("jrc_seasonality", "WGS84",    30, "static"),
    "worldcover":      ("worldcover",      "WGS84",    10, "worldcover"),
}

WORLDCOVER_CLASSES = {
    10: "trees",  20: "shrubland", 30: "grassland", 40: "cropland",
    50: "builtup", 60: "bare",     80: "water",      90: "wetland",
}

LARGE_COLLECTIONS = {"s1_vv", "s1_vh", "ndvi_250m", "evi_250m"}
META_COLS = ["lat", "lon", "variable", "collection"]

# Join keys for the monthly ward table (analogous to training JOIN_KEYS)
JOIN_KEYS = ["ward_name", "ward_lon", "ward_lat", "year", "month"]


s3 = boto3.client("s3")
_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32637", always_xy=True)


# ── S3 helpers ────────────────────────────────────────────────────────────────

def s3_load_parquet(key: str) -> pd.DataFrame:
    log.info("  Loading s3://%s/%s ...", SM_BUCKET, key)
    obj = s3.get_object(Bucket=SM_BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def s3_upload_parquet(df: pd.DataFrame, key: str) -> None:
    tmp = f"/tmp/{key.split('/')[-1]}"
    df.to_parquet(tmp, index=False, engine="pyarrow", compression="snappy")
    s3.upload_file(tmp, SM_BUCKET, key)
    log.info("  Uploaded -> s3://%s/%s  (%.1f MB)", SM_BUCKET, key,
             os.path.getsize(tmp) / 1e6)
    os.remove(tmp)


def s3_exists(key: str) -> bool:
    try:
        s3.head_object(Bucket=SM_BUCKET, Key=key)
        return True
    except Exception:
        return False


def pixel_key(stem: str) -> str:
    return f"{PC_PREFIX}/{stem}.parquet"


# ── Coordinate helpers ────────────────────────────────────────────────────────

def ward_coords(wards: pd.DataFrame, crs: str):
    """Return (lons, lats) arrays for ward centroids, optionally projected."""
    lons = wards["ward_lon"].values.astype("float64")
    lats = wards["ward_lat"].values.astype("float64")
    if crs == "UTM37N":
        lons, lats = _to_utm.transform(lons, lats)
    return lons, lats


def snap_to_grid(pixel_lons, pixel_lats, hh_lons, hh_lats):
    ctr_lon = np.searchsorted(pixel_lons, hh_lons).clip(0, len(pixel_lons) - 1)
    ctr_lat = np.searchsorted(-pixel_lats, -hh_lats).clip(0, len(pixel_lats) - 1)
    return ctr_lon, ctr_lat


def half_win(resolution_m: int) -> int:
    return int(np.ceil(WINDOW_KM * 1000 / resolution_m))


def window_mean_2d(data_2d, ctr_lon_idxs, ctr_lat_idxs, n_lat, n_lon, hw):
    n = len(ctr_lon_idxs)
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        lat_lo = max(ctr_lat_idxs[i] - hw, 0)
        lat_hi = min(ctr_lat_idxs[i] + hw + 1, n_lat)
        lon_lo = max(ctr_lon_idxs[i] - hw, 0)
        lon_hi = min(ctr_lon_idxs[i] + hw + 1, n_lon)
        window = data_2d[lat_lo:lat_hi, lon_lo:lon_hi]
        if window.size > 0:
            out[i] = np.nanmean(window)
    return out


# ── PART 0: Sample points within ward polygons ──────────���─────────────────────

def sample_ward_points(
    n_points: int = N_SAMPLE_POINTS,
    bbox: Optional[tuple] = None,
) -> pd.DataFrame:
    """
    Load Kenya ADMIN3 ward boundaries from S3 and return a DataFrame of
    sample points within each ward polygon.

    For each ward, a regular grid of ceil(sqrt(n_points)) × ceil(sqrt(n_points))
    candidate points is laid over the bounding box and all candidates that fall
    inside the polygon are kept. This gives between 1 and n_points points per
    ward depending on polygon shape. Wards where no grid point falls inside
    (very small or narrow polygons) fall back to the centroid.

    Averaging 20km-window means across these points at extraction time is
    equivalent to what training did by averaging over household GPS points
    within each ward, and generalises to wards with no survey history.

    Parameters
    ----------
    n_points : int
        Target number of sample points per ward. The actual count per ward
        may be lower for small or irregular polygons.
    bbox : (min_lon, min_lat, max_lon, max_lat), optional
        Clip to a bounding box (e.g. Marsabit) before sampling.

    Returns
    -------
    pd.DataFrame with columns: ward_name, ward_lon, ward_lat
        Multiple rows per ward_name when n_points > 1.
    """
    import math
    import tempfile
    import geopandas as gpd
    from shapely.geometry import Point

    log.info("Loading ward boundaries from s3://%s/%s ...", SM_BUCKET, WARD_BOUNDARIES_KEY)
    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as tmp:
        s3.download_file(SM_BUCKET, WARD_BOUNDARIES_KEY, tmp.name)
        local_path = tmp.name

    wards = gpd.read_file(local_path).to_crs("EPSG:4326")
    os.remove(local_path)

    if bbox is not None:
        wards = wards.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].reset_index(drop=True)

    name_col = "shapeName" if "shapeName" in wards.columns else "ADM3_EN"
    wards["ward_name"] = wards[name_col].astype(str)

    grid_side = math.ceil(math.sqrt(n_points))
    records = []

    for _, row in wards.iterrows():
        poly = row.geometry
        name = row["ward_name"]

        minx, miny, maxx, maxy = poly.bounds
        # Build a grid_side × grid_side grid over the bounding box
        lons = np.linspace(minx, maxx, grid_side)
        lats = np.linspace(miny, maxy, grid_side)
        candidates = [
            (lon, lat)
            for lon in lons
            for lat in lats
            if poly.contains(Point(lon, lat))
        ]

        if not candidates:
            # Fallback: use the centroid (always inside for convex polygons;
            # representative for concave ones)
            c = poly.centroid
            candidates = [(c.x, c.y)]

        for lon, lat in candidates:
            records.append({"ward_name": name, "ward_lon": lon, "ward_lat": lat})

    result = pd.DataFrame(records)
    n_wards  = result["ward_name"].nunique()
    avg_pts  = len(result) / n_wards if n_wards else 0
    log.info("  %d wards, %d total sample points (avg %.1f per ward)",
             n_wards, len(result), avg_pts)
    return result


# ── PART 1: Extraction ────────────────────────────────────────────────────────

def extract_temporal(
    var_name: str,
    stem: str,
    crs: str,
    resolution_m: int,
    wards: pd.DataFrame,
    time_start: str,
    time_end: str,
) -> Optional[pd.DataFrame]:
    """
    Extract monthly 20km-window mean from pixel parquet for each ward centroid.
    Returns DataFrame with columns: JOIN_KEYS + [var_name]
    """
    log.info("[temporal] %s  crs=%s  res=%dm", var_name, crs, resolution_m)
    df = s3_load_parquet(pixel_key(stem))

    if var_name == "soil_moisture":
        return _extract_soil_moisture(df, wards, time_end)

    month_cols = sorted([
        c for c in df.columns
        if c not in META_COLS and len(c) == 7 and time_start <= c <= time_end
    ])
    if not month_cols:
        log.warning("  No month columns in [%s, %s] — skipping %s", time_start, time_end, var_name)
        return None

    log.info("  %d pixels x %d months", len(df), len(month_cols))
    pixel_lons = np.sort(df["lon"].unique()).astype("float64")
    pixel_lats = np.sort(df["lat"].unique())[::-1].astype("float64")
    n_lon, n_lat = len(pixel_lons), len(pixel_lats)
    df = df.sort_values(["lat", "lon"], ascending=[False, True]).reset_index(drop=True)

    w_lons, w_lats = ward_coords(wards, crs)
    ctr_lon_idxs, ctr_lat_idxs = snap_to_grid(pixel_lons, pixel_lats, w_lons, w_lats)
    hw    = half_win(resolution_m)
    n_w   = len(wards)
    is_large = var_name in LARGE_COLLECTIONS
    log.info("  HALF_WIN=%d  wards=%d", hw, n_w)

    records = []
    for idx, col in enumerate(month_cols):
        data_2d = df[col].values.reshape(n_lat, n_lon).astype("float32")
        values  = window_mean_2d(data_2d, ctr_lon_idxs, ctr_lat_idxs, n_lat, n_lon, hw)
        t = pd.to_datetime(f"{col}-01")
        for i in range(n_w):
            records.append((
                wards["ward_name"].iloc[i],
                float(wards["ward_lon"].iloc[i]),
                float(wards["ward_lat"].iloc[i]),
                int(t.year), int(t.month),
                float(values[i]),
            ))
        if is_large:
            del data_2d; gc.collect()
        if (idx + 1) % 24 == 0:
            log.info("  ... %d / %d months", idx + 1, len(month_cols))

    del df; gc.collect()
    result = pd.DataFrame(records, columns=JOIN_KEYS + [var_name])
    result[var_name] = result[var_name].astype("float32")
    result = result.sort_values(JOIN_KEYS).reset_index(drop=True)
    log.info("  Shape=%s  Null=%.1f%%  Mean=%.4f",
             result.shape, result[var_name].isna().mean() * 100,
             result[var_name].dropna().mean())
    return result


def _extract_soil_moisture(
    df: pd.DataFrame,
    wards: pd.DataFrame,
    time_end: str,
) -> pd.DataFrame:
    """Soil moisture: 4 layers (swvl1-4) with prefixed columns."""
    layers = ["swvl1", "swvl2", "swvl3", "swvl4"]
    hw     = half_win(9000)
    w_lons, w_lats = ward_coords(wards, "WGS84")
    n_w    = len(wards)
    all_dfs = []

    for layer in layers:
        sub = df[df["variable"] == layer].copy()
        sub = sub.rename(columns={c: c.replace(f"{layer}_", "")
                                   for c in sub.columns if c.startswith(f"{layer}_")})
        sub = sub.drop(columns=["variable", "collection"], errors="ignore")

        month_cols = sorted([
            c for c in sub.columns
            if c not in ["lat", "lon"] and len(c) == 7 and c <= time_end
        ])
        if not month_cols:
            continue

        pixel_lons = np.sort(sub["lon"].unique()).astype("float64")
        pixel_lats = np.sort(sub["lat"].unique())[::-1].astype("float64")
        n_lon, n_lat = len(pixel_lons), len(pixel_lats)
        sub = sub.sort_values(["lat", "lon"], ascending=[False, True]).reset_index(drop=True)
        ctr_lon_idxs, ctr_lat_idxs = snap_to_grid(pixel_lons, pixel_lats, w_lons, w_lats)

        records = []
        for col in month_cols:
            data_2d = sub[col].values.reshape(n_lat, n_lon).astype("float32")
            values  = window_mean_2d(data_2d, ctr_lon_idxs, ctr_lat_idxs, n_lat, n_lon, hw)
            t = pd.to_datetime(f"{col}-01")
            for i in range(n_w):
                records.append((
                    wards["ward_name"].iloc[i],
                    float(wards["ward_lon"].iloc[i]),
                    float(wards["ward_lat"].iloc[i]),
                    int(t.year), int(t.month),
                    float(values[i]),
                ))

        layer_df = pd.DataFrame(records, columns=JOIN_KEYS + [layer])
        all_dfs.append(layer_df)
        log.info("  %s: mean=%.4f  null=%.1f%%",
                 layer, layer_df[layer].mean(), layer_df[layer].isna().mean() * 100)

    result = all_dfs[0]
    for ldf in all_dfs[1:]:
        result = result.merge(ldf, on=JOIN_KEYS, how="outer")
    return result.sort_values(JOIN_KEYS).reset_index(drop=True)


def extract_static(
    var_name: str,
    stem: str,
    wards: pd.DataFrame,
    time_start: str,
    time_end: str,
) -> pd.DataFrame:
    """Static layer: compute ward centroid value once, broadcast to all months."""
    log.info("[static] %s", var_name)
    df = s3_load_parquet(pixel_key(stem))
    val_candidates = [c for c in df.columns if c not in META_COLS]
    val_col = next((c for c in ["value", var_name] if c in val_candidates),
                   val_candidates[0])
    pixel_lons = df["lon"].values.astype("float64")
    pixel_lats = df["lat"].values.astype("float64")
    pixel_vals = df[val_col].values.astype("float32")
    del df; gc.collect()

    w_lons, w_lats = ward_coords(wards, "WGS84")
    deg_win = WINDOW_KM / 111.0
    n_w     = len(wards)
    values  = np.full(n_w, np.nan, dtype=np.float32)
    for i in range(n_w):
        mask = (
            (pixel_lons >= w_lons[i] - deg_win) & (pixel_lons <= w_lons[i] + deg_win) &
            (pixel_lats >= w_lats[i] - deg_win) & (pixel_lats <= w_lats[i] + deg_win)
        )
        if mask.sum() > 0:
            values[i] = np.nanmean(pixel_vals[mask])

    del pixel_lons, pixel_lats, pixel_vals; gc.collect()
    log.info("  Null=%.1f%%  Mean=%.4f",
             np.isnan(values).mean() * 100, float(np.nanmean(values)))

    time_df = pd.DataFrame({
        "year":  [t.year  for t in pd.date_range(time_start, time_end, freq="MS")],
        "month": [t.month for t in pd.date_range(time_start, time_end, freq="MS")],
        "_key":  1,
    })
    ward_static = pd.DataFrame({
        "ward_name": wards["ward_name"].values,
        "ward_lon":  wards["ward_lon"].values,
        "ward_lat":  wards["ward_lat"].values,
        var_name:    values,
        "_key":      1,
    })
    result = ward_static.merge(time_df, on="_key").drop(columns="_key")
    return result.sort_values(JOIN_KEYS).reset_index(drop=True)


def extract_worldcover(
    wards: pd.DataFrame,
    time_start: str,
    time_end: str,
) -> pd.DataFrame:
    """WorldCover class fractions within 20km window, broadcast to all months."""
    log.info("[worldcover] Loading...")
    df = s3_load_parquet(pixel_key("worldcover"))
    val_candidates = [c for c in df.columns if c not in META_COLS]
    val_col = next((c for c in ["value", "worldcover"] if c in val_candidates),
                   val_candidates[0])
    pixel_lons = df["lon"].values.astype("float64")
    pixel_lats = df["lat"].values.astype("float64")
    pixel_vals = df[val_col].values
    del df; gc.collect()

    w_lons, w_lats = ward_coords(wards, "WGS84")
    deg_win = WINDOW_KM / 111.0
    n_w     = len(wards)
    ward_wc = pd.DataFrame({
        "ward_name": wards["ward_name"].values,
        "ward_lon":  wards["ward_lon"].values,
        "ward_lat":  wards["ward_lat"].values,
    })

    for code, name in WORLDCOVER_CLASSES.items():
        fractions = np.full(n_w, np.nan, dtype=np.float32)
        for i in range(n_w):
            mask = (
                (pixel_lons >= w_lons[i] - deg_win) & (pixel_lons <= w_lons[i] + deg_win) &
                (pixel_lats >= w_lats[i] - deg_win) & (pixel_lats <= w_lats[i] + deg_win)
            )
            window = pixel_vals[mask]
            if window.size > 0:
                fractions[i] = float((window == code).sum()) / window.size
        ward_wc[f"wc_{name}"] = fractions
        log.info("  wc_%s: mean=%.3f", name, float(np.nanmean(fractions)))

    del pixel_lons, pixel_lats, pixel_vals; gc.collect()

    time_df = pd.DataFrame({
        "year":  [t.year  for t in pd.date_range(time_start, time_end, freq="MS")],
        "month": [t.month for t in pd.date_range(time_start, time_end, freq="MS")],
        "_key":  1,
    })
    ward_wc["_key"] = 1
    result = ward_wc.merge(time_df, on="_key").drop(columns="_key")
    return result.sort_values(JOIN_KEYS).reset_index(drop=True)


def run_extraction(
    wards: pd.DataFrame,
    time_start: str,
    time_end: str,
    output_prefix: str,
    skip_collections: Optional[set] = None,
) -> pd.DataFrame:
    """
    Extract all collections for the given time range and ward centroids.
    Caches per-variable parquets on S3 to allow restartable runs.
    Returns merged monthly ward DataFrame.
    """
    log.info("\n-- PART 1: EXTRACTION --")

    time_index = pd.DataFrame({
        "year":  [t.year  for t in pd.date_range(time_start, time_end, freq="MS")],
        "month": [t.month for t in pd.date_range(time_start, time_end, freq="MS")],
        "_key":  1,
    })
    ward_index = wards[["ward_name", "ward_lon", "ward_lat"]].copy()
    ward_index["_key"] = 1
    canonical = (
        ward_index.merge(time_index, on="_key")
        .drop(columns="_key")
        .astype({"year": int, "month": int})
        .sort_values(JOIN_KEYS)
        .reset_index(drop=True)
    )
    log.info("Canonical index: %d rows (%d wards x %d months)",
             len(canonical), len(wards),
             len(pd.date_range(time_start, time_end, freq="MS")))

    extracted = {}

    for var_name, (stem, crs, res_m, ctype) in COLLECTIONS.items():
        if ctype not in ("temporal", "static"):
            continue
        if skip_collections and var_name in skip_collections:
            log.info("[skip] %s -- excluded via --skip-collections", var_name)
            continue
        ck = f"{output_prefix}/extracted/{var_name}.parquet"
        if s3_exists(ck):
            log.info("[skip] %s -- cache hit", var_name)
            extracted[var_name] = s3_load_parquet(ck)
            continue
        try:
            if ctype == "temporal":
                df_out = extract_temporal(var_name, stem, crs, res_m, wards, time_start, time_end)
            else:
                df_out = extract_static(var_name, stem, wards, time_start, time_end)
            if df_out is not None:
                s3_upload_parquet(df_out, ck)
                extracted[var_name] = df_out
        except Exception as e:
            log.error("FAILED %s: %s", var_name, e, exc_info=True)

    ck = f"{output_prefix}/extracted/worldcover.parquet"
    if skip_collections and "worldcover" in skip_collections:
        log.info("[skip] worldcover -- excluded via --skip-collections")
    elif s3_exists(ck):
        log.info("[skip] worldcover -- cache hit")
        extracted["worldcover"] = s3_load_parquet(ck)
    else:
        try:
            df_out = extract_worldcover(wards, time_start, time_end)
            s3_upload_parquet(df_out, ck)
            extracted["worldcover"] = df_out
        except Exception as e:
            log.error("FAILED worldcover: %s", e, exc_info=True)

    log.info("\nMerging %d collections...", len(extracted))
    merged = canonical.copy()
    for var_name, df in extracted.items():
        feat_cols = [c for c in df.columns if c not in JOIN_KEYS]
        merged = merged.merge(df[JOIN_KEYS + feat_cols], on=JOIN_KEYS, how="left")
        log.info("  + %-20s  shape=%s", var_name, merged.shape)

    return merged.sort_values(JOIN_KEYS).reset_index(drop=True)


# ── PART 2: Feature Engineering (identical tiers to marsabit_feature_pipeline) ─

def compute_dem_roughness(ward_lats, ward_lons):
    log.info("Computing DEM terrain roughness...")
    df = s3_load_parquet(pixel_key("dem"))
    val_candidates = [c for c in df.columns if c not in META_COLS]
    val_col = next((c for c in ["value", "dem"] if c in val_candidates),
                   val_candidates[0])
    pixel_lons = df["lon"].values.astype("float64")
    pixel_lats = df["lat"].values.astype("float64")
    pixel_vals = df[val_col].values.astype("float32")
    del df; gc.collect()

    deg_win     = WINDOW_KM / 111.0
    n           = len(ward_lats)
    dem_std_v   = np.full(n, np.nan, dtype="float32")
    dem_range_v = np.full(n, np.nan, dtype="float32")
    for i in range(n):
        mask  = (
            (pixel_lons >= ward_lons[i] - deg_win) & (pixel_lons <= ward_lons[i] + deg_win) &
            (pixel_lats >= ward_lats[i] - deg_win) & (pixel_lats <= ward_lats[i] + deg_win)
        )
        valid = pixel_vals[mask]; valid = valid[~np.isnan(valid)]
        if len(valid) > 1:
            dem_std_v[i]   = float(np.std(valid))
            dem_range_v[i] = float(np.max(valid) - np.min(valid))
    del pixel_lons, pixel_lats, pixel_vals; gc.collect()
    return dem_std_v, dem_range_v


def tier0_preprocess(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Tier 0: ET/PET forward fill...")
    for col in ["et", "pet"]:
        if col in df.columns:
            df[col] = (
                df.groupby(["ward_name", "ward_lon", "ward_lat", "year"])[col]
                .transform(lambda x: x.ffill().bfill())
            )
            log.info("  %s filled. Null=%.1f%%", col, df[col].isna().mean() * 100)
    return df


def tier1_direct_indices(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Tier 1: Direct indices...")

    # ── Rename chirps → ppt (training convention) ────────────────────────────
    if "chirps" in df.columns and "ppt" not in df.columns:
        df = df.rename(columns={"chirps": "ppt"})
        log.info("  Renamed chirps → ppt")

    # ── Soil moisture composites ─────────────────────────────────────────────
    if all(c in df.columns for c in ["swvl1", "swvl2", "swvl3", "swvl4"]):
        df["soil_composite"] = (
            0.4 * df["swvl1"] + 0.3 * df["swvl2"]
            + 0.2 * df["swvl3"] + 0.1 * df["swvl4"]
        )
        df["soil_shallow_deep"] = (
            df["swvl1"] / df["swvl4"].replace(0, np.nan)
        ).clip(0, 10)
        # Per-ward monthly anomaly
        W_MONTH = ["ward_name", "ward_lon", "ward_lat", "month"]
        clim_mean = df.groupby(W_MONTH)["soil_composite"].transform("mean")
        df["soil_composite_anom"] = df["soil_composite"] - clim_mean
        log.info("  soil_composite, soil_shallow_deep, soil_composite_anom")

    if all(c in df.columns for c in ["sr_nir", "sr_red"]):
        nir, red = df["sr_nir"], df["sr_red"]
        df["evi2"]  = (2.5*(nir-red) / (nir+2.4*red+1)).clip(-1, 1)
        df["savi"]  = (1.5*(nir-red) / (nir+red+0.5)).clip(-1, 1)
        df["msavi"] = ((2*nir+1 - np.sqrt((2*nir+1)**2 - 8*(nir-red))) / 2).clip(-1, 1)
        df["osavi"] = ((nir-red) / (nir+red+0.16)).clip(-1, 1)
        log.info("  EVI2, SAVI, MSAVI, OSAVI")

    if all(c in df.columns for c in ["sr_nir", "sr_swir1"]):
        nir, swir1 = df["sr_nir"], df["sr_swir1"]
        df["ndwi"] = ((nir-swir1) / (nir+swir1+1e-6)).clip(-1, 1)
        df["lswi"] = ((nir-swir1) / (nir+swir1+1e-6)).clip(-1, 1)
        log.info("  NDWI, LSWI")

    if all(c in df.columns for c in ["sr_nir", "sr_swir2"]):
        df["nbr"] = ((df["sr_nir"]-df["sr_swir2"]) /
                     (df["sr_nir"]+df["sr_swir2"]+1e-6)).clip(-1, 1)
        log.info("  NBR")

    if all(c in df.columns for c in ["sr_red", "sr_nir", "sr_swir1"]):
        df["bsi"] = ((df["sr_red"]+df["sr_swir1"]) /
                     (df["sr_nir"]+df["sr_swir1"]+1e-6)).clip(0, 5)
        log.info("  BSI")

    if all(c in df.columns for c in ["sr_swir1", "sr_swir2"]):
        df["swir_ratio"] = (df["sr_swir1"] / (df["sr_swir2"]+1e-6)).clip(0, 10)
        log.info("  SWIR ratio")

    if all(c in df.columns for c in ["s2_nir", "s2_red"]):
        df["s2_ndvi"] = ((df["s2_nir"]-df["s2_red"]) /
                         (df["s2_nir"]+df["s2_red"]+1e-6)).clip(-1, 1)
        log.info("  S2 NDVI")

    if all(c in df.columns for c in ["s2_nir", "s2_swir1"]):
        df["s2_ndwi"] = ((df["s2_nir"]-df["s2_swir1"]) /
                         (df["s2_nir"]+df["s2_swir1"]+1e-6)).clip(-1, 1)
        log.info("  S2 NDWI")

    if all(c in df.columns for c in ["s1_vv", "s1_vh"]):
        vv_db, vh_db = df["s1_vv"], df["s1_vh"]
        df["sar_vv_vh_ratio"] = (vv_db - vh_db).clip(-20, 20)
        vv_lin = 10 ** (vv_db / 10)
        vh_lin = 10 ** (vh_db / 10)
        df["sar_rvi"] = (4*vh_lin / (vv_lin+vh_lin+1e-9)).clip(0, 1)
        log.info("  SAR VV/VH ratio, RVI")

    if all(c in df.columns for c in ["et", "pet"]):
        df["et_deficit"]  = (df["pet"] - df["et"]).clip(lower=0)
        df["et_fraction"] = (df["et"] / df["pet"].replace(0, np.nan)).clip(0, 1)
        log.info("  ET deficit, ET fraction")

    if all(c in df.columns for c in ["lst_day", "lst_night"]):
        df["lst_diurnal_range"] = (df["lst_day"] - df["lst_night"]).clip(lower=0)
        log.info("  LST diurnal range")

    df["is_lrld"] = ((df["month"] >= 3) & (df["month"] <= 9)).astype("float32")
    df["is_mam"]  = df["month"].isin([3, 4, 5]).astype("float32")
    df["is_ond"]  = df["month"].isin([10, 11, 12]).astype("float32")
    log.info("  Season flags")
    return df


def tier2_temporal_windows(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Tier 2: Temporal window features...")
    df  = df.sort_values(JOIN_KEYS)
    grp = df.groupby(["ward_name", "ward_lon", "ward_lat"], sort=False)

    TEMPORAL_VARS = [c for c in [
        "ndvi_250m", "evi_250m", "lst_day", "lst_night",
        "lai", "fpar", "sr_nir", "sr_red",
        "sar_rvi", "sar_vv_vh_ratio", "s1_vv", "s1_vh",
        "et_deficit", "et_fraction", "lst_diurnal_range",
        "ppt", "soil_composite",
    ] if c in df.columns]

    for col in TEMPORAL_VARS:
        for lag in [1, 2, 3]:
            df[f"{col}_lag{lag}"] = grp[col].shift(lag)
        df[f"{col}_diff1"]      = grp[col].transform(lambda x: x.diff(1))
        prev                    = grp[col].shift(1)
        df[f"{col}_ratio1"]     = (df[col] / prev.replace(0, np.nan)).clip(-10, 10)
        df[f"{col}_yoy_diff"]   = grp[col].transform(lambda x: x.diff(12))
        prev12                  = grp[col].shift(12)
        df[f"{col}_yoy_ratio"]  = (df[col] / prev12.replace(0, np.nan)).clip(-10, 10)
        df[f"{col}_roll3_mean"] = grp[col].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        df[f"{col}_roll3_std"]  = grp[col].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).std())
        df[f"{col}_roll3_sum"]  = grp[col].transform(
            lambda x: x.shift(1).rolling(3, min_periods=1).sum())

    log.info("  Applied to %d variables", len(TEMPORAL_VARS))
    return df


def tier3_drought_composites(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Tier 3: Drought composites...")
    clim_mask = df["year"].isin(CLIM_YEARS)
    W_MONTH   = ["ward_name", "ward_lon", "ward_lat", "month"]

    if "ndvi_250m" in df.columns:
        clim = (df.loc[clim_mask].groupby(W_MONTH)["ndvi_250m"]
                .agg(ndvi_clim_mean="mean", ndvi_clim_std="std",
                     ndvi_p02=lambda x: x.quantile(0.02),
                     ndvi_p98=lambda x: x.quantile(0.98))
                .reset_index())
        df = df.merge(clim, on=W_MONTH, how="left")
        df["ndvi_anom"]     = df["ndvi_250m"] - df["ndvi_clim_mean"]
        df["ndvi_anom_std"] = df["ndvi_anom"] / df["ndvi_clim_std"].replace(0, np.nan)
        rng = (df["ndvi_p98"] - df["ndvi_p02"]).replace(0, np.nan)
        df["vci"] = ((df["ndvi_250m"] - df["ndvi_p02"]) / rng * 100).clip(0, 100)
        df = df.drop(columns=["ndvi_clim_mean", "ndvi_clim_std", "ndvi_p02", "ndvi_p98"])
        log.info("  NDVI anomaly, VCI")

    if "lst_day" in df.columns:
        clim = (df.loc[clim_mask].groupby(W_MONTH)["lst_day"]
                .agg(lst_clim_mean="mean",
                     lst_p02=lambda x: x.quantile(0.02),
                     lst_p98=lambda x: x.quantile(0.98))
                .reset_index())
        df = df.merge(clim, on=W_MONTH, how="left")
        df["lst_anom"] = df["lst_day"] - df["lst_clim_mean"]
        rng = (df["lst_p98"] - df["lst_p02"]).replace(0, np.nan)
        df["tci"] = ((df["lst_p98"] - df["lst_day"]) / rng * 100).clip(0, 100)
        df = df.drop(columns=["lst_clim_mean", "lst_p02", "lst_p98"])
        log.info("  LST anomaly, TCI")

    if all(c in df.columns for c in ["vci", "tci"]):
        df["vhi"]             = (0.5*df["vci"] + 0.5*df["tci"]).clip(0, 100)
        df["drought_mild"]    = (df["vhi"] < 40).astype("float32")
        df["drought_severe"]  = (df["vhi"] < 20).astype("float32")
        df["drought_extreme"] = (df["vhi"] < 10).astype("float32")
        log.info("  VHI, drought flags")

    if all(c in df.columns for c in ["ndvi_anom_std", "lst_anom"]):
        df["compound_stress"] = (df["ndvi_anom_std"].clip(upper=0).abs() *
                                 df["lst_anom"].clip(lower=0))
        log.info("  Compound stress")

    df  = df.sort_values(JOIN_KEYS)
    grp = df.groupby(["ward_name", "ward_lon", "ward_lat"], sort=False)
    for col in ["vci", "tci", "vhi"]:
        if col in df.columns:
            for lag in [1, 2, 3]:
                df[f"{col}_lag{lag}"] = grp[col].shift(lag)
            df[f"{col}_roll3_mean"] = grp[col].transform(
                lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    log.info("  VCI/TCI/VHI lags 1-3 and rolling mean")
    return df


def tier4_fire(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Tier 4: Fire features...")
    if "fire_mask" not in df.columns:
        log.warning("  fire_mask not found -- skipping")
        return df

    df  = df.sort_values(JOIN_KEYS)
    grp = df.groupby(["ward_name", "ward_lon", "ward_lat"], sort=False)

    df["fire_detected"]         = (df["fire_mask"] >= 7).astype("float32")
    df["fire_cumulative_count"] = grp["fire_detected"].transform("cumsum")

    def months_since_fire(series):
        counts, c = [], 0
        for v in series.shift(1).fillna(0):
            c = 0 if v >= 7 else c + 1
            counts.append(c)
        return pd.Series(counts, index=series.index)

    df["months_since_fire"] = grp["fire_mask"].transform(months_since_fire)
    if "fire_frp_sum" in df.columns:
        df["fire_frp_cumulative"] = grp["fire_frp_sum"].transform("cumsum")
    df["fire_count_12m"] = grp["fire_detected"].transform(
        lambda x: x.shift(1).rolling(12, min_periods=1).sum())
    log.info("  fire_detected, cumulative_count, months_since_fire, fire_count_12m")
    return df


def tier4_longterm_ndvi(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Tier 4: Long-term NDVI statistics...")
    if "ndvi_250m" not in df.columns:
        return df

    W_KEYS = ["ward_name", "ward_lon", "ward_lat"]
    annual = (df.groupby(W_KEYS + ["year"])["ndvi_250m"].mean()
              .reset_index().rename(columns={"ndvi_250m": "ndvi_annual"}))

    hh_stats = (annual.groupby(W_KEYS)["ndvi_annual"]
                .agg(ndvi_lt_mean="mean", ndvi_lt_std="std",
                     ndvi_lt_p10=lambda x: x.quantile(0.10),
                     ndvi_lt_p25=lambda x: x.quantile(0.25),
                     ndvi_lt_p75=lambda x: x.quantile(0.75),
                     ndvi_lt_p90=lambda x: x.quantile(0.90))
                .reset_index())
    hh_stats["ndvi_lt_cv"] = (hh_stats["ndvi_lt_std"] /
                               hh_stats["ndvi_lt_mean"].replace(0, np.nan))

    p20 = (annual.groupby(W_KEYS)["ndvi_annual"].quantile(0.20)
           .reset_index().rename(columns={"ndvi_annual": "ndvi_p20"}))
    annual = annual.merge(p20, on=W_KEYS, how="left")
    drought_count = (annual
                     .assign(is_drought=(annual["ndvi_annual"] < annual["ndvi_p20"]).astype(int))
                     .groupby(W_KEYS)["is_drought"].sum().reset_index()
                     .rename(columns={"is_drought": "ndvi_drought_year_count"}))
    hh_stats = hh_stats.merge(drought_count, on=W_KEYS, how="left")

    for season, months in [("mam", [3,4,5]), ("ond", [10,11,12]), ("jfas", [1,2,6,7,8,9])]:
        seas = (df[df["month"].isin(months)].groupby(W_KEYS)["ndvi_250m"]
                .mean().reset_index().rename(columns={"ndvi_250m": f"ndvi_mean_{season}"}))
        hh_stats = hh_stats.merge(seas, on=W_KEYS, how="left")

    df = df.merge(hh_stats, on=W_KEYS, how="left")
    log.info("  ndvi_lt_cv, P10/P25/P75/P90, drought_year_count, MAM/OND/JFAS")
    return df


def run_feature_engineering(df: pd.DataFrame, wards: pd.DataFrame) -> pd.DataFrame:
    log.info("\n-- PART 2: FEATURE ENGINEERING --")

    unique_wards = df[["ward_name", "ward_lon", "ward_lat"]].drop_duplicates()
    dem_std_v, dem_range_v = compute_dem_roughness(
        unique_wards["ward_lat"].values, unique_wards["ward_lon"].values)
    ward_topo = pd.DataFrame({
        "ward_name":  unique_wards["ward_name"].values,
        "ward_lon":   unique_wards["ward_lon"].values,
        "ward_lat":   unique_wards["ward_lat"].values,
        "dem_std":    dem_std_v,
        "dem_range":  dem_range_v,
    })
    df = df.merge(ward_topo, on=["ward_name", "ward_lon", "ward_lat"], how="left")
    log.info("DEM roughness added")

    df = tier0_preprocess(df)
    df = tier1_direct_indices(df)
    df = tier2_temporal_windows(df)
    df = tier3_drought_composites(df)
    df = tier4_fire(df)
    df = tier4_longterm_ndvi(df)

    log.info("Adding month dummies (m_2 to m_12)...")
    dummies = pd.get_dummies(df["month"], prefix="m", drop_first=True, dtype="float32")
    df = pd.concat([df, dummies], axis=1)
    return df.sort_values(JOIN_KEYS).reset_index(drop=True)


# ── PART 3: Season Aggregation ────────────────────────────────────────────────

def assign_biannual(month: int) -> str:
    return "LRLD" if 3 <= month <= 9 else "SRSD"


def assign_quadseasonal(month: int) -> str:
    if month in [3, 4, 5, 6]:    return "LRS"
    elif month in [7, 8, 9]:     return "LRS_dry"
    elif month in [10, 11, 12]:  return "SRS"
    else:                        return "SRS_dry"


def get_season_year(month: int, year: int, scheme_season: str) -> int:
    """Jan/Feb belong to the previous season for SRSD and SRS_dry."""
    if scheme_season in ("SRSD", "SRS_dry") and month in [1, 2]:
        return year - 1
    return year


def aggregate_to_seasons(
    df: pd.DataFrame,
    feat_cols: list,
) -> dict:
    """
    Aggregate monthly ward feature DataFrame to each season scheme.

    Returns dict with keys "biannual", "quadseasonal", "monthly",
    each containing a DataFrame with (ward_name, season/year/month, season_year, features).
    """
    log.info("\n-- PART 3: SEASON AGGREGATION --")

    df = df.copy()
    df["biannual_season"] = df["month"].apply(assign_biannual)
    df["quad_season"]     = df["month"].apply(assign_quadseasonal)
    df["biannual_year"]   = df.apply(
        lambda r: get_season_year(r["month"], r["year"], r["biannual_season"]), axis=1)
    df["quad_year"]       = df.apply(
        lambda r: get_season_year(r["month"], r["year"], r["quad_season"]), axis=1)

    group_biannual = df.groupby(["ward_name", "biannual_season", "biannual_year"])[feat_cols].mean().reset_index()
    group_biannual.rename(columns={"biannual_season": "season", "biannual_year": "season_year"}, inplace=True)
    log.info("Biannual: %s", group_biannual.shape)

    group_quad = df.groupby(["ward_name", "quad_season", "quad_year"])[feat_cols].mean().reset_index()
    group_quad.rename(columns={"quad_season": "season", "quad_year": "season_year"}, inplace=True)
    log.info("Quadseasonal: %s", group_quad.shape)

    group_monthly = df.groupby(["ward_name", "year", "month"])[feat_cols].mean().reset_index()
    log.info("Monthly: %s", group_monthly.shape)

    return {
        "biannual":     group_biannual,
        "quadseasonal": group_quad,
        "monthly":      group_monthly,
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main(
    time_start: str = DEFAULT_TIME_START,
    time_end:   str = DEFAULT_TIME_END,
    output_prefix: Optional[str] = None,
    scheme: Optional[str] = None,
    bbox: Optional[tuple] = (36.0, 1.2, 39.0, 4.5),
    n_points: int = N_SAMPLE_POINTS,
    skip_collections: Optional[set] = None,
):
    if output_prefix is None:
        output_prefix = (
            f"dzd-ayr06tncl712p3/5t7l23o0xvt99j/dev/data/inference/"
            f"ward_features_{time_start}_{time_end}"
        )

    log.info("=" * 60)
    log.info("Ward Feature Pipeline (Inference)")
    log.info("Bucket        : %s", SM_BUCKET)
    log.info("Time range    : %s → %s", time_start, time_end)
    log.info("Sample points : %d per ward (grid)", n_points)
    log.info("Output        : s3://%s/%s", SM_BUCKET, output_prefix)
    log.info("=" * 60)

    wards = sample_ward_points(n_points=n_points, bbox=bbox)
    df    = run_extraction(wards, time_start, time_end, output_prefix,
                           skip_collections=skip_collections)
    df    = run_feature_engineering(df, wards)

    feat_cols = [
        c for c in df.columns
        if c not in JOIN_KEYS + [
            "biannual_season", "quad_season", "biannual_year", "quad_year"
        ]
    ]

    null_summary = df[feat_cols].isna().mean().sort_values(ascending=False)
    log.info("\nFinal monthly shape: %s", df.shape)
    log.info("Total features: %d", len(feat_cols))
    log.info("Null rate top 15:\n%s", null_summary.head(15).round(3).to_string())

    season_dfs = aggregate_to_seasons(df, feat_cols)

    schemes_to_write = [scheme] if scheme else ["biannual", "quadseasonal", "monthly"]
    outputs = {}
    for s in schemes_to_write:
        key = f"{output_prefix}/ward_features_{s}.parquet"
        s3_upload_parquet(season_dfs[s], key)
        outputs[s] = f"s3://{SM_BUCKET}/{key}"
        log.info("Written %s: s3://%s/%s", s, SM_BUCKET, key)

    log.info("\n" + "=" * 60)
    log.info("Ward feature pipeline complete.")
    for s, uri in outputs.items():
        log.info("  %s: %s", s, uri)
    log.info("=" * 60)
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ward-level satellite feature pipeline for inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--time-start",        default=DEFAULT_TIME_START,
                        help="Start month YYYY-MM")
    parser.add_argument("--time-end",          default=DEFAULT_TIME_END,
                        help="End month YYYY-MM")
    parser.add_argument("--output-prefix",     default=None,
                        help="S3 key prefix for outputs (omit s3://bucket/ prefix)")
    parser.add_argument("--scheme",            default=None,
                        choices=["biannual", "quadseasonal", "monthly"],
                        help="Run only a specific season scheme (default: all three)")
    parser.add_argument("--n-sample-points",   default=N_SAMPLE_POINTS, type=int,
                        help="Number of grid sample points per ward polygon")
    parser.add_argument("--no-bbox",           action="store_true",
                        help="Disable Marsabit bounding box filter (use all wards)")
    parser.add_argument("--skip-collections",  nargs="*", default=None,
                        help="Collection names to skip during extraction (e.g. s1_vv s1_vh)")
    args = parser.parse_args()

    bbox = None if args.no_bbox else (36.0, 1.2, 39.0, 4.5)
    main(
        time_start=args.time_start,
        time_end=args.time_end,
        output_prefix=args.output_prefix,
        scheme=args.scheme,
        bbox=bbox,
        n_points=args.n_sample_points,
        skip_collections=set(args.skip_collections) if args.skip_collections else None,
    )
