"""
marsabit_feature_pipeline.py
=============================
Full Marsabit feature extraction and engineering pipeline.
Reads pixel parquets and labels from S3. Writes engineered
features parquet back to S3.

S3 layout (SM_BUCKET):
  planetary_computer/
    ndvi_250m.parquet, evi_250m.parquet, lai.parquet, fpar.parquet,
    lst_day.parquet, lst_night.parquet, et.parquet, pet.parquet,
    sr_red.parquet, sr_nir.parquet, sr_swir1.parquet, sr_swir2.parquet,
    fire_mask.parquet, fire_frp_max.parquet, fire_frp_sum.parquet,
    s1_vv.parquet, s1_vh.parquet,
    s2_red.parquet, s2_nir.parquet, s2_swir1.parquet,
    gpp.parquet, chirps.parquet, soil_moisture.parquet,
    dem.parquet, jrc_occurrence.parquet, jrc_seasonality.parquet,
    worldcover.parquet
  ibli/
    target_data_pipeline.csv

Output:
  planetary_computer/pc_features_engineered.parquet

Usage:
  pip install pyproj --quiet
  python marsabit_feature_pipeline.py 2>&1 | tee pipeline.log
"""

import io, os, gc, logging, warnings
import numpy as np
import pandas as pd
import boto3
from pyproj import Transformer

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

SM_BUCKET  = "amazon-sagemaker-575108933641-us-east-1-c422b90ce861"
SM_BASE    = "dzd-ayr06tncl712p3/5t7l23o0xvt99j/dev/data/training"
PC_PREFIX  = f"{SM_BASE}/planetary_computer"
IBLI_KEY   = f"{SM_BASE}/ibli/target_data_pipeline.csv"
OUTPUT_KEY = f"{PC_PREFIX}/pc_features_engineered.parquet"

# ── Spatial / temporal config ─────────────────────────────────────────────────
WINDOW_KM  = 20
TIME_START = "2008-01"
TIME_END   = "2025-12"
CLIM_YEARS = list(range(2008, 2021))

# ── Collection definitions ────────────────────────────────────────────────────
# (s3_stem, crs, resolution_m, collection_type)
# crs: "WGS84" | "UTM37N"
# collection_type: "temporal" | "static" | "worldcover"
COLLECTIONS = {
    # MODIS — WGS84
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
    # Sentinel-1 — UTM37N
    "s1_vv":           ("s1_vv",           "UTM37N",  100, "temporal"),
    "s1_vh":           ("s1_vh",           "UTM37N",  100, "temporal"),
    # Sentinel-2 — UTM37N
    "s2_red":          ("s2_red",          "UTM37N",  100, "temporal"),
    "s2_nir":          ("s2_nir",          "UTM37N",  100, "temporal"),
    "s2_swir1":        ("s2_swir1",        "UTM37N",  100, "temporal"),
    # New v2 data sources — WGS84
    "gpp":             ("gpp",             "WGS84",   500, "temporal"),
    "chirps":          ("chirps",          "WGS84",  5000, "temporal"),
    "soil_moisture":   ("soil_moisture",   "WGS84",  9000, "temporal"),
    # Static — WGS84
    "dem":             ("dem",             "WGS84",    30, "static"),
    "jrc_occurrence":  ("jrc_occurrence",  "WGS84",    30, "static"),
    "jrc_seasonality": ("jrc_seasonality", "WGS84",    30, "static"),
    # Categorical
    "worldcover":      ("worldcover",      "WGS84",    10, "worldcover"),
}

WORLDCOVER_CLASSES = {
    10: "trees",  20: "shrubland", 30: "grassland", 40: "cropland",
    50: "builtup", 60: "bare",     80: "water",      90: "wetland",
}

LARGE_COLLECTIONS = {"s1_vv", "s1_vh", "ndvi_250m", "evi_250m"}
META_COLS = ["lat", "lon", "variable", "collection"]
JOIN_KEYS = ["hhid", "gps_latitude", "gps_longitude", "year", "month"]


s3 = boto3.client("s3")


# ── S3 helpers ────────────────────────────────────────────────────────────────
def s3_load_parquet(key):
    log.info("  Loading s3://%s/%s ...", SM_BUCKET, key)
    obj = s3.get_object(Bucket=SM_BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def s3_load_csv(key):
    log.info("  Loading s3://%s/%s ...", SM_BUCKET, key)
    obj = s3.get_object(Bucket=SM_BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def s3_upload_parquet(df, key):
    tmp = f"/tmp/{key.split('/')[-1]}"
    df.to_parquet(tmp, index=False, engine="pyarrow", compression="snappy")
    s3.upload_file(tmp, SM_BUCKET, key)
    log.info("  Uploaded -> s3://%s/%s  (%.1f MB)", SM_BUCKET, key,
             os.path.getsize(tmp) / 1e6)
    os.remove(tmp)


def s3_exists(key):
    try:
        s3.head_object(Bucket=SM_BUCKET, Key=key)
        return True
    except Exception:
        return False


def pixel_key(stem):
    return f"{PC_PREFIX}/{stem}.parquet"


# ── Coordinate utilities ──────────────────────────────────────────────────────
_to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32637", always_xy=True)


def hh_coords(hh, crs):
    lons = hh["gps_longitude"].values.astype("float64")
    lats = hh["gps_latitude"].values.astype("float64")
    if crs == "UTM37N":
        lons, lats = _to_utm.transform(lons, lats)
    return lons, lats


def snap_to_grid(pixel_lons, pixel_lats, hh_lons, hh_lats):
    """pixel_lats descending, pixel_lons ascending."""
    ctr_lon = np.searchsorted(pixel_lons, hh_lons).clip(0, len(pixel_lons) - 1)
    ctr_lat = np.searchsorted(-pixel_lats, -hh_lats).clip(0, len(pixel_lats) - 1)
    return ctr_lon, ctr_lat


def half_win(resolution_m):
    return int(np.ceil(WINDOW_KM * 1000 / resolution_m))


def window_mean_2d(data_2d, ctr_lon_idxs, ctr_lat_idxs, n_lat, n_lon, hw):
    n_hh = len(ctr_lon_idxs)
    out  = np.full(n_hh, np.nan, dtype=np.float32)
    for i in range(n_hh):
        lat_lo = max(ctr_lat_idxs[i] - hw, 0)
        lat_hi = min(ctr_lat_idxs[i] + hw + 1, n_lat)
        lon_lo = max(ctr_lon_idxs[i] - hw, 0)
        lon_hi = min(ctr_lon_idxs[i] + hw + 1, n_lon)
        window = data_2d[lat_lo:lat_hi, lon_lo:lon_hi]
        if window.size > 0:
            out[i] = np.nanmean(window)
    return out


# ── PART 1: EXTRACTION ────────────────────────────────────────────────────────

def load_households():
    log.info("Loading IBLI household GPS coordinates...")
    df = s3_load_csv(IBLI_KEY)
    for col in ["gps_longitude", "gps_latitude", "hhid", "year", "month"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["gps_longitude", "gps_latitude", "hhid"])
    df["gps_latitude"]  = df["gps_latitude"].round(6)
    df["gps_longitude"] = df["gps_longitude"].round(6)
    hh = (df[["hhid", "gps_longitude", "gps_latitude"]]
          .drop_duplicates().reset_index(drop=True))
    hh["hhid"] = hh["hhid"].astype(int)
    log.info("  %d unique households", len(hh))
    return hh


def extract_temporal(var_name, stem, crs, resolution_m, hh):
    """
    Extract monthly 20km window mean from pixel parquet.
    Wide format: rows=pixels, cols=YYYY-MM.
    Soil moisture has prefixed columns (swvl1_YYYY-MM) — handled separately.
    """
    log.info("[temporal] %s  crs=%s  res=%dm", var_name, crs, resolution_m)
    df = s3_load_parquet(pixel_key(stem))

    # Soil moisture: 4 layers as separate rows with prefixed columns
    if var_name == "soil_moisture":
        return _extract_soil_moisture(df, hh)

    month_cols = sorted([c for c in df.columns
                         if c not in META_COLS and len(c) == 7 and c <= TIME_END])
    if not month_cols:
        log.warning("  No month columns — skipping %s", var_name)
        return None

    log.info("  %d pixels x %d months", len(df), len(month_cols))
    pixel_lons = np.sort(df["lon"].unique()).astype("float64")
    pixel_lats = np.sort(df["lat"].unique())[::-1].astype("float64")
    n_lon, n_lat = len(pixel_lons), len(pixel_lats)
    df = df.sort_values(["lat", "lon"], ascending=[False, True]).reset_index(drop=True)

    hh_lons, hh_lats = hh_coords(hh, crs)
    ctr_lon_idxs, ctr_lat_idxs = snap_to_grid(pixel_lons, pixel_lats, hh_lons, hh_lats)
    hw   = half_win(resolution_m)
    n_hh = len(hh)
    log.info("  HALF_WIN=%d  window=%dx%d  households=%d", hw, 2*hw+1, 2*hw+1, n_hh)

    is_large = var_name in LARGE_COLLECTIONS
    records  = []

    for idx, col in enumerate(month_cols):
        data_2d = df[col].values.reshape(n_lat, n_lon).astype("float32")
        values  = window_mean_2d(data_2d, ctr_lon_idxs, ctr_lat_idxs, n_lat, n_lon, hw)
        t = pd.to_datetime(f"{col}-01")
        for i in range(n_hh):
            records.append((int(hh["hhid"].iloc[i]),
                            float(hh["gps_latitude"].iloc[i]),
                            float(hh["gps_longitude"].iloc[i]),
                            int(t.year), int(t.month), float(values[i])))
        if is_large:
            del data_2d; gc.collect()
        if (idx + 1) % 24 == 0:
            log.info("  ... %d / %d months", idx + 1, len(month_cols))

    del df; gc.collect()
    result = pd.DataFrame(records,
        columns=["hhid", "gps_latitude", "gps_longitude", "year", "month", var_name])
    result[var_name] = result[var_name].astype("float32")
    result = result.sort_values(JOIN_KEYS).reset_index(drop=True)
    log.info("  Shape=%s  Null=%.1f%%  Mean=%.4f",
             result.shape, result[var_name].isna().mean() * 100,
             result[var_name].dropna().mean())
    return result


def _extract_soil_moisture(df, hh):
    """
    Soil moisture parquet has 4 rows per pixel (swvl1-4) with prefixed cols.
    Extract each layer, strip prefix, return wide DataFrame:
    cols = [JOIN_KEYS..., swvl1, swvl2, swvl3, swvl4]
    """
    layers  = ["swvl1", "swvl2", "swvl3", "swvl4"]
    hw      = half_win(9000)
    hh_lons, hh_lats = hh_coords(hh, "WGS84")
    n_hh    = len(hh)
    all_dfs = []

    for layer in layers:
        sub = df[df["variable"] == layer].copy()
        # Strip layer prefix from column names: swvl1_2008-01 -> 2008-01
        sub = sub.rename(columns={c: c.replace(f"{layer}_", "")
                                   for c in sub.columns if c.startswith(f"{layer}_")})
        sub = sub.drop(columns=["variable", "collection"], errors="ignore")

        month_cols = sorted([c for c in sub.columns
                              if c not in ["lat", "lon"] and len(c) == 7 and c <= TIME_END])
        if not month_cols:
            continue

        pixel_lons = np.sort(sub["lon"].unique()).astype("float64")
        pixel_lats = np.sort(sub["lat"].unique())[::-1].astype("float64")
        n_lon, n_lat = len(pixel_lons), len(pixel_lats)
        sub = sub.sort_values(["lat", "lon"], ascending=[False, True]).reset_index(drop=True)
        ctr_lon_idxs, ctr_lat_idxs = snap_to_grid(pixel_lons, pixel_lats, hh_lons, hh_lats)

        records = []
        for col in month_cols:
            data_2d = sub[col].values.reshape(n_lat, n_lon).astype("float32")
            values  = window_mean_2d(data_2d, ctr_lon_idxs, ctr_lat_idxs, n_lat, n_lon, hw)
            t = pd.to_datetime(f"{col}-01")
            for i in range(n_hh):
                records.append((int(hh["hhid"].iloc[i]),
                                float(hh["gps_latitude"].iloc[i]),
                                float(hh["gps_longitude"].iloc[i]),
                                int(t.year), int(t.month), float(values[i])))

        layer_df = pd.DataFrame(records,
            columns=["hhid", "gps_latitude", "gps_longitude", "year", "month", layer])
        all_dfs.append(layer_df)
        log.info("  %s: mean=%.4f  null=%.1f%%",
                 layer, layer_df[layer].mean(), layer_df[layer].isna().mean() * 100)

    result = all_dfs[0]
    for ldf in all_dfs[1:]:
        result = result.merge(ldf, on=JOIN_KEYS, how="outer")
    return result.sort_values(JOIN_KEYS).reset_index(drop=True)


def extract_static(var_name, stem, hh):
    """Direct pixel lookup within 20km window. Value broadcast to all months."""
    log.info("[static] %s", var_name)
    df = s3_load_parquet(pixel_key(stem))
    val_candidates = [c for c in df.columns if c not in META_COLS]
    val_col = next((c for c in ["value", var_name] if c in val_candidates),
                   val_candidates[0])
    pixel_lons = df["lon"].values.astype("float64")
    pixel_lats = df["lat"].values.astype("float64")
    pixel_vals = df[val_col].values.astype("float32")
    del df; gc.collect()

    hh_lons, hh_lats = hh_coords(hh, "WGS84")
    deg_win = WINDOW_KM / 111.0
    n_hh    = len(hh)
    values  = np.full(n_hh, np.nan, dtype=np.float32)
    for i in range(n_hh):
        mask = ((pixel_lons >= hh_lons[i] - deg_win) & (pixel_lons <= hh_lons[i] + deg_win) &
                (pixel_lats >= hh_lats[i] - deg_win) & (pixel_lats <= hh_lats[i] + deg_win))
        if mask.sum() > 0:
            values[i] = np.nanmean(pixel_vals[mask])

    del pixel_lons, pixel_lats, pixel_vals; gc.collect()
    log.info("  Null=%.1f%%  Mean=%.4f", np.isnan(values).mean() * 100,
             float(np.nanmean(values)))

    time_df   = pd.DataFrame({
        "year":  [t.year  for t in pd.date_range(TIME_START, TIME_END, freq="MS")],
        "month": [t.month for t in pd.date_range(TIME_START, TIME_END, freq="MS")],
        "_key":  1,
    })
    hh_static = pd.DataFrame({
        "hhid": hh["hhid"].values.astype(int),
        "gps_latitude": hh["gps_latitude"].values,
        "gps_longitude": hh["gps_longitude"].values,
        var_name: values, "_key": 1,
    })
    result = hh_static.merge(time_df, on="_key").drop(columns="_key")
    return result.sort_values(JOIN_KEYS).reset_index(drop=True)


def extract_worldcover(hh):
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

    hh_lons, hh_lats = hh_coords(hh, "WGS84")
    deg_win = WINDOW_KM / 111.0
    n_hh    = len(hh)
    hh_wc   = pd.DataFrame({"hhid": hh["hhid"].values.astype(int),
                             "gps_latitude": hh["gps_latitude"].values,
                             "gps_longitude": hh["gps_longitude"].values})

    for code, name in WORLDCOVER_CLASSES.items():
        fractions = np.full(n_hh, np.nan, dtype=np.float32)
        for i in range(n_hh):
            mask = ((pixel_lons >= hh_lons[i] - deg_win) & (pixel_lons <= hh_lons[i] + deg_win) &
                    (pixel_lats >= hh_lats[i] - deg_win) & (pixel_lats <= hh_lats[i] + deg_win))
            window = pixel_vals[mask]
            if window.size > 0:
                fractions[i] = float((window == code).sum()) / window.size
        hh_wc[f"wc_{name}"] = fractions
        log.info("  wc_%s: mean=%.3f", name, float(np.nanmean(fractions)))

    del pixel_lons, pixel_lats, pixel_vals; gc.collect()
    time_df = pd.DataFrame({
        "year":  [t.year  for t in pd.date_range(TIME_START, TIME_END, freq="MS")],
        "month": [t.month for t in pd.date_range(TIME_START, TIME_END, freq="MS")],
        "_key":  1,
    })
    hh_wc["_key"] = 1
    result = hh_wc.merge(time_df, on="_key").drop(columns="_key")
    return result.sort_values(JOIN_KEYS).reset_index(drop=True)


def run_extraction(hh):
    log.info("\n-- PART 1: EXTRACTION --")

    time_index = pd.DataFrame({
        "year":  [t.year  for t in pd.date_range(TIME_START, TIME_END, freq="MS")],
        "month": [t.month for t in pd.date_range(TIME_START, TIME_END, freq="MS")],
        "_key":  1,
    })
    hh_index = hh[["hhid", "gps_latitude", "gps_longitude"]].copy()
    hh_index["_key"] = 1
    canonical = (hh_index.merge(time_index, on="_key").drop(columns="_key")
                 .astype({"hhid": int, "year": int, "month": int})
                 .sort_values(JOIN_KEYS).reset_index(drop=True))
    log.info("Canonical index: %d rows (%d HH x %d months)",
             len(canonical), len(hh),
             len(pd.date_range(TIME_START, TIME_END, freq="MS")))

    extracted = {}

    for var_name, (stem, crs, res_m, ctype) in COLLECTIONS.items():
        if ctype not in ("temporal", "static"):
            continue
        ck = f"{PC_PREFIX}/{var_name}_extracted.parquet"
        if s3_exists(ck):
            log.info("[skip] %s -- cache hit", var_name)
            extracted[var_name] = s3_load_parquet(ck)
            continue
        try:
            df_out = (extract_temporal(var_name, stem, crs, res_m, hh)
                      if ctype == "temporal"
                      else extract_static(var_name, stem, hh))
            if df_out is not None:
                s3_upload_parquet(df_out, ck)
                extracted[var_name] = df_out
        except Exception as e:
            log.error("FAILED %s: %s", var_name, e, exc_info=True)

    ck = f"{PC_PREFIX}/worldcover_extracted.parquet"
    if s3_exists(ck):
        log.info("[skip] worldcover -- cache hit")
        extracted["worldcover"] = s3_load_parquet(ck)
    else:
        try:
            df_out = extract_worldcover(hh)
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


# ── PART 2: FEATURE ENGINEERING ───────────────────────────────────────────────

def compute_dem_roughness(hh_lats, hh_lons):
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
    n_hh        = len(hh_lats)
    dem_std_v   = np.full(n_hh, np.nan, dtype="float32")
    dem_range_v = np.full(n_hh, np.nan, dtype="float32")
    for i in range(n_hh):
        mask  = ((pixel_lons >= hh_lons[i] - deg_win) & (pixel_lons <= hh_lons[i] + deg_win) &
                 (pixel_lats >= hh_lats[i] - deg_win) & (pixel_lats <= hh_lats[i] + deg_win))
        valid = pixel_vals[mask]; valid = valid[~np.isnan(valid)]
        if len(valid) > 1:
            dem_std_v[i]   = float(np.std(valid))
            dem_range_v[i] = float(np.max(valid) - np.min(valid))
    del pixel_lons, pixel_lats, pixel_vals; gc.collect()
    log.info("  dem_std mean=%.1fm  dem_range mean=%.1fm",
             np.nanmean(dem_std_v), np.nanmean(dem_range_v))
    return dem_std_v, dem_range_v


def tier0_preprocess(df):
    """ET/PET from MOD16A3GF are annual -- forward-fill within each year."""
    log.info("Tier 0: ET/PET forward fill...")
    for col in ["et", "pet"]:
        if col in df.columns:
            df[col] = (df.groupby(["hhid", "gps_latitude", "gps_longitude", "year"])[col]
                       .transform(lambda x: x.ffill().bfill()))
            log.info("  %s filled. Null=%.1f%%", col, df[col].isna().mean() * 100)
    return df


def tier1_direct_indices(df):
    """Spectral indices, SAR features, water balance, season flags."""
    log.info("Tier 1: Direct indices...")

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


def tier2_temporal_windows(df):
    """Lags 1-3, adjacent diff/ratio, YoY diff/ratio, 3-month rolling."""
    log.info("Tier 2: Temporal window features...")
    df  = df.sort_values(JOIN_KEYS)
    grp = df.groupby(["hhid", "gps_latitude", "gps_longitude"], sort=False)

    TEMPORAL_VARS = [c for c in [
        "ndvi_250m", "evi_250m", "lst_day", "lst_night",
        "lai", "fpar", "sr_nir", "sr_red",
        "sar_rvi", "sar_vv_vh_ratio", "s1_vv", "s1_vh",
        "et_deficit", "et_fraction", "lst_diurnal_range",
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


def tier3_drought_composites(df):
    """VCI, TCI, VHI, drought flags, compound stress + VCI/TCI/VHI lags."""
    log.info("Tier 3: Drought composites...")
    clim_mask = df["year"].isin(CLIM_YEARS)
    HH_MONTH  = ["hhid", "gps_latitude", "gps_longitude", "month"]

    if "ndvi_250m" in df.columns:
        clim = (df.loc[clim_mask].groupby(HH_MONTH)["ndvi_250m"]
                .agg(ndvi_clim_mean="mean", ndvi_clim_std="std",
                     ndvi_p02=lambda x: x.quantile(0.02),
                     ndvi_p98=lambda x: x.quantile(0.98))
                .reset_index())
        df = df.merge(clim, on=HH_MONTH, how="left")
        df["ndvi_anom"]     = df["ndvi_250m"] - df["ndvi_clim_mean"]
        df["ndvi_anom_std"] = df["ndvi_anom"] / df["ndvi_clim_std"].replace(0, np.nan)
        rng = (df["ndvi_p98"] - df["ndvi_p02"]).replace(0, np.nan)
        df["vci"] = ((df["ndvi_250m"] - df["ndvi_p02"]) / rng * 100).clip(0, 100)
        df = df.drop(columns=["ndvi_clim_mean", "ndvi_clim_std", "ndvi_p02", "ndvi_p98"])
        log.info("  NDVI anomaly, VCI")

    if "lst_day" in df.columns:
        clim = (df.loc[clim_mask].groupby(HH_MONTH)["lst_day"]
                .agg(lst_clim_mean="mean",
                     lst_p02=lambda x: x.quantile(0.02),
                     lst_p98=lambda x: x.quantile(0.98))
                .reset_index())
        df = df.merge(clim, on=HH_MONTH, how="left")
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
    grp = df.groupby(["hhid", "gps_latitude", "gps_longitude"], sort=False)
    for col in ["vci", "tci", "vhi"]:
        if col in df.columns:
            for lag in [1, 2, 3]:
                df[f"{col}_lag{lag}"] = grp[col].shift(lag)
            df[f"{col}_roll3_mean"] = grp[col].transform(
                lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    log.info("  VCI/TCI/VHI lags 1-3 and rolling mean")
    return df


def tier4_fire(df):
    """fire_detected, cumulative_count, months_since_fire, fire_count_12m."""
    log.info("Tier 4: Fire features...")
    if "fire_mask" not in df.columns:
        log.warning("  fire_mask not found -- skipping")
        return df

    df  = df.sort_values(JOIN_KEYS)
    grp = df.groupby(["hhid", "gps_latitude", "gps_longitude"], sort=False)

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


def tier4_longterm_ndvi(df):
    """Long-term NDVI stats per household: CV, percentiles, drought count, seasonal means."""
    log.info("Tier 4: Long-term NDVI statistics...")
    if "ndvi_250m" not in df.columns:
        return df

    HH_KEYS = ["hhid", "gps_latitude", "gps_longitude"]
    annual  = (df.groupby(HH_KEYS + ["year"])["ndvi_250m"].mean()
               .reset_index().rename(columns={"ndvi_250m": "ndvi_annual"}))

    hh_stats = (annual.groupby(HH_KEYS)["ndvi_annual"]
                .agg(ndvi_lt_mean="mean", ndvi_lt_std="std",
                     ndvi_lt_p10=lambda x: x.quantile(0.10),
                     ndvi_lt_p25=lambda x: x.quantile(0.25),
                     ndvi_lt_p75=lambda x: x.quantile(0.75),
                     ndvi_lt_p90=lambda x: x.quantile(0.90))
                .reset_index())
    hh_stats["ndvi_lt_cv"] = (hh_stats["ndvi_lt_std"] /
                               hh_stats["ndvi_lt_mean"].replace(0, np.nan))

    p20 = (annual.groupby(HH_KEYS)["ndvi_annual"].quantile(0.20)
           .reset_index().rename(columns={"ndvi_annual": "ndvi_p20"}))
    annual = annual.merge(p20, on=HH_KEYS, how="left")
    drought_count = (annual
                     .assign(is_drought=(annual["ndvi_annual"] < annual["ndvi_p20"]).astype(int))
                     .groupby(HH_KEYS)["is_drought"].sum().reset_index()
                     .rename(columns={"is_drought": "ndvi_drought_year_count"}))
    hh_stats = hh_stats.merge(drought_count, on=HH_KEYS, how="left")

    for season, months in [("mam",[3,4,5]), ("ond",[10,11,12]), ("jfas",[1,2,6,7,8,9])]:
        seas = (df[df["month"].isin(months)].groupby(HH_KEYS)["ndvi_250m"]
                .mean().reset_index().rename(columns={"ndvi_250m": f"ndvi_mean_{season}"}))
        hh_stats = hh_stats.merge(seas, on=HH_KEYS, how="left")

    df = df.merge(hh_stats, on=HH_KEYS, how="left")
    log.info("  ndvi_lt_cv, P10/P25/P75/P90, drought_year_count, MAM/OND/JFAS")
    return df


def run_feature_engineering(df, hh):
    log.info("\n-- PART 2: FEATURE ENGINEERING --")

    hh_unique = df[["hhid", "gps_latitude", "gps_longitude"]].drop_duplicates()
    dem_std_v, dem_range_v = compute_dem_roughness(
        hh_unique["gps_latitude"].values, hh_unique["gps_longitude"].values)
    hh_topo = pd.DataFrame({
        "hhid":          hh_unique["hhid"].values,
        "gps_latitude":  hh_unique["gps_latitude"].values,
        "gps_longitude": hh_unique["gps_longitude"].values,
        "dem_std":       dem_std_v,
        "dem_range":     dem_range_v,
    })
    df = df.merge(hh_topo, on=["hhid", "gps_latitude", "gps_longitude"], how="left")
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


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Marsabit Feature Pipeline -- Native SageMaker S3")
    log.info("Bucket : %s", SM_BUCKET)
    log.info("PC prefix : %s", PC_PREFIX)
    log.info("Output : %s", OUTPUT_KEY)
    log.info("=" * 60)

    hh = load_households()
    df = run_extraction(hh)
    df = run_feature_engineering(df, hh)

    feat_cols    = [c for c in df.columns if c not in JOIN_KEYS]
    null_summary = df[feat_cols].isna().mean().sort_values(ascending=False)
    log.info("\nFinal shape: %s", df.shape)
    log.info("Total features: %d", len(feat_cols))
    log.info("Null rate top 15:\n%s", null_summary.head(15).round(3).to_string())
    log.info("Rows per year:\n%s", df.groupby("year")["hhid"].count().to_string())

    s3_upload_parquet(df, OUTPUT_KEY)

    log.info("\n" + "=" * 60)
    log.info("Pipeline complete.")
    log.info("Output: s3://%s/%s", SM_BUCKET, OUTPUT_KEY)
    log.info("=" * 60)
    return df


if __name__ == "__main__":
    main()
