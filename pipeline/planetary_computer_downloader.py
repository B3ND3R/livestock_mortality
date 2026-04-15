"""
planetary_computer_downloader.py

Downloads collections from Microsoft Planetary Computer as monthly median
composites and saves them to Parquet — one file per collection/variable.

Output schema (wide format, one row per pixel):
    lat        – WGS84 latitude
    lon        – WGS84 longitude
    variable   – variable name (e.g. "ndvi")
    collection – Planetary Computer collection id
    <YYYY-MM>  – one column per month, float32 pixel value (NaN = missing)

Static collections (esa-worldcover, cop-dem-glo-30, jrc-gsw) have no time
dimension — they are saved with a single value column instead of month columns.

Usage
-----
    python planetary_computer_downloader.py

R2 upload
---------
Set these env vars before running to auto-upload each Parquet file to
Cloudflare R2 (or any S3-compatible store) as it completes:

    export R2_ENDPOINT=https://<account_id>.r2.cloudflarestorage.com
    export R2_BUCKET=marsabit-data
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
"""

from __future__ import annotations

import gc
import logging
import os
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import odc.stac
import pandas as pd
import planetary_computer as pc
import pystac_client

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bounding box and date range
# ---------------------------------------------------------------------------

BBOX       = [36.013414, 1.2755, 38.960949, 4.459143]   # Marsabit County
START_DATE = "2008-01-01"
END_DATE   = "2026-01-01"


# ---------------------------------------------------------------------------
# Collection configuration
# ---------------------------------------------------------------------------
#
# Each entry maps a short variable name → config dict.
#
# Required keys:
#   collection_id – Planetary Computer collection slug
#   band          – asset/band name to load (str), or [red, nir] for NDVI
#   scale_factor  – divide raw int values by this to get physical units
#   resolution    – output spatial resolution in metres
#   groupby       – odc.stac groupby key (use "solar_day" for time-series)
#   date_field    – item property with the scene date:
#                     "start_datetime" → MODIS
#                     None            → use item.datetime (Landsat, Sentinel)
#
# Optional keys:
#   is_static     – True for single-epoch datasets (DEM, WorldCover, JRC-GSW)
#                   These are downloaded once, not as monthly composites.
#   extra_load_kw – dict of extra kwargs passed straight to odc.stac.stac_load
#                   e.g. {"nodata": 0} for categorical layers

COLLECTIONS: dict[str, dict[str, Any]] = {

    # ── MODIS Vegetation Indices 250m (EVI + NDVI) ──────────────────────────
    # 16-day composites, 2000-present.  Scale: raw / 10000 → [-1, 1]
    "ndvi_250m": {
        "collection_id": "modis-13Q1-061",
        "band":          "250m_16_days_NDVI",
        "scale_factor":  10_000,
        "resolution":    250,
        "groupby":       "solar_day",
        "date_field":    "start_datetime",
    },
    "evi_250m": {
        "collection_id": "modis-13Q1-061",
        "band":          "250m_16_days_EVI",
        "scale_factor":  10_000,
        "resolution":    250,
        "groupby":       "solar_day",
        "date_field":    "start_datetime",
    },

    # ── MODIS LAI / FPAR 500m ───────────────────────────────────────────────
    # 8-day composites, 2002-present.
    # LAI  scale: raw / 10  → [0, 10]  m²/m²
    # FPAR scale: raw / 100 → [0, 1]   fraction
    "lai": {
        "collection_id": "modis-15A2H-061",
        "band":          "Lai_500m",
        "scale_factor":  10,
        "resolution":    500,
        "groupby":       "solar_day",
        "date_field":    "start_datetime",
    },
    "fpar": {
        "collection_id": "modis-15A2H-061",
        "band":          "Fpar_500m",
        "scale_factor":  100,
        "resolution":    500,
        "groupby":       "solar_day",
        "date_field":    "start_datetime",
    },

    # ── MODIS Thermal Anomalies / Fire Daily 1km ─────────────────────────────
    # Daily, 2000-present.
    # FireMask: categorical flag (0=no data, 3=non-fire, 7=fire, 8=high-conf fire)
    # MaxFRP:   maximum fire radiative power (MW), scale raw / 10
    "fire_mask": {
        "collection_id": "modis-14A1-061",
        "band":          "FireMask",
        "scale_factor":  1,          # categorical — no scaling needed
        "resolution":    1000,
        "groupby":       "solar_day",
        "date_field":    "start_datetime",
        "extra_load_kw": {"nodata": 0},
    },
    "fire_frp": {
        "collection_id": "modis-14A1-061",
        "band":          "MaxFRP",
        "scale_factor":  10,         # raw / 10 → MW
        "resolution":    1000,
        "groupby":       "solar_day",
        "date_field":    "start_datetime",
    },

    # ── MODIS Surface Reflectance 8-Day 500m ────────────────────────────────
    # Bands 1-7, 8-day composites, 2000-present.
    # All surface reflectance bands: scale raw / 10000 → [0, 1]
    # Bands used here: sur_refl_b01 (red 620-670nm), sur_refl_b02 (NIR 841-876nm),
    #                  sur_refl_b06 (SWIR1 1628-1652nm), sur_refl_b07 (SWIR2 2105-2155nm)
    "sr_red": {
        "collection_id": "modis-09A1-061",
        "band":          "sur_refl_b01",
        "scale_factor":  10_000,
        "resolution":    500,
        "groupby":       "solar_day",
        "date_field":    "start_datetime",
    },
    "sr_nir": {
        "collection_id": "modis-09A1-061",
        "band":          "sur_refl_b02",
        "scale_factor":  10_000,
        "resolution":    500,
        "groupby":       "solar_day",
        "date_field":    "start_datetime",
    },
    "sr_swir1": {
        "collection_id": "modis-09A1-061",
        "band":          "sur_refl_b06",
        "scale_factor":  10_000,
        "resolution":    500,
        "groupby":       "solar_day",
        "date_field":    "start_datetime",
    },
    "sr_swir2": {
        "collection_id": "modis-09A1-061",
        "band":          "sur_refl_b07",
        "scale_factor":  10_000,
        "resolution":    500,
        "groupby":       "solar_day",
        "date_field":    "start_datetime",
    },

    # ── MODIS LST 8-Day 1km ─────────────────────────────────────────────────
    # 8-day composites, 2000-present.
    # Scale: raw * 0.02 → Kelvin  (equivalent to raw / 50)
    "lst_day": {
        "collection_id": "modis-11A2-061",
        "band":          "LST_Day_1km",
        "scale_factor":  50,         # raw / 50 = raw * 0.02 → Kelvin
        "resolution":    1000,
        "groupby":       "solar_day",
        "date_field":    "start_datetime",
    },
    "lst_night": {
        "collection_id": "modis-11A2-061",
        "band":          "LST_Night_1km",
        "scale_factor":  50,
        "resolution":    1000,
        "groupby":       "solar_day",
        "date_field":    "start_datetime",
    },

    # ── MODIS Net ET 500m (Annual) ───────────────────────────────────────────
    # Annual composites, 2001-present.
    # ET scale: raw / 10 → kg/m² (mm of water equivalent)
    "et": {
        "collection_id": "modis-16A3GF-061",
        "band":          "ET_500m",
        "scale_factor":  10,
        "resolution":    500,
        "groupby":       "solar_day",
        "date_field":    "start_datetime",
    },
    "pet": {
        "collection_id": "modis-16A3GF-061",
        "band":          "PET_500m",
        "scale_factor":  10,
        "resolution":    500,
        "groupby":       "solar_day",
        "date_field":    "start_datetime",
    },

    # ── JRC Global Surface Water 30m ─────────────────────────────────────────
    # Static product (1984-2020 composite). Downloaded once, no monthly loop.
    # occurrence:  % of time water was present  [0-100]
    # seasonality: number of months water was present in 2020  [0-12]
    "jrc_occurrence": {
        "collection_id": "jrc-gsw",
        "band":          "occurrence",
        "scale_factor":  1,
        "resolution":    30,
        "groupby":       "solar_day",
        "date_field":    None,
        "is_static":     True,
    },
    "jrc_seasonality": {
        "collection_id": "jrc-gsw",
        "band":          "seasonality",
        "scale_factor":  1,
        "resolution":    30,
        "groupby":       "solar_day",
        "date_field":    None,
        "is_static":     True,
    },

    # ── ESA WorldCover 10m ───────────────────────────────────────────────────
    # Static land-cover classification (2020 and 2021 epochs only).
    # Band "map": integer class codes, no scaling needed.
    # Class codes: 10=Trees, 20=Shrubland, 30=Grassland, 40=Cropland,
    #              50=Built-up, 60=Bare, 70=Snow/Ice, 80=Water,
    #              90=Herbaceous wetland, 95=Mangroves, 100=Moss/Lichen
    "worldcover": {
        "collection_id": "esa-worldcover",
        "band":          "map",
        "scale_factor":  1,
        "resolution":    10,
        "groupby":       "solar_day",
        "date_field":    None,
        "is_static":     True,
        "extra_load_kw": {"nodata": 0},
    },

    # ── Copernicus DEM GLO-30 ────────────────────────────────────────────────
    # Static 30m DEM. Elevation in metres above EGM2008 geoid.
    # Scale: raw / 1 (already in metres, float32)
    "dem": {
        "collection_id": "cop-dem-glo-30",
        "band":          "data",
        "scale_factor":  1,
        "resolution":    30,
        "groupby":       "solar_day",
        "date_field":    None,
        "is_static":     True,
    },

    # ── Sentinel-2 L2A 10m ──────────────────────────────────────────────────
    # ~5-day revisit, 2015-present. Bottom-of-atmosphere reflectance.
    # Scale: raw / 10000 → [0, 1].
    # Bands loaded: B04 (red), B08 (NIR), B11 (SWIR1), B12 (SWIR2)
    "s2_red": {
        "collection_id": "sentinel-2-l2a",
        "band":          "B04",
        "scale_factor":  10_000,
        "resolution":    10,
        "groupby":       "solar_day",
        "date_field":    None,
    },
    "s2_nir": {
        "collection_id": "sentinel-2-l2a",
        "band":          "B08",
        "scale_factor":  10_000,
        "resolution":    10,
        "groupby":       "solar_day",
        "date_field":    None,
    },
    "s2_swir1": {
        "collection_id": "sentinel-2-l2a",
        "band":          "B11",
        "scale_factor":  10_000,
        "resolution":    20,          # B11 is 20m native
        "groupby":       "solar_day",
        "date_field":    None,
    },

    # ── Sentinel-1 GRD SAR 10m ──────────────────────────────────────────────
    # ~12-day revisit, 2014-present. C-band backscatter in linear power units.
    # Values are already in dB (10*log10 scale) on Planetary Computer.
    # Scale: 1 (no scaling — values are float dB, typically -25 to +5 dB)
    "s1_vv": {
        "collection_id": "sentinel-1-grd",
        "band":          "vv",
        "scale_factor":  1,
        "resolution":    10,
        "groupby":       "solar_day",
        "date_field":    None,
    },
    "s1_vh": {
        "collection_id": "sentinel-1-grd",
        "band":          "vh",
        "scale_factor":  1,
        "resolution":    10,
        "groupby":       "solar_day",
        "date_field":    None,
    },
}


# ---------------------------------------------------------------------------
# MODIS Sinusoidal → WGS84 reprojection
# ---------------------------------------------------------------------------

_MODIS_SINU_WKT = """PROJCS["Sinusoidal",
    GEOGCS["GCS_Undefined",
        DATUM["Undefined",
            SPHEROID["User_Defined_Spheroid",6371007.181,0.0]],
        PRIMEM["Greenwich",0.0],
        UNIT["Degree",0.0174532925199433]],
    PROJECTION["Sinusoidal"],
    PARAMETER["False_Easting",0.0],
    PARAMETER["False_Northing",0.0],
    PARAMETER["Central_Meridian",0.0],
    UNIT["Meter",1.0]]"""


def _reproject_coords(
    x_coords: np.ndarray,
    y_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert MODIS Sinusoidal coords to WGS84 lon/lat if needed."""
    if abs(x_coords).max() <= 180 and abs(y_coords).max() <= 90:
        return x_coords, y_coords
    try:
        from pyproj import CRS, Transformer
    except ImportError:
        log.warning("pyproj not installed – coords kept as Sinusoidal metres.")
        return x_coords, y_coords

    modis_crs = CRS.from_wkt(_MODIS_SINU_WKT)
    tf = Transformer.from_crs(modis_crs, "EPSG:4326", always_xy=True)
    lon_1d = np.array([tf.transform(x, y_coords[0])[0] for x in x_coords])
    lat_1d = np.array([tf.transform(x_coords[0], y)[1] for y in y_coords])
    return lon_1d, lat_1d


# ---------------------------------------------------------------------------
# Retry / back-off
# ---------------------------------------------------------------------------

@dataclass
class RetryConfig:
    max_attempts:   int   = 5
    base_delay:     float = 2.0
    max_delay:      float = 120.0
    backoff_factor: float = 2.0

    def sleep_durations(self):
        d = self.base_delay
        for _ in range(self.max_attempts - 1):
            yield min(d, self.max_delay)
            d *= self.backoff_factor


def _is_transient(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(k in msg for k in (
        "403", "401", "sas", "token", "expired",
        "429", "rate limit", "too many requests",
        "connection", "timeout", "ssl",
    ))


def with_retry(fn, *args, retry: RetryConfig | None = None, **kwargs):
    retry = retry or RetryConfig()
    last: Exception | None = None
    for attempt, delay in enumerate([0.0, *retry.sleep_durations()], start=1):
        if delay:
            log.warning("  Retry %d/%d in %.0fs  (%s)", attempt, retry.max_attempts, delay, last)
            time.sleep(delay)
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last = exc
            if not _is_transient(exc):
                raise
    raise RuntimeError(f"All {retry.max_attempts} attempts failed: {last}") from last


# ---------------------------------------------------------------------------
# R2 / S3 upload
# ---------------------------------------------------------------------------

def _upload_to_r2(local_path: Path, object_key: str) -> None:
    endpoint = os.getenv("R2_ENDPOINT")
    bucket   = os.getenv("R2_BUCKET")
    if not (endpoint and bucket):
        return
    try:
        import boto3
        s3 = boto3.client("s3", endpoint_url=endpoint)
        log.info("  Uploading %s → s3://%s/%s", local_path.name, bucket, object_key)
        s3.upload_file(str(local_path), bucket, object_key)
        log.info("  Upload complete.")
    except Exception as exc:
        log.error("  R2 upload failed: %s", exc)


# ---------------------------------------------------------------------------
# Core downloader
# ---------------------------------------------------------------------------

class PlanetaryComputerDownloader:

    def __init__(
        self,
        bbox:       list[float],
        start_date: str,
        end_date:   str,
        output_dir: str | Path = "./outputs",
        retry:      RetryConfig | None = None,
    ) -> None:
        self.bbox       = bbox
        self.start_date = start_date
        self.end_date   = end_date
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.retry      = retry or RetryConfig()

        self._catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=pc.sign_inplace,
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, collections: dict[str, dict[str, Any]]) -> list[Path]:
        written: list[Path] = []
        total = len(collections)

        for i, (var_name, cfg) in enumerate(collections.items(), start=1):
            log.info("")
            log.info("=" * 60)
            log.info("[%d/%d]  %s  (%s)", i, total, var_name, cfg["collection_id"])
            log.info("=" * 60)

            if cfg.get("is_static"):
                path = self._download_static(var_name, cfg)
            else:
                path = self._download_timeseries(var_name, cfg)

            if path:
                written.append(path)
                _upload_to_r2(path, path.name)
            else:
                log.warning("  Skipped %s – no data returned.", var_name)

        log.info("")
        log.info("Finished. %d / %d collections written.", len(written), total)
        return written

    # ------------------------------------------------------------------
    # Time-series pipeline (most collections)
    # ------------------------------------------------------------------

    def _download_timeseries(self, var_name: str, cfg: dict) -> Path | None:
        items = self._search_items(cfg["collection_id"])
        if not items:
            return None
        log.info("  Found %d items.", len(items))

        month_map   = self._group_by_month(items, cfg.get("date_field"))
        full_months = pd.period_range(self.start_date, self.end_date, freq="M")
        log.info(
            "  %d months in range | %d have data | %d will be NaN",
            len(full_months), len(month_map),
            len(full_months) - len(set(full_months) & set(month_map)),
        )

        monthly_arrays: dict[str, np.ndarray] = {}
        x_coords = y_coords = None

        for i, month in enumerate(full_months):
            col = str(month)

            if month in month_map:
                result = self._load_month(
                    month_map[month], cfg["band"],
                    cfg["resolution"], cfg.get("groupby", "solar_day"),
                    cfg.get("extra_load_kw", {}),
                )
                if result is not None:
                    if x_coords is None:
                        x_coords = result.x.values
                        y_coords = result.y.values
                    monthly_arrays[col] = (
                        result.astype("float32").values / cfg["scale_factor"]
                    )
                    del result
                    gc.collect()
                    continue

            if x_coords is not None:
                monthly_arrays[col] = np.full(
                    (len(y_coords), len(x_coords)), np.nan, dtype="float32"
                )

            if (i + 1) % 24 == 0:
                log.info("  Progress: %d / %d months …", i + 1, len(full_months))

        if not monthly_arrays or x_coords is None:
            return None

        return self._save_parquet(var_name, cfg, x_coords, y_coords, monthly_arrays)

    # ------------------------------------------------------------------
    # Static pipeline (DEM, WorldCover, JRC-GSW)
    # ------------------------------------------------------------------

    def _download_static(self, var_name: str, cfg: dict) -> Path | None:
        """Download a single-epoch dataset — no monthly compositing."""
        items = self._search_items(cfg["collection_id"])
        if not items:
            return None
        log.info("  Found %d items (static).", len(items))

        result = self._load_month(
            items, cfg["band"],
            cfg["resolution"], cfg.get("groupby", "solar_day"),
            cfg.get("extra_load_kw", {}),
        )
        if result is None:
            return None

        x_coords = result.x.values
        y_coords = result.y.values
        # Store as a single column named "value"
        arrays = {"value": result.astype("float32").values / cfg["scale_factor"]}
        del result
        gc.collect()

        return self._save_parquet(var_name, cfg, x_coords, y_coords, arrays)

    # ------------------------------------------------------------------
    # Build DataFrame and write Parquet
    # ------------------------------------------------------------------

    def _save_parquet(
        self,
        var_name:  str,
        cfg:       dict,
        x_coords:  np.ndarray,
        y_coords:  np.ndarray,
        arrays:    dict[str, np.ndarray],
    ) -> Path | None:
        lon_1d, lat_1d = _reproject_coords(x_coords, y_coords)
        lon_grid, lat_grid = np.meshgrid(lon_1d, lat_1d)

        df = pd.DataFrame({
            "lat":        lat_grid.ravel().astype("float32"),
            "lon":        lon_grid.ravel().astype("float32"),
            "variable":   var_name,
            "collection": cfg["collection_id"],
        })

        for col, arr in sorted(arrays.items()):
            if arr.shape == (len(lat_1d), len(lon_1d)):
                df[col] = arr.ravel()
            else:
                df[col] = np.nan

        pct_valid = df[list(arrays.keys())].notna().values.mean() * 100
        log.info(
            "  DataFrame  rows=%d  cols=%d  valid=%.1f%%",
            len(df), len(arrays), pct_valid,
        )

        out_path = self.output_dir / f"{var_name}.parquet"
        df.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")
        log.info("  Saved → %s  (%.1f MB)", out_path, out_path.stat().st_size / 1e6)
        return out_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _search_items(self, collection_id: str) -> list:
        def _do():
            return list(
                self._catalog.search(
                    collections=[collection_id],
                    bbox=self.bbox,
                    datetime=f"{self.start_date}/{self.end_date}",
                ).item_collection()
            )
        return with_retry(_do, retry=self.retry)

    def _group_by_month(self, items: list, date_field: str | None) -> dict:
        month_map: dict[pd.Period, list] = {}
        for item in items:
            raw = item.properties.get(date_field) if date_field else item.datetime
            if raw is None:
                continue
            period = pd.to_datetime(raw).to_period("M")
            month_map.setdefault(period, []).append(item)
        return month_map

    def _load_month(
        self,
        month_items:  list,
        band:         str | list[str],
        resolution:   int,
        groupby:      str,
        extra_load_kw: dict,
    ):
        def _do():
            signed = [pc.sign(item) for item in month_items]
            bands  = [band] if isinstance(band, str) else band

            stack = odc.stac.stac_load(
                signed,
                bands=bands,
                bbox=self.bbox,
                resolution=resolution,
                groupby=groupby,
                **extra_load_kw,
            )

            # Derived NDVI when band is [red_name, nir_name]
            if isinstance(band, list) and len(band) == 2:
                r  = stack[band[0]].astype("float32")
                n  = stack[band[1]].astype("float32")
                da = (n - r) / (n + r)
            else:
                da = stack[band]

            if "time" in da.dims and len(da.time) > 1:
                return da.median(dim="time").compute()
            return (
                da.squeeze("time", drop=True).compute()
                if "time" in da.dims
                else da.compute()
            )

        try:
            return with_retry(_do, retry=self.retry)
        except Exception as exc:
            log.warning("  Load failed after all retries: %s", exc)
            return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dl = PlanetaryComputerDownloader(
        bbox=BBOX,
        start_date=START_DATE,
        end_date=END_DATE,
        output_dir="./outputs",
    )
    dl.run(COLLECTIONS)
