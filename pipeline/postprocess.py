"""
postprocess.py

ADMIN3 Ward-Level Postprocessing for Livestock Mortality Risk Predictions.

Takes household-level model predictions with GPS coordinates and aggregates
them to Kenya ADMIN3 ward boundaries (IEBC wards) using a spatial join.

Outputs per ward
----------------
1. Risk level   - categorical (Normal / Concerning / Critical), relative to
                  average predicted loss across all observations
2. Confidence   - prediction agreement within ward (1 - normalized std)
3. Top features - SHAP-based feature importance aggregated per ward

Can be used standalone or as a SageMaker Pipeline step.

Artifacts logged to MLflow
--------------------------
- postprocessing/ward_predictions.csv
- postprocessing/ward_predictions.geojson
- postprocessing/ward_predictions.tif  (3-band GeoTIFF)

GeoTIFF bands
-------------
  Band 1 : risk_level_encoded   int16   0=Normal 1=Concerning 2=Critical  nodata=-1
  Band 2 : confidence           float32 0–1                               nodata=-9999
  Band 3 : top_feature_importance float32 mean |SHAP| of #1 feature/ward  nodata=-9999
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Constants ────────────────────────────────────────────────────────────────

TRACKING_SERVER_ARN = (
    "arn:aws:sagemaker:us-east-1:575108933641:"
    "mlflow-tracking-server/lmr-tracking-server-5t7l23o0xvt99j-chws71x3trpelj-dev"
)

# Default relative risk thresholds (fraction above average predicted loss).
# "Normal"     = ward mean is 0–10% above the global average
# "Concerning" = ward mean is 10–20% above the global average
# "Critical"   = ward mean is >20% above the global average
DEFAULT_RISK_THRESHOLDS = {
    "Normal": (0.0, 0.10),
    "Concerning": (0.10, 0.20),
    "Critical": (0.20, float("inf")),
}


def load_admin3_boundaries(shapefile_path: str) -> "gpd.GeoDataFrame":
    """
    Load Kenya ADMIN3 (ward) boundary polygons from a local or S3 path.

    Parameters
    ----------
    shapefile_path : str
        Local path or S3 URI to the ADMIN3 shapefile/GeoJSON/GeoPackage.

    Returns
    -------
    gpd.GeoDataFrame with columns: ADM3_EN, ADM3_PCODE, ADM2_EN, ADM1_EN, geometry
    """
    import geopandas as gpd

    gdf = gpd.read_file(shapefile_path)

    # Standardize column names to ADM3_EN, ADM3_PCODE, ADM2_EN, ADM1_EN
    required = ["ADM3_EN", "ADM3_PCODE", "ADM2_EN", "ADM1_EN", "geometry"]
    missing = [c for c in required if c not in gdf.columns]
    if missing:
        # Try case-insensitive match first
        col_map = {c.upper(): c for c in gdf.columns}
        for req in missing[:]:
            if req.upper() in col_map:
                gdf = gdf.rename(columns={col_map[req.upper()]: req})

        # Fallback: geoBoundaries format (shapeName, shapeID, etc.)
        geo_boundaries_map = {
            "ADM3_EN": "shapeName",
            "ADM3_PCODE": "shapeID",
        }
        still_missing = [c for c in required if c not in gdf.columns]
        rename = {geo_boundaries_map[req]: req for req in still_missing if geo_boundaries_map.get(req) in gdf.columns}
        if rename:
            gdf = gdf.rename(columns=rename)

        # Fallback: explicit mapping for Kenya IEBC wards shapefile column names
        iebc_col_map = {
            "ADM3_EN": "IEBC_WARDS",
            "ADM3_PCODE": "PCODE",
            "ADM2_EN": "FIRST_DIST",
            "ADM1_EN": "FIRST_PROV",
        }
        still_missing = [c for c in required if c not in gdf.columns]
        rename = {iebc_col_map[req]: req for req in still_missing if iebc_col_map.get(req) in gdf.columns}
        if rename:
            gdf = gdf.rename(columns=rename)

        # Fill any remaining missing admin columns with "Unknown"
        for col in ["ADM2_EN", "ADM1_EN"]:
            if col not in gdf.columns:
                gdf[col] = "Unknown"

    still_missing = [c for c in required if c not in gdf.columns]
    if still_missing:
        raise ValueError(
            f"ADMIN3 shapefile is missing required columns: {still_missing}. "
            f"Available columns: {gdf.columns.tolist()}"
        )

    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)  # assume WGS84 if no CRS is defined
    else:
        gdf = gdf.to_crs(epsg=4326)  # reproject to WGS84
    return gdf[required]


def assign_risk_level(
    ward_mean: float,
    global_mean: float,
    thresholds: Dict[str, Tuple[float, float]],
) -> str:
    """
    Classify a ward's risk based on how far its mean prediction exceeds
    the global average.

    Parameters
    ----------
    ward_mean : float
        Mean predicted loss ratio for this ward.
    global_mean : float
        Mean predicted loss ratio across all observations.
    thresholds : dict
        Risk level name → (lower_pct, upper_pct) relative to global mean.
    """
    if global_mean <= 0:
        return "Normal"
    pct_above = (ward_mean - global_mean) / global_mean
    pct_above = max(pct_above, 0.0)  # below-average → Normal

    for level, (lo, hi) in thresholds.items():
        if lo <= pct_above < hi:
            return level
    return "Critical"


def compute_ward_confidence(predictions: pd.Series) -> float:
    """
    Confidence score based on prediction agreement within a ward.

    Uses 1 - (std / cap). When all predictions agree (std=0), confidence is
    1.0.  When predictions are highly dispersed, confidence approaches 0.

    Cap is set to 0.5 (conservative upper bound for std of a [0,1] variable).
    """
    if len(predictions) <= 1:
        return 1.0
    std = predictions.std()
    confidence = max(0.0, 1.0 - (std / 0.5))
    return round(confidence, 4)


def compute_shap_values(
    model,
    X: pd.DataFrame,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Compute SHAP values for a set of predictions using TreeExplainer.

    Returns a DataFrame aligned to X with one column per feature.
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X[feature_names])
    return pd.DataFrame(shap_values, columns=feature_names, index=X.index)


def rasterize_ward_predictions(
    ward_geo: "gpd.GeoDataFrame",
    output_path: str,
    resolution: float = 0.01,
    risk_encoding: Optional[Dict[str, int]] = None,
) -> None:
    """
    Rasterize ward-level predictions to a 3-band GeoTIFF (EPSG:4326).

    Parameters
    ----------
    ward_geo : gpd.GeoDataFrame
        Ward GeoDataFrame with columns: risk_level, confidence, top_features.
    output_path : str
        Local file path for the output GeoTIFF.
    resolution : float
        Pixel size in degrees (default 0.01 ≈ 1.1 km at equator).
    risk_encoding : dict, optional
        Mapping from risk-level name to integer code.
        Defaults to Normal=0, Concerning=1, Critical=2.

    Bands
    -----
    1 : risk_level_encoded   (float32, nodata=-9999)
    2 : confidence           (float32, nodata=-9999)
    3 : top_feature_importance (float32, nodata=-9999) — mean |SHAP| of #1 feature
    """
    import rasterio
    from rasterio.features import rasterize as rio_rasterize
    from rasterio.transform import from_bounds

    if risk_encoding is None:
        risk_encoding = {"Normal": 0, "Concerning": 1, "Critical": 2}

    nodata = -9999.0

    minx, miny, maxx, maxy = ward_geo.total_bounds
    width = max(1, int(np.ceil((maxx - minx) / resolution)))
    height = max(1, int(np.ceil((maxy - miny) / resolution)))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    def _shapes(values):
        """Yield (geometry, float_value) pairs, skipping NaN."""
        for geom, val in zip(ward_geo.geometry, values):
            if pd.isna(val):
                continue
            yield geom, float(val)

    # Band 1: risk level encoded as 0 / 1 / 2
    risk_encoded = ward_geo["risk_level"].map(risk_encoding)
    band1 = rio_rasterize(
        _shapes(risk_encoded),
        out_shape=(height, width),
        transform=transform,
        fill=nodata,
        dtype="float32",
    )

    # Band 2: confidence
    band2 = rio_rasterize(
        _shapes(ward_geo["confidence"]),
        out_shape=(height, width),
        transform=transform,
        fill=nodata,
        dtype="float32",
    )

    # Band 3: importance of the top SHAP feature per ward
    top_importances = []
    for val in ward_geo["top_features"]:
        try:
            features = json.loads(val) if isinstance(val, str) else val
            top_importances.append(features[0]["importance"] if features else np.nan)
        except Exception:
            top_importances.append(np.nan)

    band3 = rio_rasterize(
        _shapes(pd.Series(top_importances, index=ward_geo.index)),
        out_shape=(height, width),
        transform=transform,
        fill=nodata,
        dtype="float32",
    )

    band_tags = [
        {"name": "risk_level_encoded", "encoding": json.dumps(risk_encoding)},
        {"name": "confidence"},
        {"name": "top_feature_importance"},
    ]

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype="float32",
        crs="EPSG:4326",
        transform=transform,
        nodata=nodata,
        compress="lzw",
    ) as dst:
        for band_idx, (band_data, tags) in enumerate(
            zip([band1, band2, band3], band_tags), start=1
        ):
            dst.write(band_data, band_idx)
            dst.update_tags(band_idx, **tags)


def aggregate_top_features(
    shap_df: pd.DataFrame,
    top_n: int = 5,
) -> List[Dict[str, float]]:
    """
    Return top_n features ranked by mean |SHAP| across rows.

    Returns list of dicts: [{"feature": "ndvi", "importance": 0.032}, ...]
    """
    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
    top = mean_abs_shap.head(top_n)
    return [{"feature": f, "importance": round(float(v), 6)} for f, v in top.items()]


def run_postprocess(
    predictions_s3_path: str,
    experiment_name: str,
    run_id: str,
    training_run_id: str,
    admin3_shapefile_path: str,
    label_column: str = "tlu_loss_ratio",
    prediction_column: str = "prediction",
    lat_column: str = "gps_latitude",
    lon_column: str = "gps_longitude",
    feature_names: Optional[List[str]] = None,
    top_n_features: int = 5,
    risk_thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
    geotiff_resolution: float = 0.01,
    output_s3_prefix: Optional[str] = None,
    granularity: str = "household",
    compute_shap: bool = True,
    training_label_mean: Optional[float] = None,
) -> Tuple[str, str, str]:
    """
    Postprocess model predictions by aggregating to ADMIN3 ward level.

    Can be called standalone or from a SageMaker Pipeline @step.

    Parameters
    ----------
    predictions_s3_path : str
        S3 URI to CSV with columns: lat, lon, prediction, and feature columns.
    experiment_name : str
        MLflow experiment name.
    run_id : str
        Parent MLflow run ID.
    training_run_id : str
        MLflow run ID where the trained model artifact lives.
    admin3_shapefile_path : str
        S3 URI or local path to ADMIN3 ward boundary shapefile/GeoJSON.
    label_column : str
        Target column name (used for reference, not required in predictions).
    prediction_column : str
        Column containing model predictions.
    lat_column, lon_column : str
        GPS coordinate column names.
    feature_names : list[str], optional
        Feature columns for SHAP. Defaults to the standard 35-feature set.
    top_n_features : int
        Number of top SHAP features to report per ward.
    risk_thresholds : dict, optional
        Override risk thresholds. Keys are level names, values are
        (lower_pct, upper_pct) tuples representing the fraction above
        the global average prediction.

    Returns
    -------
    geotiff_resolution : float
        GeoTIFF pixel size in degrees (default 0.01 ≈ 1.1 km at equator).
    output_s3_prefix : str, optional
        Explicit base S3 URI for output files. When None (default), outputs
        are written alongside the predictions file (existing behaviour).
    granularity : str
        "ward"      — input is already at ward level (skips spatial join and
                      GPS column requirement; ward_name column used directly).
        "household" — input is household-level with GPS coords (default,
                      existing behaviour).
    compute_shap : bool
        When False, skip SHAP computation (faster; top_features will be "[]").
        Default True.
    training_label_mean : float, optional
        When provided, use this value as the global mean for risk-level
        thresholds instead of computing it from the prediction distribution.
        Recommended for ward-level inference so thresholds are anchored to
        the training label distribution.

    Returns
    -------
    ward_csv_s3_path : str
        S3 path to ward-level predictions CSV.
    ward_geojson_s3_path : str
        S3 path to ward-level predictions GeoJSON.
    ward_geotiff_s3_path : str
        S3 path to 3-band GeoTIFF (risk_level / confidence / top_feature_importance).
    """
    import geopandas as gpd
    import mlflow
    from shapely.geometry import Point

    try:
        import sagemaker_mlflow  # noqa: F401
    except Exception:
        pass

    if feature_names is None:
        feature_names = [
            "soil", "ppt", "pdsi", "vpd", "ndvi", "lai", "lst",
            "soil_lag1", "soil_lag2", "soil_lag3",
            "ppt_lag1", "ppt_lag2", "ppt_lag3",
            "pdsi_lag1", "pdsi_lag2", "pdsi_lag3",
            "vpd_lag1", "vpd_lag2", "vpd_lag3",
            "ndvi_lag1", "ndvi_lag2", "ndvi_lag3",
            "lai_lag1", "lai_lag2", "lai_lag3",
            "lst_lag1", "lst_lag2", "lst_lag3",
            "month_sin", "month_cos", "hhid_tlu_enc",
        ]

    thresholds = risk_thresholds if risk_thresholds is not None else DEFAULT_RISK_THRESHOLDS

    mlflow.set_tracking_uri(TRACKING_SERVER_ARN)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_id=run_id):
        with mlflow.start_run(run_name="PostProcessing", nested=True):

            # ── 1. Load predictions ──────────────────────────────────────
            # Ward-level data may be parquet; household-level is CSV.
            if predictions_s3_path.endswith(".parquet"):
                pred_df = pd.read_parquet(predictions_s3_path)
            else:
                pred_df = pd.read_csv(predictions_s3_path)
            print(f"Loaded {len(pred_df)} predictions from {predictions_s3_path}")

            if granularity == "household":
                coord_cols = [lat_column, lon_column]
                missing_coords = [c for c in coord_cols if c not in pred_df.columns]
                if missing_coords:
                    raise ValueError(f"Missing GPS columns in predictions file: {missing_coords}")

            # Generate predictions inline if prediction column is absent
            available_features = [f for f in feature_names if f in pred_df.columns]
            if prediction_column not in pred_df.columns:
                print(f"No '{prediction_column}' column found — generating predictions from model")
                model_uri = f"runs:/{training_run_id}/model"
                inference_model = mlflow.sklearn.load_model(model_uri)
                pred_df[prediction_column] = inference_model.predict(
                    pred_df[available_features]
                )

            # Use training label mean when provided so risk thresholds are
            # anchored to the training distribution rather than this batch.
            if training_label_mean is not None:
                global_mean = float(training_label_mean)
                print(f"Using training_label_mean as global mean: {global_mean:.6f}")
            else:
                global_mean = float(pred_df[prediction_column].mean())
                print(f"Global mean prediction: {global_mean:.6f}")

            # ── 2. Load ADMIN3 boundaries ────────────────────────────────
            admin3 = load_admin3_boundaries(admin3_shapefile_path)
            print(f"Loaded {len(admin3)} ADMIN3 ward boundaries")

            # ── 3. Spatial join (household) or direct merge (ward) ───────
            if granularity == "ward":
                # Data is already at ward level — join on ward_name to get
                # ADM3 codes and geometry; no GPS columns needed.
                if "ward_name" not in pred_df.columns:
                    raise ValueError(
                        "granularity='ward' requires a 'ward_name' column in predictions."
                    )
                joined = pred_df.merge(
                    admin3.rename(columns={"ADM3_EN": "ward_name"})[
                        ["ward_name", "ADM3_PCODE", "ADM2_EN", "ADM1_EN"]
                    ],
                    on="ward_name",
                    how="left",
                )
                unmatched = int(joined["ADM3_PCODE"].isna().sum())
                if unmatched > 0:
                    print(f"Warning: {unmatched} ward(s) could not be matched to ADMIN3 boundaries")
                mlflow.log_metric("unmatched_wards", unmatched)
                mlflow.log_metric("total_wards", len(joined))
            else:
                # Household granularity — spatial join points to polygons
                geometry = [
                    Point(lon, lat)
                    for lon, lat in zip(pred_df[lon_column], pred_df[lat_column])
                ]
                points_gdf = gpd.GeoDataFrame(pred_df, geometry=geometry, crs="EPSG:4326")

                joined = gpd.sjoin(points_gdf, admin3, how="left", predicate="within")

                unmatched = int(joined["ADM3_PCODE"].isna().sum())
                if unmatched > 0:
                    print(f"Warning: {unmatched}/{len(joined)} points outside ward boundaries — using nearest ward")
                    unmatched_idx = joined[joined["ADM3_PCODE"].isna()].index
                    unmatched_pts = points_gdf.loc[unmatched_idx]
                    nearest = gpd.sjoin_nearest(
                        unmatched_pts, admin3, how="left", distance_col="_dist"
                    )
                    for col in ["ADM3_EN", "ADM3_PCODE", "ADM2_EN", "ADM1_EN"]:
                        joined.loc[unmatched_idx, col] = nearest[col].values

                mlflow.log_metric("unmatched_points", unmatched)
                mlflow.log_metric("total_points", len(joined))

            # ── 4. Compute SHAP values ───────────────────────────────────
            shap_df = None

            if compute_shap and available_features:
                try:
                    model_uri = f"runs:/{training_run_id}/model"
                    raw_model = mlflow.sklearn.load_model(model_uri)

                    shap_df = compute_shap_values(
                        raw_model, pred_df, available_features
                    )
                    shap_df["ADM3_PCODE"] = joined["ADM3_PCODE"].values
                    print(f"Computed SHAP values for {len(available_features)} features")
                except Exception as e:
                    print(f"SHAP computation skipped: {e}")
            elif not compute_shap:
                print("SHAP computation skipped (compute_shap=False)")

            # ── 5. Aggregate per ward ────────────────────────────────────
            ward_results = []
            for pcode, group in joined.groupby("ADM3_PCODE"):
                preds = group[prediction_column]
                mean_score = float(preds.mean())

                ward_info = {
                    "ADM3_PCODE": pcode,
                    "ADM3_EN": group["ADM3_EN"].iloc[0] if "ADM3_EN" in group.columns else group.get("ward_name", pd.Series([pcode])).iloc[0],
                    "ADM2_EN": group["ADM2_EN"].iloc[0],
                    "ADM1_EN": group["ADM1_EN"].iloc[0],
                    "mean_predicted_loss_ratio": round(mean_score, 6),
                    "median_predicted_loss_ratio": round(float(preds.median()), 6),
                    "max_predicted_loss_ratio": round(float(preds.max()), 6),
                    "n_observations": int(len(preds)),
                    "risk_level": assign_risk_level(mean_score, global_mean, thresholds),
                    "confidence": compute_ward_confidence(preds),
                }

                # Top SHAP features for this ward
                if shap_df is not None:
                    ward_shap = shap_df[shap_df["ADM3_PCODE"] == pcode][available_features]
                    if len(ward_shap) > 0:
                        ward_info["top_features"] = json.dumps(
                            aggregate_top_features(ward_shap, top_n=top_n_features)
                        )
                    else:
                        ward_info["top_features"] = "[]"
                elif "top_features" in group.columns:
                    # Pre-computed top features (ward-level inference path)
                    val = group["top_features"].iloc[0]
                    ward_info["top_features"] = val if pd.notna(val) else "[]"
                else:
                    ward_info["top_features"] = "[]"

                ward_results.append(ward_info)

            ward_df = pd.DataFrame(ward_results)

            # Merge geometry back for GeoJSON output
            ward_geo = admin3.merge(
                ward_df, on="ADM3_PCODE", how="inner", suffixes=("", "_agg")
            )
            for col in ["ADM3_EN_agg", "ADM2_EN_agg", "ADM1_EN_agg"]:
                if col in ward_geo.columns:
                    ward_geo = ward_geo.drop(columns=[col])

            print(f"Aggregated predictions for {len(ward_df)} wards")
            print(f"Risk distribution:\n{ward_df['risk_level'].value_counts().to_string()}")

            # ── 6. Log to MLflow ─────────────────────────────────────────
            mlflow.log_params({
                "n_wards_with_predictions": int(len(ward_df)),
                "top_n_features": top_n_features,
                "risk_thresholds": json.dumps(
                    {k: list(v) for k, v in thresholds.items()}
                ),
                "global_mean_prediction": round(global_mean, 6),
                "granularity": granularity,
                "compute_shap": compute_shap,
            })
            mlflow.log_metrics({
                "mean_ward_confidence": float(ward_df["confidence"].mean()),
                "pct_critical_risk": float(
                    (ward_df["risk_level"] == "Critical").mean()
                ),
                "pct_concerning_or_critical": float(
                    ward_df["risk_level"].isin(["Concerning", "Critical"]).mean()
                ),
            })

            with tempfile.TemporaryDirectory() as tmp_dir:
                csv_path = os.path.join(tmp_dir, "ward_predictions.csv")
                ward_df.to_csv(csv_path, index=False)
                mlflow.log_artifact(csv_path, artifact_path="postprocessing")

                geojson_path = os.path.join(tmp_dir, "ward_predictions.geojson")
                ward_geo.to_file(geojson_path, driver="GeoJSON")
                mlflow.log_artifact(geojson_path, artifact_path="postprocessing")

                geotiff_path = os.path.join(tmp_dir, "ward_predictions.tif")
                rasterize_ward_predictions(
                    ward_geo,
                    output_path=geotiff_path,
                    resolution=geotiff_resolution,
                )
                mlflow.log_artifact(geotiff_path, artifact_path="postprocessing")

                print("Logged MLflow artifacts:")
                print(" - postprocessing/ward_predictions.csv")
                print(" - postprocessing/ward_predictions.geojson")
                print(" - postprocessing/ward_predictions.tif")

            # ── 7. Write outputs to S3 ───────────────────────────────────
            import s3fs

            if output_s3_prefix is not None:
                base_dir = output_s3_prefix.rstrip("/")
            else:
                base_dir = predictions_s3_path.rsplit("/", 1)[0]

            csv_s3 = f"{base_dir}/ward_predictions.csv"
            geojson_s3 = f"{base_dir}/ward_predictions.geojson"
            geotiff_s3 = f"{base_dir}/ward_predictions.tif"

            ward_df.to_csv(csv_s3, index=False)

            # geopandas/fiona and rasterio may not support S3 URIs directly;
            # write to temp files then upload via s3fs
            fs = s3fs.S3FileSystem()

            with tempfile.NamedTemporaryFile(suffix=".geojson", delete=True) as tmp:
                ward_geo.to_file(tmp.name, driver="GeoJSON")
                fs.put(tmp.name, geojson_s3)

            with tempfile.NamedTemporaryFile(suffix=".tif", delete=True) as tmp:
                rasterize_ward_predictions(
                    ward_geo,
                    output_path=tmp.name,
                    resolution=geotiff_resolution,
                )
                fs.put(tmp.name, geotiff_s3)

            print(f"Ward CSV:     {csv_s3}")
            print(f"Ward GeoJSON: {geojson_s3}")
            print(f"Ward GeoTIFF: {geotiff_s3}")

            return csv_s3, geojson_s3, geotiff_s3
