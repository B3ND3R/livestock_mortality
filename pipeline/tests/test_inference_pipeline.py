"""
test_inference_pipeline.py

Validates the four key requirements of the ward-level inference pipeline:

  1. Predictions are made at the ward (admin 3) level based on features
     aggregated to ward level.
  2. Models are loaded from the expected S3 path.
  3. Per-timepoint .csv, .geojson, and .tif files are generated.
  4. GeoJSON follows the required schema (FeatureCollection / Feature /
     property names / top_features as an array of dicts).

Run from the pipeline/ directory:
    python test_inference_pipeline.py          # full integration test
    pytest test_inference_pipeline.py -v       # same tests via pytest
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import traceback
from typing import Any, Dict, List

# ── Constants ─────────────────────────────────────────────────────────────────

S3_BUCKET     = "amazon-sagemaker-575108933641-us-east-1-c422b90ce861"
MODEL_BASE    = "dzd-ayr06tncl712p3/5t7l23o0xvt99j/shared/lmr_example_models"
MODEL_PREFIX  = f"s3://{S3_BUCKET}/{MODEL_BASE}"
SEASON        = "biannual"

INPUT_PARQUET = (
    f"s3://{S3_BUCKET}/dzd-ayr06tncl712p3/5t7l23o0xvt99j/dev/data/inference/"
    "ward_features_2022-01_2023-12/ward_features_biannual.parquet"
)
OUTPUT_BASE = (
    f"s3://{S3_BUCKET}/dzd-ayr06tncl712p3/5t7l23o0xvt99j/dev/outputs/"
    f"inference-test-{int(time.time())}"
)

REQUIRED_MODEL_FILES = [
    "xgboost_model.joblib",
    "lgbm_model.joblib",
    "rf_model.joblib",
    "ridge_model.joblib",
    "ensemble_weights.json",
    "feature_names.json",
    "feature_scaler.joblib",
    "train_medians.json",
    "run_metadata.json",
]

REQUIRED_GEOJSON_PROPS = [
    "ADM3_EN",
    "pcode",
    "mean_predicted_loss_ratio",
    "median_predicted_loss_ratio",
    "max_predicted_loss_ratio",
    "confidence",
    "risk_level",
    "n_observations",
    "top_features",
]

VALID_RISK_LEVELS = {"Normal", "Concerning", "Critical"}

# ── Helpers ───────────────────────────────────────────────────────────────────

_PASS = "[PASS]"
_FAIL = "[FAIL]"


def _assert(condition: bool, msg: str) -> None:
    if not condition:
        raise AssertionError(msg)


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_model_files_exist_on_s3():
    """Requirement 2: models are at the expected S3 path."""
    import boto3

    s3 = boto3.client("s3")
    key_prefix = f"{MODEL_BASE}/{SEASON}"

    missing = []
    for filename in REQUIRED_MODEL_FILES:
        key = f"{key_prefix}/{filename}"
        try:
            s3.head_object(Bucket=S3_BUCKET, Key=key)
        except Exception:
            missing.append(key)

    _assert(
        not missing,
        f"Missing model files at s3://{S3_BUCKET}/{key_prefix}/: {missing}",
    )
    print(f"  All {len(REQUIRED_MODEL_FILES)} model files found at "
          f"s3://{S3_BUCKET}/{key_prefix}/")


def test_input_parquet_is_ward_aggregated():
    """
    Requirement 1: input parquet is pre-aggregated at the ward level.

    Checks that:
    - a 'ward_name' column exists (no lat/lon GPS columns)
    - each (ward_name, season, season_year) tuple is unique — i.e. already
      one row per ward per timepoint, not household-level
    """
    import pandas as pd

    df = pd.read_parquet(INPUT_PARQUET)
    print(f"  Input parquet shape: {df.shape}")
    print(f"  Columns (first 10): {df.columns.tolist()[:10]}")

    _assert("ward_name" in df.columns,
            "Input parquet must have a 'ward_name' column")

    # Must NOT have raw GPS columns (would indicate household-level data)
    gps_cols = [c for c in ["gps_latitude", "gps_longitude", "lat", "lon"]
                if c in df.columns]
    _assert(
        len(gps_cols) == 0,
        f"Input parquet should NOT have GPS coordinate columns for ward-level "
        f"inference, but found: {gps_cols}",
    )

    # Each (ward_name, season, season_year) must be unique
    key_cols = [c for c in ["ward_name", "season", "season_year"] if c in df.columns]
    n_rows = len(df)
    n_unique_keys = len(df[key_cols].drop_duplicates())
    _assert(
        n_rows == n_unique_keys,
        f"Input parquet has {n_rows} rows but only {n_unique_keys} unique "
        f"{key_cols} combinations — expected one row per ward per timepoint",
    )

    print(f"  {n_rows} rows, all unique by {key_cols} — ward-level aggregation confirmed")
    print(f"  Unique wards: {df['ward_name'].nunique()}")
    if "season" in df.columns:
        print(f"  Seasons present: {sorted(df['season'].unique())}")
    if "season_year" in df.columns:
        print(f"  Years present: {sorted(df['season_year'].unique())}")


def _validate_geojson_schema(geojson_path: str) -> None:
    """
    Requirement 4: GeoJSON must match the required schema.

    Validates:
    - top-level type == "FeatureCollection" with name == "ward_predictions"
    - each feature has type == "Feature"
    - all required property keys present
    - pcode is a string (renamed from ADM3_PCODE)
    - top_features is a list of dicts with 'feature' (str) and 'importance' (float)
    - confidence is a float in [0, 1]
    - risk_level is one of Normal / Concerning / Critical
    - geometry is not null and has 'type' and 'coordinates'
    - ADM2_EN and ADM1_EN are excluded (not in properties)
    """
    with open(geojson_path) as f:
        gj = json.load(f)

    _assert(gj.get("type") == "FeatureCollection",
            f"Expected type='FeatureCollection', got: {gj.get('type')}")
    _assert(gj.get("name") == "ward_predictions",
            f"Expected name='ward_predictions', got: {gj.get('name')}")
    _assert("features" in gj and isinstance(gj["features"], list),
            "GeoJSON must have a 'features' list")
    _assert(len(gj["features"]) > 0, "GeoJSON features list is empty")

    for i, feat in enumerate(gj["features"]):
        ctx = f"features[{i}]"
        _assert(feat.get("type") == "Feature",
                f"{ctx}: expected type='Feature', got: {feat.get('type')}")

        props = feat.get("properties", {})

        # All required properties present
        for key in REQUIRED_GEOJSON_PROPS:
            _assert(key in props,
                    f"{ctx}: missing required property '{key}'")

        # ADM2_EN and ADM1_EN must NOT appear in properties
        for excluded in ["ADM2_EN", "ADM1_EN"]:
            _assert(excluded not in props,
                    f"{ctx}: property '{excluded}' should be excluded from GeoJSON output")

        # pcode is a string
        _assert(isinstance(props["pcode"], str),
                f"{ctx}: 'pcode' must be a string, got {type(props['pcode'])}")

        # confidence is a float in [0, 1]
        conf = props["confidence"]
        _assert(isinstance(conf, (int, float)),
                f"{ctx}: 'confidence' must be numeric, got {type(conf)}")
        _assert(0.0 <= conf <= 1.0,
                f"{ctx}: 'confidence' must be in [0,1], got {conf}")

        # risk_level is valid
        _assert(props["risk_level"] in VALID_RISK_LEVELS,
                f"{ctx}: 'risk_level' must be one of {VALID_RISK_LEVELS}, "
                f"got '{props['risk_level']}'")

        # n_observations is a positive integer
        n_obs = props["n_observations"]
        _assert(isinstance(n_obs, int) and n_obs > 0,
                f"{ctx}: 'n_observations' must be a positive int, got {n_obs!r}")

        # top_features is a list (NOT a string) of dicts with 'feature'+'importance'
        tf = props["top_features"]
        _assert(isinstance(tf, list),
                f"{ctx}: 'top_features' must be a list (not a string), got {type(tf)}")
        for j, entry in enumerate(tf):
            _assert(isinstance(entry, dict),
                    f"{ctx} top_features[{j}]: expected dict, got {type(entry)}")
            _assert("feature" in entry and isinstance(entry["feature"], str),
                    f"{ctx} top_features[{j}]: must have 'feature' (str)")
            _assert("importance" in entry and isinstance(entry["importance"], float),
                    f"{ctx} top_features[{j}]: must have 'importance' (float)")

        # geometry present and has expected structure
        geom = feat.get("geometry")
        _assert(geom is not None, f"{ctx}: geometry must not be null")
        _assert("type" in geom, f"{ctx}: geometry must have 'type'")
        _assert("coordinates" in geom, f"{ctx}: geometry must have 'coordinates'")


def test_geojson_format_with_mock_data():
    """
    Requirement 4: unit test for _write_ward_geojson using mock ward data.

    Constructs a minimal ward GeoDataFrame and verifies the written GeoJSON
    passes schema validation without needing a full pipeline run.
    """
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Polygon
    from postprocess import _write_ward_geojson

    mock_top_features = json.dumps([
        {"feature": "ndvi_anom_roll6", "importance": 0.041200},
        {"feature": "lst_day_lag2",    "importance": 0.029800},
        {"feature": "pdsi_lag4",       "importance": 0.015400},
    ])

    ward_geo = gpd.GeoDataFrame(
        {
            "ADM3_EN":   ["Dukana", "Sagante/Jaldessa"],
            "ADM3_PCODE": ["KE0212", "KE1192"],
            "ADM2_EN":   ["Moyale", "Moyale"],
            "ADM1_EN":   ["Marsabit", "Marsabit"],
            "mean_predicted_loss_ratio":   [0.022695, 0.045100],
            "median_predicted_loss_ratio": [0.022643, 0.044200],
            "max_predicted_loss_ratio":    [0.022902, 0.048900],
            "confidence":   [0.9998, 0.9875],
            "risk_level":   ["Normal", "Normal"],
            "n_observations": [5, 5],
            "top_features": [mock_top_features, mock_top_features],
        },
        geometry=[
            Polygon([(37.1, 4.2), (37.3, 4.2), (37.3, 4.4), (37.1, 4.4)]),
            Polygon([(37.9, 2.3), (38.1, 2.3), (38.1, 2.5), (37.9, 2.5)]),
        ],
        crs="EPSG:4326",
    )

    with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as f:
        tmp_path = f.name

    try:
        _write_ward_geojson(ward_geo, tmp_path)
        _validate_geojson_schema(tmp_path)
        with open(tmp_path) as f:
            gj = json.load(f)
        print(f"  Mock GeoJSON written: {len(gj['features'])} features, schema valid")
    finally:
        os.unlink(tmp_path)


# ── Integration test ──────────────────────────────────────────────────────────

def test_end_to_end_pipeline():
    """
    Requirements 1–4: full end-to-end run of all three pipeline steps.

    Validates:
    - step 1 (preprocess): features selected / imputed at ward level
    - step 2 (inference):  ensemble predictions produced
    - step 3 (postprocess): per-timepoint .csv / .geojson / .tif written to S3
    - GeoJSON schema matches requirement 4
    - model artifacts are loaded from MODEL_PREFIX (requirement 2)
    """
    import boto3
    import pandas as pd
    import s3fs
    import shap as shap_lib
    import joblib

    from inference_config import WARD_BOUNDARIES_S3_KEY
    from inference_preprocess import run_inference_preprocess
    from inference import run_inference
    from postprocess import run_postprocess

    s3 = boto3.client("s3")
    fs = s3fs.S3FileSystem()

    # ── Step 1: Preprocess ────────────────────────────────────────────────────
    print(f"\n  Running preprocess (input: {INPUT_PARQUET})")
    features_s3, features_ridge_s3, metadata_s3, label_mean = run_inference_preprocess(
        input_data_s3_path=INPUT_PARQUET,
        model_s3_prefix=MODEL_PREFIX,
        season_scheme=SEASON,
        output_s3_base_uri=OUTPUT_BASE,
    )

    # Verify files landed on S3
    for uri in [features_s3, features_ridge_s3, metadata_s3]:
        bucket, key = uri[5:].split("/", 1)
        s3.head_object(Bucket=bucket, Key=key)  # raises if missing
    print(f"  Preprocess outputs present on S3")

    # Validate features are at ward level (no GPS cols)
    X_raw = pd.read_parquet(features_s3)
    meta  = pd.read_parquet(metadata_s3)
    gps_in_features = [c for c in ["lat", "lon", "gps_latitude", "gps_longitude"]
                       if c in X_raw.columns]
    _assert(len(gps_in_features) == 0,
            f"Preprocessed features contain GPS columns: {gps_in_features}")
    _assert("ward_name" in meta.columns,
            "Metadata must contain 'ward_name' column")
    print(f"  Feature matrix: {X_raw.shape} (no GPS columns — ward-level confirmed)")
    print(f"  Metadata wards: {meta['ward_name'].nunique()} unique wards")
    print(f"  label_mean from run_metadata: {label_mean:.6f}")

    # ── Step 2: Model inference ───────────────────────────────────────────────
    print(f"\n  Running inference (models from: {MODEL_PREFIX}/{SEASON}/)")
    predictions_s3 = run_inference(
        features_s3=features_s3,
        features_ridge_s3=features_ridge_s3,
        metadata_s3=metadata_s3,
        model_s3_prefix=MODEL_PREFIX,
        season_scheme=SEASON,
        output_s3_base_uri=OUTPUT_BASE,
    )

    pred_df = pd.read_parquet(predictions_s3)
    _assert("prediction" in pred_df.columns,
            f"Predictions parquet must have 'prediction' column. Columns: {pred_df.columns.tolist()}")
    _assert("ward_name" in pred_df.columns,
            f"Predictions parquet must have 'ward_name' column. Columns: {pred_df.columns.tolist()}")
    _assert(pred_df["prediction"].notna().all(),
            "All predictions must be non-null")
    print(f"  Predictions: {len(pred_df)} rows, "
          f"mean={pred_df['prediction'].mean():.4f}, "
          f"std={pred_df['prediction'].std():.4f}")

    # ── Compute SHAP and add top_features before postprocess ─────────────────
    print("\n  Computing SHAP values (XGBoost TreeExplainer)...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        fn_local  = os.path.join(tmp_dir, "feature_names.json")
        xgb_local = os.path.join(tmp_dir, "xgboost_model.joblib")
        key_prefix = f"{MODEL_BASE}/{SEASON}"
        s3.download_file(S3_BUCKET, f"{key_prefix}/feature_names.json", fn_local)
        s3.download_file(S3_BUCKET, f"{key_prefix}/xgboost_model.joblib", xgb_local)
        with open(fn_local) as f:
            feature_names = json.load(f)
        xgb_model = joblib.load(xgb_local)

    explainer = shap_lib.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X_raw.values)
    shap_df   = pd.DataFrame(shap_vals, columns=feature_names)
    shap_df["ward_name"] = meta["ward_name"].values

    ward_shap = (
        shap_df.groupby("ward_name")[feature_names]
        .apply(lambda g: g.abs().mean())
    )

    def _top5(row):
        top = row.nlargest(5)
        return json.dumps([
            {"feature": feat, "importance": round(float(v), 6)}
            for feat, v in top.items()
        ])

    ward_top = ward_shap.apply(_top5, axis=1).reset_index()
    ward_top.columns = ["ward_name", "top_features"]

    pred_df = pred_df.merge(ward_top, on="ward_name", how="left")
    pred_df["top_features"] = pred_df["top_features"].fillna("[]")
    pred_df.to_parquet(predictions_s3, index=False)
    print(f"  SHAP computed for {len(ward_top)} wards")

    # ── Step 3: Postprocess ───────────────────────────────────────────────────
    print("\n  Running postprocess...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        bounds_local = os.path.join(tmp_dir, "boundaries.geojson")
        s3.download_file(S3_BUCKET, WARD_BOUNDARIES_S3_KEY, bounds_local)

        base_dir, output_dirs = run_postprocess(
            predictions_s3_path=predictions_s3,
            experiment_name="lmr-ward-inference-test",
            run_id=None,
            training_run_id="",
            admin3_shapefile_path=bounds_local,
            prediction_column="prediction",
            feature_names=feature_names,
            top_n_features=5,
            output_s3_prefix=OUTPUT_BASE,
            granularity="ward",
            compute_shap=False,
            training_label_mean=label_mean,
            season_scheme=SEASON,
        )

    _assert(len(output_dirs) > 0, "Postprocess must produce at least one timepoint output")
    print(f"  Timepoints written: {len(output_dirs)}")

    # ── Validate per-timepoint output files ───────────────────────────────────
    # Requirement 3: .csv, .geojson, .tif must exist per timepoint
    # Requirement 4: GeoJSON must match schema
    print("\n  Validating per-timepoint outputs:")

    for tp_prefix in output_dirs:
        tp_label = tp_prefix.rstrip("/").split("/")[-1]
        bucket, key_base = tp_prefix[5:].split("/", 1)
        key_base = key_base.rstrip("/")

        for ext in ["ward_predictions.csv", "ward_predictions.geojson", "ward_predictions.tif"]:
            key = f"{key_base}/{ext}"
            s3.head_object(Bucket=bucket, Key=key)  # raises if missing

        # Download GeoJSON and validate schema
        with tempfile.NamedTemporaryFile(suffix=".geojson", delete=False) as f:
            tmp_geojson = f.name
        try:
            s3.download_file(bucket, f"{key_base}/ward_predictions.geojson", tmp_geojson)
            _validate_geojson_schema(tmp_geojson)
            with open(tmp_geojson) as f:
                gj = json.load(f)
            n_feat = len(gj["features"])
        finally:
            os.unlink(tmp_geojson)

        # Also check CSV has expected columns
        csv_uri = f"s3://{bucket}/{key_base}/ward_predictions.csv"
        csv_df = pd.read_csv(csv_uri)
        for col in ["ADM3_PCODE", "mean_predicted_loss_ratio", "risk_level", "confidence", "n_observations"]:
            _assert(col in csv_df.columns, f"CSV missing column '{col}'")

        print(f"    {tp_label}: {n_feat} wards, .csv/.geojson/.tif all present, GeoJSON schema valid")

    print(f"\n  All outputs written under: {OUTPUT_BASE}/")


# ── Test runner ───────────────────────────────────────────────────────────────

TESTS: List[Dict[str, Any]] = [
    {
        "name": "Req 2  — model files exist at expected S3 path",
        "fn":   test_model_files_exist_on_s3,
    },
    {
        "name": "Req 1  — input parquet is ward-aggregated (not household-level)",
        "fn":   test_input_parquet_is_ward_aggregated,
    },
    {
        "name": "Req 4  — GeoJSON schema valid (unit test with mock data)",
        "fn":   test_geojson_format_with_mock_data,
    },
    {
        "name": "Req 1–4 — end-to-end pipeline run and output validation",
        "fn":   test_end_to_end_pipeline,
    },
]


def main():
    passed = 0
    failed = 0

    print("=" * 70)
    print("LMR Inference Pipeline — Test Suite")
    print("=" * 70)

    for test in TESTS:
        print(f"\nTEST: {test['name']}")
        print("-" * 70)
        try:
            test["fn"]()
            print(f"{_PASS} {test['name']}")
            passed += 1
        except Exception as exc:
            print(f"{_FAIL} {test['name']}")
            print(f"       {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
