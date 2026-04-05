"""
Local end-to-end test of the inference pipeline steps.
Run from the pipeline/ directory:  python test_pipeline_local.py
"""

from inference_config import S3_BUCKET, MODEL_BASE_PREFIX, WARD_BOUNDARIES_S3_KEY

INPUT_PARQUET = (
    f"s3://{S3_BUCKET}/dzd-ayr06tncl712p3/5t7l23o0xvt99j/dev/data/inference/"
    "ward_features_2022-01_2023-12/ward_features_biannual.parquet"
)
MODEL_PREFIX = f"s3://{S3_BUCKET}/{MODEL_BASE_PREFIX}"
SEASON = "biannual"
OUTPUT_BASE = f"s3://{S3_BUCKET}/dzd-ayr06tncl712p3/5t7l23o0xvt99j/dev/outputs/inference-test"

# ── Step 1: Preprocess ──────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: InferencePreprocess")
print("=" * 60)

from inference_preprocess import run_inference_preprocess

features_s3, features_ridge_s3, metadata_s3, label_mean = run_inference_preprocess(
    input_data_s3_path=INPUT_PARQUET,
    model_s3_prefix=MODEL_PREFIX,
    season_scheme=SEASON,
    output_s3_base_uri=OUTPUT_BASE,
)
print(f"\nStep 1 outputs:")
print(f"  features:       {features_s3}")
print(f"  features_ridge: {features_ridge_s3}")
print(f"  metadata:       {metadata_s3}")
print(f"  label_mean:     {label_mean}")

# ── Step 2: Model Inference ─────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: ModelInference")
print("=" * 60)

from inference import run_inference

predictions_s3 = run_inference(
    features_s3=features_s3,
    features_ridge_s3=features_ridge_s3,
    metadata_s3=metadata_s3,
    model_s3_prefix=MODEL_PREFIX,
    season_scheme=SEASON,
    output_s3_base_uri=OUTPUT_BASE,
)
print(f"\nStep 2 output:")
print(f"  predictions: {predictions_s3}")

# ── Step 3: Postprocess ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: InferencePostprocess")
print("=" * 60)

import json
import joblib
import tempfile
import shap as shap_lib
import boto3
import pandas as pd
from postprocess import run_postprocess

s3 = boto3.client("s3")

def _parse_s3(uri):
    parts = uri[5:].split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""

with tempfile.TemporaryDirectory() as tmp_dir:
    import os

    # Download ward boundaries
    bounds_local = os.path.join(tmp_dir, "boundaries.geojson")
    s3.download_file(S3_BUCKET, WARD_BOUNDARIES_S3_KEY, bounds_local)
    admin3_local_path = bounds_local

    # Download feature names + XGBoost model for SHAP
    bucket, key_prefix = _parse_s3(f"{MODEL_PREFIX}/{SEASON}")
    fn_local = os.path.join(tmp_dir, "feature_names.json")
    xgb_local = os.path.join(tmp_dir, "xgboost_model.joblib")
    s3.download_file(bucket, f"{key_prefix}/feature_names.json", fn_local)
    s3.download_file(bucket, f"{key_prefix}/xgboost_model.joblib", xgb_local)
    with open(fn_local) as f:
        feature_names = json.load(f)
    xgb_model = joblib.load(xgb_local)

    # Load data
    X_raw = pd.read_parquet(features_s3)
    metadata_df = pd.read_parquet(metadata_s3)
    pred_df = pd.read_parquet(predictions_s3)

    # Compute SHAP using XGBoost TreeExplainer
    explainer = shap_lib.TreeExplainer(xgb_model)
    shap_vals = explainer.shap_values(X_raw.values)
    shap_df = pd.DataFrame(shap_vals, columns=feature_names)
    shap_df["ward_name"] = metadata_df["ward_name"].values

    ward_shap = (
        shap_df.groupby("ward_name")[feature_names]
        .apply(lambda g: g.abs().mean())
    )

    def _top_features(row, n=5):
        top = row.nlargest(n)
        return json.dumps([
            {"feature": feat, "importance": round(float(v), 6)}
            for feat, v in top.items()
        ])

    ward_top = ward_shap.apply(_top_features, axis=1).reset_index()
    ward_top.columns = ["ward_name", "top_features"]

    pred_df = pred_df.merge(ward_top, on="ward_name", how="left")
    pred_df["top_features"] = pred_df["top_features"].fillna("[]")
    pred_df.to_parquet(predictions_s3, index=False)
    print(f"SHAP top features computed for {len(ward_top)} wards")

    csv_s3, geojson_s3, geotiff_s3 = run_postprocess(
        predictions_s3_path=predictions_s3,
        experiment_name="lmr-ward-inference-test",
        run_id=None,
        training_run_id="",
        admin3_shapefile_path=admin3_local_path,
        prediction_column="prediction",
        feature_names=feature_names,
        top_n_features=5,
        output_s3_prefix=OUTPUT_BASE,
        granularity="ward",
        compute_shap=False,
        training_label_mean=label_mean,
    )

print(f"\nStep 3 outputs:")
print(f"  CSV:     {csv_s3}")
print(f"  GeoJSON: {geojson_s3}")
print(f"  GeoTIFF: {geotiff_s3}")

print("\n" + "=" * 60)
print("ALL STEPS COMPLETED SUCCESSFULLY")
print("=" * 60)
