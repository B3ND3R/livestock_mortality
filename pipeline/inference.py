"""
inference.py

SageMaker Pipeline step 2 of 3: run ensemble inference using 4 pre-trained
models (XGBoost, LightGBM, Random Forest, Ridge).

Steps
-----
1. Load preprocessed_features.parquet (raw median-imputed) and
   preprocessed_features_ridge.parquet (Ridge-scaled) from S3
2. Download all 4 model joblibs + ensemble_weights.json from
   model_s3_prefix/{season_scheme}/
3. For each model: predict using X_ridge for Ridge, X_raw for others
4. Normalize ensemble weights: w_i / sum(w)  (handles zero-weight models)
5. Compute weighted average prediction
6. Merge with metadata, write predictions_with_metadata.parquet to S3
7. Return predictions S3 path
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict


MODEL_FILES = {
    "xgboost": "xgboost_model.joblib",
    "lgbm":    "lgbm_model.joblib",
    "rf":      "rf_model.joblib",
    "ridge":   "ridge_model.joblib",
}


def run_inference(
    features_s3: str,
    features_ridge_s3: str,
    metadata_s3: str,
    model_s3_prefix: str,
    season_scheme: str,
    output_s3_base_uri: str,
) -> str:
    """
    Run weighted ensemble inference and write predictions with metadata.

    Parameters
    ----------
    features_s3 : str
        S3 URI to preprocessed_features.parquet (raw median-imputed).
    features_ridge_s3 : str
        S3 URI to preprocessed_features_ridge.parquet (Ridge-scaled).
    metadata_s3 : str
        S3 URI to inference_metadata.parquet.
    model_s3_prefix : str
        S3 prefix containing one subfolder per season scheme.
    season_scheme : str
        One of "biannual", "quadseasonal", "monthly".
    output_s3_base_uri : str
        Base S3 URI where predictions_with_metadata.parquet will be written.

    Returns
    -------
    predictions_s3 : str
        S3 URI to predictions_with_metadata.parquet containing all metadata
        columns plus a "prediction" column with the ensemble output.
    """
    import boto3
    import joblib
    import numpy as np
    import pandas as pd

    scheme_prefix = _join_s3(model_s3_prefix, season_scheme)
    bucket, key_prefix = _parse_s3_uri(scheme_prefix)

    s3 = boto3.client("s3")

    with tempfile.TemporaryDirectory() as tmp:

        # ── 1. Load preprocessed features ────────────────────────────────────
        X_raw   = pd.read_parquet(features_s3)
        X_ridge = pd.read_parquet(features_ridge_s3)
        metadata_df = pd.read_parquet(metadata_s3)

        print(f"Loaded features: {X_raw.shape}, ridge: {X_ridge.shape}, metadata: {metadata_df.shape}")

        # ── 2. Download models and ensemble weights ───────────────────────────
        weights_path = os.path.join(tmp, "ensemble_weights.json")
        s3.download_file(bucket, f"{key_prefix}/ensemble_weights.json", weights_path)
        with open(weights_path) as f:
            raw_weights: Dict[str, float] = json.load(f)
        print(f"Raw ensemble weights: {raw_weights}")

        models = {}
        for model_name, filename in MODEL_FILES.items():
            local_path = os.path.join(tmp, filename)
            s3.download_file(bucket, f"{key_prefix}/{filename}", local_path)
            models[model_name] = joblib.load(local_path)
            print(f"Loaded {model_name} from s3://{bucket}/{key_prefix}/{filename}")

        # ── 3. Predict with each model ────────────────────────────────────────
        predictions: Dict[str, np.ndarray] = {}
        for model_name, model in models.items():
            X = X_ridge if model_name == "ridge" else X_raw
            preds = model.predict(X.values)
            predictions[model_name] = preds
            print(f"{model_name}: mean={preds.mean():.6f}, std={preds.std():.6f}")

        # ── 4. Normalize ensemble weights ─────────────────────────────────────
        # raw_weights may contain zero-weight models (e.g. monthly: rf=1.0, rest=0)
        weight_sum = sum(raw_weights.values())
        if weight_sum <= 0:
            raise ValueError(
                f"Sum of ensemble weights is {weight_sum}; cannot normalize. "
                f"Weights: {raw_weights}"
            )
        normalized_weights = {k: v / weight_sum for k, v in raw_weights.items()}
        print(f"Normalized weights: {normalized_weights}")

        # ── 5. Weighted average ───────────────────────────────────────────────
        ensemble_pred = np.zeros(len(X_raw), dtype=np.float64)
        for model_name, weight in normalized_weights.items():
            if weight > 0 and model_name in predictions:
                ensemble_pred += weight * predictions[model_name]

        print(f"Ensemble: mean={ensemble_pred.mean():.6f}, std={ensemble_pred.std():.6f}")

        # ── 6. Merge with metadata and write ─────────────────────────────────
        result_df = metadata_df.copy()
        result_df["prediction"] = ensemble_pred

        out_prefix = output_s3_base_uri.rstrip("/")
        predictions_s3 = f"{out_prefix}/predictions_with_metadata.parquet"
        result_df.to_parquet(predictions_s3, index=False)

        print(f"Predictions written to: {predictions_s3}")
        print(f"Output shape: {result_df.shape}")

    return predictions_s3


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_s3_uri(s3_uri: str):
    assert s3_uri.startswith("s3://"), f"Expected s3:// URI, got: {s3_uri}"
    parts = s3_uri[5:].split("/", 1)
    bucket = parts[0]
    key    = parts[1] if len(parts) > 1 else ""
    return bucket, key


def _join_s3(*parts: str) -> str:
    base = parts[0].rstrip("/")
    for p in parts[1:]:
        base = base + "/" + p.strip("/")
    return base
