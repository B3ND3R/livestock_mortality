"""
inference_preprocess.py

SageMaker Pipeline step 1 of 3: preprocess ward+season input data for
ensemble inference.

Steps
-----
1. Download feature_names.json, train_medians.json, run_metadata.json,
   and feature_scaler.joblib from model_s3_prefix/{season_scheme}/
2. Load input ward+season parquet from S3
3. Select and order columns to match feature_names.json
4. Impute NaNs using train_medians.json values
5. Apply feature_scaler.joblib to produce Ridge-scaled features
6. Write two parquets to S3:
     - preprocessed_features.parquet        (raw median-imputed, for XGBoost/LightGBM/RF)
     - preprocessed_features_ridge.parquet  (scaled, for Ridge)
7. Write metadata parquet (ward_name, season, season_year) to S3
8. Return (features_s3, features_ridge_s3, metadata_s3, label_mean)
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Tuple


def run_inference_preprocess(
    input_data_s3_path: str,
    model_s3_prefix: str,
    season_scheme: str,
    output_s3_base_uri: str,
) -> Tuple[str, str, str, float]:
    """
    Preprocess input ward+season data for ensemble inference.

    Parameters
    ----------
    input_data_s3_path : str
        S3 URI to pre-aggregated ward+season parquet.
        Expected columns: ward_name, season (or year+month for monthly),
        season_year, plus all feature columns.
    model_s3_prefix : str
        S3 prefix containing one subfolder per season scheme
        (e.g. "s3://bucket/path/to/lmr_example_models").
    season_scheme : str
        One of "biannual", "quadseasonal", "monthly".
    output_s3_base_uri : str
        Base S3 URI where preprocessed outputs will be written.

    Returns
    -------
    features_s3 : str
        S3 URI to preprocessed_features.parquet (raw median-imputed).
    features_ridge_s3 : str
        S3 URI to preprocessed_features_ridge.parquet (Ridge-scaled).
    metadata_s3 : str
        S3 URI to inference_metadata.parquet (ward_name, season, season_year).
    label_mean : float
        Training label mean from run_metadata.json — used by postprocessor
        for risk level thresholds.
    """
    import boto3
    import joblib
    import numpy as np
    import pandas as pd

    # ── Parse S3 prefix ──────────────────────────────────────────────────────
    scheme_prefix = _join_s3(model_s3_prefix, season_scheme)
    bucket, key_prefix = _parse_s3_uri(scheme_prefix)

    s3 = boto3.client("s3")

    with tempfile.TemporaryDirectory() as tmp:

        # ── 1. Download model artifacts ──────────────────────────────────────
        feature_names_path = os.path.join(tmp, "feature_names.json")
        train_medians_path = os.path.join(tmp, "train_medians.json")
        run_metadata_path  = os.path.join(tmp, "run_metadata.json")
        scaler_path        = os.path.join(tmp, "feature_scaler.joblib")

        for filename, local_path in [
            ("feature_names.json",    feature_names_path),
            ("train_medians.json",    train_medians_path),
            ("run_metadata.json",     run_metadata_path),
            ("feature_scaler.joblib", scaler_path),
        ]:
            s3.download_file(bucket, f"{key_prefix}/{filename}", local_path)
            print(f"Downloaded s3://{bucket}/{key_prefix}/{filename}")

        with open(feature_names_path) as f:
            feature_names = json.load(f)
        with open(train_medians_path) as f:
            train_medians = json.load(f)
        with open(run_metadata_path) as f:
            run_metadata = json.load(f)

        label_mean = float(run_metadata["label_mean"])
        print(f"label_mean from run_metadata: {label_mean}")
        print(f"Feature count: {len(feature_names)}")

        scaler = joblib.load(scaler_path)

        # ── 2. Load input data ───────────────────────────────────────────────
        df = pd.read_parquet(input_data_s3_path)
        print(f"Loaded input data: {df.shape} from {input_data_s3_path}")

        # ── 3. Extract metadata columns ──────────────────────────────────────
        meta_cols = _detect_metadata_columns(df, season_scheme)
        missing_meta = [c for c in meta_cols if c not in df.columns]
        if missing_meta:
            raise ValueError(
                f"Input parquet is missing metadata columns: {missing_meta}. "
                f"Available columns: {df.columns.tolist()}"
            )
        metadata_df = df[meta_cols].copy()

        # ── 4. Select and order feature columns ──────────────────────────────
        missing_features = [f for f in feature_names if f not in df.columns]
        if missing_features:
            raise ValueError(
                f"Input parquet is missing {len(missing_features)} feature "
                f"column(s): {missing_features[:10]}{'...' if len(missing_features) > 10 else ''}"
            )
        X = df[feature_names].copy()

        # ── 5. Impute NaNs using training medians ────────────────────────────
        nan_counts_before = X.isna().sum().sum()
        for feat, median_val in train_medians.items():
            if feat in X.columns:
                X[feat] = X[feat].fillna(median_val)
        # Any remaining NaNs (features not in train_medians) → 0
        remaining = X.isna().sum().sum()
        if remaining > 0:
            print(f"Warning: {remaining} NaN(s) remain after median imputation — filling with 0")
            X = X.fillna(0.0)
        print(f"Imputed {nan_counts_before} NaN(s) using training medians")

        # ── 6. Scale features for Ridge ──────────────────────────────────────
        X_scaled_arr = scaler.transform(X.values)
        X_ridge = pd.DataFrame(X_scaled_arr, columns=feature_names, index=X.index)

        # ── 7. Write outputs to S3 ────────────────────────────────────────────
        out_prefix = output_s3_base_uri.rstrip("/")
        features_s3      = f"{out_prefix}/preprocessed_features.parquet"
        features_ridge_s3 = f"{out_prefix}/preprocessed_features_ridge.parquet"
        metadata_s3      = f"{out_prefix}/inference_metadata.parquet"

        X.to_parquet(features_s3, index=False)
        X_ridge.to_parquet(features_ridge_s3, index=False)
        metadata_df.to_parquet(metadata_s3, index=False)

        print(f"Features (raw):   {features_s3}")
        print(f"Features (Ridge): {features_ridge_s3}")
        print(f"Metadata:         {metadata_s3}")

    return features_s3, features_ridge_s3, metadata_s3, label_mean


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_metadata_columns(df: "pd.DataFrame", season_scheme: str) -> list:
    """
    Return the metadata column names expected for this season scheme.

    Monthly data uses year+month columns; biannual/quadseasonal use season.
    """
    base = ["ward_name"]
    if season_scheme == "monthly":
        time_cols = ["year", "month"]
    else:
        time_cols = ["season"]
    time_cols += ["season_year"]
    return base + time_cols


def _parse_s3_uri(s3_uri: str):
    """Return (bucket, key) from an s3:// URI."""
    assert s3_uri.startswith("s3://"), f"Expected s3:// URI, got: {s3_uri}"
    parts = s3_uri[5:].split("/", 1)
    bucket = parts[0]
    key    = parts[1] if len(parts) > 1 else ""
    return bucket, key


def _join_s3(*parts: str) -> str:
    """Join S3 URI parts, preserving the s3:// prefix."""
    base = parts[0].rstrip("/")
    for p in parts[1:]:
        base = base + "/" + p.strip("/")
    return base
