"""
preprocess.py

Runs in the SageMaker Pipelines RemoteFunction step before training.

Reads a merged parquet from S3, applies cleaning/encoding checks, and writes
time-ordered expanding-window (walk-forward) cross-validation splits back to S3.

Fixes implemented
-----------------
1) Consistent imputation:
   - Train/val/test all use the SAME imputation values (training medians of the
     final/widest training window).
   - Previously, test used scaler.mean_ as a proxy, which changes missing-value
     behavior and can distort downstream predictions.

2) step_months implemented properly:
   - Validation windows are length = step_months periods (not just one period).

3) Fold indexing logic simplified and corrected:
   - Fold i uses:
       train = periods [0 : min_train_months + i*step_months]
       val   = next step_months periods
   - Last step_months periods are held out as the final test set.

Artifacts logged to MLflow
--------------------------
- preprocessing/feature_scaler.joblib
- preprocessing/train_medians.json
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Dict, List, Optional, Tuple


# ── Constants ────────────────────────────────────────────────────────────────
# Hard-coded MLflow tracking server ARN (per your request)
TRACKING_SERVER_ARN = (
    "arn:aws:sagemaker:us-east-1:575108933641:"
    "mlflow-tracking-server/lmr-tracking-server-5t7l23o0xvt99j-chws71x3trpelj-dev"
)


def run_preprocess(
    raw_data_s3_path: str,
    output_prefix: str,
    experiment_name: str,
    run_name: str,
    output_s3_base_uri: str,
    label_column: str = "tlu_loss_ratio",
    date_column: str = "month",
    min_train_months: int = 12,
    step_months: int = 1,
    feature_names: Optional[List[str]] = None,
) -> Tuple[List[Dict], str, str, str]:
    """
    Preprocess data and produce expanding-window CV fold paths on S3.

    Parameters
    ----------
    raw_data_s3_path : str
        S3 URI to merged parquet (must be readable by pandas with s3fs).
    output_prefix : str
        Key prefix under output_s3_base_uri where fold CSVs will be written.
    experiment_name : str
        MLflow experiment name.
    run_name : str
        MLflow parent run name (often the pipeline execution ID).
    output_s3_base_uri : str
        Base S3 URI like "s3://my-bucket/some/prefix" (no trailing slash preferred).
    label_column : str
        Target column.
    date_column : str
        Time ordering column (Period, datetime, int, etc).
    min_train_months : int
        Minimum number of periods in the initial training window.
    step_months : int
        Number of periods per validation window, and expansion step.

    Returns
    -------
    fold_paths : list[dict]
        Each dict has keys: fold_index, train, val, train_rows, val_rows, val_periods.
    test_s3_path : str
        S3 path to held-out final test set CSV.
    experiment_name : str
    run_id : str
        Parent MLflow run id.
    """
    import pandas as pd
    import joblib
    import mlflow

    # Ensure the ARN tracking URI scheme is registered (no-op if not installed).
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

    if step_months < 1:
        raise ValueError("step_months must be >= 1")
    if min_train_months < 1:
        raise ValueError("min_train_months must be >= 1")

    mlflow.set_tracking_uri(TRACKING_SERVER_ARN)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        with mlflow.start_run(run_name="DataPreprocessing", nested=True):

            # ---- Load ----
            try:
                df = pd.read_parquet(raw_data_s3_path)
            except Exception as e:
                raise ValueError(f"Could not read parquet file: {raw_data_s3_path}") from e

            required_cols = [date_column, label_column] + list(feature_names)
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Columns missing from data: {missing_cols}. "
                    f"Available: {df.columns.tolist()}"
                )

            df = df[[date_column, label_column] + list(feature_names)].copy()

            initial_shape = df.shape
            missing_before = int(df.isnull().sum().sum())

            # Drop missing labels globally (safe).
            df = df.dropna(subset=[label_column]).reset_index(drop=True)

            # Sort by time (critical).
            df = df.sort_values(date_column).reset_index(drop=True)

            # Unique periods in sorted order
            sorted_periods = df[date_column].dropna().unique()
            n_periods = len(sorted_periods)

            # We hold out the final step_months as test set.
            # Need enough periods for: initial train + at least one val window + test window
            min_required = min_train_months + step_months + step_months
            if n_periods < min_required:
                raise ValueError(
                    f"Dataset has only {n_periods} unique periods in '{date_column}', "
                    f"but needs at least {min_required} for "
                    f"min_train_months={min_train_months}, step_months={step_months} "
                    f"(train + val + test)."
                )

            mlflow.log_params({
                "raw_row_count": initial_shape[0],
                "raw_col_count": initial_shape[1],
                "missing_values_before_imputation": missing_before,
                "cleaned_row_count": df.shape[0],
                "label_column": label_column,
                "date_column": date_column,
                "n_unique_periods": int(n_periods),
                "min_train_months": int(min_train_months),
                "step_months": int(step_months),
                "raw_data_s3_path": raw_data_s3_path,
                "output_prefix": output_prefix,
                "output_s3_base_uri": output_s3_base_uri,
            })
            mlflow.log_metrics({
                "label_mean": float(df[label_column].mean()),
                "label_std": float(df[label_column].std()),
                "label_min": float(df[label_column].min()),
                "label_max": float(df[label_column].max()),
            })
            mlflow.log_input(
                mlflow.data.from_pandas(df, raw_data_s3_path, targets=label_column),
                context="DataPreprocessing",
            )

            # ---- Split periods into CV pool and test pool ----
            test_periods = list(sorted_periods[-step_months:])
            cv_periods = list(sorted_periods[:-step_months])

            test_df_raw = df[df[date_column].isin(test_periods)].reset_index(drop=True)

            # ---- Build expanding-window folds ----
            fold_paths: List[Dict] = []
            cols_to_save = [label_column] + list(feature_names)

            # final objects used for test/inference scaling (from widest training window)
            final_scaler = None
            final_train_medians = None

            fold_index = 0
            train_end = min_train_months  # number of periods in train window

            while True:
                val_start = train_end
                val_end = train_end + step_months
                if val_end > len(cv_periods):
                    break  # no more full validation windows

                train_periods = cv_periods[:train_end]
                val_periods = cv_periods[val_start:val_end]

                train_df_raw_fold = df[df[date_column].isin(train_periods)].reset_index(drop=True)
                val_df_raw_fold = df[df[date_column].isin(val_periods)].reset_index(drop=True)

                train_df, val_df, scaler, train_medians = _impute_and_scale(
                    train_df_raw=train_df_raw_fold,
                    val_df_raw=val_df_raw_fold,
                    feature_names=feature_names,
                )

                # Keep from the widest training window (last fold)
                final_scaler = scaler
                final_train_medians = train_medians

                train_path = _write_csv(
                    train_df[cols_to_save],
                    s3_base_uri=output_s3_base_uri,
                    prefix=output_prefix,
                    name=f"fold_{fold_index}/train",
                )
                val_path = _write_csv(
                    val_df[cols_to_save],
                    s3_base_uri=output_s3_base_uri,
                    prefix=output_prefix,
                    name=f"fold_{fold_index}/val",
                )

                fold_paths.append({
                    "fold_index": fold_index,
                    "train_periods_n": len(train_periods),
                    "val_periods_n": len(val_periods),
                    "val_periods": [str(p) for p in val_periods],
                    "train_rows": int(len(train_df)),
                    "val_rows": int(len(val_df)),
                    "train": train_path,
                    "val": val_path,
                })

                print(
                    f"Fold {fold_index}: train_periods={len(train_periods)} "
                    f"val_periods={len(val_periods)} "
                    f"train_rows={len(train_df)} val_rows={len(val_df)}"
                )

                fold_index += 1
                train_end += step_months  # expanding window

            if not fold_paths or final_scaler is None or final_train_medians is None:
                raise RuntimeError(
                    "No CV folds were produced. Check min_train_months/step_months vs. dataset size."
                )

            # ---- Transform test set using final fold's medians + scaler ----
            test_df = _apply_impute_and_scale(
                df_raw=test_df_raw,
                scaler=final_scaler,
                feature_names=feature_names,
                train_medians=final_train_medians,
            )
            test_s3_path = _write_csv(
                test_df[cols_to_save],
                s3_base_uri=output_s3_base_uri,
                prefix=output_prefix,
                name="test",
            )
            print(f"Test set: periods={list(map(str, test_periods))} rows={len(test_df)} → {test_s3_path}")

            # ---- Persist scaler + medians as MLflow artifacts ----
            with tempfile.TemporaryDirectory() as tmp_dir:
                scaler_path = os.path.join(tmp_dir, "feature_scaler.joblib")
                joblib.dump(final_scaler, scaler_path)
                mlflow.log_artifact(scaler_path, artifact_path="preprocessing")

                medians_path = os.path.join(tmp_dir, "train_medians.json")
                with open(medians_path, "w") as f:
                    json.dump({k: float(v) for k, v in final_train_medians.to_dict().items()}, f, indent=2)
                mlflow.log_artifact(medians_path, artifact_path="preprocessing")

                print("Logged MLflow artifacts:")
                print(" - preprocessing/feature_scaler.joblib")
                print(" - preprocessing/train_medians.json")

            mlflow.log_params({
                "n_cv_folds": int(len(fold_paths)),
                "test_periods": str([str(p) for p in test_periods]),
                "test_rows": int(len(test_df)),
            })

    return fold_paths, test_s3_path, experiment_name, run_id


def _impute_and_scale(
    train_df_raw: "pd.DataFrame",
    val_df_raw: "pd.DataFrame",
    feature_names: List[str],
) -> Tuple["pd.DataFrame", "pd.DataFrame", "StandardScaler", "pd.Series"]:
    """
    Fit median imputation + StandardScaler on training features only,
    then apply to train and val.

    Returns (train_out, val_out, scaler, train_medians).
    """
    from sklearn.preprocessing import StandardScaler

    # 1) Median imputation values from training only
    train_medians = train_df_raw[feature_names].median()

    # 2) Impute
    train_features = train_df_raw[feature_names].fillna(train_medians)
    val_features = val_df_raw[feature_names].fillna(train_medians)

    # 3) Scale (fit on train only)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    val_scaled = scaler.transform(val_features)

    # 4) Reassemble
    train_out = train_df_raw.copy()
    val_out = val_df_raw.copy()
    train_out[feature_names] = train_scaled
    val_out[feature_names] = val_scaled

    return train_out, val_out, scaler, train_medians


def _apply_impute_and_scale(
    df_raw: "pd.DataFrame",
    scaler: "StandardScaler",
    feature_names: List[str],
    train_medians: "pd.Series",
) -> "pd.DataFrame":
    """
    Apply training medians (not means) then transform with a pre-fitted StandardScaler.
    This keeps test/inference preprocessing consistent with training-time behavior.
    """
    out = df_raw.copy()
    out[feature_names] = out[feature_names].fillna(train_medians)
    out[feature_names] = scaler.transform(out[feature_names])
    return out


def _s3_join(base: str, *parts: str) -> str:
    """Join S3 URI parts safely: _s3_join("s3://b/p", "a", "c") -> "s3://b/p/a/c"."""
    base = base.rstrip("/")
    clean = [p.strip("/").replace("//", "/") for p in parts if p is not None and p != ""]
    return "/".join([base] + clean)


def _write_csv(df: "pd.DataFrame", s3_base_uri: str, prefix: str, name: str) -> str:
    """
    Write a DataFrame to S3 as CSV and return the S3 URI.
    Requires s3fs/fsspec support in the runtime.
    """
    path = _s3_join(s3_base_uri, prefix, f"{name}.csv")
    df.to_csv(path, index=False)
    return path
