"""
Runs in the ProcessingStep before training. Reads a merged parquet from S3,
applies cleaning/encoding, and writes time-ordered cross-validation splits
back to S3 using an expanding-window strategy appropriate for time-series data.

CV strategy
-----------
Rather than a random holdout, we use expanding-window (walk-forward) CV:
  fold 1 : train = months [0 .. MIN_TRAIN_MONTHS-1],  val = month MIN_TRAIN_MONTHS
  fold 2 : train = months [0 .. MIN_TRAIN_MONTHS],     val = month MIN_TRAIN_MONTHS+1
  ...
  fold N : train = months [0 .. T-2],                  val = month T-1

The last fold's validation window is held back as a final test set.

A minimum training window (MIN_TRAIN_MONTHS) is enforced to ensure the model
has seen at least one full seasonal cycle and has meaningful lag-feature history
before any validation is attempted.

Imputation & scaling
--------------------
Median imputation and StandardScaler normalization are fit ONLY on each fold's
training split, then applied to the corresponding val/test splits. This prevents
any information from leaking across the temporal boundary. The scaler for the
final (widest) training window is persisted as an MLflow artifact so that it can
be applied identically at inference time.
"""

import os
import tempfile

# ── Constants ─────────────────────────────────────────────────────────────────
TRACKING_SERVER_ARN = (
    "arn:aws:sagemaker:us-east-1:575108933641:"
    "mlflow-tracking-server/lmr-tracking-server-5t7l23o0xvt99j-chws71x3trpelj-dev"
)
S3_BUCKET = (
    "amazon-sagemaker-575108933641-us-east-1-c422b90ce861"
    "/dzd-ayr06tncl712p3/5t7l23o0xvt99j/dev"
)


def run_preprocess(
    raw_data_s3_path: str,
    output_prefix: str,
    experiment_name: str,
    run_name: str,
    label_column: str = "tlu_loss_ratio",
    date_column: str = "month",          # column that identifies the time period
    min_train_months: int = 12,          # burn-in: min months before first validation fold
    step_months: int = 1,               # how many months to advance per fold
    feature_names: list = None,
) -> tuple:
    """
    Preprocess data and produce expanding-window CV fold paths on S3.

    Imputation and scaling are fit on each fold's training split only, then
    applied to the corresponding val/test splits to prevent data leakage.
    The scaler from the final (widest) fold is logged as an MLflow artifact.

    Returns
    -------
    fold_paths : list[dict]
        Each dict has keys 'train', 'val' with S3 CSV paths, plus 'fold_index'.
    test_s3_path : str
        S3 path to the held-out final test set (last fold's val window).
    experiment_name : str
    run_id : str
    """
    import mlflow
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.preprocessing import StandardScaler

    if feature_names is None:
        feature_names = [
            'soil', 'ppt', 'pdsi', 'vpd', 'ndvi', 'lai', 'lst',
            'soil_lag1', 'soil_lag2', 'soil_lag3',
            'ppt_lag1',  'ppt_lag2',  'ppt_lag3',
            'pdsi_lag1', 'pdsi_lag2', 'pdsi_lag3',
            'vpd_lag1',  'vpd_lag2',  'vpd_lag3',
            'ndvi_lag1', 'ndvi_lag2', 'ndvi_lag3',
            'lai_lag1',  'lai_lag2',  'lai_lag3',
            'lst_lag1',  'lst_lag2',  'lst_lag3',
            'month_sin', 'month_cos', 'hhid_tlu_enc',
        ]

    mlflow.set_tracking_uri(TRACKING_SERVER_ARN)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        with mlflow.start_run(run_name="DataPreprocessing", nested=True):

            # ── Load ──────────────────────────────────────────────────────────
            try:
                df = pd.read_parquet(raw_data_s3_path)
            except Exception as e:
                raise ValueError(
                    f"Could not read parquet file: {raw_data_s3_path}"
                ) from e

            required_cols = [date_column, label_column] + feature_names
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                raise ValueError(
                    f"Columns missing from data: {missing_cols}. "
                    f"Available: {df.columns.tolist()}"
                )

            df = df[[date_column, label_column] + feature_names].copy()

            # ── Validate ──────────────────────────────────────────────────────
            initial_shape = df.shape
            missing_before = int(df.isnull().sum().sum())

            # ── Drop rows where the label is missing — always safe to do globally
            # because we never use the label to compute imputation statistics.
            df = df.dropna(subset=[label_column])
            df = df.reset_index(drop=True)

            # ── Sort by time — critical for walk-forward CV ───────────────────
            # date_column should already be a period/month integer or datetime;
            # sorting ensures temporal order is preserved.
            df = df.sort_values(date_column).reset_index(drop=True)
            sorted_periods = df[date_column].unique()   # sorted unique time periods
            n_periods = len(sorted_periods)

            if n_periods <= min_train_months:
                raise ValueError(
                    f"Dataset has only {n_periods} unique periods in '{date_column}', "
                    f"but min_train_months={min_train_months}. "
                    "Reduce min_train_months or supply more data."
                )

            # ── Log dataset stats ─────────────────────────────────────────────
            mlflow.log_params({
                "raw_row_count":                    initial_shape[0],
                "raw_col_count":                    initial_shape[1],
                "missing_values_before_imputation": missing_before,
                "cleaned_row_count":                df.shape[0],
                "label_column":                     label_column,
                "date_column":                      date_column,
                "n_unique_periods":                 n_periods,
                "min_train_months":                 min_train_months,
                "step_months":                      step_months,
            })
            mlflow.log_metrics({
                "label_mean": float(df[label_column].mean()),
                "label_std":  float(df[label_column].std()),
                "label_min":  float(df[label_column].min()),
                "label_max":  float(df[label_column].max()),
            })
            mlflow.log_input(
                mlflow.data.from_pandas(df, raw_data_s3_path, targets=label_column),
                context="DataPreprocessing",
            )

            # ── Build expanding-window folds ──────────────────────────────────
            #
            # sorted_periods = [p0, p1, p2, ..., p_{T-1}]
            #
            # We treat the LAST fold's validation set as the held-out test set,
            # so we stop building folds one step before the end.
            #
            # fold i: train on periods [p0 .. p_{min_train_months - 1 + i*step - 1}]
            #         val   on periods [p_{min_train_months + i*step} ..
            #                           p_{min_train_months + i*step + step - 1}]
            #
            # The final period window becomes the test set.

            fold_paths = []

            # Last `step_months` periods = held-out test set (raw — will be
            # transformed using the final fold's fitted scaler below).
            test_periods      = sorted_periods[-(step_months):]
            train_val_periods = sorted_periods[:n_periods - step_months]

            test_df_raw = df[df[date_column].isin(test_periods)].reset_index(drop=True)

            # Walk-forward folds over the remaining periods
            fold_index    = 0
            val_end_idx   = min_train_months   # exclusive upper bound for val period index
            final_scaler  = None               # will hold the last fold's fitted scaler

            while val_end_idx <= len(train_val_periods):
                val_period    = train_val_periods[val_end_idx - 1]      # single val period
                train_periods = train_val_periods[:val_end_idx - 1]     # everything before it

                if len(train_periods) < min_train_months:
                    # Not enough training data yet; advance and try again
                    val_end_idx += step_months
                    continue

                train_df_raw = df[df[date_column].isin(train_periods)].reset_index(drop=True)
                val_df_raw   = df[df[date_column] == val_period].reset_index(drop=True)

                # ── Impute then scale — fit on train only ─────────────────────
                train_df, val_df, scaler = _impute_and_scale(
                    train_df_raw, val_df_raw, feature_names
                )

                # Keep the scaler from the widest training window (last fold)
                # so it can be used at inference time.
                final_scaler = scaler

                train_path = _write_csv(train_df, output_prefix, f"fold_{fold_index}/train", S3_BUCKET)
                val_path   = _write_csv(val_df,   output_prefix, f"fold_{fold_index}/val",   S3_BUCKET)

                fold_paths.append({
                    "fold_index":  fold_index,
                    "val_period":  str(val_period),
                    "train_rows":  len(train_df),
                    "val_rows":    len(val_df),
                    "train":       train_path,
                    "val":         val_path,
                })

                print(
                    f"Fold {fold_index}: train={len(train_periods)} periods "
                    f"({len(train_df)} rows), val={val_period} ({len(val_df)} rows)"
                )

                fold_index  += 1
                val_end_idx += step_months

            # ── Transform test set using the final fold's scaler ──────────────
            # The test set must be imputed and scaled with statistics derived
            # solely from the final training window — never from test data itself.
            if final_scaler is None:
                raise RuntimeError(
                    "No CV folds were produced. Check min_train_months vs. dataset size."
                )

            test_df = _apply_scaler(test_df_raw, final_scaler, feature_names)
            test_s3_path = _write_csv(test_df, output_prefix, "test", S3_BUCKET)
            print(
                f"Test set: {len(test_periods)} period(s), "
                f"{len(test_df)} rows → {test_s3_path}"
            )

            # ── Persist the final scaler as an MLflow artifact ────────────────
            with tempfile.TemporaryDirectory() as tmp_dir:
                scaler_path = os.path.join(tmp_dir, "feature_scaler.joblib")
                joblib.dump(final_scaler, scaler_path)
                mlflow.log_artifact(scaler_path, artifact_path="preprocessing")
                print(f"Scaler logged to MLflow artifact: preprocessing/feature_scaler.joblib")

            mlflow.log_params({
                "n_cv_folds":   fold_index,
                "test_periods": str(list(test_periods)),
                "test_rows":    len(test_df),
            })

            print(f"Created {fold_index} CV folds + 1 held-out test set.")

    return fold_paths, test_s3_path, experiment_name, run_id


# ── Imputation & scaling helpers ──────────────────────────────────────────────

def _impute_and_scale(
    train_df: "pd.DataFrame",
    val_df: "pd.DataFrame",
    feature_names: list,
) -> tuple:
    """
    Fit median imputation and StandardScaler on `train_df`, then apply both
    to `train_df` and `val_df`.

    Imputation is handled manually (rather than via sklearn Pipeline) so that
    the per-column medians can be inspected or logged independently if needed.

    Handles three edge cases that cause StandardScaler's divide-by-zero
    RuntimeWarnings:
      1. All-NaN column in the training split  → median() returns NaN →
         fillna() is a no-op → NaN reaches fit_transform().
         Fix: fall back to 0.0 for any column whose training median is NaN.
      2. Non-numeric column in feature_names   → median() silently skips it.
         Fix: restrict to numeric dtype before computing medians.
      3. Val/test column has NaNs in a position that was fully observed in
         train, so the training median exists but the val NaN survives a
         fillna on a different dtype.
         Fix: cast feature columns to float64 before imputation.

    Parameters
    ----------
    train_df, val_df : pd.DataFrame
        Raw splits for one fold (may contain NaNs in feature columns).
    feature_names : list[str]
        Columns to impute and scale; non-feature columns are passed through.

    Returns
    -------
    train_out, val_out : pd.DataFrame
        Imputed and scaled copies.
    scaler : StandardScaler
        Fitted scaler (needed to transform the held-out test set).
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    # Cast to float64 first so that dtype mismatches don't silently block fillna.
    train_features = train_df[feature_names].astype(np.float64)
    val_features   = val_df[feature_names].astype(np.float64)

    # 1. Fit medians on training features only.
    #    median() is already numeric-only after the cast above.
    train_medians = train_features.median()

    # 2. Guard: if a column is entirely NaN in training, its median is NaN and
    #    fillna() will be a no-op, passing NaN straight into StandardScaler.
    #    Fall back to 0.0 (post-scaling mean) for those columns and warn loudly.
    all_nan_cols = train_medians[train_medians.isna()].index.tolist()
    if all_nan_cols:
        import warnings
        warnings.warn(
            f"The following feature columns are entirely NaN in the training "
            f"split and will be filled with 0.0: {all_nan_cols}. "
            "Consider dropping or investigating these features.",
            RuntimeWarning,
            stacklevel=2,
        )
        train_medians = train_medians.fillna(0.0)

    # 3. Apply median fill to both splits using only training statistics.
    train_features = train_features.fillna(train_medians)
    val_features   = val_features.fillna(train_medians)

    # 4. Hard assertion: NaNs must be gone before we touch the scaler.
    #    This turns a silent corrupt-scaler bug into an immediate, clear error.
    remaining_train_nans = train_features.isna().sum().sum()
    remaining_val_nans   = val_features.isna().sum().sum()
    if remaining_train_nans or remaining_val_nans:
        bad_cols = (
            train_features.columns[train_features.isna().any()].tolist()
            + val_features.columns[val_features.isna().any()].tolist()
        )
        raise ValueError(
            f"NaNs remain after imputation in columns: {list(set(bad_cols))}. "
            "Imputation did not fully cover these columns — check for non-numeric "
            "dtypes or columns that are entirely NaN in both splits."
        )

    # 5. Fit scaler on imputed training features only.
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    val_scaled   = scaler.transform(val_features)

    # 6. Reconstruct DataFrames, preserving non-feature columns unchanged.
    train_out = train_df.copy()
    val_out   = val_df.copy()

    train_out[feature_names] = train_scaled
    val_out[feature_names]   = val_scaled

    return train_out, val_out, scaler


def _apply_scaler(
    df: "pd.DataFrame",
    scaler: "StandardScaler",
    feature_names: list,
) -> "pd.DataFrame":
    """
    Apply pre-fitted imputation and StandardScaler to an arbitrary split —
    used for the held-out test set.

    Because sklearn's StandardScaler does not store the imputation medians,
    we use the scaler's fitted mean_ as the fill value. For normally
    distributed features this is a close approximation; if exact median
    imputation at inference time is critical, persist the medians separately
    (e.g. as a JSON artifact alongside the scaler).

    Applies the same float64 cast and post-imputation NaN assertion as
    _impute_and_scale to prevent divide-by-zero inside StandardScaler.
    """
    import numpy as np

    out = df.copy()

    # Cast to float64 to match the dtype used during fitting.
    out[feature_names] = out[feature_names].astype(np.float64)

    # Fill NaNs using the training mean stored inside the fitted scaler.
    fill_values = dict(zip(feature_names, scaler.mean_))
    out[feature_names] = out[feature_names].fillna(fill_values)

    # Hard assertion before transform — same guard as _impute_and_scale.
    remaining_nans = out[feature_names].isna().sum().sum()
    if remaining_nans:
        bad_cols = out[feature_names].columns[out[feature_names].isna().any()].tolist()
        raise ValueError(
            f"NaNs remain in test set after imputation in columns: {bad_cols}. "
            "These columns may be entirely NaN in the test split."
        )

    out[feature_names] = scaler.transform(out[feature_names])
    return out


# ── S3 write helper ───────────────────────────────────────────────────────────

def _write_csv(df: "pd.DataFrame", prefix: str, name: str, bucket: str) -> str:
    """Write a DataFrame to S3 as CSV and return the S3 URI."""
    path = f"s3://{bucket}/{prefix}/{name}.csv"
    df.to_csv(path, index=False)
    return path