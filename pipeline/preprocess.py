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
"""

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
    date_column: str = "month",              # column that identifies the time period
    min_train_months: int = 12,              # burn-in: min months before first validation fold
    step_months: int = 1,                    # how many months to advance per fold
    feature_names: list = None,
) -> tuple:
    """
    Preprocess data and produce expanding-window CV fold paths on S3.

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

            # ── Clean ─────────────────────────────────────────────────────────
            df = df.dropna(subset=[label_column])
            df = df.fillna(df.median(numeric_only=True))
            df = df.reset_index(drop=True)

            # ── Sort by time — critical for walk-forward CV ───────────────────
            # date_column should already be a period/month integer or datetime;
            # sorting ensures temporal order is preserved.
            df = df.sort_values(date_column).reset_index(drop=True)
            sorted_periods = df[date_column].unique()  # sorted unique time periods
            n_periods = len(sorted_periods)

            if n_periods <= min_train_months:
                raise ValueError(
                    f"Dataset has only {n_periods} unique periods in '{date_column}', "
                    f"but min_train_months={min_train_months}. "
                    "Reduce min_train_months or supply more data."
                )

            # ── Log dataset stats ─────────────────────────────────────────────
            mlflow.log_params({
                "raw_row_count":                  initial_shape[0],
                "raw_col_count":                  initial_shape[1],
                "missing_values_before_imputation": missing_before,
                "cleaned_row_count":              df.shape[0],
                "label_column":                   label_column,
                "date_column":                    date_column,
                "n_unique_periods":               n_periods,
                "min_train_months":               min_train_months,
                "step_months":                    step_months,
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

            # Last `step_months` periods = held-out test set
            test_periods  = sorted_periods[-(step_months):]
            train_val_periods = sorted_periods[:n_periods - step_months]

            test_df = df[df[date_column].isin(test_periods)].reset_index(drop=True)
            test_s3_path = _write_csv(test_df, output_prefix, "test", S3_BUCKET)
            print(f"Test set: {len(test_periods)} period(s), {len(test_df)} rows → {test_s3_path}")

            # Walk-forward folds over the remaining periods
            fold_index = 0
            val_end_idx = min_train_months  # exclusive upper bound for val period index

            while val_end_idx <= len(train_val_periods):
                val_period  = train_val_periods[val_end_idx - 1]           # single val period
                train_periods = train_val_periods[:val_end_idx - 1]        # everything before it

                if len(train_periods) < min_train_months:
                    # Not enough training data yet; advance and try again
                    val_end_idx += step_months
                    continue

                train_df = df[df[date_column].isin(train_periods)].reset_index(drop=True)
                val_df   = df[df[date_column] == val_period].reset_index(drop=True)

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

            mlflow.log_params({
                "n_cv_folds":       fold_index,
                "test_periods":     str(list(test_periods)),
                "test_rows":        len(test_df),
            })

            print(f"Created {fold_index} CV folds + 1 held-out test set.")

    return fold_paths, test_s3_path, experiment_name, run_id


# ── Helper ────────────────────────────────────────────────────────────────────

def _write_csv(df: "pd.DataFrame", prefix: str, name: str, bucket: str) -> str:
    """Write a DataFrame to S3 as CSV and return the S3 URI."""
    path = f"s3://{bucket}/{prefix}/{name}.csv"
    df.to_csv(path, index=False)
    return path