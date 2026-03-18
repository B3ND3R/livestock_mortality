"""
Runs in the ProcessingStep before training. Reads raw CSV from S3, apply any
cleaning/encoding, select the target and feature columns, and write train/test
split CSVs back to /opt/ml/processing/train and /opt/ml/processing/test
(standard paths for root path in container filesystem).
"""

def run_preprocess(
    raw_data_s3_path: str,
    output_prefix: str,
    experiment_name: str,
    run_name: str,
    label_column: str = "tlu_loss",
    test_size: float = 0.2,
    feature_names = ['soil', 'ppt', 'pdsi', 'vpd', 'ndvi', 'lai', 'lst', 'soil_lag1', 'soil_lag2', 'soil_lag3',
                     'ppt_lag1', 'ppt_lag2', 'ppt_lag3', 'pdsi_lag1', 'pdsi_lag2', 'pdsi_lag3', 'vpd_lag1',
                     'vpd_lag2', 'vpd_lag3', 'ndvi_lag1', 'ndvi_lag2', 'ndvi_lag3', 'lai_lag1', 'lai_lag2',
                     'lai_lag3', 'lst_lag1', 'lst_lag2', 'lst_lag3', 'month_sin', 'month_cos', 'hhid_tlu_enc']
) -> tuple:
    import mlflow
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split

    tracking_server_arn = "arn:aws:sagemaker:us-east-1:575108933641:mlflow-tracking-server/lmr-tracking-server-5t7l23o0xvt99j-chws71x3trpelj-dev"
    mlflow.set_tracking_uri(tracking_server_arn)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        with mlflow.start_run(run_name="DataPreprocessing", nested=True):

            # --- Load ---
            try:
                df = pd.read_parquet(raw_data_s3_path)
                df = df[[label_column] + feature_names]
            except ValueError as e:
                raise ValueError(
                    f"Raw data must be a parquet file. Invalid file: {raw_data_s3_path}"
                ) from e

            # --- Validate ---
            if label_column not in df.columns:
                raise ValueError(f"Label column '{label_column}' not found. Columns: {df.columns.tolist()}")

            initial_shape = df.shape
            missing_before = df.isnull().sum().sum()

            # --- Clean ---
            df = df.dropna(subset=[label_column])           # drop rows with missing target
            df = df.fillna(df.median(numeric_only=True))    # impute remaining NaNs with median
            df = df.reset_index(drop=True)

            # --- Log dataset stats to MLflow ---
            param_dict = {
                "raw_row_count": initial_shape[0],
                "raw_col_count": initial_shape[1],
                "missing_values_before_imputation": int(missing_before),
                "cleaned_row_count": df.shape[0],
                "label_column": label_column,
                "test_size": test_size,
            }
            
            print(f"parameters: {param_dict}")
            mlflow.log_params(param_dict)

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

            # --- Split (no stratify — regression target) ---
            train_df, temp_df = train_test_split(df, test_size=test_size, random_state=88)
            val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=88)
            # 80/10/10 split if test_size = 20

            train_df = train_df.reset_index(drop=True)
            val_df   = val_df.reset_index(drop=True)
            test_df  = test_df.reset_index(drop=True)

            mlflow.log_params({
                "train_size": len(train_df),
                "val_size": len(val_df),
                "test_size_rows": len(test_df),
            })

            # --- Save splits to S3 ---
            bucket = 'amazon-sagemaker-575108933641-us-east-1-c422b90ce861/dzd-ayr06tncl712p3/5t7l23o0xvt99j/dev'
            train_s3_path = f"s3://{bucket}/{output_prefix}/train.csv"
            val_s3_path   = f"s3://{bucket}/{output_prefix}/val.csv"
            test_s3_path  = f"s3://{bucket}/{output_prefix}/test.csv"

            train_df.to_csv(train_s3_path, index=False)
            val_df.to_csv(val_s3_path,     index=False)
            test_df.to_csv(test_s3_path,   index=False)

    return train_s3_path, val_s3_path, test_s3_path, experiment_name, run_id