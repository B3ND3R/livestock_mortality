"""
run_inference_all_schemes.py

Upserts the lmr-ward-inference-pipeline and starts one execution per
season scheme (biannual, quadseasonal, monthly) for the 2018-10 → 2020-12
feature window.
"""

import sagemaker
from sagemaker.workflow.execution_variables import ExecutionVariables
from sagemaker.workflow.function_step import step
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline

from inference_config import S3_BUCKET, MODEL_BASE_PREFIX, WARD_BOUNDARIES_S3_KEY

# ---------------------------------------------------------------------------
# Session / role
# ---------------------------------------------------------------------------
sagemaker_session = sagemaker.session.Session()
role = sagemaker.get_execution_role()

pipeline_name       = "lmr-ward-inference-pipeline"
instance_type       = ParameterString(name="InferenceInstanceType", default_value="ml.m5.xlarge")
tracking_server_arn = (
    "arn:aws:sagemaker:us-east-1:575108933641:"
    "mlflow-tracking-server/lmr-tracking-server-5t7l23o0xvt99j-chws71x3trpelj-dev"
)
experiment_name = "lmr-ward-inference"

# ---------------------------------------------------------------------------
# Pipeline parameters
# ---------------------------------------------------------------------------
FEATURE_WINDOW  = "2018-10_2020-12"
PERIOD_LABEL    = "2019-2020"
BASE_INPUT      = f"s3://{S3_BUCKET}/dzd-ayr06tncl712p3/5t7l23o0xvt99j/dev/data/inference/ward_features_{FEATURE_WINDOW}"
BASE_OUTPUT     = f"s3://{S3_BUCKET}/dzd-ayr06tncl712p3/5t7l23o0xvt99j/dev/outputs"
MODEL_S3_PREFIX = f"s3://{S3_BUCKET}/{MODEL_BASE_PREFIX}"

InputDataS3Path = ParameterString(name="InputDataS3Path", default_value=f"{BASE_INPUT}/ward_features_biannual.parquet")
ModelS3Prefix   = ParameterString(name="ModelS3Prefix",   default_value=MODEL_S3_PREFIX)
SeasonScheme    = ParameterString(name="SeasonScheme",    default_value="biannual")
OutputS3Prefix  = ParameterString(name="OutputS3Prefix",  default_value=f"{BASE_OUTPUT}/{PERIOD_LABEL}-biannual")

# ---------------------------------------------------------------------------
# Step definitions (mirror of inference_pipeline.ipynb)
# ---------------------------------------------------------------------------
REQUIREMENTS = "requirements.txt"

@step(name="InferencePreprocess", instance_type=instance_type, dependencies=REQUIREMENTS)
def inference_preprocess(input_data_s3_path, model_s3_prefix, season_scheme, output_s3_base_uri):
    from inference_preprocess import run_inference_preprocess
    return run_inference_preprocess(
        input_data_s3_path=input_data_s3_path,
        model_s3_prefix=model_s3_prefix,
        season_scheme=season_scheme,
        output_s3_base_uri=output_s3_base_uri,
    )

@step(name="ModelInference", instance_type=instance_type, dependencies=REQUIREMENTS)
def model_inference(features_s3, features_ridge_s3, metadata_s3, model_s3_prefix, season_scheme, output_s3_base_uri):
    from inference import run_inference
    return run_inference(
        features_s3=features_s3,
        features_ridge_s3=features_ridge_s3,
        metadata_s3=metadata_s3,
        model_s3_prefix=model_s3_prefix,
        season_scheme=season_scheme,
        output_s3_base_uri=output_s3_base_uri,
    )

@step(name="InferencePostprocess", instance_type=instance_type, dependencies=REQUIREMENTS)
def inference_postprocess(
    predictions_s3_path, features_s3, metadata_s3,
    model_s3_prefix, season_scheme, output_s3_prefix,
    training_label_mean, experiment_name, run_id,
):
    import boto3, json, os, tempfile, joblib, pandas as pd
    import shap as shap_lib
    from inference_config import S3_BUCKET, WARD_BOUNDARIES_S3_KEY
    from postprocess import run_postprocess

    s3 = boto3.client("s3")

    def _parse_s3(uri):
        parts = uri[5:].split("/", 1)
        return parts[0], parts[1] if len(parts) > 1 else ""

    with tempfile.TemporaryDirectory() as tmp:
        bounds_local = os.path.join(tmp, "boundaries.geojson")
        s3.download_file(S3_BUCKET, WARD_BOUNDARIES_S3_KEY, bounds_local)

        bucket, key_prefix = _parse_s3(f"{model_s3_prefix}/{season_scheme}")
        fn_local  = os.path.join(tmp, "feature_names.json")
        xgb_local = os.path.join(tmp, "xgboost_model.joblib")
        s3.download_file(bucket, f"{key_prefix}/feature_names.json", fn_local)
        s3.download_file(bucket, f"{key_prefix}/xgboost_model.joblib", xgb_local)
        with open(fn_local) as f:
            feature_names = json.load(f)
        xgb_model = joblib.load(xgb_local)

        X_raw       = pd.read_parquet(features_s3)
        metadata_df = pd.read_parquet(metadata_s3)
        pred_df     = pd.read_parquet(predictions_s3_path)

        try:
            explainer = shap_lib.TreeExplainer(xgb_model)
            shap_vals = explainer.shap_values(X_raw.values)
            shap_df   = pd.DataFrame(shap_vals, columns=feature_names)
            shap_df["ward_name"] = metadata_df["ward_name"].values
            ward_shap = shap_df.groupby("ward_name")[feature_names].apply(lambda g: g.abs().mean())

            def _top_features(row, n=5):
                top = row.nlargest(n)
                return json.dumps([{"feature": feat, "importance": round(float(v), 6)} for feat, v in top.items()])

            ward_top = ward_shap.apply(_top_features, axis=1).reset_index()
            ward_top.columns = ["ward_name", "top_features"]
            pred_df = pred_df.merge(ward_top, on="ward_name", how="left")
            pred_df["top_features"] = pred_df["top_features"].fillna("[]")
            pred_df.to_parquet(predictions_s3_path, index=False)
            print(f"SHAP top features computed for {len(ward_top)} wards")
        except Exception as e:
            print(f"SHAP computation failed: {e}")

        return run_postprocess(
            predictions_s3_path=predictions_s3_path,
            experiment_name=experiment_name,
            run_id=run_id,
            training_run_id="",
            admin3_shapefile_path=bounds_local,
            prediction_column="prediction",
            feature_names=feature_names,
            top_n_features=5,
            output_s3_prefix=output_s3_prefix,
            granularity="ward",
            compute_shap=False,
            training_label_mean=training_label_mean,
        )

# ---------------------------------------------------------------------------
# Wire DAG
# ---------------------------------------------------------------------------
preprocess_step  = inference_preprocess(
    input_data_s3_path=InputDataS3Path,
    model_s3_prefix=ModelS3Prefix,
    season_scheme=SeasonScheme,
    output_s3_base_uri=OutputS3Prefix,
)
inference_step   = model_inference(
    features_s3=preprocess_step[0],
    features_ridge_s3=preprocess_step[1],
    metadata_s3=preprocess_step[2],
    model_s3_prefix=ModelS3Prefix,
    season_scheme=SeasonScheme,
    output_s3_base_uri=OutputS3Prefix,
)
postprocess_step = inference_postprocess(
    predictions_s3_path=inference_step,
    features_s3=preprocess_step[0],
    metadata_s3=preprocess_step[2],
    model_s3_prefix=ModelS3Prefix,
    season_scheme=SeasonScheme,
    output_s3_prefix=OutputS3Prefix,
    training_label_mean=preprocess_step[3],
    experiment_name=experiment_name,
    run_id=ExecutionVariables.PIPELINE_EXECUTION_ID,
)

# ---------------------------------------------------------------------------
# Upsert pipeline
# ---------------------------------------------------------------------------
pipeline = Pipeline(
    name=pipeline_name,
    parameters=[instance_type, InputDataS3Path, ModelS3Prefix, SeasonScheme, OutputS3Prefix],
    steps=[postprocess_step],
    sagemaker_session=sagemaker_session,
)
pipeline.upsert(role_arn=role)
print(f"Pipeline '{pipeline_name}' upserted.")

# ---------------------------------------------------------------------------
# Start one execution per scheme
# ---------------------------------------------------------------------------
SCHEMES = ["biannual", "quadseasonal", "monthly"]
executions = []

for scheme in SCHEMES:
    params = {
        "InputDataS3Path": f"{BASE_INPUT}/ward_features_{scheme}.parquet",
        "ModelS3Prefix":   MODEL_S3_PREFIX,
        "SeasonScheme":    scheme,
        "OutputS3Prefix":  f"{BASE_OUTPUT}/{PERIOD_LABEL}-{scheme}",
    }
    exec_ = pipeline.start(parameters=params)
    executions.append((scheme, exec_))
    print(f"[{scheme}] Execution ARN: {exec_.arn}")

# ---------------------------------------------------------------------------
# Wait for all three
# ---------------------------------------------------------------------------
for scheme, exec_ in executions:
    print(f"Waiting for [{scheme}] ...")
    exec_.wait()
    desc = exec_.describe()
    status = desc["PipelineExecutionStatus"]
    print(f"[{scheme}] Status: {status}")
    if status != "Succeeded":
        raise RuntimeError(f"[{scheme}] pipeline execution failed with status: {status}")

print("\nAll three scheme executions complete.")
