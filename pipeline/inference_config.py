"""
inference_config.py

Constants for the ward/season inference pipeline.
"""

S3_BUCKET = "amazon-sagemaker-575108933641-us-east-1-c422b90ce861"

# Base S3 prefix for model artifact folders (one subfolder per season scheme).
# Each subfolder contains: xgboost_model.joblib, lgbm_model.joblib,
# rf_model.joblib, ridge_model.joblib, feature_scaler.joblib,
# feature_names.json, train_medians.json, ensemble_weights.json, run_metadata.json
MODEL_BASE_PREFIX = (
    "dzd-ayr06tncl712p3/5t7l23o0xvt99j/shared/final_lmr_ward_results/inference_bundle"
)

# S3 key for the Kenya ADMIN3 ward boundary GeoJSON
WARD_BOUNDARIES_S3_KEY = (
    "dzd-ayr06tncl712p3/5t7l23o0xvt99j/shared/geoBoundaries-KEN-ADM3.geojson"
)

SEASON_SCHEMES = ["biannual", "quadseasonal", "monthly"]
