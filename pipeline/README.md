# Livestock Mortality Risk (LMR) Pipeline

## Overview

This directory contains two pipelines:

1. **Training pipeline** — trains a ward/season ensemble model (XGBoost, LightGBM, RF, Ridge) on historical IBLI survey data merged with Planetary Computer satellite features.
2. **Inference pipeline** — runs the trained model on new satellite data to produce ward-level risk maps (GeoJSON + GeoTIFF).

---
## EDA
### Prerequisites

#### Data Directory Structure
Assumes Kenya survey (IBLI) data is organized in the following directory structure:
```
./data
└── IBLIData_CSV_PublicZipped
    ├── HH_location_shifted.csv
    ├── IBLI_sales.csv
    ├── S0A Household Identification information.csv
    ├── S0B Comments.csv
    ├── S1 Household Information.csv
    ├── S10 Herd Migration and Satellite Camps.csv
    ...
    └── S9B Other Assistance.csv
```

## Training Pipeline
Run using `livestock_mortality/pipeline/sagemaker_pipelines_mlflow_updated.ipynb`. Calls modules to pre- and post-process data.

---

## Inference Pipeline

The inference pipeline has two stages: **feature extraction** and **model inference**.

### Stage 1 — Ward Feature Extraction

`inference_ward_feature_pipeline.py` extracts satellite features at ward level and produces pre-aggregated parquets for each season scheme. Run this whenever you want to generate predictions for a new time window.

```sh
# All three season schemes, default time range (2020-01 → 2024-12)
python inference_ward_feature_pipeline.py

# Single scheme for a specific window
python inference_ward_feature_pipeline.py \
    --scheme biannual \
    --time-start 2022-01 \
    --time-end 2023-12

# All Kenya wards (no Marsabit bounding box filter)
python inference_ward_feature_pipeline.py --no-bbox

# Denser spatial sampling within each ward (see note below)
python inference_ward_feature_pipeline.py --n-sample-points 25
```

Outputs are written to S3 under:
```
s3://<bucket>/dzd-.../dev/data/inference/ward_features_<start>_<end>/
  ward_features_biannual.parquet
  ward_features_quadseasonal.parquet
  ward_features_monthly.parquet
```

#### Spatial sampling approach

Training computed ward-level features by averaging satellite window means across
all household GPS points within each ward (from the IBLI survey). This pipeline
cannot use household GPS — it must produce features for **any ward**, including
those not in the survey.

The current approach samples a regular grid of N points within each ward polygon
(default: 9, a 3×3 grid) and averages their 20km-window means. This better
represents within-ward spatial heterogeneity than a single centroid, and
generalises to wards with no survey history.

Narrow or small polygons where no grid point falls inside automatically fall back
to the centroid. Increase `--n-sample-points` for larger or more heterogeneous
wards at the cost of proportionally longer runtime.

#### Longer-term improvement — pixel-in-polygon aggregation

The grid-sampling approach is still an approximation: the 20km window around
each sample point extends beyond the ward boundary and the number of points
inside the polygon varies by ward shape.

A more accurate approach is to aggregate all pixels whose centres fall *within*
the ward boundary directly, using a geopandas spatial join on the pixel lat/lon
columns already present in each parquet. This eliminates the window radius
entirely and gives a true within-ward mean for every variable. The main cost is
loading the full pixel parquet and running the spatial join per variable
(~1–2 minutes per variable on a `ml.m5.xlarge`). Key changes required in
`inference_ward_feature_pipeline.py`:

- Replace `extract_temporal` with a version that loads the pixel parquet,
  creates a GeoDataFrame from the `lat`/`lon` columns, spatial-joins to ward
  polygons with `gpd.sjoin`, then groups by `ward_name` and takes `.mean()`.
- Remove the `window_mean_2d` / `snap_to_grid` / `half_win` helpers.
- Remove `WINDOW_KM` and the sample-point machinery.

### Stage 2 — Model Inference (SageMaker Pipeline)

Once the ward feature parquets are in S3, run the SageMaker inference pipeline.
The recommended entry point for running all three schemes in one command is
`run_inference_all_schemes.py`. The notebook `inference_pipeline.ipynb` can also
be used to run a single scheme manually.

#### Season schemes

Stage 1 produces one parquet per scheme. Each scheme aggregates the underlying
monthly ward features differently:

| Scheme | Seasons per year | Season labels | Time columns in parquet |
|--------|-----------------|---------------|------------------------|
| `biannual` | 2 | `MAM` (Mar–May), `OND` (Oct–Dec) | `season`, `season_year` |
| `quadseasonal` | 4 | `LRLD`, `MAM`, `SRSD`, `OND` | `season`, `season_year` |
| `monthly` | 12 | Jan–Dec | `year`, `month` |

`season_year` follows the meteorological convention: January and February belong
to the previous year's season for SRSD/SRS_dry schemes.

#### SageMaker pipeline steps

| Step | Script | Input | Output |
|------|--------|-------|--------|
| `InferencePreprocess` | `inference_preprocess.py` | Ward feature parquet | Imputed features (raw + Ridge-scaled) + metadata parquet |
| `ModelInference` | `inference.py` | Preprocessed features | `predictions_with_metadata.parquet` |
| `InferencePostprocess` | `postprocess.py` | Predictions parquet | Per-timepoint GeoJSON + GeoTIFF + CSV |

All steps run on `ml.m5.xlarge` instances (configurable via `InferenceInstanceType`).

#### Ensemble model

Each scheme folder under `ModelS3Prefix` contains four independently trained
models plus a weight file:

```
lmr_example_models/<scheme>/
  xgboost_model.joblib
  lgbm_model.joblib
  rf_model.joblib
  ridge_model.joblib
  ensemble_weights.json      # e.g. {"xgboost": 0.4, "lgbm": 0.3, "rf": 0.2, "ridge": 0.1}
  feature_names.json
  feature_scaler.joblib      # StandardScaler applied to Ridge inputs only
  train_medians.json         # Per-feature medians used for NaN imputation
  run_metadata.json          # Includes label_mean for risk-level thresholds
```

At inference time the weights are normalized to sum to 1 and a weighted average
is taken across all four model predictions. Models with weight 0 are skipped.
Ridge receives Ridge-scaled features (`feature_scaler.joblib`); XGBoost, LightGBM
and RF receive raw median-imputed features.

`ModelInference` writes per-model predictions alongside the ensemble average
(`pred_xgboost`, `pred_lgbm`, `pred_rf`, `pred_ridge`) so that the postprocessor
can use ensemble member disagreement as the confidence signal (see below).

#### Outputs — per-timepoint subdirectories

`InferencePostprocess` splits predictions by timepoint and writes a separate set
of output files for each season or month under the `OutputS3Prefix`:

```
<OutputS3Prefix>/
  2019OND/
    ward_predictions.csv      # one row per ward: risk_level, confidence, top SHAP features
    ward_predictions.geojson  # same with ward geometries
    ward_predictions.tif      # 3-band GeoTIFF (see below)
  2020MAM/
    ward_predictions.csv / .geojson / .tif
  ...
```

For monthly scheme the subdirectory names are `<year><MonthAbbr>` (e.g. `2019Jan`);
for biannual/quadseasonal they are `<season_year><season>` (e.g. `2019OND`).

**GeoTIFF bands:**

| Band | Field | Type | No-data |
|------|-------|------|---------|
| 1 | `risk_level_encoded` | float32 — 0=Normal, 1=Concerning, 2=Critical | −9999 |
| 2 | `confidence` | float32 0–1 | −9999 |
| 3 | `top_feature_importance` | float32 — mean \|SHAP\| of #1 feature per ward | −9999 |

**Confidence score** is computed as `1 - (std / 0.5)`, clamped to [0, 1]:

- **Ward-level granularity** (this pipeline): confidence reflects *ensemble disagreement* — the spread of the four model predictions (`pred_xgboost`, `pred_lgbm`, `pred_rf`, `pred_ridge`) for that ward. Low spread → high confidence.
- **Household-level granularity**: confidence reflects *spatial consistency* — the spread of household-level predictions within the ward polygon. Low spread → high confidence.

The cap of 0.5 is a conservative upper bound for the std of a [0, 1]-bounded variable.

**Risk levels** are assigned relative to the training label mean (anchored via
`run_metadata.json`) so thresholds are consistent across time windows:

| Level | Condition |
|-------|-----------|
| Normal | ward mean ≤ 5% above training mean |
| Concerning | 5-10% above training mean |
| Critical | > 10% above training mean |

#### Running all three schemes

```sh
# Edit FEATURE_WINDOW and PERIOD_LABEL at the top of the script, then:
python run_inference_all_schemes.py
```

The script upserts the pipeline once and starts three parallel executions
(one per scheme), then waits for all three to complete.

#### End-to-end example (2019–2020)

```sh
# Stage 1 — set --time-start 3 months before first season of interest
python inference_ward_feature_pipeline.py \
    --time-start 2018-10 \
    --time-end 2020-12 \
    --skip-collections s1_vv s1_vh s2_red s2_nir s2_swir1
```

Stage 1 outputs (S3):
```
ward_features_2018-10_2020-12/ward_features_biannual.parquet
ward_features_2018-10_2020-12/ward_features_quadseasonal.parquet
ward_features_2018-10_2020-12/ward_features_monthly.parquet
```

Then set `FEATURE_WINDOW = "2018-10_2020-12"` and `PERIOD_LABEL = "2019-2020"` in
`run_inference_all_schemes.py` and run it. Outputs land at:
```
dev/outputs/2019-2020-biannual/2019OND/ward_predictions.{csv,geojson,tif}
dev/outputs/2019-2020-biannual/2020MAM/ward_predictions.{csv,geojson,tif}
dev/outputs/2019-2020-quadseasonal/2019LRLD/...
dev/outputs/2019-2020-monthly/2019Jan/...
dev/outputs/2019-2020-monthly/2019Feb/...
...
```

> **Note:** Skip `s1_vv`, `s1_vh` (~1–1.2 GB each) and `s2_red`, `s2_nir`,
> `s2_swir1` (~600 MB each) — these are not used by the 29 model features and
> will OOM the instance if loaded.
