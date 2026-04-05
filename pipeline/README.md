# Livestock Mortality Risk (LMR) Pipeline

## Overview

This directory contains two pipelines:

1. **Training pipeline** — trains a ward/season ensemble model (XGBoost, LightGBM, RF, Ridge) on historical IBLI survey data merged with Planetary Computer satellite features.
2. **Inference pipeline** — runs the trained model on new satellite data to produce ward-level risk maps (GeoJSON + GeoTIFF).

---

## Training Pipeline

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

### Create Training Dataset
```sh
cd pipeline
pixi run prepare_targets.py
```
Creates `data/target_data_pipeline.csv`

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

Once the ward feature parquets are in S3, run the SageMaker inference pipeline
from `inference_pipeline.ipynb`.

The pipeline has three `@step` functions:

| Step | Script | Input | Output |
|------|--------|-------|--------|
| `InferencePreprocess` | `inference_preprocess.py` | Ward feature parquet | Imputed features (raw + Ridge-scaled) |
| `ModelInference` | `inference.py` | Preprocessed features | `predictions_with_metadata.parquet` |
| `InferencePostprocess` | `postprocess.py` | Predictions | GeoJSON + GeoTIFF + CSV |

#### Running the pipeline

Open `inference_pipeline.ipynb` and set the parameters in the **Pipeline Parameters** cell:

| Parameter | Description | Example |
|-----------|-------------|---------|
| `InputDataS3Path` | S3 URI to the ward feature parquet from Stage 1 | `s3://.../ward_features_biannual.parquet` |
| `ModelS3Prefix` | S3 prefix containing `biannual/`, `quadseasonal/`, `monthly/` model folders | `s3://.../lmr_example_models` |
| `SeasonScheme` | Must match the parquet from Stage 1 | `biannual` |
| `OutputS3Prefix` | Where to write GeoJSON, GeoTIFF, and CSV results | `s3://.../inference-outputs` |

Then run all cells. The final cell starts the SageMaker Pipeline execution and waits for completion.

#### End-to-end example (biannual, 2023)

```sh
# Stage 1 — extract features
python inference_ward_feature_pipeline.py \
    --scheme biannual \
    --time-start 2022-07 \
    --time-end 2023-06 \
    --output-prefix dzd-.../dev/data/inference/ward_features_2023_biannual
```

Then in `inference_pipeline.ipynb`, set:
```python
InputDataS3Path = "s3://<bucket>/dzd-.../dev/data/inference/ward_features_2023_biannual/ward_features_biannual.parquet"
ModelS3Prefix   = "s3://<bucket>/dzd-.../shared/lmr_example_models"
SeasonScheme    = "biannual"
OutputS3Prefix  = "s3://<bucket>/dzd-.../dev/outputs/inference-outputs/2023-biannual"
```
