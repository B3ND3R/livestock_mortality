#!/usr/bin/env python3
"""
lmr_pipeline_script.py  —  v3
LMR Pipeline — SageMaker Processing Job entry point.

Key changes from v2:
  - CV years: 2008-2015 + 2019 (all_cv_years), test = 2020
  - min_train_years = 1 (expanding from fold 0)
  - survey_gap_years = [2016, 2017, 2018] (2015 has data)
  - mortality_thresholds = [0.05, 0.10, 0.15] — metrics at all three
  - Bootstrap 95% CI on Spearman per fold (1000 resamples)
  - Weighted fold selection score: 0.5×fold7 + 0.3×mature + 0.2×early
  - Test set evaluation on 2020 (separate from grid search)
  - MLflow local only (ARN not supported as tracking URI)
  - Feature phases updated: leaky ndvi_lt_* replaced with *_exp names,
    jrc_seasonality removed from WATER_BLOCK
  - Output prefix: lmr-pipeline-v2-full
"""

import sys, os, subprocess
from pathlib import Path

# ── Install dependencies ───────────────────────────────────────────────────────
print("Installing dependencies...")

subprocess.check_call(
    [sys.executable, "-m", "pip", "install",
     "numpy==1.26.4", "--upgrade", "--quiet", "--break-system-packages"],
    stderr=subprocess.DEVNULL
)
subprocess.check_call(
    [sys.executable, "-m", "pip", "install",
     "shapely", "geopandas",
     "--upgrade", "--force-reinstall", "--quiet", "--break-system-packages"],
    stderr=subprocess.DEVNULL
)
for pkg in [
    "pandas==2.0.3",
    "xgboost==1.7.6", "lightgbm", "optuna==3.6.1", "shap",
    "mlflow>=2.8,<3.0", "scipy", "pyarrow",
    "requests-auth-aws-sigv4", "joblib", "scikit-learn>=1.3",
]:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", pkg,
         "--quiet", "--break-system-packages", "--upgrade"],
        stderr=subprocess.DEVNULL
    )
print("Dependencies installed")

# ── Path setup ─────────────────────────────────────────────────────────────────
INPUT_DIR = Path("/opt/ml/processing/input/data")
if not INPUT_DIR.exists():
    INPUT_DIR = Path("/opt/ml/processing/input")
OUTPUT_DIR = Path("/opt/ml/processing/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRIOR_RESULTS_DIR = Path(os.environ.get(
    "LMR_PRIOR_RESULTS_DIR", "/opt/ml/processing/input/prior_results"))

os.chdir(INPUT_DIR)
LOG_DIR = OUTPUT_DIR / "lmrp_results"
LOG_DIR.mkdir(parents=True, exist_ok=True)
print(f"Input:         {INPUT_DIR}")
print(f"Prior results: {PRIOR_RESULTS_DIR} (exists={PRIOR_RESULTS_DIR.exists()})")
print(f"Output:        {OUTPUT_DIR}")

# ── Imports ────────────────────────────────────────────────────────────────────
import json, warnings, time
from collections import defaultdict

import numpy as np
import pandas as pd
import geopandas as gpd
import xgboost as xgb
import lightgbm as lgb
import optuna
import shap
import matplotlib
matplotlib.use("Agg")
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_squared_error, mean_absolute_error,
    roc_auc_score, f1_score, precision_score, recall_score,
)
from joblib import Parallel, delayed
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from mlflow.tracking import MlflowClient

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
print(f"Imports OK  |  numpy={np.__version__}  pandas={pd.__version__}")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    "pipeline_name":        "LMR",
    "experiment_name":      "marsabit_livestock_mortality_v3",
    "run_description":      "full_grid_v3_clean",

    "log_local":            True,
    "log_remote":           False,   # MLflow ARN not supported as URI — local only

    "features_path":        str(INPUT_DIR / "pc_features_engineered_v3.parquet"),
    "monthly_targets_path": str(INPUT_DIR / "target_data_pipeline.csv"),
    "seasonal_targets_path":str(INPUT_DIR / "target_data_seasonal.csv"),
    "gps_path":             str(INPUT_DIR / "HH_location_shifted.csv"),
    "ward_boundaries_path": str(INPUT_DIR / "geoBoundaries-KEN-ADM3.geojson"),
    "marsabit_bbox":        (36.0, 1.2, 39.0, 4.5),

    "observed_only":         False,
    "drop_negatives":        True,
    "drop_nan_ratio":        True,
    "drop_over1":            False,
    "drop_extreme_over10":   False,
    "cap_target_at_1":       False,
    "target_transform":      None,
    "sample_weight_over1":   1.0,
    "sample_weight_over10":  1.0,

    "target_col":            "tlu_loss_ratio",
    "species":               ["total", "camels", "cattle", "shoats"],

    "run_biannual":          True,
    "run_quadseasonal":      True,
    "run_monthly":           True,

    # ── CV structure (v3) ────────────────────────────────────────────────────
    # Folds 0-7: val years 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2019
    # Test: 2020 — never touched during grid search
    "train_years":      list(range(2008, 2016)) + [2019],
    "val_years":        [2019],
    "test_years":       [2020],
    "all_cv_years":     list(range(2008, 2016)) + [2019],
    "min_train_years":  1,
    "survey_gap_years": [2016, 2017, 2018],

    # ── Fold selection weights ───────────────────────────────────────────────
    # Fold 7 (val=2019): post-gap, most deployment-relevant
    # Mature folds (val >= 2012): stable, 4+ training years
    # Early folds (val 2010-2011): drought years, downweighted
    # Fold 0 (val=2009): excluded from selection, reported only
    "fold_weight_postgap":  0.5,   # fold 7 (val=2019)
    "fold_weight_mature":   0.3,   # folds val 2012-2015
    "fold_weight_early":    0.2,   # folds val 2010-2011

    # ── Signal thresholds ────────────────────────────────────────────────────
    # Metrics computed at all three — SME selects operationally appropriate one
    "mortality_thresholds": [0.05, 0.10, 0.15],
    "mortality_threshold":  0.05,  # kept for backward compat in some helpers

    # ── Bootstrap CI ─────────────────────────────────────────────────────────
    "bootstrap_ci":          True,
    "bootstrap_n_resamples": 1000,
    "bootstrap_ci_level":    0.95,

    "models_to_run":         ["xgboost", "rf", "lgbm", "ridge"],
    "ensemble_method":       "weighted_avg",
    "ensemble_weight_metric":"spearman_r",

    "tuning_enabled":        True,
    "tuning_method":         "bayesian",
    "tuning_n_trials_p1_p3": 15,
    "tuning_n_trials_p4_p7": 25,
    "tuning_n_trials":       20,
    "tuning_seasonal":       "once",
    "tuning_monthly":        "once",
    "tuning_min_rows_nested":20,
    "tuning_cv_inner":       3,

    "primary_metric":        "spearman_r",
    "drought_years":         [2010, 2011, 2016, 2017, 2020, 2021],
    "drought_threshold":     0.10,

    # ── Dynamic base selection ───────────────────────────────────────────────
    "dynamic_base_spearman_weight":     0.6,
    "dynamic_base_hitrate_weight":      0.2,
    "dynamic_base_falsetrigger_weight": 0.2,
    "dynamic_base_min_spearman":        0.25,
    "dynamic_base_use_importance":      True,
    "dynamic_base_top_n_features":      12,
    "dynamic_base_min_importance_pct":  0.02,

    "collinearity_threshold":  0.85,
    "shap_dependence_enabled": False,

    "phases_to_run": "all",
}

LOG_DIR = OUTPUT_DIR / "lmrp_results"
LOG_DIR.mkdir(parents=True, exist_ok=True)
print(f"Config loaded — v3 (clean features, expanding window climatology)")
print(f"  CV years: {CONFIG['all_cv_years']}")
print(f"  Test year: {CONFIG['test_years']}")
print(f"  Thresholds: {CONFIG['mortality_thresholds']}")

# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

MODEL_REGISTRY = {
    "xgboost": {
        "factory": lambda p: XGBRegressor(**p),
        "defaults": {
            "objective": "reg:squarederror", "max_depth": 4,
            "learning_rate": 0.05, "n_estimators": 1000,
            "min_child_weight": 2, "subsample": 0.8,
            "colsample_bytree": 0.5, "reg_lambda": 0.5,
            "tree_method": "hist", "early_stopping_rounds": 30,
            "verbosity": 0, "nthread": 8,
        },
        "param_space": {
            "max_depth":        ("suggest_int",   3, 8),
            "learning_rate":    ("suggest_float", 0.01, 0.2, True),
            "min_child_weight": ("suggest_int",   1, 10),
            "subsample":        ("suggest_float", 0.6, 1.0),
            "colsample_bytree": ("suggest_float", 0.4, 1.0),
            "reg_lambda":       ("suggest_float", 0.1, 5.0),
            "reg_alpha":        ("suggest_float", 0.0, 2.0),
        },
        "supports_early_stopping": True,
        "supports_sample_weight":  True,
    },
    "rf": {
        "factory": lambda p: RandomForestRegressor(**p),
        "defaults": {
            "n_estimators": 500, "max_features": "sqrt",
            "min_samples_leaf": 2, "n_jobs": 8, "random_state": 42,
        },
        "param_space": {
            "n_estimators":     ("suggest_int",         100, 1000),
            "max_depth":        ("suggest_int",         3, 20),
            "max_features":     ("suggest_categorical", ["sqrt","log2",0.5]),
            "min_samples_leaf": ("suggest_int",         1, 10),
            "min_samples_split":("suggest_int",         2, 20),
        },
        "supports_early_stopping": False,
        "supports_sample_weight":  True,
    },
    "lgbm": {
        "factory": lambda p: LGBMRegressor(**p),
        "defaults": {
            "n_estimators": 1000, "learning_rate": 0.05,
            "num_leaves": 31, "min_child_samples": 10,
            "subsample": 0.8, "colsample_bytree": 0.5,
            "reg_lambda": 0.5, "early_stopping_rounds": 30,
            "verbosity": -1, "n_jobs": 8, "random_state": 42,
        },
        "param_space": {
            "learning_rate":     ("suggest_float", 0.01, 0.2, True),
            "num_leaves":        ("suggest_int",   15, 127),
            "min_child_samples": ("suggest_int",   5, 50),
            "subsample":         ("suggest_float", 0.6, 1.0),
            "colsample_bytree":  ("suggest_float", 0.4, 1.0),
            "reg_lambda":        ("suggest_float", 0.1, 5.0),
            "reg_alpha":         ("suggest_float", 0.0, 2.0),
        },
        "supports_early_stopping": True,
        "supports_sample_weight":  True,
    },
    "ridge": {
        "factory": lambda p: Ridge(**p),
        "defaults": {"alpha": 1.0},
        "param_space": {"alpha": ("suggest_float", 0.001, 100.0, True)},
        "supports_early_stopping": False,
        "supports_sample_weight":  True,
    },
}

def build_model(model_name, params=None):
    cfg = MODEL_REGISTRY[model_name]
    p   = {**cfg["defaults"], **(params or {})}
    return cfg["factory"](p)

def get_optuna_params(model_name, trial):
    space  = MODEL_REGISTRY[model_name]["param_space"]
    result = {}
    for k, v in space.items():
        method = v[0]; args = v[1:]
        if method == "suggest_float" and len(args) == 3 and args[2] is True:
            result[k] = trial.suggest_float(k, args[0], args[1], log=True)
        elif method == "suggest_float":
            result[k] = trial.suggest_float(k, args[0], args[1])
        elif method == "suggest_int":
            result[k] = trial.suggest_int(k, args[0], args[1])
        elif method == "suggest_categorical":
            result[k] = trial.suggest_categorical(k, args[0])
    return result

print("Model registry loaded")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════
print("Loading features...")
feat_df = pd.read_parquet(CONFIG["features_path"])
feat_df["hhid"]  = feat_df["hhid"].astype(int)
feat_df["year"]  = feat_df["year"].astype(int)
feat_df["month"] = feat_df["month"].astype(int)
print(f"  Features: {feat_df.shape}")

print("Loading targets...")
mon_raw  = pd.read_csv(CONFIG["monthly_targets_path"])
seas_raw = pd.read_csv(CONFIG["seasonal_targets_path"])
mon_raw["hhid"]  = mon_raw["hhid"].astype(int)
mon_raw["year"]  = mon_raw["year"].astype(int)
mon_raw["month"] = mon_raw["month"].astype(int)
seas_raw["hhid"] = seas_raw["hhid"].astype(int)

print("Loading GPS...")
locations = pd.read_csv(CONFIG["gps_path"])
locations = locations.sort_values(["hhid","round"])
for col in ["gps_latitude","gps_longitude"]:
    locations[col] = locations.groupby("hhid")[col].transform(lambda x: x.ffill().bfill())
hh_gps = (locations[["hhid","gps_latitude","gps_longitude"]]
          .drop_duplicates("hhid").dropna().reset_index(drop=True))
hh_gps["hhid"] = hh_gps["hhid"].astype(int)

NON_FEAT = {
    "hhid","year","month","gps_latitude","gps_longitude",
    "ibli_date","ibli_dekad","season","data_observed","round",
    "season_start_year","tlu_loss_total","tlu_loss_ratio_total",
    "tlu_loss_camels","tlu_loss_ratio_camels",
    "tlu_loss_cattle","tlu_loss_ratio_cattle",
    "tlu_loss_shoats","tlu_loss_ratio_shoats",
    "tlu_loss","tlu_loss_ratio","sample_weight",
    "tlu_stock_total","tlu_stock_camels","tlu_stock_cattle","tlu_stock_shoats",
}
all_feat_cols = [c for c in feat_df.columns if c not in NON_FEAT]
print(f"  Available features: {len(all_feat_cols)}")

def clean_monthly(raw_df, observed_only=False):
    df = raw_df.copy()
    if observed_only:
        df = df[df["data_observed"] == 1]
    if "tlu_loss" in df.columns:
        nan_mask = df["tlu_loss_ratio"].isna() & df["tlu_loss"].notna() & (df["tlu_loss"] > 0)
        df = df[~nan_mask]
    df = df[df["tlu_loss_ratio"] >= 0]
    df["is_drought_year"] = df["year"].isin(CONFIG["drought_years"]).astype(int)
    df["sample_weight"]   = 1.0
    return df

mon = clean_monthly(mon_raw, CONFIG["observed_only"])
print(f"  Monthly cleaned: {len(mon)} rows")

# ── Ward assignment ────────────────────────────────────────────────────────────
print("Assigning wards...")
wards = gpd.read_file(CONFIG["ward_boundaries_path"])
bbox  = CONFIG["marsabit_bbox"]
wards = wards.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]].reset_index(drop=True).to_crs("EPSG:4326")

hh_gdf = gpd.GeoDataFrame(
    hh_gps,
    geometry=gpd.points_from_xy(hh_gps["gps_longitude"], hh_gps["gps_latitude"]),
    crs="EPSG:4326"
)
hh_ward = gpd.sjoin(hh_gdf, wards[["shapeName","shapeID","geometry"]],
                    how="left", predicate="within")
hh_ward = hh_ward[["hhid","shapeName","shapeID"]].rename(
    columns={"shapeName":"ward_name","shapeID":"ward_id"})

unassigned = hh_ward[hh_ward["ward_name"].isna()]["hhid"].tolist()
if unassigned:
    ward_centroids = wards.copy()
    ward_centroids["centroid"] = wards.geometry.centroid
    for _, row in hh_gdf[hh_gdf["hhid"].isin(unassigned)].iterrows():
        dists   = ward_centroids["centroid"].distance(row.geometry)
        nearest = wards.iloc[dists.idxmin()]
        hh_ward.loc[hh_ward["hhid"]==row["hhid"], "ward_name"] = nearest["shapeName"]
        hh_ward.loc[hh_ward["hhid"]==row["hhid"], "ward_id"]   = nearest["shapeID"]

feat_df = feat_df.merge(hh_ward[["hhid","ward_name"]], on="hhid", how="left")
mon     = mon.merge(hh_ward[["hhid","ward_name"]], on="hhid", how="left")
print(f"  {wards['shapeName'].nunique()} wards assigned")

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE PHASES  — v3: leaky ndvi_lt_* replaced with *_exp, jrc_seasonality removed
# ══════════════════════════════════════════════════════════════════════════════
PAPER_ANCILLARY = ["tci","jrc_occurrence","wc_builtup","wc_cropland","is_lrld"]
SME_CORE        = ["ndvi_250m","tci","ppt","lswi"]
MONTH_DUMMIES   = ["m_2","m_3","m_4","m_5","m_6","m_7","m_8","m_9","m_10","m_11","m_12"]
WATER_BLOCK     = ["ndwi","jrc_occurrence","wc_water"]          # jrc_seasonality removed
SOIL_BLOCK      = ["swvl1","swvl2","swvl3","swvl4","soil_composite","soil_shallow_deep","soil_composite_anom"]
SAR_BLOCK       = ["s1_vv","s1_vh","sar_rvi","sar_vv_vh_ratio"]
LAG_SUITE_FULL  = [
    "ndvi_250m_lag1","ndvi_250m_lag2","ndvi_250m_lag3",
    "ppt_lag1","ppt_lag2","ppt_lag3",
    "tci_lag1","tci_lag2","tci_lag3",
    "lst_day_lag1","lst_day_lag2","lst_day_lag3",
    "fpar_lag1","fpar_lag2","fpar_lag3",
    "vci_lag1","vci_lag2","vci_lag3",
    "vhi_lag1","vhi_lag2","vhi_lag3",
    "gpp_lag1","gpp_lag2","gpp_lag3",
]
ROLLING_SUITE = [
    "ndvi_250m_roll3_mean","ndvi_250m_roll3_std","evi_250m_roll3_mean",
    "lst_day_roll3_mean","lst_day_roll3_std",
    "ppt_roll3_sum","ppt_roll3_mean","ppt_roll6_sum","ppt_roll12_sum",
    "gpp_roll3_mean","gpp_roll6_mean","tci_roll3_mean",
    "vci_roll3_mean","vhi_roll3_mean","et_deficit_roll3_mean",
    "swvl1_roll3","swvl2_roll3","sar_rvi_roll3_mean",
]
YOY_SUITE = [
    "ndvi_250m_yoy_diff","ndvi_250m_yoy_ratio","evi_250m_yoy_diff",
    "ppt_yoy_diff","gpp_yoy_diff","lst_day_yoy_diff",
]
DERIVED_INDICES = [
    "nbr","bsi","swir_ratio","lue","lue_anomaly","gpp_fpar_decoupling",
    # ndvi_lt_* replaced with expanding window versions (clean)
    "ndvi_lt_mean_exp","ndvi_lt_std_exp","ndvi_lt_p10_exp","ndvi_lt_p90_exp","ndvi_lt_cv_exp",
    "ndvi_mean_mam_exp","ndvi_mean_ond_exp","ndvi_mean_jfas_exp",
    "ndvi_drought_year_count_expanding",
    "fire_detected","fire_cumulative_count","fire_count_12m","months_since_fire",
    "wc_trees","wc_shrubland","wc_grassland",
    "dem","dem_std","dem_range",
]
DYNAMIC_BASE = "DYNAMIC_BASE_PLACEHOLDER"

FEATURE_PHASES = {
    "p0a_ndvi_paper":   {"phase":0,"dynamic_base":None,"features":["ndvi_250m"]+PAPER_ANCILLARY},
    "p0b_lai_a_paper":  {"phase":0,"dynamic_base":None,"features":["lai"]+PAPER_ANCILLARY},
    "p0c_lai_p_paper":  {"phase":0,"dynamic_base":None,"features":["laih_mean","laiw_mean"]+PAPER_ANCILLARY},
    "p1a_sme_baseline": {"phase":1,"dynamic_base":None,"features":SME_CORE+MONTH_DUMMIES},
    "p1b_fpar":         {"phase":1,"dynamic_base":None,"features":SME_CORE+["fpar"]+MONTH_DUMMIES},
    "p1c_lai":          {"phase":1,"dynamic_base":None,"features":SME_CORE+["lai"]+MONTH_DUMMIES},
    "p1d_evi":          {"phase":1,"dynamic_base":None,"features":["evi_250m","tci","ppt","lswi"]+MONTH_DUMMIES},
    "p1e_osavi":        {"phase":1,"dynamic_base":None,"features":["osavi","tci","ppt","lswi"]+MONTH_DUMMIES},
    "p1f_sar":          {"phase":1,"dynamic_base":None,"features":SME_CORE+["s1_vv","s1_vh","sar_rvi"]+MONTH_DUMMIES},
    "p1g_vhi":          {"phase":1,"dynamic_base":None,"features":SME_CORE+["vhi"]+MONTH_DUMMIES},
    "p1h_gpp":          {"phase":1,"dynamic_base":None,"features":["gpp","tci","ppt","lswi"]+MONTH_DUMMIES},
    "p1i_s2":           {"phase":1,"dynamic_base":None,"features":["s2_ndvi","s2_ndwi","tci","ppt","lswi"]+MONTH_DUMMIES},
    "p1j_stress_composite":{"phase":1,"dynamic_base":None,"features":SME_CORE+[
        "et_deficit","et_fraction","months_since_rain","ppt_above_30",
        "drought_mild","drought_severe","drought_extreme","compound_stress",
        "soil_composite","soil_deficit","soil_cum_deficit_3m","soil_cum_deficit_6m",
        "months_since_fire"]+MONTH_DUMMIES},
    "p2a_anomaly_baseline":{"phase":2,"dynamic_base":None,"features":["ndvi_anom","lst_anom","ppt_anomaly","lswi"]+MONTH_DUMMIES},
    "p2b_anomaly_fpar":    {"phase":2,"dynamic_base":None,"features":["ndvi_anom","lst_anom","ppt_anomaly","lswi","fpar"]+MONTH_DUMMIES},
    "p2c_anomaly_vhi":     {"phase":2,"dynamic_base":None,"features":["ndvi_anom","lst_anom","ppt_anomaly","lswi","vhi"]+MONTH_DUMMIES},
    "p2d_soil_anomaly":    {"phase":2,"dynamic_base":None,"features":["ndvi_anom","lst_anom","ppt_anomaly","lswi","swvl1_anom","swvl2_anom","swvl3_anom","swvl4_anom"]+MONTH_DUMMIES},
    "p2e_gpp_anomaly":     {"phase":2,"dynamic_base":None,"features":["gpp_anomaly","gpp_deficit","lst_anom","ppt_anomaly","lswi"]+MONTH_DUMMIES},
    "p3a_lag1_ndvi":       {"phase":3,"dynamic_base":None,"features":SME_CORE+["fpar","ndvi_250m_lag1","lst_day_lag1","fpar_lag1","ppt_lag1"]+MONTH_DUMMIES},
    "p3b_lag3_ndvi":       {"phase":3,"dynamic_base":None,"features":SME_CORE+["fpar","ndvi_250m_lag3","lst_day_lag3","fpar_lag3","ppt_lag3"]+MONTH_DUMMIES},
    "p3c_lag1_evi":        {"phase":3,"dynamic_base":None,"features":["evi_250m","tci","ppt","lswi","fpar","evi_250m_lag1","lst_day_lag1","fpar_lag1","ppt_lag1"]+MONTH_DUMMIES},
    "p3d_lag1_lag3_ndvi":  {"phase":3,"dynamic_base":None,"features":SME_CORE+["fpar","ndvi_250m_lag1","lst_day_lag1","fpar_lag1","ppt_lag1","ndvi_250m_lag3","lst_day_lag3","fpar_lag3","ppt_lag3"]+MONTH_DUMMIES},
    "p3e_roll3":           {"phase":3,"dynamic_base":None,"features":SME_CORE+["ndvi_250m_roll3_mean","lst_day_roll3_mean","ppt_roll3_mean","tci_roll3_mean"]+MONTH_DUMMIES},
    "p3f_phenology":       {"phase":3,"dynamic_base":None,"features":SME_CORE+["ndvi_amplitude","peak_ndvi","green_months","sos_month","eos_month","season_length","season_length_anom"]+MONTH_DUMMIES},
    "p4a_water_ind":       {"phase":4,"dynamic_base":None,    "features":SME_CORE+WATER_BLOCK+MONTH_DUMMIES},
    "p4a_water_dyn":       {"phase":4,"dynamic_base":"phase3","features":[DYNAMIC_BASE]+WATER_BLOCK+MONTH_DUMMIES},
    "p4b_soil_ind":        {"phase":4,"dynamic_base":None,    "features":SME_CORE+SOIL_BLOCK+MONTH_DUMMIES},
    "p4b_soil_dyn":        {"phase":4,"dynamic_base":"phase3","features":[DYNAMIC_BASE]+SOIL_BLOCK+MONTH_DUMMIES},
    "p4c_sar_ind":         {"phase":4,"dynamic_base":None,    "features":SME_CORE+SAR_BLOCK+MONTH_DUMMIES},
    "p4c_sar_dyn":         {"phase":4,"dynamic_base":"phase3","features":[DYNAMIC_BASE]+SAR_BLOCK+MONTH_DUMMIES},
    "p4d_water_soil_ind":  {"phase":4,"dynamic_base":None,    "features":SME_CORE+WATER_BLOCK+SOIL_BLOCK+MONTH_DUMMIES},
    "p4d_water_soil_dyn":  {"phase":4,"dynamic_base":"phase3","features":[DYNAMIC_BASE]+WATER_BLOCK+SOIL_BLOCK+MONTH_DUMMIES},
    "p4e_all_env_ind":     {"phase":4,"dynamic_base":None,    "features":SME_CORE+WATER_BLOCK+SOIL_BLOCK+SAR_BLOCK+MONTH_DUMMIES},
    "p4e_all_env_dyn":     {"phase":4,"dynamic_base":"phase3","features":[DYNAMIC_BASE]+WATER_BLOCK+SOIL_BLOCK+SAR_BLOCK+MONTH_DUMMIES},
    "p5a_full_lags_dyn":   {"phase":5,"dynamic_base":"phase4","features":[DYNAMIC_BASE]+LAG_SUITE_FULL+MONTH_DUMMIES},
    "p5b_full_lags_sar_dyn":{"phase":5,"dynamic_base":"phase4","features":[DYNAMIC_BASE]+LAG_SUITE_FULL+SAR_BLOCK+MONTH_DUMMIES},
    "p6a_rolling_dyn":     {"phase":6,"dynamic_base":"phase5","features":[DYNAMIC_BASE]+ROLLING_SUITE+MONTH_DUMMIES},
    "p6b_rolling_yoy_dyn": {"phase":6,"dynamic_base":"phase5","features":[DYNAMIC_BASE]+ROLLING_SUITE+YOY_SUITE+MONTH_DUMMIES},
    "p7a_derived_dyn":     {"phase":7,"dynamic_base":"phase6","features":[DYNAMIC_BASE]+DERIVED_INDICES+MONTH_DUMMIES},
    "p7b_derived_no_terrain_dyn":{"phase":7,"dynamic_base":"phase6","features":[DYNAMIC_BASE]+[f for f in DERIVED_INDICES if f not in ["dem","dem_std","dem_range"]]+MONTH_DUMMIES},
    "p_full":              {"phase":99,"dynamic_base":None,"features":None},
}

def resolve_phase_cols_static(phase_cfg, available):
    if phase_cfg["features"] is None:
        return list(available)
    if DYNAMIC_BASE in phase_cfg["features"]:
        return []
    seen    = set()
    deduped = [f for f in phase_cfg["features"] if not (f in seen or seen.add(f))]
    return [f for f in deduped if f in available]

phase_cols_map = {}
for name, cfg in FEATURE_PHASES.items():
    if cfg["dynamic_base"] is None:
        phase_cols_map[name] = resolve_phase_cols_static(cfg, all_feat_cols)
    else:
        phase_cols_map[name] = []

print("Feature phases loaded")

# ══════════════════════════════════════════════════════════════════════════════
# DYNAMIC BASE RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════
def get_best_phase_features(phase_number, species, results_df, importance_df, available):
    fallback     = [f for f in ["ndvi_250m","tci","ppt","lswi"] if f in available]
    top_n        = CONFIG["dynamic_base_top_n_features"]
    min_imp_pct  = CONFIG["dynamic_base_min_importance_pct"]
    min_spearman = CONFIG["dynamic_base_min_spearman"]
    w_sp = CONFIG["dynamic_base_spearman_weight"]
    w_hr = CONFIG["dynamic_base_hitrate_weight"]
    w_ft = CONFIG["dynamic_base_falsetrigger_weight"]

    if results_df is None or len(results_df) == 0:
        return fallback, "SME_CORE_fallback", 0.0, True

    phase_results = results_df[results_df["phase"].str.startswith(f"p{phase_number}")].copy()
    if phase_results.empty:
        return fallback, "SME_CORE_fallback", 0.0, True

    hr_col = phase_results["hit_rate"].fillna(0) if "hit_rate" in phase_results.columns else pd.Series(0, index=phase_results.index)
    ft_col = phase_results["false_trigger_rate"].fillna(1) if "false_trigger_rate" in phase_results.columns else pd.Series(1, index=phase_results.index)

    phase_results["selection_score"] = (
        w_sp * phase_results["spearman_r"].fillna(0).clip(lower=0) +
        w_hr * hr_col.clip(lower=0) +
        w_ft * (1 - ft_col.clip(0, 1))
    )

    species_results = phase_results[phase_results["species"] == species]
    if species_results.empty:
        species_results = phase_results

    best_phase_name  = species_results.groupby("phase")["selection_score"].mean().idxmax()
    best_phase_score = float(species_results.groupby("phase")["spearman_r"].mean().max())
    print(f"    Phase {phase_number} best for {species}: {best_phase_name} (Spearman={best_phase_score:.3f})")

    if CONFIG["dynamic_base_use_importance"] and importance_df is not None and len(importance_df) > 0:
        imp = importance_df[(importance_df["phase"] == best_phase_name) & (importance_df["species"] == species)]
        if imp.empty:
            imp = importance_df[importance_df["phase"] == best_phase_name]
        if not imp.empty:
            feat_imp = imp.groupby("feature")["importance"].mean().sort_values(ascending=False)
            max_imp  = feat_imp.max()
            if max_imp > 0:
                top_by_n   = set(feat_imp.head(top_n).index)
                top_by_pct = set(feat_imp[feat_imp >= max_imp * min_imp_pct].index)
                best_cols  = [f for f in (top_by_n | top_by_pct) if f in available]
                if best_cols:
                    print(f"    Importance-based: {len(best_cols)} features")
                    return best_cols, best_phase_name, best_phase_score, False

    if best_phase_score < min_spearman:
        return fallback, "SME_CORE_fallback", best_phase_score, True

    best_cols = [f for f in phase_cols_map.get(best_phase_name, []) if f in available]
    if not best_cols:
        return fallback, "SME_CORE_fallback", best_phase_score, True

    return best_cols, best_phase_name, best_phase_score, False


def resolve_phase_cols_dynamic(phase_cfg, available, species, results_df, importance_df):
    phase_number = int(phase_cfg["dynamic_base"].replace("phase",""))
    base_cols, base_phase, base_score, used_fallback = get_best_phase_features(
        phase_number, species, results_df, importance_df, available)
    additional = [f for f in phase_cfg["features"] if f != DYNAMIC_BASE and f in available]
    seen  = set(base_cols)
    full  = base_cols + [f for f in additional if f not in seen]
    return full, base_phase, base_score, used_fallback

# ══════════════════════════════════════════════════════════════════════════════
# SEASON DATASETS
# ══════════════════════════════════════════════════════════════════════════════
def assign_biannual(month):
    return "LRLD" if 3 <= month <= 9 else "SRSD"

def assign_quadseasonal(month):
    if month in [3,4,5,6]:    return "LRS"
    elif month in [7,8,9]:    return "LRS_dry"
    elif month in [10,11,12]: return "SRS"
    else:                     return "SRS_dry"

def get_season_year(month, year, scheme_season):
    if scheme_season in ["SRSD","SRS_dry"] and month in [1,2]:
        return year - 1
    return year

feat_df["biannual_season"] = feat_df["month"].apply(assign_biannual)
feat_df["quad_season"]     = feat_df["month"].apply(assign_quadseasonal)
feat_df["biannual_year"]   = feat_df.apply(lambda r: get_season_year(r["month"],r["year"],r["biannual_season"]),axis=1)
feat_df["quad_year"]       = feat_df.apply(lambda r: get_season_year(r["month"],r["year"],r["quad_season"]),axis=1)

valid_years   = sorted(set(CONFIG["all_cv_years"] + CONFIG["test_years"]))
ward_datasets = {}

if CONFIG["run_biannual"]:
    feat_seas = feat_df.groupby(["ward_name","biannual_season","biannual_year"])[all_feat_cols].mean().reset_index()
    feat_seas.rename(columns={"biannual_season":"season","biannual_year":"season_year"},inplace=True)
    seas_tgt  = seas_raw.merge(hh_ward[["hhid","ward_name"]],on="hhid",how="left")
    seas_tgt["is_drought_year"] = seas_tgt["season_start_year"].isin(CONFIG["drought_years"]).astype(int)
    label_cols = [c for c in seas_tgt.columns if "tlu_loss_ratio" in c or c in ("data_observed","is_drought_year")]
    seas_agg   = seas_tgt.groupby(["ward_name","season","season_start_year"])[label_cols].agg(
        {**{c:"mean" for c in label_cols if c!="is_drought_year"},"is_drought_year":"max"}).reset_index()
    seas_agg.rename(columns={"season_start_year":"season_year"},inplace=True)
    seas_agg   = seas_agg.merge(feat_seas,on=["ward_name","season","season_year"],how="inner")
    seas_agg   = seas_agg[seas_agg["season_year"].isin(valid_years)].reset_index(drop=True)
    ward_datasets["biannual"] = {"df":seas_agg,"year_col":"season_year","scheme":"biannual"}
    print(f"Biannual: {seas_agg.shape}")

if CONFIG["run_quadseasonal"]:
    feat_quad = feat_df.groupby(["ward_name","quad_season","quad_year"])[all_feat_cols].mean().reset_index()
    feat_quad.rename(columns={"quad_season":"season","quad_year":"season_year"},inplace=True)
    mon_quad  = mon.copy()
    mon_quad["season"]      = mon_quad["month"].apply(assign_quadseasonal)
    mon_quad["season_year"] = mon_quad.apply(lambda r: get_season_year(r["month"],r["year"],r["season"]),axis=1)
    quad_tgt  = (mon_quad.groupby(["ward_name","season","season_year"])
                 [[CONFIG["target_col"],"sample_weight","is_drought_year"]]
                 .agg({CONFIG["target_col"]:"mean","sample_weight":"mean","is_drought_year":"max"})
                 .reset_index())
    quad_df   = quad_tgt.merge(feat_quad,on=["ward_name","season","season_year"],how="inner")
    quad_df   = quad_df[quad_df["season_year"].isin(valid_years)].reset_index(drop=True)
    ward_datasets["quadseasonal"] = {"df":quad_df,"year_col":"season_year","scheme":"quadseasonal"}
    print(f"Quadseasonal: {quad_df.shape}")

if CONFIG["run_monthly"]:
    feat_mon_ward = feat_df.groupby(["ward_name","year","month"])[all_feat_cols].mean().reset_index()
    mon_tgt = (mon.groupby(["ward_name","year","month"])
               [[CONFIG["target_col"],"sample_weight","is_drought_year"]]
               .agg({CONFIG["target_col"]:"mean","sample_weight":"mean","is_drought_year":"max"})
               .reset_index())
    mon_df  = mon_tgt.merge(feat_mon_ward,on=["ward_name","year","month"],how="inner")
    mon_df  = mon_df[mon_df["year"].isin(valid_years)].reset_index(drop=True)
    ward_datasets["monthly"] = {"df":mon_df,"year_col":"year","scheme":"monthly"}
    print(f"Monthly: {mon_df.shape}")

# ══════════════════════════════════════════════════════════════════════════════
# METRICS & CV
# ══════════════════════════════════════════════════════════════════════════════
def bootstrap_spearman_ci(y_true, y_pred, n_resamples=1000, ci_level=0.95):
    """Bootstrap 95% CI on Spearman R."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask   = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    n = len(y_true)
    if n < 5:
        return np.nan, np.nan
    rng     = np.random.default_rng(42)
    boot_rs = []
    for _ in range(n_resamples):
        idx    = rng.integers(0, n, size=n)
        sp_r, _= stats.spearmanr(y_true[idx], y_pred[idx])
        if np.isfinite(sp_r):
            boot_rs.append(sp_r)
    if not boot_rs:
        return np.nan, np.nan
    alpha = 1 - ci_level
    lo    = float(np.percentile(boot_rs, 100 * alpha / 2))
    hi    = float(np.percentile(boot_rs, 100 * (1 - alpha / 2)))
    return lo, hi


def compute_metrics(y_true, y_pred, is_drought=None):
    """
    Compute full metric set including:
    - Spearman_R with bootstrap CI
    - RMSE, R2, MAE, skill score
    - Hit rate + false trigger rate at ALL three thresholds (5%, 10%, 15%)
    - Drought-year metrics where applicable
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask   = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 3:
        return {}

    sp_r, sp_p = stats.spearmanr(y_true, y_pred)
    r2         = float(r2_score(y_true, y_pred))
    rmse       = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae        = float(mean_absolute_error(y_true, y_pred))
    pearson_r  = float(np.corrcoef(y_true, y_pred)[0,1])
    naive_rmse = float(np.sqrt(mean_squared_error(y_true, np.full_like(y_true, y_true.mean()))))
    skill      = float(1 - rmse / naive_rmse) if naive_rmse > 0 else np.nan

    m = {
        "spearman_r":  float(sp_r),
        "spearman_p":  float(sp_p),
        "r2":          r2,
        "rmse":        rmse,
        "mae":         mae,
        "pearson_r":   pearson_r,
        "skill_score": skill,
        "naive_rmse":  naive_rmse,
        "n":           len(y_true),
    }

    # Bootstrap CI on Spearman
    if CONFIG["bootstrap_ci"]:
        ci_lo, ci_hi = bootstrap_spearman_ci(
            y_true, y_pred,
            n_resamples=CONFIG["bootstrap_n_resamples"],
            ci_level=CONFIG["bootstrap_ci_level"],
        )
        m["spearman_r_ci_lo"] = ci_lo
        m["spearman_r_ci_hi"] = ci_hi

    # Hit rate / false trigger at all three thresholds
    for thr in CONFIG["mortality_thresholds"]:
        thr_key = str(int(thr * 100))   # "5", "10", "15"
        y_bin  = (y_true > thr).astype(int)
        yp_bin = (y_pred > thr).astype(int)
        if y_bin.sum() > 0 and y_bin.sum() < len(y_bin):
            m[f"hit_rate_{thr_key}pct"]           = float(recall_score(y_bin, yp_bin, zero_division=0))
            m[f"false_trigger_{thr_key}pct"]      = float(1-precision_score(y_bin,yp_bin,zero_division=1)) if yp_bin.sum()>0 else 0.0
            m[f"f1_{thr_key}pct"]                 = float(f1_score(y_bin,yp_bin,zero_division=0))
            try:    m[f"auc_{thr_key}pct"]        = float(roc_auc_score(y_bin, y_pred))
            except: m[f"auc_{thr_key}pct"]        = np.nan
        else:
            m[f"hit_rate_{thr_key}pct"]      = np.nan
            m[f"false_trigger_{thr_key}pct"] = np.nan
            m[f"f1_{thr_key}pct"]            = np.nan
            m[f"auc_{thr_key}pct"]           = np.nan

    # Keep primary threshold aliases for backward compat
    m["hit_rate"]           = m.get("hit_rate_5pct", np.nan)
    m["false_trigger_rate"] = m.get("false_trigger_5pct", np.nan)

    # Drought-year metrics
    if is_drought is not None:
        is_drought = np.array(is_drought, dtype=bool)[mask]
        if is_drought.sum() >= 3:
            yt_d, yp_d        = y_true[is_drought], y_pred[is_drought]
            m["drought_rmse"]     = float(np.sqrt(mean_squared_error(yt_d, yp_d)))
            m["drought_bias"]     = float((yp_d - yt_d).mean())
            m["drought_r2"]       = float(r2_score(yt_d, yp_d))
            dr_sp, _              = stats.spearmanr(yt_d, yp_d)
            m["drought_spearman"] = float(dr_sp)
        else:
            m.update({"drought_rmse":np.nan,"drought_bias":np.nan,
                      "drought_r2":np.nan,"drought_spearman":np.nan})
    return m


def get_folds(years, min_train=None):
    """
    Expanding window LOWO folds.
    Fold 0: train=[years[0]], val=years[1]  (with min_train_years=1)
    Gap years (2016-2018) are excluded automatically.
    """
    min_train = min_train or CONFIG["min_train_years"]
    years     = sorted([y for y in years if y not in CONFIG["survey_gap_years"]])
    return [
        {"fold": i - min_train, "train_years": years[:i], "val_year": years[i]}
        for i in range(min_train, len(years))
    ]


def compute_fold_selection_score(fold_results):
    """
    Weighted fold selection score for model selection.
    Excludes fold 0 (val=2009, 1 training year — not deployment-relevant).

    Weights:
      0.5 × Spearman of fold 7 (val=2019, post-gap)
      0.3 × mean Spearman of mature folds (val >= 2012)
      0.2 × mean Spearman of early folds (val 2010-2011)
    """
    if not fold_results:
        return 0.0

    df = pd.DataFrame(fold_results)
    if "spearman_r" not in df.columns or "val_year" not in df.columns:
        return 0.0

    w_pg = CONFIG["fold_weight_postgap"]
    w_mt = CONFIG["fold_weight_mature"]
    w_er = CONFIG["fold_weight_early"]

    postgap = df[df["val_year"] == 2019]["spearman_r"].mean()
    mature  = df[df["val_year"].isin([2012,2013,2014,2015])]["spearman_r"].mean()
    early   = df[df["val_year"].isin([2010,2011])]["spearman_r"].mean()

    postgap = postgap if np.isfinite(postgap) else 0.0
    mature  = mature  if np.isfinite(mature)  else 0.0
    early   = early   if np.isfinite(early)   else 0.0

    return float(w_pg * postgap + w_mt * mature + w_er * early)


print("Metrics helpers defined")

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def fit_model(model_name, X_train, y_train, X_val, y_val,
              sample_weight=None, params=None,
              X_train_scaled=None, X_val_scaled=None):
    """
    Fit a single model. Ridge uses StandardScaler-scaled inputs (X_train_scaled,
    X_val_scaled) if provided — all other models use raw median-imputed inputs.
    Scaling is applied externally so the scaler can be saved for inference.
    """
    model = build_model(model_name, params)
    cfg   = MODEL_REGISTRY[model_name]
    sw    = sample_weight if cfg["supports_sample_weight"] else None

    # Ridge uses scaled features for defensible L2 regularisation
    Xt = X_train_scaled if (model_name == "ridge" and X_train_scaled is not None) else X_train
    Xv = X_val_scaled   if (model_name == "ridge" and X_val_scaled   is not None) else X_val

    if cfg["supports_early_stopping"] and model_name in ("xgboost","lgbm"):
        fit_kwargs = {"eval_set":[(Xv, y_val)]}
        if sw is not None: fit_kwargs["sample_weight"] = sw
        if model_name == "xgboost": fit_kwargs["verbose"] = False
        model.fit(Xt, y_train, **fit_kwargs)
    else:
        model.fit(Xt, y_train, **({"sample_weight":sw} if sw is not None else {}))
    return model


def tune_model_bayesian(model_name, X_train, y_train,
                         objective_metric, n_trials=None, cv_inner=None):
    n_trials  = n_trials  or CONFIG["tuning_n_trials"]
    cv_inner  = cv_inner  or CONFIG["tuning_cv_inner"]
    direction = "maximize" if objective_metric in ("spearman_r","r2","auc") else "minimize"

    def optuna_objective(trial):
        try:
            params      = get_optuna_params(model_name, trial)
            full_params = {**MODEL_REGISTRY[model_name]["defaults"], **params}
            full_params.pop("early_stopping_rounds", None)
            model       = MODEL_REGISTRY[model_name]["factory"](full_params)
            scores, kf  = [], int(np.floor(len(X_train) / (cv_inner + 1)))
            if kf < 5: return np.nan
            for fi in range(cv_inner):
                vs, ve = fi*kf, fi*kf+kf
                Xv, yv = X_train.iloc[vs:ve], y_train.iloc[vs:ve]
                Xt, yt = pd.concat([X_train.iloc[:vs],X_train.iloc[ve:]]), pd.concat([y_train.iloc[:vs],y_train.iloc[ve:]])
                med = Xt.median()
                Xt, Xv = Xt.fillna(med), Xv.fillna(med)
                # Scale for Ridge — fit on inner train fold only
                if model_name == "ridge":
                    _sc = StandardScaler()
                    Xt  = pd.DataFrame(_sc.fit_transform(Xt), columns=Xt.columns)
                    Xv  = pd.DataFrame(_sc.transform(Xv),     columns=Xv.columns)
                model.fit(Xt, yt)
                preds = model.predict(Xv)
                if objective_metric == "spearman_r":
                    s, _ = stats.spearmanr(yv, preds)
                    scores.append(s if np.isfinite(s) else -1)
                elif objective_metric == "r2":
                    scores.append(r2_score(yv, preds))
                elif objective_metric == "rmse":
                    scores.append(-np.sqrt(mean_squared_error(yv, preds)))
            return np.mean(scores) if scores else np.nan
        except Exception:
            return np.nan

    study = optuna.create_study(direction=direction)
    study.optimize(optuna_objective, n_trials=n_trials, n_jobs=4,
                   catch=(Exception,), show_progress_bar=False)
    completed = [t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
    if not completed:
        best = {**MODEL_REGISTRY[model_name]["defaults"]}
        best.pop("early_stopping_rounds", None)
        return best, np.nan
    best = {**MODEL_REGISTRY[model_name]["defaults"], **study.best_params}
    best.pop("early_stopping_rounds", None)
    return best, study.best_value


def ensemble_predict(predictions_dict, weights_dict=None, method="weighted_avg"):
    preds = np.array(list(predictions_dict.values()))
    names = list(predictions_dict.keys())
    if method == "simple_avg":
        return preds.mean(axis=0)
    elif method == "weighted_avg":
        if not weights_dict:
            return preds.mean(axis=0)
        weights = np.array([max(weights_dict.get(n,0),0) for n in names], dtype=float)
        if weights.sum() == 0: return preds.mean(axis=0)
        weights /= weights.sum()
        return (preds * weights[:,None]).sum(axis=0)
    return preds.mean(axis=0)

print("Training helpers defined")

# ══════════════════════════════════════════════════════════════════════════════
# MLFLOW — local only
# ══════════════════════════════════════════════════════════════════════════════
mlflow.set_tracking_uri(f"file://{LOG_DIR.absolute()}/mlruns")
mlflow.set_experiment(CONFIG["experiment_name"])
print("MLflow: local tracking")


def build_run_name(phase_name, scheme, species, models, ensemble, tuning_obj=None):
    tuning_str = f"_tune={tuning_obj}" if tuning_obj else ""
    return (f"LMR/phase={phase_name}/scheme={scheme}/"
            f"species={species}/models={'+'.join(models)}/ens={ensemble}/all_data{tuning_str}")


def log_run(run_name, metrics, params, importance_df=None):
    try:
        with mlflow.start_run(run_name=run_name, nested=True) as run:
            mlflow.log_params(params)
            mlflow.log_metrics({k:v for k,v in metrics.items()
                                if isinstance(v,(int,float)) and np.isfinite(v)})
            if importance_df is not None and not importance_df.empty:
                imp_path = LOG_DIR / f"imp_{run_name.replace('/','_')[:80]}.csv"
                importance_df.to_csv(imp_path, index=False)
                try: mlflow.log_artifact(str(imp_path))
                except: pass
            return run.info.run_id
    except Exception as e:
        print(f"    MLflow log_run failed (non-fatal): {e}")
        return "mlflow_failed"

# ══════════════════════════════════════════════════════════════════════════════
# S3 CHECKPOINTING  — write after every experiment, enable resume on failure
# ══════════════════════════════════════════════════════════════════════════════
_SM_BUCKET    = "amazon-sagemaker-575108933641-us-east-1-c422b90ce861"
_SM_PREFIX    = os.environ.get("LMR_S3_PREFIX", "lmr-pipeline-v2-full")

# Resolve job name — Processing Jobs write metadata to a known path
def _resolve_job_name():
    # Prefer explicit env var set by launcher, fall back to SageMaker env vars
    for var in ["LMR_JOB_NAME", "TRAINING_JOB_NAME", "SAGEMAKER_JOB_NAME"]:
        val = os.environ.get(var)
        if val:
            return val
    # Try SageMaker config files
    import json as _json, glob as _glob
    for cfg_path in _glob.glob("/opt/ml/config/*.json"):
        try:
            with open(cfg_path) as f:
                cfg = _json.load(f)
            for key in ["ProcessingJobName", "TrainingJobName", "JobName"]:
                if key in cfg and isinstance(cfg[key], str):
                    return cfg[key]
        except Exception:
            pass
    return "unknown"

_JOB_NAME = _resolve_job_name()
_CKPT_PREFIX  = f"{_SM_PREFIX}/checkpoints/{_JOB_NAME}"
_s3_client    = None

def _get_s3():
    global _s3_client
    if _s3_client is None:
        import boto3 as _b3
        _s3_client = _b3.client("s3", region_name="us-east-1")
    return _s3_client

def s3_save_checkpoint(all_results, all_importance):
    """Write fold_metrics and feature_importance checkpoints directly to S3."""
    try:
        s3 = _get_s3()
        if all_results:
            buf = pd.DataFrame(all_results).to_csv(index=False).encode()
            s3.put_object(Bucket=_SM_BUCKET,
                          Key=f"{_CKPT_PREFIX}/fold_metrics_checkpoint.csv",
                          Body=buf)
        if all_importance:
            buf = pd.concat(all_importance).to_csv(index=False).encode()
            s3.put_object(Bucket=_SM_BUCKET,
                          Key=f"{_CKPT_PREFIX}/feature_importance_checkpoint.csv",
                          Body=buf)
    except Exception as e:
        print(f"    S3 checkpoint failed (non-fatal): {e}")
        # Fall back to local write
        if all_results:
            pd.DataFrame(all_results).to_csv(
                LOG_DIR / "fold_metrics_checkpoint.csv", index=False)
        if all_importance:
            pd.concat(all_importance).to_csv(
                LOG_DIR / "feature_importance_checkpoint.csv", index=False)

def s3_load_checkpoint():
    """
    Load existing checkpoint from S3 on job start.
    Returns (results_df, importance_df) or (None, None) if no checkpoint exists.
    Enables resume — already-completed experiments will be skipped.
    """
    try:
        s3 = _get_s3()
        import io as _io
        results_df    = None
        importance_df = None
        try:
            obj = s3.get_object(Bucket=_SM_BUCKET,
                                Key=f"{_CKPT_PREFIX}/fold_metrics_checkpoint.csv")
            results_df = pd.read_csv(_io.BytesIO(obj["Body"].read()))
            print(f"  Resumed fold_metrics checkpoint: {len(results_df)} rows")
        except Exception:
            pass
        try:
            obj = s3.get_object(Bucket=_SM_BUCKET,
                                Key=f"{_CKPT_PREFIX}/feature_importance_checkpoint.csv")
            importance_df = pd.read_csv(_io.BytesIO(obj["Body"].read()))
            print(f"  Resumed feature_importance checkpoint: {len(importance_df)} rows")
        except Exception:
            pass
        return results_df, importance_df
    except Exception as e:
        print(f"  S3 checkpoint load failed (non-fatal): {e}")
        return None, None

print(f"S3 checkpointing: s3://{_SM_BUCKET}/{_CKPT_PREFIX}/")

# ══════════════════════════════════════════════════════════════════════════════
# COLLINEARITY
# ══════════════════════════════════════════════════════════════════════════════
def compute_collinearity_report(X_train, feat_names, phase_name, scheme, species):
    threshold = CONFIG["collinearity_threshold"]
    if X_train.shape[1] < 2: return
    try:
        corr    = X_train.corr(method="pearson").abs()
        upper   = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        flagged = (upper.stack().reset_index()
                   .rename(columns={"level_0":"feature_a","level_1":"feature_b",0:"correlation"})
                   .query(f"correlation >= {threshold}")
                   .sort_values("correlation", ascending=False))
        if not flagged.empty:
            flagged[["phase","scheme","species"]] = phase_name, scheme, species
            path = LOG_DIR / f"collinearity_{phase_name}_{scheme}_{species}.csv"
            flagged.to_csv(path, index=False)
    except Exception as e:
        print(f"    Collinearity failed (non-fatal): {e}")

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_experiment(dataset_name, dataset_cfg, phase_name, feat_cols_static,
                   models_to_run, ensemble_method, tuning_obj=None,
                   results_df_so_far=None, importance_df_so_far=None):
    df         = dataset_cfg["df"]
    year_col   = dataset_cfg["year_col"]
    scheme     = dataset_cfg["scheme"]
    is_monthly = (scheme == "monthly")
    phase_cfg  = FEATURE_PHASES[phase_name]
    phase_num  = phase_cfg["phase"]
    is_dynamic = phase_cfg["dynamic_base"] is not None

    label_map = {}
    for sp in CONFIG["species"]:
        candidate = f"tlu_loss_ratio_{sp}"
        if candidate in df.columns:
            label_map[sp] = candidate
        elif sp == "total" and CONFIG["target_col"] in df.columns:
            label_map["total"] = CONFIG["target_col"]

    all_results    = []
    all_importance = []

    for species, label_col in label_map.items():
        # CV only — exclude test year
        df_cv = df[~df[year_col].isin(CONFIG["test_years"])].dropna(subset=[label_col]).copy()
        if len(df_cv) < 10:
            continue

        if is_dynamic:
            feat_cols, base_phase, base_score, used_fallback = resolve_phase_cols_dynamic(
                phase_cfg, all_feat_cols, species, results_df_so_far, importance_df_so_far)
            dynamic_base_info = {
                "dynamic_base_phase":    base_phase,
                "dynamic_base_score":    round(float(base_score), 4),
                "dynamic_base_fallback": used_fallback,
            }
        else:
            feat_cols         = feat_cols_static
            dynamic_base_info = {"dynamic_base_phase":"none","dynamic_base_score":None,"dynamic_base_fallback":False}

        if not feat_cols:
            print(f"   No features for {phase_name}/{species} — skipping")
            continue

        cv_years = [y for y in sorted(df_cv[year_col].unique()) if y in CONFIG["all_cv_years"]]
        folds    = get_folds(cv_years)
        if not folds:
            continue

        n_trials = CONFIG["tuning_n_trials_p1_p3"] if phase_num <= 3 else CONFIG["tuning_n_trials_p4_p7"]

        best_params_per_model = {}
        if CONFIG["tuning_enabled"] and tuning_obj:
            tune_strategy = CONFIG["tuning_monthly"] if is_monthly else CONFIG["tuning_seasonal"]
            if tune_strategy == "once":
                # Tune on all available training data excluding test year
                train_df = df_cv[df_cv[year_col].isin(CONFIG["train_years"])].copy()
                avail    = [c for c in feat_cols if c in train_df.columns]
                if len(train_df) >= CONFIG["tuning_min_rows_nested"] and avail:
                    med    = train_df[avail].median()
                    avail  = [f for f in avail if pd.notna(med[f])]
                    X_full = train_df[avail].fillna(med[avail])
                    y_full = train_df[label_col].fillna(0)
                    for mn in models_to_run:
                        # Ridge tunes on scaled features — scaler fitted on X_full
                        if mn == "ridge":
                            _sc = StandardScaler()
                            X_tune = pd.DataFrame(_sc.fit_transform(X_full), columns=X_full.columns)
                        else:
                            X_tune = X_full
                        best_p, best_v = tune_model_bayesian(mn, X_tune, y_full, tuning_obj, n_trials=n_trials)
                        best_params_per_model[mn] = best_p
                        print(f"    Tuned {mn} ({tuning_obj}={best_v:.4f})")

        fold_importance = defaultdict(lambda: defaultdict(float))
        fold_count      = 0
        avail           = []
        fold_results    = []

        for fold in folds:
            train_df = df_cv[df_cv[year_col].isin(fold["train_years"])].copy()
            val_df   = df_cv[df_cv[year_col] == fold["val_year"]].copy()
            if len(train_df) < 5 or len(val_df) < 3:
                continue

            avail = [c for c in feat_cols if c in train_df.columns]
            if not avail: continue

            med   = train_df[avail].median()
            avail = [f for f in avail if pd.notna(med[f])]
            if not avail: continue
            med     = med[avail]
            X_train = train_df[avail].fillna(med)
            y_train = train_df[label_col].fillna(0)
            X_val   = val_df[avail].fillna(med)
            y_val   = val_df[label_col].fillna(0)
            sw      = train_df["sample_weight"].values if "sample_weight" in train_df.columns else None
            drought = val_df["is_drought_year"].values if "is_drought_year" in val_df.columns else None

            # Scale for Ridge — fit on training fold only, apply to val
            _fold_scaler   = StandardScaler()
            X_train_scaled = pd.DataFrame(_fold_scaler.fit_transform(X_train), columns=avail)
            X_val_scaled   = pd.DataFrame(_fold_scaler.transform(X_val),       columns=avail)

            fold_preds  = {}
            fold_scores = {}

            for mn in models_to_run:
                try:
                    params = best_params_per_model.get(mn)
                    model  = fit_model(mn, X_train, y_train, X_val, y_val,
                                       sample_weight=sw, params=params,
                                       X_train_scaled=X_train_scaled,
                                       X_val_scaled=X_val_scaled)
                    # Predict using scaled inputs for Ridge, raw for tree models
                    X_pred          = X_val_scaled if mn == "ridge" else X_val
                    preds           = model.predict(X_pred)
                    fold_preds[mn]  = preds
                    sp_r, _         = stats.spearmanr(y_val, preds)
                    fold_scores[mn] = float(sp_r) if np.isfinite(sp_r) else 0.0

                    if hasattr(model, "get_booster"):
                        for feat, score in model.get_booster().get_fscore().items():
                            fold_importance[mn][feat] += score
                    elif hasattr(model, "feature_importances_"):
                        for feat, score in zip(avail, model.feature_importances_):
                            fold_importance[mn][feat] += score
                except Exception as e:
                    print(f"    {mn} failed fold {fold['fold']}: {e}")
                    continue

            if not fold_preds:
                continue
            fold_count += 1

            ensemble_pred = ensemble_predict(fold_preds, fold_scores, ensemble_method)
            m = compute_metrics(y_val, ensemble_pred, is_drought=drought)
            if not m:
                continue
            for mn, sp in fold_scores.items():
                m[f"spearman_{mn}"] = sp

            m.update({
                "fold":        fold["fold"],
                "val_year":    fold["val_year"],
                "train_years": str(fold["train_years"]),
                "phase":       phase_name,
                "scheme":      scheme,
                "species":     species,
                "label":       label_col,
                "n_train":     len(train_df),
                "n_val":       len(val_df),
                "n_features":  len(avail),
                "models":      "+".join(models_to_run),
                "ensemble":    ensemble_method,
                "tuning_obj":  tuning_obj or "none",
                "observed_only": CONFIG["observed_only"],
                **dynamic_base_info,
            })
            all_results.append(m)
            fold_results.append(m)

            print(f"    Fold {fold['fold']} val={fold['val_year']} "
                  f"n={len(train_df)}/{len(val_df)} | "
                  f"Spearman={m.get('spearman_r',np.nan):.3f} "
                  f"RMSE={m.get('rmse',np.nan):.4f} "
                  f"HitRate@10%={m.get('hit_rate_10pct',np.nan):.3f}")

        # Collinearity for later phases
        if phase_num >= 5 and fold_count > 0 and avail:
            try:
                train_all = df_cv[df_cv[year_col].isin(CONFIG["train_years"])].copy()
                avail_tr  = [c for c in feat_cols if c in train_all.columns]
                if len(avail_tr) > 1:
                    X_tr = train_all[avail_tr].fillna(train_all[avail_tr].median())
                    compute_collinearity_report(X_tr, avail_tr, phase_name, scheme, species)
            except Exception as e:
                print(f"    Collinearity failed (non-fatal): {e}")

        # Feature importance
        if fold_count > 0:
            imp_rows = []
            for mn, feat_scores in fold_importance.items():
                for feat, total_score in feat_scores.items():
                    imp_rows.append({
                        "model": mn, "feature": feat,
                        "importance": total_score / fold_count,
                        "phase": phase_name, "scheme": scheme, "species": species,
                    })
            if imp_rows:
                imp_df = pd.DataFrame(imp_rows).sort_values(
                    ["model","importance"], ascending=[True, False])
                all_importance.append(imp_df)

        # Weighted fold selection score for this experiment
        fold_selection_score = compute_fold_selection_score(fold_results)

        # MLflow logging
        species_results = [r for r in all_results if r["species"] == species]
        if not species_results:
            continue

        run_name    = build_run_name(phase_name, scheme, species, models_to_run, ensemble_method, tuning_obj)
        fold_df     = pd.DataFrame(species_results)
        metric_cols = ["spearman_r","r2","rmse","skill_score","hit_rate",
                       "false_trigger_rate","drought_rmse","drought_bias"]
        avail_cols  = [c for c in metric_cols if c in fold_df.columns]
        agg_metrics = fold_df[avail_cols].mean().to_dict()
        agg_metrics["n_folds"]              = len(fold_df)
        agg_metrics["fold_selection_score"] = fold_selection_score

        # Also log multi-threshold summaries
        for thr_key in ["5pct","10pct","15pct"]:
            hr_col = f"hit_rate_{thr_key}"
            ft_col = f"false_trigger_{thr_key}"
            if hr_col in fold_df.columns:
                agg_metrics[f"mean_{hr_col}"] = float(fold_df[hr_col].mean())
            if ft_col in fold_df.columns:
                agg_metrics[f"mean_{ft_col}"] = float(fold_df[ft_col].mean())

        log_params = {
            "phase":                  phase_name,
            "scheme":                 scheme,
            "species":                species,
            "models":                 "+".join(models_to_run),
            "ensemble":               ensemble_method,
            "n_features":             len(avail) if avail else 0,
            "observed_only":          str(CONFIG["observed_only"]),
            "tuning":                 tuning_obj or "none",
            "dynamic_base_phase":     dynamic_base_info["dynamic_base_phase"],
            "dynamic_base_fallback":  str(dynamic_base_info["dynamic_base_fallback"]),
        }
        imp_combined = pd.concat(all_importance) if all_importance else pd.DataFrame()
        run_id = log_run(run_name, agg_metrics, log_params, imp_combined)

        for r in all_results:
            if r["species"] == species and "run_id" not in r:
                r["run_id"] = run_id

    return all_results, all_importance

print("Experiment runner defined")

# ══════════════════════════════════════════════════════════════════════════════
# LEADERBOARD
# ══════════════════════════════════════════════════════════════════════════════
def build_leaderboard(results_df):
    if results_df.empty:
        return pd.DataFrame()
    results_df = results_df.copy()
    if "obs_note" not in results_df.columns:
        results_df["obs_note"] = "all_data"
    group_cols = ["phase","scheme","species","models","ensemble","obs_note","tuning_obj","n_features"]
    group_cols = [c for c in group_cols if c in results_df.columns]

    # Base aggregation — straightforward, no lambdas
    agg_dict = {
        "spearman_r":   ("spearman_r",  "mean"),
        "spearman_std": ("spearman_r",  "std"),
        "r2":           ("r2",          "mean"),
        "rmse":         ("rmse",        "mean"),
        "skill_score":  ("skill_score", "mean"),
        "n_folds":      ("r2",          "count"),
    }
    for thr_key in ["5pct","10pct","15pct"]:
        for metric in ["hit_rate","false_trigger"]:
            col = f"{metric}_{thr_key}"
            if col in results_df.columns:
                agg_dict[f"mean_{col}"] = (col, "mean")
    if "drought_rmse" in results_df.columns:
        agg_dict["drought_rmse"] = ("drought_rmse", "mean")

    agg = results_df.groupby(group_cols).agg(**agg_dict).reset_index()

    # Add postgap and mature Spearman separately — safer than lambda in agg
    postgap = (results_df[results_df["val_year"] == 2019]
               .groupby(group_cols)["spearman_r"].mean()
               .reset_index().rename(columns={"spearman_r": "spearman_r_postgap"}))
    mature  = (results_df[results_df["val_year"].isin([2012,2013,2014,2015])]
               .groupby(group_cols)["spearman_r"].mean()
               .reset_index().rename(columns={"spearman_r": "spearman_r_mature"}))

    agg = agg.merge(postgap, on=group_cols, how="left")
    agg = agg.merge(mature,  on=group_cols, how="left")

    return agg.sort_values("spearman_r", ascending=False)

# ══════════════════════════════════════════════════════════════════════════════
# TEST SET EVALUATION  — 2020, run once after grid search complete
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_test_set(best_phase_name, best_tuning_obj, results_df, importance_df):
    """
    Train the best selected model on ALL CV data (2008-2015 + 2019),
    evaluate on 2020 (test year). Saves test_results.csv separately.
    Called once after grid search — never during model selection.
    """
    print(f"\n{'='*70}")
    print(f"TEST SET EVALUATION — year {CONFIG['test_years']}")
    print(f"  Best phase: {best_phase_name}  tuning: {best_tuning_obj}")
    print(f"{'='*70}")

    test_results = []

    for scheme_name, dataset_cfg in active_schemes.items():
        df       = dataset_cfg["df"]
        year_col = dataset_cfg["year_col"]
        scheme   = dataset_cfg["scheme"]

        phase_cfg        = FEATURE_PHASES[best_phase_name]
        is_dynamic       = phase_cfg["dynamic_base"] is not None
        feat_cols_static = phase_cols_map.get(best_phase_name, [])

        label_map = {}
        for sp in CONFIG["species"]:
            candidate = f"tlu_loss_ratio_{sp}"
            if candidate in df.columns:
                label_map[sp] = candidate
            elif sp == "total" and CONFIG["target_col"] in df.columns:
                label_map["total"] = CONFIG["target_col"]

        for species, label_col in label_map.items():
            # Train on everything except test year
            train_df = df[~df[year_col].isin(CONFIG["test_years"])].dropna(subset=[label_col]).copy()
            test_df  = df[df[year_col].isin(CONFIG["test_years"])].dropna(subset=[label_col]).copy()

            if len(train_df) < 10 or len(test_df) < 3:
                continue

            if is_dynamic:
                feat_cols, _, _, _ = resolve_phase_cols_dynamic(
                    phase_cfg, all_feat_cols, species,
                    results_df, importance_df)
            else:
                feat_cols = feat_cols_static

            if not feat_cols:
                continue

            avail = [c for c in feat_cols if c in train_df.columns]
            if not avail: continue
            med   = train_df[avail].median()
            avail = [f for f in avail if pd.notna(med[f])]
            if not avail: continue
            med      = med[avail]
            X_train  = train_df[avail].fillna(med)
            y_train  = train_df[label_col].fillna(0)
            X_test   = test_df[avail].fillna(med)
            y_test   = test_df[label_col].fillna(0)
            drought  = test_df["is_drought_year"].values if "is_drought_year" in test_df.columns else None

            # Scale for Ridge — fit on training data only
            _test_scaler   = StandardScaler()
            X_train_scaled = pd.DataFrame(_test_scaler.fit_transform(X_train), columns=avail)
            X_test_scaled  = pd.DataFrame(_test_scaler.transform(X_test),      columns=avail)

            # Get best params from tuning (use results from best fold if available)
            best_params = {}
            fold_preds  = {}
            fold_scores = {}

            for mn in CONFIG["models_to_run"]:
                try:
                    model = fit_model(mn, X_train, y_train, X_test, y_test,
                                      params=best_params.get(mn),
                                      X_train_scaled=X_train_scaled,
                                      X_val_scaled=X_test_scaled)
                    # Predict using scaled inputs for Ridge
                    X_pred = X_test_scaled if mn == "ridge" else X_test
                    preds = model.predict(X_pred)
                    fold_preds[mn]  = preds
                    sp_r, _         = stats.spearmanr(y_test, preds)
                    fold_scores[mn] = float(sp_r) if np.isfinite(sp_r) else 0.0
                except Exception as e:
                    print(f"    {mn} test eval failed: {e}")

            if not fold_preds:
                continue

            ensemble_pred = ensemble_predict(fold_preds, fold_scores, CONFIG["ensemble_method"])
            m = compute_metrics(y_test, ensemble_pred, is_drought=drought)
            if not m:
                continue

            m.update({
                "phase":      best_phase_name,
                "scheme":     scheme_name,
                "species":    species,
                "tuning_obj": best_tuning_obj,
                "val_year":   CONFIG["test_years"][0],
                "n_train":    len(train_df),
                "n_test":     len(test_df),
                "split":      "test",
            })
            test_results.append(m)
            print(f"  {scheme_name}/{species}: Spearman={m.get('spearman_r',np.nan):.3f}  "
                  f"RMSE={m.get('rmse',np.nan):.4f}  "
                  f"HitRate@10%={m.get('hit_rate_10pct',np.nan):.3f}")

    if test_results:
        test_df_out = pd.DataFrame(test_results)
        test_df_out.to_csv(LOG_DIR / "test_results_2020.csv", index=False)
        print(f"\nTest results saved: {len(test_results)} rows")
    return test_results

# ══════════════════════════════════════════════════════════════════════════════
# GRID SEARCH
# ══════════════════════════════════════════════════════════════════════════════
all_results    = []
all_importance = []
t0 = time.time()

_phases_env = os.environ.get("LMR_PHASES_TO_RUN")
if _phases_env:
    phases_to_run = json.loads(_phases_env)
    print(f"Phases from env: {phases_to_run}")
else:
    phases_to_run = list(FEATURE_PHASES.keys())
    print(f"Running all {len(phases_to_run)} phases")

prior_results_df    = None
prior_importance_df = None
if PRIOR_RESULTS_DIR.exists():
    for _candidate in [
        PRIOR_RESULTS_DIR / "fold_metrics_checkpoint.csv",
        PRIOR_RESULTS_DIR / "lmrp_results" / "fold_metrics_checkpoint.csv",
    ]:
        if _candidate.exists():
            prior_results_df = pd.read_csv(_candidate)
            print(f"Loaded prior fold_metrics: {len(prior_results_df)} rows")
            break
    for _candidate in [
        PRIOR_RESULTS_DIR / "feature_importance_checkpoint.csv",
        PRIOR_RESULTS_DIR / "lmrp_results" / "feature_importance_checkpoint.csv",
    ]:
        if _candidate.exists():
            prior_importance_df = pd.read_csv(_candidate)
            print(f"Loaded prior feature_importance: {len(prior_importance_df)} rows")
            break

CONFIG["observed_only"] = False

# ── Resume from S3 checkpoint if available ────────────────────────────────────
# On a fresh run this returns (None, None). On a resumed run it loads prior
# results and skips already-completed experiments in the loop below.
print("Checking S3 for existing checkpoint (resume support)...")
_ckpt_results, _ckpt_importance = s3_load_checkpoint()
if _ckpt_results is not None:
    # Merge checkpoint with prior tier results
    if prior_results_df is not None:
        prior_results_df = pd.concat([prior_results_df, _ckpt_results], ignore_index=True)
    else:
        prior_results_df = _ckpt_results
    # Seed all_results so already-done experiments are skipped
    all_results    = _ckpt_results.to_dict("records")
    print(f"  Resuming: {len(all_results)} fold results already done")
    # Build completed experiment set for skip logic
    _done_experiments = set(
        zip(
            _ckpt_results["phase"],
            _ckpt_results["scheme"],
            _ckpt_results["tuning_obj"].fillna("none"),
        )
    )
else:
    _done_experiments = set()
    print("  No checkpoint found — starting fresh")

if _ckpt_importance is not None:
    all_importance = [_ckpt_importance]
    if prior_importance_df is not None:
        prior_importance_df = pd.concat([prior_importance_df, _ckpt_importance], ignore_index=True)
    else:
        prior_importance_df = _ckpt_importance

active_schemes = {}
for scheme_name, ds_key in [("biannual","biannual"),("quadseasonal","quadseasonal"),("monthly","monthly")]:
    if CONFIG.get(f"run_{scheme_name}") and ds_key in ward_datasets:
        active_schemes[scheme_name] = ward_datasets[ds_key]

print(f"\n{'='*70}")
print(f"LMR GRID SEARCH — v3 clean")
print(f"  Phases:         {len(phases_to_run)}")
print(f"  Schemes:        {list(active_schemes.keys())}")
print(f"  CV years:       {CONFIG['all_cv_years']}")
print(f"  Test year:      {CONFIG['test_years']}")
print(f"  Thresholds:     {CONFIG['mortality_thresholds']}")
print(f"  Bootstrap CI:   {CONFIG['bootstrap_ci']} ({CONFIG['bootstrap_n_resamples']} resamples)")
print(f"  Prior results:  {prior_results_df is not None}")
print(f"{'='*70}\n")

with mlflow.start_run(run_name=f"LMR_{CONFIG['run_description']}"):
    try:
        mlflow.log_params({k:str(v) for k,v in CONFIG.items()
                           if not k.endswith("_path")})
    except Exception as e:
        print(f"  MLflow log_params failed (non-fatal): {e}")

    exp_num = 0

    for phase_name in phases_to_run:
        phase_cfg        = FEATURE_PHASES[phase_name]
        phase_num        = phase_cfg["phase"]
        is_dynamic       = phase_cfg["dynamic_base"] is not None
        feat_cols_static = phase_cols_map.get(phase_name, [])

        if not is_dynamic and not feat_cols_static:
            print(f"\n  Skipping {phase_name} — no features resolved")
            continue

        if not CONFIG["tuning_enabled"] or phase_num in (0, 99):
            phase_tuning_objectives = [None]
        else:
            phase_tuning_objectives = ["spearman_r", "rmse"]

        for scheme_name, dataset_cfg in active_schemes.items():
            for tuning_obj in phase_tuning_objectives:
                exp_num += 1
                elapsed = (time.time() - t0) / 60
                print(f"\n  [{exp_num}] {phase_name} | {scheme_name} | "
                      f"tuning={tuning_obj or 'none'} | elapsed={elapsed:.0f}m")

                current_results_df = pd.DataFrame(all_results) if all_results else None
                current_imp_df     = pd.concat(all_importance) if all_importance else None

                results_df_so_far = (
                    pd.concat([prior_results_df, current_results_df], ignore_index=True)
                    if prior_results_df is not None and current_results_df is not None
                    else prior_results_df if prior_results_df is not None
                    else current_results_df
                )
                importance_df_so_far = (
                    pd.concat([prior_importance_df, current_imp_df], ignore_index=True)
                    if prior_importance_df is not None and current_imp_df is not None
                    else prior_importance_df if prior_importance_df is not None
                    else current_imp_df
                )

                # Skip if already completed in a previous run (resume support)
                exp_key = (phase_name, scheme_name, tuning_obj or "none")
                if exp_key in _done_experiments:
                    print(f"  [{exp_num}] SKIP (already done): {phase_name} | {scheme_name} | {tuning_obj or 'none'}")
                    continue

                results, importance = run_experiment(
                    dataset_name         = scheme_name,
                    dataset_cfg          = dataset_cfg,
                    phase_name           = phase_name,
                    feat_cols_static     = feat_cols_static,
                    models_to_run        = CONFIG["models_to_run"],
                    ensemble_method      = CONFIG["ensemble_method"],
                    tuning_obj           = tuning_obj,
                    results_df_so_far    = results_df_so_far,
                    importance_df_so_far = importance_df_so_far,
                )
                all_results.extend(results)
                all_importance.extend(importance)

                # Save checkpoint to S3 after every experiment (enables monitoring + resume)
                s3_save_checkpoint(all_results, all_importance)

elapsed = (time.time() - t0) / 60
print(f"\nGrid search complete: {elapsed:.1f}m | {len(all_results)} fold results")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS + TEST SET EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
import shutil

results_df    = pd.DataFrame(all_results)
importance_df = pd.concat(all_importance, ignore_index=True) if all_importance else pd.DataFrame()

results_df.to_csv(LOG_DIR / "fold_metrics.csv", index=False)
if not importance_df.empty:
    importance_df.to_csv(LOG_DIR / "feature_importance.csv", index=False)

leaderboard_df = build_leaderboard(results_df)
if not leaderboard_df.empty:
    leaderboard_df.to_csv(LOG_DIR / "leaderboard_final.csv", index=False)

    # Select best model by weighted fold selection score
    # Use biannual/total as primary ranking slice
    lb_primary = leaderboard_df[
        (leaderboard_df["scheme"] == "biannual") &
        (leaderboard_df["species"] == "total")
    ]
    if not lb_primary.empty and "spearman_r_postgap" in lb_primary.columns:
        # Compute selection score on leaderboard
        lb_primary = lb_primary.copy()
        lb_primary["fold_selection_score"] = (
            CONFIG["fold_weight_postgap"] * lb_primary["spearman_r_postgap"].fillna(0) +
            CONFIG["fold_weight_mature"]  * lb_primary["spearman_r_mature"].fillna(0)
        )
        best_row      = lb_primary.sort_values("fold_selection_score", ascending=False).iloc[0]
        best_phase    = best_row["phase"]
        best_tuning   = best_row.get("tuning_obj","none")
        print(f"\n=== BEST MODEL (biannual, total, by weighted fold score) ===")
        print(f"  Phase:            {best_phase}")
        print(f"  Tuning:           {best_tuning}")
        print(f"  Spearman (mean):  {best_row.get('spearman_r',np.nan):.3f}")
        print(f"  Spearman (post-gap fold 7): {best_row.get('spearman_r_postgap',np.nan):.3f}")
        print(f"  Spearman (mature): {best_row.get('spearman_r_mature',np.nan):.3f}")
        print(f"  RMSE:             {best_row.get('rmse',np.nan):.4f}")
        for thr_key in ["5pct","10pct","15pct"]:
            print(f"  HitRate@{thr_key}:   {best_row.get(f'mean_hit_rate_{thr_key}',np.nan):.3f}  "
                  f"FalseTrigger@{thr_key}: {best_row.get(f'mean_false_trigger_{thr_key}',np.nan):.3f}")

        # Run test set evaluation on best model
        evaluate_test_set(best_phase, best_tuning, results_df, importance_df)
    else:
        print("\nCould not identify best model for test evaluation")

# Copy all outputs to top-level output dir for S3 sync
for f in LOG_DIR.glob("*"):
    if f.is_file() and not f.name.startswith("."):
        shutil.copy(f, OUTPUT_DIR / f.name)

print(f"\nOutputs saved to {OUTPUT_DIR}")
print("Job complete.")
