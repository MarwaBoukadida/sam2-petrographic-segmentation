# -*- coding: utf-8 -*-
"""Paths and settings for SAM2 optimization and evaluation."""

import os

# Input paths: relative to the working directory; no dataset is distributed.
PATCH_DIR = "patches_data/"
CHECKPOINT_PATH = "checkpoints/sam2.1_hiera_large.pt"
MODEL_CFG_PATH = "configs/sam2.1/sam2.1_hiera_l.yaml"

# Output directories.
RESULTS_DIR = os.path.join(".", "optuna_results")
os.makedirs(RESULTS_DIR, exist_ok=True)
TRIALS_DETAILS_DIR = os.path.join(RESULTS_DIR, "trials_details")
os.makedirs(TRIALS_DETAILS_DIR, exist_ok=True)
ANALYSIS_DIR = os.path.join(RESULTS_DIR, "analysis")

# Optuna study configuration.
# A resumed database continues the same study; use a new one for a new dataset.
# N_TRIALS is the requested budget for each call to study.optimize.

N_TRIALS = 30
OPTUNA_STUDY_NAME = "sam2_geological_loso"
OPTUNA_DB_URL = "sqlite:///optuna_sam2_loso.db"
OPTUNA_PRUNER_TYPE = "median"
OPTUNA_PRUNER_STARTUP_TRIALS = 5
OPTUNA_PRUNER_WARMUP_STEPS = 1

# Border settings for optimization.
# Hold-out evaluation uses the dataset constructor defaults.
EXCLUDE_BORDERS = True
BORDER_MARGIN = 0
BORDER_KEEP_RATIO = 1.0

# Validation configuration.
ENABLE_FULL_LOSO = True

# Validation sampling ratio; each fold uses at least 20 eligible patches,
# or all patches when fewer are available.
SAMPLING_RATIO = 0.2

# Logging.
LOG_LEVEL = "INFO"
SAVE_LOGS_TO_FILE = True
LOG_FILE = os.path.join(RESULTS_DIR, "run.log")

# Memory cleanup frequency (patches).
MEMORY_CLEANUP_FREQUENCY = 50

# Metric weights.
METRIC_WEIGHTS = {
    "iou": 0.5,
    "dice": 0.3,
    "purity": 0.2
}
assert abs(sum(METRIC_WEIGHTS.values()) - 1.0) < 1e-9, "Les poids doivent sommer à 1"

# Optional visualizations for selected high- and low-scoring masks.
SAVE_INDIVIDUAL_VISUALIZATIONS = False
SAVE_TOP_BOTTOM_MASKS = True
# Enable the 2-by-3 mask panel.
SAVE_2X2_VISUALIZATIONS = True
N_BEST_MASKS_PER_PATCH = 5
N_WORST_MASKS_PER_PATCH = 3

# Visualization budget (None means unlimited).
MAX_VISUALIZATIONS_PER_TRIAL = None

# PNG resolution.
VISUALIZATION_DPI = 300

# Optimization plots.

RUN_POSTHOC_ANALYSIS = True
ANALYSIS_FIGURES = [
    "box_by_trial",
    "box_by_image_per_trial",
    "pareto",
    "parallel_coords",
    "importance",
    "pdp",
    "heatmap2d"
]

# Parameter-analysis settings.
METRICS_FOR_ANALYSIS = ["weighted_score", "IoU", "Dice", "purity"]
DEFAULT_SCORE = "weighted_score"
TOP_K_TRIALS = 2
WORST_K_PERCENT = 10
PDP_BINS = 8
