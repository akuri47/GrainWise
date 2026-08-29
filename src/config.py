"""
Rice Grain Quality Analyzer — Configuration Module
====================================================
Central configuration for all constants, paths, variety data,
FAQ thresholds, price tables, and model hyperparameters.
"""

import os

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_ROOT
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
WEB_DIR = os.path.join(PROJECT_ROOT, "web")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Ensure output directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data paths
ARUCO_IMAGE_PATH = os.path.join(DATA_DIR, "ArUCo.jpg")
WHOLE_GRAIN_DIR = os.path.join(DATA_DIR, "whole_grain")
BROKEN_GRAIN_DIR = os.path.join(DATA_DIR, "Broken_grain")
CHALKY_GRAIN_DIR = os.path.join(DATA_DIR, "chalky_grain")
DAMAGED_GRAIN_DIR = os.path.join(DATA_DIR, "Damaged_grain")
DISCOLORED_GRAIN_DIR = os.path.join(DATA_DIR, "Discolored_grain")
FOREIGN_MATTER_DIR = os.path.join(DATA_DIR, "Foreign_matter")

# Defect category directories — full-plate source images (label -> path)
DEFECT_CATEGORIES = {
    "whole":      WHOLE_GRAIN_DIR,
    "broken":     BROKEN_GRAIN_DIR,
    "chalky":     CHALKY_GRAIN_DIR,
    "damaged":    DAMAGED_GRAIN_DIR,
    "discolored": DISCOLORED_GRAIN_DIR,
    "foreign":    FOREIGN_MATTER_DIR,
}

# Extracted individual grain crops (created by extract_grains.py)
EXTRACTED_GRAINS_DIR = os.path.join(DATA_DIR, "extracted_grains")
EXTRACTED_DEFECT_CATEGORIES = {
    "whole":      os.path.join(EXTRACTED_GRAINS_DIR, "whole"),
    "broken":     os.path.join(EXTRACTED_GRAINS_DIR, "broken"),
    "chalky":     os.path.join(EXTRACTED_GRAINS_DIR, "chalky"),
    "damaged":    os.path.join(EXTRACTED_GRAINS_DIR, "Damaged"),
    "discolored": os.path.join(EXTRACTED_GRAINS_DIR, "Discolored"),
    "foreign":    os.path.join(EXTRACTED_GRAINS_DIR, "foreign"),
}

# ============================================================
# ArUCo Calibration
# ============================================================
ARUCO_MARKER_SIZE_MM = 10.0  # Legacy: used as default width
ARUCO_MARKER_WIDTH_MM = 10.0   # Real-world width of the marker (mm)
ARUCO_MARKER_HEIGHT_MM = 6.0   # Real-world height of the marker (mm)
ARUCO_DICT_TYPE = "DICT_4X4_50"  # ArUCo dictionary type

# ============================================================
# Multi-Label Grain Classification
# ============================================================
# A grain shorter than this fraction of avg whole grain length = broken
BROKEN_LENGTH_THRESHOLD = 0.75  # 3/4 of avg whole grain length

# Structural status labels (determined by length rule)
GRAIN_STRUCTURAL_LABELS = ["whole", "broken", "foreign"]

# Defect type labels (determined by model prediction)
GRAIN_DEFECT_LABELS = ["clean", "chalky", "damaged", "discolored"]

# ============================================================
# Grain Segmentation Parameters (User-specified pipeline)
# ============================================================
GAUSSIAN_BLUR_KERNEL = (5, 5)  # Gaussian blur kernel size
MORPH_KERNEL_SIZE = (5, 5)     # Morphological opening kernel
MORPH_ITERATIONS = 2           # Morphological opening iterations
CONTOUR_AREA_MIN = 100         # Minimum contour area (pixels²)
CONTOUR_AREA_MAX = 50000       # Maximum contour area (pixels²)

# ============================================================
# Variety Database
# ============================================================
# User-provided data: variety name → {category, dry_weight_g, folder_name}
# category is based on Aspect Ratio (AR):
#   Short: AR < 2.0, Medium: 2.0 ≤ AR ≤ 3.0, Long: AR > 3.0
VARIETY_DATABASE = {
    "swarna": {
        "display_name": "SWARNA",
        "category": "short",
        "dry_weight_g": 0.0139818,
        "folder_name": "swarna",
    },
    "1010": {
        "display_name": "1010",
        "category": "medium",
        "dry_weight_g": 0.0186494,
        "folder_name": "1010",
    },
    "1001": {
        "display_name": "1001",
        "category": "short",
        "dry_weight_g": 0.018,
        "folder_name": "1001",
    },
    "ganga_kaveri": {
        "display_name": "Ganga Kaveri",
        "category": "short",
        "dry_weight_g": 0.016,
        "folder_name": "ganga_kaveri",
    },
    "mansuri": {
        "display_name": "Mansuri",
        "category": "medium",
        "dry_weight_g": 0.0149156,
        "folder_name": "mansuri",
    },
    "golden_mansuri": {
        "display_name": "Golden Mansuri",
        "category": "medium",
        "dry_weight_g": 0.023,
        "folder_name": "golden_mansuri",
    },
    "nati_mansuri": {
        "display_name": "Nati Mansuri",
        "category": "short",
        "dry_weight_g": 0.019,
        "folder_name": "Nati_mansuri",
    },
    "sonam": {
        "display_name": "Sonam",
        "category": "long",
        "dry_weight_g": 0.026,
        "folder_name": "Sonam",
    },
    "ir64": {
        "display_name": "IR64",
        "category": "medium",
        "dry_weight_g": 0.021,
        "folder_name": "IR64",
    },
}

# Map folder names to variety keys for reverse lookup
FOLDER_TO_VARIETY = {}
for key, val in VARIETY_DATABASE.items():
    FOLDER_TO_VARIETY[val["folder_name"].lower()] = key

# ============================================================
# FAQ Standards — Government of India
# ============================================================
# Thresholds for quality grading (percentage values)
FAQ_STANDARDS = {
    "grade_a": {
        "moisture_max": 14.0,
        "foreign_matter_max": 0.5,
        "broken_max": 5.0,
        "damaged_max": 1.0,
        "discolored_max": 1.0,
        "chalky_max": 3.0,
    },
    "grade_b": {
        "moisture_max": 14.0,
        "foreign_matter_max": 1.0,
        "broken_max": 15.0,
        "damaged_max": 3.0,
        "discolored_max": 3.0,
        "chalky_max": 5.0,
    },
    "common": {
        "moisture_max": 14.0,
        "foreign_matter_max": 2.0,
        "broken_max": 25.0,
        "damaged_max": 5.0,
        "discolored_max": 5.0,
        "chalky_max": 7.0,
    },
}

# Weights for composite quality score (higher weight = more important)
QUALITY_WEIGHTS = {
    "moisture": 0.20,
    "foreign_matter": 0.20,
    "broken": 0.20,
    "damaged": 0.15,
    "discolored": 0.10,
    "chalky": 0.15,
}

# ============================================================
# Shelf-Life Estimation Parameters — Halving Rule Model
# ============================================================
# Based on IRRI Rice Knowledge Bank & NDSU Extension (Hellevang)
# Formula: shelf_life_days = BASELINE_DAYS
#          * 2^(-(T - T_REF) / TEMP_HALVING_C)
#          * 2^(-(MC - MC_REF) / MC_HALVING_PCT)
#          * defect_factor
# Clamped to [MIN_SHELF_LIFE_DAYS, MAX_SHELF_LIFE_DAYS]

BASELINE_SHELF_LIFE_DAYS = 365      # Anchor: ~12 months at T_REF / MC_REF
T_REF_C = 21.0                       # Reference temperature (°C)
MC_REF_PCT = 14.0                    # Reference moisture content (% wb)
TEMP_HALVING_C = 5.0                 # Shelf life halves every +5°C
MC_HALVING_PCT = 2.0                 # Shelf life halves every +2% MC

# Defect penalties (linear, floored at 0.5)
BREAKAGE_PENALTY_PER_PCT = 0.02      # 2% shelf-life reduction per 1% broken grains
DAMAGE_PENALTY_PER_PCT = 0.03        # 3% shelf-life reduction per 1% damaged grains
DEFECT_FACTOR_FLOOR = 0.5            # Defect factor never below 50%

# Output clamping
MIN_SHELF_LIFE_DAYS = 10             # Minimum output (days)
MAX_SHELF_LIFE_DAYS = 1095           # Maximum output (days) = 3 years

# Humidity drift flag threshold
HUMIDITY_DRIFT_THRESHOLD_RH = 70.0   # Flag if sustained RH > 70%

# Risk categorization thresholds (MC-based, combined with temperature)
# Risk escalates if T > HIGH_TEMP_THRESHOLD_C within a given MC band
HIGH_TEMP_THRESHOLD_C = 30.0

# ============================================================
# Price Recommendation (₹ per quintal, approximate 2024 MSP/mandi rates)
# ============================================================
PRICE_TABLE = {
    "swarna":         {"grade_a": 2300, "grade_b": 2100, "common": 1900, "rejected": 1500},
    "1010":           {"grade_a": 2400, "grade_b": 2200, "common": 2000, "rejected": 1600},
    "1001":           {"grade_a": 2250, "grade_b": 2050, "common": 1850, "rejected": 1450},
    "ganga_kaveri":   {"grade_a": 2200, "grade_b": 2000, "common": 1800, "rejected": 1400},
    "mansuri":        {"grade_a": 2500, "grade_b": 2300, "common": 2100, "rejected": 1700},
    "golden_mansuri": {"grade_a": 2550, "grade_b": 2350, "common": 2150, "rejected": 1750},
    "nati_mansuri":   {"grade_a": 2350, "grade_b": 2150, "common": 1950, "rejected": 1550},
    "sonam":          {"grade_a": 2600, "grade_b": 2400, "common": 2200, "rejected": 1800},
    "ir64":           {"grade_a": 2450, "grade_b": 2250, "common": 2050, "rejected": 1650},
}

# Price adjustment per FAQ parameter exceedance (₹ deduction per % over limit)
PRICE_DEDUCTION_PER_PCT = {
    "foreign_matter": 50,
    "broken": 20,
    "damaged": 30,
    "discolored": 25,
    "chalky": 15,
}

# ============================================================
# Defect Classifier (ResNet-18) Hyperparameters
# ============================================================
RESNET_INPUT_SIZE = 224          # Input image size for ResNet
RESNET_NUM_CLASSES = 6           # whole, broken, chalky, damaged, discolored, foreign
RESNET_BATCH_SIZE = 16
RESNET_EPOCHS = 50
RESNET_LEARNING_RATE = 1e-4
RESNET_WEIGHT_DECAY = 1e-4
RESNET_TRAIN_SPLIT = 0.8        # 80% train, 20% val
RESNET_FREEZE_UNTIL = "layer3"  # Freeze layers before this

# Class labels for defect classifier
DEFECT_LABELS = ["whole", "broken", "chalky", "damaged", "discolored", "foreign"]

# ============================================================
# Variety Classifier Hyperparameters
# ============================================================
VARIETY_N_ESTIMATORS = 200       # Random Forest trees
VARIETY_MAX_DEPTH = 10
VARIETY_RANDOM_STATE = 42

# Features used for variety classification (image-level aggregated — legacy)
VARIETY_FEATURES = [
    # Shape features (calibrated in mm)
    "mean_length_mm", "mean_width_mm", "mean_aspect_ratio",
    "mean_area_mm2", "std_length_mm", "std_width_mm",
    "median_length_mm", "median_width_mm",
    # Morphological shape descriptors
    "mean_circularity", "mean_solidity", "mean_eccentricity",
    # Normalized color features
    "mean_rg_ratio", "mean_rb_ratio", "mean_gb_ratio", "mean_norm_s",
]

# Per-grain features for individual grain variety classification
PER_GRAIN_VARIETY_FEATURES = [
    "length_mm", "width_mm", "aspect_ratio", "area_mm2",
    "circularity", "solidity", "eccentricity",
    # Normalized color features (ratios are lighting-invariant)
    "rg_ratio", "rb_ratio", "gb_ratio", "norm_s",
]

# If any secondary variety exceeds this % of total grains, flag as mixed
MIXED_VARIETY_THRESHOLD_PCT = 10.0
