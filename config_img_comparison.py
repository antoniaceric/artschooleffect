# ============================================================
# config_img_comparison.py  —  Clean, structured configuration
# ============================================================

import os

# -------------------------------
# 0) Global toggles
# -------------------------------
RANDOM_SEED = 42
VERBOSE     = True

# -------------------------------
# 1) (Optional) QIP extraction (upstream; not needed if CSVs already exist)
# -------------------------------
QIP_SCRIPT_PATH = r"C:\Users\anton\Documents\Python\Aesthetics-Toolbox-main\Aesthetics-Toolbox-main\QIP_machine_script.py"
QIP_REPO_DIR    = r"C:\Users\anton\Documents\Python\Aesthetics-Toolbox-main\Aesthetics-Toolbox-main"
AT_WEIGHTS_PATH = r"C:\Users\anton\Documents\Python\Aesthetics-Toolbox-main\Aesthetics-Toolbox-main\AT\bvlc_alexnet_conv1.npy"

# If you ever regenerate QIPs from images (optional):
BEFORE_IMG_ROOT     = r"C:\Users\anton\Documents\Python\Image_properties\Images_before_artschool"
AFTER_IMG_ROOT      = r"C:\Users\anton\Documents\Python\Image_properties\Images_2years_artschool"
SUBJECT_DIR_REGEX   = r"^sub_\d+"

# Flags for the upstream QIP machine (keep for reference)
QIP_FLAGS = {
    'Image size (pixels)': True, 'Aspect ratio': True, 'RMS contrast': True,
    'Luminance entropy': True,   'Complexity': True,   'Edge density': True,
    'Color entropy': True,
    'means RGB': True, 'means Lab': True, 'means HSV': True,
    'std RGB': True,   'std Lab': True,   'std HSV': True,
    'Mirror symmetry': True, 'DCM': True, 'Balance': True,
    'left-right': True, 'up-down': True, 'left-right & up-down': True,
    'Slope Redies': True, 'Slope Spehar': True, 'Slope Mather': True,
    'Sigma': True, '2-dimensional': True, '3-dimensional': True,
    'PHOG-based': True, 'CNN-based': True, 'Anisotropy': True,
    'Homogeneity': True, '1st-order': True, '2nd-order': True,
    'Sparseness': True, 'Variability': True,
}

# -------------------------------
# 2) Merge settings (per-subject CSVs → one CSV per timepoint)
# -------------------------------
MERGE_IN_DIR  = r"C:\Users\anton\Documents\Python\Image_properties\Results"
MERGE_OUT_DIR = r"C:\Users\anton\Documents\Python\Image_properties\Results\merged"

# Filenames like: Image_properties_BEFORE_sub01.csv / Image_properties_AFTER_sub01.csv
TIMEPOINT_PATTERNS = {
    "before": "*_BEFORE_*.csv",
    "after":  "*_AFTER_*.csv",
}

# Extract "subject" from filename (case-insensitive)
SUBJECT_REGEX = r"(?i)image_properties_(?:before|after)_(?P<subject>sub\d+)\.csv$"

# Identify image column if you want duplicate filtering by (subject, img_file)
IMG_COL_CANDIDATES = ["img_file", "img", "file", "filename"]
DROP_DUPLICATES    = True

# Output names for merged files
MERGED_FILENAMES = {
    "before": "merged_before.csv",
    "after":  "merged_after.csv",
}

# -------------------------------
# 3) Analysis I/O (consumed by the analysis script)
# -------------------------------
SUBJECT_COL  = "subject"
BEFORE_LABEL = "before"
AFTER_LABEL  = "after"   # labels used in plots / long-format models

# Point to the merged outputs from Section 2
CSV_BEFORE = r"C:\Users\anton\Documents\Python\Image_properties\Results\merged\merged_before.csv"
CSV_YEAR2  = r"C:\Users\anton\Documents\Python\Image_properties\Results\merged\merged_after.csv"

# Where analysis results will be written
OUT_DIR = r"C:\Users\anton\Documents\Python\Image_properties\Analysis_Out"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MERGE_OUT_DIR, exist_ok=True)

# -------------------------------
# 4) Optional metadata (for school/years/medium tests)
# -------------------------------
METADATA_CSV = r"C:\Users\anton\Documents\Python\Image_properties\Results\metadata_preprocessed.csv"
SCHOOL_COL   = "school"
YEARS_COL    = "years_art_school"
MEDIUM_COL   = "medium"

# -------------------------------
# 5) Filtering (drop size-like or specific metrics from inference)
# -------------------------------
SIZE_PATTERNS  = ["size","width","height","pixels","pixel","px","area","dpi","megapixel","mpx"]
METRIC_EXCLUDE = []   # add exact names or regex patterns to exclude specific QIPs

# -------------------------------
# 6) Analysis options
# -------------------------------
USE_ANOVARM      = True        # within-subject ANOVA (Time), in addition to paired tests
K_RANGE          = [2,3,4,5,6] # k-means silhouette sweep
KMEANS_N_STARTS  = 25
DPI              = 200
TOP_N_MIXED      = 12          # #QIPs for MixedLM robustness (image-level)
PCA_N_COMPONENTS = 2
HEATMAP_MAX_METRICS = 40
SPAGHETTI_TOP_N     = 8

# -------------------------------
# 7) Plot grouping (for effect summaries)
# -------------------------------
GROUP_RULES = {
    "Color":            [r"RGB", r"HSV", r"Lab", r"Color entropy"],
    "Luminance":        [r"Luminance", r"RMS contrast", r"image size", r"Aspect ratio"],
    "Edges/Texture":    [r"Edge", r"1st-order", r"2nd-order", r"Homogeneity", r"Anisotropy", r"Complexity"],
    "Symmetry/Balance": [r"Mirror symmetry", r"Balance", r"DCM", r"left-right", r"up-down"],
    "Fourier":          [r"Slope", r"Sigma"],
    "Fractal":          [r"Fractal", r"2-dimensional", r"3-dimensional"],
    "Self-similarity":  [r"PHOG", r"CNN-based", r"Self-similarity", r"Sparseness", r"Variability"],
}

# -------------------------------
# 8) Optional targets for “toward-target” Δ (used by analysis script if set)
#     Add exact QIP column names and target values to compute Δ toward target.
#     Example: TARGETS = {"Slope Redies": -2.0, "2D Fractal dimension": 1.7}
# -------------------------------
TARGETS = {
    # "Slope Redies": -2.0,
    # "2D Fractal dimension": 1.7,
}
