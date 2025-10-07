# Merge image property CSVs from different timepoints into unified tables.
# Assumes each CSV has a column with image file paths, and extracts subject IDs from filenames.
# using settings from config_img_comparison.py

import os
import re
import glob
import pandas as pd
import numpy as np

import config_img_comparison as CFG

# --------- pull settings from config (with safe defaults) ----------
MERGE_IN_DIR       = getattr(CFG, "MERGE_IN_DIR",       r".")
MERGE_OUT_DIR      = getattr(CFG, "MERGE_OUT_DIR",      r"./merged")
TIMEPOINT_PATTERNS = getattr(CFG, "TIMEPOINT_PATTERNS", {"before": "*_BEFORE_*.csv", "after": "*_AFTER_*.csv"})
SUBJECT_REGEX      = getattr(CFG, "SUBJECT_REGEX",      r"(?i)image_properties_(?:before|after)_(?P<subject>sub\d+)\.csv$")
IMG_COL_CANDIDATES = getattr(CFG, "IMG_COL_CANDIDATES", ["img_file", "img", "file", "filename"])
DROP_DUPLICATES    = bool(getattr(CFG, "DROP_DUPLICATES", True))
MERGED_FILENAMES   = getattr(CFG, "MERGED_FILENAMES", {})

os.makedirs(MERGE_OUT_DIR, exist_ok=True)

def find_img_col(df: pd.DataFrame):
    for c in IMG_COL_CANDIDATES:
        if c in df.columns:
            return c
    return None

def parse_subject_from_filename(path: str) -> str:
    fname = os.path.basename(path)
    m = re.search(SUBJECT_REGEX, fname, flags=re.IGNORECASE)
    if m and "subject" in m.groupdict():
        return m.group("subject")
    # fallback: everything after last '_' without extension
    base = os.path.splitext(fname)[0]
    return base.split("_")[-1]

def merge_one_timepoint(tp_key: str, glob_pattern: str) -> pd.DataFrame:
    search = os.path.join(MERGE_IN_DIR, glob_pattern)
    paths = sorted(glob.glob(search))
    if not paths:
        raise FileNotFoundError(f"No CSVs for timepoint='{tp_key}' with pattern '{glob_pattern}' in {MERGE_IN_DIR}")

    prepped = []
    all_cols = set()
    print(f"[MERGE] timepoint='{tp_key}' | files={len(paths)}")

    # 1) read all, add subject + timepoint, collect union of columns
    for p in paths:
        df = pd.read_csv(p)
        subj = parse_subject_from_filename(p)
        df.insert(0, "subject", subj)
        df.insert(1, "timepoint", tp_key)
        prepped.append(df)
        all_cols.update(df.columns)

    # 2) align to union of columns (stable order)
    ordered_cols = ["subject", "timepoint"] + sorted([c for c in all_cols if c not in ("subject", "timepoint")])
    aligned = [df.reindex(columns=ordered_cols) for df in prepped]

    big = pd.concat(aligned, ignore_index=True, sort=False)

    # 3) optional de-dup by (subject, img_file)
    if DROP_DUPLICATES:
        img_col = find_img_col(big)
        if img_col:
            n0 = len(big)
            big = big.drop_duplicates(subset=["subject", img_col], keep="first")
            dropped = n0 - len(big)
            if dropped > 0:
                print(f"[MERGE] {tp_key}: dropped {dropped} duplicates by (subject, {img_col})")

    # basic stats
    n_subj = big["subject"].nunique()
    n_rows = len(big)
    n_num  = big.select_dtypes(include=[np.number]).shape[1]
    print(f"[MERGE] {tp_key}: subjects={n_subj}, rows={n_rows}, numeric_cols={n_num}")
    return big

def main():
    outputs = {}
    for tp_key, pat in TIMEPOINT_PATTERNS.items():
        df = merge_one_timepoint(tp_key, pat)
        out_name = MERGED_FILENAMES.get(tp_key, f"merged_{tp_key}.csv")
        out_path = os.path.join(MERGE_OUT_DIR, out_name)
        df.to_csv(out_path, index=False)
        outputs[tp_key] = out_path
        print(f"[SAVE] {tp_key} → {out_path}")

    print("\n[DONE] Merged outputs:")
    for tp, p in outputs.items():
        print(f" - {tp}: {p}")

if __name__ == "__main__":
    main()
