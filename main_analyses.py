# ================================================
# Art School Effect — Portfolio + ICC + Paired/ANOVA + Clustering (PRINTS)
# [MODIFIED to use the two files from your merge script]
# ================================================
# RQ1: Do QIPs change with training?  (paired tests + optional within-subject ANOVA)
# RQ2: Do change patterns cluster, and are clusters/school differences evident? (clustering + ANOVA/MANOVA)
# -----------------------------------------------

# --- Stability tweaks for Windows OpenMP/MKL (pre-import) ---
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from statsmodels.multivariate.manova import MANOVA

# Optional within-subject ANOVA (Time)
try:
    from statsmodels.stats.anova import AnovaRM
    HAVE_ANOVARM = True
except Exception:
    HAVE_ANOVARM = False

# Optional seaborn
try:
    import seaborn as sns
    HAVE_SEABORN = True
except Exception:
    HAVE_SEABORN = False

# ---------------- CONFIG ----------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# This script can still load defaults from config_img_comparison.py
# but we will override the file paths below.
import importlib
import config_img_comparison as CFG
importlib.reload(CFG)

np.random.seed(getattr(CFG, "RANDOM_SEED", 42))
VERBOSE         = getattr(CFG, "VERBOSE", True)
DPI             = getattr(CFG, "DPI", 200)
USE_ANOVARM     = getattr(CFG, "USE_ANOVARM", True) and HAVE_ANOVARM
K_RANGE         = getattr(CFG, "K_RANGE", [2,3,4,5,6])
KMEANS_N_STARTS = getattr(CFG, "KMEANS_N_STARTS", 25)
TOP_N_MIXED     = getattr(CFG, "TOP_N_MIXED", 12)

SUBJECT_COL     = getattr(CFG, "SUBJECT_COL", "subject") # This should be "subject" (which your merge script adds)
BEFORE_LABEL    = getattr(CFG, "BEFORE_LABEL", "before")
AFTER_LABEL     = getattr(CFG, "AFTER_LABEL", "after")

# --- [MODIFIED] Paths now point to the output of your merge script ---
CSV_BEFORE    = r"C:\Users\anton\Documents\Python\Image_properties\Results\merged\merged_before.csv"
CSV_YEAR2     = r"C:\Users\anton\Documents\Python\Image_properties\Results\merged\merged_after.csv"
# ---------------------------------------------------------------------

OUT_DIR         = CFG.OUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# Optional metadata
METADATA_CSV    = getattr(CFG, "METADATA_CSV", None)
SCHOOL_COL      = getattr(CFG, "SCHOOL_COL", "school")
YEARS_COL       = getattr(CFG, "YEARS_COL", "years_art_school")
MEDIUM_COL      = getattr(CFG, "MEDIUM_COL", "medium")

# Exclusions
DEFAULT_SIZE_PATTERNS = ["size","width","height","pixels","pixel","px","area","dpi","megapixel","mpx"]
SIZE_PATTERNS = [p.lower() for p in getattr(CFG, "SIZE_PATTERNS", DEFAULT_SIZE_PATTERNS)]
METRIC_EXCLUDE = getattr(CFG, "METRIC_EXCLUDE", []) + ["Aspect ratio"]

# Visuals
HEATMAP_MAX_METRICS = getattr(CFG, "HEATMAP_MAX_METRICS", 40)
SPAGHETTI_TOP_N     = getattr(CFG, "SPAGHETTI_TOP_N", 8)
PCA_N_COMPONENTS    = getattr(CFG, "PCA_N_COMPONENTS", 2)

TARGETS = getattr(CFG, "TARGETS", {})

def log(msg: str):
    if VERBOSE:
        print(msg)

# ---------------- HELPERS ----------------
def is_size_like(colname: str) -> bool:
    n = str(colname).lower()
    return any(p in n for p in SIZE_PATTERNS)

def matches_exclude(colname: str) -> bool:
    if not METRIC_EXCLUDE: return False
    for pat in METRIC_EXCLUDE:
        try:
            if re.search(pat, colname, flags=re.IGNORECASE):
                return True
        except re.error:
            if str(pat).lower() == str(colname).lower():
                return True
    return False

def hedges_g_paired(diff):
    n = len(diff)
    if n < 2: return np.nan
    sd = np.std(diff, ddof=1)
    if sd == 0: return np.nan
    d = np.mean(diff) / sd
    J = 1 - (3/(4*n - 1)) if n > 2 else 1.0
    return d * J

def rank_biserial_paired(diff):
    pos = np.sum(diff > 0); neg = np.sum(diff < 0)
    n_pairs = pos + neg
    return (pos - neg)/n_pairs if n_pairs > 0 else np.nan

def shapiro_p_safe(arr):
    arr = np.asarray(arr)
    if np.sum(np.isfinite(arr)) < 3: return np.nan
    if len(arr) > 5000: return np.nan
    try:
        return stats.shapiro(arr)[1]
    except Exception:
        return np.nan

def cramers_v(chi2, n, r, c):
    denom = min(r-1, c-1)
    return np.sqrt((chi2 / n) / denom) if denom > 0 else np.nan

# Normalize IDs like 'sub01', '1', '1.0' -> '1' (string)
def _norm_ids(s: pd.Series) -> pd.Series:
    # Extract the first group of digits
    digits = s.astype(str).str.extract(r'(\d+)', expand=False)

    # Convert to a standard integer (handles "01" -> 1 and "1" -> 1)
    # .astype('Int64') supports NaN values (pd.NA)
    numeric = pd.to_numeric(digits, errors='coerce').astype('Int64')

    # Convert back to a plain string for matching
    # "1" -> "1", "33" -> "33", pd.NA -> "<NA>"
    return numeric.astype(str).replace("<NA>", pd.NA)

# ---------- ICC (reliability of portfolio mean) ----------
def icc1k_unbalanced(df_long, subject_col, value_col):
    """
    ICC(1) and ICC(1,k) for unbalanced designs using mean n_bar.
    """
    d = df_long.dropna(subset=[subject_col, value_col]).copy()
    if d[subject_col].nunique() < 2:
        return np.nan, np.nan, np.nan, np.nan
    grp = d.groupby(subject_col)[value_col]
    means = grp.mean()
    counts = grp.count()
    nbar = counts.mean()
    grand = d[value_col].mean()
    ss_between = ((means - grand)**2 * counts).sum()
    ss_within  = ((d[value_col] - d.groupby(subject_col)[value_col].transform('mean'))**2).sum()
    df_between = d[subject_col].nunique() - 1
    df_within  = len(d) - d[subject_col].nunique()
    if df_between <= 0 or df_within <= 0:
        return np.nan, np.nan, np.nan, np.nan
    ms_between = ss_between / df_between
    ms_within  = ss_within  / df_within
    denom = ms_between + (nbar - 1) * ms_within
    icc1  = (ms_between - ms_within) / denom if denom > 0 else np.nan
    icc1k = (nbar * icc1) / (1 + (nbar - 1) * icc1) if not np.isnan(icc1) else np.nan
    return icc1, icc1k, nbar, ms_within

# ---------------- LOAD DATA ----------------
# This section is identical to your original script, but now
# CSV_BEFORE and CSV_YEAR2 point to your new merged files.
log(f"\n[SETUP] Using config: {getattr(CFG, '__file__', '<unknown>')}")
log(f"[PATH] BEFORE CSV: {CSV_BEFORE}")
log(f"[PATH] AFTER  CSV: {CSV_YEAR2}")
if METADATA_CSV:
    log(f"[PATH] META   CSV: {METADATA_CSV}")
else:
    log("[PATH] META   CSV: <none>")

for _p in (CSV_BEFORE, CSV_YEAR2):
    if not os.path.exists(_p):
        _folder = os.path.dirname(_p) or "."
        have = [f for f in os.listdir(_folder) if f.lower().endswith(".csv")] if os.path.isdir(_folder) else []
        raise FileNotFoundError(
            f"\nCSV not found:\n  {_p}\n"
            f"Folder contents ({os.path.abspath(_folder)}):\n  " + ("\n  ".join(have) if have else "(no CSVs found)")
        )

before_raw = pd.read_csv(CSV_BEFORE)
after_raw  = pd.read_csv(CSV_YEAR2)
log(f"[LOAD] BEFORE rows: {len(before_raw):,} | AFTER rows: {len(after_raw):,}")

# Your merge script adds the "subject" column, so this check should pass.
if SUBJECT_COL not in before_raw.columns or SUBJECT_COL not in after_raw.columns:
    raise ValueError(f"'{SUBJECT_COL}' column is required in both CSVs.")

# ---- Restrict to subjects present in metadata (using normalized IDs)
if METADATA_CSV and os.path.exists(METADATA_CSV):
    _meta_tmp = pd.read_csv(METADATA_CSV)
    if SUBJECT_COL in _meta_tmp.columns:
        keep_norm = set(_norm_ids(_meta_tmp[SUBJECT_COL]).dropna())
        before_raw = before_raw[_norm_ids(before_raw[SUBJECT_COL]).isin(keep_norm)].copy()
        after_raw  = after_raw [_norm_ids(after_raw [SUBJECT_COL]).isin(keep_norm)].copy()
        log(f"[FILTER] Kept subjects present in metadata (normalized ids): "
            f"before={before_raw[SUBJECT_COL].nunique()}, after={after_raw[SUBJECT_COL].nunique()}")
    else:
        log("[FILTER] Metadata present but missing SUBJECT_COL; not filtering to metadata subjects.")

before_num = before_raw.select_dtypes(include=[np.number]).copy()
after_num  = after_raw.select_dtypes(include=[np.number]).copy()
common = before_num.columns.intersection(after_num.columns).tolist()
log(f"[NUMERIC] Common numeric columns (raw): {len(common)}")

before = pd.concat([before_raw[[SUBJECT_COL]], before_num[common]], axis=1)
after  = pd.concat([after_raw [[SUBJECT_COL]], after_num [common]], axis=1)

# Remove columns that are all-NaN / zero-variance across both waves
numeric_cols = []
for c in common:
    a = before[c]; b = after[c]
    if (a.count() + b.count()) == 0:
        continue
    if (a.nunique(dropna=True) <= 1) and (b.nunique(dropna=True) <= 1):
        continue
    numeric_cols.append(c)

# Exclude size-like and user-specified metrics
important_cols, dropped_size, dropped_user = [], [], []
for c in numeric_cols:
    if is_size_like(c):
        dropped_size.append(c); continue
    if matches_exclude(c):
        dropped_user.append(c); continue
    important_cols.append(c)

log(f"[FILTER] Dropped size-like cols: {len(dropped_size)}")
if dropped_size: log("         -> " + ", ".join(dropped_size[:12]) + (" ..." if len(dropped_size)>12 else ""))
log(f"[FILTER] Dropped user-excluded cols: {len(dropped_user)}")
if dropped_user: log("         -> " + ", ".join(dropped_user[:12]) + (" ..." if len(dropped_user)>12 else ""))
log(f"[KEEP]   Important QIPs for analysis: {len(important_cols)}")

# ---------------- IMAGE COUNTS ----------------
img_counts_before = before.groupby(SUBJECT_COL).size()
img_counts_after  = after.groupby(SUBJECT_COL).size()
log(f"[IMAGES] Median images per participant - Before: {img_counts_before.median():.1f}, After: {img_counts_after.median():.1f}")

# ---------------- PORTFOLIO MEANS (primary unit) ----------------
subj_before = before.groupby(SUBJECT_COL)[important_cols].mean()
subj_after  = after .groupby(SUBJECT_COL)[important_cols].mean()
common_subjects = subj_before.index.intersection(subj_after.index)
subj_before = subj_before.loc[common_subjects].sort_index()
subj_after  = subj_after .loc[common_subjects].sort_index()
delta = subj_after - subj_before

# Within-portfolio SD (consistency)
sd_before = before.groupby(SUBJECT_COL)[important_cols].std(ddof=1).loc[common_subjects]
sd_after  = after .groupby(SUBJECT_COL)[important_cols].std(ddof=1).loc[common_subjects]
delta_sd  = sd_after - sd_before

# Optional derived "toward-target" deltas (if TARGETS provided)
if isinstance(TARGETS, dict) and len(TARGETS) > 0:
    toward_cols = []
    for qip_name, target_val in TARGETS.items():
        if qip_name in subj_before.columns and qip_name in subj_after.columns:
            colname = f"Delta_toward_target__{qip_name}"
            before_dist = (subj_before[qip_name] - target_val).abs()
            after_dist  = (subj_after[qip_name]  - target_val).abs()
            delta[colname] = (after_dist - before_dist) * -1.0  # positive = moved closer to target
            toward_cols.append(colname)
    if toward_cols:
        log(f"[DERIVED] Added toward-target metrics: {len(toward_cols)}")
        important_cols = important_cols + toward_cols  # include for completeness

# Save key tables
subj_before.to_csv(os.path.join(OUT_DIR, "portfolio_means_before.csv"), encoding="utf-8", index=True)
subj_after .to_csv(os.path.join(OUT_DIR, "portfolio_means_after.csv"),  encoding="utf-8", index=True)
delta      .to_csv(os.path.join(OUT_DIR, "portfolio_delta.csv"),        encoding="utf-8", index=True)
delta_sd   .to_csv(os.path.join(OUT_DIR, "portfolio_delta_withinSD.csv"), encoding="utf-8", index=True)
log(f"[SAVE]  Portfolio means & delta saved to OUT_DIR.")

log(f"[DELTA] Subjects with both waves: {len(common_subjects)} | Metrics: {delta.shape[1]}")

# ---------------- RELIABILITY (ICC) ----------------
log("\n[RELIABILITY] Estimating ICC(1,k) per QIP (per wave) to justify portfolio means")
icc_rows = []
for wave_name, df_wave in [(AFTER_LABEL, after), (BEFORE_LABEL, before)]:
    for q in important_cols:
        icc1, icc1k, nbar, msw = icc1k_unbalanced(
            df_wave[[SUBJECT_COL, q]].rename(columns={q:"val"}),
            subject_col=SUBJECT_COL, value_col="val"
        )
        icc_rows.append({"QIP": q, "Wave": wave_name, "ICC1": icc1, "ICC1k_meanN": icc1k, "mean_n_images": nbar, "MS_within": msw})

icc_tbl = pd.DataFrame(icc_rows)
icc_tbl.to_csv(os.path.join(OUT_DIR, "icc_reliability_by_qip_wave.csv"), index=False, encoding="utf-8")
if not icc_tbl.empty:
    med_icc = icc_tbl.groupby("Wave")["ICC1k_meanN"].median().round(3)
    log(f"[RELIABILITY] Median ICC(1,k) by wave (k = mean images/person): {med_icc.to_dict()}")

# ---------------- RQ1: Before vs After per QIP ----------------
log("\n[RQ1] Testing Before vs After per QIP (paired) and summarizing changes")
paired_rows = []
for q in [c for c in delta.columns if c in subj_before.columns]:  # only original QIPs, not toward-target unless included in means
    d = (subj_after[q] - subj_before[q]).values
    n = len(d)
    if n < 2:
        continue
    p_norm = shapiro_p_safe(d)
    if not np.isnan(p_norm) and p_norm > 0.05:
        t_stat, pval = stats.ttest_rel(subj_after[q], subj_before[q], nan_policy='omit')
        test = "Paired t"
        eff  = hedges_g_paired(d)
        effname = "Hedges_g (paired)"
    else:
        try:
            w_stat, pval = stats.wilcoxon(d, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
        except Exception:
            w_stat, pval = (np.nan, 1.0)
        t_stat = w_stat
        test = "Wilcoxon"
        eff  = rank_biserial_paired(d)
        effname = "Rank-biserial r (paired)"
    paired_rows.append({
        "QIP": q, "Test": test, "Statistic": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "p_value": float(pval), "Effect_name": effname, "Effect": float(eff) if np.isfinite(eff) else np.nan,
        "n_subjects": int(n), "mean_delta": float(np.mean(d)), "sd_delta": float(np.std(d, ddof=1)) if n>1 else np.nan
    })

paired_tbl = pd.DataFrame(paired_rows).sort_values("p_value") if paired_rows else pd.DataFrame(columns=["QIP","p_value"])
if not paired_tbl.empty:
    rej, p_fdr = fdrcorrection(paired_tbl["p_value"].values, alpha=0.05, method='indep')
    paired_tbl["p_FDR"] = p_fdr
    paired_tbl["Significant_FDR_0.05"] = rej
paired_tbl.to_csv(os.path.join(OUT_DIR, "rq1_paired_tests.csv"), index=False, encoding="utf-8")

if not paired_tbl.empty:
    n_sig = int(paired_tbl["Significant_FDR_0.05"].sum())
    log(f"[RQ1] Tested QIPs: {len(paired_tbl)} | FDR-significant (q<.05): {n_sig}")
    top = paired_tbl.copy()
    top["abs_eff"] = top["Effect"].abs()
    top = top.sort_values(["Significant_FDR_0.05","p_FDR","abs_eff"], ascending=[False,True,False]).head(min(15, len(top)))
    log("[RQ1] Top changed QIPs (by FDR then |effect|):")
    for _, r in top.iterrows():
        log(f"   - {r['QIP']}: Δ={r['mean_delta']:.3g}, p={r['p_value']:.3g}, q={r.get('p_FDR',np.nan):.3g}, {r['Effect_name']}={r['Effect']:.3g}")

# Optional: Within-subject ANOVA (Time) per QIP
if USE_ANOVARM:
    log("\n[RQ1-ANOVA] Within-subject ANOVA (Time) per QIP (AnovaRM)")
    arows = []
    long = subj_before.assign(Time=BEFORE_LABEL).reset_index().melt(id_vars=[SUBJECT_COL,"Time"], var_name="QIP", value_name="Value")
    long2= subj_after .assign(Time=AFTER_LABEL ).reset_index().melt(id_vars=[SUBJECT_COL,"Time"], var_name="QIP", value_name="Value")
    long_all = pd.concat([long, long2], ignore_index=True)
    for q in subj_before.columns:
        dfq = long_all[long_all["QIP"]==q]
        if dfq[SUBJECT_COL].nunique() < 2:
            continue
        try:
            res = AnovaRM(dfq, depvar="Value", subject=SUBJECT_COL, within=["Time"]).fit()
            tbl = res.anova_table
            if "Time" in tbl.index:
                F = tbl.loc["Time","F Value"]; p = tbl.loc["Time","Pr > F"]; df1 = tbl.loc["Time","Num DF"]; df2 = tbl.loc["Time","Den DF"]
                arows.append({"QIP": q, "F_Time": F, "df1": df1, "df2": df2, "p_value": p})
        except Exception:
            arows.append({"QIP": q, "F_Time": np.nan, "df1": np.nan, "df2": np.nan, "p_value": np.nan})
    aov_tbl = pd.DataFrame(arows).sort_values("p_value") if arows else pd.DataFrame(columns=["QIP","p_value"])
    if not aov_tbl.empty:
        rej, p_fdr = fdrcorrection(aov_tbl["p_value"].fillna(1.0).values, alpha=0.05, method='indep')
        aov_tbl["p_FDR"] = p_fdr
        aov_tbl["Significant_FDR_0.05"] = rej
        aov_tbl.to_csv(os.path.join(OUT_DIR, "rq1_within_subject_anova.csv"), index=False, encoding="utf-8")
        log(f"[RQ1-ANOVA] Finished AnovaRM; FDR-significant (q<.05): {int(aov_tbl['Significant_FDR_0.05'].sum())}")
else:
    log("[RQ1-ANOVA] AnovaRM not run (disabled or unavailable).")

# Portfolio consistency: did within-portfolio SD decrease?
log("\n[RQ1] Testing within-portfolio consistency (did SD decrease?)")
cons_rows = []
for q in sd_before.columns:
    dsd = (sd_after[q] - sd_before[q]).values
    n = len(dsd)
    if n < 2:
        continue
    p_norm = shapiro_p_safe(dsd)
    if not np.isnan(p_norm) and p_norm > 0.05:
        t_stat, pval = stats.ttest_1samp(dsd, 0.0)
        test = "Paired t on SD-delta (After-Before)"
        eff  = hedges_g_paired(dsd)
        effname = "Hedges_g (paired)"
    else:
        try:
            w_stat, pval = stats.wilcoxon(dsd, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
        except Exception:
            w_stat, pval = (np.nan, 1.0)
        t_stat = w_stat
        test = "Wilcoxon on SD-delta"
        eff  = rank_biserial_paired(dsd)
        effname = "Rank-biserial r (paired)"
    cons_rows.append({"QIP": q, "Test": test, "p_value": float(pval), "Effect_name": effname, "Effect": float(eff) if np.isfinite(eff) else np.nan})
cons_tbl = pd.DataFrame(cons_rows).sort_values("p_value") if cons_rows else pd.DataFrame(columns=["QIP","p_value"])
if not cons_tbl.empty:
    rej, p_fdr = fdrcorrection(cons_tbl["p_value"].values, alpha=0.05, method='indep')
    cons_tbl["p_FDR"] = p_fdr
    cons_tbl["Significant_FDR_0.05"] = rej
cons_tbl.to_csv(os.path.join(OUT_DIR, "rq1_within_portfolio_consistency.csv"), index=False, encoding="utf-8")
if not cons_tbl.empty:
    log(f"[RQ1] Portfolio-consistency: tested {len(cons_tbl)} QIPs; FDR-significant={int(cons_tbl['Significant_FDR_0.05'].sum())}")

# ---------------- RQ1 Figures ----------------
fig_dir = os.path.join(OUT_DIR, "figs_rq1"); os.makedirs(fig_dir, exist_ok=True)
if not paired_tbl.empty and HAVE_SEABORN:
    sel = paired_tbl.copy()
    sel["abs_eff"] = sel["Effect"].abs()
    sel = sel.sort_values(["Significant_FDR_0.05","p_FDR","abs_eff"], ascending=[False,True,False])
    top_names = sel["QIP"].tolist()[:min(HEATMAP_MAX_METRICS, len(sel))]
    Z = (delta[top_names] - delta[top_names].mean()) / delta[top_names].std(ddof=1)
    plt.figure(figsize=(max(6, 0.35*len(top_names)+3), max(4, 0.30*len(Z)+2)))
    sns.heatmap(Z, cmap="vlag", center=0, cbar_kws={"label":"Z(Δ) across subjects"})
    plt.title("Per-subject ΔQIP (z-scored) — top metrics")
    plt.tight_layout()
    fp = os.path.join(fig_dir, "heatmap_delta_top.png")
    plt.savefig(fp, dpi=DPI); plt.close()
    log(f"[FIG]   Heatmap (top ΔQIPs) -> {fp}")

# ---------------- RQ2: Clustering + School differences ----------------
log("\n[RQ2] Clustering Δ-profiles and testing cluster-school association")

X_df = delta.dropna(axis=1, how='any').copy()
if X_df.shape[0] < 2 or X_df.shape[1] < 2:
    log("[RQ2] Skipped clustering (insufficient subjects/metrics).")
else:
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values)
    clu_dir = os.path.join(OUT_DIR, "figs_clustering"); os.makedirs(clu_dir, exist_ok=True)

    # Hierarchical dendrogram (Ward)
    try:
        Z = linkage(X, method="ward")
        plt.figure(figsize=(6,4))
        dendrogram(Z, labels=X_df.index.tolist(), leaf_rotation=90)
        plt.title("Hierarchical clustering (Ward) — subjects by ΔQIPs")
        plt.ylabel("Distance")
        plt.tight_layout()
        fp = os.path.join(clu_dir, "dendrogram_subjects.png")
        plt.savefig(fp, dpi=DPI); plt.close()
        log(f"[FIG]   Dendrogram -> {fp}")
    except Exception as e:
        log(f"[warn] Ward linkage failed: {e}")

    # k-means with silhouette selection
    best_k, best_s = None, -np.inf
    sil_scores = []
    log("[CLUST] Silhouette sweep over k: " + ", ".join(map(str, K_RANGE)))
    for k in K_RANGE:
        if k >= X.shape[0]:
            log(f"[CLUST] skip k={k} (k must be < n_subjects={X.shape[0]})"); continue
        try:
            km = KMeans(n_clusters=k, n_init=KMEANS_N_STARTS, random_state=getattr(CFG,"RANDOM_SEED",42))
            labels = km.fit_predict(X)
            s = silhouette_score(X, labels)
            sil_scores.append((k, s))
            log(f"        k={k}: silhouette={s:.3f}")
            if s > best_s:
                best_s, best_k = s, k
        except Exception as e:
            log(f"[warn] k={k} silhouette failed: {e}")

    if sil_scores:
        ks, ss = zip(*sil_scores)
        plt.figure(figsize=(5,3.2))
        plt.plot(ks, ss, marker='o')
        plt.xlabel("k"); plt.ylabel("Silhouette")
        plt.title("Silhouette scores over k")
        plt.tight_layout()
        fp = os.path.join(clu_dir, "silhouette_over_k.png")
        plt.savefig(fp, dpi=DPI); plt.close()
        log(f"[FIG]   Silhouette curve -> {fp}")

    if best_k is None:
        log("[CLUST] Could not determine best k; skipping k-means labels.")
    else:
        km = KMeans(n_clusters=best_k, n_init=KMEANS_N_STARTS, random_state=getattr(CFG,"RANDOM_SEED",42))
        cluster_labels = km.fit_predict(X)
        cluster_assignments = pd.DataFrame({SUBJECT_COL: X_df.index, "Cluster": cluster_labels})
        clu_path = os.path.join(OUT_DIR, "subject_clusters_kmeans.csv")
        cluster_assignments.to_csv(clu_path, index=False, encoding="utf-8")
        sizes = cluster_assignments["Cluster"].value_counts().sort_index()
        log(f"[CLUST] Best k={best_k} (silhouette={best_s:.3f}); cluster sizes: " + ", ".join([f"{c}:{n}" for c,n in sizes.items()]))
        log(f"[SAVE]  Cluster assignments -> {clu_path}")

        # Visualize clusters in PCA space
        if X_df.shape[1] >= 2:
            pca2 = StandardScaler(with_mean=True, with_std=True).fit_transform(X)  # already standardized X
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=getattr(CFG,"RANDOM_SEED",42))
            Z2 = pca.fit_transform(X)
            exp2 = pca.explained_variance_ratio_ * 100
            plt.figure(figsize=(6,5))
            for c in np.unique(cluster_labels):
                idx = cluster_labels == c
                plt.scatter(Z2[idx,0], Z2[idx,1], label=f"Cluster {c}")
            for i, subj in enumerate(X_df.index):
                plt.text(Z2[i,0], Z2[i,1], str(subj), fontsize=8, ha='left', va='bottom')
            plt.xlabel(f"PC1 ({exp2[0]:.1f}%)"); plt.ylabel(f"PC2 ({exp2[1]:.1f}%)")
            plt.legend()
            plt.title(f"k-means clusters (k={best_k}) in PCA space")
            plt.tight_layout()
            fp = os.path.join(clu_dir, "kmeans_clusters_on_delta_pca.png")
            plt.savefig(fp, dpi=DPI); plt.close()
            log(f"[FIG]   Cluster PCA plot -> {fp}")

         
        log("\n[CLUST] Calculating cluster centroids (mean ΔQIP profile per cluster)")
        try:
            # Merge cluster assignments back onto the original delta data
            delta_with_clusters = delta.reset_index().merge(
                cluster_assignments, on=SUBJECT_COL, how='inner'
            )

            # Calculate centroids (mean delta per QIP for each cluster)
            cluster_centroids = delta_with_clusters.groupby('Cluster')[important_cols].mean().T # Transpose for better readability
            centroid_path = os.path.join(OUT_DIR, "cluster_centroids_mean_delta.csv")
            cluster_centroids.to_csv(centroid_path, encoding="utf-8")
            log(f"[SAVE]  Cluster centroids -> {centroid_path}")

            # Compare centroids using independent t-tests per QIP
            log("[CLUST] Comparing cluster centroids (independent t-tests per QIP)")
            ttest_results = []
            cluster_0_data = delta_with_clusters[delta_with_clusters['Cluster'] == 0]
            cluster_1_data = delta_with_clusters[delta_with_clusters['Cluster'] == 1]

            # Ensure there are enough subjects in each cluster for t-tests (at least 2)
            if len(cluster_0_data) >= 2 and len(cluster_1_data) >= 2:
                for qip in important_cols:
                    try:
                        t_stat, p_val = stats.ttest_ind(
                            cluster_0_data[qip].dropna(),
                            cluster_1_data[qip].dropna(),
                            equal_var=False # Welch's t-test (safer for small/unequal N)
                        )
                        ttest_results.append({
                            "QIP": qip,
                            "t_statistic": t_stat,
                            "p_value": p_val,
                            "Mean_Cluster0": cluster_centroids.loc[qip, 0],
                            "Mean_Cluster1": cluster_centroids.loc[qip, 1]
                        })
                    except Exception as e_ttest:
                        log(f"[warn] T-test failed for {qip}: {e_ttest}")
                        ttest_results.append({
                            "QIP": qip, "t_statistic": np.nan, "p_value": np.nan,
                            "Mean_Cluster0": cluster_centroids.loc[qip, 0],
                            "Mean_Cluster1": cluster_centroids.loc[qip, 1]
                        })

                if ttest_results:
                    ttest_df = pd.DataFrame(ttest_results).sort_values("p_value")
                    # Optional: Add FDR correction if desired, though likely underpowered
                    # rej_fdr, p_fdr = fdrcorrection(ttest_df["p_value"].fillna(1.0).values, alpha=0.05, method='indep')
                    # ttest_df["p_FDR"] = p_fdr
                    ttest_path = os.path.join(OUT_DIR, "cluster_comparison_ttests.csv")
                    ttest_df.to_csv(ttest_path, index=False, encoding="utf-8")
                    log(f"[SAVE]  Cluster comparison t-tests -> {ttest_path}")
                    # Log top differentiating QIPs (uncorrected p-value)
                    top_diff = ttest_df.nsmallest(5, 'p_value')
                    log("[CLUST] Top 5 QIPs differentiating clusters (uncorrected p-value):")
                    for _, row in top_diff.iterrows():
                         log(f"  - {row['QIP']}: p={row['p_value']:.3g}, Mean Clu0={row['Mean_Cluster0']:.3g}, Mean Clu1={row['Mean_Cluster1']:.3g}")

            else:
                log("[CLUST] Skipped t-tests: one or both clusters have fewer than 2 members.")

        except Exception as e_clust_analysis:
            log(f"[warn] Failed to calculate or compare cluster centroids: {e_clust_analysis}")
       
# ---------------- Cluster x School association (normalized ids) ----------------
if METADATA_CSV and os.path.exists(METADATA_CSV) and 'cluster_labels' in locals():
    try:
        meta = pd.read_csv(METADATA_CSV).drop_duplicates(subset=[SUBJECT_COL])
        if SUBJECT_COL in meta.columns and SCHOOL_COL in meta.columns:
            meta['_merge_key'] = _norm_ids(meta[SUBJECT_COL])

            subj_raw = pd.Index(X_df.index)
            subj_norm = pd.Series(subj_raw.astype(str)).str.extract(r'(\d+)', expand=False)
            clu_df = pd.DataFrame({
                SUBJECT_COL: subj_raw.astype(str),
                '_merge_key': subj_norm,
                'Cluster': cluster_labels
            })

            clu_df = clu_df.merge(meta[['_merge_key', SCHOOL_COL]], on='_merge_key', how='left')

            ct = pd.crosstab(clu_df["Cluster"], clu_df[SCHOOL_COL], dropna=False)
            ct_path = os.path.join(OUT_DIR, "cluster_by_school_crosstab.csv")
            ct.to_csv(ct_path, encoding="utf-8")

            from scipy.stats import chi2_contingency
            if ct.shape[1] >= 2 and int(ct.values.sum()) > 0:
                chi2, p_chi, dof, exp = chi2_contingency(ct.values)
                V = cramers_v(chi2, int(ct.values.sum()), ct.shape[0], ct.shape[1])
                with open(os.path.join(OUT_DIR, "cluster_by_school_chi2.txt"), "w", encoding="utf-8") as f:
                    f.write(f"Chi-square = {chi2:.4f}\n")
                    f.write(f"df = {dof}\n")
                    f.write(f"p = {p_chi:.6g}\n")
                    f.write(f"CramersV = {V:.4f}\n")
                log(f"[ASSOC] Cluster x School: chi2={chi2:.2f}, p={p_chi:.4g}, V={V:.3f} -> {ct_path}")
            else:
                log("[ASSOC] Not enough non-missing school levels for chi-square; saved crosstab only.")
        else:
            log("[ASSOC] Metadata missing required columns for Cluster x School.")
    except Exception as e:
        log(f"[warn] Cluster x School association failed: {e}")
else:
    log("[ASSOC] No metadata CSV; skipping Cluster x School.")

# ---------------- School differences in Δ (ANOVA) ----------------
log("\n[RQ2] Testing school differences in Δ (per-QIP ANOVA)")
if METADATA_CSV and os.path.exists(METADATA_CSV):
    meta = pd.read_csv(METADATA_CSV).drop_duplicates(subset=[SUBJECT_COL])
    if SUBJECT_COL in meta.columns and SCHOOL_COL in meta.columns:
        meta['_merge_key'] = _norm_ids(meta[SUBJECT_COL])

        ds = delta.reset_index()
        if ds.columns[0] != SUBJECT_COL:
            ds = ds.rename(columns={ds.columns[0]: SUBJECT_COL})
        ds['_merge_key'] = _norm_ids(ds[SUBJECT_COL])

        ds = ds.merge(meta[['_merge_key', SCHOOL_COL]], on='_merge_key', how="left").dropna(subset=[SCHOOL_COL])

        dv_cols = [c for c in delta.columns if c in ds.columns]
        ds = ds.dropna(subset=dv_cols, how="any")

        if ds[SCHOOL_COL].nunique() >= 2 and len(dv_cols) >= 2 and len(ds) >= 3:
            from statsmodels.formula.api import ols
            from statsmodels.stats.anova import anova_lm
            rows = []
            for q in dv_cols:
                try:
                    model = ols('val ~ C(school)', data=pd.DataFrame({
                        "val": ds[q].values, "school": ds[SCHOOL_COL].values
                    })).fit()
                    aov = anova_lm(model, typ=2)
                    F = aov.loc["C(school)","F"]; p = aov.loc["C(school)","PR(>F)"]
                    df1 = aov.loc["C(school)","df"]; df2 = aov.loc["Residual","df"]
                    ss_effect = aov.loc["C(school)","sum_sq"]; ss_total = ss_effect + aov.loc["Residual","sum_sq"]
                    eta2 = ss_effect/ss_total if ss_total > 0 else np.nan
                    rows.append({"QIP": q, "F": F, "df1": df1, "df2": df2, "p_value": p, "eta2": eta2})
                except Exception:
                    rows.append({"QIP": q, "F": np.nan, "df1": np.nan, "df2": np.nan, "p_value": np.nan, "eta2": np.nan})
            aov_tbl = pd.DataFrame(rows).sort_values("p_value") if rows else pd.DataFrame(columns=["QIP","p_value"])
            if not aov_tbl.empty:
                rej, p_fdr = fdrcorrection(aov_tbl["p_value"].fillna(1.0).values, alpha=0.05, method='indep')
                aov_tbl["p_FDR"] = p_fdr
                aov_tbl["Significant_FDR_0.05"] = rej
                aov_tbl.to_csv(os.path.join(OUT_DIR, "rq2_anova_delta_by_school.csv"), index=False, encoding="utf-8")
                log(f"[RQ2-ANOVA] Δ-by-school ANOVAs: QIPs tested={len(aov_tbl)} | FDR-significant={int(aov_tbl['Significant_FDR_0.05'].sum())}")
            
# ---------------- Mixed-effects robustness (patched) ----------------
log("\n[ROBUSTNESS] Image-level robustness: simple MixedLM (subject random intercept) with fallback to cluster-robust OLS")

try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM
    HAVE_MIXED = True
except Exception:
    HAVE_MIXED = False
    log("[ROBUSTNESS] Statsmodels MixedLM unavailable; skipping robustness.")

MIN_SUBJECTS_FOR_MIXED = 6  # skip MixedLM for tiny N

def _make_long(df, time_label, qip_list):
    d = df[[SUBJECT_COL] + qip_list].copy()
    d["Time"] = time_label
    return d.melt(id_vars=[SUBJECT_COL, "Time"], var_name="QIP", value_name="Value")

if HAVE_MIXED and not paired_tbl.empty and len(common_subjects) >= MIN_SUBJECTS_FOR_MIXED:
    # Rank QIPs for robustness
    if "p_FDR" in paired_tbl.columns and not paired_tbl["p_FDR"].isna().all():
        ranked_qips = paired_tbl.sort_values("p_FDR")["QIP"].tolist()
    else:
        ranked_qips = paired_tbl.sort_values("p_value")["QIP"].tolist()

    top_qips = ranked_qips[:min(TOP_N_MIXED, len(ranked_qips))]
    if top_qips:
        long_before = _make_long(before, BEFORE_LABEL, top_qips)
        long_after  = _make_long(after,  AFTER_LABEL,  top_qips)
        long_all = pd.concat([long_before, long_after], ignore_index=True)

        rob_rows = []
        for q in top_qips:
            dfq = long_all[long_all["QIP"] == q]
            if dfq[SUBJECT_COL].nunique() < 2:
                continue

            # Try MixedLM with subject random intercept
            mixed_ok = False
            try:
                md  = MixedLM.from_formula("Value ~ C(Time)", groups=SUBJECT_COL, data=dfq)
                mdf = md.fit(reml=True, method="lbfgs", maxiter=200)
                coef_name = f"C(Time)[T.{AFTER_LABEL}]"
                if coef_name in mdf.params.index:
                    b = mdf.params[coef_name]
                    se = mdf.bse[coef_name]
                    z  = b / se if se > 0 else np.nan
                    p  = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else np.nan
                    rob_rows.append({"QIP": q, "model": "MixedLM", "beta_Time(After vs Before)": b, "SE": se, "z": z, "p_value": p})
                    mixed_ok = True
            except Exception:
                pass

            # Fallback: OLS with cluster-robust SEs
            if not mixed_ok:
                try:
                    ols = smf.ols("Value ~ C(Time)", data=dfq).fit(cov_type="cluster", cov_kwds={"groups": dfq[SUBJECT_COL]})
                    coef_name = f"C(Time)[T.{AFTER_LABEL}]"
                    if coef_name in ols.params.index:
                        b = ols.params[coef_name]
                        se = ols.bse[coef_name]
                        p  = ols.pvalues[coef_name]
                        rob_rows.append({"QIP": q, "model": "OLS_cluster", "beta_Time(After vs Before)": b, "SE": se, "z": np.nan, "p_value": p})
                except Exception:
                    rob_rows.append({"QIP": q, "model": "failed", "beta_Time(After vs Before)": np.nan, "SE": np.nan, "z": np.nan, "p_value": np.nan})

        if rob_rows:
            rob_tbl = pd.DataFrame(rob_rows)
            if "p_value" in rob_tbl.columns:
                rob_tbl = rob_tbl.sort_values("p_value")
            outp = os.path.join(OUT_DIR, "robust_time_effect_image_level.csv")
            rob_tbl.to_csv(outp, index=False, encoding="utf-8")
            n_sig = (rob_tbl["p_value"] < 0.05).sum() if "p_value" in rob_tbl else 0
            log(f"[ROBUSTNESS] Saved image-level robustness ({len(rob_tbl)} fits); p<.05 in {n_sig} (uncorrected). -> {outp}")
        else:
            log("[ROBUSTNESS] Mixed/OLS produced no analyzable rows; skipping table.")
    else:
        log("[ROBUSTNESS] No top QIPs available to test; skipping.")
else:
    reason = []
    if not HAVE_MIXED: reason.append("MixedLM unavailable")
    if paired_tbl.empty: reason.append("no paired results")
    if len(common_subjects) < MIN_SUBJECTS_FOR_MIXED: reason.append(f"n_subjects<{MIN_SUBJECTS_FOR_MIXED}")
    log("[ROBUSTNESS] Skipped (" + ", ".join(reason) + ").")

# ---------------- SUMMARY ----------------
log("\n[SUMMARY] Key outputs written to OUT_DIR:")
summary_files = [
    "portfolio_means_before.csv",
    "portfolio_means_after.csv",
    "portfolio_delta.csv",
    "portfolio_delta_withinSD.csv",
    "icc_reliability_by_qip_wave.csv",
    "rq1_paired_tests.csv",
    "rq1_within_subject_anova.csv" if USE_ANOVARM and HAVE_ANOVARM else None,
    "rq1_within_portfolio_consistency.csv",
    "figs_rq1/heatmap_delta_top.png",
    "subject_clusters_kmeans.csv",
    "cluster_by_school_crosstab.csv",
    "cluster_by_school_chi2.txt",
    "rq2_anova_delta_by_school.csv",
    "figs_clustering/dendrogram_subjects.png",
    "figs_clustering/silhouette_over_k.png",
    "figs_clustering/kmeans_clusters_on_delta_pca.png",
    "robust_time_effect_image_level.csv"
]
for f in summary_files:
    if f:
        p = os.path.join(OUT_DIR, f)
        if os.path.exists(p):
            print(" -", f)

log("\n[DONE] RQ1 & RQ2 analysis complete.")

# ---------------- Descriptive plots: Top-10 most-changing QIPs ----------------
try:
    desc_dir = os.path.join(OUT_DIR, "figs_rq1")
    os.makedirs(desc_dir, exist_ok=True)

    # --- pick top-10 QIPs by |effect| if paired_tbl exists, else by |mean Δ|
    if 'paired_tbl' in locals() and not paired_tbl.empty and 'Effect' in paired_tbl.columns:
        _rank = paired_tbl.copy()
        _rank['abs_eff'] = _rank['Effect'].abs()
        top_qips = _rank.sort_values(['abs_eff', 'p_value'], ascending=[False, True])['QIP'].tolist()[:10]
    else:
        # fallback: rank by absolute mean Δ
        _md = (delta.mean(axis=0)
               .reindex(delta.columns)
               .abs()
               .sort_values(ascending=False))
        top_qips = _md.index.tolist()[:10]

    top_qips = [q for q in top_qips if q in subj_before.columns and q in subj_after.columns]
    if len(top_qips) == 0:
        log("[DESC] No QIPs available for top-10 plots; skipping.")
    else:
        # ---------- Spaghetti plot grid (per subject lines) ----------
        import matplotlib.pyplot as plt
        n = len(top_qips)
        ncols = 2
        nrows = int(np.ceil(n / ncols))
        plt.figure(figsize=(10, max(6, 2.6 * nrows)))

        x = np.array([0, 1], dtype=float)
        for i, q in enumerate(top_qips, start=1):
            ax = plt.subplot(nrows, ncols, i)

            # per-subject lines
            y0 = subj_before[q].loc[common_subjects]
            y1 = subj_after[q].loc[common_subjects]
            # ensure aligned indices
            y0, y1 = y0.align(y1, join='inner')

            for s in y0.index:
                yy = [y0.loc[s], y1.loc[s]]
                ax.plot(x, yy, marker='o', linewidth=1.0, alpha=0.6)

            # mean line
            m0, m1 = np.nanmean(y0.values), np.nanmean(y1.values)
            ax.plot(x, [m0, m1], marker='o', linewidth=3.0, alpha=0.9)

            ax.set_xticks([0, 1], [BEFORE_LABEL, AFTER_LABEL])
            ax.set_title(q, fontsize=10)
            ax.grid(True, alpha=0.2)
            # tidy y-lims (pad 5%)
            y_all = np.r_[y0.values, y1.values]
            if np.isfinite(y_all).any():
                lo, hi = np.nanmin(y_all), np.nanmax(y_all)
                pad = (hi - lo) * 0.05 if hi > lo else (abs(hi) + 1e-6) * 0.05
                ax.set_ylim(lo - pad, hi + pad)

        plt.tight_layout()
        fp_spag = os.path.join(desc_dir, "top10_spaghetti.png")
        plt.savefig(fp_spag, dpi=DPI)
        plt.close()
        log(f"[FIG]   Top-10 QIPs spaghetti -> {fp_spag}")

        # ---------- Mean Δ bar with 95% CI ----------
        # compute Δ per subject (after - before) for the selected QIPs
        dsel = (subj_after[top_qips] - subj_before[top_qips]).copy()
        # stats across subjects
        mean_delta = dsel.mean(axis=0)
        sd = dsel.std(axis=0, ddof=1)
        n_subj = (dsel.notna()).sum(axis=0).astype(float)
        se = sd / np.sqrt(np.maximum(n_subj, 1.0))
        # 95% CI (t-based, df=n-1)
        from scipy.stats import t
        ci_mult = t.ppf(0.975, np.maximum(n_subj - 1.0, 1.0))
        ci = se * ci_mult

        # sort bars by |mean Δ|
        order = mean_delta.abs().sort_values(ascending=False).index.tolist()

        plt.figure(figsize=(max(8, 0.6 * len(order) + 2), 4))
        pos = np.arange(len(order))
        md = mean_delta.reindex(order).values
        ci_vals = ci.reindex(order).values

        # bars + error bars
        plt.bar(pos, md)
        plt.errorbar(pos, md, yerr=ci_vals, fmt='none', capsize=4, linewidth=1.2)

        plt.xticks(pos, order, rotation=45, ha='right')
        plt.ylabel("Mean Δ (after − before)")
        plt.title("Top-10 ΔQIPs — mean change with 95% CI")
        plt.grid(axis='y', alpha=0.2)
        plt.tight_layout()
        fp_bar = os.path.join(desc_dir, "top10_mean_delta_bar.png")
        plt.savefig(fp_bar, dpi=DPI)
        plt.close()
        log(f"[FIG]   Top-10 mean Δ bar -> {fp_bar}")

except Exception as e:
    log(f"[warn] Descriptive top-10 plots failed: {e}")