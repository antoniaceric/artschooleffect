## Compare image properties (multi-subject): before vs after with inter-individual analyses ##

# =========================== IMPORTS & CONFIG ===========================
import os, sys, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

# Ensure we import config from this script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import importlib
import config_img_comparison as CFG
importlib.reload(CFG)

np.random.seed(getattr(CFG, "RANDOM_SEED", 42))

SUBJECT_COL = getattr(CFG, "SUBJECT_COL", "subject")
BEFORE_LABEL = getattr(CFG, "BEFORE_LABEL", "before")
AFTER_LABEL  = getattr(CFG, "AFTER_LABEL", "year2")

OUT_DIR = CFG.OUT_DIR
os.makedirs(OUT_DIR, exist_ok=True)

# Sanity check files
for _p in (CFG.CSV_BEFORE, CFG.CSV_YEAR2):
    if not os.path.exists(_p):
        _folder = os.path.dirname(_p) or "."
        _have = []
        if os.path.isdir(_folder):
            try:
                _have = [f for f in os.listdir(_folder) if f.lower().endswith(".csv")]
            except Exception:
                pass
        raise FileNotFoundError(
            f"\nCSV not found:\n  {_p}\n"
            f"Folder contents ({os.path.abspath(_folder)}):\n  " + ("\n  ".join(_have) if _have else "(no CSVs found)")
        )

print("Using config:", getattr(CFG, "__file__", "<unknown>"))
print("CSV_BEFORE:", CFG.CSV_BEFORE)
print("CSV_YEAR2 :", CFG.CSV_YEAR2)

# ================================ LOAD =================================
def load_numeric_overlap_with_subject(a_path, b_path, subject_col=SUBJECT_COL):
    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)
    if subject_col not in a.columns or subject_col not in b.columns:
        raise ValueError(f"'{subject_col}' column is required in both merged CSVs.")

    # numeric overlap
    a_num = a.select_dtypes(include=[np.number])
    b_num = b.select_dtypes(include=[np.number])
    common = a_num.columns.intersection(b_num.columns)

    # keep subject col + numeric common
    a = pd.concat([a[[subject_col]], a_num[common]], axis=1)
    b = pd.concat([b[[subject_col]], b_num[common]], axis=1)

    # drop all-NaN/zero-variance cols in BOTH sets
    keep = []
    for c in common:
        a_c = a[c].dropna()
        b_c = b[c].dropna()
        if (a_c.size + b_c.size) == 0:
            continue
        if (a_c.nunique() <= 1) and (b_c.nunique() <= 1):
            continue
        keep.append(c)

    a = pd.concat([a[[subject_col]], a[keep]], axis=1)
    b = pd.concat([b[[subject_col]], b[keep]], axis=1)
    return a, b, keep

df_before_all, df_after_all, metric_cols = load_numeric_overlap_with_subject(CFG.CSV_BEFORE, CFG.CSV_YEAR2)

# ====================== GROUP-LEVEL (all images pooled) =================
# (Retain your original approach for comparability with previous results)
def descriptives_by_time(df_before_all, df_after_all, metric_cols):
    desc_before = df_before_all[metric_cols].describe().T
    desc_after  = df_after_all[metric_cols].describe().T
    desc = pd.concat(
        [desc_before[['count','mean','std','min','50%','max']].rename(columns={'50%':'median'}),
         desc_after [['count','mean','std','min','50%','max']].rename(columns={'50%':'median'})],
        axis=1, keys=[BEFORE_LABEL, AFTER_LABEL]
    )
    return desc

group_desc = descriptives_by_time(df_before_all, df_after_all, metric_cols)
group_desc.to_csv(os.path.join(OUT_DIR, "descriptives_group_level.csv"), index=True)

# Classic independent-sample tests on pooled rows (as before)
def hedges_g(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    sp_den = (nx + ny - 2)
    if sp_den <= 0:
        return np.nan
    sp = np.sqrt(((nx-1)*sx2 + (ny-1)*sy2) / sp_den)
    if sp <= 0:
        return np.nan
    d = (np.mean(x) - np.mean(y)) / sp
    J = 1 - (3/(4*(nx + ny) - 9)) if (nx + ny) > 2 else 1.0
    return d * J

def rank_biserial_from_mwu(x, y):
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan
    U = stats.mannwhitneyu(x, y, alternative='two-sided')[0]
    return 1 - 2*U/(n1*n2)

rows = []
for col in metric_cols:
    x = df_before_all[col].dropna().values
    y = df_after_all[col].dropna().values
    if len(x) < 2 or len(y) < 2:
        continue

    def shapiro_p(z):
        if 3 <= len(z) <= 5000:
            try: return stats.shapiro(z)[1]
            except Exception: return np.nan
        return np.nan

    p_norm1, p_norm2 = shapiro_p(x), shapiro_p(y)
    both_normal = (not np.isnan(p_norm1) and p_norm1 > 0.05) and (not np.isnan(p_norm2) and p_norm2 > 0.05)

    if both_normal:
        stat, pval = stats.ttest_ind(x, y, equal_var=False)
        test = "Welch t"
        effect = hedges_g(x, y)   # signed: Before - After
        eff_name = "Hedges_g (Before-After)"
    else:
        u, pval = stats.mannwhitneyu(x, y, alternative='two-sided')
        stat = u
        test = "Mann-Whitney U"
        effect = rank_biserial_from_mwu(x, y)  # positive => Before > After
        eff_name = "Rank-biserial r (Before>After)"

    rows.append({
        "QIP": col,
        "Test": test,
        "Statistic": float(stat),
        "p_value": float(pval),
        "Effect_name": eff_name,
        "Effect": float(effect) if np.isfinite(effect) else np.nan,
        "mean_Before": float(np.mean(x)) if len(x) else np.nan,
        "mean_After":  float(np.mean(y)) if len(y) else np.nan,
        "n_rows_Before": int(len(x)),
        "n_rows_After":  int(len(y)),
    })

group_tests = pd.DataFrame(rows).sort_values("p_value")
if not group_tests.empty:
    rej, p_fdr = fdrcorrection(group_tests["p_value"].values, alpha=0.05, method='indep')
    group_tests["p_FDR"] = p_fdr
    group_tests["Significant_FDR_0.05"] = rej

group_tests.to_csv(os.path.join(OUT_DIR, "inferential_group_level.csv"), index=False)

# ===================== SUBJECT-LEVEL (paired across subjects) ===========
# Per-subject means per timepoint
subj_mean_before = df_before_all.groupby(SUBJECT_COL)[metric_cols].mean()
subj_mean_after  = df_after_all.groupby(SUBJECT_COL)[metric_cols].mean()

# Align subjects present in both timepoints
common_subjects = subj_mean_before.index.intersection(subj_mean_after.index)
subj_mean_before = subj_mean_before.loc[common_subjects].sort_index()
subj_mean_after  = subj_mean_after.loc[common_subjects].sort_index()

# Deltas per subject and metric: After - Before
delta = subj_mean_after - subj_mean_before
delta.to_csv(os.path.join(OUT_DIR, "subject_metric_delta.csv"), index=True)

# Paired tests across subjects per metric
def cohens_dz(diff):
    # mean(diff) / sd(diff), sd with ddof=1
    if len(diff) < 2: return np.nan
    sd = np.std(diff, ddof=1)
    return np.mean(diff) / sd if sd > 0 else np.nan

paired_rows = []
for col in metric_cols:
    # Pairwise diffs
    d = delta[col].dropna().values
    n = len(d)
    if n < 2:
        continue

    # Normality of diffs (Shapiro)
    p_norm = np.nan
    if 3 <= n <= 5000:
        try:
            p_norm = stats.shapiro(d)[1]
        except Exception:
            p_norm = np.nan

    if not np.isnan(p_norm) and p_norm > 0.05:
        # Paired t-test
        t_stat, pval = stats.ttest_1samp(d, 0.0, nan_policy='omit')
        test = "Paired t (diff vs 0)"
        effect = cohens_dz(d)  # signed: After - Before
        eff_name = "Cohen_dz (After-Before)"
    else:
        # Wilcoxon signed-rank
        # If all diffs are zero, wilcoxon fails; catch it
        try:
            w_stat, pval = stats.wilcoxon(d, zero_method="wilcox", correction=False, alternative="two-sided", mode="auto")
        except Exception:
            w_stat, pval = (np.nan, 1.0)
        t_stat = w_stat
        test = "Wilcoxon signed-rank"
        # rank-biserial for paired can be approximated:
        pos = np.sum(d > 0); neg = np.sum(d < 0)
        n_pairs = pos + neg
        r_rb = (pos - neg) / n_pairs if n_pairs > 0 else np.nan
        effect = r_rb
        eff_name = "Rank-biserial (paired)"

    paired_rows.append({
        "QIP": col,
        "Test": test,
        "Statistic": float(t_stat) if np.isfinite(t_stat) else np.nan,
        "p_value": float(pval),
        "Effect_name": eff_name,
        "Effect": float(effect) if np.isfinite(effect) else np.nan,
        "n_subjects": int(n),
        "mean_delta": float(np.mean(d)),
        "sd_delta": float(np.std(d, ddof=1)) if n > 1 else np.nan
    })

paired_tests = pd.DataFrame(paired_rows).sort_values("p_value")
if not paired_tests.empty:
    rej, p_fdr = fdrcorrection(paired_tests["p_value"].values, alpha=0.05, method='indep')
    paired_tests["p_FDR"] = p_fdr
    paired_tests["Significant_FDR_0.05"] = rej

paired_tests.to_csv(os.path.join(OUT_DIR, "inferential_subject_level_paired.csv"), index=False)

# ------------- Inter-individual similarity (pattern of change) ----------
# Build subject change vectors (Δ across metrics), standardized per metric for comparability
delta_std = (delta - delta.mean(axis=0)) / delta.std(axis=0, ddof=1)
# subject-by-subject similarity (Pearson corr of standardized delta vectors)
subj_corr = delta_std.T.corr()  # corr over metrics
subj_corr.to_csv(os.path.join(OUT_DIR, "subject_change_similarity_corr.csv"))

# Summary: average pairwise similarity (off-diagonal mean)
if subj_corr.shape[0] > 1:
    m = subj_corr.values
    avg_sim = (np.sum(m) - np.trace(m)) / (m.shape[0]*(m.shape[0]-1))
else:
    avg_sim = np.nan

with open(os.path.join(OUT_DIR, "interindividual_summary.txt"), "w") as f:
    f.write(f"Average pairwise similarity of change (Pearson r across metrics): {avg_sim:.3f}\n")
    f.write(f"Subjects included: {', '.join(map(str, delta_std.index.tolist()))}\n")

# ================================ FIGURES ================================
# Helpers
def safe_title(s): return s.replace("_"," ")

# 1) Spaghetti plots (subject trajectories) for top metrics by paired FDR
spag_dir = os.path.join(OUT_DIR, "figs_spaghetti")
os.makedirs(spag_dir, exist_ok=True)
TOP_N_SPAG = getattr(CFG, "SPAGHETTI_TOP_N", 8)

if not paired_tests.empty:
    top_metrics = paired_tests.sort_values("p_FDR").head(TOP_N_SPAG)["QIP"].tolist()
else:
    top_metrics = metric_cols[:TOP_N_SPAG]

for m in top_metrics:
    if m not in subj_mean_before.columns: 
        continue
    plt.figure(figsize=(5.8,4.2))
    x = [0, 1]
    for subj in common_subjects:
        y0 = subj_mean_before.loc[subj, m]
        y1 = subj_mean_after.loc[subj, m]
        if np.isfinite(y0) and np.isfinite(y1):
            plt.plot(x, [y0, y1], marker='o', alpha=0.7)
            # add subject label near the AFTER point
            plt.text(
                x[1] + 0.02, y1, str(subj),
                fontsize=8, va='center'
            )
    plt.xticks(x, [BEFORE_LABEL, AFTER_LABEL])
    plt.title(f"{safe_title(m)} (subject means)")
    plt.ylabel(m)
    plt.tight_layout()
    plt.savefig(os.path.join(spag_dir, f"{re.sub(r'[^A-Za-z0-9_-]+','_',m)}.png"), dpi=getattr(CFG, "DPI", 200))
    plt.close()

# 2) Heatmap of per-subject changes (Δ = After - Before)
heat_dir = os.path.join(OUT_DIR, "figs_heatmaps")
os.makedirs(heat_dir, exist_ok=True)

# Choose up to HEATMAP_MAX_METRICS by strongest |effect| (paired dz or r)
HEAT_CAP = getattr(CFG, "HEATMAP_MAX_METRICS", 40)
if not paired_tests.empty:
    # order by absolute standardized effect where available (Cohen dz preferred)
    def _score(row):
        e = row["Effect"]
        if pd.isna(e): return 0.0
        return abs(e)
    ranked = paired_tests.copy()
    ranked["abs_effect"] = ranked.apply(_score, axis=1)
    sel_metrics = ranked.sort_values(["Significant_FDR_0.05","abs_effect"], ascending=[False,False])["QIP"].tolist()
else:
    sel_metrics = list(metric_cols)

sel_metrics = sel_metrics[:HEAT_CAP]
delta_sel = delta[sel_metrics].copy()

# Z-score across subjects (columns = metrics)
delta_sel_z = (delta_sel - delta_sel.mean(axis=0)) / delta_sel.std(axis=0, ddof=1)

plt.figure(figsize=(max(6, 0.3*len(sel_metrics)+3), max(4, 0.25*len(delta_sel_z)+2)))
im = plt.imshow(delta_sel_z.values, aspect='auto', interpolation='nearest')
plt.colorbar(im, fraction=0.046, pad=0.04, label="Z(Δ) across subjects")
plt.yticks(range(len(delta_sel_z.index)), delta_sel_z.index)
plt.xticks(range(len(sel_metrics)), [safe_title(m) for m in sel_metrics], rotation=45, ha='right')
plt.title("Per-subject change (z-scored per metric)")
plt.tight_layout()
plt.savefig(os.path.join(heat_dir, "heatmap_subject_changes.png"), dpi=getattr(CFG, "DPI", 200))
plt.close()

# 3) PCA of change vectors (subjects positioned by their overall change pattern)
from sklearn.decomposition import PCA

# Drop metrics with any NaN in delta for PCA
delta_pca_input = delta.dropna(axis=1, how='any')
pcs = min(getattr(CFG, "PCA_N_COMPONENTS", 2), max(1, min(delta_pca_input.shape)-1))
if pcs >= 2 and delta_pca_input.shape[0] >= 2 and delta_pca_input.shape[1] >= 2:
    pca = PCA(n_components=pcs, random_state=getattr(CFG, "RANDOM_SEED", 42))
    X = delta_pca_input.values
    # Standardize metrics columns (z across subjects)
    X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)
    Z = pca.fit_transform(X)
    exp = pca.explained_variance_ratio_[:2] * 100.0

    plt.figure(figsize=(6,5))
    plt.scatter(Z[:,0], Z[:,1])
    for i, subj in enumerate(delta_pca_input.index):
        plt.text(Z[i,0], Z[i,1], str(subj), fontsize=9, ha='left', va='bottom')
    plt.xlabel(f"PC1 ({exp[0]:.1f}%)")
    plt.ylabel(f"PC2 ({exp[1]:.1f}%)")
    plt.title("PCA of subject change vectors (Δ)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pca_subject_changes.png"), dpi=getattr(CFG, "DPI", 200))
    plt.close()

    # %% ====================== GROUP-LEVEL PLOTTING ======================
def safe_title(s): 
    return s.replace("_", " ")

def plot_violin_swarm_group(metric, savepath):
    """
    Group-level: plot pooled rows (all images across all subjects)
    for BEFORE vs AFTER as violin + jittered swarm.
    """
    a = pd.DataFrame({"value": df_before_all[metric], "Group": BEFORE_LABEL}).dropna()
    b = pd.DataFrame({"value": df_after_all[metric],  "Group": AFTER_LABEL}).dropna()
    ab = pd.concat([a, b], ignore_index=True)
    if ab.empty:
        return

    plt.figure(figsize=(5,4))
    # Violin (in Before/After order)
    parts = plt.violinplot(
        [a["value"].values, b["value"].values],
        showmeans=True, showmedians=False
    )
    # Jittered swarm
    rng = np.random.default_rng(getattr(CFG, "RANDOM_SEED", 42))
    for i, vals in enumerate([a["value"].values, b["value"].values], start=1):
        if len(vals) == 0:
            continue
        xj = rng.normal(loc=i, scale=0.04, size=len(vals))
        plt.plot(xj, vals, 'o', alpha=0.35, markersize=3)

    plt.xticks([1,2], [BEFORE_LABEL, AFTER_LABEL])
    plt.ylabel(metric)
    plt.title(f"{safe_title(metric)} (group level)")
    plt.tight_layout()
    plt.savefig(savepath, dpi=getattr(CFG, "DPI", 200))
    plt.close()

def assign_group(metric_name):
    name = metric_name.lower()
    for g, patterns in CFG.GROUP_RULES.items():
        for pat in patterns:
            if re.search(pat.lower(), name):
                return g
    return "Other"

# ---- Top-N group-level metrics by FDR (falls back to raw p if FDR missing) ----
group_figs_dir = os.path.join(OUT_DIR, "figs_group_top_metrics")
os.makedirs(group_figs_dir, exist_ok=True)

if "p_FDR" in group_tests.columns and not group_tests["p_FDR"].isna().all():
    top_group = group_tests.sort_values("p_FDR").head(CFG.TOP_N_PLOTS)
else:
    top_group = group_tests.sort_values("p_value").head(CFG.TOP_N_PLOTS)

for m in top_group["QIP"]:
    png = os.path.join(group_figs_dir, f"{re.sub(r'[^A-Za-z0-9_-]+','_',m)}.png")
    try:
        plot_violin_swarm_group(m, png)
    except Exception as e:
        print(f"[warn] could not plot group metric {m}: {e}")

# ---- Group-level metric groups bar (using group-level effects) ----
# Convert effects to a common scale for visualization:
#   Hedges' g stays as-is; rank-biserial r -> approx d via r * pi/sqrt(3)
plot_rows_group = []
for _, row in group_tests.iterrows():
    m = row["QIP"]
    eff = row.get("Effect", np.nan)
    if pd.isna(eff):
        continue
    if "Hedges_g" in str(row.get("Effect_name", "")):
        d = eff
    else:
        d = eff * (np.pi / np.sqrt(3))  # compare-ish scale for plotting
    plot_rows_group.append({"QIP": m, "d_std": d, "Group": assign_group(m)})

df_eff_group = pd.DataFrame(plot_rows_group)
if not df_eff_group.empty:
    grp = df_eff_group.groupby("Group")["d_std"]
    group_means = grp.mean().sort_values(key=lambda s: s.abs(), ascending=False)
    group_se = grp.apply(lambda x: x.std(ddof=1)/np.sqrt(len(x)))  # SE across metrics in group

    plt.figure(figsize=(7,4.5))
    order = group_means.index.tolist()
    y = group_means.values
    yerr = group_se.reindex(order).values
    x = np.arange(len(order))
    plt.bar(x, y, yerr=yerr, capsize=4)
    plt.axhline(0, linestyle="--", linewidth=1)
    plt.xticks(x, order, rotation=20, ha='right')
    plt.ylabel(f"Mean standardized effect ({BEFORE_LABEL} − {AFTER_LABEL})")
    plt.title("Group-level metric-group differences (mean effect ± SE)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "metric_groups_bar_group_level.png"), dpi=getattr(CFG, "DPI", 200))
    plt.close()


# =============================== SUMMARY ===============================
print("\nSaved:")
print(" -", os.path.join(OUT_DIR, "descriptives_group_level.csv"))
print(" -", os.path.join(OUT_DIR, "inferential_group_level.csv"))
print(" -", os.path.join(OUT_DIR, "subject_metric_delta.csv"))
print(" -", os.path.join(OUT_DIR, "inferential_subject_level_paired.csv"))
print(" -", os.path.join(OUT_DIR, "subject_change_similarity_corr.csv"))
print(" -", os.path.join(OUT_DIR, "interindividual_summary.txt"))
print(" -", os.path.join(spag_dir, "/*.png"))
print(" -", os.path.join(heat_dir, "heatmap_subject_changes.png"))
print(" -", os.path.join(OUT_DIR, "pca_subject_changes.png"))
print(" -", os.path.join(group_figs_dir, "/*.png"))
print(" -", os.path.join(OUT_DIR, "metric_groups_bar_group_level.png"))


# %% ===================== SUBJECT CLUSTERING & SIMILARITY =====================
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import seaborn as sns

# --- Prepare subject-level change data (delta is already After-Before) ---
delta_std = delta.dropna(axis=1, how='any')  # drop metrics with missing for some subjects
if not delta_std.empty:
    # Z-score metrics across subjects to give them equal weight
    scaler = StandardScaler()
    X = scaler.fit_transform(delta_std.values)

    # --- 1) Hierarchical clustering ---
    Z = linkage(X, method="ward")  # hierarchical clustering on subjects
    dendro_dir = os.path.join(OUT_DIR, "figs_clustering")
    os.makedirs(dendro_dir, exist_ok=True)

    plt.figure(figsize=(6, 4))
    dendrogram(Z, labels=delta_std.index.tolist(), leaf_rotation=90)
    plt.title("Hierarchical clustering of subjects (Δ profiles)")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(os.path.join(dendro_dir, "dendrogram_subjects.png"), dpi=getattr(CFG, "DPI", 200))
    plt.close()

    # --- 2) Clustered heatmap of subject deltas ---
    plt.figure(figsize=(8, max(4, 0.4*len(delta_std))))
    sns.clustermap(
        delta_std, method="ward", cmap="coolwarm", center=0,
        yticklabels=True, xticklabels=True,
        figsize=(10, max(6, 0.35*len(delta_std))),
    )
    plt.savefig(os.path.join(dendro_dir, "heatmap_clustered.png"), dpi=getattr(CFG, "DPI", 200))
    plt.close()

    # --- 3) Subject similarity matrix (correlations between change vectors) ---
    corr = np.corrcoef(X)
    corr_df = pd.DataFrame(corr, index=delta_std.index, columns=delta_std.index)
    corr_df.to_csv(os.path.join(OUT_DIR, "subject_change_similarity_corr.csv"))

    plt.figure(figsize=(6, 5))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="vlag", center=0,
                cbar_kws={"label": "Correlation of Δ vectors"})
    plt.title("Inter-subject similarity of change profiles")
    plt.tight_layout()
    plt.savefig(os.path.join(dendro_dir, "heatmap_subject_similarity.png"), dpi=getattr(CFG, "DPI", 200))
    plt.close()

    # --- 4) Optionally extract cluster assignments ---
    clusters = fcluster(Z, t=2, criterion="maxclust")  # adjust 't' for desired cluster count
    cluster_assignments = pd.DataFrame({
        "Subject": delta_std.index,
        "Cluster": clusters
    })
    cluster_assignments.to_csv(os.path.join(OUT_DIR, "subject_clusters.csv"), index=False)

    print("Saved subject clustering results:")
    print(" -", os.path.join(dendro_dir, "dendrogram_subjects.png"))
    print(" -", os.path.join(dendro_dir, "heatmap_clustered.png"))
    print(" -", os.path.join(OUT_DIR, "subject_change_similarity_corr.csv"))
    print(" -", os.path.join(dendro_dir, "heatmap_subject_similarity.png"))
    print(" -", os.path.join(OUT_DIR, "subject_clusters.csv"))
else:
    print("[warn] Not enough data for clustering (delta_std empty).")
