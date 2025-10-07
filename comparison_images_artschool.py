## Compare image properties before and in the first 2 years of artschool ##

# %% =========================== CONFIG ===========================
import os
import re
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection
import matplotlib.pyplot as plt
import config_img_comparison as CFG

# ---- use config values (no hardcoded paths/options here) ----
os.makedirs(CFG.OUT_DIR, exist_ok=True)

# Helpful: fail early with a clear message if a CSV path is wrong
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

# %% ========================= LOAD & CLEAN =========================
def load_numeric_overlap(a_path, b_path):
    a = pd.read_csv(a_path)
    b = pd.read_csv(b_path)

    # Keep only numeric columns that exist in BOTH
    a_num = a.select_dtypes(include=[np.number])
    b_num = b.select_dtypes(include=[np.number])
    common = a_num.columns.intersection(b_num.columns)

    a = a_num[common].copy()
    b = b_num[common].copy()

    # Drop columns that are completely NaN or zero-variance in BOTH sets
    keep = []
    for c in common:
        a_c = a[c].dropna()
        b_c = b[c].dropna()
        if (a_c.size + b_c.size) == 0:
            continue
        if (a_c.nunique() <= 1) and (b_c.nunique() <= 1):
            continue
        keep.append(c)
    return a[keep], b[keep]

df_before, df_year2 = load_numeric_overlap(CFG.CSV_BEFORE, CFG.CSV_YEAR2)

# Descriptives
desc_before = df_before.describe().T
desc_year2  = df_year2.describe().T
desc = pd.concat(
    [desc_before[['count','mean','std','min','50%','max']].rename(columns={'50%':'median'}),
     desc_year2[['count','mean','std','min','50%','max']].rename(columns={'50%':'median'})],
    axis=1, keys=['Before','After']
)
desc.to_csv(os.path.join(CFG.OUT_DIR, "descriptives.csv"), index=False)

# %% ====================== TESTS & EFFECT SIZES ====================
def hedges_g(x, y):
    # unbiased Cohen's d
    nx, ny = len(x), len(y)
    sx2, sy2 = np.var(x, ddof=1), np.var(y, ddof=1)
    # pooled SD (unequal n)
    sp = np.sqrt(((nx-1)*sx2 + (ny-1)*sy2) / (nx+ny-2)) if (nx+ny-2) > 0 else np.nan
    d = (np.mean(x) - np.mean(y)) / sp if sp > 0 else np.nan
    # small-sample correction
    J = 1 - (3/(4*(nx+ny)-9)) if (nx+ny) > 2 else 1.0
    return d * J

def rank_biserial_from_mwu(x, y):
    # r_rb = 1 - 2U/(n1*n2)
    n1, n2 = len(x), len(y)
    if n1 == 0 or n2 == 0:
        return np.nan
    U = stats.mannwhitneyu(x, y, alternative='two-sided')[0]
    return 1 - 2*U/(n1*n2)

rows = []
for col in df_before.columns:
    x = df_before[col].dropna().values
    y = df_year2[col].dropna().values

    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        continue

    # Shapiro only if 3 <= n <= 5000 (heuristic), otherwise skip normality
    def shapiro_p(z):
        if 3 <= len(z) <= 5000:
            try:
                return stats.shapiro(z)[1]
            except Exception:
                return np.nan
        return np.nan

    p_norm1 = shapiro_p(x)
    p_norm2 = shapiro_p(y)
    both_normal = (not np.isnan(p_norm1) and p_norm1 > 0.05) and (not np.isnan(p_norm2) and p_norm2 > 0.05)

    if both_normal:
        # Welch's t-test
        t_stat, pval = stats.ttest_ind(x, y, equal_var=False)
        test = "Welch t"
        effect = hedges_g(x, y)   # signed: Before - Year2
        eff_name = "Hedges_g (Before-After)"
    else:
        # Mann-Whitney U
        u_stat, pval = stats.mannwhitneyu(x, y, alternative='two-sided')
        t_stat = u_stat
        test = "Mann-Whitney U"
        effect = rank_biserial_from_mwu(x, y)  # signed: positive means Before > Year2
        eff_name = "Rank-biserial r (Before>After)"

    rows.append({
        "QIP": col,
        "n_Before": n1, "n_After": n2,
        "mean_Before": np.mean(x), "mean_After": np.mean(y),
        "std_Before": np.std(x, ddof=1), "std_After": np.std(y, ddof=1),
        "Test": test,
        "Statistic": t_stat,
        "p_value": pval,
        "Effect": effect,
        "Effect_name": eff_name,
        "Shapiro_p_Before": p_norm1,
        "Shapiro_p_After": p_norm2
    })

res = pd.DataFrame(rows).sort_values("p_value")
if not res.empty:
    # FDR correction
    reject, p_fdr = fdrcorrection(res["p_value"].values, alpha=0.05, method='indep')
    res["p_FDR"] = p_fdr
    res["Significant_FDR_0.05"] = reject
else:
    res["p_FDR"] = []
    res["Significant_FDR_0.05"] = []

res.to_csv(os.path.join(CFG.OUT_DIR, "inferential_results.csv"), index=False)
print("Saved:")
print(" -", os.path.join(CFG.OUT_DIR, "descriptives.csv"))
print(" -", os.path.join(CFG.OUT_DIR, "inferential_results.csv"))

# %% ====================== PLOTTING HELPERS ========================
def safe_title(s):
    return s.replace("_", " ")

def plot_violin_swarm(metric, savepath):
    # combine into long form
    a = pd.DataFrame({"value": df_before[metric], "Group": "Before"}).dropna()
    b = pd.DataFrame({"value": df_year2[metric],  "Group": "After"}).dropna()
    ab = pd.concat([a, b], ignore_index=True)
    if ab.empty:
        return
    plt.figure(figsize=(5,4))
    # Violin
    parts = plt.violinplot([a["value"].values, b["value"].values], showmeans=True, showmedians=False)
    # Swarm (rudimentary): jittered scatter
    rng = np.random.default_rng(42)
    for i, vals in enumerate([a["value"].values, b["value"].values], start=1):
        xj = rng.normal(loc=i, scale=0.04, size=len(vals))
        plt.plot(xj, vals, 'o', alpha=0.35, markersize=3)
    plt.xticks([1,2], ["Before", "After"])
    plt.ylabel(metric)
    plt.title(safe_title(metric))
    plt.tight_layout()
    plt.savefig(savepath, dpi=CFG.DPI)
    plt.close()

def assign_group(metric_name):
    name = metric_name.lower()
    for g, patterns in CFG.GROUP_RULES.items():
        for pat in patterns:
            if re.search(pat.lower(), name):
                return g
    return "Other"

# %% ===================== FIGURES: TOP-N METRICS ===================
if not res.empty:
    top = res.sort_values("p_FDR").head(CFG.TOP_N_PLOTS)
    figs_dir = os.path.join(CFG.OUT_DIR, "figs_top_metrics")
    os.makedirs(figs_dir, exist_ok=True)
    for m in top["QIP"]:
        png = os.path.join(figs_dir, f"{m.replace(os.sep,'_').replace(' ','_')}.png")
        try:
            plot_violin_swarm(m, png)
        except Exception as e:
            print(f"[warn] could not plot {m}: {e}")

# %% ===================== FIGURES: METRIC GROUPS ===================
# Compute standardized effects per metric, then summarize per group.
# We'll use signed Hedges' g when available; otherwise convert rank-biserial r ~> approx d via d = r * pi/np.sqrt(3)
# (purely for visualization comparability)
plot_rows = []
for _, row in res.iterrows():
    m = row["QIP"]
    eff = row["Effect"]
    if pd.isna(eff):
        continue
    if "Hedges_g" in row["Effect_name"]:
        d = eff
    else:
        # rank-biserial to Cohen's d (approx):
        d = eff * (np.pi / np.sqrt(3))
    plot_rows.append({"QIP": m, "d_std": d, "Group": assign_group(m)})

df_eff = pd.DataFrame(plot_rows)
if not df_eff.empty:
    grp = df_eff.groupby("Group")["d_std"]
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
    plt.ylabel("Mean standardized effect (Before − After)")
    plt.title("Metric-group differences (mean effect ± SE across metrics)")
    plt.tight_layout()
    plt.savefig(os.path.join(CFG.OUT_DIR, "metric_groups_bar.png"), dpi=CFG.DPI)
    plt.close()

# %% =================== OPTIONAL: SUMMARY PRINT ====================
if not res.empty:
    show_cols = ["QIP","Test","p_value","p_FDR","Significant_FDR_0.05",
                 "Effect_name","Effect","mean_Before","mean_After","n_Before","n_After"]
    print("\nTop differences (by FDR):")
    print(res.sort_values("p_FDR")[show_cols].head(15).to_string(index=False))
else:
    print("No comparable numeric metrics found between the two CSVs.")
