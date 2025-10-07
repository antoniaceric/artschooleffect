# The Art School Effect

This repository contains scripts to analyze **Quantitative Image Properties (QIPs)** in artworks created **before and after art school**.  
It forms the **second step** of the workflow: after extracting QIPs with the **Aesthetics Toolbox** (Redies et al. 2025), these scripts merge, compare, and statistically evaluate the results to trace measurable changes in artistic development.

---

## Step 1 — Analyze images with the Aesthetics Toolbox

Before running any scripts here, you need to extract QIP values for your artworks.

1. Download and install the **Aesthetics Toolbox** (Redies et al., 2025).  
   → https://github.com/rbartho/Aesthetics-Toolbox  
2. Use it to process your image folders (e.g., *before* and *after* art school*).  
3. Each subject or group should end up with CSV files containing QIP values.

Once you have these CSVs, continue with the analysis scripts below.

---

## Step 2 — QIP Analysis Before vs. After

| Script | Purpose |
|--------|----------|
| **config_img_comparison.py** | Central configuration file – set all paths, filenames, and analysis options here. |
| **merge_subjects.py** | Merges individual subject CSVs (from the Aesthetics Toolbox) into combined datasets for each timepoint. |
| **comparison_images_artschool.py** | Runs overall comparisons between *before* and *after* artworks (pooled image level). |
| **comparison_multiple_subjects.py** | Performs subject-level paired tests, PCA, and clustering across participants. |
| **main_analyses.py** | Executes the complete analysis pipeline: reliability (ICC), paired stats, ANOVA, clustering, and school-level effects. |

### Usage
1. Adjust parameters in `config_img_comparison.py`
2. Run the desired script, for example:
   ```bash
   python main_analyses.py
   ```

---

### Requirements
Install dependencies:
```bash
pip install -r requirements.txt
```
or  
```bash
conda env create -f environment.yml
conda activate aesth
```

---

### Output
Results are saved in the configured output folder (`OUT_DIR`):

- CSVs: statistics, ICC, and clustering results  
- Plots: heatmaps, PCA, bar charts, and violin plots  
- Optional metadata-based analyses (e.g., by school, medium, or year)

---

### Background
This toolkit is part of *The Art School Effect* project, investigating how formal art education shapes the visual and structural properties of artistic production through computational image analysis.

---

### Contact
**Antonia Ceric**  
antonia.ceric@gmail.com

