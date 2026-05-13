# HH Table 4: theme determinants. Regress bank-year loadings on
# lagged accounting chars with year + SIC4 FE.

import os
import warnings
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def normal_sf(x):
    """Survival function for standard normal (1 - CDF). No scipy needed."""
    return 0.5 * math.erfc(x / math.sqrt(2))

# Paths
BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE, "..", "data")
OUTPUT_DIR = os.path.join(BASE, "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOAD_PATH   = os.path.join(DATA_DIR, "outputs_textual_factors_v2",
                           "bank_year_loadings_v7_xsecstd.csv")
FUND_PATH   = os.path.join(DATA_DIR, "bank_fundamentals_hh_extended.csv")
SAMPLE_KEYS_PATH = os.path.join(OUTPUT_DIR, "main_regression_sample_keys.csv")
THEMES_PATH = os.path.join(DATA_DIR, "outputs_textual_factors_v2",
                           "final_themes.csv")


# 1. LOAD AND MERGE DATA

print("Loading data...")
loadings = pd.read_csv(LOAD_PATH)
fund     = pd.read_csv(FUND_PATH)
themes   = pd.read_csv(THEMES_PATH)

# Build label mapping: top 5 words per cluster (consistent with 18c)
theme_labels = {}
for _, r in themes.iterrows():
    cid = int(r["cluster_id"])
    tw = str(r.get("top_words", ""))
    words = [w.strip().replace("_", " ") for w in tw.split(",")][:5]
    words = [w for w in words if w]
    theme_labels[cid] = ", ".join(words) if words else f"Cluster {cid}"

# Short labels for heatmap (top 2 words only)
theme_labels_short = {}
for _, r in themes.iterrows():
    cid = int(r["cluster_id"])
    tw = str(r.get("top_words", ""))
    words = [w.strip().replace("_", " ") for w in tw.split(",")][:2]
    words = [w for w in words if w]
    theme_labels_short[cid] = ", ".join(words) if words else f"Cluster {cid}"

# Theme columns
theme_cols = [c for c in loadings.columns if c.startswith("topic_loading_")]
print(f"  Loadings: {len(loadings):,} bank-years, {len(theme_cols)} themes")
print(f"  Fundamentals: {len(fund):,} bank-years")

# Merge: loadings have CIK, fundamentals have permno → link via permno
# (Use permno-based linking to match 11_build_pairwise_dataset.py)
link_path = os.path.join(DATA_DIR, "permno_cik_wrds_extended.csv")
link = pd.read_csv(link_path)
link = link.dropna(subset=["cik","permno"])
link["cik"] = link["cik"].astype(int)
link_permno = link.drop_duplicates(subset=["permno"])[["permno","cik"]]

fund = fund.merge(link_permno, on="permno", how="inner")
fund["cik"] = fund["cik"].astype(int)
# A few gvkeys share a cik (parent-subsidiary mergers); dedupe on (cik, fyear)
fund = fund.drop_duplicates(subset=["cik","fyear"], keep="first")

# Loadings are for year t; use fund from fyear t (contemporaneous).
# Rationale: this matches the fund observation the MAIN pairwise
# regression uses for this bank-year (in 16j the loading at t−1 and
# fund at t−1 are paired at reg_year t; here we re-express that pair
# as loading_t + fund_t). Ensures 18d runs on exactly the same
# population as the main regression after the sample-keys filter.
loadings["cik"] = loadings["cik"].astype(int)
fund_lag = fund.copy()
fund_lag["year"] = fund_lag["fyear"]

MERGE_VARS = ["cik", "year", "log_assets", "log_age", "cash_assets",
              "loans_assets", "capital", "neg_earn", "sich"]
merged = loadings.merge(fund_lag[MERGE_VARS], on=["cik", "year"], how="inner")
print(f"  Merged: {len(merged):,} bank-years with both loadings and fundamentals")

# Restrict to the main pairwise regression sample (consistency with 16j)
keys = pd.read_csv(SAMPLE_KEYS_PATH)
keys["cik"] = keys["cik"].astype(int)
keys = keys.rename(columns={"loading_year": "year"})[["cik", "year"]]
merged = merged.merge(keys, on=["cik", "year"], how="inner")
print(f"  After restricting to main regression sample: {len(merged):,} bank-years "
      f"({merged['cik'].nunique()} banks)")


# 2. RUN THEME-BY-THEME REGRESSIONS (with SIC4 fixed effects)

print("\nRunning theme determinant regressions (with SIC4 FE)...")

CONTROLS = ["log_assets", "log_age", "cash_assets", "loans_assets",
            "capital", "neg_earn"]

for c in CONTROLS:
    merged[c] = pd.to_numeric(merged[c], errors="coerce")

# SIC4 fixed effects (group rare codes into "other")
merged["sich"] = pd.to_numeric(merged["sich"], errors="coerce")
merged["sic4"] = merged["sich"].astype("Int64").astype(str)
sic4_counts = merged["sic4"].value_counts()
valid_sic4 = sic4_counts[sic4_counts >= 5].index
merged.loc[~merged["sic4"].isin(valid_sic4), "sic4"] = "other"
n_sic4 = merged["sic4"].nunique()
print(f"  SIC4 fixed effects: {n_sic4} groups")

# Year fixed effects
merged["year_fe"] = merged["year"].astype(str)
n_years = merged["year_fe"].nunique()
print(f"  Year fixed effects: {n_years} years")

merged_clean = merged.dropna(subset=CONTROLS)
print(f"  Clean sample: {len(merged_clean):,} bank-years ({merged_clean['cik'].nunique()} banks)")

# OLS with FE via dummy variables (numpy)
# Build design matrix: controls + SIC4 dummies + year dummies + intercept
print("  Building design matrix with FE dummies...")
X_ctrl = merged_clean[CONTROLS].values  # (N, k)

# SIC4 dummies (drop first for identification)
sic4_dummies = pd.get_dummies(merged_clean["sic4"], prefix="sic", drop_first=True).values
year_dummies = pd.get_dummies(merged_clean["year_fe"], prefix="yr", drop_first=True).values
intercept = np.ones((len(merged_clean), 1))

X_full = np.hstack([intercept, X_ctrl, sic4_dummies, year_dummies])
print(f"  Design matrix: {X_full.shape[0]} obs × {X_full.shape[1]} regressors "
      f"({len(CONTROLS)} controls + {sic4_dummies.shape[1]} SIC4 + {year_dummies.shape[1]} year + intercept)")

# Pre-compute (X'X)^{-1} X' for speed (shared across all 39 regressions)
XtX_inv = np.linalg.inv(X_full.T @ X_full)
XtX_inv_Xt = XtX_inv @ X_full.T

n_obs = X_full.shape[0]
n_params = X_full.shape[1]
k_ctrl = len(CONTROLS)

results = []
for col in sorted(theme_cols):
    tid = int(col.replace("topic_loading_", ""))
    label = theme_labels.get(tid, f"Cluster {tid}")

    y = merged_clean[col].values
    if np.isnan(y).any():
        continue

    # OLS: beta = (X'X)^{-1} X'y
    beta_hat = XtX_inv_Xt @ y
    residuals = y - X_full @ beta_hat
    sse = residuals @ residuals
    sst = np.sum((y - y.mean()) ** 2)
    r2 = 1 - sse / sst if sst > 0 else 0
    adj_r2 = 1 - (1 - r2) * (n_obs - 1) / (n_obs - n_params - 1)

    # HC1 heteroskedasticity-robust standard errors
    meat = (X_full * residuals[:, None]).T @ (X_full * residuals[:, None])
    hc1_factor = n_obs / (n_obs - n_params)
    var_beta = hc1_factor * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.maximum(np.diag(var_beta), 0))
    t_stats = np.where(se > 0, beta_hat / se, 0)

    # Extract control coefficients (indices 1 through k_ctrl, skipping intercept)
    row = {
        "theme_id": tid,
        "theme_label": label,
        "n": n_obs,
        "adj_r2": adj_r2,
    }
    for j, ctrl in enumerate(CONTROLS):
        idx = j + 1  # +1 because intercept is at index 0
        row[f"{ctrl}_beta"] = beta_hat[idx]
        row[f"{ctrl}_tstat"] = t_stats[idx]
        # Two-sided p-value (normal approx, valid with N>7000)
        row[f"{ctrl}_pval"] = 2 * normal_sf(abs(t_stats[idx]))
    results.append(row)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("adj_r2", ascending=False)
print(f"\n  Completed: {len(results_df)} themes")


# 3. OUTPUT: CSV

csv_path = os.path.join(OUTPUT_DIR, "table4_theme_determinants.csv")
results_df.to_csv(csv_path, index=False)
print(f"\n  Saved: {csv_path}")


# 4. OUTPUT: FORMATTED TEXT TABLE

def sig_star(p):
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""

lines = []
lines.append("Table 4: Baseline Semantic Themes and Bank Characteristics")
lines.append("=" * 140)
lines.append(f"Sample: {len(merged_clean):,} bank-years, {merged_clean['cik'].nunique()} banks")
lines.append(f"Dependent variable: Bank's z-scored loading on each theme (cross-sectionally standardized)")
lines.append(f"Independent variables: Fiscal-year-t accounting characteristics (same fiscal year as the 10-K)")
lines.append(f"Fixed effects: SIC4 industry ({n_sic4} groups) + Year ({n_years} years)")
lines.append("")

ctrl_headers = {
    "log_assets": ("Log", "Assets"),
    "log_age": ("Log", "Age"),
    "cash_assets": ("Cash/", "Assets"),
    "loans_assets": ("Loans/", "Assets"),
    "capital": ("Equity/", "Assets"),
    "neg_earn": ("Neg.", "Earn."),
}
header = f"{'Theme':<30} " + " ".join(f"{v[0]:<16}" for v in ctrl_headers.values()) + f" {'Adj':>6}"
subheader = f"{'':<30} " + " ".join(f"{v[1]:<16}" for v in ctrl_headers.values()) + f" {'R²':>6}"
lines.append(header)
lines.append(subheader)
lines.append("-" * 140)

for _, row in results_df.iterrows():
    parts = [f"{row['theme_label']:<30}"]
    for ctrl in CONTROLS:
        beta = row.get(f"{ctrl}_beta", np.nan)
        tstat = row.get(f"{ctrl}_tstat", np.nan)
        pval = row.get(f"{ctrl}_pval", 1.0)
        if pd.notna(beta):
            star = sig_star(pval)
            parts.append(f"{beta:>7.3f}{star:<3} ({tstat:>5.2f})")
        else:
            parts.append(f"{'':>16}")
    parts.append(f"{row['adj_r2']:>6.3f}")
    lines.append(" ".join(parts))

lines.append("-" * 140)
lines.append("* p<0.10, ** p<0.05, *** p<0.01. t-statistics in parentheses.")
lines.append(f"All regressions include SIC4 and year fixed effects.")
lines.append(f"Dependent variable is the cross-sectionally standardized theme loading (z-scored within year).")
lines.append(f"Independent variables are lagged one year.")

txt_path = os.path.join(OUTPUT_DIR, "table4_theme_determinants.txt")
with open(txt_path, "w") as f:
    f.write("\n".join(lines))
print(f"  Saved: {txt_path}")


# 5. OUTPUT: HEATMAP VISUALIZATION (standardized coefficients)

# Raw coefficients are not comparable across variables because the
# variables have very different scales (e.g., cash_assets
# ~ 0.06 vs log_assets ~ 8). Solution: multiply each coefficient
# by the SD of that variable → "standardized beta" = effect of a
# 1-SD increase in X on the theme loading (already in z-score units).
print("\nCreating heatmap (standardized coefficients)...")

ctrl_display = {
    "log_assets": "Log Assets",
    "log_age": "Log Age",
    "cash_assets": "Cash/Assets",
    "loans_assets": "Loans/Assets",
    "capital": "Equity/Assets",
    "neg_earn": "Neg. Earnings",
}

# Compute SD of each control variable in the sample
ctrl_sds = {c: merged_clean[c].std() for c in CONTROLS}
print("  Variable SDs (for standardization):")
for c, sd in ctrl_sds.items():
    print(f"    {c:30s}  SD = {sd:.6f}")

heat_data = []
for _, row in results_df.iterrows():
    tid = row["theme_id"]
    short_label = theme_labels_short.get(tid, row["theme_label"][:30])
    # Append adj. R² to label so reader sees explanatory power at a glance
    r2_val = row.get("adj_r2", 0)
    entry = {"Theme": f"{short_label}  (R²={r2_val:.2f})"}
    for ctrl, display in ctrl_display.items():
        raw_beta = row.get(f"{ctrl}_beta", np.nan)
        # Standardized beta = raw_beta × SD(X)
        # Interpretation: 1 SD increase in X → this many SD change in loading
        if pd.notna(raw_beta):
            entry[display] = raw_beta * ctrl_sds[ctrl]
        else:
            entry[display] = np.nan
    heat_data.append(entry)

heat_df = pd.DataFrame(heat_data).set_index("Theme")
heat_df = heat_df.dropna(how="all")

fig, ax = plt.subplots(figsize=(12, max(8, len(heat_df) * 0.35)))

vmax = np.nanpercentile(np.abs(heat_df.values), 95)
im = ax.imshow(heat_df.values, cmap="RdBu_r", aspect="auto",
               vmin=-vmax, vmax=vmax)

ax.set_xticks(range(len(heat_df.columns)))
ax.set_xticklabels(heat_df.columns, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(heat_df.index)))
ax.set_yticklabels(heat_df.index, fontsize=8)

# Add significance stars
for i, (_, row) in enumerate(results_df.iterrows()):
    for j, ctrl in enumerate(ctrl_display.keys()):
        pval = row.get(f"{ctrl}_pval", 1.0)
        star = sig_star(pval)
        if star:
            val = heat_df.iloc[i, j] if not np.isnan(heat_df.iloc[i, j]) else 0
            ax.text(j, i, star, ha="center", va="center",
                    fontsize=7, color="white" if abs(val) > vmax * 0.6 else "black")

plt.colorbar(im, ax=ax, label="Standardized coefficient (effect of 1 SD change)", shrink=0.8)
ax.set_title("Theme Determinants — Standardized Coefficients\n"
             "(Effect of 1 SD increase in bank characteristic on theme loading)",
             fontsize=12, fontweight="bold")
plt.tight_layout()

heatmap_path = os.path.join(OUTPUT_DIR, "table4_theme_determinants_heatmap.png")
plt.savefig(heatmap_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"  Saved: {heatmap_path}")

print("\nDone!")
