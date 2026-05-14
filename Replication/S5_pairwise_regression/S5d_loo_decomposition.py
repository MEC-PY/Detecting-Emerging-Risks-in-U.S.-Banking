# Leave-one-out individual theme decomposition (HH 2019 style).
# For each quarter and theme: delta_R2 = adj_R2(full) - adj_R2(full minus theme k).

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE, "..", "data")
OUTPUT_DIR = os.path.join(BASE, "..", "output")
INPUT_PATH = os.path.join(OUTPUT_DIR, "pairwise_with_theme_products_v7.parquet")
THEMES_CSV = os.path.join(DATA_DIR, "outputs_textual_factors_v2", "final_themes.csv")

themes_df = pd.read_csv(THEMES_CSV)
theme_labels = {}
for _, r in themes_df.iterrows():
    cid = int(r["cluster_id"])
    label = str(r.get("label", "") or r.get("taxonomy_match", "") or "")
    theme_labels[cid] = label if label else f"cluster_{cid}"

# Load data
print("Loading pairwise dataset...")
df = pd.read_parquet(INPUT_PATH)

PROD_COLS = sorted([c for c in df.columns if c.startswith("prod_topic_loading_")])
CONTROLS  = [c for c in [
    "log_assets", "log_age", "cash_assets", "loans_assets",
    "capital", "neg_earn",
    "same_sic2", "same_sic3", "same_sic4",
] if c in df.columns]

print(f"  {len(PROD_COLS)} themes, {len(CONTROLS)} controls")
df = df.dropna(subset=["covariance"] + PROD_COLS)
print(f"  {len(df):,} rows, {df['quarter'].nunique()} quarters")

ALL_REGRESSORS = PROD_COLS + CONTROLS
K = len(PROD_COLS)
MIN_OBS = 200


def adj_r2(y, X):
    """OLS adjusted R² using numpy. X should NOT include intercept column."""
    n = len(y)
    # Add intercept
    X_with_const = np.column_stack([np.ones(n), X])
    p = X_with_const.shape[1]
    # OLS via normal equations
    try:
        beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan
    resid = y - X_with_const @ beta
    ss_res = np.dot(resid, resid)
    ss_tot = np.dot(y - y.mean(), y - y.mean())
    if ss_tot == 0:
        return np.nan
    r2 = 1 - ss_res / ss_tot
    adj = 1 - (1 - r2) * (n - 1) / (n - p)
    return adj


def ols_beta_pvalue(y, X, col_idx):
    """Get beta and p-value for column col_idx from full OLS."""
    n = len(y)
    X_with_const = np.column_stack([np.ones(n), X])
    p = X_with_const.shape[1]
    try:
        beta = np.linalg.lstsq(X_with_const, y, rcond=None)[0]
    except np.linalg.LinAlgError:
        return np.nan, np.nan
    resid = y - X_with_const @ beta
    ss_res = np.dot(resid, resid)
    dof = n - p
    if dof <= 0:
        return beta[col_idx + 1], np.nan  # +1 for intercept
    mse = ss_res / dof
    try:
        XtX_inv = np.linalg.inv(X_with_const.T @ X_with_const)
    except np.linalg.LinAlgError:
        return beta[col_idx + 1], np.nan
    se = np.sqrt(np.abs(mse * XtX_inv[col_idx + 1, col_idx + 1]))
    if se == 0:
        return beta[col_idx + 1], np.nan
    from scipy import stats as sp_stats
    t_stat = beta[col_idx + 1] / se
    pval = 2 * sp_stats.t.sf(np.abs(t_stat), dof)
    return beta[col_idx + 1], pval


# Check if scipy is available for p-values
try:
    from scipy import stats as sp_stats
    HAS_SCIPY = True
    print("  scipy available for p-values")
except ImportError:
    HAS_SCIPY = False
    print("  WARNING: scipy not available, p-values will be NaN")

# Main loop
quarters = sorted(df["quarter"].unique())
print(f"\nRunning leave-one-out decomposition for {len(quarters)} quarters × {K} themes...")

theme_results = []

for q_idx, q in enumerate(quarters):
    sub = df[df["quarter"] == q]
    n = len(sub)
    if n < MIN_OBS:
        continue

    y = sub["covariance"].values.astype(np.float64)
    X_all = sub[ALL_REGRESSORS].values.astype(np.float64)

    # Full model adj R²
    r2_full = adj_r2(y, X_all)

    # Get betas and p-values from full model
    betas_pvals = {}
    if HAS_SCIPY:
        for k_idx, col in enumerate(PROD_COLS):
            b, p = ols_beta_pvalue(y, X_all, k_idx)
            betas_pvals[col] = (b, p)

    # Leave-one-out for each theme
    for k_idx, col in enumerate(PROD_COLS):
        # Build X without this theme column
        X_loo = np.delete(X_all, k_idx, axis=1)
        r2_loo = adj_r2(y, X_loo)
        delta = r2_full - r2_loo

        theme_id = int(col.replace("prod_topic_loading_", ""))
        beta_val, pval = betas_pvals.get(col, (np.nan, np.nan))

        theme_results.append({
            "quarter": q,
            "year": int(str(q).split("Q")[0]),
            "theme_id": theme_id,
            "theme_label": theme_labels.get(theme_id, ""),
            "theme_col": col,
            "adj_r2_full": r2_full,
            "adj_r2_loo": r2_loo,
            "delta_adj_r2": delta,
            "beta": beta_val,
            "pvalue": pval,
        })

    if (q_idx + 1) % 5 == 0:
        print(f"  {q_idx+1}/{len(quarters)} quarters done ({q})")

# Save
theme_df = pd.DataFrame(theme_results)
out_csv = os.path.join(OUTPUT_DIR, "individual_theme_r2_v7_xsecstd.csv")
theme_df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")
print(f"  {len(theme_df):,} rows ({theme_df['theme_id'].nunique()} themes × {theme_df['quarter'].nunique()} quarters)")

# Quick summary
print("\nTop 10 themes by mean delta_adj_r2 (leave-one-out):")
top = (theme_df.groupby("theme_id")["delta_adj_r2"]
       .mean().sort_values(ascending=False).head(10))
for tid, val in top.items():
    lbl = theme_labels.get(tid, str(tid))[:40]
    print(f"  {tid:3d}  {lbl:40s}  mean ΔR² = {val*10000:.2f} × 10⁴")
