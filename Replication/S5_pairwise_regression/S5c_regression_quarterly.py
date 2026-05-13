# HH quarterly covariance regression with cross-sectionally standardized loadings.
# Uses pre-GFC baseline (2006-2007) z-scoring + rolling 3-year robustness.
# Requires: output/pairwise_with_theme_products_v7.parquet

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

# Paths
BASE        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE, "..", "data")
OUTPUT_DIR  = os.path.join(BASE, "..", "output")
INPUT_PATH  = os.path.join(OUTPUT_DIR, "pairwise_with_theme_products_v7.parquet")
THEMES_PATH = os.path.join(DATA_DIR, "outputs_textual_factors_v2", "final_themes.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load theme labels
themes_df = pd.read_csv(THEMES_PATH)
theme_labels = {}
for _, r in themes_df.iterrows():
    cid = int(r["cluster_id"])
    label = str(r.get("label", "") or r.get("taxonomy_match", "") or "")
    theme_labels[cid] = label if label else f"cluster_{cid}"

# Load data
print("Loading pairwise dataset with cross-sec standardized products (v7)...")
df = pd.read_parquet(INPUT_PATH)
print(f"  Shape          : {df.shape}")
print(f"  Year range     : {df['year'].min()} – {df['year'].max()}")
print(f"  Quarters       : {df['quarter'].nunique()}")
print(f"  Unique pairs   : {df[['cik_i','cik_j']].drop_duplicates().shape[0]:,}")

# Variable definitions
PROD_COLS = sorted([c for c in df.columns if c.startswith("prod_topic_loading_")])
CONTROLS  = [c for c in [
    # Six of HH's seven bank-level controls (loss_prov_allow_assets dropped —
    # see 11_build_pairwise_dataset.py docstring)
    "log_assets", "log_age", "cash_assets", "loans_assets",
    "capital", "neg_earn",
    "same_sic2", "same_sic3", "same_sic4",
] if c in df.columns]

print(f"\nTextual theme regressors : {len(PROD_COLS)}")
print(f"Control regressors       : {len(CONTROLS)}")

# Diagnostic: product statistics
prod_vals = df[PROD_COLS].values
print(f"\nProduct stats (xsec-std loadings):")
print(f"  min={prod_vals.min():.4f}  max={prod_vals.max():.4f}")
print(f"  mean={prod_vals.mean():.4f}  std={prod_vals.std():.4f}")
print(f"  zeros: {(prod_vals == 0).sum() / prod_vals.size * 100:.1f}%")
print(f"  negatives: {(prod_vals < 0).sum() / prod_vals.size * 100:.1f}%")

# NOTE: covariance has ALREADY been winsorized per-quarter at 1/99 inside
# 11_build_pairwise_dataset.py, which matches Hanley & Hoberg (2019) fn. 19
# and their Panel C note: "winsorize the covariance estimates in each quarter
# at the 1%/99% level." We intentionally do NOT apply a second pooled clip
# here, because that would not replicate HH's procedure.

# Drop rows missing covariance or any theme product
df = df.dropna(subset=["covariance"] + PROD_COLS)
print(f"\nAfter dropping missing    : {len(df):,} rows")
print(f"Quarters remaining        : {df['quarter'].nunique()}")

# Build regression formulas
text_terms = " + ".join(PROD_COLS)
ctrl_terms = " + ".join(CONTROLS)

formula_full = f"covariance ~ {text_terms} + {ctrl_terms}"
formula_ctrl = f"covariance ~ {ctrl_terms}"

print(f"\nFull model : {len(PROD_COLS)} text regressors + {len(CONTROLS)} controls")
print(f"Ctrl model : {len(CONTROLS)} controls only")

# Run per-quarter regressions
print("\n" + "="*80)
print(f"  {'Quarter':<12}  {'N':>8}  {'AdjR2_f':>8}  {'AdjR2_c':>8}  "
      f"{'dAdjR2':>8}  {'(raw dR2)':>11}")
print("="*80)

results = []
quarters = sorted(df["quarter"].unique())
MIN_OBS  = 200

for q in quarters:
    sub = df[df["quarter"] == q].copy()
    n   = len(sub)

    if n < MIN_OBS:
        print(f"  {q:<12}  {n:>8,}  (skipped — too few observations)")
        continue

    try:
        res_full = smf.ols(formula_full, data=sub).fit()
        r2_full      = res_full.rsquared
        adj_r2_full  = res_full.rsquared_adj

        res_ctrl = smf.ols(formula_ctrl, data=sub).fit()
        r2_ctrl      = res_ctrl.rsquared
        adj_r2_ctrl  = res_ctrl.rsquared_adj

        delta_r2     = r2_full      - r2_ctrl
        delta_adj_r2 = adj_r2_full  - adj_r2_ctrl

        betas = {col: res_full.params.get(col, np.nan) for col in PROD_COLS}

        row = {
            "quarter"      : q,
            "year"         : int(str(q).split("Q")[0]),
            "q_num"        : int(str(q).split("Q")[1]),
            "n"            : n,
            "r2_full"      : r2_full,
            "r2_ctrl"      : r2_ctrl,
            "adj_r2_full"  : adj_r2_full,
            "adj_r2_ctrl"  : adj_r2_ctrl,
            "delta_r2"     : delta_r2,
            "delta_adj_r2" : delta_adj_r2,
        }
        row.update({f"beta_{k}": v for k, v in betas.items()})
        results.append(row)

        print(f"  {q:<12}  {n:>8,}  {adj_r2_full:>8.4f}  {adj_r2_ctrl:>8.4f}  "
              f"{delta_adj_r2:>8.4f}  ({delta_r2:+.4f} raw)")

    except Exception as e:
        print(f"  {q:<12}  ERROR: {e}")

print("="*80)

results_df = pd.DataFrame(results)
print(f"\nCompleted: {len(results_df)} quarters")
print(f"Mean  delta_adj_r2  : {results_df['delta_adj_r2'].mean():.4f}")
print(f"Median delta_adj_r2 : {results_df['delta_adj_r2'].median():.4f}")
peak_idx = results_df['delta_adj_r2'].idxmax()
print(f"Peak  delta_adj_r2  : {results_df.loc[peak_idx,'delta_adj_r2']:.4f}  "
      f"(quarter {results_df.loc[peak_idx,'quarter']})")

# Z-SCORE NORMALISATION — DUAL BASELINES

def z_score_baseline(series, baseline_mask, label):
    baseline = series[baseline_mask]
    mu = baseline.mean()
    sig = baseline.std()
    z = (series - mu) / sig if sig > 0 else series - mu
    print(f"\n  {label}:")
    print(f"    Quarters in baseline : {baseline_mask.sum()}")
    print(f"    Baseline mean        : {mu:.5f}")
    print(f"    Baseline std         : {sig:.5f}")
    print(f"    Z-score range        : {z.min():.2f} to {z.max():.2f}")
    return z, mu, sig

# Pre-GFC baseline (2006-2007)
mask_pregfc = results_df["year"].between(2006, 2007)
results_df["z_score_pregfc"], mu_pregfc, sig_pregfc = z_score_baseline(
    results_df["delta_adj_r2"], mask_pregfc, "Pre-GFC baseline (2006-2007)")

# Post-GFC baseline (2010-2015)
mask_postgfc = results_df["year"].between(2010, 2015)
results_df["z_score_postgfc"], mu_postgfc, sig_postgfc = z_score_baseline(
    results_df["delta_adj_r2"], mask_postgfc, "Post-GFC baseline (2010-2015)")

# Rolling 3-year baseline
print("\n  Rolling 3-year baseline z-scores:")
rolling_z = []
for idx, row in results_df.iterrows():
    yr = row["year"]
    qn = row["q_num"]
    # Baseline: 3 years ending 1 year before this quarter
    baseline = results_df[results_df["year"].between(yr - 4, yr - 2)]
    if len(baseline) >= 4:
        mu_r = baseline["delta_adj_r2"].mean()
        sig_r = baseline["delta_adj_r2"].std()
        z_r = (row["delta_adj_r2"] - mu_r) / sig_r if sig_r > 0 else 0
    else:
        z_r = np.nan
    rolling_z.append(z_r)
results_df["z_score_rolling"] = rolling_z

# Primary z-score
results_df["z_score"] = results_df["z_score_pregfc"]

# Save quarterly results
out_csv = os.path.join(OUTPUT_DIR, "quarterly_marginal_r2_v7_xsecstd.csv")
results_df.to_csv(out_csv, index=False)
print(f"\nSaved: {out_csv}")

# Subperiod summary
print("\n" + "="*70)
print("SUBPERIOD SUMMARY")
print("="*70)
subperiods = [
    ("Pre-GFC baseline (2006-2007)", results_df["year"].between(2006, 2007)),
    ("GFC (2008-2009)",              results_df["year"].between(2008, 2009)),
    ("Post-crisis (2010-2015)",      results_df["year"].between(2010, 2015)),
    ("Post-reform (2016-2019)",      results_df["year"].between(2016, 2019)),
    ("COVID+ (2020-2021)",           results_df["year"].between(2020, 2021)),
    ("Rate hike (2022-2024)",        results_df["year"] >= 2022),
]

fmt = "  {:<35}  mean dAdjR2={:.4f}  z_preGFC={:>6.2f}  z_roll={:>6.2f}  (n={} qtrs)"
for label, mask in subperiods:
    sub = results_df[mask]
    if len(sub) == 0:
        continue
    z_roll = sub["z_score_rolling"].dropna().mean()
    print(fmt.format(label, sub["delta_adj_r2"].mean(),
                     sub["z_score"].mean(), z_roll, len(sub)))

# FIGURE 1: Emerging Risk Measure (H&H Figure 1 equivalent)
print("\nGenerating Figure 1 (3 panels: pre-GFC z, post-GFC z, rolling z)...")
results_df["date"] = pd.PeriodIndex(results_df["quarter"], freq="Q").to_timestamp()

fig, axes = plt.subplots(3, 1, figsize=(15, 14), sharex=False)

def plot_panel(ax, zseries, title, baseline_span=None):
    colors = ["steelblue" if z >= 0 else "lightcoral" for z in zseries]
    ax.bar(results_df["date"], zseries, width=60, color=colors, alpha=0.85, zorder=3)
    ax.axhline(0, color="black", linewidth=0.8, zorder=4)
    if baseline_span:
        ax.axvspan(*baseline_span, alpha=0.12, color="green", label="Baseline window", zorder=1)
        ax.legend(fontsize=9, loc="upper left")
    ax.set_ylabel("Z-score", fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

# Panel A: Pre-GFC baseline
plot_panel(axes[0], results_df["z_score_pregfc"],
           "Panel A — Pre-GFC Baseline (2006–2007)",
           (pd.Timestamp("2006-01-01"), pd.Timestamp("2007-12-31")))

# Panel B: Post-GFC baseline
plot_panel(axes[1], results_df["z_score_postgfc"],
           "Panel B — Post-GFC Baseline (2010–2015)",
           (pd.Timestamp("2010-01-01"), pd.Timestamp("2015-12-31")))

# Panel C: Rolling 3-year baseline
z_rolling_clean = results_df["z_score_rolling"].fillna(0)
plot_panel(axes[2], z_rolling_clean,
           "Panel C — Rolling 3-Year Baseline (t-4 to t-2)")

axes[2].set_xlabel("Quarter", fontsize=11)

fig.suptitle(
    f"Aggregate Emerging Risk Index — Cross-Sectionally Standardized Loadings\n"
    f"US Banks {results_df['year'].min()}-{results_df['year'].max()}",
    fontsize=14, fontweight="bold", y=1.01)

plt.tight_layout()
out_plot = os.path.join(OUTPUT_DIR, "hh_emerging_risk_figure1_v7_xsecstd.png")
plt.savefig(out_plot, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved plot : {out_plot}")

# INDIVIDUAL THEME DECOMPOSITION
print("\n" + "="*70)
print("INDIVIDUAL THEME DECOMPOSITION")
print("="*70)

theme_results = []
for q_idx, q in enumerate(quarters):
    sub = df[df["quarter"] == q].copy()
    n   = len(sub)
    if n < MIN_OBS:
        continue

    # Full model with all themes (needed for leave-one-out)
    res_full_q = smf.ols(formula_full, data=sub).fit()
    # HC1-robust fit for p-values (consistent with Tables 5, 6, 7)
    res_full_q_hc1 = smf.ols(formula_full, data=sub).fit(cov_type='HC1')
    adj_r2_full_q = res_full_q.rsquared_adj

    for col in PROD_COLS:
        try:
            # Leave-one-out: full model minus this one theme
            other_cols = [c for c in PROD_COLS if c != col]
            f_loo = f"covariance ~ {' + '.join(other_cols)} + {ctrl_terms}"
            res_loo = smf.ols(f_loo, data=sub).fit()
            adj_r2_loo = res_loo.rsquared_adj
            delta_loo = adj_r2_full_q - adj_r2_loo

            theme_id = int(col.replace("prod_topic_loading_", ""))
            theme_results.append({
                "quarter": q,
                "year": int(str(q).split("Q")[0]),
                "theme_id": theme_id,
                "theme_label": theme_labels.get(theme_id, ""),
                "theme_col": col,
                "adj_r2_full": adj_r2_full_q,
                "adj_r2_loo": adj_r2_loo,
                "delta_adj_r2": delta_loo,
                "beta": res_full_q.params.get(col, np.nan),
                "pvalue": res_full_q_hc1.pvalues.get(col, np.nan),
            })
        except Exception:
            pass

    if (q_idx + 1) % 10 == 0:
        print(f"  Processed {q_idx+1}/{len(quarters)} quarters...")

theme_df = pd.DataFrame(theme_results)
theme_csv = os.path.join(OUTPUT_DIR, "individual_theme_r2_v7_xsecstd.csv")
theme_df.to_csv(theme_csv, index=False)
print(f"Saved: {theme_csv}")

if len(theme_df):
    # Top themes during GFC specifically
    gfc_themes = theme_df[theme_df["year"].between(2008, 2009)]
    if len(gfc_themes):
        avg_gfc = (gfc_themes.groupby(["theme_id", "theme_label"])["delta_adj_r2"]
                   .mean().sort_values(ascending=False).head(15))
        print(f"\n  Top 15 themes by avg individual ΔAdj.R² during GFC (2008-2009):")
        for (tid, tlabel), val in avg_gfc.items():
            print(f"    [{tid:3d}] {tlabel:25s}  avg ΔAdj.R² = {val:.5f}")

    # Top themes overall
    avg_all = (theme_df.groupby(["theme_id", "theme_label"])["delta_adj_r2"]
               .mean().sort_values(ascending=False).head(15))
    print(f"\n  Top 15 themes overall:")
    for (tid, tlabel), val in avg_all.items():
        print(f"    [{tid:3d}] {tlabel:25s}  avg ΔAdj.R² = {val:.5f}")

# POOLED SUMMARY
print("\n" + "="*55)
print("POOLED SUMMARY (all quarters, cluster SE on cik_i)")
print("="*55)

res_pooled_full = smf.ols(formula_full, data=df).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["cik_i"]},
)
res_pooled_ctrl = smf.ols(formula_ctrl, data=df).fit(
    cov_type="cluster",
    cov_kwds={"groups": df["cik_i"]},
)
r2_pf  = res_pooled_full.rsquared
r2_pc  = res_pooled_ctrl.rsquared
delta  = r2_pf - r2_pc
print(f"  N              : {int(res_pooled_full.nobs):,}")
print(f"  R²_full        : {r2_pf:.4f}")
print(f"  R²_ctrl        : {r2_pc:.4f}")
print(f"  ΔR² (pooled)   : {delta:.4f}")

# Save pooled β summary
summary_rows = []
for col in PROD_COLS:
    b    = res_pooled_full.params.get(col, np.nan)
    se   = res_pooled_full.bse.get(col, np.nan)
    t    = res_pooled_full.tvalues.get(col, np.nan)
    p    = res_pooled_full.pvalues.get(col, np.nan)
    stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
    theme_id = int(col.replace("prod_topic_loading_", ""))
    summary_rows.append({
        "theme_col"    : col,
        "theme_id"     : theme_id,
        "theme_label"  : theme_labels.get(theme_id, ""),
        "beta"         : b,
        "se"           : se,
        "t"            : t,
        "pvalue"       : p,
        "significance" : stars,
    })

summary_df = pd.DataFrame(summary_rows)
out_summary = os.path.join(OUTPUT_DIR, "regression_summary_hh_v7_xsecstd.csv")
summary_df.to_csv(out_summary, index=False)
print(f"\nPooled theme-level β table saved: {out_summary}")

top10 = summary_df.dropna(subset=["pvalue"]).nsmallest(10, "pvalue")
print("\nTop 10 most significant themes (pooled regression):")
print(f"  {'ID':>4}  {'Label':>25}  {'β':>12}  {'SE':>12}  {'p':>10}  {'sig':>4}")
print(f"  {'─'*70}")
for _, row in top10.iterrows():
    print(f"  {row['theme_id']:>4}  {row['theme_label']:>25}  {row['beta']:>12.6f}  "
          f"{row['se']:>12.6f}  {row['pvalue']:>10.4f}  {row['significance']:>4}")

# COMPARISON: V7 (xsec-std) vs V6 (SVD-normed) — GFC signal diagnostic
# Purely diagnostic; does not affect any canonical output.
v6_path = os.path.join(OUTPUT_DIR, "quarterly_marginal_r2_v6_svdnorm.csv")
if os.path.exists(v6_path):
    print("\n" + "="*70)
    print("COMPARISON: V7 (xsec-std) vs V6 (SVD-normed)")
    print("="*70)
    v6 = pd.read_csv(v6_path)[["quarter", "delta_adj_r2"]].rename(
        columns={"delta_adj_r2": "delta_adj_r2_v6"})
    v7 = results_df[["quarter", "delta_adj_r2"]].rename(
        columns={"delta_adj_r2": "delta_adj_r2_v7"})
    comp = v7.merge(v6, on="quarter", how="inner")

    for label, yr_range in [("Pre-GFC (2006-07)", (2006, 2007)),
                             ("GFC (2008-09)", (2008, 2009)),
                             ("Post-GFC (2010-15)", (2010, 2015)),
                             ("Recent (2020-24)", (2020, 2024))]:
        mask = comp["quarter"].str[:4].astype(int).between(*yr_range)
        sub = comp[mask]
        if len(sub) > 0:
            v7_mean = float(sub["delta_adj_r2_v7"].mean())
            v6_mean = float(sub["delta_adj_r2_v6"].mean())
            ratio = v7_mean / v6_mean if v6_mean != 0 else float("inf")
            print(f"  {label:20s}  v7={v7_mean:.4f}  v6={v6_mean:.4f}  "
                  f"ratio(v7/v6)={ratio:.2f}")

print(f"\n{'='*55}")
print("DONE. Key outputs:")
print(f"  Quarterly ΔR²       →  {out_csv}")
print(f"  Figure 1            →  {out_plot}")
print(f"  β table             →  {out_summary}")
print(f"  Theme decomposition →  {theme_csv}")
