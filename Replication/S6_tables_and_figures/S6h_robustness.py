# Robustness tests (HH Table 7). Checks marginal text R2
# across subperiods, size splits, SIC homogeneity, and
# theme concentration (drop top-5 contributors).

import os
import sys
import warnings
import math
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import f as f_dist

warnings.filterwarnings("ignore")

# Paths
BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE, "..", "data")
OUTPUT_DIR = os.path.join(BASE, "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARQUET_PATH = os.path.join(OUTPUT_DIR, "pairwise_with_theme_products_v7.parquet")
FUND_PATH    = os.path.join(DATA_DIR, "bank_fundamentals_hh_extended.csv")
LINK_PATH    = os.path.join(DATA_DIR, "permno_cik_wrds_extended.csv")
IND_R2_PATH  = os.path.join(OUTPUT_DIR, "individual_theme_r2_v7_xsecstd.csv")

# HH Control Variables
HH_CONTROLS = [
    # Six of HH's seven bank-level controls (loss_prov_allow_assets dropped —
    # comp.bank not in WRDS subscription; see 11_build_pairwise_dataset.py).
    "log_assets", "log_age",
    "cash_assets", "loans_assets",
    "capital", "neg_earn",
]
SIC_CONTROLS = ["same_sic2", "same_sic3", "same_sic4"]
ALL_CONTROLS = HH_CONTROLS + SIC_CONTROLS

# Minimum observations per quarter for a stable regression (matches 16j)
MIN_OBS = 200



# 1. LOAD DATA — parquet-first with CSV reconstruction fallback

print("=" * 70)
print("18f_robustness.py — Robustness Tests for Main Covariance Regression")
print("=" * 70)

USE_PARQUET = False
try:
    print(f"\nAttempting to load canonical parquet: {PARQUET_PATH}")
    pair_df = pd.read_parquet(PARQUET_PATH)
    USE_PARQUET = True
    print(f"  ✓ Loaded canonical parquet: {pair_df.shape}")
    print(f"  Year range : {pair_df['year'].min()} – {pair_df['year'].max()}")
    print(f"  Quarters   : {pair_df['quarter'].nunique()}")
except (ImportError, FileNotFoundError) as e:
    print(f"  ✗ Parquet load failed ({type(e).__name__}): {e}")
    print("  → Falling back to CSV reconstruction (numbers will NOT be bit-")
    print("    identical to 16j canonical; expect ~2% sample shrinkage).")

# ── Identify theme product columns in parquet
if USE_PARQUET:
    PROD_COLS = sorted([c for c in pair_df.columns
                        if c.startswith("prod_topic_loading_")])
    print(f"  Theme product cols: {len(PROD_COLS)}")
    print(f"  Control cols found: "
          f"{[c for c in ALL_CONTROLS if c in pair_df.columns]}")

    # NOTE: covariance has ALREADY been winsorized per-quarter at 1/99 inside
    # 11_build_pairwise_dataset.py. This matches Hanley & Hoberg (2019) fn. 19
    # ("winsorize the covariance estimates in each quarter at the 1%/99% level").
    # We deliberately do NOT apply a second pooled clip, to replicate HH exactly.

    # Drop rows missing covariance or any theme product
    before = len(pair_df)
    pair_df = pair_df.dropna(subset=["covariance"] + PROD_COLS)
    print(f"  Rows after dropna: {len(pair_df):,} (dropped {before - len(pair_df):,})")

    # ── Re-merge bank-level log_assets & sich for subsample splits ─
    # (Parquet only holds pair-product controls, not raw bank-level values.
    # We need log_assets_i, log_assets_j, sich_i, sich_j to build the
    # within-year median split for Large vs Small.)
    print("\nRe-merging bank-level characteristics for subsample splits...")
    link = pd.read_csv(LINK_PATH)
    link = link.dropna(subset=["cik", "permno"])
    link["permno"] = link["permno"].astype(int)
    link["cik"] = link["cik"].astype(int).astype(str).str.zfill(10)
    permno_to_cik = dict(zip(link["permno"], link["cik"]))

    fund_raw = pd.read_csv(FUND_PATH)
    fund_raw["permno"] = pd.to_numeric(fund_raw["permno"], errors="coerce")
    fund_raw = fund_raw.dropna(subset=["permno"])
    fund_raw["permno"] = fund_raw["permno"].astype(int)
    fund_raw["cik"] = fund_raw["permno"].map(permno_to_cik)
    fund_raw = fund_raw.dropna(subset=["cik"])
    fund_raw["cik"] = fund_raw["cik"].astype(str).str.zfill(10)
    fund_raw["fyear"] = pd.to_numeric(fund_raw["fyear"], errors="coerce").astype("Int64")
    fund_raw["sich"] = pd.to_numeric(fund_raw["sich"], errors="coerce")
    fund_raw["log_assets_bank"] = pd.to_numeric(fund_raw["log_assets"], errors="coerce")

    fund_slim = fund_raw[["cik", "fyear", "log_assets_bank", "sich"]].drop_duplicates(
        subset=["cik", "fyear"], keep="last")

    # Within-year median of bank-level log_assets (used for Large/Small split)
    yearly_median = fund_slim.groupby("fyear")["log_assets_bank"].median().to_dict()
    print(f"  Yearly median log_assets computed for {len(yearly_median)} years")

    # Merge in bank i's and bank j's characteristics (prior-fiscal-year cross
    # section). A LEFT JOIN on (cik, fyear) means row count CANNOT change; the
    # only thing the merge can produce is missing values when a bank-year is
    # absent from the fundamentals CSV. We assert row-count invariance, then
    # track how many pairs get NaN-ed out for the Large/Small split so the
    # subsample N is fully transparent.
    n_before_merge = len(pair_df)
    pair_df["fyear"] = pair_df["year"] - 1

    pair_df = pair_df.merge(
        fund_slim.rename(columns={
            "cik": "cik_i",
            "log_assets_bank": "log_assets_i_bank",
            "sich": "sich_i",
        }),
        on=["cik_i", "fyear"], how="left",
    )
    pair_df = pair_df.merge(
        fund_slim.rename(columns={
            "cik": "cik_j",
            "log_assets_bank": "log_assets_j_bank",
            "sich": "sich_j",
        }),
        on=["cik_j", "fyear"], how="left",
    )

    assert len(pair_df) == n_before_merge, (
        f"Re-merge changed row count: {n_before_merge:,} -> {len(pair_df):,}. "
        "Check fund_slim for duplicate (cik, fyear) keys."
    )
    print(f"  Re-merge row-count invariant OK ({len(pair_df):,})")

    n_missing_i = pair_df["log_assets_i_bank"].isna().sum()
    n_missing_j = pair_df["log_assets_j_bank"].isna().sum()
    print(f"  Missing bank-i log_assets after merge: {n_missing_i:,}")
    print(f"  Missing bank-j log_assets after merge: {n_missing_j:,}")

    # Build large_i / large_j flags using within-year median (only when both
    # sides matched; pairs with any missing side are excluded from the size
    # subsamples, matching the HH median-split logic).
    pair_df["median_log_assets"] = pair_df["fyear"].map(yearly_median)
    pair_df["large_i"] = (pair_df["log_assets_i_bank"]
                           >= pair_df["median_log_assets"]).astype("Int64")
    pair_df["large_j"] = (pair_df["log_assets_j_bank"]
                           >= pair_df["median_log_assets"]).astype("Int64")

    pair_df = pair_df.drop(columns=["fyear"])

    n_has_size = pair_df[["large_i", "large_j"]].notna().all(axis=1).sum()
    pct_has_size = 100.0 * n_has_size / len(pair_df)
    print(f"  Rows with size flags: {n_has_size:,} / {len(pair_df):,} "
          f"({pct_has_size:.1f}%)")
    if pct_has_size < 99.0:
        print(f"  ⚠ WARNING: {100-pct_has_size:.1f}% of pairs dropped from "
              f"size subsamples due to missing fundamentals re-merge. "
              f"Investigate before trusting Large/Small rows.")

else:
    # ══════════════════════════════════════════════════════════════
    # FALLBACK: CSV reconstruction (sandbox path, no pyarrow)
    # ══════════════════════════════════════════════════════════════
    from itertools import combinations

    CRSP_PATH  = os.path.join(DATA_DIR, "crsp_daily_banks_2006_2024.csv")
    LOAD_PATH  = os.path.join(DATA_DIR, "outputs_textual_factors_v2",
                                "bank_year_loadings_v7_xsecstd.csv")
    SAMPLE_KEYS_PATH = os.path.join(OUTPUT_DIR, "main_regression_sample_keys.csv")

    # Linkage
    print("\nLoading PERMNO → CIK linkage...")
    link = pd.read_csv(LINK_PATH)
    link = link.dropna(subset=["cik", "permno"])
    link["permno"] = link["permno"].astype(int)
    link["cik"] = link["cik"].astype(int).astype(str).str.zfill(10)
    permno_to_cik = dict(zip(link["permno"], link["cik"]))

    # CRSP daily
    print("Loading CRSP daily returns...")
    crsp = pd.read_csv(CRSP_PATH, usecols=["PERMNO", "DATE", "RET"])
    crsp["DATE"] = pd.to_datetime(crsp["DATE"], format="%Y%m%d")
    crsp["RET"]  = pd.to_numeric(crsp["RET"], errors="coerce")
    crsp = crsp.dropna(subset=["RET"])
    crsp["PERMNO"] = crsp["PERMNO"].astype(int)
    crsp["cik"] = crsp["PERMNO"].map(permno_to_cik)
    crsp = crsp.dropna(subset=["cik"])
    crsp["year"]    = crsp["DATE"].dt.year
    crsp["quarter"] = crsp["DATE"].dt.to_period("Q").astype(str)
    crsp = crsp[crsp["year"] >= 2006]
    quarters_available = sorted(crsp["quarter"].unique())

    # Fundamentals
    print("Loading fundamentals...")
    fund = pd.read_csv(FUND_PATH)
    fund["permno"] = pd.to_numeric(fund["permno"], errors="coerce")
    fund = fund.dropna(subset=["permno"])
    fund["permno"] = fund["permno"].astype(int)
    fund["cik"] = fund["permno"].map(permno_to_cik)
    fund = fund.dropna(subset=["cik"])
    fund["sich"] = pd.to_numeric(fund["sich"], errors="coerce")

    fund_lookup = {}
    for _, r in fund.iterrows():
        key = (r["cik"], int(r["fyear"]))
        vals = {}
        for c in HH_CONTROLS:
            vals[c] = r[c] if pd.notna(r.get(c)) else np.nan
        vals["sich"] = r["sich"] if pd.notna(r.get("sich")) else np.nan
        fund_lookup[key] = vals

    yearly_assets = fund.groupby("fyear")["log_assets"].median().to_dict()

    # Loadings
    print("Loading theme loadings...")
    loadings = pd.read_csv(LOAD_PATH)
    loadings["cik"] = loadings["cik"].astype(str).str.zfill(10)
    theme_cols = sorted([c for c in loadings.columns
                          if c.startswith("topic_loading_")])
    loading_lookup = {}
    for _, r in loadings.iterrows():
        key = (r["cik"], int(r["year"]))
        loading_lookup[key] = np.array([r[c] for c in theme_cols])

    # Sample-key filter
    sample_keys = pd.read_csv(SAMPLE_KEYS_PATH)
    sample_keys["cik"] = sample_keys["cik"].astype(int).astype(str).str.zfill(10)
    sample_key_set = set(
        (r["cik"], int(r["loading_year"])) for _, r in sample_keys.iterrows()
    )
    fund_lookup    = {k: v for k, v in fund_lookup.items()    if k in sample_key_set}
    loading_lookup = {k: v for k, v in loading_lookup.items() if k in sample_key_set}

    PROD_COLS = [f"prod_{tc}" for tc in theme_cols]

    def build_quarter_pairwise_reconstruction(quarter_str, crsp_q):
        q_year = int(quarter_str[:4])
        load_year = q_year - 1
        pivot = crsp_q.pivot_table(index="DATE", columns="cik", values="RET")
        banks = sorted(pivot.columns.tolist())
        if len(banks) < 10:
            return None
        cov_matrix = pivot.cov()
        pairs = list(combinations(banks, 2))
        covs = np.array([cov_matrix.loc[i, j] for i, j in pairs])
        lo, hi = np.nanpercentile(covs, [1, 99])
        covs = np.clip(covs, lo, hi)
        records = []
        for idx, (ci, cj) in enumerate(pairs):
            if np.isnan(covs[idx]):
                continue
            fund_i = fund_lookup.get((ci, load_year))
            fund_j = fund_lookup.get((cj, load_year))
            load_i = loading_lookup.get((ci, load_year))
            load_j = loading_lookup.get((cj, load_year))
            if fund_i is None or fund_j is None:
                continue
            if load_i is None or load_j is None:
                continue
            ctrl_vals = {}
            skip = False
            for c in HH_CONTROLS:
                vi, vj = fund_i[c], fund_j[c]
                if np.isnan(vi) or np.isnan(vj):
                    skip = True
                    break
                ctrl_vals[c] = vi * vj
            if skip:
                continue
            sich_i = fund_i.get("sich", np.nan)
            sich_j = fund_j.get("sich", np.nan)
            si_str = str(int(sich_i)) if not np.isnan(sich_i) else ""
            sj_str = str(int(sich_j)) if not np.isnan(sich_j) else ""
            ctrl_vals["same_sic2"] = int(len(si_str) >= 2 and len(sj_str) >= 2
                                          and si_str[:2] == sj_str[:2])
            ctrl_vals["same_sic3"] = int(len(si_str) >= 3 and len(sj_str) >= 3
                                          and si_str[:3] == sj_str[:3])
            ctrl_vals["same_sic4"] = int(len(si_str) >= 4 and len(sj_str) >= 4
                                          and si_str[:4] == sj_str[:4])
            theme_prods = load_i * load_j
            assets_i = fund_i["log_assets"]
            assets_j = fund_j["log_assets"]
            median_assets = yearly_assets.get(load_year, np.nan)
            large_i = int(assets_i >= median_assets) if not np.isnan(median_assets) else -1
            large_j = int(assets_j >= median_assets) if not np.isnan(median_assets) else -1
            rec = {
                "covariance": covs[idx],
                "year": q_year, "quarter": quarter_str,
                "large_i": large_i, "large_j": large_j,
            }
            rec.update(ctrl_vals)
            for ti, tc in enumerate(theme_cols):
                rec[f"prod_{tc}"] = theme_prods[ti]
            records.append(rec)
        if len(records) < 100:
            return None
        return pd.DataFrame(records)

    # Build the full reconstructed df once by looping over quarters
    frames = []
    for qi, quarter in enumerate(quarters_available):
        crsp_q = crsp[crsp["quarter"] == quarter]
        qdf = build_quarter_pairwise_reconstruction(quarter, crsp_q)
        if qdf is not None:
            frames.append(qdf)
    pair_df = pd.concat(frames, ignore_index=True)
    print(f"\nReconstructed pair_df: {pair_df.shape}")



# 2. TOP-5 THEMES (for concentration test)

print("\nIdentifying top-5 themes by average marginal R²...")
ind_r2 = pd.read_csv(IND_R2_PATH)
theme_rank = ind_r2.groupby("theme_col")["delta_adj_r2"].mean().sort_values(ascending=False)
top5_prod_cols = [t if t.startswith("prod_") else f"prod_{t}"
                   for t in theme_rank.head(5).index.tolist()]
top5_prod_cols = [t for t in top5_prod_cols if t in PROD_COLS]
remaining_prod_cols = [t for t in PROD_COLS if t not in top5_prod_cols]
print(f"  Top 5: {[t.replace('prod_topic_loading_', '') for t in top5_prod_cols]}")
print(f"  Remaining: {len(remaining_prod_cols)} themes")



# 3. REGRESSION ENGINE — statsmodels formula interface (matches 16j)

def run_spec(sub_df, prod_cols, ctrl_cols):
    """
    Run controls-only and full OLS on a (filtered) quarter's pairwise data
    using statsmodels formula interface to match 16j's regression exactly.
    Returns dict with n_pairs, adj_r2_ctrl, adj_r2_full, delta_adj_r2,
    f_stat, f_pvalue.
    """
    n = len(sub_df)
    if n < MIN_OBS:
        return None

    # Drop constant control columns (happens when a subsample fixes a SIC dummy)
    active_ctrl = [c for c in ctrl_cols if c in sub_df.columns
                    and sub_df[c].std() > 1e-12]

    if sub_df["covariance"].std() <= 0:
        return None

    # Formulas
    text_terms = " + ".join(prod_cols)
    ctrl_terms = " + ".join(active_ctrl)
    formula_full = f"covariance ~ {text_terms} + {ctrl_terms}"
    formula_ctrl = f"covariance ~ {ctrl_terms}"

    try:
        res_ctrl = smf.ols(formula_ctrl, data=sub_df).fit()
        res_full = smf.ols(formula_full, data=sub_df).fit()
        # HC1 robust fit for Wald F-test
        res_full_hc1 = smf.ols(formula_full, data=sub_df).fit(cov_type='HC1')
    except Exception:
        return None

    adj_r2_ctrl = res_ctrl.rsquared_adj
    adj_r2_full = res_full.rsquared_adj
    delta       = adj_r2_full - adj_r2_ctrl

    # Classical incremental F-test (SSR-based, for reference)
    k_themes    = len(prod_cols)
    ssr_ctrl    = res_ctrl.ssr
    ssr_full    = res_full.ssr
    df_full     = res_full.df_resid
    if df_full > 0 and ssr_full > 0 and k_themes > 0:
        f_stat   = ((ssr_ctrl - ssr_full) / k_themes) / (ssr_full / df_full)
        f_pvalue = f_dist.sf(f_stat, k_themes, df_full)
    else:
        f_stat   = np.nan
        f_pvalue = np.nan

    # HC1-robust Wald F-test: joint H0 that all theme coefficients = 0
    try:
        # Build restriction string: test each theme product column = 0
        restrictions = ", ".join([f"{c} = 0" for c in prod_cols
                                  if c in res_full_hc1.params.index])
        if restrictions:
            wald_result = res_full_hc1.f_test(restrictions)
            f_stat_robust  = float(wald_result.fvalue)
            f_pvalue_robust = float(wald_result.pvalue)
        else:
            f_stat_robust  = np.nan
            f_pvalue_robust = np.nan
    except Exception:
        f_stat_robust  = np.nan
        f_pvalue_robust = np.nan

    return {
        "n_pairs": n,
        "adj_r2_ctrl": adj_r2_ctrl,
        "adj_r2_full": adj_r2_full,
        "delta_adj_r2": delta,
        "f_stat": f_stat,
        "f_pvalue": f_pvalue,
        "f_stat_robust": f_stat_robust,
        "f_pvalue_robust": f_pvalue_robust,
    }



# 4. SPECIFICATIONS

specs = [
    ("Full sample (baseline)",
     lambda df, q: df,
     PROD_COLS),
    ("Pre-GFC (2006Q1–2008Q2)",
     lambda df, q: df if q <= "2008Q2" else None,
     PROD_COLS),
    ("Post-GFC / pre-COVID (2009Q1–2019Q4)",
     lambda df, q: df if "2009Q1" <= q <= "2019Q4" else None,
     PROD_COLS),
    ("COVID + post (2020Q1–2024Q4)",
     lambda df, q: df if q >= "2020Q1" else None,
     PROD_COLS),
    ("Large–Large pairs",
     lambda df, q: df[(df["large_i"] == 1) & (df["large_j"] == 1)],
     PROD_COLS),
    ("Small–Small pairs",
     lambda df, q: df[(df["large_i"] == 0) & (df["large_j"] == 0)],
     PROD_COLS),
    ("Same SIC4 pairs (homogeneous)",
     lambda df, q: df[df["same_sic4"] == 1],
     PROD_COLS),
    ("Different SIC4 pairs (heterogeneous)",
     lambda df, q: df[df["same_sic4"] == 0],
     PROD_COLS),
    ("Excluding top 5 themes (34 remaining)",
     lambda df, q: df,
     remaining_prod_cols),
]



# 5. RUN ALL ROBUSTNESS TESTS (quarter by quarter)

spec_results = {name: [] for name, _, _ in specs}
quarters = sorted(pair_df["quarter"].astype(str).unique())
print(f"\nProcessing {len(quarters)} quarters across {len(specs)} specs...")
print("-" * 70)

for qi, quarter in enumerate(quarters):
    qdf = pair_df[pair_df["quarter"].astype(str) == quarter]
    n_q = len(qdf)
    if n_q < MIN_OBS:
        continue

    for name, filter_fn, prod_cols in specs:
        sub = filter_fn(qdf, quarter)
        if sub is None or len(sub) < MIN_OBS:
            continue
        result = run_spec(sub, prod_cols, ALL_CONTROLS)
        if result is not None:
            result["quarter"] = quarter
            spec_results[name].append(result)

    if (qi + 1) % 10 == 0 or qi == 0:
        n_specs_active = sum(1 for name in spec_results if len(spec_results[name]) > 0)
        print(f"  [{qi+1:3d}/{len(quarters)}] {quarter}: "
              f"{n_q:,} pairs, {n_specs_active} specs active")
    sys.stdout.flush()

print("-" * 70)
print("All quarters processed.")



# 6. AGGREGATE RESULTS

print("\nAggregating results...")
summary_rows = []
for name, _, prod_cols in specs:
    qresults = spec_results[name]
    if len(qresults) == 0:
        print(f"  {name}: no valid quarters, skipping")
        continue
    rdf = pd.DataFrame(qresults)
    row = {
        "specification":   name,
        "quarters":        len(rdf),
        "avg_n_pairs":     int(rdf["n_pairs"].mean()),
        "n_themes":        len(prod_cols),
        "avg_ctrl_r2":     rdf["adj_r2_ctrl"].mean(),
        "avg_full_r2":     rdf["adj_r2_full"].mean(),
        "avg_delta_r2":    rdf["delta_adj_r2"].mean(),
        "median_delta_r2": rdf["delta_adj_r2"].median(),
        "pct_positive":    (rdf["delta_adj_r2"] > 0).mean() * 100,
        "pct_f_sig_01":    (rdf["f_pvalue"] < 0.01).mean() * 100,
        "pct_f_sig_05":    (rdf["f_pvalue"] < 0.05).mean() * 100,
        "avg_f_stat":      rdf["f_stat"].mean(),
        # HC1-robust Wald F-test statistics
        "pct_f_robust_sig_01": (rdf["f_pvalue_robust"] < 0.01).mean() * 100,
        "pct_f_robust_sig_05": (rdf["f_pvalue_robust"] < 0.05).mean() * 100,
        "pct_f_robust_sig_10": (rdf["f_pvalue_robust"] < 0.10).mean() * 100,
        "avg_f_stat_robust":   rdf["f_stat_robust"].mean(),
    }
    summary_rows.append(row)
    print(f"  {name}: {row['quarters']}Q, Avg Δ R²={row['avg_delta_r2']:.6f}, "
          f"Δ>0: {row['pct_positive']:.0f}%, "
          f"F sig 1%: {row['pct_f_sig_01']:.0f}% (classical) / "
          f"{row['pct_f_robust_sig_01']:.0f}% (HC1-robust), "
          f"avg F: {row['avg_f_stat']:.2f} / {row['avg_f_stat_robust']:.2f}")

summary_df = pd.DataFrame(summary_rows)



# 7. SAVE CSV

csv_path = os.path.join(OUTPUT_DIR, "robustness_table.csv")
summary_df.to_csv(csv_path, index=False)
print(f"\nSaved: {csv_path}")



# 8. FORMATTED TEXT TABLE

lines = []
lines.append("Table 7: Robustness of Marginal Text Contribution to Pairwise Covariance")
lines.append("=" * 170)
lines.append("")
lines.append(f"{'Specification':<42} {'Q':>4} {'Avg N':>8} {'K':>4} "
             f"{'Ctrl R²':>8} {'Full R²':>8} {'Avg Δ R²':>10} {'Med Δ R²':>10} "
             f"{'Δ>0':>5} {'F sig':>6} {'avg F':>7} {'F sig':>6} {'avg F':>7}")
lines.append(f"{'':42} {'':>4} {'pairs':>8} {'':>4} "
             f"{'(adj)':>8} {'(adj)':>8} {'(adj)':>10} {'(adj)':>10} "
             f"{'(%)':>5} {'1%(%)':>6} {'':>7} {'1%HC1':>6} {'HC1':>7}")
lines.append("-" * 170)
for _, row in summary_df.iterrows():
    lines.append(
        f"{row['specification']:<42} {row['quarters']:>4} {row['avg_n_pairs']:>8,} "
        f"{row['n_themes']:>4} "
        f"{row['avg_ctrl_r2']:>8.4f} {row['avg_full_r2']:>8.4f} "
        f"{row['avg_delta_r2']:>10.6f} {row['median_delta_r2']:>10.6f} "
        f"{row['pct_positive']:>5.0f} {row['pct_f_sig_01']:>6.0f} "
        f"{row['avg_f_stat']:>7.2f} {row['pct_f_robust_sig_01']:>6.0f} "
        f"{row['avg_f_stat_robust']:>7.2f}"
    )
lines.append("-" * 170)
lines.append("")
lines.append("Notes:")
if USE_PARQUET:
    lines.append("  Data source: pairwise_with_theme_products_v7.parquet (canonical pipeline,")
    lines.append("  produced by 11_build_pairwise_dataset.py → 15i_build_pairwise_theme_products_")
    lines.append("  v7_xsecstd.py). Full-sample baseline row is bit-identical to 16j.")
else:
    lines.append("  Data source: CSV reconstruction (fallback path — pyarrow/fastparquet not")
    lines.append("  available; numbers will differ from 16j canonical by ~2%).")
lines.append("  Each row re-runs the quarterly pairwise covariance regression on the specified")
lines.append("  subsample using statsmodels OLS on per-quarter winsorized (1/99 pctl) daily return")
lines.append("  covariance. Controls: pairwise products of log_assets, log_age, cash/assets,")
lines.append("  loans/assets, capital, neg_earnings + SIC2/3/4 same-industry dummies (six of HH's")
lines.append("  seven bank-level controls; loss_prov./assets excluded — comp.bank not in")
lines.append("  subscription, no funda proxy).")
lines.append("  Theme regressors: pairwise products of cross-sectionally standardized loadings")
lines.append("  (z_ik × z_jk).")
lines.append("  Ctrl R² = controls-only adj. R². Full R² = controls + text themes adj. R².")
lines.append("  Avg Δ R² = average marginal contribution of text themes across quarters.")
lines.append("  Δ>0    = pct of quarters where text themes improve adj. R² over controls only.")
lines.append("  F sig 1% = pct of quarters where classical incremental F-test rejects H₀ at 1%.")
lines.append("  avg F = average classical incremental F statistic across quarters.")
lines.append("  F sig 1%HC1 = pct of quarters where HC1-robust Wald F-test rejects H₀ at 1%.")
lines.append("  avg F HC1 = average HC1-robust Wald F statistic across quarters.")
lines.append("  K = number of theme product variables in the regression.")
lines.append("  Bank size: within-year median split on log(total assets) from prior fiscal year.")

txt_path = os.path.join(OUTPUT_DIR, "robustness_table.txt")
with open(txt_path, "w") as f:
    f.write("\n".join(lines))
print(f"Saved: {txt_path}")

print("\n" + "=" * 70)
print(f"DONE — 18f Robustness Table  (data source: "
      f"{'parquet ✓' if USE_PARQUET else 'CSV reconstruction'})")
print("=" * 70)
