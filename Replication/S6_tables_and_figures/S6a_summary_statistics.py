# Summary statistics (HH Table 2 equivalent).
# Panels A-C: bank chars, pair chars, time-series vars.
# Outputs to output/summary_statistics.csv and .txt.

import os
import numpy as np
import pandas as pd

# Paths
BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE, "..", "data")
OUTPUT_DIR = os.path.join(BASE, "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FUND_PATH    = os.path.join(DATA_DIR, "bank_fundamentals_hh.csv")
LOAD_PATH    = os.path.join(DATA_DIR, "outputs_textual_factors_v2",
                            "bank_year_loadings_v7_xsecstd.csv")
THEMES_PATH  = os.path.join(DATA_DIR, "outputs_textual_factors_v2",
                            "final_themes.csv")
META_PATH    = os.path.join(DATA_DIR, "outputs_textual_factors_v2",
                            "document_metadata.csv")
QMARG_PATH   = os.path.join(OUTPUT_DIR, "quarterly_marginal_r2_v7_xsecstd.csv")
CRSP_PATH    = os.path.join(DATA_DIR, "crsp_daily_banks_2006_2024.csv")
PAIRWISE_PATH = os.path.join(OUTPUT_DIR, "pairwise_with_theme_products_v7.parquet")
SAMPLE_KEYS_PATH = os.path.join(OUTPUT_DIR, "main_regression_sample_keys.csv")
LINK_PATH    = os.path.join(DATA_DIR, "permno_cik_wrds_extended.csv")


def describe_var(series, name):
    """Return descriptive stats for a single variable."""
    s = series.dropna()
    return {
        "Variable": name,
        "Mean": s.mean(),
        "SD": s.std(),
        "Minimum": s.min(),
        "Median": s.median(),
        "Maximum": s.max(),
        "N_obs": len(s),
    }


def format_row(row, n_fmt="int", decimals=3):
    """Format a single row of the summary stats table."""
    n = row["N_obs"]
    if n_fmt == "millions":
        n_str = f"{n/1e6:.1f}M"
    elif n_fmt == "int":
        n_str = f"{n:,.0f}"
    else:
        n_str = f"{n}"
    d = decimals
    return (f"{row['Variable']:<40} {row['Mean']:>10.{d}f} {row['SD']:>10.{d}f} "
            f"{row['Minimum']:>10.{d}f} {row['Median']:>10.{d}f} "
            f"{row['Maximum']:>10.{d}f} {n_str:>12}")



# PANEL A: Bank Characteristics (Compustat Annual Fundamentals)

print("=" * 70)
print("PANEL A: Bank Characteristics (Compustat)")
print("=" * 70)

fund = pd.read_csv(FUND_PATH)
fund = fund[(fund["fyear"] >= 2005) & (fund["fyear"] <= 2024)]
fund["sich"] = pd.to_numeric(fund.get("sich"), errors="coerce")
fund["sic4"] = fund["sich"].astype("Int64")
fund["sic2"] = (fund["sic4"] // 100).astype("Int64")

print(f"  Unfiltered fundamentals: {len(fund):,} bank-years, "
      f"{fund['gvkey'].nunique()} unique gvkeys")

# Restrict to the main regression sample for consistency with Table 4
# Panel A must describe the 7,384 bank-years actually used in the pairwise
# covariance regression (after CIK linkage, loading availability, and the
# main_regression_sample_keys filter produced by 11_build_pairwise_dataset.py).
link = pd.read_csv(LINK_PATH)
link = link.dropna(subset=["cik", "permno"])
link["cik"] = link["cik"].astype(int)
link_permno = link.drop_duplicates(subset=["permno"])[["permno", "cik"]]

fund = fund.merge(link_permno, on="permno", how="inner")
fund["cik"] = fund["cik"].astype(int)
fund = fund.drop_duplicates(subset=["cik", "fyear"], keep="first")

keys = pd.read_csv(SAMPLE_KEYS_PATH)
keys["cik"] = keys["cik"].astype(int)
keys = keys.rename(columns={"loading_year": "fyear"})[["cik", "fyear"]].drop_duplicates()
fund = fund.merge(keys, on=["cik", "fyear"], how="inner")

print(f"  Main regression sample: {len(fund):,} bank-years, "
      f"{fund['cik'].nunique()} unique banks")
print(f"  Year range: {fund['fyear'].min()} - {fund['fyear'].max()}")

panel_a = []
# All variables come from Compustat (funda + bank supplement)
panel_a_vars = [
    ("log_assets", "ln(total assets)"),           # Compustat funda: at
    ("log_age", "ln(bank age)"),                   # CRSP: years since first listing
    ("cash_assets", "Cash/assets"),                # Compustat funda: che / at
    ("loans_assets", "Loans/assets"),              # Compustat funda: rect / at (lntal unavailable)
    ("capital", "Capital (equity/assets)"),         # Compustat funda: ceq / at
    ("neg_earn", "Negative earnings dummy"),        # Compustat funda: ni < 0
    ("at", "Total assets ($M)"),                   # Compustat funda: at
]
for col, label in panel_a_vars:
    if col in fund.columns:
        s = pd.to_numeric(fund[col], errors="coerce")
        if s.notna().sum() > 0:
            panel_a.append(describe_var(s, label))
        else:
            print(f"  WARNING: {col} has no valid values, skipping")
    else:
        print(f"  NOTE: {col} not in fundamentals, skipping")

panel_a_df = pd.DataFrame(panel_a)


# PANEL B: Bank-Pair Characteristics (from pairwise dataset)

print("\n" + "=" * 70)
print("PANEL B: Bank-Pair Characteristics")
print("=" * 70)

panel_b = []
try:
    pw = pd.read_parquet(PAIRWISE_PATH)
    print(f"  Pairwise observations: {len(pw):,}")

    if "covariance" in pw.columns:
        panel_b.append(describe_var(pw["covariance"], "Bank-pair daily covariance"))
    if "same_sic4" in pw.columns:
        panel_b.append(describe_var(pw["same_sic4"], "Same 4-digit SIC"))
    if "same_sic3" in pw.columns:
        panel_b.append(describe_var(pw["same_sic3"], "Same 3-digit SIC"))
    if "same_sic2" in pw.columns:
        panel_b.append(describe_var(pw["same_sic2"], "Same 2-digit SIC"))

    # Control variables in pairwise regression (z_i × z_j products)
    for ctrl_col, ctrl_label in [
        ("log_assets", "ln(assets)_i × ln(assets)_j"),
        ("log_age", "ln(age)_i × ln(age)_j"),
        ("cash_assets", "(Cash/assets)_i × (Cash/assets)_j"),
        ("loans_assets", "(Loans/assets)_i × (Loans/assets)_j"),
        ("capital", "(Capital)_i × (Capital)_j"),
    ]:
        if ctrl_col in pw.columns:
            panel_b.append(describe_var(pw[ctrl_col], ctrl_label))

except Exception as e:
    print(f"  WARNING: Could not read pairwise dataset: {e}")
    print(f"  Panel B will use quarterly regression summary instead.")
    qmarg_tmp = pd.read_csv(QMARG_PATH)
    panel_b.append(describe_var(qmarg_tmp["n"], "Observations per quarter"))

panel_b_df = pd.DataFrame(panel_b)


# PANEL C: Time-Series Variables

print("\n" + "=" * 70)
print("PANEL C: Time-Series Variables")
print("=" * 70)

qmarg = pd.read_csv(QMARG_PATH)
print(f"  Quarters: {len(qmarg)}")
our_quarters = set(qmarg["quarter"])

panel_c = []

# VIX
try:
    import pandas_datareader.data as web
    vix_raw = web.DataReader("VIXCLS", "fred", "2006-01-01", "2024-12-31")
    vix_raw = vix_raw.dropna()
    vix_raw["quarter"] = vix_raw.index.to_period("Q").astype(str)
    vix_q = vix_raw.groupby("quarter")["VIXCLS"].mean()
    vix_q = vix_q[vix_q.index.isin(our_quarters)]
    panel_c.append(describe_var(vix_q, "VIX level"))
    print(f"  VIX loaded: {len(vix_q)} quarters")
except Exception as e:
    print(f"  VIX not available ({e.__class__.__name__}), skipping")

# Avg pairwise covariance & SD returns from CRSP
try:
    print("  Loading CRSP for covariance & volatility...")
    crsp = pd.read_csv(CRSP_PATH)
    # Normalise column names to lowercase (extraction script uppercased them)
    crsp.columns = [c.lower() for c in crsp.columns]
    # Handle date column — stored as integer YYYYMMDD by extraction script
    date_col = "date" if "date" in crsp.columns else crsp.columns[0]
    crsp["date"] = pd.to_datetime(crsp[date_col].astype(str), format="%Y%m%d", errors="coerce")
    crsp = crsp.dropna(subset=["date"])
    crsp["ret"] = pd.to_numeric(crsp["ret"], errors="coerce")
    crsp = crsp.dropna(subset=["ret", "permno"])
    crsp["permno"] = crsp["permno"].astype(int)
    crsp["month"] = crsp["date"].dt.to_period("M")
    crsp["quarter_str"] = crsp["date"].dt.to_period("Q").astype(str)
    print(f"  CRSP: {len(crsp):,} daily obs")

    # Avg pairwise covariance per quarter (from DAILY returns)
    print("  Computing avg pairwise covariance (daily returns, per quarter)...")
    cov_list = []
    for q in sorted(crsp["quarter_str"].unique()):
        if q not in our_quarters:
            continue
        qdata = crsp[crsp["quarter_str"] == q]
        piv = qdata.pivot_table(index="date", columns="permno", values="ret")
        if len(piv) < 10 or piv.shape[1] < 5:
            continue
        # Drop banks with too few daily obs in this quarter
        piv = piv.dropna(axis=1, thresh=int(len(piv) * 0.5))
        if piv.shape[1] < 5:
            continue
        cov_mat = piv.cov()
        mask = ~np.eye(len(cov_mat), dtype=bool)
        avg_cov = np.nanmean(cov_mat.values[mask])
        if not np.isnan(avg_cov):
            cov_list.append({"quarter": q, "avg_cov": avg_cov})
    cov_df = pd.DataFrame(cov_list)
    if len(cov_df) > 0:
        panel_c.append(describe_var(cov_df["avg_cov"], "Avg daily pair covariance"))
        print(f"  Avg covariance: {len(cov_df)} quarters")

    # Cross-sectional SD of quarterly returns
    # Compound daily → quarterly return per bank, then take cross-sectional SD
    qret = crsp.groupby(["permno", "quarter_str"])["ret"].apply(
        lambda x: ((1 + x).prod() - 1)
    ).reset_index()
    qret.columns = ["permno", "quarter_str", "qret"]
    vol_q = qret.groupby("quarter_str")["qret"].std().reset_index()
    vol_q.columns = ["quarter_str", "xsec_sd"]
    vol_q = vol_q[vol_q["quarter_str"].isin(our_quarters)]
    if len(vol_q) > 0:
        panel_c.append(describe_var(vol_q["xsec_sd"],
                                    "Cross-sectional SD quarterly returns"))
        print(f"  SD returns: {len(vol_q)} quarters")

    # Save for 18b to reuse
    ts_path = os.path.join(OUTPUT_DIR, "quarterly_market_measures.csv")
    ts_out = cov_df.copy() if len(cov_df) > 0 else pd.DataFrame(columns=["quarter"])
    if len(vol_q) > 0:
        vol_merge = vol_q.rename(columns={"quarter_str": "quarter"})
        ts_out = ts_out.merge(vol_merge, on="quarter", how="outer")
    ts_out.to_csv(ts_path, index=False)
    print(f"  Saved market measures: {ts_path}")

except Exception as e:
    print(f"  CRSP-based stats failed: {e}")
    import traceback; traceback.print_exc()

# EPU
try:
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context
    epu_url = "https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.csv"
    epu_raw = pd.read_csv(epu_url)
    epu_cols = [c for c in epu_raw.columns if "ncert" in c.lower() or "index" in c.lower()]
    if epu_cols:
        epu_raw["date"] = pd.to_datetime(
            epu_raw["Year"].astype(str) + "-" + epu_raw["Month"].astype(str).str.zfill(2) + "-15"
        )
        epu_raw["quarter"] = epu_raw["date"].dt.to_period("Q").astype(str)
        epu_q = epu_raw.groupby("quarter")[epu_cols[0]].mean()
        epu_q = epu_q[epu_q.index.isin(our_quarters)]
        panel_c.append(describe_var(epu_q, "Econ policy uncertainty (EPU)"))
        print(f"  EPU loaded: {len(epu_q)} quarters")
except Exception as e:
    print(f"  EPU not available ({e.__class__.__name__}), skipping")

# Our R² measures
panel_c.append(describe_var(qmarg["adj_r2_ctrl"],
                            "Cov model adj. R² (controls only)"))
panel_c.append(describe_var(qmarg["delta_adj_r2"],
                            "Cov model adj. R² (text themes, marginal)"))

panel_c_df = pd.DataFrame(panel_c)


# INCREMENTAL F-TEST: Do text themes jointly improve the model?

print("\n" + "=" * 70)
print("INCREMENTAL F-TEST (text themes)")
print("=" * 70)

import math

K_THEMES = 39
P_CTRL   = 9    # 6 bank-level controls + 3 SIC indicators

def f_pvalue(f_stat, df1, df2):
    """P-value from F via Wilson-Hilferty normal approximation to chi2."""
    chi2_stat = f_stat * df1
    if chi2_stat <= 0:
        return 1.0
    z = (pow(chi2_stat / df1, 1/3) - (1 - 2/(9*df1))) / pow(2/(9*df1), 0.5)
    return 0.5 * math.erfc(z / math.sqrt(2))

f_results = []
for _, row in qmarg.iterrows():
    n_obs = int(row["n"])
    r2_full = row["r2_full"]
    r2_ctrl = row["r2_ctrl"]
    delta_r2 = r2_full - r2_ctrl
    df1 = K_THEMES
    df2 = n_obs - P_CTRL - K_THEMES - 1
    if df2 > 0 and (1 - r2_full) > 0 and delta_r2 > 0:
        f_stat = (delta_r2 / df1) / ((1 - r2_full) / df2)
        p_val  = f_pvalue(f_stat, df1, df2)
    else:
        f_stat, p_val = 0.0, 1.0
    sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
    f_results.append({"quarter": row["quarter"], "f_stat": f_stat,
                       "p_val": p_val, "sig": sig, "delta_r2": delta_r2})

f_df = pd.DataFrame(f_results)
n_sig01 = (f_df["p_val"] < 0.01).sum()
n_sig05 = (f_df["p_val"] < 0.05).sum()
n_sig10 = (f_df["p_val"] < 0.10).sum()
n_tot   = len(f_df)

print(f"  H0: All {K_THEMES} theme coefficients = 0")
print(f"  Significant at 1%:   {n_sig01}/{n_tot} quarters ({n_sig01/n_tot*100:.0f}%)")
print(f"  Significant at 5%:   {n_sig05}/{n_tot} quarters ({n_sig05/n_tot*100:.0f}%)")
print(f"  Significant at 10%:  {n_sig10}/{n_tot} quarters ({n_sig10/n_tot*100:.0f}%)")
print(f"  Not significant:     {n_tot - n_sig10}/{n_tot} quarters ({(n_tot-n_sig10)/n_tot*100:.0f}%)")
print(f"  Average F-statistic: {f_df['f_stat'].mean():.1f}")
print(f"  Max F-statistic:     {f_df['f_stat'].max():.1f} ({f_df.loc[f_df['f_stat'].idxmax(), 'quarter']})")

# Save F-test results
f_path = os.path.join(OUTPUT_DIR, "incremental_f_tests.csv")
f_df.to_csv(f_path, index=False)
print(f"  Saved: {f_path}")


# LOADING MATRIX STATS

print("\n" + "=" * 70)
print("LOADING MATRIX STATS")
print("=" * 70)

loadings = pd.read_csv(LOAD_PATH)
themes   = pd.read_csv(THEMES_PATH)
print(f"  Bank-year observations (loadings, unfiltered): {len(loadings):,}")
print(f"  Themes: {len(themes)}")

# Panel D: use the filtered regression sample so every panel describes the
# same 7,384 bank-years.
banks_per_year = fund.groupby("fyear")["cik"].nunique()
banks_per_year.index.name = "year"
print(f"\n  Banks per year in loading matrix:")
for yr, n in banks_per_year.items():
    print(f"    {yr}: {n}")


# SIC DISTRIBUTION

SIC2_LABELS = {
    60: "Depository Institutions (60xx)",
    61: "Non-Depository Credit (61xx)",
}

sic2_dist = fund["sic2"].value_counts().sort_index()
sic4_dist = fund["sic4"].value_counts().sort_values(ascending=False)

print(f"\n  SIC distribution:")
print(f"    Unique SIC4 codes: {fund['sic4'].nunique()}")
for sic2, count in sic2_dist.items():
    label = SIC2_LABELS.get(sic2, f"SIC {sic2}xx")
    pct = count / len(fund) * 100
    print(f"    {label}: {count:,} ({pct:.1f}%)")


# FORMATTED OUTPUT (matching HH Table 2 format)

print("\n" + "=" * 70)
print("FORMATTED OUTPUT")
print("=" * 70)

COL_HEADER = (f"{'Variable':<40} {'Mean':>10} {'SD':>10} {'Minimum':>10} "
              f"{'Median':>10} {'Maximum':>10} {'# obs':>12}")
SEPARATOR  = "-" * 102

output_text = []
output_text.append("Table 2")
output_text.append("Summary Statistics")
output_text.append("=" * 102)
output_text.append(COL_HEADER)
output_text.append(SEPARATOR)

# Panel A: Bank characteristics (Compustat + CRSP-derived)
output_text.append(f"{'':>25}A. Bank characteristics (Compustat / CRSP)")
for _, row in panel_a_df.iterrows():
    output_text.append(format_row(row))

# Panel B: Bank-pair characteristics (from pairwise regression dataset)
output_text.append(f"{'':>25}B. Bank-pair characteristics")
for _, row in panel_b_df.iterrows():
    output_text.append(format_row(row, n_fmt="millions"))

# Panel C: Time-series variables
output_text.append(f"{'':>25}C. Time-series variables")
for _, row in panel_c_df.iterrows():
    # Use 6 decimals for covariance (very small values), 3 for everything else
    dec = 6 if "covariance" in row["Variable"].lower() else 3
    output_text.append(format_row(row, decimals=dec))

output_text.append(SEPARATOR)
output_text.append("")
output_text.append(
    f"Summary statistics for our sample of {len(fund):,} bank-year observations "
    f"from {fund['fyear'].min()} to {fund['fyear'].max()}. "
    f"Panel A reports bank-level characteristics constructed entirely from Compustat "
    f"Annual Fundamentals (funda). The Compustat Bank supplement (comp.bank) is not "
    f"included in the WRDS subscription used here, so HH's loan-portfolio and "
    f"loan-loss-provision items (lntal, rll, pll, npatac) are not directly available. "
    f"Loans/assets is approximated as rect/at (rect captures interest-earning "
    f"receivables from customers for bank filers and is a defensible stand-in for "
    f"the Call Report loan portfolio). HH's loss-provisions-to-assets, non-performing-"
    f"assets-to-assets, and BHC-indicator controls are excluded because no defensible "
    f"funda proxy exists; we therefore run HH's specification with six of their "
    f"seven bank-level controls plus the three pair-level SIC industry indicators. "
    f"The three continuous ratios (cash/assets, loans/assets, capital) are "
    f"winsorized at the 1st and 99th percentiles. ln(bank age) is computed from "
    f"the bank's first appearance in CRSP daily data. "
    f"Panel B reports bank-pair-quarter characteristics from the pairwise regression "
    f"dataset. The bank-pair daily covariance is the quarterly covariance of CRSP daily "
    f"stock returns for a pair of banks, winsorized at 1% and 99%. "
    f"Panel C reports time-series variables across {len(qmarg)} quarterly observations. "
    f"The controls-only adjusted R² is from a regression of bank-pairwise covariance on "
    f"accounting characteristics and SIC industry dummies. The marginal text R² is the "
    f"incremental improvement when textual information from {len(themes)} semantic themes "
    f"is added to the pairwise covariance regression."
)

# Add F-test summary to formatted output
output_text.append("")
output_text.append("Incremental F-test: H0 = all 39 theme coefficients jointly zero")
output_text.append("-" * 60)
output_text.append(f"  Significant at 1%:  {n_sig01}/{n_tot} quarters ({n_sig01/n_tot*100:.0f}%)")
output_text.append(f"  Significant at 5%:  {n_sig05}/{n_tot} quarters ({n_sig05/n_tot*100:.0f}%)")
output_text.append(f"  Significant at 10%: {n_sig10}/{n_tot} quarters ({n_sig10/n_tot*100:.0f}%)")
output_text.append(f"  Average F({K_THEMES}, ~{qmarg['n'].mean():,.0f}): {f_df['f_stat'].mean():.1f}")
output_text.append(f"  Peak: F = {f_df['f_stat'].max():.1f} in {f_df.loc[f_df['f_stat'].idxmax(), 'quarter']}")
output_text.append("-" * 60)

# Add Panel D: Coverage by year
output_text.append("")
output_text.append("Panel D: Sample Coverage by Year")
output_text.append("-" * 40)
output_text.append(f"{'Year':>6} {'N Banks':>10}")
output_text.append("-" * 40)
for yr, n in banks_per_year.items():
    output_text.append(f"{yr:>6} {n:>10}")
output_text.append("-" * 40)

# Add Panel E: SIC distribution
output_text.append("")
output_text.append("Panel E: Industry Distribution (SIC Codes)")
output_text.append("-" * 60)
output_text.append(f"{'SIC2':>6} {'Description':<35} {'N':>8} {'%':>8}")
output_text.append("-" * 60)
for sic2, count in sic2_dist.items():
    label = SIC2_LABELS.get(sic2, f"SIC {sic2}xx")
    pct = count / len(fund) * 100
    output_text.append(f"{sic2:>6} {label:<35} {count:>8,} {pct:>7.1f}%")
output_text.append("-" * 60)
output_text.append(f"\nTop 10 SIC4 codes:")
output_text.append(f"{'SIC4':>6} {'N':>8} {'%':>8}")
for sic4, count in sic4_dist.head(10).items():
    pct = count / len(fund) * 100
    output_text.append(f"{sic4:>6} {count:>8,} {pct:>7.1f}%")

full_text = "\n".join(output_text)
print(full_text)

# Save
txt_path = os.path.join(OUTPUT_DIR, "summary_statistics.txt")
with open(txt_path, "w") as f:
    f.write(full_text)

# Save combined CSV
all_panels = pd.concat([
    panel_a_df.assign(Panel="A: Bank Characteristics (Compustat / CRSP)"),
    panel_b_df.assign(Panel="B: Bank-Pair Characteristics"),
    panel_c_df.assign(Panel="C: Time-Series Variables"),
], ignore_index=True)

csv_path = os.path.join(OUTPUT_DIR, "summary_statistics.csv")
all_panels.to_csv(csv_path, index=False)

# Save banks per year
bpy_path = os.path.join(OUTPUT_DIR, "banks_per_year.csv")
banks_per_year.reset_index().to_csv(bpy_path, index=False)

print(f"\nSaved: {txt_path}")
print(f"Saved: {csv_path}")
print(f"Saved: {bpy_path}")
print("Done!")
