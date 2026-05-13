# Build HH-style quarterly pairwise dataset.
# Dep var: quarterly covariance of daily returns, winsorized 1%/99%.
# Controls are pairwise dot products of bank-level variables (HH Eq 3).
# Accounting controls use prior fiscal year (fyear = quarter_year - 1).
# Output: output/pairwise_dataset.parquet

import os
import pandas as pd
import numpy as np
from itertools import combinations

BASE        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE, "..", "data")
OUTPUT_DIR  = os.path.join(BASE, "..", "output")

# Use original 2006-2024 CRSP (smaller, fits in memory) with extended link/fundamentals
CRSP_PATH   = os.path.join(DATA_DIR, "crsp_daily_banks_2006_2024.csv")
LINK_PATH   = os.path.join(DATA_DIR, "permno_cik_wrds_extended.csv")
FUND_PATH   = os.path.join(DATA_DIR, "bank_fundamentals_hh_extended.csv")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "pairwise_dataset.parquet")

# HH CONTROL VARIABLES (Panel A + B)
# Each becomes a pairwise dot product: c_i * c_j
HH_CONTROLS = [
    # Panel A — Compustat
    "log_assets",
    "log_age",
    # Panel B — Call Report items approximated from funda (six of HH's seven
    # bank-level controls; loss_prov_allow_assets dropped because we have no
    # defensible proxy — see module docstring)
    "cash_assets",
    "loans_assets",          # rect/at from funda (defensible for banks)
    "capital",
    "neg_earn",
    # EXCLUDED (comp.bank not in our WRDS subscription, no defensible proxy):
    #   "loss_prov_allow_assets"  (pll from comp.bank)
    #   "npa_assets"              (npatac from comp.bank)
    #   "bhc_dummy"               (Call Report RSSD9364)
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# LOAD DAILY RETURNS
print("Loading CRSP daily data...")
crsp = pd.read_csv(CRSP_PATH)

crsp["DATE"]    = pd.to_datetime(crsp["DATE"], format="%Y%m%d")
crsp["year"]    = crsp["DATE"].dt.year
crsp["quarter"] = crsp["DATE"].dt.to_period("Q")
crsp            = crsp[["PERMNO", "DATE", "RET", "year", "quarter"]]

# Drop missing or non-numeric returns
crsp["RET"] = pd.to_numeric(crsp["RET"], errors="coerce")
crsp        = crsp.dropna(subset=["RET"])

# Only need 2006+ for covariance (text starts 2005, earliest cov quarter = 2006Q1)
crsp = crsp[crsp["year"] >= 2006].copy()
print(f"  After year filter (>= 2006): {len(crsp):,} rows")

# LINK PERMNO → CIK
print("Linking PERMNO to CIK...")
link = pd.read_csv(LINK_PATH)

crsp = crsp.merge(link[["permno", "cik"]], left_on="PERMNO", right_on="permno", how="inner")
crsp = crsp.drop(columns=["permno"])
crsp["cik"] = crsp["cik"].astype(str).str.split(".").str[0].str.zfill(10)

# COMPUTE QUARTERLY COVARIANCE
# Winsorized at 1%/99% per quarter (HH footnote 19)
print("Computing quarterly covariances (with winsorization)...")

pairwise_records = []

for q, group in crsp.groupby("quarter"):
    pivot      = group.pivot_table(index="DATE", columns="cik", values="RET")
    cov_matrix = pivot.cov()
    banks      = cov_matrix.columns.tolist()

    # Collect all pair covariances for this quarter
    pairs = list(combinations(banks, 2))
    covs  = [cov_matrix.loc[i, j] for i, j in pairs]

    # Winsorize at 1%/99% within this quarter
    cov_arr = np.array(covs, dtype=float)
    lo = np.nanpercentile(cov_arr, 1)
    hi = np.nanpercentile(cov_arr, 99)
    cov_arr_w = np.clip(cov_arr, lo, hi)

    for (ci, cj), cov_w in zip(pairs, cov_arr_w):
        pairwise_records.append({
            "cik_i":      ci,
            "cik_j":      cj,
            "year":       q.year,
            "quarter":    str(q),
            "covariance": cov_w,
        })

pair_df = pd.DataFrame(pairwise_records)
print(f"  Total pairwise rows: {len(pair_df):,}")

# LOAD FUNDAMENTALS
print("Loading fundamentals...")
fund = pd.read_csv(FUND_PATH)

# Map gvkey → CIK via permno link
fund = fund.merge(link[["permno", "cik"]], on="permno", how="inner")
fund["cik"] = fund["cik"].astype(str).str.split(".").str[0].str.zfill(10)

# MERGE FUNDAMENTALS WITH 1-YEAR LAG
# HH: "ex ante data from the prior fiscal year t-1"
# Quarter in year t  →  fyear = t - 1
print("Merging fundamentals (lagged by 1 fiscal year)...")
pair_df["fyear"] = pair_df["year"] - 1

fund_i = fund.rename(columns={"cik": "cik_i"})
fund_j = fund.rename(columns={"cik": "cik_j"})

pair_df = pair_df.merge(
    fund_i,
    left_on=["cik_i", "fyear"],
    right_on=["cik_i", "fyear"],
    how="left"
)

pair_df = pair_df.merge(
    fund_j,
    left_on=["cik_j", "fyear"],
    right_on=["cik_j", "fyear"],
    how="left",
    suffixes=("_i", "_j")
)

pair_df = pair_df.drop(columns=["fyear"])

# CONSTRUCT PAIRWISE CONTROLS (DOT PRODUCTS)
# HH Equation 3: X_{i,j,t-1} = X_{i,t-1} · X_{j,t-1}
print("Constructing pairwise controls...")

for c in HH_CONTROLS:
    ci, cj = f"{c}_i", f"{c}_j"
    if ci in pair_df.columns and cj in pair_df.columns:
        pair_df[c] = pair_df[ci] * pair_df[cj]
    else:
        print(f"  WARNING: {c} not found in fundamentals — skipping")

# SIC SAME-INDUSTRY INDICATORS
for n_digits, col_name in [(2, "same_sic2"), (3, "same_sic3"), (4, "same_sic4")]:
    pair_df[col_name] = (
        pair_df["sich_i"].astype(str).str[:n_digits] ==
        pair_df["sich_j"].astype(str).str[:n_digits]
    ).astype(int)

# CLEAN UP INTERMEDIATE COLUMNS
drop_cols = [
    col for col in pair_df.columns
    if (col.endswith("_i") or col.endswith("_j"))
    and col not in ("cik_i", "cik_j")
]
pair_df = pair_df.drop(columns=drop_cols)

# Drop rows missing any core control
available_controls = [c for c in HH_CONTROLS if c in pair_df.columns]
pair_df = pair_df.dropna(subset=available_controls)

# Final CIK formatting
pair_df["cik_i"] = pair_df["cik_i"].astype(str).str.split(".").str[0].str.zfill(10)
pair_df["cik_j"] = pair_df["cik_j"].astype(str).str.split(".").str[0].str.zfill(10)

# SUMMARY
print(f"\nFinal rows: {len(pair_df):,}")
print(f"Quarters:   {pair_df['quarter'].nunique()}")
print(f"Unique CIKs in pairs: {pd.concat([pair_df['cik_i'], pair_df['cik_j']]).nunique()}")
print(f"\nControl variable coverage (non-null rate):")
print(pair_df[available_controls + ["same_sic2", "same_sic3", "same_sic4"]].notna().mean().round(3).to_string())

# SAVE
print("\nSaving pairwise dataset...")
pair_df.to_parquet(OUTPUT_PATH, index=False)
print(f"Saved to: {OUTPUT_PATH}")
