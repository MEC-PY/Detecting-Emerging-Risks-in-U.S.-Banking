# WRDS data extraction for HH (2019) replication, 1997-2024 sample.
# Pulls Compustat fundamentals, bank supplement, CRSP daily returns,
# and builds HH control variables. Outputs go to data/.

import os
import wrds
import numpy as np
import pandas as pd

BASE      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE, "..", "data")
FUND_PATH = os.path.join(DATA_DIR, "bank_fundamentals_hh_extended.csv")
CRSP_PATH = os.path.join(DATA_DIR, "crsp_daily_banks_1997_2024.csv")
LINK_PATH = os.path.join(DATA_DIR, "permno_cik_wrds_extended.csv")

os.makedirs(DATA_DIR, exist_ok=True)
SIC_MIN    = 6000
SIC_MAX    = 6199
YEAR_START = 1996    # one year back so 1997 can lag to 1996
YEAR_END   = 2024

db = wrds.Connection()

# 1. COMPUSTAT ANNUAL FUNDAMENTALS
print("\n[1/6] Pulling Compustat annual fundamentals (INDL + FS)...")

funda = db.raw_sql(f"""
    SELECT
        gvkey, fyear, datadate, sich, conm, indfmt,
        at,     -- total assets (universal)
        ni,     -- net income
        ceq,    -- book equity
        che,    -- cash & short-term investments (INDL)
        ch,     -- cash (FS format equivalent of che)
        lt      -- total liabilities (diagnostics)
    FROM comp.funda
    WHERE sich BETWEEN {SIC_MIN} AND {SIC_MAX}
      AND fyear BETWEEN {YEAR_START} AND {YEAR_END}
      AND indfmt  IN ('INDL', 'FS')
      AND datafmt = 'STD'
      AND popsrc  = 'D'
      AND consol  = 'C'
      AND at > 0
""")

# Prefer INDL over FS when both exist for same gvkey-fyear.
funda["_fmt_rank"] = funda["indfmt"].map({"INDL": 0, "FS": 1}).fillna(2)
funda = funda.sort_values(["gvkey", "fyear", "_fmt_rank"])
funda = funda.drop_duplicates(subset=["gvkey", "fyear"])
funda = funda.drop(columns="_fmt_rank")

# Unify cash variable
funda["che"] = funda["che"].combine_first(funda["ch"])
funda = funda.drop(columns="ch", errors="ignore")

print(f"  {len(funda):,} rows, {funda['gvkey'].nunique()} unique gvkeys")

# 2. COMPUSTAT BANK SUPPLEMENT
print("\n[2/6] Pulling Compustat Bank supplement...")

BANK_QUERY = """
    SELECT gvkey, fyear, lntal, rll, pll, npatac
    FROM {table}
    WHERE fyear BETWEEN {start} AND {end}
"""

bank = None
for tbl in ["comp.bank", "compa.bank", "comp_na_annual_update.bank"]:
    try:
        bank = db.raw_sql(BANK_QUERY.format(table=tbl, start=YEAR_START, end=YEAR_END))
        print(f"  {tbl}: {len(bank):,} rows, {bank['gvkey'].nunique()} gvkeys")
        break
    except Exception as e:
        print(f"  {tbl} not available: {e.__class__.__name__}")

if bank is None:
    print("  WARNING: comp.bank not available — using rect/dp from funda as fallback")
    funda_extra = db.raw_sql(f"""
        SELECT gvkey, fyear, indfmt, rect, dp
        FROM comp.funda
        WHERE sich BETWEEN {SIC_MIN} AND {SIC_MAX}
          AND fyear BETWEEN {YEAR_START} AND {YEAR_END}
          AND indfmt IN ('INDL', 'FS')
          AND datafmt = 'STD' AND popsrc = 'D' AND consol = 'C'
    """)
    funda_extra["_fmt_rank"] = funda_extra["indfmt"].map({"INDL": 0, "FS": 1}).fillna(2)
    funda_extra = funda_extra.sort_values(["gvkey", "fyear", "_fmt_rank"])
    funda_extra = funda_extra.drop_duplicates(subset=["gvkey", "fyear"])
    funda_extra = funda_extra.drop(columns=["_fmt_rank", "indfmt"], errors="ignore")
    bank = funda_extra.rename(columns={"rect": "lntal", "dp": "pll"})
    bank["rll"]    = np.nan
    bank["npatac"] = np.nan
    bank["_approximated"] = True

# 3. CRSP-COMPUSTAT LINK (gvkey -> permno)
print("\n[3/6] Pulling CRSP–Compustat link table...")

ccm = db.raw_sql("""
    SELECT gvkey, lpermno AS permno,
           linktype, linkprim, linkdt, linkenddt
    FROM crsp.ccmxpf_linktable
    WHERE linktype IN ('LU','LC')
      AND linkprim IN ('P','C')
""")

ccm["linkdt"]    = pd.to_datetime(ccm["linkdt"])
ccm["linkenddt"] = pd.to_datetime(ccm["linkenddt"]).fillna(pd.Timestamp("2099-12-31"))
print(f"  {len(ccm):,} link rows")

# 4. CIK FROM COMPUSTAT COMPANY TABLE
print("\n[4/6] Pulling CIK from Compustat company table...")

cik_map = db.raw_sql("""
    SELECT gvkey, cik
    FROM comp.company
    WHERE cik IS NOT NULL
""")
cik_map["cik"] = pd.to_numeric(cik_map["cik"], errors="coerce")
cik_map = cik_map.dropna(subset=["cik"])
cik_map["cik"] = cik_map["cik"].astype(int)
print(f"  {len(cik_map):,} gvkey→CIK mappings")

# 5. CRSP DAILY RETURNS (1997-2024)
print("\n[5/6] Identifying PERMNOs and pulling CRSP daily returns (1997-2024)...")

bank_gvkeys = set(funda["gvkey"].astype(str).unique())
ccm_banks   = ccm[ccm["gvkey"].astype(str).isin(bank_gvkeys)]
permnos     = ccm_banks["permno"].dropna().astype(int).unique().tolist()
print(f"  {len(permnos)} unique PERMNOs to pull")

# Pull in batches of 500
BATCH = 500
crsp_chunks = []
for i in range(0, len(permnos), BATCH):
    batch    = permnos[i : i + BATCH]
    perm_str = ",".join(str(p) for p in batch)
    chunk = db.raw_sql(f"""
        SELECT a.permno, a.date, a.ret, a.prc, a.shrout, a.vol,
               b.exchcd, b.shrcd, b.ticker
        FROM crsp.dsf a
        LEFT JOIN crsp.msenames b
            ON  a.permno = b.permno
            AND a.date BETWEEN b.namedt AND COALESCE(b.nameendt, '2099-12-31')
        WHERE a.permno IN ({perm_str})
          AND a.date BETWEEN '1997-01-01' AND '2024-12-31'
    """)
    crsp_chunks.append(chunk)
    print(f"  Batch {i//BATCH + 1}/{-(-len(permnos)//BATCH)}: {len(chunk):,} rows")

crsp = pd.concat(crsp_chunks, ignore_index=True)
crsp.columns = [c.upper() for c in crsp.columns]
crsp["DATE"] = pd.to_datetime(crsp["DATE"]).dt.strftime("%Y%m%d").astype(int)

# CRSP first appearance for bank age calculation
first_crsp = db.raw_sql(f"""
    SELECT permno, MIN(EXTRACT(year FROM date))::int AS first_crsp_year
    FROM crsp.dsf
    WHERE permno IN ({','.join(str(p) for p in permnos)})
    GROUP BY permno
""")

db.close()
print("  WRDS connection closed.")
print(f"  Total CRSP rows: {len(crsp):,}")

# 6. BUILD FUNDAMENTALS
print("\n[6/6] Merging and constructing HH control variables...")

fund = funda.merge(bank[["gvkey","fyear","lntal","rll","pll","npatac"]],
                   on=["gvkey","fyear"], how="left")

fund["datadate"] = pd.to_datetime(fund["datadate"])

# Date-constrained CCM join
fund = fund.merge(ccm[["gvkey","permno","linkdt","linkenddt"]], on="gvkey", how="left")
fund = fund[
    (fund["datadate"] >= fund["linkdt"]) &
    (fund["datadate"] <= fund["linkenddt"])
].copy()

# Bank age
fund = fund.merge(first_crsp, on="permno", how="left")
fund = fund.sort_values(["gvkey","fyear","permno"])
fund = fund.drop_duplicates(subset=["gvkey","fyear"])

print(f"  After merge: {len(fund):,} rows, {fund['gvkey'].nunique()} banks")

# Construct HH variables
at = fund["at"].replace(0, np.nan)

fund["log_assets"]             = np.log(at)
fund["log_age"]                = np.log((fund["fyear"] - fund["first_crsp_year"] + 1).clip(lower=1))
fund["cash_assets"]            = fund["che"] / at
fund["loans_assets"]           = fund["lntal"] / at
fund["loss_prov_allow_assets"] = (fund["rll"].fillna(0) + fund["pll"].fillna(0)) / at
fund["capital"]                = fund["ceq"] / at
fund["neg_earn"]               = (fund["ni"] < 0).astype(float)
fund["npa_assets"]             = fund["npatac"] / at
fund["bhc_dummy"]              = np.nan

# Winsorise ratios
for v in ["cash_assets","loans_assets","loss_prov_allow_assets","capital","npa_assets"]:
    lo, hi = fund[v].quantile(0.01), fund[v].quantile(0.99)
    fund[v] = fund[v].clip(lo, hi)

funda = funda.drop(columns="indfmt", errors="ignore")
keep = ["gvkey","permno","fyear","sich","conm",
        "log_assets","log_age","cash_assets","loans_assets",
        "loss_prov_allow_assets","capital","neg_earn","bhc_dummy","npa_assets",
        "at","ni","ceq"]
fund = fund[[c for c in keep if c in fund.columns]]

# Diagnostics
print(f"\n  Final: {len(fund):,} rows, {fund['gvkey'].nunique()} unique banks")
print(f"  Year range: {fund['fyear'].min()} – {fund['fyear'].max()}")
print("\n  Missing value rates:")
check = ["log_assets","log_age","cash_assets","loans_assets",
         "loss_prov_allow_assets","capital","neg_earn","npa_assets"]
print(fund[[c for c in check if c in fund.columns]].isna().mean().round(3).to_string())

# Per-year bank counts
print("\n  Banks per year:")
year_counts = fund.groupby("fyear")["permno"].nunique()
for yr, cnt in year_counts.items():
    print(f"    {yr}: {cnt} banks")

# Save
print("\nSaving outputs...")

fund.to_csv(FUND_PATH, index=False)
print(f"  Saved: {FUND_PATH}  ({len(fund):,} rows)")

crsp.to_csv(CRSP_PATH, index=False)
print(f"  Saved: {CRSP_PATH}  ({len(crsp):,} rows)")

# PERMNO–CIK link
link = (
    ccm_banks[["gvkey","permno"]]
    .drop_duplicates()
    .merge(cik_map, on="gvkey", how="left")
)
link.to_csv(LINK_PATH, index=False)
print(f"  Saved: {LINK_PATH}  ({len(link):,} rows, {link['cik'].notna().sum()} with CIK)")

print("\nDone. Next steps:")
print("  1. Run 01b_data_management_extended.py (builds bank_years from 1997)")
print("  2. Run 03 (build cik_years_to_scrape)")
print("  3. Run 04b (download 10-K risk text — handles pre-2005)")
