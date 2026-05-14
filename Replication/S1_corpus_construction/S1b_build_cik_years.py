import os
import json
import pandas as pd

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "..", "data")

# Load inputs
bank_years = pd.read_csv(os.path.join(DATA_DIR, "bank_years_to_scrape.csv"))
# Force CIK to load as string to avoid float corruption (e.g., 1036030.0)
permno_cik = pd.read_csv(os.path.join(DATA_DIR, "permno_cik_wrds.csv"), dtype={"CIK": str})

# Normalize column names
bank_years.columns = [c.upper() for c in bank_years.columns]
permno_cik.columns = [c.upper() for c in permno_cik.columns]

# Ensure PERMNO is integer in both datasets (avoid silent merge mismatches)
bank_years["PERMNO"] = bank_years["PERMNO"].astype(int)
permno_cik["PERMNO"] = permno_cik["PERMNO"].astype(int)

# Merge PERMNO -> CIK
m = bank_years.merge(
    permno_cik[["PERMNO", "CIK"]],
    on="PERMNO",
    how="left"
)

# --- Mapping diagnostics (PERMNO-level) ---
# Unique PERMNOs in the bank_years universe
permno_universe = pd.DataFrame({"PERMNO": bank_years["PERMNO"].dropna().astype(int).unique()})

# PERMNO -> CIK mapping availability
permno_map = permno_universe.merge(
    permno_cik[["PERMNO", "CIK"]],
    on="PERMNO",
    how="left"
)
permno_map["has_cik"] = permno_map["CIK"].notna().astype(int)

n_permno_total = int(len(permno_map))
n_permno_mapped = int(permno_map["has_cik"].sum())
n_permno_unmapped = int(n_permno_total - n_permno_mapped)
permno_success_rate = (n_permno_mapped / n_permno_total) if n_permno_total else 0.0

# Fail loudly if mapping rate unexpectedly drops
if permno_success_rate < 0.95:
    raise ValueError(
        f"PERMNO->CIK mapping rate below 95% ({permno_success_rate:.1%}). "
        "Investigate before proceeding with scraping."
    )

# Save unmapped PERMNOs for inspection
permno_unmapped_df = permno_map[permno_map["has_cik"] == 0].copy()
permno_unmapped_df.to_csv(os.path.join(DATA_DIR, "permno_without_cik.csv"), index=False)

# Drop banks without CIK (cannot scrape 10-Ks)
m = m.dropna(subset=["CIK"]).copy()

# Ensure clean types and fix any corrupted CIKs (e.g., "1036030.0")
m["CIK"] = (
    m["CIK"]
    .astype(str)
    .str.replace(".0", "", regex=False)   # remove float artifact if present
    .str.strip()
    .str.zfill(10)                        # enforce 10-digit SEC format
)
m["YEAR"] = m["YEAR"].astype(int)

# Keep unique CIK–YEAR pairs
cik_years = (
    m[["CIK", "YEAR"]]
    .drop_duplicates()
    .sort_values(["CIK", "YEAR"])
    .reset_index(drop=True)
)

# --- Mapping diagnostics (CIK-YEAR level) ---
# Share of bank_year rows (PERMNO-YEAR) that have a CIK
n_bank_year_rows_total = int(len(bank_years))
n_bank_year_rows_mapped = int(m[["PERMNO", "YEAR"]].drop_duplicates().shape[0])
# Note: m has only rows with CIK, but may have duplicate PERMNO-YEAR due to merges; we drop dupes above.
bank_year_row_success_rate = (n_bank_year_rows_mapped / n_bank_year_rows_total) if n_bank_year_rows_total else 0.0

# Save
cik_years.to_csv(os.path.join(DATA_DIR, "cik_years_to_scrape.csv"), index=False)

# Write a small summary JSON for thesis logging/reproducibility
summary = {
    "permno_total": n_permno_total,
    "permno_mapped": n_permno_mapped,
    "permno_unmapped": n_permno_unmapped,
    "permno_success_rate": permno_success_rate,
    "bank_year_rows_total": n_bank_year_rows_total,
    "bank_year_rows_mapped": n_bank_year_rows_mapped,
    "bank_year_row_success_rate": bank_year_row_success_rate,
    "cik_unique": int(cik_years["CIK"].nunique()),
    "cik_year_rows": int(len(cik_years)),
    "year_min": int(cik_years["YEAR"].min()) if len(cik_years) else None,
    "year_max": int(cik_years["YEAR"].max()) if len(cik_years) else None,
}
with open(os.path.join(DATA_DIR, "cik_mapping_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Saved cik_years_to_scrape.csv")
print("Saved permno_without_cik.csv")
print("Saved cik_mapping_summary.json")
print(f"PERMNO->CIK mapped: {n_permno_mapped}/{n_permno_total} ({permno_success_rate:.1%})")
print(f"PERMNO-YEAR rows with CIK: {n_bank_year_rows_mapped}/{n_bank_year_rows_total} ({bank_year_row_success_rate:.1%})")
print("Number of CIKs:", cik_years["CIK"].nunique())
print("Number of CIK-YEAR observations:", len(cik_years))
print("Year range:", cik_years["YEAR"].min(), "-", cik_years["YEAR"].max())