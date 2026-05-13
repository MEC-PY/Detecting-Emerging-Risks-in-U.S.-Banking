# Build bank_years_to_scrape.csv from CRSP daily returns (1997-2024).
# Keeps firm-years with >= 200 trading days.

import os
import pandas as pd

# Use absolute paths relative to this script's location
BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, "..", "data")

df = pd.read_csv(os.path.join(DATA_DIR, "crsp_daily_banks_1997_2024.csv"))
df["DATE"] = pd.to_datetime(df["DATE"], format="%Y%m%d")
df["YEAR"] = df["DATE"].dt.year

df = df.sort_values(["PERMNO", "DATE"]).reset_index(drop=True)

# Check for duplicates
dup_mask = df.duplicated(subset=["PERMNO", "DATE"], keep=False)
dup_df = df[dup_mask].copy()

if not dup_df.empty:
    check_cols = ["RET", "PRC", "SHROUT", "VOL", "EXCHCD", "SHRCD", "TICKER"]
    consistency_check = (
        dup_df.groupby(["PERMNO", "DATE"])[check_cols]
              .nunique()
              .max()
              .max()
    )
    if consistency_check > 1:
        raise ValueError("Non-identical duplicates detected! Investigate before dropping.")
    else:
        print("Duplicate rows are identical. Safe to drop.")

df = df.drop_duplicates(subset=["PERMNO", "DATE"])
print("Rows after dropping duplicates:", len(df))

print(df.head())
print(df.tail())
print("Years:", df["YEAR"].min(), "-", df["YEAR"].max())
print("Number of banks:", df["PERMNO"].nunique())

bank_years = (
    df.groupby(["PERMNO", "YEAR"])
      .size()
      .reset_index(name="n_days")
)

# Keep only firm-years with at least 200 trading days
bank_years = bank_years[bank_years["n_days"] >= 200]

bank_years.to_csv(os.path.join(DATA_DIR, "bank_years_to_scrape.csv"), index=False)
print("Saved bank_years_to_scrape.csv with rows:", len(bank_years))

# Per-year summary
print("\nBanks per year:")
for yr, grp in bank_years.groupby("YEAR"):
    print(f"  {yr}: {grp['PERMNO'].nunique()} banks")
