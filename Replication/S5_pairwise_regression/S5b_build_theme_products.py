# Build pairwise theme products using cross-sectionally standardized loadings.
# Output: output/pairwise_with_theme_products_v7.parquet

import os
import numpy as np
import pandas as pd

BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE, "..", "data")
OUTPUT_DIR = os.path.join(BASE, "..", "output")

LOADINGS_PATH = os.path.join(DATA_DIR,   "outputs_textual_factors_v2", "bank_year_loadings_v7_xsecstd.csv")
PAIRWISE_PATH = os.path.join(OUTPUT_DIR, "pairwise_dataset.parquet")
OUTPUT_PATH   = os.path.join(OUTPUT_DIR, "pairwise_with_theme_products_v7.parquet")

print("Loading bank-year loadings (v7, cross-sectionally standardized)...")
loadings = pd.read_csv(LOADINGS_PATH)
loadings["cik"] = loadings["cik"].astype(str).str.zfill(10)
loading_cols = [c for c in loadings.columns if c.startswith("topic_loading_")]
print(f"  Bank-year rows : {len(loadings):,}")
print(f"  Themes (k)     : {len(loading_cols)}")
print(f"  Year range     : {loadings['year'].min()} – {loadings['year'].max()}")

vals = loadings[loading_cols].values
print(f"  Loading stats  : min={vals.min():.6f}  max={vals.max():.6f}  "
      f"mean={vals.mean():.6f}  zeros={((vals==0).sum()/vals.size*100):.1f}%")

# Load existing pairwise dataset
print("\nLoading pairwise dataset...")
pair_df = pd.read_parquet(PAIRWISE_PATH)
print(f"  Shape          : {pair_df.shape}")
print(f"  Year range     : {pair_df['year'].min()} – {pair_df['year'].max()}")

pair_df["cik_i"] = pair_df["cik_i"].astype(str).str.zfill(10)
pair_df["cik_j"] = pair_df["cik_j"].astype(str).str.zfill(10)

# ── Merge loading for bank i (loading year = pair year - 1) ──
print("\nMerging theme loadings for bank i...")
load_i = loadings[["cik", "year"] + loading_cols].copy()
load_i = load_i.rename(columns={"cik": "cik_i"})
load_i["pair_year"] = load_i["year"] + 1
load_i = load_i.rename(columns={c: f"i_{c}" for c in loading_cols})

pair_df = pair_df.merge(
    load_i[["cik_i", "pair_year"] + [f"i_{c}" for c in loading_cols]],
    left_on  = ["cik_i", "year"],
    right_on = ["cik_i", "pair_year"],
    how      = "left",
).drop(columns=["pair_year"])

n_matched_i = pair_df[[f"i_{loading_cols[0]}"]].notna().sum().iloc[0]
print(f"  Rows with bank i loading: {n_matched_i:,} / {len(pair_df):,} "
      f"({n_matched_i/len(pair_df)*100:.1f}%)")

# Merge loading for bank j
print("Merging theme loadings for bank j...")
load_j = loadings[["cik", "year"] + loading_cols].copy()
load_j = load_j.rename(columns={"cik": "cik_j"})
load_j["pair_year"] = load_j["year"] + 1
load_j = load_j.rename(columns={c: f"j_{c}" for c in loading_cols})

pair_df = pair_df.merge(
    load_j[["cik_j", "pair_year"] + [f"j_{c}" for c in loading_cols]],
    left_on  = ["cik_j", "year"],
    right_on = ["cik_j", "pair_year"],
    how      = "left",
).drop(columns=["pair_year"])

n_matched_both = pair_df[[f"i_{loading_cols[0]}", f"j_{loading_cols[0]}"]].notna().all(axis=1).sum()
print(f"  Rows with BOTH loadings: {n_matched_both:,} / {len(pair_df):,} "
      f"({n_matched_both/len(pair_df)*100:.1f}%)")

# Compute pairwise theme products
print("\nComputing pairwise products...")
prod_cols = []
for col in loading_cols:
    prod_col = f"prod_{col}"
    pair_df[prod_col] = pair_df[f"i_{col}"] * pair_df[f"j_{col}"]
    prod_cols.append(prod_col)

print(f"  Created {len(prod_cols)} pairwise product columns")

# Drop individual loadings
drop_cols = [f"i_{c}" for c in loading_cols] + [f"j_{c}" for c in loading_cols]
pair_df = pair_df.drop(columns=drop_cols)

# Drop missing
pair_df = pair_df.dropna(subset=prod_cols[:1])
print(f"\nAfter dropping missing: {len(pair_df):,} rows")
print(f"Quarters: {pair_df['quarter'].nunique()}")

# Sanity check
sample_prod = pair_df[prod_cols].values
print(f"\nProduct stats (xsec-std):")
print(f"  min={sample_prod.min():.6f}  max={sample_prod.max():.6f}  "
      f"mean={sample_prod.mean():.6f}  std={sample_prod.std():.6f}")

# Save
print(f"\nSaving to: {OUTPUT_PATH}")
pair_df.to_parquet(OUTPUT_PATH, index=False)
print(f"  File size: {os.path.getsize(OUTPUT_PATH) / 1e9:.2f} GB")
print(f"  Final shape: {pair_df.shape}")
print("Done.")
