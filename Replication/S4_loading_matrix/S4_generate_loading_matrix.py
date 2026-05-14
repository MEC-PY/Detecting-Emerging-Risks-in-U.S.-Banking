# Generate bank-year loading matrix with cross-sectional standardization.
# Within each year, z-score each theme's loading across banks to remove
# secular trend in loading magnitudes from declining sparsity over time.

import os
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(BASE, "..", "data", "outputs_textual_factors_v2")

FINAL_THEMES_PATH = os.path.join(OUT_DIR, "final_themes.csv")
TOPICS_PATH       = os.path.join(OUT_DIR, "first_doc_topics.csv")
META_PATH         = os.path.join(OUT_DIR, "document_metadata.csv")
OUTPUT_PATH       = os.path.join(OUT_DIR, "bank_year_loadings_v7_xsecstd.csv")

# Load final themes
final = pd.read_csv(FINAL_THEMES_PATH)
final_ids = [int(x) for x in final["cluster_id"].tolist()]
print(f"Final themes: {len(final_ids)}")

# Load topic loadings
print("\nLoading topic matrix (pre-SVD normalized)...")
topics = pd.read_csv(TOPICS_PATH)
all_loading_cols = sorted([c for c in topics.columns if c.startswith("topic_loading_")])
print(f"  Total clusters: {len(all_loading_cols)}")
print(f"  Documents: {len(topics):,}")

# Select final themes
keep_cols = [f"topic_loading_{cid}" for cid in final_ids
             if f"topic_loading_{cid}" in topics.columns]
print(f"  Selected {len(keep_cols)} themes")

# Load metadata
print("\nLoading metadata...")
meta = pd.read_csv(META_PATH)
meta["cik"] = meta["cik"].astype(str).str.split(".").str[0].str.zfill(10)

# Merge
keep_final = ["document"] + keep_cols
df = topics[keep_final].merge(meta[["document", "cik", "year"]], on="document", how="inner")
print(f"  After merge: {len(df):,} rows")

# Keep first 10-K per bank-year
df = df.sort_values("document")
df = df.drop_duplicates(subset=["cik", "year"], keep="first")
print(f"  After dedup: {len(df):,} rows")
print(f"  Unique banks: {df['cik'].nunique()}")
print(f"  Year range:   {df['year'].min()} – {df['year'].max()}")

# PRE-STANDARDIZATION diagnostics
X_raw = df[keep_cols].values
print(f"\n  RAW loading statistics (before cross-sectional std):")
print(f"    Min:       {X_raw.min():.6f}")
print(f"    Max:       {X_raw.max():.6f}")
print(f"    Mean:      {X_raw.mean():.6f}")
print(f"    Std:       {X_raw.std():.6f}")
print(f"    Zeros:     {(X_raw == 0).sum()} / {X_raw.size} "
      f"({(X_raw == 0).sum() / X_raw.size * 100:.1f}%)")

print(f"\n  Per-year RAW loading magnitude:")
for yr, grp in df.groupby("year"):
    vals = grp[keep_cols].values
    mean_abs = np.abs(vals).mean()
    zeros_pct = (vals == 0).sum() / vals.size * 100
    print(f"    {yr}: mean_abs={mean_abs:.4f}  zeros={zeros_pct:.1f}%  n={len(grp)}")
# CROSS-SECTIONAL STANDARDIZATION

print("\n" + "=" * 60)
print("CROSS-SECTIONAL STANDARDIZATION (within-year z-score)")
print("=" * 60)

for col in keep_cols:
    yr_mean = df.groupby("year")[col].transform("mean")
    yr_std  = df.groupby("year")[col].transform("std")
    # Protect against zero std (theme not mentioned by anyone that year)
    yr_std = yr_std.replace(0, 1.0)
    df[col] = (df[col] - yr_mean) / yr_std

# POST-STANDARDIZATION diagnostics
X_std = df[keep_cols].values
print(f"\n  STANDARDIZED loading statistics:")
print(f"    Min:       {X_std.min():.4f}")
print(f"    Max:       {X_std.max():.4f}")
print(f"    Mean:      {X_std.mean():.6f}  (should be ~0)")
print(f"    Std:       {X_std.std():.4f}   (should be ~1)")

print(f"\n  Per-year STANDARDIZED loading magnitude:")
for yr, grp in df.groupby("year"):
    vals = grp[keep_cols].values
    mean_val = vals.mean()
    std_val  = vals.std()
    mean_abs = np.abs(vals).mean()
    print(f"    {yr}: mean={mean_val:.4f}  std={std_val:.4f}  "
          f"mean_abs={mean_abs:.4f}  n={len(grp)}")

# Verify: no secular trend in mean_abs
yearly_mean_abs = []
for yr, grp in df.groupby("year"):
    yearly_mean_abs.append(np.abs(grp[keep_cols].values).mean())
print(f"\n  Mean_abs range across years: "
      f"{min(yearly_mean_abs):.4f} – {max(yearly_mean_abs):.4f}")
print(f"  Ratio (last/first):  {yearly_mean_abs[-1]/yearly_mean_abs[0]:.2f}x  "
      f"(should be ~1.0)")

# Save
out = df[["cik", "year"] + keep_cols].copy()
out.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved: {OUTPUT_PATH}")
print(f"Shape: {out.shape}")
