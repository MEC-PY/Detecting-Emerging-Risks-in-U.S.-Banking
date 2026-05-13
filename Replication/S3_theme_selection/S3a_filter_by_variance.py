# Filter clusters by coefficient of variation (CV = std/mean).
# Removes flat/boilerplate clusters, keeps temporally dynamic ones.
# Uses CV rather than absolute std so niche emerging-risk topics
# with small absolute loadings aren't penalised.
# Restricted to the canonical 7,384 regression sample to avoid look-ahead.

import os
import re
import ast
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(BASE, "..", "data", "outputs_textual_factors_v2")

TOPICS_CSV   = os.path.join(OUT_DIR, "first_doc_topics.csv")
METADATA_CSV = os.path.join(OUT_DIR, "document_metadata.csv")
WORDS_CSV    = os.path.join(OUT_DIR, "topics_words.csv")

# Canonical regression-sample keys (cik, loading_year) used by 16j/18a-f.
# Restricting CV computation to this key set pre-empts any look-ahead from
# embedding-only documents influencing theme selection.
SAMPLE_KEYS_CSV = os.path.join(BASE, "..", "data", "main_regression_sample_keys.csv")

# Settings
# Coefficient of variation floor: remove clusters with CV (std/mean) BELOW this.
# CV measures relative temporal volatility, normalised for loading magnitude.
# A CV of 0.35 means a cluster's year-to-year variation is at least 35% of its
# average level — this cleanly separates dynamic risk topics from flat boilerplate
# while preserving niche emerging risk topics with small absolute loadings.
CV_FLOOR_THRESHOLD = 0.40

# Minimum mean loading: safety net to avoid clusters that are essentially empty
# (near-zero loadings across the board, making CV unstable).
MIN_MEAN_LOADING = 0.05

# Minimum years a cluster must be active to be considered at all
MIN_YEARS_ACTIVE = 1

print("Loading outputs...")
topics   = pd.read_csv(TOPICS_CSV)
metadata = pd.read_csv(METADATA_CSV)
words    = pd.read_csv(WORDS_CSV)

loading_cols = [c for c in topics.columns if c.startswith("topic_loading_")]
print(f"  Documents     : {len(topics):,}")
print(f"  Clusters found: {len(loading_cols)}")

# Restrict to canonical 7,384 regression sample
# CV is computed ONLY on the 7,384 bank-years that survive all three filters
# (CRSP returns, non-missing Compustat controls, regression year 2006-2024).
# This is the SAME set of documents used downstream in 16j and 18a-f, so no
# look-ahead-only document can influence which clusters are flagged as high-CV.
print("\nRestricting to canonical regression sample...")
sample_keys = pd.read_csv(SAMPLE_KEYS_CSV)
sample_keys["cik"] = sample_keys["cik"].astype(int)
sample_keys["year"] = sample_keys["loading_year"].astype(int)
sample_set = set(zip(sample_keys["cik"].tolist(), sample_keys["year"].tolist()))
print(f"  Canonical sample keys: {len(sample_set):,}")

metadata_all = metadata.copy()
metadata_all["cik"] = metadata_all["cik"].astype(int)
metadata_all["year"] = metadata_all["year"].astype(int)
metadata_canon = metadata_all[
    metadata_all.apply(lambda r: (r["cik"], r["year"]) in sample_set, axis=1)
].copy()
canonical_docs = set(metadata_canon["document"].tolist())
print(f"  Documents in canonical sample: {len(canonical_docs):,}")
print(f"  Documents excluded (embedding-only): {len(metadata_all) - len(canonical_docs):,}")

print("\nMelting to long format...")
long_full = topics.melt(
    id_vars="document",
    value_vars=loading_cols,
    var_name="col",
    value_name="topic_loading",
)
long_full["cluster_id"] = long_full["col"].str.extract(r"topic_loading_(\d+)").astype(int)
long_full = long_full.drop(columns="col")
long_full = long_full.merge(metadata_all[["document", "year"]], on="document", how="left")
long_full = long_full.dropna(subset=["year"])
long_full["year"] = long_full["year"].astype(int)

# Canonical sample subset (used for the main CV calculation)
long = long_full[long_full["document"].isin(canonical_docs)].copy()
print(f"  Long-format rows: full={len(long_full):,}, canonical={len(long):,}")

print("Computing temporal variance per cluster on canonical sample...")


def _cluster_cv(df_long):
    annual = (
        df_long
        .groupby(["cluster_id", "year"])["topic_loading"]
        .mean()
        .reset_index()
        .rename(columns={"topic_loading": "mean_loading"})
    )
    s = (
        annual
        .groupby("cluster_id")["mean_loading"]
        .agg(temporal_std="std", mean_loading="mean", n_years="count")
        .reset_index()
    )
    s["temporal_std"] = s["temporal_std"].fillna(0)
    s["cv"] = s["temporal_std"] / s["mean_loading"].replace(0, np.nan)
    s["cv"] = s["cv"].fillna(0)
    return s


stats = _cluster_cv(long)
stats = stats[stats["n_years"] >= MIN_YEARS_ACTIVE].copy()

# Also compute on the full 8,253-doc corpus so the robustness appendix has
# both numbers side by side. Downstream code uses the canonical `cv` column;
# `cv_full_corpus` is carried for diagnostics only.
stats_full = _cluster_cv(long_full).rename(
    columns={
        "temporal_std": "temporal_std_full",
        "mean_loading": "mean_loading_full",
        "n_years": "n_years_full",
        "cv": "cv_full_corpus",
    }
)
stats = stats.merge(stats_full[["cluster_id", "cv_full_corpus", "mean_loading_full"]],
                    on="cluster_id", how="left")

stats_sorted = stats.sort_values("cv", ascending=False).reset_index(drop=True)

print(f"  Clusters with >= {MIN_YEARS_ACTIVE} active years: {len(stats_sorted)}")
print(f"\n  Temporal std distribution:")
print(stats_sorted["temporal_std"].describe().round(4))
print(f"\n  CV (std/mean) distribution:")
print(stats_sorted["cv"].describe().round(4))

# Apply CV floor + minimum mean loading
kept = stats_sorted[
    (stats_sorted["cv"] > CV_FLOOR_THRESHOLD) &
    (stats_sorted["mean_loading"] >= MIN_MEAN_LOADING)
].copy()
dropped = stats_sorted[~stats_sorted["cluster_id"].isin(kept["cluster_id"])]
kept_ids = set(kept["cluster_id"].tolist())

print(f"\n  CV floor threshold  : cv > {CV_FLOOR_THRESHOLD}")
print(f"  Min mean loading    : >= {MIN_MEAN_LOADING}")
print(f"  Clusters removed    : {len(dropped)}  (flat boilerplate)")
print(f"  Clusters kept       : {len(kept)}  (for Streamlit review)")

# Save outputs

# 1. Full variance stats (all clusters, for reference)
stats_sorted.to_csv(os.path.join(OUT_DIR, "cluster_variance_stats.csv"), index=False)
print(f"\nSaved cluster_variance_stats.csv  ({len(stats_sorted)} rows)")

# 2. Filtered wide-format topics
kept_cols = ["document"] + [
    f"topic_loading_{cid}" for cid in sorted(kept_ids)
    if f"topic_loading_{cid}" in topics.columns
]
filtered_topics = topics[kept_cols].copy()
filtered_topics.to_csv(os.path.join(OUT_DIR, "first_doc_topics_filtered.csv"), index=False)
print(f"Saved first_doc_topics_filtered.csv  "
      f"({len(filtered_topics)} docs, {len(kept_cols)-1} clusters)")

# 3. Filtered words
filtered_words = words[words["topic"].isin(kept_ids)].copy()
filtered_words.to_csv(os.path.join(OUT_DIR, "topics_words_filtered.csv"), index=False)
print(f"Saved topics_words_filtered.csv")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].hist(stats_sorted["cv"], bins=60, edgecolor="white", color="steelblue")
axes[0].axvline(CV_FLOOR_THRESHOLD, color="red", linestyle="--",
                label=f"CV Floor = {CV_FLOOR_THRESHOLD}\n"
                      f"Removed {len(dropped)} clusters below")
axes[0].set_title("Distribution of CV = std/mean (all clusters)")
axes[0].set_xlabel("Coefficient of Variation (temporal volatility relative to mean)")
axes[0].set_ylabel("Number of Clusters")
axes[0].legend(fontsize=8)

top10_ids = stats_sorted.head(10)["cluster_id"].tolist()
top10_annual = (
    long[long["cluster_id"].isin(top10_ids)]
    .groupby(["cluster_id", "year"])["topic_loading"]
    .mean()
    .reset_index()
    .rename(columns={"topic_loading": "mean_loading"})
)
for cid, grp in top10_annual.groupby("cluster_id"):
    axes[1].plot(grp["year"], grp["mean_loading"], marker="o", markersize=3,
                 label=f"cluster {cid}")
axes[1].set_title("Top 10 Highest-CV Clusters (time series)")
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Mean Loading")
axes[1].legend(fontsize=7, ncol=2)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cluster_variance_plot.png"), dpi=150)
plt.close()
print(f"Saved cluster_variance_plot.png")

def parse_top_words(dist_str, n=6):
    try:
        cleaned = re.sub(r"np\.float64\(([^)]+)\)", r"\1", dist_str)
        d = ast.literal_eval(cleaned)
        return sorted(d, key=d.get, reverse=True)[:n]
    except Exception:
        return []

word_lookup = dict(zip(words["topic"], words["topic_distribution"]))

print("\n--- Top 20 most dynamic clusters ---")
for _, row in stats_sorted.head(20).iterrows():
    cid       = int(row["cluster_id"])
    top_words = parse_top_words(word_lookup.get(cid, ""))
    print(f"  Cluster {cid:4d} | std={row['temporal_std']:.5f} "
          f"| years={int(row['n_years'])} | {', '.join(top_words)}")

print(f"\nDone. Now run the Streamlit app to manually review and select clusters.")
print(f"  cd '{os.path.dirname(BASE)}'")
print(f"  streamlit run Scripts/cluster_review_app.py")
