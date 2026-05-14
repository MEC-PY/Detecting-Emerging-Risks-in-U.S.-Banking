# Identify redundant cluster pairs via Spearman correlation on annual
# mean loadings and cosine similarity on document-level loading vectors.
# Produces flagged pairs for manual review, does not auto-remove.

import os
import numpy as np
import pandas as pd


# PATHS

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(BASE, "..", "data", "outputs_textual_factors_v2")

T1_CSV     = os.path.join(OUT_DIR, "track1_selected.csv")
T2_CSV     = os.path.join(OUT_DIR, "track2_selected.csv")
TOPICS_CSV = os.path.join(OUT_DIR, "first_doc_topics.csv")
META_CSV   = os.path.join(OUT_DIR, "document_metadata.csv")


# SETTINGS

SPEARMAN_THRESHOLD = 0.90    # Flag pairs with |Spearman| > this
COSINE_THRESHOLD   = 0.70    # Flag pairs with cosine similarity > this
# NOTE: With only 20 annual data points, Spearman is noisy and over-flags.
# Cosine on document-level loadings is the primary redundancy signal.
# A pair is truly redundant when BOTH metrics are high.


# HELPERS

def spearman_corr(x, y):
    """Pure numpy Spearman rank correlation."""
    n = len(x)
    if n < 3:
        return 0.0
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    mx, my = rx.mean(), ry.mean()
    num = np.sum((rx - mx) * (ry - my))
    den = np.sqrt(np.sum((rx - mx)**2) * np.sum((ry - my)**2))
    return float(num / den) if den > 0 else 0.0


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    return float(dot / (na * nb)) if (na > 0 and nb > 0) else 0.0


# MAIN
def main():
    print("=" * 70)
    print("REDUNDANCY FILTER")
    print("=" * 70)

    # ---- Load selected clusters from both tracks ----
    t1 = pd.read_csv(T1_CSV)
    t2 = pd.read_csv(T2_CSV)

    # Normalise columns for union
    common_cols = ["cluster_id", "label", "cv", "mean_loading",
                   "singular_value", "top_words", "track"]
    for col in common_cols:
        if col not in t1.columns:
            t1[col] = ""
        if col not in t2.columns:
            t2[col] = ""

    union = pd.concat([t1[common_cols], t2[common_cols]], ignore_index=True)
    union = union.drop_duplicates(subset="cluster_id")
    union = union.sort_values("singular_value", ascending=False).reset_index(drop=True)

    # Add taxonomy info from Track 2 where available
    tax_cols = ["taxonomy_match", "taxonomy_source"]
    for col in tax_cols:
        if col in t2.columns:
            tax_map = dict(zip(t2["cluster_id"], t2[col]))
            union[col] = union["cluster_id"].map(tax_map).fillna("")

    cids = list(union["cluster_id"])
    n = len(cids)
    print(f"\n  Union: {n} clusters")

    # ---- Load document-level loadings ----
    topics = pd.read_csv(TOPICS_CSV)
    metadata = pd.read_csv(META_CSV)

    loading_cols = [f"topic_loading_{cid}" for cid in cids]
    missing = [c for c in loading_cols if c not in topics.columns]
    if missing:
        print(f"  WARNING: {len(missing)} clusters missing from topics file: "
              f"{[c.replace('topic_loading_', '') for c in missing]}")
        loading_cols = [c for c in loading_cols if c in topics.columns]
        cids = [int(c.replace("topic_loading_", "")) for c in loading_cols]
        n = len(cids)

    # Document-level loading matrix (N_docs x N_clusters)
    doc_matrix = topics[loading_cols].values

    # ---- Compute annual mean profiles ----
    merged = pd.DataFrame({
        "year": metadata["year"],
    })
    for i, cid in enumerate(cids):
        merged[f"c_{cid}"] = doc_matrix[:, i]

    merged = merged.dropna(subset=["year"])
    merged["year"] = merged["year"].astype(int)
    annual = merged.groupby("year").mean()
    years = annual.index.values
    print(f"  Years: {years.min()}-{years.max()} ({len(years)} years)")

    # ---- Pairwise Spearman on annual profiles ----
    print(f"\n  Computing pairwise Spearman correlations ({n}x{n})...")
    annual_profiles = {}
    for cid in cids:
        annual_profiles[cid] = annual[f"c_{cid}"].values

    spearman_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            rho = spearman_corr(annual_profiles[cids[i]], annual_profiles[cids[j]])
            spearman_matrix[i, j] = rho
            spearman_matrix[j, i] = rho
        spearman_matrix[i, i] = 1.0

    # ---- Pairwise cosine on document-level loadings ----
    print(f"  Computing pairwise cosine similarities ({n}x{n})...")
    cosine_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            cs = cosine_sim(doc_matrix[:, i], doc_matrix[:, j])
            cosine_matrix[i, j] = cs
            cosine_matrix[j, i] = cs
        cosine_matrix[i, i] = 1.0

    # ---- Flag redundant pairs ----
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            rho = spearman_matrix[i, j]
            cs  = cosine_matrix[i, j]
            flagged_spearman = abs(rho) > SPEARMAN_THRESHOLD
            flagged_cosine   = cs > COSINE_THRESHOLD

            if flagged_spearman or flagged_cosine:
                cid_a, cid_b = cids[i], cids[j]
                r_a = union[union["cluster_id"] == cid_a].iloc[0]
                r_b = union[union["cluster_id"] == cid_b].iloc[0]
                pairs.append({
                    "cluster_a": cid_a,
                    "cluster_b": cid_b,
                    "label_a":   r_a.get("label", ""),
                    "label_b":   r_b.get("label", ""),
                    "track_a":   r_a.get("track", ""),
                    "track_b":   r_b.get("track", ""),
                    "sv_a":      r_a.get("singular_value", 0),
                    "sv_b":      r_b.get("singular_value", 0),
                    "spearman":  round(rho, 4),
                    "cosine":    round(cs, 4),
                    "flag_spearman": flagged_spearman,
                    "flag_cosine":   flagged_cosine,
                    "top_words_a": r_a.get("top_words", ""),
                    "top_words_b": r_b.get("top_words", ""),
                })

    pairs_df = pd.DataFrame(pairs)
    if len(pairs_df):
        pairs_df = pairs_df.sort_values("cosine", ascending=False).reset_index(drop=True)

    # ---- Summary: how many pairs per cluster ----
    pair_counts = {}
    for _, p in pairs_df.iterrows():
        pair_counts[p["cluster_a"]] = pair_counts.get(p["cluster_a"], 0) + 1
        pair_counts[p["cluster_b"]] = pair_counts.get(p["cluster_b"], 0) + 1

    union["redundancy_pairs"] = union["cluster_id"].map(pair_counts).fillna(0).astype(int)

    # ---- Save ----
    union.to_csv(os.path.join(OUT_DIR, "union_clusters.csv"), index=False)
    pairs_df.to_csv(os.path.join(OUT_DIR, "redundancy_pairs.csv"), index=False)

    print(f"\n  Flagged pairs: {len(pairs_df)}")
    print(f"    Spearman > {SPEARMAN_THRESHOLD}: "
          f"{pairs_df['flag_spearman'].sum() if len(pairs_df) else 0}")
    print(f"    Cosine   > {COSINE_THRESHOLD}: "
          f"{pairs_df['flag_cosine'].sum() if len(pairs_df) else 0}")
    print(f"    Both:     "
          f"{(pairs_df['flag_spearman'] & pairs_df['flag_cosine']).sum() if len(pairs_df) else 0}")

    n_involved = len(pair_counts)
    n_clean = n - n_involved
    print(f"\n  Clusters involved in at least one pair: {n_involved}")
    print(f"  Clusters with no redundancy flags:      {n_clean}")

    # Print the pairs
    if len(pairs_df):
        print(f"\n  {'='*90}")
        print(f"  FLAGGED REDUNDANT PAIRS")
        print(f"  {'='*90}")
        for _, p in pairs_df.iterrows():
            flags = []
            if p["flag_spearman"]:
                flags.append(f"Spearman={p['spearman']:.2f}")
            if p["flag_cosine"]:
                flags.append(f"Cosine={p['cosine']:.2f}")
            print(f"  [{p['cluster_a']:3d}] vs [{p['cluster_b']:3d}]  "
                  f"{' | '.join(flags):30s}  "
                  f"SV={p['sv_a']:.0f}/{p['sv_b']:.0f}")
            print(f"    A: {str(p['top_words_a'])[:60]}")
            print(f"    B: {str(p['top_words_b'])[:60]}")
            print()

    print(f"\n  Saved: union_clusters.csv, redundancy_pairs.csv")


if __name__ == "__main__":
    main()
