# Consolidate redundant themes via iterative cosine-based removal,
# then confirm VIF < 10 on the final set (raw loadings, HH footnote 24).

import os
import numpy as np
import pandas as pd
from collections import defaultdict


# PATHS

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(BASE, "..", "data", "outputs_textual_factors_v2")

T1_CSV     = os.path.join(OUT_DIR, "track1_selected.csv")
T2_CSV     = os.path.join(OUT_DIR, "track2_selected.csv")
TOPICS_CSV = os.path.join(OUT_DIR, "first_doc_topics.csv")
SV_CSV     = os.path.join(OUT_DIR, "singular_values.csv")
META_CSV   = os.path.join(OUT_DIR, "document_metadata.csv")


# SETTINGS

COSINE_THRESHOLD = 0.90   # Pairs above this trigger iterative removal
VIF_THRESHOLD    = 10.0   # HH footnote 24
VIF_SAMPLE_PAIRS = 50000
RANDOM_SEED      = 42



# HELPERS

def cosine_sim(a, b):
    dot = np.dot(a, b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(dot / (na * nb)) if na > 0 and nb > 0 else 0.0


def find_connected_components(graph):
    """Find connected components in an adjacency dict."""
    visited = set()
    components = []

    def dfs(node, comp):
        visited.add(node)
        comp.add(node)
        for nb in graph.get(node, set()):
            if nb not in visited:
                dfs(nb, comp)

    for node in graph:
        if node not in visited:
            comp = set()
            dfs(node, comp)
            components.append(comp)
    return components


def compute_vif(X, col_idx):
    """VIF for column col_idx via OLS on all other columns."""
    y = X[:, col_idx]
    others = np.delete(X, col_idx, axis=1)
    Z = np.hstack([np.ones((others.shape[0], 1)), others])
    try:
        beta = np.linalg.lstsq(Z, y, rcond=None)[0]
        y_hat = Z @ beta
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        if ss_tot == 0:
            return float("inf")
        r2 = min(1 - ss_res / ss_tot, 0.9999)
        return 1.0 / (1.0 - r2)
    except Exception:
        return float("inf")


# MAIN
def main():
    print("=" * 70)
    print("THEME CONSOLIDATION")
    print("=" * 70)

    # ---- Load union of selected clusters ----
    t1 = pd.read_csv(T1_CSV)
    t2 = pd.read_csv(T2_CSV)

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

    # Add taxonomy info
    if "taxonomy_match" in t2.columns:
        tax_map = dict(zip(t2["cluster_id"], t2["taxonomy_match"]))
        union["taxonomy_match"] = union["cluster_id"].map(tax_map).fillna("")
    else:
        union["taxonomy_match"] = ""

    cids = list(union["cluster_id"])
    n = len(cids)
    sv_map = dict(zip(union["cluster_id"], union["singular_value"]))
    print(f"\n  Union: {n} clusters")

    # ---- Load document-level loadings ----
    topics = pd.read_csv(TOPICS_CSV)
    loading_cols = [f"topic_loading_{cid}" for cid in cids
                    if f"topic_loading_{cid}" in topics.columns]
    cids = [int(c.replace("topic_loading_", "")) for c in loading_cols]
    n = len(cids)

    S = topics[loading_cols].values  # (n_docs x n_themes)
    print(f"  Documents: {S.shape[0]}, Themes: {n}")

    # ---- Compute pairwise cosine on raw loadings ----
    print(f"\n  Computing pairwise cosine similarities...")
    cos_matrix = np.zeros((n, n))
    for i in range(n):
        cos_matrix[i, i] = 1.0
        for j in range(i + 1, n):
            cs = cosine_sim(S[:, i], S[:, j])
            cos_matrix[i, j] = cs
            cos_matrix[j, i] = cs

    # Save cosine matrix
    cos_df = pd.DataFrame(cos_matrix, index=cids, columns=cids)
    cos_df.to_csv(os.path.join(OUT_DIR, "cosine_similarity_matrix.csv"))

    # ---- Iterative cosine-based removal ----
    # Like HH's iterative VIF removal: find highest-cosine pair,
    # remove the lower-SV member, repeat until no pair > threshold.
    # This avoids the connected-components chaining problem.
    active_set = set(cids)
    removed = {}       # cid → {reason, replaced_by, cosine, step}
    removal_log = []
    step = 0

    while True:
        # Find the highest cosine pair among active clusters
        best_cos = 0
        best_pair = None
        active_list = sorted(active_set)
        for i, ci in enumerate(active_list):
            idx_i = cids.index(ci)
            for j in range(i + 1, len(active_list)):
                cj = active_list[j]
                idx_j = cids.index(cj)
                cs = cos_matrix[idx_i, idx_j]
                if cs > best_cos:
                    best_cos = cs
                    best_pair = (ci, cj)

        if best_cos <= COSINE_THRESHOLD or best_pair is None:
            break

        step += 1
        ci, cj = best_pair
        sv_i, sv_j = sv_map.get(ci, 0), sv_map.get(cj, 0)

        # Remove the lower-SV member
        if sv_i >= sv_j:
            drop, keep = cj, ci
        else:
            drop, keep = ci, cj

        r_drop = union[union["cluster_id"] == drop].iloc[0]
        r_keep = union[union["cluster_id"] == keep].iloc[0]
        label_drop = str(r_drop.get("label", "") or r_drop.get("taxonomy_match", "") or "")
        label_keep = str(r_keep.get("label", "") or r_keep.get("taxonomy_match", "") or "")
        tw_drop = str(r_drop.get("top_words", ""))[:45]

        removed[drop] = {
            "step": step,
            "replaced_by": keep,
            "cosine": round(best_cos, 3),
        }
        active_set.discard(drop)

        removal_log.append({
            "step": step,
            "removed_id": drop,
            "removed_sv": sv_map.get(drop, 0),
            "removed_label": label_drop,
            "removed_words": tw_drop,
            "kept_id": keep,
            "kept_sv": sv_map.get(keep, 0),
            "kept_label": label_keep,
            "cosine": round(best_cos, 3),
        })

        print(f"\n  Step {step}: cosine={best_cos:.3f}")
        print(f"    DROP [{drop:3d}] SV={sv_map.get(drop,0):7.0f} {label_drop:25s} {tw_drop}")
        print(f"    KEEP [{keep:3d}] SV={sv_map.get(keep,0):7.0f} {label_keep}")

    kept = active_set

    # ---- Chain-break: restore orphaned clusters ----
    # If A was absorbed by B, but B was later removed (absorbed by C),
    # follow the chain to the final representative. If the final rep
    # is in the kept set, A is fine. If NOT, the chain is broken —
    # restore the highest-SV member of that orphan chain.
    def find_final_rep(cid):
        """Follow replaced_by chain to its end."""
        visited = {cid}
        cur = cid
        while cur in removed:
            nxt = removed[cur]["replaced_by"]
            if nxt in visited:
                return None  # cycle (shouldn't happen)
            visited.add(nxt)
            cur = nxt
        return cur  # this is in the kept set (or is the end of chain)

    orphan_chains = defaultdict(list)  # final_rep → [removed cids in its chain]
    for cid in removed:
        final = find_final_rep(cid)
        if final in kept:
            pass  # fine, has a living representative
        else:
            # The whole chain is orphaned — shouldn't happen with our logic
            # since the final rep should always be in kept_set
            pass

    # More targeted fix: detect when a removed cluster's DIRECT representative
    # was itself later removed, creating conceptual distance.
    # For each removed cluster, check: is its replaced_by still in kept?
    # If not, the cluster was absorbed by something that no longer exists.
    restored = set()
    chains_to_restore = defaultdict(list)

    for cid in list(removed.keys()):
        rep = removed[cid]["replaced_by"]
        if rep not in kept:
            # rep was also removed — this is a broken chain
            # Collect the full chain
            chain = [cid]
            cur = rep
            while cur in removed:
                chain.append(cur)
                cur = removed[cur]["replaced_by"]
            # cur is now the final survivor in kept
            # But all members of chain were absorbed through intermediaries
            # that no longer exist. The chain members may be conceptually
            # very different from the final survivor.
            chain_key = cur  # final survivor
            chains_to_restore[chain_key].extend(chain)

    # For each broken chain, restore the highest-SV member (excluding
    # the final survivor which is already kept)
    for final_survivor, chain_members in chains_to_restore.items():
        unique_members = list(set(chain_members))
        if not unique_members:
            continue
        # Sort by singular value, restore the top one
        unique_members.sort(key=lambda c: sv_map.get(c, 0), reverse=True)
        best = unique_members[0]
        restored.add(best)
        kept.add(best)
        r_best = union[union["cluster_id"] == best].iloc[0]
        label_best = str(r_best.get("label", "") or r_best.get("taxonomy_match", "") or "")
        r_surv = union[union["cluster_id"] == final_survivor].iloc[0]
        label_surv = str(r_surv.get("label", "") or r_surv.get("taxonomy_match", "") or "")
        print(f"\n  RESTORED [{best:3d}] SV={sv_map.get(best,0):7.0f} {label_best}")
        print(f"    (was chained through removed intermediaries to [{final_survivor}] {label_surv})")
        print(f"    Chain members: {unique_members}")
        # Remove from removed dict so VIF uses it
        del removed[best]

    if restored:
        print(f"\n  Restored {len(restored)} clusters from broken chains")
    else:
        print(f"\n  No broken chains detected")

    pd.DataFrame(removal_log).to_csv(
        os.path.join(OUT_DIR, "redundancy_groups.csv"), index=False)

    n_removed = len(removed)
    n_kept = len(kept)
    print(f"\n  Kept: {n_kept} themes (including {len(restored)} restored)")
    print(f"  Removed: {n_removed} themes")

    # ---- Print removed ----
    print(f"\n  --- REMOVED ({n_removed}, in order) ---")
    for r in removal_log:
        print(f"    Step {r['step']}: [{r['removed_id']:3d}] SV={r['removed_sv']:7.0f} "
              f"(cosine {r['cosine']:.3f} with [{r['kept_id']}])")

    # ================================================================
    # VIF CONFIRMATION on the kept set
    # ================================================================
    print(f"\n{'='*70}")
    print(f"VIF CONFIRMATION on {n_kept} consolidated themes")
    print(f"{'='*70}")

    kept_cids = sorted(kept)
    kept_cols = [f"topic_loading_{c}" for c in kept_cids
                 if f"topic_loading_{c}" in topics.columns]
    kept_cids = [int(c.replace("topic_loading_", "")) for c in kept_cols]
    S_kept = topics[kept_cols].values
    n_kept_final = len(kept_cids)

    # --- PRIMARY: VIF on raw loadings (HH 2019, footnote 24) ---
    print(f"\n  --- VIF on RAW LOADINGS (HH style, primary check) ---")
    print(f"  Loading matrix: {S_kept.shape}")

    vif_raw = {}
    for k in range(n_kept_final):
        vif_raw[kept_cids[k]] = compute_vif(S_kept, k)

    max_vif_raw_cid = max(vif_raw, key=vif_raw.get)
    max_vif_raw = vif_raw[max_vif_raw_cid]
    print(f"  Max VIF (raw): {max_vif_raw:.1f} (cluster {max_vif_raw_cid})")
    print(f"  All VIF <= {VIF_THRESHOLD}: {'YES' if max_vif_raw <= VIF_THRESHOLD else 'NO'}")

    # --- SUPPLEMENTARY: VIF on pairwise products ---
    print(f"\n  --- VIF on PAIRWISE PRODUCTS (supplementary check) ---")
    rng = np.random.RandomState(RANDOM_SEED)
    n_docs = S_kept.shape[0]
    idx_i = rng.randint(0, n_docs, VIF_SAMPLE_PAIRS * 3)
    idx_j = rng.randint(0, n_docs, VIF_SAMPLE_PAIRS * 3)
    mask = idx_i < idx_j
    idx_i, idx_j = idx_i[mask][:VIF_SAMPLE_PAIRS], idx_j[mask][:VIF_SAMPLE_PAIRS]

    X_prod = S_kept[idx_i, :] * S_kept[idx_j, :]
    print(f"  Pairwise product matrix: {X_prod.shape}")

    vif_prod = {}
    for k in range(n_kept_final):
        vif_prod[kept_cids[k]] = compute_vif(X_prod, k)

    max_vif_prod_cid = max(vif_prod, key=vif_prod.get)
    max_vif_prod = vif_prod[max_vif_prod_cid]
    print(f"  Max VIF (products): {max_vif_prod:.1f} (cluster {max_vif_prod_cid})")
    print(f"  All VIF <= {VIF_THRESHOLD}: {'YES' if max_vif_prod <= VIF_THRESHOLD else 'NO'}")

    # ---- Build final output (use raw-loading VIF as the reported column) ----
    final = union[union["cluster_id"].isin(kept_cids)].copy()
    final["vif"] = final["cluster_id"].map(vif_raw).round(2)
    final["vif_products"] = final["cluster_id"].map(vif_prod).round(2)
    final = final.sort_values("singular_value", ascending=False).reset_index(drop=True)
    final.to_csv(os.path.join(OUT_DIR, "final_themes.csv"), index=False)

    print(f"\n  --- FINAL THEME SET ({n_kept_final} themes) ---")
    for _, r in final.iterrows():
        cid = int(r["cluster_id"])
        sv = r["singular_value"]
        vif = r["vif"]
        cv = r["cv"]
        label = str(r.get("label", "") or r.get("taxonomy_match", "") or "")
        tw = str(r.get("top_words", ""))[:50]
        print(f"    [{cid:3d}] SV={sv:7.0f} VIF={vif:4.1f} CV={cv:.2f} "
              f"{label:25s} {tw}")

    print(f"\n  Saved: final_themes.csv, redundancy_groups.csv, "
          f"cosine_similarity_matrix.csv")


if __name__ == "__main__":
    main()
