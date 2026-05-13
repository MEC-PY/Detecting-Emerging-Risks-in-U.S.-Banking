# Streamlit app for Track 2: review taxonomy-matched low-CV clusters.
# Shows taxonomy category/source alongside each cluster.
# Exports track2_selected.csv.

import os
import re
import ast
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


# PATHS

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(BASE, "..", "data", "outputs_textual_factors_v2")

TAXONOMY_CSV = os.path.join(OUT_DIR, "track2_taxonomy_candidates.csv")
WORDS_CSV    = os.path.join(OUT_DIR, "topics_words.csv")
SV_CSV       = os.path.join(OUT_DIR, "singular_values.csv")
META_CSV     = os.path.join(OUT_DIR, "document_metadata.csv")
TOPICS_CSV   = os.path.join(OUT_DIR, "first_doc_topics.csv")
EXPORT_CSV   = os.path.join(OUT_DIR, "track2_selected.csv")

PAGE_SIZE = 10


# PAGE CONFIG

st.set_page_config(page_title="Track 2 — Taxonomy Review", layout="wide")


# DATA LOADING

@st.cache_data
def load_data():
    tax_df   = pd.read_csv(TAXONOMY_CSV)
    words_df = pd.read_csv(WORDS_CSV)
    sv_df    = pd.read_csv(SV_CSV)
    metadata = pd.read_csv(META_CSV)
    topics   = pd.read_csv(TOPICS_CSV)

    # Only the selected ones (taxonomy_score >= threshold)
    selected = tax_df[tax_df["selected"] == True].copy()
    selected = selected.sort_values("singular_value", ascending=False).reset_index(drop=True)

    # Build annual mean loadings for time series
    cids = list(selected["cluster_id"])
    loading_cols = [f"topic_loading_{cid}" for cid in cids
                    if f"topic_loading_{cid}" in topics.columns]
    if loading_cols:
        long = topics.melt(
            id_vars="document", value_vars=loading_cols,
            var_name="col", value_name="loading",
        )
        long["cluster_id"] = long["col"].str.extract(r"topic_loading_(\d+)").astype(int)
        long.drop(columns="col", inplace=True)
        long = long.merge(metadata[["document", "year"]], on="document", how="left")
        long.dropna(subset=["year"], inplace=True)
        long["year"] = long["year"].astype(int)
        annual = (
            long.groupby(["cluster_id", "year"])["loading"]
            .mean().reset_index()
            .rename(columns={"loading": "mean_loading"})
        )
    else:
        annual = pd.DataFrame(columns=["cluster_id", "year", "mean_loading"])

    sv_lookup = dict(zip(sv_df["cluster"], sv_df["leading_singular"]))
    return selected, annual, words_df, sv_lookup


@st.cache_data
def parse_word_weights(dist_str):
    try:
        cleaned = re.sub(r"np\.float64\(([^)]+)\)", r"\1", str(dist_str))
        d = ast.literal_eval(cleaned)
        return sorted(d.items(), key=lambda x: -x[1])
    except Exception:
        return []


def fmt_words(dist_str, n=8):
    pairs = parse_word_weights(dist_str)[:n]
    return ", ".join(f"{w} ({v:.2f})" for w, v in pairs) if pairs else ""


tax_df, annual, words_df, sv_lookup = load_data()
word_lookup = dict(zip(words_df["topic"], words_df["topic_distribution"]))
all_ids = list(tax_df["cluster_id"])


# SESSION STATE — all taxonomy matches start as KEPT

if "t2_status" not in st.session_state:
    if os.path.exists(EXPORT_CSV):
        prev = pd.read_csv(EXPORT_CSV)
        prev_ids = set(prev["cluster_id"])
        st.session_state.t2_status = {
            cid: "kept" if cid in prev_ids else "removed"
            for cid in all_ids
        }
    else:
        st.session_state.t2_status = {cid: "kept" for cid in all_ids}

if "t2_labels" not in st.session_state:
    # Pre-fill labels from taxonomy match
    if os.path.exists(EXPORT_CSV):
        prev = pd.read_csv(EXPORT_CSV)
        st.session_state.t2_labels = dict(zip(prev["cluster_id"], prev.get("label", "")))
    else:
        st.session_state.t2_labels = dict(zip(
            tax_df["cluster_id"],
            tax_df["taxonomy_match"].fillna(""),
        ))

if "t2_page" not in st.session_state:
    st.session_state.t2_page = 0
if "t2_focus" not in st.session_state:
    st.session_state.t2_focus = all_ids[0] if all_ids else None


# SIDEBAR

st.sidebar.title("Track 2 — Taxonomy Review")
st.sidebar.markdown(
    f"**{len(all_ids)}** clusters matched the risk taxonomy.\n\n"
    "All start as **kept**. Deselect weak or false matches."
)
st.sidebar.markdown("---")

n_kept    = sum(1 for v in st.session_state.t2_status.values() if v == "kept")
n_removed = sum(1 for v in st.session_state.t2_status.values() if v == "removed")
st.sidebar.markdown(f"✅ **{n_kept}** kept · ❌ **{n_removed}** removed")
st.sidebar.markdown("---")

# Category filter
categories = sorted(tax_df["taxonomy_match"].dropna().unique())
cat_filter = st.sidebar.selectbox("Filter by category", ["All"] + categories)

keyword_filter = st.sidebar.text_input(
    "Filter by keyword", placeholder="e.g. mortgage, cyber",
).strip().lower()

status_filter = st.sidebar.radio(
    "Show", options=["All", "Kept", "Removed"], horizontal=True,
)

sort_option = st.sidebar.selectbox(
    "Sort by",
    options=["Singular Value", "Taxonomy Score", "CV", "Cluster ID"],
)

st.sidebar.markdown("---")
if st.sidebar.button("💾 Export kept clusters", type="primary"):
    rows = []
    for cid in all_ids:
        if st.session_state.t2_status.get(cid) == "kept":
            r = tax_df[tax_df["cluster_id"] == cid]
            if len(r):
                r = r.iloc[0]
                rows.append({
                    "cluster_id"      : cid,
                    "label"           : st.session_state.t2_labels.get(cid, ""),
                    "taxonomy_match"  : r["taxonomy_match"],
                    "taxonomy_score"  : int(r["taxonomy_score"]),
                    "taxonomy_source" : r["taxonomy_source"],
                    "matched_keywords": r["matched_keywords"],
                    "cv"              : float(r["cv"]),
                    "mean_loading"    : float(r["mean_loading"]),
                    "singular_value"  : float(r["singular_value"]),
                    "top_words"       : r["top_words"],
                    "track"           : "track2_taxonomy",
                })
    if rows:
        out = pd.DataFrame(rows).sort_values("singular_value", ascending=False)
        out.to_csv(EXPORT_CSV, index=False)
        st.sidebar.success(f"Saved {len(rows)} clusters → track2_selected.csv")
    else:
        st.sidebar.warning("No clusters kept.")


# FILTER + SORT

display = tax_df.copy()

if cat_filter != "All":
    display = display[display["taxonomy_match"] == cat_filter]

if keyword_filter:
    def has_kw(cid):
        return any(keyword_filter in w
                   for w, _ in parse_word_weights(word_lookup.get(cid, ""))[:30])
    display = display[display["cluster_id"].apply(has_kw)]

if status_filter == "Kept":
    display = display[display["cluster_id"].apply(
        lambda c: st.session_state.t2_status.get(c) == "kept")]
elif status_filter == "Removed":
    display = display[display["cluster_id"].apply(
        lambda c: st.session_state.t2_status.get(c) == "removed")]

sort_map = {
    "Singular Value":  ("singular_value", False),
    "Taxonomy Score":  ("taxonomy_score", False),
    "CV":              ("cv", False),
    "Cluster ID":      ("cluster_id", True),
}
scol, sasc = sort_map[sort_option]
display = display.sort_values(scol, ascending=sasc).reset_index(drop=True)

total_pages = max(1, (len(display) + PAGE_SIZE - 1) // PAGE_SIZE)
st.session_state.t2_page = min(st.session_state.t2_page, total_pages - 1)


# MAIN TABLE

st.title("Track 2 — Taxonomy-Matched Cluster Review")
st.caption(f"Showing {len(display)} clusters  |  Low-CV, matched to risk taxonomy")

# Pagination
pc1, pc2, pc3 = st.columns([1, 2, 1])
with pc1:
    if st.button("← Prev", disabled=(st.session_state.t2_page == 0)):
        st.session_state.t2_page -= 1
        st.rerun()
with pc2:
    st.markdown(
        f"<div style='text-align:center'>Page {st.session_state.t2_page+1} / {total_pages}</div>",
        unsafe_allow_html=True,
    )
with pc3:
    if st.button("Next →", disabled=(st.session_state.t2_page >= total_pages - 1)):
        st.session_state.t2_page += 1
        st.rerun()

# Header
hdr = st.columns([0.4, 0.5, 0.4, 0.4, 1.5, 3.0, 0.6, 0.6, 0.6])
hdr[0].markdown("**#**")
hdr[1].markdown("**SV**")
hdr[2].markdown("**CV**")
hdr[3].markdown("**Score**")
hdr[4].markdown("**Taxonomy Match**")
hdr[5].markdown("**Top Words (weight)**")

page_start = st.session_state.t2_page * PAGE_SIZE
page_df    = display.iloc[page_start : page_start + PAGE_SIZE]

for _, row in page_df.iterrows():
    cid = int(row["cluster_id"])
    cur = st.session_state.t2_status.get(cid, "kept")
    sv_val = row["singular_value"]
    cv_val = row["cv"]
    tax_match = row["taxonomy_match"]
    tax_score = int(row["taxonomy_score"])
    matched_kw = row.get("matched_keywords", "")

    fade = "opacity:0.4;" if cur == "removed" else ""

    cols = st.columns([0.4, 0.5, 0.4, 0.4, 1.5, 3.0, 0.6, 0.6, 0.6])
    cols[0].markdown(f"**{cid}**")
    cols[1].markdown(f"<span style='{fade}'>{sv_val:.0f}</span>", unsafe_allow_html=True)
    cols[2].markdown(f"<span style='{fade}'>{cv_val:.2f}</span>", unsafe_allow_html=True)
    cols[3].markdown(f"<span style='{fade}'>{tax_score}</span>", unsafe_allow_html=True)
    cols[4].markdown(
        f"<span style='font-size:0.85em; {fade}'>"
        f"**{tax_match}**<br><span style='color:gray;font-size:0.8em'>{matched_kw}</span>"
        f"</span>",
        unsafe_allow_html=True,
    )
    cols[5].markdown(
        f"<span style='font-size:0.85em; {fade}'>"
        f"{fmt_words(word_lookup.get(cid, ''), n=7)}</span>",
        unsafe_allow_html=True,
    )
    with cols[6]:
        if st.button("✅", key=f"t2k_{cid}", use_container_width=True,
                      type="primary" if cur == "kept" else "secondary"):
            st.session_state.t2_status[cid] = "kept"
            st.rerun()
    with cols[7]:
        if st.button("❌", key=f"t2r_{cid}", use_container_width=True,
                      type="primary" if cur == "removed" else "secondary"):
            st.session_state.t2_status[cid] = "removed"
            st.rerun()
    with cols[8]:
        if st.button("🔍", key=f"t2i_{cid}", use_container_width=True):
            st.session_state.t2_focus = cid
            st.rerun()


# DETAIL PANEL

st.markdown("---")
fid = st.session_state.get("t2_focus")
if fid is not None and fid in set(all_ids):
    r = tax_df[tax_df["cluster_id"] == fid]
    if len(r):
        r = r.iloc[0]
        cur_status = st.session_state.t2_status.get(fid, "kept")
        status_lbl = {"kept": "✅ Kept", "removed": "❌ Removed"}[cur_status]

        st.markdown(f"### Cluster {fid} — {status_lbl}")

        # Taxonomy info box
        st.info(
            f"**Taxonomy match:** {r['taxonomy_match']}  \n"
            f"**Score:** {int(r['taxonomy_score'])} keyword matches  \n"
            f"**Matched keywords:** {r['matched_keywords']}  \n"
            f"**Source:** {r['taxonomy_source']}  \n"
            f"**All category matches:** {r.get('all_matches', '')}"
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Singular Value", f"{r['singular_value']:,.0f}")
        m2.metric("CV", f"{r['cv']:.3f}")
        m3.metric("Mean Loading", f"{r['mean_loading']:.4f}")
        m4.metric("Taxonomy Score", f"{int(r['taxonomy_score'])}")

        # Label input
        new_label = st.text_input(
            "Label (editable)", value=st.session_state.t2_labels.get(fid, ""),
            placeholder="e.g. Real Estate Lending", key=f"t2_lbl_{fid}",
        )
        st.session_state.t2_labels[fid] = new_label

        # Word weights chart + table
        dist_str = word_lookup.get(fid, "")
        ww = parse_word_weights(dist_str)

        if ww:
            max_w = ww[0][1]
            col_chart, col_table = st.columns([3, 2])

            with col_chart:
                top_n = min(25, len(ww))
                cw = [w for w, _ in ww[:top_n]][::-1]
                cv_ = [v for _, v in ww[:top_n]][::-1]

                # Highlight matched keywords
                matched_set = set(str(r.get("matched_keywords", "")).split(", "))
                colors = ["#ff6b6b" if w in matched_set else "#4da6ff" for w in cw]

                fig = go.Figure(go.Bar(
                    x=cv_, y=cw, orientation="h",
                    marker_color=colors,
                    text=[f"{v:.3f}" for v in cv_],
                    textposition="outside",
                ))
                fig.update_layout(
                    title=f"Top {top_n} Words (red = taxonomy match)",
                    xaxis_title="Weight",
                    height=max(300, top_n * 24),
                    margin=dict(l=10, r=30, t=40, b=30),
                    plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e",
                    font=dict(color="white", size=11),
                )
                st.plotly_chart(fig, use_container_width=True)

            with col_table:
                ww_df = pd.DataFrame(ww, columns=["Word", "Weight"])
                ww_df["Rank"] = range(1, len(ww_df) + 1)
                ww_df["% of Top"] = (ww_df["Weight"] / max_w * 100).round(1)
                ww_df["Matched"] = ww_df["Word"].apply(
                    lambda w: "✓" if w in matched_set else ""
                )
                ww_df = ww_df[["Rank", "Word", "Weight", "% of Top", "Matched"]]
                st.dataframe(
                    ww_df,
                    column_config={
                        "Rank":     st.column_config.NumberColumn("Rank",     width=50, format="%d"),
                        "Word":     st.column_config.TextColumn("Word",       width=160),
                        "Weight":   st.column_config.NumberColumn("Weight",   width=80, format="%.4f"),
                        "% of Top": st.column_config.ProgressColumn("% of Top", min_value=0, max_value=100, width=80),
                        "Matched":  st.column_config.TextColumn("Match",      width=50),
                    },
                    hide_index=True,
                    height=min(600, len(ww_df) * 36 + 40),
                    use_container_width=True,
                )

        # Time series
        st.markdown("**Annual Mean Loading**")
        cl_ann = annual[annual["cluster_id"] == fid].sort_values("year")
        if len(cl_ann):
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cl_ann["year"], y=cl_ann["mean_loading"],
                mode="lines+markers",
                line=dict(color="#4da6ff", width=2), marker=dict(size=6),
            ))
            fig.update_layout(
                xaxis_title="Year", yaxis_title="Mean Loading",
                height=300, margin=dict(l=40, r=20, t=20, b=40),
                plot_bgcolor="#1e1e1e", paper_bgcolor="#1e1e1e",
                font=dict(color="white"),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Action buttons
        ka, kb, _ = st.columns([2, 2, 6])
        with ka:
            if st.button("✅ Keep", use_container_width=True, key="t2_dk"):
                st.session_state.t2_status[fid] = "kept"
                st.rerun()
        with kb:
            if st.button("❌ Remove", use_container_width=True, key="t2_dr"):
                st.session_state.t2_status[fid] = "removed"
                st.rerun()
else:
    st.info("Click 🔍 on a cluster to inspect it.")
