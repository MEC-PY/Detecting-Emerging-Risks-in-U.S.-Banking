# Theme drivers by period: decompose aggregate delta-R2 into
# per-theme contributions for each elevated episode.

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# Paths
BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE, "..", "data")
OUTPUT_DIR = os.path.join(BASE, "..", "output")
THEME_R2   = os.path.join(OUTPUT_DIR, "individual_theme_r2_v7_xsecstd.csv")
THEMES_CSV = os.path.join(DATA_DIR, "outputs_textual_factors_v2", "final_themes.csv")


# 1. LOAD DATA

print("Loading individual theme R² data...")
df = pd.read_csv(THEME_R2)
df["date"] = pd.to_datetime(df["quarter"].str.replace(
    r"(\d{4})Q(\d)", lambda m: f"{m.group(1)}-{int(m.group(2))*3:02d}-28", regex=True
))
df["year"] = df["date"].dt.year
print(f"  {len(df):,} rows: {df['theme_id'].nunique()} themes × {df['quarter'].nunique()} quarters")

# Load z-scores (pre-computed by 18c)
ZSCORE_CSV = os.path.join(OUTPUT_DIR, "individual_theme_zscore_timeseries.csv")
if os.path.exists(ZSCORE_CSV):
    zdf = pd.read_csv(ZSCORE_CSV, usecols=["quarter", "theme_id", "z_score"])
    df = df.merge(zdf, on=["quarter", "theme_id"], how="left")
    print(f"  Merged z-scores from {ZSCORE_CSV}")
else:
    print("  WARNING: z-score CSV not found, peak_z will be NaN")
    df["z_score"] = np.nan

# Theme labels: economic names (primary) + top 3 cluster words (secondary)
themes_meta = pd.read_csv(THEMES_CSV)
cluster_words = {}
for _, r in themes_meta.iterrows():
    cid = int(r["cluster_id"])
    tw = str(r.get("top_words", ""))
    words = [w.strip().replace("_", " ") for w in tw.split(",")][:2]
    words = [w for w in words if w]
    cluster_words[cid] = ", ".join(words) if words else f"Cluster {cid}"

# Economic names for the main themes (based on first SVD bigram / economic content)
economic_names = {
    165: "credit rating",
    242: "counterparty risk",
    250: "fraud and cyber risk",
    22:  "mortgage lending",
    247: "derivatives",
    35:  "FDIC / Dodd-Frank",
    207: "real estate",
    54:  "tax law",
    179: "regulatory and industry change",
    227: "residential",
    105: "security breaches",
    218: "technology infrastructure",
    344: "regulatory scrutiny",
    147: "disruption",
    114: "dividend / distribution",
    99:  "mortgage-backed securities",
    38:  "regulatory enforcement",
    80:  "prepayment",
    149: "monetary policy",
    259: "Dodd-Frank / liquidation",
    373: "OTTI charges",
    307: "phishing / malware",
    339: "healthcare / misc.",
    221: "financial instruments",
    351: "held-to-maturity",
    287: "natural disasters",
    10:  "capital adequacy",
    383: "labor and staffing",
    283: "sovereign / macro",
    245: "accounting standards",
}

# Use raw cluster words as labels (transparent for thesis)
theme_labels = dict(cluster_words)

df["label"] = df["theme_id"].map(theme_labels).fillna(df["theme_id"].astype(str))



# 2. DEFINE EPISODES (from rolling z-score plot peaks)

# These periods correspond to the three main clusters of elevated
# rolling z-scores in Panel C of Figure 4, plus the pre-SVB buildup.

episodes = [
    {
        "name": "GFC (2008)",
        "quarters": ["2008Q1", "2008Q2", "2008Q3", "2008Q4"],
    },
    {
        "name": "Mid-Decade Elevation (2014Q3–2015)",
        "quarters": ["2014Q3", "2014Q4", "2015Q1", "2015Q2", "2015Q3", "2015Q4"],
    },
    {
        "name": "Rate & Regulatory (2016)",
        "quarters": ["2016Q1", "2016Q2", "2016Q3", "2016Q4"],
    },
    {
        "name": "COVID Low-Rate Buildup (2021)",
        "quarters": ["2021Q1", "2021Q2", "2021Q3", "2021Q4"],
    },
    {
        "name": "Rate Hikes / SVB (2022)",
        "quarters": ["2022Q1", "2022Q2", "2022Q3", "2022Q4"],
    },
]

N_TOP = 5  # top themes per panel



# 3. COMPUTE PER-EPISODE THEME RANKINGS

print("\nComputing per-episode theme rankings...")

episode_data = []
for ep in episodes:
    qs = ep["quarters"]
    subset = df[df["quarter"].isin(qs)]

    # Average ΔR² and average p-value per theme across the episode quarters
    agg = subset.groupby("theme_id").agg(
        avg_dr2=("delta_adj_r2", "mean"),
        avg_pval=("pvalue", "mean"),
        peak_z=("z_score", "max"),
    ).reset_index()

    # Convert to basis points
    agg["avg_dr2_bps"] = agg["avg_dr2"] * 10_000

    # Significance stars based on average p-value
    def stars(p):
        if pd.isna(p):
            return ""
        if p < 0.01:
            return " ***"
        if p < 0.05:
            return " **"
        if p < 0.10:
            return " *"
        return ""

    agg["stars"] = agg["avg_pval"].apply(stars)
    agg["label"] = agg["theme_id"].map(theme_labels)

    # Sort by average ΔR² and take top N
    top = agg.sort_values("avg_dr2_bps", ascending=False).head(N_TOP)

    print(f"\n  {ep['name'].replace(chr(10), ' ')}:")
    for _, row in top.iterrows():
        print(f"    {row['label'][:45]:45s}  ΔR²={row['avg_dr2_bps']:5.1f} bps"
              f"  peak_z={row['peak_z']:5.1f}{row['stars']}")

    episode_data.append({
        "name": ep["name"],
        "quarters": qs,
        "top": top,
    })



# 4. COLOR SCHEME (by theme category)

# Group themes into broad economic categories for color-coding
category_map = {
    # Credit / Lending
    165: "Credit & Lending",    # credit rating
    22:  "Credit & Lending",    # mortgage lending
    207: "Credit & Lending",    # real estate
    227: "Credit & Lending",    # residential
    99:  "Credit & Lending",    # MBS
    351: "Credit & Lending",    # held-to-maturity
    373: "Credit & Lending",    # OTTI
    # Derivatives / Markets
    247: "Derivatives & Markets",  # derivatives
    149: "Derivatives & Markets",  # monetary policy
    147: "Derivatives & Markets",  # disruption
    # Regulatory
    344: "Regulatory",          # regulatory scrutiny
    35:  "Regulatory",          # FDIC/Dodd-Frank
    38:  "Regulatory",          # regulatory enforcement
    259: "Regulatory",          # Dodd-Frank/liquidation
    179: "Regulatory",          # reg/industry change
    # Operational / Cyber
    250: "Operational & Cyber",  # fraud/cyber
    105: "Operational & Cyber",  # security breaches
    307: "Operational & Cyber",  # phishing
    218: "Operational & Cyber",  # tech infrastructure
    # Macro / Other
    242: "Macro & Counterparty", # counterparty
    114: "Macro & Counterparty", # dividend
    54:  "Tax & Accounting",     # tax law
    221: "Tax & Accounting",     # financial instruments
    80:  "Other",                # prepayment
    287: "Other",                # natural disasters
    339: "Other",                # healthcare/misc
}

category_colors = {
    "Credit & Lending":      "#E8A838",  # gold
    "Derivatives & Markets": "#6FB05C",  # green
    "Regulatory":            "#E86850",  # red
    "Operational & Cyber":   "#9B59B6",  # purple
    "Macro & Counterparty":  "#3498DB",  # blue
    "Tax & Accounting":      "#F39C12",  # orange
    "Other":                 "#95A5A6",  # gray
}

def get_color(tid):
    cat = category_map.get(tid, "Other")
    return category_colors.get(cat, "#95A5A6")

def get_category(tid):
    return category_map.get(tid, "Other")



# 5. PLOT THE FIGURE

print("\nCreating figure...")

n_eps = len(episode_data)
# Set global font sizes BEFORE creating figure
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 14,
})

# Layout: 2 columns, 3 rows (6th cell = legend)
fig, axes = plt.subplots(3, 2, figsize=(18, 20))

ax_list = [
    axes[0, 0],  # GFC
    axes[0, 1],  # Oil & Cyber
    axes[1, 0],  # Rate & Regulatory
    axes[1, 1],  # COVID Low-Rate
    axes[2, 0],  # Rate Hikes / SVB
]
# Hide the 6th panel (will put legend there)
axes[2, 1].set_visible(False)

for idx, ep in enumerate(episode_data):
    ax = ax_list[idx]
    top = ep["top"].reset_index(drop=True)

    # Bars (horizontal, sorted bottom-to-top so biggest is at top)
    top_sorted = top.sort_values("avg_dr2_bps", ascending=True)
    y_pos = range(len(top_sorted))
    colors = [get_color(tid) for tid in top_sorted["theme_id"]]

    bars = ax.barh(
        y_pos,
        top_sorted["avg_dr2_bps"],
        color=colors,
        edgecolor="none",
        alpha=0.9,
        height=0.50,
    )

    # Labels: "top 3 words ***"
    labels = []
    for _, row in top_sorted.iterrows():
        lbl = row["label"]
        if len(lbl) > 40:
            lbl = lbl[:37] + "..."
        labels.append(f"{lbl}{row['stars']}")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_xlabel("Avg. Δ Adj. R² (× 10⁴)", fontsize=13)
    ax.set_title(ep["name"], fontweight="bold", fontsize=15, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)

# Legend
legend_patches = [
    mpatches.Patch(color=c, label=cat)
    for cat, c in category_colors.items()
]
fig.legend(
    handles=legend_patches,
    loc="center",
    bbox_to_anchor=(0.75, 0.17),
    fontsize=13,
    title="Theme Categories",
    title_fontsize=14,
    frameon=True,
    ncol=2,
)

# Add note about method
fig.text(
    0.02, 0.01,
    "Bars show average marginal Δ adj. R² (× 10⁴) across all quarters in each episode.\n"
    "Labels show top 2 cluster bigrams.  Significance: *** p<0.01, ** p<0.05, * p<0.10 (average p-value across episode).",
    fontsize=14, fontstyle="italic", va="bottom",
)

fig.suptitle(
    "Theme Drivers of Emerging Risk Peaks",
    fontsize=18, fontweight="bold", y=0.98,
)
fig.tight_layout(rect=[0, 0.03, 1, 0.95])

out_path = os.path.join(OUTPUT_DIR, "figure_theme_drivers_by_period.png")
fig.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")
print("Done.")
