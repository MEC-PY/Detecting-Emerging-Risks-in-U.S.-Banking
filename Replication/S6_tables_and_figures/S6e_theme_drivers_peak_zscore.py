# Same as S6d but ranks themes by peak z-score instead of average.
# Captures single most elevated quarter per theme per episode.

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
    print("  ERROR: z-score CSV not found — cannot rank by z-score!")
    raise FileNotFoundError(ZSCORE_CSV)

# Theme labels: top 2 cluster bigrams
themes_meta = pd.read_csv(THEMES_CSV)
cluster_words = {}
for _, r in themes_meta.iterrows():
    cid = int(r["cluster_id"])
    tw = str(r.get("top_words", ""))
    words = [w.strip().replace("_", " ") for w in tw.split(",")][:2]
    words = [w for w in words if w]
    cluster_words[cid] = ", ".join(words) if words else f"Cluster {cid}"

theme_labels = dict(cluster_words)
df["label"] = df["theme_id"].map(theme_labels).fillna(df["theme_id"].astype(str))



# 2. DEFINE EPISODES

episodes = [
    {
        "name": "Global Financial Crisis (2008)",  # GFC
        "quarters": ["2008Q1", "2008Q2", "2008Q3", "2008Q4"],
    },
    {
        "name": "Mid-Decade Elevation (2014Q3–2015)",
        "quarters": ["2014Q3", "2014Q4", "2015Q1", "2015Q2", "2015Q3", "2015Q4"],
    },
    {
        "name": "Mid-Decade Elevation (2016)",
        "quarters": ["2016Q1", "2016Q2", "2016Q3", "2016Q4"],
    },
    {
        "name": "Rate-Hiking Buildup (2021)",
        "quarters": ["2021Q1", "2021Q2", "2021Q3", "2021Q4"],
    },
    {
        "name": "Rate-Hiking Buildup (2022)",
        "quarters": ["2022Q1", "2022Q2", "2022Q3", "2022Q4"],
    },
]

MIN_PEAK_Z = 10   # minimum peak z-score to qualify as "emerging"
SIG_LEVEL  = 0.05  # significance threshold (standard 5% level)



# 3. COMPUTE PER-EPISODE THEME RANKINGS (by peak z-score)

print(f"\nComputing per-episode theme rankings (peak z ≥ {MIN_PEAK_Z}, p < {SIG_LEVEL})...")

episode_data = []
for ep in episodes:
    qs = ep["quarters"]
    subset = df[df["quarter"].isin(qs)]

    agg = subset.groupby("theme_id").agg(
        avg_zscore=("z_score", "mean"),
        peak_z=("z_score", "max"),
        avg_dr2=("delta_adj_r2", "mean"),
        avg_pval=("pvalue", "mean"),
        min_pval=("pvalue", "min"),   # best single-quarter p-value
    ).reset_index()

    # Convert ΔR² to basis points for reference
    agg["avg_dr2_bps"] = agg["avg_dr2"] * 10_000

    # Significance stars based on best single-quarter p-value
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

    agg["stars"] = agg["min_pval"].apply(stars)
    agg["label"] = agg["theme_id"].map(theme_labels)

    # Dual filter: significant at 5% level AND peak z ≥ threshold
    agg_sig = agg[(agg["min_pval"] <= SIG_LEVEL) & (agg["peak_z"] >= MIN_PEAK_Z)].copy()

    # Sort by PEAK z-score (show ALL that qualify, no fixed top-N cutoff)
    top = agg_sig.sort_values("peak_z", ascending=False)

    n_total_sig = len(agg[agg["min_pval"] <= SIG_LEVEL])
    print(f"\n  {ep['name']}:  {len(top)} themes qualify (of {n_total_sig} significant at {SIG_LEVEL})")
    for _, row in top.iterrows():
        print(f"    {row['label'][:45]:45s}  peak_z={row['peak_z']:6.1f}"
              f"  avg_z={row['avg_zscore']:6.1f}"
              f"  ΔR²={row['avg_dr2_bps']:5.1f} bps{row['stars']}")

    episode_data.append({
        "name": ep["name"],
        "quarters": qs,
        "top": top,
    })



# 4. COLOR SCHEME (same as ΔR² version)

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
    383: "Other",                # labor
    10:  "Other",                # capital adequacy
    283: "Other",                # sovereign/macro
    245: "Tax & Accounting",     # accounting standards
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



# 5. PLOT THE FIGURE

print("\nCreating figure...")

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 14,
})

fig, axes = plt.subplots(3, 2, figsize=(18, 22))

ax_list = [
    axes[0, 0],  # GFC
    axes[0, 1],  # Oil & Cyber
    axes[1, 0],  # Rate & Regulatory
    axes[1, 1],  # COVID Low-Rate
    axes[2, 0],  # Rate Hikes / SVB
]
axes[2, 1].set_visible(False)

for idx, ep in enumerate(episode_data):
    ax = ax_list[idx]
    top = ep["top"].reset_index(drop=True)

    # Bars sorted bottom-to-top so biggest is at top
    top_sorted = top.sort_values("peak_z", ascending=True)
    y_pos = range(len(top_sorted))
    bar_height = 0.50 if len(top_sorted) <= 7 else 0.40
    bars = ax.barh(
        y_pos,
        top_sorted["peak_z"],
        color="#4878A8",
        edgecolor="none",
        alpha=0.85,
        height=bar_height,
    )

    # Labels
    labels = []
    for _, row in top_sorted.iterrows():
        lbl = row["label"]
        if len(lbl) > 40:
            lbl = lbl[:37] + "..."
        labels.append(f"{lbl}{row['stars']}")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_xlabel("Peak Rolling Z-Score", fontsize=13)
    ax.set_title(ep["name"], fontweight="bold", fontsize=15, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)

fig.suptitle(
    "Emerging Theme Drivers by Stress Episode (Peak Z-Score Ranking)",
    fontsize=18, fontweight="bold", y=0.98,
)
fig.tight_layout(rect=[0, 0.01, 1, 0.95])

out_path = os.path.join(OUTPUT_DIR, "figure_theme_drivers_by_period_peak_zscore.png")
fig.savefig(out_path, dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out_path}")
print("Done.")
