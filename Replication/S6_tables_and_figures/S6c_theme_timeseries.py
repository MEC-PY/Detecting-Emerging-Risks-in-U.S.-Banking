# Individual theme time series plots (HH Figs 5, 6, A1).
# Per-theme marginal R2 z-scored with rolling baseline.

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# Paths
BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE, "..", "data")
OUTPUT_DIR = os.path.join(BASE, "..", "output")
THEME_R2   = os.path.join(OUTPUT_DIR, "individual_theme_r2_v7_xsecstd.csv")
THEMES_CSV = os.path.join(DATA_DIR, "outputs_textual_factors_v2", "final_themes.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. LOAD DATA

print("Loading individual theme R² data...")
df = pd.read_csv(THEME_R2)
print(f"  {len(df):,} rows: {df['theme_id'].nunique()} themes × {df['quarter'].nunique()} quarters")

# Parse quarter → date
df["date"] = pd.to_datetime(df["quarter"].str.replace(
    r"(\d{4})Q(\d)", lambda m: f"{m.group(1)}-{int(m.group(2))*3:02d}-28", regex=True
))
df["year"] = df["date"].dt.year


# 2. BUILD THEME LABELS (fill nan labels with readable names)

print("\nBuilding theme labels...")
themes = pd.read_csv(THEMES_CSV)

# Use top 5 words from each cluster as label (more transparent for thesis)
theme_labels = {}
for _, r in themes.iterrows():
    cid = int(r["cluster_id"])
    tw = str(r.get("top_words", ""))
    words = [w.strip().replace("_", " ") for w in tw.split(",")][:5]
    words = [w for w in words if w]  # drop empty
    theme_labels[cid] = ", ".join(words) if words else f"Cluster {cid}"

# Apply labels to dataframe
df["label"] = df["theme_id"].map(theme_labels).fillna(df["theme_id"].astype(str))

print(f"  Labeled {len(theme_labels)} themes:")
for tid in sorted(theme_labels.keys()):
    print(f"    {tid:3d}: {theme_labels[tid]}")



# 3. COMPUTE LAGGED ROLLING Z-SCORES PER THEME

# Same methodology as aggregate: for year Y, baseline = years (Y-4) to (Y-2).
# This ensures current spikes don't inflate their own baseline.
print("\nComputing lagged rolling z-scores per theme (baseline: years t-4 to t-2)...")

def lagged_zscore_per_theme(group):
    """Z-score each quarter's delta_adj_r2 against years (Y-4) to (Y-2)."""
    z_vals = []
    for _, row in group.iterrows():
        yr = row["year"]
        baseline = group[group["year"].between(yr - 4, yr - 2)]
        if len(baseline) >= 4:  # need at least ~1 year of baseline data
            mu = baseline["delta_adj_r2"].mean()
            sig = baseline["delta_adj_r2"].std()
            z = (row["delta_adj_r2"] - mu) / sig if sig > 0 else 0
        else:
            z = np.nan
        z_vals.append(z)
    return pd.Series(z_vals, index=group.index)

df = df.sort_values(["theme_id", "quarter"])
theme_ids = sorted(df["theme_id"].unique())
for i, tid in enumerate(theme_ids):
    mask = df["theme_id"] == tid
    df.loc[mask, "z_score"] = lagged_zscore_per_theme(df.loc[mask].copy())
    if (i + 1) % 10 == 0:
        print(f"  ... computed {i+1}/{len(theme_ids)} themes")

print(f"  Done. Z-score NaN rate: {df['z_score'].isna().mean():.1%} (expected for first 3 years)")

# Save z-score time series
zscore_path = os.path.join(OUTPUT_DIR, "individual_theme_zscore_timeseries.csv")
df.to_csv(zscore_path, index=False)
print(f"  Saved: {zscore_path}")



# 4. IDENTIFY TOP THEMES PER CRISIS PERIOD

print("\nIdentifying top themes per crisis period...")

crisis_periods = {
    "GFC (2007-2009)":              (2007, 2009),
    "COVID-19 (2020)":              (2020, 2020),
    "Rate Hikes / SVB (2022-2023)": (2022, 2023),
}

top_themes = {}
for period_name, (y_start, y_end) in crisis_periods.items():
    period_data = df[df["year"].between(y_start, y_end)]
    avg_z = period_data.groupby("theme_id")["z_score"].mean().sort_values(ascending=False)
    top_themes[period_name] = avg_z.head(10).index.tolist()
    print(f"\n  {period_name}:")
    for tid in avg_z.head(7).index:
        label = theme_labels.get(tid, f"Theme {tid}")
        print(f"    {label:35s}  avg z = {avg_z[tid]:6.2f}")



# 5. PLOTTING HELPERS

# Color scheme: significant bars (p < 0.05) get strong color,
# non-significant bars get faint gray. Purely data-driven — no
# hardcoded crisis period shading.
SIG_LEVEL   = 0.05
COLOR_SIG   = "#4878A8"   # strong blue for significant quarters
COLOR_NSIG  = "#C8C8C8"   # light gray for non-significant quarters


def plot_theme_panels(ids_to_plot, title, filename, n_max=7):
    """Create stacked panel figure. Bars colored by statistical significance.
    Blue = significant at 5%, gray = not significant."""
    ids_to_plot = ids_to_plot[:n_max]
    n = len(ids_to_plot)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for i, tid in enumerate(ids_to_plot):
        ax = axes[i]
        tdata = df[df["theme_id"] == tid].sort_values("date")
        valid = tdata.dropna(subset=["z_score"])

        vals = valid["z_score"].clip(lower=-5)
        sig = valid["pvalue"] < SIG_LEVEL

        # Significant = blue, non-significant = gray
        colors = [COLOR_SIG if s else COLOR_NSIG for s in sig]

        ax.bar(valid["date"], vals, width=75, color=colors, alpha=0.85,
               edgecolor="none")
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)

        # Count significant quarters for subtitle
        n_sig = sig.sum()
        n_tot = len(valid)
        label_text = theme_labels.get(tid, f"Theme {tid}")
        ax.set_title(f"{label_text}   ({n_sig}/{n_tot} quarters significant at 5%)",
                     fontsize=10, fontweight="bold", loc="left")
        ax.set_ylabel("z-score", fontsize=8)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        for label in ax.get_xticklabels():
            label.set_rotation(45)
            label.set_ha("right")

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()

    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")



# 6. RANK THEMES BY SIGNIFICANCE AND PEAK Z-SCORE

# Instead of pre-defining crisis periods, rank themes purely by data:
# (1) number of significant quarters, (2) peak z-score as tiebreaker.
print("\nRanking themes by number of significant quarters...")

theme_sig_stats = []
for tid in theme_ids:
    tdata = df[df["theme_id"] == tid].dropna(subset=["z_score"])
    n_sig = (tdata["pvalue"] < SIG_LEVEL).sum()
    n_tot = len(tdata)
    peak_z = tdata["z_score"].max() if len(tdata) > 0 else 0
    avg_z = tdata["z_score"].mean() if len(tdata) > 0 else 0
    theme_sig_stats.append({
        "theme_id": tid,
        "label": theme_labels.get(tid, f"Theme {tid}"),
        "n_sig": n_sig,
        "n_total": n_tot,
        "pct_sig": n_sig / n_tot if n_tot > 0 else 0,
        "peak_z": peak_z,
        "avg_z": avg_z,
    })

sig_df = pd.DataFrame(theme_sig_stats).sort_values(
    ["n_sig", "peak_z"], ascending=[False, False]
)

print("\n  All themes ranked by significant quarters:")
for _, row in sig_df.iterrows():
    print(f"    {row['label'][:50]:50s}  sig: {row['n_sig']:2.0f}/{row['n_total']:.0f} "
          f"({row['pct_sig']:.0%})  peak z = {row['peak_z']:6.1f}")

# Save ranking
sig_rank_path = os.path.join(OUTPUT_DIR, "theme_significance_ranking.csv")
sig_df.to_csv(sig_rank_path, index=False)
print(f"\n  Saved: {sig_rank_path}")



# 7. FIGURE 5: TOP THEMES BY SIGNIFICANCE (most frequently significant)

print("\nCreating Figure 5: Most frequently significant themes...")
top7_by_sig = sig_df.head(7)["theme_id"].tolist()

plot_theme_panels(
    top7_by_sig,
    title="Most Frequently Significant Emerging Risk Themes\n"
          "(Individual Theme Marginal R² Z-Scores, Rolling 3-Year Lagged Baseline)",
    filename="figure5_crisis_themes.png",
)


# 8. FIGURE 6: TOP THEMES BY PEAK Z-SCORE (highest spikes)

print("\nCreating Figure 6: Highest-peak emerging risk themes...")
# Rank by peak z-score (which themes had the most extreme spikes?)
top7_by_peak = sig_df.sort_values("peak_z", ascending=False).head(7)["theme_id"].tolist()

# Remove duplicates with Figure 5 to show different themes
top7_by_peak_unique = [t for t in top7_by_peak if t not in top7_by_sig]
# Fill remaining slots from peak ranking
remaining = [t for t in sig_df.sort_values("peak_z", ascending=False)["theme_id"]
             if t not in top7_by_sig and t not in top7_by_peak_unique]
top7_fig6 = (top7_by_peak_unique + remaining)[:7]

# If there's heavy overlap, just use peak ranking directly
if len(top7_fig6) < 7:
    top7_fig6 = sig_df.sort_values("peak_z", ascending=False).head(7)["theme_id"].tolist()

plot_theme_panels(
    top7_fig6,
    title="Highest-Peak Emerging Risk Themes\n"
          "(Individual Theme Marginal R² Z-Scores, Rolling 3-Year Lagged Baseline)",
    filename="figure6_recent_themes.png",
)


# 9. FIGURE A1: REMAINING THEMES (same layout as Fig 5 & 6)

# All themes NOT already shown in Figure 5 or 6, split across
# multiple pages of 7 panels each (same stacked layout).
print("\nCreating Figure A1 pages: remaining themes (same layout as Fig 5 & 6)...")

shown_in_fig5_fig6 = set(top7_by_sig) | set(top7_fig6)
remaining_ids = [t for t in sig_df["theme_id"].tolist() if t not in shown_in_fig5_fig6]
print(f"  {len(shown_in_fig5_fig6)} themes in Fig 5+6, {len(remaining_ids)} remaining for A1")

THEMES_PER_PAGE = 7
n_pages = int(np.ceil(len(remaining_ids) / THEMES_PER_PAGE))

for page in range(n_pages):
    start = page * THEMES_PER_PAGE
    end = min(start + THEMES_PER_PAGE, len(remaining_ids))
    page_ids = remaining_ids[start:end]
    page_num = page + 1

    plot_theme_panels(
        page_ids,
        title=f"Individual Emerging Risk Themes (continued)\n"
              f"(Rolling 3-Year Lagged Z-Scores — Blue = significant at 5%, Gray = not significant)",
        filename=f"figureA1_{page_num}_themes.png",
        n_max=THEMES_PER_PAGE,
    )



# 10. SUMMARY

print("\n" + "=" * 70)
print("SUMMARY: Theme Significance Statistics")
print("=" * 70)
print(f"\nTotal theme-quarter observations: {len(df)}")
print(f"Significant at 5%: {(df['pvalue'] < SIG_LEVEL).sum()} / {len(df)} "
      f"({(df['pvalue'] < SIG_LEVEL).mean():.1%})")
print(f"Significant at 1%: {(df['pvalue'] < 0.01).sum()} / {len(df)} "
      f"({(df['pvalue'] < 0.01).mean():.1%})")
print(f"\nTop 10 themes by significant quarters:")
for _, row in sig_df.head(10).iterrows():
    print(f"  {row['label'][:50]:50s}  {row['n_sig']:2.0f}/{row['n_total']:.0f} sig "
          f"({row['pct_sig']:.0%})  peak z = {row['peak_z']:.1f}")

print(f"\nDone! All figures saved to: {OUTPUT_DIR}")
