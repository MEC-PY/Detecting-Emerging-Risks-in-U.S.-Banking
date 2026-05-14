# HH Figure 4 + Table 3: VIX, EPU, covariance measures and
# model R-squared z-scores. Correlation matrix of quarterly series.

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ssl

warnings.filterwarnings("ignore")
ssl._create_default_https_context = ssl._create_unverified_context

# Paths
BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE, "..", "data")
OUTPUT_DIR = os.path.join(BASE, "..", "output")
CRSP_PATH  = os.path.join(DATA_DIR, "crsp_daily_banks_2006_2024.csv")
QMARG_PATH = os.path.join(OUTPUT_DIR, "quarterly_marginal_r2_v7_xsecstd.csv")
MKT_PATH   = os.path.join(OUTPUT_DIR, "quarterly_market_measures.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# 1. LOAD OUR R² TIME SERIES

print("Loading quarterly R² data...")
qmarg = pd.read_csv(QMARG_PATH)
qmarg["date"] = pd.to_datetime(qmarg["quarter"].str.replace(
    r"(\d{4})Q(\d)", lambda m: f"{m.group(1)}-{int(m.group(2))*3:02d}-28", regex=True
))
print(f"  {len(qmarg)} quarters loaded ({qmarg['quarter'].iloc[0]} to {qmarg['quarter'].iloc[-1]})")


# 2. LOAD MARKET MEASURES (pre-computed by 18a, or compute here)

our_quarters = set(qmarg["quarter"])

if os.path.exists(MKT_PATH):
    print("\nLoading pre-computed market measures from 18a...")
    mkt = pd.read_csv(MKT_PATH)
    cov_df = mkt[["quarter", "avg_cov"]].dropna()
    vol_df = mkt[["quarter", "xsec_sd"]].dropna()
    print(f"  {len(cov_df)} quarters covariance, {len(vol_df)} quarters SD returns")
else:
    print("\nComputing market measures from CRSP (18a output not found)...")
    crsp = pd.read_csv(CRSP_PATH)
    crsp.columns = [c.lower() for c in crsp.columns]
    crsp["date"] = pd.to_datetime(crsp["date"].astype(str), format="%Y%m%d", errors="coerce")
    crsp = crsp.dropna(subset=["date"])
    crsp["ret"] = pd.to_numeric(crsp["ret"], errors="coerce")
    crsp = crsp.dropna(subset=["ret", "permno"])
    crsp["permno"] = crsp["permno"].astype(int)
    crsp["quarter"] = crsp["date"].dt.to_period("Q").astype(str)

    # Avg pairwise covariance (daily returns per quarter)
    cov_list = []
    for q in sorted(crsp["quarter"].unique()):
        if q not in our_quarters:
            continue
        qdata = crsp[crsp["quarter"] == q]
        piv = qdata.pivot_table(index="date", columns="permno", values="ret")
        if len(piv) < 10 or piv.shape[1] < 5:
            continue
        piv = piv.dropna(axis=1, thresh=int(len(piv) * 0.5))
        if piv.shape[1] < 5:
            continue
        cov_mat = piv.cov()
        mask = ~np.eye(len(cov_mat), dtype=bool)
        avg_cov = np.nanmean(cov_mat.values[mask])
        if not np.isnan(avg_cov):
            cov_list.append({"quarter": q, "avg_cov": avg_cov})
    cov_df = pd.DataFrame(cov_list)

    # Cross-sectional SD of quarterly returns
    qret = crsp.groupby(["permno", "quarter"])["ret"].apply(
        lambda x: ((1 + x).prod() - 1)).reset_index()
    qret.columns = ["permno", "quarter", "qret"]
    vol_df = qret.groupby("quarter")["qret"].std().reset_index()
    vol_df.columns = ["quarter", "xsec_sd"]
    vol_df = vol_df[vol_df["quarter"].isin(our_quarters)]
    print(f"  {len(cov_df)} quarters covariance, {len(vol_df)} quarters SD returns")


# 3. VIX (CBOE Volatility Index)

# Priority: (1) local CSV with quarterly FRED data, (2) live FRED download,
# (3) CRSP-based proxy as last resort.
VIX_LOCAL = os.path.join(BASE, "vix_quarterly_fred.csv")
print("\nLoading VIX...")
vix_df = None

# --- Option 1: local pre-downloaded FRED data ---
if os.path.exists(VIX_LOCAL):
    vix_df = pd.read_csv(VIX_LOCAL)
    print(f"  VIX loaded from local file: {len(vix_df)} quarters (FRED VIXCLS)")

# --- Option 2: try FRED CSV API ---
if vix_df is None:
    try:
        fred_url = ("https://fred.stlouisfed.org/graph/fredgraph.csv"
                    "?id=VIXCLS&cosd=2006-01-01&coed=2024-12-31")
        vix_raw = pd.read_csv(fred_url, parse_dates=["DATE"])
        vix_raw = vix_raw.rename(columns={"DATE": "date", "VIXCLS": "vix"})
        vix_raw["vix"] = pd.to_numeric(vix_raw["vix"], errors="coerce")
        vix_raw = vix_raw.dropna(subset=["vix"])
        vix_raw["quarter"] = vix_raw["date"].dt.to_period("Q").astype(str)
        vix_df = vix_raw.groupby("quarter")["vix"].mean().reset_index()
        # Save locally for future runs
        vix_df.to_csv(VIX_LOCAL, index=False)
        print(f"  VIX loaded from FRED: {len(vix_df)} quarters (saved locally)")
    except Exception as e:
        print(f"  FRED download failed: {e}")

# --- Option 3: CRSP proxy as last resort ---
if vix_df is None:
    print("  Computing VIX proxy from CRSP daily volatility...")
    crsp_tmp = pd.read_csv(CRSP_PATH)
    crsp_tmp.columns = [c.lower() for c in crsp_tmp.columns]
    crsp_tmp["date"] = pd.to_datetime(crsp_tmp["date"].astype(str),
                                       format="%Y%m%d", errors="coerce")
    crsp_tmp["ret"] = pd.to_numeric(crsp_tmp["ret"], errors="coerce")
    crsp_tmp = crsp_tmp.dropna(subset=["ret", "date"])
    crsp_tmp["quarter"] = crsp_tmp["date"].dt.to_period("Q").astype(str)
    daily_vol = crsp_tmp.groupby("quarter")["ret"].std().reset_index()
    daily_vol.columns = ["quarter", "daily_std"]
    daily_vol["vix"] = daily_vol["daily_std"] * np.sqrt(252) * 100
    vix_df = daily_vol[["quarter", "vix"]]
    print(f"  VIX proxy computed: {len(vix_df)} quarters")


# 4. EPU (Baker, Bloom & Davis 2016)

print("\nLoading EPU...")
epu_df = None
EPU_LOCAL = os.path.join(BASE, "epu_quarterly_local.csv")

# --- Option 1: local pre-downloaded data ---
if os.path.exists(EPU_LOCAL):
    epu_raw = pd.read_csv(EPU_LOCAL)
    epu_cols = [c for c in epu_raw.columns if "ncert" in c.lower() or "index" in c.lower()]
    if epu_cols:
        epu_raw["date"] = pd.to_datetime(
            epu_raw["Year"].astype(str) + "-" +
            epu_raw["Month"].astype(str).str.zfill(2) + "-15")
        epu_raw["quarter"] = epu_raw["date"].dt.to_period("Q").astype(str)
        epu_df = epu_raw.groupby("quarter")[epu_cols[0]].mean().reset_index()
        epu_df.columns = ["quarter", "epu"]
        print(f"  EPU loaded from local file: {len(epu_df)} quarters")

# --- Option 2: try downloading from policyuncertainty.com ---
if epu_df is None:
    try:
        epu_url = "https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.csv"
        epu_raw = pd.read_csv(epu_url)
        epu_cols = [c for c in epu_raw.columns if "ncert" in c.lower() or "index" in c.lower()]
        if epu_cols:
            epu_raw["date"] = pd.to_datetime(
                epu_raw["Year"].astype(str) + "-" +
                epu_raw["Month"].astype(str).str.zfill(2) + "-15")
            epu_raw["quarter"] = epu_raw["date"].dt.to_period("Q").astype(str)
            epu_df = epu_raw.groupby("quarter")[epu_cols[0]].mean().reset_index()
            epu_df.columns = ["quarter", "epu"]
            print(f"  EPU loaded from web: {len(epu_df)} quarters")
    except Exception as e:
        print(f"  EPU download failed: {e}")


# 5. MERGE ALL TIME SERIES

print("\nMerging all time series...")
ts = qmarg[["quarter", "date", "adj_r2_ctrl", "adj_r2_full", "delta_adj_r2"]].copy()
ts = ts.merge(cov_df, on="quarter", how="left")
ts = ts.merge(vol_df, on="quarter", how="left")
if vix_df is not None:
    ts = ts.merge(vix_df, on="quarter", how="left")
if epu_df is not None:
    ts = ts.merge(epu_df, on="quarter", how="left")
print(f"  Merged: {len(ts)} quarters, columns: {list(ts.columns)}")



# 6. LAGGED ROLLING Z-SCORES (matching pipeline approach)

# We use a LAGGED rolling baseline: for each quarter in year Y, the
# baseline is years (Y-4) to (Y-2). This means current spikes don't
# inflate their own baseline, which is crucial for detecting emerging
# risks. This also handles the secular trend in disclosures.
print("\nComputing lagged rolling z-scores (baseline: years t-4 to t-2)")

def lagged_rolling_zscore(df, col):
    """Z-score using lagged 3-year window (years t-4 to t-2), matching pipeline."""
    z_vals = []
    for _, row in df.iterrows():
        yr = int(row["quarter"][:4])
        baseline = df[df["quarter"].str[:4].astype(int).between(yr - 4, yr - 2)]
        if len(baseline) >= 4:
            mu = baseline[col].mean()
            sig = baseline[col].std()
            z = (row[col] - mu) / sig if sig > 0 else 0
        else:
            z = np.nan
        z_vals.append(z)
    return pd.Series(z_vals, index=df.index)



# 7. FIGURE 4 PANEL A: RISK METRICS (raw levels, bar charts)

print("\nCreating Figure 4 Panel A: Risk metrics (raw levels)...")

panel_a_vars = []
if "vix" in ts.columns and ts["vix"].notna().sum() > 5:
    panel_a_vars.append(("vix", "VIX level"))
panel_a_vars.append(("avg_cov", "Average covariance"))
panel_a_vars.append(("xsec_sd", "SD returns (financials)"))
if "epu" in ts.columns and ts["epu"].notna().sum() > 5:
    panel_a_vars.append(("epu", "EPU USA"))

n_a = len(panel_a_vars)
fig_a, axes_a = plt.subplots(n_a, 1, figsize=(14, 2.8 * n_a), sharex=False)
if n_a == 1:
    axes_a = [axes_a]

bar_color = "#4878A8"  # consistent blue like HH

for i, (col, title) in enumerate(panel_a_vars):
    ax = axes_a[i]
    valid = ts.dropna(subset=[col])

    # Scale covariance for readability (×10,000)
    if col == "avg_cov":
        vals = valid[col] * 10000
        title = "Average covariance (×10⁴)"
    else:
        vals = valid[col]

    ax.bar(valid["date"], vals, width=75, color=bar_color, alpha=0.8,
           edgecolor="none")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_ylabel("")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

fig_a.suptitle("(A) Risk metrics", fontsize=13, fontweight="bold",
               x=0.08, ha="left", y=0.98)
fig_a.tight_layout(rect=[0, 0, 1, 0.96])

fig_a_path = os.path.join(OUTPUT_DIR, "figure4a_risk_metrics.png")
fig_a.savefig(fig_a_path, dpi=200, bbox_inches="tight")
plt.close(fig_a)
print(f"  Saved: {fig_a_path}")


# 8. FIGURE 4 PANEL B: COVARIANCE MODELS (z-scores, bar charts)

print("\nCreating Figure 4 Panel B: Covariance models...")

# Pre-computed z-score from pipeline (lagged rolling, years t-4 to t-2)
ts["z_delta_adj_r2"] = qmarg["z_score_rolling"].values

# Panel B: 3 subplots showing genuinely different things
# 1. Controls-only R² (raw) — how much do accounting vars explain?
# 2. Marginal text R² (raw) — actual magnitude of text contribution
# 3. Marginal text R² (z-score) — when is text contribution unusually high?
panel_b_specs = [
    {"col": "adj_r2_ctrl", "title": "Controls-only adj. R²",
     "ylabel": "adj. R²", "use_zscore": False},
    {"col": "delta_adj_r2", "title": "Marginal text adj. R² (Δ adj. R²)",
     "ylabel": "Δ adj. R²", "use_zscore": False},
    {"col": "adj_r2_full", "title": "Full model adj. R² (controls + text)",
     "ylabel": "adj. R²", "use_zscore": False},
    {"col": "z_delta_adj_r2", "title": "Emerging risk index (rolling-z Δ adj. R²)",
     "ylabel": "z-score", "use_zscore": True},
]

n_b = len(panel_b_specs)
fig_b, axes_b = plt.subplots(n_b, 1, figsize=(14, 2.8 * n_b), sharex=False)

delta_color = "#C0392B"  # red for the delta overlay on the full-model row

for i, spec in enumerate(panel_b_specs):
    ax = axes_b[i]
    col = spec["col"]

    if col == "adj_r2_full":
        # Stacked: controls-only (blue) + delta (red) = full model
        valid_mask = ts["adj_r2_ctrl"].notna() & ts["delta_adj_r2"].notna()
        dates = ts.loc[valid_mask, "date"]
        ctrl_vals = ts.loc[valid_mask, "adj_r2_ctrl"]
        delta_vals = ts.loc[valid_mask, "delta_adj_r2"]
        ax.bar(dates, ctrl_vals, width=75, color=bar_color, alpha=0.85,
               edgecolor="none", label="Controls-only adj. R²")
        ax.bar(dates, delta_vals, width=75, bottom=ctrl_vals, color=delta_color,
               alpha=0.95, edgecolor="none", label="Text Δ adj. R²")
        ax.legend(loc="upper left", fontsize=8, frameon=False)
    else:
        valid_mask = ts[col].notna()
        dates = ts.loc[valid_mask, "date"]
        vals = ts.loc[valid_mask, col]
        colors_bar = [bar_color if v >= 0 else "#D88888" for v in vals]
        ax.bar(dates, vals, width=75, color=colors_bar, alpha=0.8, edgecolor="none")
        if spec["use_zscore"]:
            ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)

    ax.set_title(spec["title"], fontsize=11, fontweight="bold")
    ax.set_ylabel(spec["ylabel"])
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

fig_b.suptitle("(B) Covariance models", fontsize=13, fontweight="bold",
               x=0.08, ha="left", y=0.98)
fig_b.tight_layout(rect=[0, 0, 1, 0.96])

fig_b_path = os.path.join(OUTPUT_DIR, "figure4b_covariance_models.png")
fig_b.savefig(fig_b_path, dpi=200, bbox_inches="tight")
plt.close(fig_b)
print(f"  Saved: {fig_b_path}")


# 9. COMBINED FIGURE (both panels on one page, like HH)

print("\nCreating combined Figure 4...")
n_total = n_a + n_b
fig_c, axes_c = plt.subplots(n_total, 1, figsize=(14, 2.5 * n_total), sharex=True)

# Panel A: raw levels
for i, (col, title) in enumerate(panel_a_vars):
    ax = axes_c[i]
    valid = ts.dropna(subset=[col])
    if col == "avg_cov":
        vals = valid[col] * 10000
        title = "Average covariance (×10⁴)"
    else:
        vals = valid[col]
    ax.bar(valid["date"], vals, width=75, color=bar_color, alpha=0.8, edgecolor="none")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if i == 0:
        ax.annotate("(A) Risk metrics", xy=(0.01, 1.15), xycoords="axes fraction",
                     fontsize=12, fontweight="bold")

# Panel B: covariance models (already computed above)
for j, spec in enumerate(panel_b_specs):
    ax = axes_c[n_a + j]
    col = spec["col"]
    valid_mask = ts[col].notna()
    vals = ts.loc[valid_mask, col]
    colors_bar = [bar_color if v >= 0 else "#D88888" for v in vals]
    ax.bar(ts.loc[valid_mask, "date"], vals,
           width=75, color=colors_bar, alpha=0.8, edgecolor="none")
    if spec["use_zscore"]:
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_title(spec["title"], fontsize=10, fontweight="bold")
    ax.set_ylabel(spec["ylabel"], fontsize=8)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if j == 0:
        ax.annotate("(B) Covariance models", xy=(0.01, 1.15), xycoords="axes fraction",
                     fontsize=12, fontweight="bold")

axes_c[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes_c[-1].xaxis.set_major_locator(mdates.YearLocator(2))

fig_c.suptitle("Figure 4: Emerging risks comparison",
               fontsize=14, fontweight="bold", y=1.01)
fig_c.tight_layout()
fig_c_path = os.path.join(OUTPUT_DIR, "figure4_combined.png")
fig_c.savefig(fig_c_path, dpi=200, bbox_inches="tight")
plt.close(fig_c)
print(f"  Saved: {fig_c_path}")


# 10. TABLE 3: CORRELATION MATRIX

print("\nComputing Table 3: Correlation matrix...")

# HH Table 3 uses levels (not z-scores)
corr_vars = {}
if "vix" in ts.columns:
    corr_vars["VIX"] = ts["vix"]
corr_vars["Avg. pairwise cov."] = ts.get("avg_cov", pd.Series(dtype=float))
corr_vars["SD returns (fin.)"] = ts.get("xsec_sd", pd.Series(dtype=float))
if "epu" in ts.columns:
    corr_vars["EPU"] = ts["epu"]
corr_vars["Acct. var. adj. R²"] = ts["adj_r2_ctrl"]
corr_vars["Text var. adj. R²"] = ts["delta_adj_r2"]
corr_vars["Full model adj. R²"] = ts["adj_r2_full"]
# Robustness: rolling-baseline z-score of text ΔR² (our actual emerging-risk index)
corr_vars["Text ΔR² (rolling z)"] = ts["z_delta_adj_r2"]

corr_df = pd.DataFrame(corr_vars)
corr_df = corr_df.dropna(how="all")
corr_matrix = corr_df.corr()

print("\n  Pearson Correlation Matrix:")
print(corr_matrix.round(3).to_string())

# Save CSV
corr_csv_path = os.path.join(OUTPUT_DIR, "table3_correlation_matrix.csv")
corr_matrix.to_csv(corr_csv_path)

# Save formatted text
n_q = corr_df.dropna().shape[0]
txt_lines = [
    "Table 3: Pearson Correlation Coefficients",
    "=" * 80,
    f"Sample: {ts['quarter'].iloc[0]} to {ts['quarter'].iloc[-1]} ({n_q} quarters)",
    "",
    corr_matrix.round(3).to_string(),
    "",
    "Notes: VIX is the quarterly average of the CBOE Volatility Index (FRED VIXCLS).",
    "Avg. pairwise cov. is the quarterly average pairwise covariance of daily bank",
    "stock returns from CRSP. SD returns (fin.) is the cross-sectional standard",
    "deviation of compounded quarterly returns for financial firms (SIC 6000-6199).",
    "EPU is the Economic Policy Uncertainty index from Baker, Bloom, and Davis (2016).",
    "Acct. var. adj. R² is the adjusted R² from regressing bank-pair covariance on",
    "accounting characteristics and SIC industry dummies only. Text var. adj. R² is",
    f"the marginal improvement when {len([c for c in qmarg.columns if c.startswith('beta_')])} "
    "semantic theme products are added to the covariance regression.",
    "Text ΔR² (rolling z) is the same quantity z-scored against a 3-year lagged",
    "baseline (years t-4 to t-2), i.e. the emerging risk index from Figure 6.",
]
txt_path = os.path.join(OUTPUT_DIR, "table3_correlation_matrix.txt")
with open(txt_path, "w") as f:
    f.write("\n".join(txt_lines))

print(f"\n  Saved: {corr_csv_path}")
print(f"  Saved: {txt_path}")
print("\nDone!")
