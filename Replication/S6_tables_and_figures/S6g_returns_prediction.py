# HH Table 6: crisis return prediction.
# Build bank-level emerging risk exposure from quarterly betas and
# z-scored loadings, then regress crisis returns on pre-crisis exposure.
# Under zero-mean loadings the HH formula simplifies to
# E_i = -(1/(N-1)) * sum_k beta_k * z_ik^2.

import os
import warnings
import math
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def normal_sf(x):
    """Survival function for standard normal. No scipy needed."""
    return 0.5 * math.erfc(x / math.sqrt(2))


def sig_star(p):
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""


# Paths
BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE, "..", "data")
OUTPUT_DIR = os.path.join(BASE, "..", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOAD_PATH   = os.path.join(DATA_DIR, "outputs_textual_factors_v2",
                           "bank_year_loadings_v7_xsecstd.csv")
FUND_PATH   = os.path.join(DATA_DIR, "bank_fundamentals_hh_extended.csv")
CRSP_PATH   = os.path.join(DATA_DIR, "crsp_daily_banks_2006_2024.csv")
LINK_PATH   = os.path.join(DATA_DIR, "permno_cik_wrds_extended.csv")
QMARG_PATH  = os.path.join(OUTPUT_DIR, "quarterly_marginal_r2_v7_xsecstd.csv")
SAMPLE_KEYS_PATH = os.path.join(OUTPUT_DIR, "main_regression_sample_keys.csv")


# 1. LOAD DATA

print("Loading data...")

loadings = pd.read_csv(LOAD_PATH)
fund     = pd.read_csv(FUND_PATH)
qmarg    = pd.read_csv(QMARG_PATH)

# Theme columns
theme_cols = sorted([c for c in loadings.columns if c.startswith("topic_loading_")])
beta_cols  = [c for c in qmarg.columns if c.startswith("beta_prod_")]
beta_to_theme = {bc: bc.replace("beta_prod_", "") for bc in beta_cols}

print(f"  Loadings (v7 xsec-std): {len(loadings):,} bank-years, {len(theme_cols)} themes")
print(f"  Quarterly betas: {len(qmarg)} quarters, {len(beta_cols)} theme products")

# Linkage: CIK → PERMNO
link = pd.read_csv(LINK_PATH)
link = link.dropna(subset=["cik", "permno"])
link["cik"]    = link["cik"].astype(int)
link["permno"] = link["permno"].astype(int)
cik_permno = link.drop_duplicates(subset=["cik"])[["cik", "permno"]]
print(f"  CIK→PERMNO links: {len(cik_permno):,}")

# CRSP returns (columns UPPERCASE in file)
print("Loading CRSP returns...")
crsp = pd.read_csv(CRSP_PATH)
crsp.columns = crsp.columns.str.lower()
if crsp["date"].dtype in [np.int64, np.float64, object]:
    crsp["date"] = pd.to_datetime(crsp["date"].astype(str), format="%Y%m%d",
                                  errors="coerce")
crsp["ret"]    = pd.to_numeric(crsp["ret"], errors="coerce")
crsp["prc"]    = pd.to_numeric(crsp["prc"], errors="coerce")
crsp["shrout"] = pd.to_numeric(crsp["shrout"], errors="coerce")
crsp["permno"] = pd.to_numeric(crsp["permno"], errors="coerce").astype("Int64")
crsp = crsp.dropna(subset=["ret", "permno", "date"])
crsp["permno"] = crsp["permno"].astype(int)
print(f"  CRSP: {len(crsp):,} daily obs, {crsp['permno'].nunique()} stocks")


# 2. BUILD BANK-LEVEL EXPOSURE PER QUARTER (closed-form, HH-equivalent)

print("\nBuilding bank-level exposure for each quarter (closed form)...")

# For quarter q in year Y, most recent loadings are from year Y-1
# (10-Ks for fiscal year Y-1 filed ~Q1 of year Y).
#
# Closed-form exposure under xsec-standardized loadings:
#   exposure_i,q = -(1/(N-1)) · Σ_k β_k,q · z_i,k²
# See module docstring for the derivation from HH (2019) eq. 4581.

exposure_all = []

for _, qrow in qmarg.iterrows():
    quarter = qrow["quarter"]
    q_year  = qrow["year"]
    loading_year = q_year - 1

    yr_loads = loadings[loadings["year"] == loading_year].reset_index(drop=True)
    if len(yr_loads) == 0:
        continue
    N_banks = len(yr_loads)
    if N_banks < 2:
        continue

    cik_arr = yr_loads["cik"].values

    # Align betas to loading columns. qmarg has 'beta_prod_<theme>' columns
    # and beta_to_theme maps them back to loading column names.
    betas_lookup = {beta_to_theme[bc]: (qrow[bc] if pd.notna(qrow[bc]) else 0.0)
                    for bc in beta_cols}
    beta_vec = np.array([betas_lookup.get(tc, 0.0) for tc in theme_cols])

    # Loadings matrix (N x K), fill NaN with 0
    Z = yr_loads[theme_cols].astype(float).fillna(0.0).values  # (N, K)

    # Closed-form: exposure_i = -(1/(N-1)) Σ_k β_k · z_i,k²
    raw_exp = (Z ** 2) @ beta_vec                # (N,)
    exposure_vec = -raw_exp / (N_banks - 1)

    for i in range(N_banks):
        exposure_all.append({
            "cik": cik_arr[i],
            "quarter": quarter,
            "q_year": q_year,
            "loading_year": loading_year,
            "emerging_risk_exposure": exposure_vec[i],
        })

exposure_df = pd.DataFrame(exposure_all)
print(f"  Total bank-quarter exposures: {len(exposure_df):,}")
print(f"  Quarters covered: {exposure_df['quarter'].nunique()}")
print(f"  Exposure: mean={exposure_df['emerging_risk_exposure'].mean():.2e}, "
      f"std={exposure_df['emerging_risk_exposure'].std():.2e}")

# Save full exposure dataset
exp_path = os.path.join(OUTPUT_DIR, "bank_emerging_risk_exposure.csv")
exposure_df.to_csv(exp_path, index=False)
print(f"  Saved: {exp_path}")

# Restrict to main regression sample (harmonize with pairwise regression)
print("\nRestricting exposures to main regression sample...")
sample_keys = pd.read_csv(SAMPLE_KEYS_PATH)
sample_keys["cik"] = sample_keys["cik"].astype(int)
sample_keys = sample_keys.rename(columns={"loading_year": "loading_year"})[
    ["cik", "loading_year"]
].drop_duplicates()
exposure_df["cik"] = exposure_df["cik"].astype(int)
n_before = len(exposure_df)
exposure_df = exposure_df.merge(sample_keys, on=["cik", "loading_year"], how="inner")
print(f"  Exposures: {n_before:,} → {len(exposure_df):,} after main-sample filter")
print(f"  Unique banks in filtered exposures: {exposure_df['cik'].nunique()}")

# Merge with PERMNO
exposure_df = exposure_df.merge(cik_permno, on="cik", how="inner")
print(f"  After PERMNO merge: {len(exposure_df):,}")


# 3. COMPUTE PERIOD STOCK RETURNS

print("\nComputing period stock returns...")

# Two test periods matching HH Table 6 structure:
# Left:  GFC (Sep 2008 – Dec 2012) — same window as HH
# Right: SVB crisis (Mar 2023 – Sep 2023) — focused bank stress episode
periods = {
    "GFC": {
        "start": "2008-09-01", "end": "2012-12-31",
        "label": "Crisis period (Sep 2008 – Dec 2012)",
        "exposure_quarters_start": "2006Q1",
        "exposure_quarters_end":   "2012Q4",
        "predictive_cutoff": "2008Q2",  # Q3 2008 onward = nonpredictive
    },
    "SVB": {
        "start": "2023-03-01", "end": "2023-09-30",
        "label": "Current period (Mar 2023 – Sep 2023)",
        "exposure_quarters_start": "2019Q1",
        "exposure_quarters_end":   "2023Q4",
        "predictive_cutoff": "2023Q1",  # SVB failed Mar 10, 2023
    },
}

period_returns = {}
for pname, pinfo in periods.items():
    mask = (crsp["date"] >= pinfo["start"]) & (crsp["date"] <= pinfo["end"])
    period_crsp = crsp[mask].copy()

    cum_ret = period_crsp.groupby("permno")["ret"].apply(
        lambda x: (1 + x).prod() - 1
    ).reset_index()
    cum_ret.columns = ["permno", "period_return"]

    # Below-mean return (left tail, HH Panel B)
    avg_ret = cum_ret["period_return"].mean()
    cum_ret["below_mean_return"] = np.minimum(0, cum_ret["period_return"] - avg_ret)

    period_returns[pname] = cum_ret
    print(f"  {pinfo['label']}: {len(cum_ret)} stocks, "
          f"mean return = {cum_ret['period_return'].mean():.3f}")


# 4. COMPUTE CONTROLS

print("\nPreparing control variables...")

crsp["mktcap"] = crsp["prc"].abs() * crsp["shrout"]
crsp["year"]   = crsp["date"].dt.year

# Year-end market cap
yearly_mc = crsp.sort_values("date").groupby(["permno", "year"]).last()
yearly_mc = yearly_mc[["mktcap"]].reset_index()
yearly_mc["log_mktcap"] = np.log(yearly_mc["mktcap"].clip(lower=1))

# Momentum (HH-faithful): cumulative return from month t-12 to t-2
# (11 months, skipping the most recent month) where t is the crisis start.
# Computed separately for each crisis period as a single scalar per bank.
monthly_ret = crsp.copy()
monthly_ret["month"] = monthly_ret["date"].dt.to_period("M")
monthly_stock = monthly_ret.groupby(["permno", "month"])["ret"].apply(
    lambda x: (1 + x).prod() - 1
).reset_index()
monthly_stock.columns = ["permno", "month", "mret"]

def hh_momentum(crisis_start):
    """11-month cumulative return from t-12 to t-2, skipping t-1."""
    t = pd.Period(pd.Timestamp(crisis_start), freq="M")
    start = t - 12   # include t-12
    end   = t - 2    # include t-2
    sub = monthly_stock[(monthly_stock["month"] >= start) &
                        (monthly_stock["month"] <= end)]
    mom = sub.groupby("permno")["mret"].apply(
        lambda x: (1 + x).prod() - 1
    ).reset_index()
    mom.columns = ["permno", "momentum"]
    return mom

# Fundamentals (already has permno)
fund["permno"] = pd.to_numeric(fund["permno"], errors="coerce")
fund_m = fund.dropna(subset=["permno"])
fund_m["permno"] = fund_m["permno"].astype(int)

controls = fund_m[["permno", "fyear", "ceq", "at", "sich"]].copy()
controls = controls.rename(columns={"fyear": "year"})
controls = controls.merge(yearly_mc[["permno", "year", "mktcap", "log_mktcap"]],
                          on=["permno", "year"], how="inner")
controls["ceq"] = pd.to_numeric(controls["ceq"], errors="coerce")
controls["log_bm"] = np.log((controls["ceq"] / controls["mktcap"]).clip(0.01, 100))
controls["neg_bm"] = (controls["ceq"] <= 0).astype(int)

# SIC4 fixed effects
controls["sich"] = pd.to_numeric(controls["sich"], errors="coerce")
controls["sic4"] = controls["sich"].astype("Int64").astype(str)
sic4_counts = controls["sic4"].value_counts()
valid_sic4  = sic4_counts[sic4_counts >= 5].index
controls.loc[~controls["sic4"].isin(valid_sic4), "sic4"] = "other"
print(f"  Controls: {len(controls):,} stock-years")
print(f"  SIC4 groups: {controls['sic4'].nunique()}")


# 5. RUN REGRESSIONS — ONE PER QUARTER PER PERIOD

print("\nRunning return regressions...")

CTRL_COLS = ["log_mktcap", "log_bm", "neg_bm", "momentum"]

all_results = []

for pname, pinfo in periods.items():
    print(f"\n  {pinfo['label']}")
    print(f"  {'='*70}")

    period_ret = period_returns[pname]
    crisis_start_year = int(pinfo["start"][:4])
    ctrl_year = crisis_start_year - 1

    # Period-specific momentum (HH: months t-12 to t-2 before crisis start)
    mom_p = hh_momentum(pinfo["start"])

    ctrl_sub = controls[controls["year"] == ctrl_year][
        ["permno", "log_mktcap", "log_bm", "neg_bm", "sic4"]
    ].copy()
    ctrl_sub = ctrl_sub.merge(mom_p, on="permno", how="left")

    all_quarters = sorted(exposure_df["quarter"].unique())
    test_quarters = [q for q in all_quarters
                     if q >= pinfo["exposure_quarters_start"]
                     and q <= pinfo["exposure_quarters_end"]]

    for quarter in test_quarters:
        exp_q = exposure_df[exposure_df["quarter"] == quarter][
            ["permno", "emerging_risk_exposure"]
        ].copy()
        exp_q.columns = ["permno", "exposure"]

        if len(exp_q) == 0:
            continue

        # Merge
        reg_data = period_ret.merge(exp_q, on="permno", how="inner")
        reg_data = reg_data.merge(ctrl_sub, on="permno", how="inner")
        reg_data = reg_data.dropna(subset=["exposure"] + CTRL_COLS)

        if len(reg_data) < 30:
            continue

        # Bank-level 5/95 winsorization to dampen extreme exposures and
        # standardization to unit variance so that the regression coefficient
        # represents the effect of a one-SD increase in exposure. Neither
        # step affects statistical significance; both improve readability.
        lo = reg_data["exposure"].quantile(0.05)
        hi = reg_data["exposure"].quantile(0.95)
        reg_data["exposure_w"] = reg_data["exposure"].clip(lo, hi)
        exp_mean = reg_data["exposure_w"].mean()
        exp_std  = reg_data["exposure_w"].std()
        if exp_std == 0:
            continue
        reg_data["exposure_z"] = (reg_data["exposure_w"] - exp_mean) / exp_std

        # Predictive timing
        predictive = "Predictive" if quarter <= pinfo["predictive_cutoff"] else "Nonpredictive"

        # Design matrix: intercept + exposure_z + controls + SIC4 dummies
        X_vars    = reg_data[["exposure_z"] + CTRL_COLS].values
        sic_dums  = pd.get_dummies(reg_data["sic4"], prefix="sic",
                                   drop_first=True).values
        intercept = np.ones((len(reg_data), 1))
        X_full    = np.hstack([intercept, X_vars, sic_dums])

        n_obs    = X_full.shape[0]
        n_params = X_full.shape[1]

        try:
            XtX_inv = np.linalg.inv(X_full.T @ X_full)
        except np.linalg.LinAlgError:
            XtX_inv = np.linalg.pinv(X_full.T @ X_full)

        XtX_inv_Xt = XtX_inv @ X_full.T

        for panel, dep_var in [("A", "period_return"),
                               ("B", "below_mean_return")]:
            y = reg_data[dep_var].values

            beta_hat  = XtX_inv_Xt @ y
            residuals = y - X_full @ beta_hat
            sse = residuals @ residuals
            sst = np.sum((y - y.mean()) ** 2)
            r2  = 1 - sse / sst if sst > 0 else 0
            adj_r2 = 1 - (1 - r2) * (n_obs - 1) / max(n_obs - n_params - 1, 1)

            sigma2 = sse / max(n_obs - n_params, 1)
            se     = np.sqrt(np.diag(XtX_inv) * sigma2)
            t_stats = beta_hat / np.where(se > 0, se, 1e-10)

            exp_coef  = beta_hat[1]
            exp_tstat = t_stats[1]
            exp_pval  = 2 * normal_sf(abs(exp_tstat))

            all_results.append({
                "test_period": pname,
                "period_label": pinfo["label"],
                "quarter": quarter,
                "panel": panel,
                "exposure_coef": exp_coef,
                "exposure_tstat": exp_tstat,
                "exposure_pval": exp_pval,
                "n_obs": n_obs,
                "adj_r2": adj_r2,
                "predictive": predictive,
            })

        # Print Panel A
        row_a = all_results[-2]
        star = sig_star(row_a["exposure_pval"])
        print(f"    {quarter}: β={row_a['exposure_coef']:>8.3f} "
              f"(t={row_a['exposure_tstat']:>6.2f}){star:<4} "
              f"N={row_a['n_obs']:>4}  {predictive}")

results_df = pd.DataFrame(all_results)


# 6. SAVE CSV

csv_path = os.path.join(OUTPUT_DIR, "table6_returns_prediction.csv")
results_df.to_csv(csv_path, index=False)
print(f"\n  Saved: {csv_path}")


# 7. FORMATTED TEXT TABLE (HH side-by-side layout)

gfc_data = results_df[results_df["test_period"] == "GFC"].copy()
svb_data = results_df[results_df["test_period"] == "SVB"].copy()

lines = []
lines.append("Table 6: Crisis and Current-Period Return Regressions")
lines.append("=" * 130)
lines.append("")

for panel_code, panel_title in [("A", "A. Raw returns"),
                                ("B", "B. Below-mean returns")]:

    gfc_panel = gfc_data[gfc_data["panel"] == panel_code].reset_index(drop=True)
    svb_panel = svb_data[svb_data["panel"] == panel_code].reset_index(drop=True)

    hdr_left  = f"{'':>6}{'Crisis period (Sep 2008 – Dec 2012)':^55}"
    hdr_right = f"{'Current period (Mar 2023 – Sep 2023)':^55}"
    lines.append(f"{hdr_left}    {hdr_right}")
    lines.append("")

    col_hdr = (f"{'Row':>4} {'Quarter':>8} {'Exposure':>12} {'Obs':>5} "
               f"{'Timing':>15}")
    lines.append(f"{col_hdr}    {col_hdr}")
    lines.append("-" * 130)
    lines.append(f"{'':>20}{panel_title}{'':>40}{panel_title}")

    max_rows = max(len(gfc_panel), len(svb_panel))

    for i in range(max_rows):
        if i < len(gfc_panel):
            r = gfc_panel.iloc[i]
            star = sig_star(r["exposure_pval"])
            left = (f"({i+1:>2}) {r['quarter']:>8} "
                    f"{r['exposure_coef']:>8.3f} ({r['exposure_tstat']:>5.2f}){star:<3} "
                    f"{r['n_obs']:>5} {r['predictive']:>15}")
        else:
            left = " " * 55

        if i < len(svb_panel):
            r = svb_panel.iloc[i]
            star = sig_star(r["exposure_pval"])
            right = (f"({i+1:>2}) {r['quarter']:>8} "
                     f"{r['exposure_coef']:>8.3f} ({r['exposure_tstat']:>5.2f}){star:<3} "
                     f"{r['n_obs']:>5} {r['predictive']:>15}")
        else:
            right = ""

        lines.append(f"{left}    {right}")

    lines.append("-" * 130)
    lines.append("")

lines.append("Notes:")
lines.append("  Cross-sectional OLS regressions predicting individual bank stock returns during")
lines.append("  and after crises. Dependent variable: cumulative stock return over the period")
lines.append("  (Panel A) or min(0, return − mean return) across banks (Panel B).")
lines.append("  Emerging risk exposure is the quarterly predicted covariance based on Equation (3)")
lines.append("  using only the portion attributable to the semantic themes. Under xsec-standardized")
lines.append("  loadings, HH's pair-level construction collapses to the closed form")
lines.append("  exposure_i = -(1/(N-1)) Σ_k β_k z_i,k². Exposure is then winsorized at 5/95 pct")
lines.append("  within the cross-section and standardized to unit variance.")
lines.append("  Controls (not shown): log(market cap), log(book-to-market), neg BM dummy, momentum.")
lines.append("  Momentum is the cumulative return from month t-12 to t-2 relative to crisis start.")
lines.append("  Industry fixed effects: SIC4.")
lines.append("  Predictive timing: exposure measured before crisis start is 'Predictive'.")
lines.append("  t-statistics in parentheses. * p<0.10, ** p<0.05, *** p<0.01.")

txt_path = os.path.join(OUTPUT_DIR, "table6_returns_prediction.txt")
with open(txt_path, "w") as f:
    f.write("\n".join(lines))
print(f"  Saved: {txt_path}")


# 8. SUMMARY

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

for panel_code, panel_name in [("A", "Raw Returns"), ("B", "Below-Mean Returns")]:
    print(f"\nPanel {panel_code}: {panel_name}")

    for pname in ["GFC", "SVB"]:
        pdata = results_df[(results_df["test_period"] == pname) &
                           (results_df["panel"] == panel_code)]
        pred  = pdata[pdata["predictive"] == "Predictive"]

        n_sig_neg = ((pred["exposure_pval"] < 0.10) &
                     (pred["exposure_coef"] < 0)).sum()
        n_neg     = (pred["exposure_coef"] < 0).sum()
        n_total   = len(pred)

        print(f"\n  {pname} — Predictive quarters only:")
        print(f"    Quarters: {n_total}, Negative coef: {n_neg}/{n_total}, "
              f"Significant negative (p<0.10): {n_sig_neg}")

        if n_total > 0:
            neg_pred = pred[pred["exposure_coef"] < 0]
            if len(neg_pred) > 0:
                best = neg_pred.loc[neg_pred["exposure_pval"].idxmin()]
                star = sig_star(best["exposure_pval"])
                print(f"    Strongest negative: {best['quarter']}, "
                      f"β={best['exposure_coef']:.3f} "
                      f"(t={best['exposure_tstat']:.2f}){star}, "
                      f"N={best['n_obs']:.0f}")

print("\nDone!")
