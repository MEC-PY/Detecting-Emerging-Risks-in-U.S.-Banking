# Crisis return prediction using cross-sectionally standardized loadings.
# The same L_tilde used in the pairwise covariance regression enters
# directly as a bank-level regressor. The HH scalar exposure formula
# cannot be applied because demeaning causes it to collapse into squared
# deviations, so we estimate theme-by-theme regressions instead.
# Only tests the 2021/22 -> 2023 regional-bank stress episode.

import os
import warnings
import math
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def normal_sf(x):
    return 0.5 * math.erfc(x / math.sqrt(2))

def sig_star(p):
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return ""

def _betai(a, b, x):
    """Regularized incomplete beta function via continued fraction."""
    if x < 0 or x > 1: return 0.0
    if x == 0 or x == 1: return x
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    f = 1.0; c = 1.0; d = 1 - (a + b) * x / (a + 1)
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d; f = d
    for m in range(1, 200):
        num = m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))
        d = 1 + num * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1 + num / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d; f *= c * d
        num = -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))
        d = 1 + num * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1 + num / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d; delta = c * d; f *= delta
        if abs(delta - 1.0) < 1e-10: break
    return front * f

def f_sf(F, d1, d2):
    if F <= 0: return 1.0
    x = d2 / (d2 + d1 * F)
    return _betai(d2 / 2.0, d1 / 2.0, x)


# Paths
BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE, "..", "data")
OUTPUT_DIR = os.path.join(BASE, "..", "output")

# Cross-sectionally standardized loadings (same as pairwise covariance regression)
STD_LOAD_PATH = os.path.join(DATA_DIR, "outputs_textual_factors_v2",
                              "bank_year_loadings_v7_xsecstd.csv")
FUND_PATH   = os.path.join(DATA_DIR, "bank_fundamentals_hh_extended.csv")
CRSP_PATH   = os.path.join(DATA_DIR, "crsp_daily_banks_2006_2024.csv")
LINK_PATH   = os.path.join(DATA_DIR, "permno_cik_wrds_extended.csv")
SAMPLE_KEYS_PATH = os.path.join(OUTPUT_DIR, "main_regression_sample_keys.csv")

# Themes from Figure 10 (matched to episode year)
# 2021 panel: Rate-Hiking Buildup (2021)
SVB_THEMES_2021 = {
    "topic_loading_22":  "mortgage_lending",
    "topic_loading_307": "phishing_malware",
    "topic_loading_54":  "tax_law",
    "topic_loading_155": "capital_ratio",
    "topic_loading_344": "regulatory_scrutiny",
}
# 2022 panel: Rate-Hiking Buildup (2022)
SVB_THEMES_2022 = {
    "topic_loading_207": "real_estate",
    "topic_loading_155": "capital_ratio",
    "topic_loading_22":  "mortgage_lending",
    "topic_loading_80":  "prepayment",
    "topic_loading_99":  "mortgage_backed_sec",
    "topic_loading_373": "otti",
}


# 1. LOAD DATA

print("Loading data...")

std_loadings = pd.read_csv(STD_LOAD_PATH)
fund = pd.read_csv(FUND_PATH)

link = pd.read_csv(LINK_PATH)
link = link.dropna(subset=["cik", "permno"])
link["cik"]    = link["cik"].astype(int)
link["permno"] = link["permno"].astype(int)
cik_permno = link.drop_duplicates(subset=["cik"])[["cik", "permno"]]

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

sample_keys = pd.read_csv(SAMPLE_KEYS_PATH)
sample_keys["cik"] = sample_keys["cik"].astype(int)
sample_keys = sample_keys[["cik", "loading_year"]].drop_duplicates()

fund = fund.dropna(subset=["permno"])
fund["permno"] = pd.to_numeric(fund["permno"], errors="coerce").astype(int)
fund["ceq"]    = pd.to_numeric(fund["ceq"], errors="coerce")
fund["sich"]   = pd.to_numeric(fund["sich"], errors="coerce")

crsp["mktcap"] = crsp["prc"].abs() * crsp["shrout"]
crsp["year"]   = crsp["date"].dt.year
yearly_mc = crsp.sort_values("date").groupby(["permno", "year"]).last()
yearly_mc = yearly_mc[["mktcap"]].reset_index()
yearly_mc["log_mktcap"] = np.log(yearly_mc["mktcap"].clip(lower=1))

monthly_ret = crsp.copy()
monthly_ret["month"] = monthly_ret["date"].dt.to_period("M")
monthly_stock = monthly_ret.groupby(["permno", "month"])["ret"].apply(
    lambda x: (1 + x).prod() - 1
).reset_index()
monthly_stock.columns = ["permno", "month", "mret"]

def hh_momentum(crisis_start):
    t = pd.Period(pd.Timestamp(crisis_start), freq="M")
    start = t - 12; end = t - 2
    sub = monthly_stock[(monthly_stock["month"] >= start) &
                        (monthly_stock["month"] <= end)]
    mom = sub.groupby("permno")["mret"].apply(
        lambda x: (1 + x).prod() - 1
    ).reset_index()
    mom.columns = ["permno", "momentum"]
    return mom

controls = fund[["permno", "fyear", "ceq", "sich"]].copy()
controls = controls.rename(columns={"fyear": "year"})
controls = controls.merge(yearly_mc[["permno", "year", "mktcap", "log_mktcap"]],
                          on=["permno", "year"], how="inner")
controls["log_bm"] = np.log((controls["ceq"] / controls["mktcap"]).clip(0.01, 100))
controls["neg_bm"] = (controls["ceq"] <= 0).astype(int)
controls["sic4"] = controls["sich"].astype("Int64").astype(str)
sic4_counts = controls["sic4"].value_counts()
valid_sic4 = sic4_counts[sic4_counts >= 5].index
controls.loc[~controls["sic4"].isin(valid_sic4), "sic4"] = "other"


# 2. PREPARE STANDARDIZED LOADINGS

print("\nUsing cross-sectionally standardized loadings (L_tilde)...")
std_loadings["cik"] = std_loadings["cik"].astype(str).str.split(".").str[0].astype(int)
all_theme_cols = sorted([c for c in std_loadings.columns if c.startswith("topic_loading_")])
print(f"  Loadings: {len(std_loadings):,} bank-years, {len(all_theme_cols)} themes")

# Diagnostics for crisis-relevant themes
all_theme_ids = set(list(SVB_THEMES_2021.keys()) + list(SVB_THEMES_2022.keys()))
print(f"\n  Standardized loading diagnostics (crisis-relevant themes):")
for col in sorted(all_theme_ids):
    if col in std_loadings.columns:
        vals = std_loadings[col].values
        label = SVB_THEMES_2021.get(col, SVB_THEMES_2022.get(col, col))
        print(f"    {label:25s}: mean={vals.mean():+.4f}  std={vals.std():.4f}  "
              f"min={vals.min():+.4f}  max={vals.max():+.4f}")


# 3. CRISIS WINDOWS AND REGRESSIONS

def manual_ols(y, X):
    n, k = X.shape
    if n <= k + 5: return None
    try:
        beta = np.linalg.solve(X.T @ X, X.T @ y)
    except np.linalg.LinAlgError: return None
    resid = y - X @ beta
    SSR = resid @ resid
    SST = np.sum((y - y.mean()) ** 2)
    R2 = 1 - SSR / SST if SST > 0 else 0
    adj_R2 = 1 - (SSR / (n - k)) / (SST / (n - 1)) if SST > 0 else 0
    # HC1 heteroskedasticity-robust standard errors
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        meat = (X * resid[:, None]).T @ (X * resid[:, None])
        hc1_factor = n / (n - k)
        var_beta = hc1_factor * XtX_inv @ meat @ XtX_inv
        se = np.sqrt(np.maximum(np.diag(var_beta), 0))
    except: se = np.full(k, np.nan)
    t_stats = np.where(se > 0, beta / se, 0)
    p_values = np.array([2 * normal_sf(abs(t)) for t in t_stats])
    return {"beta": beta, "se": se, "t": t_stats, "p": p_values,
            "R2": R2, "adj_R2": adj_R2, "n": n, "k": k, "SSR": SSR, "SST": SST}

crisis_windows = {
    "SVB_2021_themes": {
        "start": "2023-03-01", "end": "2023-06-30",
        "label": "Regional-bank stress (Mar – Jun 2023), 2021 themes",
        "themes": SVB_THEMES_2021,
        "loading_years": [2021],
        "ctrl_year": 2022,
    },
    "SVB_2022_themes": {
        "start": "2023-03-01", "end": "2023-06-30",
        "label": "Regional-bank stress (Mar – Jun 2023), 2022 themes (proximity check)",
        "themes": SVB_THEMES_2022,
        "loading_years": [2022],
        "ctrl_year": 2022,
    },
}

print("\n" + "=" * 80)
print("CRISIS RETURN REGRESSIONS (cross-sectionally standardized loadings)")
print("=" * 80)

all_results = []

for window_name, winfo in crisis_windows.items():
    print(f"\n{'━' * 75}")
    print(f"  {winfo['label']}")
    print(f"{'━' * 75}")

    themes = winfo["themes"]
    theme_cols = list(themes.keys())
    theme_labels = list(themes.values())
    ctrl_year = winfo["ctrl_year"]
    q_themes = len(theme_cols)

    # Returns
    mask = (crsp["date"] >= winfo["start"]) & (crsp["date"] <= winfo["end"])
    cum_ret = crsp[mask].groupby("permno")["ret"].apply(
        lambda x: (1 + x).prod() - 1
    ).reset_index()
    cum_ret.columns = ["permno", "period_return"]
    avg_ret = cum_ret["period_return"].mean()
    cum_ret["below_mean_return"] = np.minimum(0, cum_ret["period_return"] - avg_ret)
    print(f"  Returns: {len(cum_ret)} stocks, mean={avg_ret:.4f}")

    mom = hh_momentum(winfo["start"])
    ctrl_sub = controls[controls["year"] == ctrl_year][
        ["permno", "log_mktcap", "log_bm", "neg_bm", "sic4"]
    ].drop_duplicates("permno").copy()
    ctrl_sub = ctrl_sub.merge(mom, on="permno", how="left")

    for loading_year in winfo["loading_years"]:
        yr_loads = std_loadings[std_loadings["year"] == loading_year].copy()

        # Filter to main sample
        yr_loads = yr_loads.merge(
            sample_keys[sample_keys["loading_year"] == loading_year][["cik"]],
            on="cik", how="inner"
        )
        yr_loads = yr_loads.merge(cik_permno, on="cik", how="inner")

        df = yr_loads.merge(cum_ret, on="permno", how="inner")
        df = df.merge(ctrl_sub, on="permno", how="inner")
        df = df.dropna(subset=theme_cols + ["period_return", "log_mktcap",
                                             "log_bm", "momentum"])

        if len(df) < 50:
            print(f"\n  Yr={loading_year}: too few obs ({len(df)}), skipping")
            continue

        # SIC dummies
        sic_dums = pd.get_dummies(df["sic4"], prefix="sic", drop_first=True).astype(float)
        sic_arr = sic_dums.values
        sic_var = sic_arr.var(axis=0)
        sic_arr = sic_arr[:, sic_var > 0]

        # SPEC 1: Theme-by-theme regression
        for panel_name, dep_var in [("Panel A: Raw returns", "period_return"),
                                     ("Panel B: Below-mean", "below_mean_return")]:
            y = df[dep_var].values

            # Restricted model (controls only)
            X_r_parts = [np.ones((len(df), 1))]
            for ctrl in ["log_mktcap", "log_bm", "neg_bm", "momentum"]:
                X_r_parts.append(df[ctrl].values.reshape(-1, 1))
            X_r_parts.append(sic_arr)
            X_r = np.hstack(X_r_parts)

            # Unrestricted model (controls + themes)
            X_u_parts = [np.ones((len(df), 1))]
            for tc in theme_cols:
                X_u_parts.append(df[tc].values.reshape(-1, 1))
            for ctrl in ["log_mktcap", "log_bm", "neg_bm", "momentum"]:
                X_u_parts.append(df[ctrl].values.reshape(-1, 1))
            X_u_parts.append(sic_arr)
            X_u = np.hstack(X_u_parts)

            res_r = manual_ols(y, X_r)
            res_u = manual_ols(y, X_u)
            if res_r is None or res_u is None:
                continue

            # F-test
            F_stat = ((res_r["SSR"] - res_u["SSR"]) / q_themes) / (res_u["SSR"] / (res_u["n"] - res_u["k"]))
            F_p = f_sf(F_stat, q_themes, res_u["n"] - res_u["k"])

            delta_R2 = res_u["R2"] - res_r["R2"]
            delta_adj = res_u["adj_R2"] - res_r["adj_R2"]

            print(f"\n  Yr={loading_year} | {panel_name} (n={res_u['n']}):")
            print(f"    Joint F({q_themes},{res_u['n']-res_u['k']}) = {F_stat:.3f}  "
                  f"p = {F_p:.4f}{sig_star(F_p)}  ΔR²={delta_R2:.4f}  Δadj-R²={delta_adj:.4f}")
            print(f"    Individual themes:")

            for idx, (tc, label) in enumerate(zip(theme_cols, theme_labels)):
                ci = 1 + idx
                coef = res_u["beta"][ci]
                t = res_u["t"][ci]
                p = res_u["p"][ci]
                s = sig_star(p)
                marker = " <──" if abs(t) >= 1.65 else ""
                print(f"      {label:25s}  coef={coef:+.4f}  t={t:+6.2f}{s}{marker}")

                all_results.append({
                    "test": "theme_by_theme",
                    "window": window_name,
                    "label": winfo["label"],
                    "panel": panel_name,
                    "loading_year": loading_year,
                    "theme": label,
                    "coef": coef, "t_stat": t, "p_value": p, "stars": s,
                    "n": res_u["n"], "R2": res_u["R2"], "adj_R2": res_u["adj_R2"],
                    "F_stat": F_stat, "F_p": F_p, "delta_R2": delta_R2,
                })

        # SPEC 2: Equal-weighted composite exposure
        df["CTE"] = df[theme_cols].mean(axis=1)
        cte_mean = df["CTE"].mean()
        cte_std = df["CTE"].std()
        print(f"\n  Yr={loading_year} | Composite exposure (equal-weighted):")
        print(f"    Stats: mean={cte_mean:.4f}  std={cte_std:.4f}  "
              f"min={df['CTE'].min():.4f}  max={df['CTE'].max():.4f}")

        for panel_name, dep_var in [("Panel A: Raw returns", "period_return"),
                                     ("Panel B: Below-mean", "below_mean_return")]:
            y = df[dep_var].values
            X_parts = [np.ones((len(df), 1)),
                       df["CTE"].values.reshape(-1, 1)]
            for ctrl in ["log_mktcap", "log_bm", "neg_bm", "momentum"]:
                X_parts.append(df[ctrl].values.reshape(-1, 1))
            X_parts.append(sic_arr)
            X = np.hstack(X_parts)

            res = manual_ols(y, X)
            if res is None: continue

            cte_coef = res["beta"][1]
            cte_t = res["t"][1]
            cte_p = res["p"][1]
            cte_s = sig_star(cte_p)
            marker = " <──" if abs(cte_t) >= 1.65 else ""
            print(f"    {panel_name}: coef={cte_coef:+.4f}  t={cte_t:+.2f}{cte_s}{marker}")

            all_results.append({
                "test": "aggregate_CTE",
                "window": window_name,
                "label": winfo["label"],
                "panel": panel_name,
                "loading_year": loading_year,
                "theme": "CTE_equal_weight",
                "coef": cte_coef, "t_stat": cte_t, "p_value": cte_p, "stars": cte_s,
                "n": res["n"], "R2": res["R2"], "adj_R2": res["adj_R2"],
                "F_stat": np.nan, "F_p": np.nan, "delta_R2": np.nan,
            })


# 4. SUMMARY

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(OUTPUT_DIR, "directional_crisis_returns.csv"), index=False)

print("\n\n" + "=" * 80)
print("SUMMARY: All significant results (p < 0.10)")
print("=" * 80)

sig = results_df[results_df["p_value"] < 0.10].sort_values(
    ["window", "panel", "loading_year", "p_value"])
for _, row in sig.iterrows():
    print(f"  {row['window']:12s} | {row['panel']:25s} | yr={row['loading_year']} | "
          f"{row['theme']:25s} | coef={row['coef']:+.4f} | t={row['t_stat']:+.2f}{row['stars']} "
          f"| test={row['test']}")

print("\nDone.")
