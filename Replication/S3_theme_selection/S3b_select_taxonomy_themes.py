# Track 2: select low-CV clusters that match a pre-specified risk taxonomy.
# Sources: HH (2019) Table 4, BCBS operational/climate risk principles,
# FSB COVID-19, OFAC/FinCEN sanctions/AML guidance.

import os
import re
import ast
import numpy as np
import pandas as pd


# PATHS

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(BASE, "..", "data", "outputs_textual_factors_v2")

STATS_CSV    = os.path.join(OUT_DIR, "cluster_variance_stats.csv")
WORDS_CSV    = os.path.join(OUT_DIR, "topics_words.csv")
SV_CSV       = os.path.join(OUT_DIR, "singular_values.csv")
FILTERED_CSV = os.path.join(OUT_DIR, "first_doc_topics_filtered.csv")


# SETTINGS

N_TOP_WORDS = 30          # How many top words per cluster to check
MIN_TAXONOMY_SCORE = 2    # Minimum keyword matches to count as a hit


# RISK TAXONOMY

# Format: { category_name: { "source": str, "keywords": set } }
#
# Keywords are matched against the top-N words of each cluster.
# Both exact matches and substring matches are checked (e.g.,
# "credit_loss" matches the keyword "credit").


RISK_TAXONOMY = {
    # ========================
    # HH (2019) Table 4 themes
    # ========================
    "real_estate": {
        "source": "HH Table 4 (#26); Basel III",
        "keywords": {
            "real_estate", "residential", "commercial_real", "mortgage",
            "housing", "foreclosure", "property", "construction",
            "home_equity", "homeowner", "family_residential",
            "owner_occupied", "estate_market", "estate_loan",
            "housing_market", "multi_family", "single_family",
        },
    },
    "credit_card": {
        "source": "HH Table 4 (#8)",
        "keywords": {
            "credit_card", "debit", "cardholder", "interchange",
            "visa", "mastercard", "card_loan", "card_issuer",
        },
    },
    "credit_risk_loans": {
        "source": "HH Table 4 (#11 Deposits, #22 OBS); Basel III Pillar 1",
        "keywords": {
            "credit", "loan", "default", "delinquency", "delinquent",
            "nonperforming", "charge_offs", "allowance", "provision",
            "impairment", "impaired", "collateral", "creditworthiness",
            "borrower", "underwriting", "classified", "substandard",
            "doubtful", "loss_loan", "credit_loss", "credit_risk",
            "problem_loan", "credit_exposure", "forbearance",
            "originated", "consumer_loan", "mortgage_loan",
        },
    },
    "interest_rate_prepayment": {
        "source": "HH Table 4 (#24 Prepayment); Basel III IRRBB",
        "keywords": {
            "interest_rate", "rate_risk", "net_interest", "margin",
            "basis_point", "repricing", "rate_sensitivity", "yield_curve",
            "prepayment", "amortization", "refinance", "adjustable_rate",
            "fixed_rate", "rate_change", "rate_environment",
            "rate_increase", "rate_decline",
        },
    },
    "derivative_hedging": {
        "source": "HH Table 4 (#12)",
        "keywords": {
            "derivative", "hedging", "hedge", "swap", "futures",
            "notional", "credit_derivative",
            "derivative_instrument", "credit_spread", "mark_to_market",
        },
    },
    "market_risk_volatility": {
        "source": "HH Table 4 (#23); Basel III Pillar 1",
        "keywords": {
            "volatility", "market_risk", "trading",
            "fluctuation", "market_condition", "market_interest",
            "market_volatility", "turmoil",
        },
    },
    "liquidity_funding": {
        "source": "HH Table 4 (#15 Funding sources); Basel III Pillar 2",
        "keywords": {
            "liquidity", "funding", "deposit", "cash_flow", "outflow",
            "inflow", "brokered_deposit", "wholesale_funding",
            "liquidity_risk", "funding_source", "liquid", "illiquid",
            "illiquidity", "fhlb", "borrowing", "capital_market",
        },
    },
    "capital_adequacy": {
        "source": "HH Table 4 (#27 Regulatory capital); Basel III Pillar 1",
        "keywords": {
            "tier", "leverage_ratio", "capital_ratio",
            "capital_requirement", "capital_adequacy", "undercapitalized",
            "well_capitalized", "capital_conservation", "regulatory_capital",
            "stress_test", "stress_testing", "risk_weighted",
        },
    },
    "securitization": {
        "source": "HH Table 4 (#29)",
        "keywords": {
            "securitization", "mortgage_backed", "asset_backed",
            "structured", "tranche", "securitized", "cdo", "clo",
            "backed_security",
        },
    },
    "operational_risk": {
        "source": "HH Table 4 (#23); Basel III Pillar 1",
        "keywords": {
            "operational", "operational_risk", "disruption",
            "interruption", "disaster",
            "business_continuity", "processing_error",
        },
    },
    "internal_controls": {
        "source": "HH Table 4 (#19)",
        "keywords": {
            "internal_control", "audit", "auditor",
            "deficiency", "sarbanes", "oxley",
            "material_weakness", "risk_management",
        },
    },
    "regulatory_compliance": {
        "source": "HH Table 4 (#27, #16 Governance); Basel III",
        "keywords": {
            "regulation", "compliance", "regulatory", "examination",
            "enforcement", "supervision", "dodd_frank", "frank_act",
            "basel", "volcker", "consent_order", "cease_desist",
            "penalty", "fine", "sanction", "noncompliance",
            "enforcement_action", "regulatory_action",
        },
    },
    "legal_litigation": {
        "source": "HH Table 4 (#20 Lawsuit)",
        "keywords": {
            "litigation", "lawsuit", "class_action", "legal",
            "court", "settlement", "judgment", "plaintiff",
            "investigation", "claim", "attorney",
        },
    },
    "reputation": {
        "source": "HH Table 4 (#28)",
        "keywords": {
            "reputation", "reputational", "publicity", "perception",
            "reputational_damage", "reputational_harm",
            "reputational_risk", "negative_publicity",
            "damage_reputation",
        },
    },
    "competition": {
        "source": "HH Table 4 (#6)",
        "keywords": {
            "competition", "competitive", "competitor", "market_share",
            "pricing_pressure", "nonbank", "disintermediation",
        },
    },
    "counterparty": {
        "source": "HH Table 4 (#7)",
        "keywords": {
            "counterparty", "counterparties", "clearing",
            "counterparty_risk", "counterparty_credit",
        },
    },
    "insurance_deposit": {
        "source": "HH Table 4 (#18 Insurance, #3 Certificate deposit)",
        "keywords": {
            "deposit_insurance", "fdic", "insured", "uninsured",
            "insurance", "insured_deposit", "insurance_fund",
        },
    },
    "rating_agencies": {
        "source": "HH Table 4 (#25)",
        "keywords": {
            "downgrade", "credit_rating", "rating_agency",
            "standard_poor", "moody", "fitch", "rating_downgrade",
        },
    },
    "dividend_distribution": {
        "source": "HH Table 4 (#13 Dividends)",
        "keywords": {
            "dividend", "buyback", "repurchase",
            "capital_distribution", "payout", "pay_dividend",
        },
    },
    "currency_exchange": {
        "source": "HH Table 4 (#9)",
        "keywords": {
            "currency", "foreign_exchange", "foreign_currency",
            "forex", "exchange_rate",
        },
    },
    "mergers_acquisition": {
        "source": "HH Table 4 (#21)",
        "keywords": {
            "merger", "acquisition", "takeover", "integration",
            "acquiring", "divestiture",
        },
    },
    "compensation_labor": {
        "source": "HH Table 4 (#5 Compensation)",
        "keywords": {
            "compensation", "salary", "wage", "labor", "employee",
            "workforce", "staffing", "talent", "retention",
            "hiring", "unemployment",
        },
    },
    "taxes": {
        "source": "HH Table 4 (#31)",
        "keywords": {
            "tax_law", "income_tax", "tax_reform",
            "deferred_tax", "tax_liability", "tax_credit",
            "taxable", "taxation",
        },
    },
    "student_loans": {
        "source": "HH Table 4 (#30)",
        "keywords": {
            "student", "student_loan", "education", "education_loan",
        },
    },
    "accounting_standards": {
        "source": "HH Table 4 (#1 Accounting)",
        "keywords": {
            "accounting", "gaap", "fasb", "asu", "cecl",
            "restatement", "impairment_model",
            "accounting_standard", "fair_value",
        },
    },
    # ========================================
    # Basel III / BCP 2024 — newer risk types
    # ========================================
    "cybersecurity": {
        "source": "BCBS, Principles for Operational Risk (2021 rev.); BCP 2024 Principle 25",
        "keywords": {
            "cyber", "cybersecurity", "breach", "hacking", "malware",
            "phishing", "ransomware", "data_breach", "security_breach",
            "cyber_attack", "information_security", "data_protection",
            "unauthorized_access", "identity_theft",
        },
    },
    "technology_risk": {
        "source": "BCBS, Principles for Operational Risk (2021 rev.); BCP 2024 Principle 25",
        "keywords": {
            "technology_risk", "technology_change", "fintech",
            "digital_banking", "digital_transformation",
            "cloud_computing", "automation",
            "information_technology", "system_failure",
        },
    },
    "climate_environmental": {
        "source": "BCBS, Climate-related financial risks (2021); TCFD (2017)",
        "keywords": {
            "climate", "environmental", "emission", "carbon",
            "esg", "flood", "hurricane", "wildfire", "drought",
            "weather", "natural_disaster", "sea_level",
            "pollution", "contamination", "hazardous", "toxic",
        },
    },
    "third_party_vendor": {
        "source": "BCBS, Principles for Operational Risk (2021 rev.)",
        "keywords": {
            "vendor", "third_party", "outsourcing", "outsource",
            "service_provider", "contractor", "supply_chain",
        },
    },
    "money_laundering_bsa": {
        "source": "BCBS AML/CFT standards; FinCEN BSA guidance",
        "keywords": {
            "money_laundering", "laundering", "bsa", "aml",
            "bank_secrecy", "suspicious", "terrorist_financing",
            "know_your_customer", "kyc", "patriot", "ofac",
        },
    },
    "pandemic_health": {
        "source": "FSB, COVID-19 Pandemic: Financial Stability Implications (2020)",
        "keywords": {
            "pandemic", "covid", "coronavirus", "virus", "disease",
            "health", "public_health", "outbreak", "epidemic",
            "contagion",
        },
    },
    "geopolitical": {
        "source": "FSB Global Monitoring Report (2022); OFAC sanctions framework",
        "keywords": {
            "geopolitical", "terrorism", "terrorist", "war",
            "sanctions", "sanction", "trade_policy", "tariff",
            "embargo", "conflict", "ukraine",
            "geopolitical_risk", "unrest",
        },
    },
    "sovereign_macro": {
        "source": "HH Table 4 (#14 FDIC, #17 Housing crisis); Basel III macroprudential",
        "keywords": {
            "recession", "downturn", "economic_downturn", "gdp",
            "unemployment", "fiscal", "monetary_policy",
            "federal_fund", "federal_reserve", "macroeconomic",
            "economic_slowdown", "recessionary",
        },
    },
}



# HELPER FUNCTIONS

def parse_top_words(dist_str, n=N_TOP_WORDS):
    """Extract top-n words from the topic_distribution string."""
    try:
        cleaned = re.sub(r"np\.float64\(([^)]+)\)", r"\1", str(dist_str))
        d = ast.literal_eval(cleaned)
        return sorted(d, key=d.get, reverse=True)[:n]
    except Exception:
        return []


def match_taxonomy(top_words, taxonomy):
    """
    Match a cluster's top words against the risk taxonomy.
    Returns list of (category, matched_words, score, source) sorted by score.

    Matching logic:
      - Exact match only: word == keyword  (no substring matching).
      - Compound-word match: if keyword contains '_', also check whether
        the individual parts appear as separate words in top_words
        (e.g., keyword "credit_risk" matches if both "credit" AND "risk"
        are in top_words). This catches multi-word terms that Word2Vec
        may have tokenised differently.

    The old substring matching (kw in w or w in kw) was dropped because
    generic short keywords like "tax", "system", "capital" matched far
    too many unrelated clusters.
    """
    top_set = set(top_words)
    matches = []
    for category, info in taxonomy.items():
        keywords = info["keywords"]
        source = info["source"]
        matched_words = set()
        for kw in keywords:
            # 1. Direct exact match
            if kw in top_set:
                matched_words.add(kw)
            # 2. Compound keyword: check if ALL parts are present
            elif "_" in kw:
                parts = kw.split("_")
                if len(parts) >= 2 and all(p in top_set for p in parts):
                    matched_words.update(parts)
        score = len(matched_words)
        if score > 0:
            matches.append((category, matched_words, score, source))
    matches.sort(key=lambda x: -x[2])
    return matches


# MAIN
def main():
    print("=" * 70)
    print("TRACK 2: Low-CV Taxonomy Selection")
    print("=" * 70)

    # Load data
    stats = pd.read_csv(STATS_CSV)
    words = pd.read_csv(WORDS_CSV)
    sv = pd.read_csv(SV_CSV)

    word_lookup = dict(zip(words["topic"], words["topic_distribution"]))
    sv_lookup = dict(zip(sv["cluster"], sv["leading_singular"]))

    # Identify high-CV clusters (Track 1) from the filtered file
    filtered = pd.read_csv(FILTERED_CSV)
    high_cv_ids = set(
        int(c.replace("topic_loading_", ""))
        for c in filtered.columns if c.startswith("topic_loading_")
    )
    print(f"\n  High-CV clusters (Track 1): {len(high_cv_ids)}")

    # Low-CV clusters = everything NOT in Track 1
    low_cv = stats[~stats["cluster_id"].isin(high_cv_ids)].copy()
    low_cv = low_cv.sort_values("mean_loading", ascending=False)
    print(f"  Low-CV clusters (Track 2 candidates): {len(low_cv)}")

    # Score each cluster
    results = []
    for _, row in low_cv.iterrows():
        cid = int(row["cluster_id"])
        top_w = parse_top_words(word_lookup.get(cid, ""))
        matches = match_taxonomy(top_w, RISK_TAXONOMY)

        # Best match with score >= MIN_TAXONOMY_SCORE
        strong = [(c, mw, s, src) for c, mw, s, src in matches if s >= MIN_TAXONOMY_SCORE]

        if strong:
            best_cat, best_words, best_score, best_source = strong[0]
            all_cats = "; ".join(f"{c}({s})" for c, _, s, _ in strong[:3])
        else:
            best_cat = ""
            best_words = set()
            best_score = 0
            best_source = ""
            all_cats = ""

        results.append({
            "cluster_id": cid,
            "cv": row["cv"],
            "mean_loading": row["mean_loading"],
            "singular_value": sv_lookup.get(cid, np.nan),
            "top_words": ", ".join(top_w[:10]),
            "taxonomy_match": best_cat,
            "taxonomy_score": best_score,
            "taxonomy_source": best_source,
            "matched_keywords": ", ".join(sorted(best_words)),
            "all_matches": all_cats,
            "selected": best_score >= MIN_TAXONOMY_SCORE,
        })

    df = pd.DataFrame(results)
    n_selected = df["selected"].sum()
    n_skipped = (~df["selected"]).sum()

    print(f"\n  Taxonomy matches (score >= {MIN_TAXONOMY_SCORE}): {n_selected}")
    print(f"  No strong match (skipped): {n_skipped}")

    # Save
    out_path = os.path.join(OUT_DIR, "track2_taxonomy_candidates.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    # Print selected
    selected = df[df["selected"]].sort_values("singular_value", ascending=False)
    print(f"\n  --- SELECTED ({len(selected)} clusters) ---")
    for _, row in selected.iterrows():
        print(f"    [{row['cluster_id']:3d}] SV={row['singular_value']:8.1f} | "
              f"CV={row['cv']:.3f} | {row['taxonomy_match']:25s} | "
              f"{row['top_words']}")

    # Category summary
    print(f"\n  --- By risk category ---")
    cat_counts = selected["taxonomy_match"].value_counts()
    for cat, count in cat_counts.items():
        src = RISK_TAXONOMY[cat]["source"]
        print(f"    {cat:30s}: {count:2d}  ({src})")


if __name__ == "__main__":
    main()
