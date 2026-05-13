# Extended 10-K risk text downloader for pre-2005 and post-2005 filings.
# Pre-2005: Item 1A wasn't mandatory, so we also search Item 7 / MD&A
# and standalone risk headers. Post-2005: standard Item 1A extraction.

from __future__ import annotations

import os
import re
import time
import json
import argparse
from pathlib import Path

import pandas as pd
import requests
from sec_edgar_toolkit import SecEdgarApi

session = requests.Session()

try:
    from bs4 import BeautifulSoup
    from bs4 import XMLParsedAsHTMLWarning
    import warnings
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except Exception:
    BeautifulSoup = None


SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data/{cik_nolead}/{acc_no_nodash}/{doc}"


def sec_get(url: str, user_agent: str, sleep_s: float = 0.5) -> requests.Response:
    headers = {
        "User-Agent": user_agent,
        "Accept": "application/json,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive",
    }
    max_retries = 5
    backoff = sleep_s
    for attempt in range(max_retries):
        r = session.get(url, headers=headers, timeout=20)
        if r.status_code == 200:
            time.sleep(sleep_s)
            return r
        if r.status_code in (403, 429):
            print(f"[WARN] {r.status_code} received. Backing off {backoff:.1f}s...")
            time.sleep(backoff)
            backoff *= 2
            continue
        r.raise_for_status()
    raise requests.HTTPError(f"Failed after retries: {url}")


def html_to_text(html: str) -> str:
    if BeautifulSoup is None:
        txt = re.sub(r"<[^>]+>", " ", html)
        txt = re.sub(r"\s+", " ", txt)
        return txt.strip()
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# Item 1A extraction patterns (post-2005)

ITEM_1A_LINE_PAT = re.compile(r"(?im)^\s*item\s+1a\b")
ITEM_1A_ANY_PAT  = re.compile(r"(?im)\bitem\s+1a\b")

STOP_ITEM_1B_LINE_PAT   = re.compile(r"(?im)^\s*item\s+1b\b")
STOP_ITEM_2_LINE_PAT    = re.compile(r"(?im)^\s*item\s+2\b")
STOP_ITEM_NEXT_LINE_PAT = re.compile(r"(?im)^\s*item\s+[2-9]\b")
STOP_ITEM_HIGH_LINE_PAT = re.compile(r"(?im)^\s*item\s+\d{2}\b")
STOP_PART_ANY_LINE_PAT  = re.compile(r"(?im)^\s*part\s+(ii|iii|iv|v)\b")
STOP_SIGNATURES_PAT     = re.compile(r"(?im)^\s*signatures?\s*$")
STOP_EXHIBIT_PAT        = re.compile(r"(?im)^\s*exhibit\s+\d")

ALL_STOP_PATS = [
    STOP_ITEM_1B_LINE_PAT, STOP_ITEM_2_LINE_PAT, STOP_ITEM_NEXT_LINE_PAT,
    STOP_ITEM_HIGH_LINE_PAT, STOP_PART_ANY_LINE_PAT, STOP_SIGNATURES_PAT,
    STOP_EXHIBIT_PAT,
]

RISK_FACTORS_LINE_PAT = re.compile(r"(?im)^\s*risk\s+factors\s*$")
TOC_PHRASE_PAT   = re.compile(r"(?im)table\s+of\s+contents")
DOT_LEADER_PAT   = re.compile(r"\.\s*\.\s*\.|\.{5,}")
PAGE_NUM_END_PAT = re.compile(r"\s\d{1,4}\s*$")
PAGE_NUM_ONLY_PAT = re.compile(r"^.{10,120}\s+\d{1,3}\s*$")


def extract_item_1a(text: str) -> str | None:
    """Extract Item 1A (Risk Factors) from plain text — same logic as 04."""
    if not text or len(text) < 2000:
        return None

    lines = text.splitlines()

    def _pos_for_line_index(i):
        return sum(len(x) + 1 for x in lines[:i])

    def _is_all_caps_heading(s):
        s = s.strip()
        if len(s) < 5 or len(s) > 80:
            return False
        letters = [c for c in s if c.isalpha()]
        if len(letters) < 4:
            return False
        upper = sum(1 for c in letters if c.isupper())
        return (upper / len(letters)) >= 0.85

    def _looks_like_toc_line(s):
        s_stripped = s.strip()
        if not s_stripped:
            return False
        if DOT_LEADER_PAT.search(s_stripped):
            return True
        if PAGE_NUM_END_PAT.search(s_stripped) and len(s_stripped) <= 120:
            if re.search(r"(?i)\bitem\b", s_stripped) or _is_all_caps_heading(s_stripped):
                return True
        return False

    def _near_toc(i):
        lo = max(0, i - 80)
        window_before = "\n".join(lines[lo:i])
        if TOC_PHRASE_PAT.search(window_before):
            return True
        toc_like = 0
        for j in range(max(0, i - 20), min(len(lines), i + 5)):
            if _looks_like_toc_line(lines[j]):
                toc_like += 1
        return toc_like >= 5

    def _in_first_pct(i, pct=0.10):
        pos = _pos_for_line_index(i)
        return pos < len(text) * pct

    def _has_real_body_after(i):
        snippet_lines = lines[i + 1 : i + 40]
        snippet = "\n".join(snippet_lines)
        if len(snippet) < 400:
            return False
        letters = [c for c in snippet if c.isalpha()]
        if not letters:
            return False
        lower = sum(1 for c in letters if c.islower())
        if (lower / len(letters)) < 0.15:
            return False
        tocish = sum(1 for ln in snippet_lines if _looks_like_toc_line(ln))
        if tocish >= 5:
            return False
        return True

    def _looks_like_risk_toc(start_line):
        snippet = lines[start_line + 1 : start_line + 60]
        if not snippet:
            return False
        page_num_lines = sum(1 for ln in snippet if PAGE_NUM_ONLY_PAT.match(ln.strip()))
        return page_num_lines >= max(3, len(snippet) * 0.25)

    # Search for Item 1A
    candidates = []
    for i, ln in enumerate(lines):
        if ITEM_1A_LINE_PAT.search(ln):
            if not _in_first_pct(i, 0.08):
                candidates.append(i)

    if not candidates:
        for i, ln in enumerate(lines):
            if RISK_FACTORS_LINE_PAT.search(ln) and ln.strip().upper() == "RISK FACTORS":
                if not _in_first_pct(i, 0.08):
                    candidates.append(i)

    for i in candidates:
        if _near_toc(i):
            continue
        if _looks_like_toc_line(lines[i]):
            continue
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        if j >= len(lines):
            continue
        next_line = lines[j].strip()
        if not (_is_all_caps_heading(next_line) or any(c.islower() for c in next_line)):
            continue
        if not _has_real_body_after(i):
            continue
        if _looks_like_risk_toc(i):
            continue

        start_pos = _pos_for_line_index(i)
        tail = text[start_pos:]
        stop_positions = []
        for pat in ALL_STOP_PATS:
            m = pat.search(tail)
            if m:
                stop_positions.append(m.start())
        chunk = tail[: min(stop_positions)].strip() if stop_positions else tail.strip()
        chunk = chunk[:300_000]
        if len(chunk) < 3000:
            return None
        return chunk

    # Fallback
    for m in re.finditer(ITEM_1A_ANY_PAT, text):
        if m.start() >= len(text) * 0.08:
            tail = text[m.start():]
            stop_positions = []
            for pat in ALL_STOP_PATS:
                m2 = pat.search(tail)
                if m2:
                    stop_positions.append(m2.start())
            chunk = tail[: min(stop_positions)].strip() if stop_positions else tail.strip()
            chunk = chunk[:300_000]
            if len(chunk) >= 3000:
                return chunk
    return None


# Pre-2005 risk text extraction

# Patterns for finding risk-related sections in pre-2005 10-Ks
RISK_SECTION_HEADERS = re.compile(
    r"(?im)^\s*("
    r"risk\s+factors?"
    r"|factors?\s+(?:that\s+)?(?:may\s+)?affect"
    r"|(?:certain\s+)?risk(?:s)?\s+(?:and\s+)?(?:uncertainties|considerations)"
    r"|risk\s+management"
    r"|credit\s+risk"
    r"|market\s+risk"
    r"|operational\s+risk"
    r"|liquidity\s+risk"
    r"|interest\s+rate\s+risk"
    r"|cautionary\s+(?:statement|note|factors)"
    r"|forward[\s-]looking\s+statements?"
    r")\s*$"
)

# Item 7 / MD&A patterns (often contains risk discussion pre-2005)
ITEM_7_PAT = re.compile(r"(?im)^\s*item\s+7\.?\s")
ITEM_7A_PAT = re.compile(r"(?im)^\s*item\s+7a\.?\s")
ITEM_8_PAT = re.compile(r"(?im)^\s*item\s+8\b")


def extract_risk_text_pre2005(text: str) -> str | None:
    """
    Extract risk-related text from a pre-2005 10-K where Item 1A may not exist.

    Strategy (mimicking H&H's metaHeuristica approach):
    1. Try Item 1A extraction first (some banks had it voluntarily)
    2. Look for standalone "Risk Factors" or similar risk section headers
    3. Extract Item 7 (MD&A) and Item 7A (Quantitative Risk Disclosures)
    4. Concatenate all found risk text

    Guardrails (to avoid accidentally using the whole 10-K):
    - Individual risk header sections: capped at 50,000 chars each
    - Item 7 (MD&A): capped at 80,000 chars (these can be huge)
    - Item 7A: capped at 40,000 chars
    - Final combined output: capped at 200,000 chars (~30,000 words)
    - Maximum word count: 45,000 words (typical Item 1A is 5k-20k words)
    - If combined text > 30% of full 10-K, something went wrong → reject

    The goal: get risk-discussion paragraphs, NOT the entire filing.
    """
    if not text or len(text) < 2000:
        return None

    # Hard limits to avoid grabbing the entire filing
    MAX_SECTION_CHARS    = 50_000   # per risk-header section
    MAX_ITEM7_CHARS      = 80_000   # MD&A can be long but cap it
    MAX_ITEM7A_CHARS     = 40_000   # quantitative risk section
    MAX_COMBINED_CHARS   = 200_000  # final output cap (~30k words)
    MAX_COMBINED_WORDS   = 45_000   # absolute word limit
    MAX_PCT_OF_FULL_10K  = 0.30     # reject if > 30% of full 10-K

    # 1. Try Item 1A first (voluntary before 2005)
    item1a = extract_item_1a(text)
    if item1a and len(item1a) >= 3000:
        return item1a

    lines = text.splitlines()
    collected_sections = []

    def _pos_for_line_index(i):
        return sum(len(x) + 1 for x in lines[:i])

    def _in_first_pct(i, pct=0.08):
        pos = _pos_for_line_index(i)
        return pos < len(text) * pct

    # 2. Find standalone risk section headers
    for i, ln in enumerate(lines):
        if _in_first_pct(i, 0.06):
            continue  # Skip TOC area
        if RISK_SECTION_HEADERS.search(ln):
            # Check this isn't a TOC line
            if DOT_LEADER_PAT.search(ln):
                continue
            if PAGE_NUM_END_PAT.search(ln.strip()) and len(ln.strip()) <= 120:
                continue

            # Extract text from this header to the next major header
            start_pos = _pos_for_line_index(i)
            tail = text[start_pos:]

            # Stop at next Item or Part heading
            stop_pats = [
                re.compile(r"(?im)^\s*item\s+\d"),
                re.compile(r"(?im)^\s*part\s+(i{1,3}|iv|v)\b"),
                STOP_SIGNATURES_PAT,
            ]
            stop_positions = []
            for pat in stop_pats:
                for m in pat.finditer(tail):
                    if m.start() > 50:  # skip self-match
                        stop_positions.append(m.start())
                        break

            section = tail[: min(stop_positions)].strip() if stop_positions else tail.strip()
            section = section[:MAX_SECTION_CHARS]  # hard cap per section

            if len(section) >= 500:
                collected_sections.append(section)

    # 3. Extract Item 7 / MD&A (common source of risk discussion pre-2005)
    for i, ln in enumerate(lines):
        if _in_first_pct(i, 0.06):
            continue
        if ITEM_7_PAT.search(ln) and not ITEM_7A_PAT.search(ln):
            start_pos = _pos_for_line_index(i)
            tail = text[start_pos:]

            stop_positions = []
            for pat in [ITEM_7A_PAT, ITEM_8_PAT, STOP_PART_ANY_LINE_PAT]:
                m = pat.search(tail)
                if m and m.start() > 50:
                    stop_positions.append(m.start())

            section = tail[: min(stop_positions)].strip() if stop_positions else tail.strip()
            section = section[:MAX_ITEM7_CHARS]  # hard cap for MD&A

            if len(section) >= 2000:
                collected_sections.append(section)
            break  # Only take first Item 7

    # 3b. Item 7A (Quantitative and Qualitative Disclosures About Market Risk)
    for i, ln in enumerate(lines):
        if _in_first_pct(i, 0.06):
            continue
        if ITEM_7A_PAT.search(ln):
            start_pos = _pos_for_line_index(i)
            tail = text[start_pos:]

            stop_positions = []
            for pat in [ITEM_8_PAT, STOP_PART_ANY_LINE_PAT]:
                m = pat.search(tail)
                if m and m.start() > 50:
                    stop_positions.append(m.start())

            section = tail[: min(stop_positions)].strip() if stop_positions else tail.strip()
            section = section[:MAX_ITEM7A_CHARS]  # hard cap for Item 7A

            if len(section) >= 500:
                collected_sections.append(section)
            break

    if not collected_sections:
        return None

    # Combine and apply guardrails
    combined = "\n\n".join(collected_sections)

    # Character cap
    combined = combined[:MAX_COMBINED_CHARS]

    # Word count cap
    word_count = len(combined.split())
    if word_count > MAX_COMBINED_WORDS:
        # Truncate to MAX_COMBINED_WORDS words
        words = combined.split()[:MAX_COMBINED_WORDS]
        combined = " ".join(words)

    # Sanity check: if combined text is > 30% of full 10-K, we probably
    # grabbed too much (means stop patterns failed). Log warning but
    # still return — the downstream LDA/textual factors will handle noise.
    pct_of_full = len(combined) / len(text) if text else 0
    if pct_of_full > MAX_PCT_OF_FULL_10K:
        print(f"    WARNING: extracted {pct_of_full:.0%} of full 10-K "
              f"({len(combined):,} / {len(text):,} chars, {word_count:,} words)")

    # Minimum length check
    if len(combined) < 2000:
        return None

    return combined


def extract_risk_text(text: str, year: int) -> tuple[str | None, str]:
    """
    Unified extraction: uses Item 1A for post-2005, risk text search for pre-2005.
    Returns (text, method) where method is "item1a", "risk_sections", or None.
    """
    if year >= 2005:
        result = extract_item_1a(text)
        if result:
            return result, "item1a"
        # Fallback: try risk section extraction even for post-2005
        result = extract_risk_text_pre2005(text)
        if result:
            return result, "risk_sections_fallback"
        return None, "no_risk_text"
    else:
        result = extract_risk_text_pre2005(text)
        if result:
            return result, "risk_sections"
        return None, "no_risk_text"



def main():
    BASE = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE, "..", "data")

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(DATA_DIR, "cik_years_to_scrape.csv"))
    ap.add_argument("--out_dir", default=os.path.join(DATA_DIR, "10k_item1a_v2"))
    ap.add_argument(
        "--user_agent",
        default="Magnus Ellegaard (Copenhagen Business School, MSc) mach21ax@student.cbs.dk",
    )
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--max_rows", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--reextract_only", action="store_true")
    ap.add_argument("--year_min", type=int, default=0, help="Only process years >= this")
    ap.add_argument("--year_max", type=int, default=9999, help="Only process years <= this")
    args = ap.parse_args()

    api = SecEdgarApi(user_agent=args.user_agent)

    df = pd.read_csv(args.input, dtype={"CIK": str, "YEAR": int})
    df.columns = [c.upper() for c in df.columns]
    assert {"CIK", "YEAR"}.issubset(df.columns)

    # Filter year range
    df = df[(df["YEAR"] >= args.year_min) & (df["YEAR"] <= args.year_max)]

    df["CIK"] = (
        df["CIK"].astype(str).str.strip()
        .str.replace(r"\D", "", regex=True)
        .replace({"": pd.NA})
    )
    df = df.dropna(subset=["CIK"]).copy()
    df["CIK"] = df["CIK"].apply(lambda x: str(int(x)).zfill(10))
    df = df.drop_duplicates(subset=["CIK", "YEAR"]).copy()
    total_rows = len(df)
    print(f"Starting download for {total_rows} CIK-YEAR observations "
          f"({df['YEAR'].min()}-{df['YEAR'].max()})...")

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    txt_dir = out_dir / "item1a"
    raw_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    # Re-extraction mode
    if args.reextract_only:
        print("Running in RE-EXTRACTION mode (no SEC downloads)...")
        raw_files = list(raw_dir.glob("*_primary.html"))
        print(f"Found {len(raw_files)} raw HTML files.")

        reextract_log = []
        for idx, raw_file in enumerate(raw_files):
            try:
                name = raw_file.stem
                parts = name.split("_")
                cik = parts[0]
                year = int(parts[1])

                if year < args.year_min or year > args.year_max:
                    continue

                out_txt = txt_dir / f"{cik}_{year}_item1a.txt"
                if out_txt.exists():
                    continue

                html = raw_file.read_text(encoding="utf-8", errors="ignore")
                text = html_to_text(html)
                result, method = extract_risk_text(text, year)

                if result:
                    out_txt.write_text(result, encoding="utf-8")
                    status = f"ok_{method}"
                else:
                    status = "still_no_risk_text"

                reextract_log.append({"cik": cik, "year": year, "status": status, "method": method})

                if (idx + 1) % 100 == 0:
                    print(f"Re-extracted {idx+1}/{len(raw_files)}")

            except Exception as e:
                reextract_log.append({
                    "cik": cik if "cik" in dir() else None,
                    "year": year if "year" in dir() else None,
                    "status": f"error_{type(e).__name__}",
                    "msg": str(e)[:200]
                })

        pd.DataFrame(reextract_log).to_csv(out_dir / "reextract_log.csv", index=False)
        print("Re-extraction complete. Log saved to:", out_dir / "reextract_log.csv")
        return

    # Normal download mode
    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    log_rows = []

    for idx, row in df.iterrows():
        cik = re.sub(r"\D", "", str(row["CIK"]))
        cik = str(int(cik)).zfill(10)
        year = int(row["YEAR"])

        out_txt = txt_dir / f"{cik}_{year}_item1a.txt"
        out_raw = raw_dir / f"{cik}_{year}_primary.html"

        if args.resume and out_txt.exists():
            continue

        try:
            filings = api.get_company_submissions(cik)

            filing = None

            def scan_block(block):
                forms = block.get("form", [])
                accession = block.get("accessionNumber", [])
                primary_docs = block.get("primaryDocument", [])
                filing_dates = block.get("filingDate", [])
                report_dates = block.get("reportDate", [])

                # For pre-2005 also accept 10-K405 and 10-KSB40 forms
                ORIGINALS = {"10-K", "10-KSB", "10-KT", "10-KSB405", "10-K405"}
                AMENDMENTS = {"10-K/A", "10-KSB/A", "10-KT/A", "10-KSB405/A", "10-K405/A"}

                rows_list = list(zip(forms, accession, primary_docs, filing_dates, report_dates))

                def _year_match(fdate, rdate):
                    if rdate and rdate.startswith(str(year)):
                        return True
                    if fdate and (fdate.startswith(str(year)) or fdate.startswith(str(year + 1))):
                        return True
                    return False

                def _make_result(f, acc, doc, fdate):
                    return {"form": f, "accessionNumber": acc, "primaryDocument": doc, "filingDate": fdate}

                for f, acc, doc, fdate, rdate in rows_list:
                    if f in ORIGINALS and _year_match(fdate, rdate):
                        return _make_result(f, acc, doc, fdate)

                for f, acc, doc, fdate, rdate in rows_list:
                    if f in AMENDMENTS and _year_match(fdate, rdate):
                        return _make_result(f, acc, doc, fdate)

                return None

            # Check recent filings
            recent = filings.get("filings", {}).get("recent", {})
            filing = scan_block(recent)

            # Check historical files if needed
            if filing is None:
                files = filings.get("filings", {}).get("files", [])
                for fmeta in files:
                    fname = fmeta.get("name")
                    if not fname:
                        continue
                    hist_url = f"https://data.sec.gov/submissions/{fname}"
                    hist_json = sec_get(hist_url, args.user_agent, sleep_s=args.sleep).json()
                    filing = scan_block(hist_json)
                    if filing:
                        break

            if filing is None:
                log_rows.append({"cik": cik, "year": year, "status": "no_10k_found"})
                continue

            acc = filing["accessionNumber"]
            doc = filing["primaryDocument"]
            cik_nolead = str(int(cik))
            acc_nodash = acc.replace("-", "")
            url = SEC_ARCHIVES.format(cik_nolead=cik_nolead, acc_no_nodash=acc_nodash, doc=doc)

            # Download
            html = sec_get(url, args.user_agent, sleep_s=args.sleep).text
            out_raw.write_text(html, encoding="utf-8")

            # Extract risk text
            text = html_to_text(html)
            result, method = extract_risk_text(text, year)

            if result is None:
                (raw_dir / f"{cik}_{year}_fulltext.txt").write_text(text, encoding="utf-8")
                log_rows.append({
                    "cik": cik, "year": year, "status": f"downloaded_no_risk_text",
                    "form": filing["form"], "method": method,
                    "text_len": len(text)
                })
                continue

            out_txt.write_text(result, encoding="utf-8")
            log_rows.append({
                "cik": cik, "year": year, "status": "ok",
                "form": filing["form"], "filingDate": filing["filingDate"],
                "method": method, "risk_text_len": len(result)
            })

        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            log_rows.append({"cik": cik, "year": year, "status": f"http_error_{code}"})
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout):
            log_rows.append({"cik": cik, "year": year, "status": "timeout"})
            continue
        except Exception as e:
            log_rows.append({"cik": cik, "year": year, "status": f"error_{type(e).__name__}", "msg": str(e)[:200]})

        if (idx + 1) % 100 == 0:
            pd.DataFrame(log_rows).to_csv(out_dir / "download_log_extended.csv", index=False)
        if (idx + 1) % 50 == 0:
            ok_count = sum(1 for r in log_rows if r["status"] == "ok")
            print(f"Processed {idx+1}/{total_rows}  ({ok_count} ok)")

    pd.DataFrame(log_rows).to_csv(out_dir / "download_log_extended.csv", index=False)

    # Summary
    log_df = pd.DataFrame(log_rows)
    print("\n" + "=" * 60)
    print("DOWNLOAD SUMMARY")
    print("=" * 60)
    print(log_df["status"].value_counts().to_string())
    if "method" in log_df.columns:
        print("\nExtraction methods:")
        print(log_df[log_df["status"] == "ok"]["method"].value_counts().to_string())
    print(f"\nLog saved to: {out_dir / 'download_log_extended.csv'}")
    print(f"Risk text files in: {txt_dir}")


if __name__ == "__main__":
    main()
