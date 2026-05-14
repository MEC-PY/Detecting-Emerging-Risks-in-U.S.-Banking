from __future__ import annotations

import os
import re
import time
import json
import argparse
from pathlib import Path

import pandas as pd
import requests
from sec_edgar_toolkit import SecEdgarApi, SecEdgarApiError


session = requests.Session()

# Optional but very helpful for cleaner text extraction:
# pip install beautifulsoup4 lxml
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
    """
    SEC request wrapper with:
    - persistent session
    - browser-like headers
    - exponential backoff on 403/429
    """

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
        # Fallback: strip tags crudely
        txt = re.sub(r"<[^>]+>", " ", html)
        txt = re.sub(r"\s+", " ", txt)
        return txt.strip()

    soup = BeautifulSoup(html, "lxml")
    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# Item markers (use line-start to avoid false matches inside prose like "items")
ITEM_1A_LINE_PAT = re.compile(r"(?im)^\s*item\s+1a\b")
ITEM_1A_ANY_PAT  = re.compile(r"(?im)\bitem\s+1a\b")

STOP_ITEM_1B_LINE_PAT   = re.compile(r"(?im)^\s*item\s+1b\b")
STOP_ITEM_2_LINE_PAT    = re.compile(r"(?im)^\s*item\s+2\b")
STOP_ITEM_NEXT_LINE_PAT = re.compile(r"(?im)^\s*item\s+[2-9]\b")   # item 2-9 at line start
STOP_ITEM_HIGH_LINE_PAT = re.compile(r"(?im)^\s*item\s+\d{2}\b")   # item 10+ (e.g. item 15)
STOP_PART_ANY_LINE_PAT  = re.compile(r"(?im)^\s*part\s+(ii|iii|iv|v)\b")  # Part II onward
STOP_SIGNATURES_PAT     = re.compile(r"(?im)^\s*signatures?\s*$")           # Signature page
STOP_EXHIBIT_PAT        = re.compile(r"(?im)^\s*exhibit\s+\d")              # Exhibit block

ALL_STOP_PATS = [
    STOP_ITEM_1B_LINE_PAT,
    STOP_ITEM_2_LINE_PAT,
    STOP_ITEM_NEXT_LINE_PAT,
    STOP_ITEM_HIGH_LINE_PAT,
    STOP_PART_ANY_LINE_PAT,
    STOP_SIGNATURES_PAT,
    STOP_EXHIBIT_PAT,
]

# Some filers label the section only as RISK FACTORS (all caps) in the body
RISK_FACTORS_LINE_PAT = re.compile(r"(?im)^\s*risk\s+factors\s*$")

# Common Table-of-Contents artifacts
TOC_PHRASE_PAT   = re.compile(r"(?im)table\s+of\s+contents")
DOT_LEADER_PAT   = re.compile(r"\.\s*\.\s*\.|\.{5,}")
PAGE_NUM_END_PAT = re.compile(r"\s\d{1,4}\s*$")
# Lines that end with a bare page number AND are short → strong TOC signal
PAGE_NUM_ONLY_PAT = re.compile(r"^.{10,120}\s+\d{1,3}\s*$")


def extract_item_1a(text: str) -> str | None:
    """Extract Item 1A (Risk Factors) from plain text.

    Key goals:
    - Avoid grabbing Table-of-Contents hits (very common for large banks).
    - Treat ALL-CAPS `RISK FACTORS` as a valid start when Item 1A is not printed in the body.
    - Stop ONLY at real structural markers (Item 1B / Item 2 / next Item / Part II),
      not at bold/Title-Case sentences that become their own line after HTML->text.
    """

    if not text or len(text) < 2000:
        return None

    lines = text.splitlines()

    def _pos_for_line_index(i: int) -> int:
        # character offset in original `text` at the start of line i
        return sum(len(x) + 1 for x in lines[:i])

    def _is_all_caps_heading(s: str) -> bool:
        s = s.strip()
        if len(s) < 5 or len(s) > 80:
            return False
        # must contain letters
        letters = [c for c in s if c.isalpha()]
        if len(letters) < 4:
            return False
        upper = sum(1 for c in letters if c.isupper())
        # allow some punctuation like '&', '-', ','
        return (upper / len(letters)) >= 0.85

    def _looks_like_toc_line(s: str) -> bool:
        s_stripped = s.strip()
        if not s_stripped:
            return False
        if DOT_LEADER_PAT.search(s_stripped):
            return True
        # many TOC entries end with a page number
        if PAGE_NUM_END_PAT.search(s_stripped) and len(s_stripped) <= 120:
            # avoid classifying normal prose lines that end with a year etc.
            # Heuristic: if the line also contains "item" or looks like a short heading.
            if re.search(r"(?i)\bitem\b", s_stripped) or _is_all_caps_heading(s_stripped):
                return True
        return False

    def _near_toc(i: int) -> bool:
        # If "Table of Contents" appears shortly BEFORE this line,
        # we are almost certainly in the TOC block.
        lo = max(0, i - 80)
        window_before = "\n".join(lines[lo:i])
        if TOC_PHRASE_PAT.search(window_before):
            return True

        # Also reject if the surrounding lines look heavily like TOC entries
        toc_like = 0
        for j in range(max(0, i - 20), min(len(lines), i + 5)):
            if _looks_like_toc_line(lines[j]):
                toc_like += 1
        return toc_like >= 5

    def _looks_like_risk_toc(start_line: int) -> bool:
        """Detect risk-factor mini-TOC: list of headings each ending with a page number.

        Pattern: many short lines where each paragraph ends with a standalone number
        (e.g. "Credit Risk May Adversely Affect Our Business\\n48").
        This catches TOC sections that don't use dot leaders.
        """
        snippet = lines[start_line + 1 : start_line + 60]
        if not snippet:
            return False
        page_num_lines = sum(1 for ln in snippet if PAGE_NUM_ONLY_PAT.match(ln.strip()))
        # If >25% of the next 60 lines look like "heading  pagenum", it's a TOC
        return page_num_lines >= max(3, len(snippet) * 0.25)

    def _in_first_pct(i: int, pct: float = 0.10) -> bool:
        """True if line i is within the first `pct` of the document (by char offset)."""
        pos = _pos_for_line_index(i)
        return pos < len(text) * pct

    def _has_real_body_after(i: int) -> bool:
        # After the candidate header, we expect lots of lowercase prose fairly quickly.
        # (TOC blocks tend to have dots/page numbers and very little lowercase body text.)
        snippet_lines = lines[i + 1 : i + 40]
        snippet = "\n".join(snippet_lines)
        if len(snippet) < 400:
            return False

        letters = [c for c in snippet if c.isalpha()]
        if not letters:
            return False
        lower = sum(1 for c in letters if c.islower())
        # require at least some real prose
        if (lower / len(letters)) < 0.15:
            return False

        # reject if the next area still looks like TOC (many dot leaders)
        tocish = sum(1 for ln in snippet_lines if _looks_like_toc_line(ln))
        if tocish >= 5:
            return False

        return True

    def _find_start() -> int | None:
        # Strategy: scan line-by-line so we can reject TOC occurrences.
        candidates: list[int] = []

        for i, ln in enumerate(lines):
            if ITEM_1A_LINE_PAT.search(ln):
                # Skip if in the very first slice of the document — almost always TOC.
                # Most 10-Ks have cover page + TOC in the first ~8% of the text.
                if _in_first_pct(i, 0.08):
                    continue
                candidates.append(i)

        # If no explicit Item 1A line, allow ALL-CAPS "RISK FACTORS" in the body.
        if not candidates:
            for i, ln in enumerate(lines):
                if RISK_FACTORS_LINE_PAT.search(ln) and ln.strip().upper() == "RISK FACTORS":
                    if not _in_first_pct(i, 0.08):
                        candidates.append(i)

        for i in candidates:
            # Reject obvious TOC hits
            if _near_toc(i):
                continue
            if _looks_like_toc_line(lines[i]):
                continue

            # Strong requirement: next non-empty line must be either:
            #  - ALL CAPS heading (like RISK FACTORS)
            #  - or actual lowercase prose
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j >= len(lines):
                continue

            next_line = lines[j].strip()

            if not (
                _is_all_caps_heading(next_line)
                or any(c.islower() for c in next_line)
            ):
                continue

            if not _has_real_body_after(i):
                continue

            # Final check: reject risk-factor mini-TOC (headings + page numbers,
            # no dot leaders — missed by _near_toc but detectable by page-num density)
            if _looks_like_risk_toc(i):
                continue

            return _pos_for_line_index(i)

        # Fallback: last resort – pick the first non-TOC occurrence anywhere
        # but still skip if it's in the first 8% of the document
        for m in re.finditer(ITEM_1A_ANY_PAT, text):
            if m.start() >= len(text) * 0.08:
                return m.start()
        return None

    start_pos = _find_start()
    if start_pos is None:
        return None

    tail = text[start_pos:]

    # Stop at the earliest structural marker that follows Item 1A.
    # Expanded set: Item 1B, Item 2+, Part II+, Signatures, Exhibits.
    stop_positions: list[int] = []
    for pat in ALL_STOP_PATS:
        m = pat.search(tail)
        if m:
            stop_positions.append(m.start())

    chunk = tail[: min(stop_positions)].strip() if stop_positions else tail.strip()

    # Maximum length guardrail — cap at 300,000 chars (~45,000 words).
    # The longest legitimate bank Item 1A sections are ~150,000–200,000 chars;
    # anything beyond 300k almost certainly means a stop pattern failed to fire
    # and we have run into unrelated document sections.
    chunk = chunk[:300_000]

    # Minimum length guardrail — real Item 1A prose is never this short
    if len(chunk) < 3000:
        return None

    return chunk


def main():
    BASE = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE, "..", "data")

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(DATA_DIR, "cik_years_to_scrape.csv"))
    ap.add_argument("--out_dir", default=os.path.join(DATA_DIR, "10k_item1a"))
    ap.add_argument(
        "--user_agent",
        default="Magnus Ellegaard (Copenhagen Business School, MSc) mach21ax@student.cbs.dk",
        help="Required by SEC.",
    )
    ap.add_argument("--sleep", type=float, default=0.2, help="Seconds between SEC requests (increase if 429).")
    ap.add_argument("--max_rows", type=int, default=0, help="For testing: limit number of cik-year rows (0 = all).")
    ap.add_argument("--resume", action="store_true", help="Skip if output already exists.")
    ap.add_argument("--reextract_only", action="store_true", help="Re-extract Item 1A from existing raw HTML files only (no SEC calls).")
    ap.add_argument("--test_cik", default="", help="If set, test only this CIK (digits or not) and ignore input file")
    ap.add_argument("--test_year", type=int, default=0, help="If set with --test_cik, test only this year")
    args = ap.parse_args()

    # Initialize SEC API (clean official interface)
    api = SecEdgarApi(user_agent=args.user_agent)

    # Optional quick test without reading the full input file
    if args.test_cik:
        tcik = re.sub(r"\D", "", str(args.test_cik))
        tcik = str(int(tcik)).zfill(10)
        tyear = int(args.test_year) if args.test_year else 2006

        print(f"[TEST] Fetching ALL filings via SEC API for {tcik}")
        filings = api.get_company_submissions(tcik)

        filing = None

        # --- Helper to scan a filings block ---
        def scan_block(block):
            forms = block.get("form", [])
            accession = block.get("accessionNumber", [])
            primary_docs = block.get("primaryDocument", [])
            filing_dates = block.get("filingDate", [])
            report_dates = block.get("reportDate", [])

            for f, acc, doc, fdate, rdate in zip(
                forms, accession, primary_docs, filing_dates, report_dates
            ):
                # Match 10-K / 10-KA
                if not f.startswith("10-K"):
                    continue

                # Prefer reportDate (fiscal year end) if available
                if rdate and rdate.startswith(str(tyear)):
                    return {
                        "form": f,
                        "accessionNumber": acc,
                        "primaryDocument": doc,
                        "filingDate": fdate,
                    }

                # Fallback: allow filingDate in following year (e.g. FY2008 filed in 2009)
                if fdate.startswith(str(tyear)) or fdate.startswith(str(tyear + 1)):
                    return {
                        "form": f,
                        "accessionNumber": acc,
                        "primaryDocument": doc,
                        "filingDate": fdate,
                    }

            return None

        # 1) Check recent filings
        recent = filings.get("filings", {}).get("recent", {})
        filing = scan_block(recent)

        # 2) If not found, check historical submission files
        if filing is None:
            files = filings.get("filings", {}).get("files", [])
            for fmeta in files:
                fname = fmeta.get("name")
                if not fname:
                    continue
                hist_url = f"https://data.sec.gov/submissions/{fname}"
                print(f"[TEST] Checking historical file: {hist_url}")
                hist_json = sec_get(hist_url, args.user_agent, sleep_s=args.sleep).json()

                # DEBUG: inspect structure of historical file
                print("[DEBUG] hist_json keys:", list(hist_json.keys())[:10])
                if "form" in hist_json:
                    print("[DEBUG] First 5 forms:", hist_json.get("form", [])[:5])
                if "filings" in hist_json:
                    print("[DEBUG] hist_json has nested filings key")

                # Historical files store arrays at root level (not under filings.recent)
                filing = scan_block(hist_json)
                if filing:
                    break

        print(f"[TEST] Filing found: {filing}")

        if filing is None:
            print("[TEST] No 10-K found for that year.")
            return

        acc = filing["accessionNumber"]
        doc = filing["primaryDocument"]
        cik_nolead = str(int(tcik))
        acc_nodash = acc.replace("-", "")
        test_url = SEC_ARCHIVES.format(cik_nolead=cik_nolead, acc_no_nodash=acc_nodash, doc=doc)
        print("[TEST] Downloading primary document:", test_url)
        html = sec_get(test_url, args.user_agent, sleep_s=args.sleep).text
        print("[TEST] Download ok. First 200 chars:\n", html[:200])
        return

    # Read with explicit dtypes so CIK is not parsed as float (avoids values like '00003906.0')
    df = pd.read_csv(args.input, dtype={"CIK": str, "YEAR": int})
    df.columns = [c.upper() for c in df.columns]
    assert {"CIK", "YEAR"}.issubset(df.columns), "Input must contain CIK and YEAR"

    # Clean CIK values: keep digits only (removes a trailing '.0' etc.) and left-pad to 10 digits
    df["CIK"] = (
        df["CIK"].astype(str)
        .str.strip()
        .str.replace(r"\D", "", regex=True)
        .replace({"": pd.NA})
    )
    df = df.dropna(subset=["CIK"]).copy()
    df["CIK"] = df["CIK"].apply(lambda x: str(int(x)).zfill(10))
    df = df.drop_duplicates(subset=["CIK", "YEAR"]).copy()
    total_rows = len(df)
    print(f"Starting download for {total_rows} CIK-YEAR observations...")

    # Defensive: ensure no duplicate CIK-YEAR rows (should already be clean)
    df = df.drop_duplicates(subset=["CIK", "YEAR"]).copy()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    txt_dir = out_dir / "item1a"
    raw_dir.mkdir(parents=True, exist_ok=True)
    txt_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Re-extraction mode: only process existing raw HTML files
    # ------------------------------------------------------------
    if args.reextract_only:
        print("Running in RE-EXTRACTION mode (no SEC downloads)...")

        raw_files = list(raw_dir.glob("*_primary.html"))
        print(f"Found {len(raw_files)} raw HTML files.")

        reextract_log = []

        for idx, raw_file in enumerate(raw_files):
            try:
                name = raw_file.stem  # e.g., 0000123456_2018_primary
                parts = name.split("_")
                cik = parts[0]
                year = int(parts[1])

                out_txt = txt_dir / f"{cik}_{year}_item1a.txt"

                # Skip if already exists
                if out_txt.exists():
                    continue

                html = raw_file.read_text(encoding="utf-8", errors="ignore")
                text = html_to_text(html)
                item1a = extract_item_1a(text)

                if item1a:
                    out_txt.write_text(item1a, encoding="utf-8")
                    status = "ok_reextract"
                else:
                    status = "still_no_item1a"

                reextract_log.append({
                    "cik": cik,
                    "year": year,
                    "status": status
                })

                if (idx + 1) % 100 == 0:
                    print(f"Re-extracted {idx+1}/{len(raw_files)}")

            except Exception as e:
                reextract_log.append({
                    "cik": cik if "cik" in locals() else None,
                    "year": year if "year" in locals() else None,
                    "status": f"error_{type(e).__name__}",
                    "msg": str(e)[:200]
                })

        pd.DataFrame(reextract_log).to_csv(out_dir / "reextract_log.csv", index=False)
        print("Re-extraction complete. Log saved to:", out_dir / "reextract_log.csv")
        return

    # Limit rows for test runs
    if args.max_rows and args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    log_rows = []

    for idx, row in df.iterrows():
        # Extra safety: sanitize again at row level
        cik = re.sub(r"\D", "", str(row["CIK"]))
        cik = str(int(cik)).zfill(10)
        cik_used_fallback = 0
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

                # Original filings (no amendment suffix) — always preferred
                ORIGINALS = {"10-K", "10-KSB", "10-KT", "10-KSB405"}
                # Amendments — only used as fallback if no original found
                AMENDMENTS = {"10-K/A", "10-KSB/A", "10-KT/A", "10-KSB405/A"}

                rows = list(zip(forms, accession, primary_docs, filing_dates, report_dates))

                def _year_match(fdate, rdate):
                    """True if this filing covers fiscal year `year`."""
                    if rdate and rdate.startswith(str(year)):
                        return True
                    if fdate and (fdate.startswith(str(year)) or fdate.startswith(str(year + 1))):
                        return True
                    return False

                def _make_result(f, acc, doc, fdate):
                    return {"form": f, "accessionNumber": acc, "primaryDocument": doc, "filingDate": fdate}

                # Pass 1: original 10-K matching the target year
                for f, acc, doc, fdate, rdate in rows:
                    if f in ORIGINALS and _year_match(fdate, rdate):
                        return _make_result(f, acc, doc, fdate)

                # Pass 2: amendment only if no original found
                for f, acc, doc, fdate, rdate in rows:
                    if f in AMENDMENTS and _year_match(fdate, rdate):
                        return _make_result(f, acc, doc, fdate)

                return None

            # 1) Check recent
            recent = filings.get("filings", {}).get("recent", {})
            filing = scan_block(recent)

            # 2) Check historical files if needed
            if filing is None:
                files = filings.get("filings", {}).get("files", [])
                for fmeta in files:
                    fname = fmeta.get("name")
                    if not fname:
                        continue
                    hist_url = f"https://data.sec.gov/submissions/{fname}"
                    hist_json = sec_get(hist_url, args.user_agent, sleep_s=args.sleep).json()
                    # Historical files store arrays at root level
                    filing = scan_block(hist_json)
                    if filing:
                        break

            if filing is None:
                log_rows.append({
                    "cik": cik,
                    "year": year,
                    "status": "no_10k_found",
                    "cik_fallback": cik_used_fallback
                })
                continue

            acc = filing["accessionNumber"]
            doc = filing["primaryDocument"]

            cik_nolead = str(int(cik))
            acc_nodash = acc.replace("-", "")
            url = SEC_ARCHIVES.format(cik_nolead=cik_nolead, acc_no_nodash=acc_nodash, doc=doc)

            # 3) Download primary doc (usually HTML)
            html = sec_get(url, args.user_agent, sleep_s=args.sleep).text
            out_raw.write_text(html, encoding="utf-8")

            # 4) Convert to text and extract Item 1A
            text = html_to_text(html)
            item1a = extract_item_1a(text)

            if item1a is None:
                # Save full text for debugging if extraction fails
                (raw_dir / f"{cik}_{year}_fulltext.txt").write_text(text, encoding="utf-8")
                log_rows.append({"cik": cik, "year": year, "status": "downloaded_no_item1a", "form": filing["form"], "cik_fallback": cik_used_fallback})
                continue

            out_txt.write_text(item1a, encoding="utf-8")
            log_rows.append({"cik": cik, "year": year, "status": "ok", "form": filing["form"], "filingDate": filing["filingDate"], "cik_fallback": cik_used_fallback})

        except requests.HTTPError as e:
            code = getattr(e.response, "status_code", None)
            log_rows.append({
                "cik": cik,
                "year": year,
                "status": f"http_error_{code}",
                "last_url": url if 'url' in locals() else sub_url,
                "cik_fallback": cik_used_fallback,
            })
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout):
            log_rows.append({
                "cik": cik,
                "year": year,
                "status": "timeout",
                "last_url": url if 'url' in locals() else sub_url,
                "cik_fallback": cik_used_fallback,
            })
            continue
        except Exception as e:
            log_rows.append({"cik": cik, "year": year, "status": f"error_{type(e).__name__}", "msg": str(e)[:200]})

        # Persist progress occasionally
        if (idx + 1) % 100 == 0:
            pd.DataFrame(log_rows).to_csv(out_dir / "download_log.csv", index=False)
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx+1}/{total_rows}")

    pd.DataFrame(log_rows).to_csv(out_dir / "download_log.csv", index=False)
    print("Done. Log saved to:", out_dir / "download_log.csv")
    print("Item 1A files in:", txt_dir)


if __name__ == "__main__":
    main()