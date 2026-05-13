# Re-extract Item 1A from raw HTML where initial extraction failed.
# Fixes iXBRL non-breaking space issue (\xa0 in headers like "Item 1A")
# that caused regex misses. No SEC downloads, just re-parses local HTML.

import os
import re
import glob
import pandas as pd
from pathlib import Path

# Import the extraction function from 04
# We'll redefine html_to_text with the nbsp fix built in

try:
    from bs4 import BeautifulSoup
    from bs4 import XMLParsedAsHTMLWarning
    import warnings
    warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
except Exception:
    BeautifulSoup = None


def html_to_text_fixed(html: str) -> str:
    """Convert HTML to text with non-breaking space normalization."""
    if BeautifulSoup is None:
        txt = re.sub(r"<[^>]+>", " ", html)
        txt = re.sub(r"\s+", " ", txt)
        # Normalize non-breaking spaces
        txt = txt.replace("\xa0", " ")
        txt = txt.replace("\u00a0", " ")
        return txt.strip()

    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # normalize non-breaking spaces (the whole point of this script)
    text = text.replace("\xa0", " ")
    text = text.replace("\u00a0", " ")
    # Also normalize other Unicode whitespace that iXBRL might use
    text = text.replace("\u2003", " ")  # em space
    text = text.replace("\u2002", " ")  # en space
    text = text.replace("\u200b", "")   # zero-width space
    text = text.replace("\ufeff", "")   # BOM

    return text.strip()


# Item 1A extraction (same patterns as S1c, works now because nbsp is normalized)

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



def main():
    BASE = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE, "..", "data")
    raw_dir = Path(DATA_DIR) / "10k_item1a_v2" / "raw"
    txt_dir = Path(DATA_DIR) / "10k_item1a_v2" / "item1a"
    txt_dir.mkdir(parents=True, exist_ok=True)

    # Find all raw HTML files
    raw_files = sorted(raw_dir.glob("*_primary.html"))
    print(f"Found {len(raw_files)} raw HTML files")

    # Find which ones DON'T have extracted text yet
    missing = []
    for rf in raw_files:
        name = rf.stem  # e.g., 0000019617_2022_primary
        parts = name.split("_")
        cik = parts[0]
        year = int(parts[1])
        out_txt = txt_dir / f"{cik}_{year}_item1a.txt"
        if not out_txt.exists():
            missing.append((rf, cik, year, out_txt))

    print(f"Missing extracted text: {len(missing)} files")
    if not missing:
        print("Nothing to re-extract!")
        return

    # Count by year
    from collections import Counter
    year_counts = Counter(yr for _, _, yr, _ in missing)
    print("\nMissing by year:")
    for yr in sorted(year_counts):
        print(f"  {yr}: {year_counts[yr]}")

    # Re-extract with nbsp fix
    log = []
    recovered = 0
    still_failed = 0

    for idx, (rf, cik, year, out_txt) in enumerate(missing):
        try:
            html = rf.read_text(encoding="utf-8", errors="ignore")
            text = html_to_text_fixed(html)  # ← THE FIX: normalizes \xa0
            item1a = extract_item_1a(text)

            if item1a:
                out_txt.write_text(item1a, encoding="utf-8")
                log.append({"cik": cik, "year": year, "status": "recovered",
                            "chars": len(item1a), "words": len(item1a.split())})
                recovered += 1
            else:
                log.append({"cik": cik, "year": year, "status": "still_failed"})
                still_failed += 1

        except Exception as e:
            log.append({"cik": cik, "year": year, "status": f"error_{type(e).__name__}",
                         "msg": str(e)[:200]})
            still_failed += 1

        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx+1}/{len(missing)} "
                  f"(recovered: {recovered}, still failed: {still_failed})")

    # Summary
    log_df = pd.DataFrame(log)
    print(f"\n{'='*60}")
    print(f"RE-EXTRACTION SUMMARY (with non-breaking space fix)")
    print(f"{'='*60}")
    print(f"  Total attempted:  {len(missing)}")
    print(f"  Recovered:        {recovered}")
    print(f"  Still failed:     {still_failed}")
    print(f"  Recovery rate:    {recovered/len(missing):.0%}")

    print(f"\nRecovered by year:")
    rec_df = log_df[log_df["status"] == "recovered"]
    for yr in sorted(rec_df["year"].unique()):
        cnt = len(rec_df[rec_df["year"] == yr])
        print(f"  {yr}: +{cnt} filings recovered")

    # New total Item 1A files
    total_after = len(list(txt_dir.glob("*_item1a.txt")))
    print(f"\nTotal Item 1A files after re-extraction: {total_after}")

    # Save log
    log_path = raw_dir.parent / "reextract_nbsp_fix_log.csv"
    log_df.to_csv(log_path, index=False)
    print(f"Log saved: {log_path}")

    # Show new per-year counts
    print(f"\nItem 1A files per year (after fix):")
    all_files = list(txt_dir.glob("*_item1a.txt"))
    year_counts_new = Counter()
    for f in all_files:
        parts = f.stem.split("_")
        yr = int(parts[1])
        year_counts_new[yr] += 1
    for yr in sorted(year_counts_new):
        print(f"  {yr}: {year_counts_new[yr]}")


if __name__ == "__main__":
    main()
