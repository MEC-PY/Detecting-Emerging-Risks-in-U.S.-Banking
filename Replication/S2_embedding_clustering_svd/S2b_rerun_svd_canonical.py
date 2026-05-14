# Re-run SVD on the canonical 7,384 bank-year regression sample only.
# Filters to main_regression_sample_keys.csv before building doc-word matrix
# to avoid look-ahead leak from embedding-only docs not in the regression.

import os
import sys
import re
import ast
import numpy as np
import pandas as pd
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_DIR = os.path.join(BASE, "..", "data", "nltk_data")
SAMPLE_KEYS_PATH = os.path.join(BASE, "..", "data", "main_regression_sample_keys.csv")

import nltk
nltk.data.path = [NLTK_DATA_DIR]

sys.path.insert(0, os.path.join(BASE, "..", "lib"))

from engine import clean_and_normalize_text, calculate_word_frequencies
from TextualFactors import (
    TextualFactors,
    transfer_document_topics,
    transfer_topic_words,
    transfer_sigular_values,
    transfer_topic_importances,
)

import spacy

# Settings (must match S2a exactly)
item1a_folder = os.path.join(BASE, "..", "data", "10k_item1a_v2", "item1a")
OUT_DIR = os.path.join(BASE, "..", "data", "outputs_textual_factors_v2")

TOKEN_MIN_LEN = 3
PEAK_YEAR_MIN_DF = 30
MAX_DF_RATIO = 0.80
USE_BIGRAMS = True
BIGRAM_MIN_COUNT = 300
N_TOPICS_PER_CLUSTER = 1

_CACHED_BIGRAM_SET = None
_SPACY_STOPWORDS = None

EXTRA_DROP_WORDS = {
    "annual", "report", "reports", "group", "page", "pages", "section", "chapter",
    "table", "tables", "figure", "figures", "statement", "statements",
    "introduction", "overview", "note", "notes", "euro",
    "company", "companies", "including", "include", "future",
    "certain", "applicable", "material",
    "limited", "ltd", "inc", "corp", "corporation", "llc", "plc", "ab", "asa", "as",
    "generally", "primarily", "substantially", "significantly", "particularly",
    "previously", "currently", "typically", "frequently", "potentially",
    "separately", "collectively", "increasingly", "directly", "indirectly",
    "effectively", "subsequently", "additionally", "accordingly", "successfully",
    "periodically", "adequately", "heavily", "relatively", "consequently",
    "historically", "furthermore", "recently", "fully", "negatively",
    "materially", "adversely",
    "ability", "able", "unable", "depend", "depends", "dependent",
    "result", "resulted", "resulting", "cause", "caused",
    "provide", "provided", "provides", "providing",
    "require", "required", "requires", "continue", "continued", "continuing",
    "increase", "increased", "increasing", "decrease", "reduce", "reduced", "reducing",
    "maintain", "obtain", "comply", "conduct", "occur",
    "expected", "based", "related", "relating",
    "described", "designed", "intended", "involve", "involves", "involved",
    "includes", "included", "established", "implement", "implemented",
    "considered", "determined", "anticipated",
    "offered", "received", "taken", "given", "adopted", "enacted",
    "proposed", "identified", "known", "issued", "performed",
    "additional", "general", "common", "existing", "current",
    "particular", "specific", "similar", "recent", "necessary",
    "appropriate", "effective", "sufficient", "adequate", "actual",
    "possible", "reasonable", "numerous", "various", "different",
    "broad", "wide", "important", "inherent", "overall", "ongoing",
    "prior", "outside", "present", "extensive", "favorable", "difficult",
    "time", "period", "date", "year", "quarter", "monthly",
    "december", "january", "quarterly",
    "number", "level", "degree", "size", "range",
    "million", "billion", "thousand", "trillion", "percent", "percentage",
    "average", "approximately", "minimum", "maximum", "total", "excess",
    "process", "course", "order", "place", "case", "form",
    "need", "meet", "face", "seek", "lack", "turn",
    "attract", "retain", "compete", "prevent", "protect",
    "achieve", "develop", "create", "expand", "grow", "generate",
    "apply", "evaluate", "satisfy", "address", "mitigate",
    "restrict", "monitor", "ensure", "predict",
    "management", "personnel", "board", "executive",
    "review", "discussion", "assessment", "evaluation",
    "reporting", "disclosure", "accounting",
    "program", "plan", "strategy", "structure", "accordance",
    "agreement", "arrangement", "approval", "authority",
    "oversight", "attention", "confidence",
    "performance", "quality", "success", "failure",
    "uncertainty", "competitive", "sensitive",
    "business", "item", "value", "cost", "expense", "benefit",
    "payment", "sale", "return", "service", "product", "customer",
    "environment",
    "government", "governmental", "political", "legislative", "legislation",
}

ADVERB_EXCEPTIONS = {
    "supply", "rally", "tally", "bully", "ally",
    "family", "assembly", "anomaly",
    "quarterly", "daily", "monthly", "yearly", "weekly",
    "orderly", "disorderly",
}


def _load_spacy_stopwords():
    global _SPACY_STOPWORDS
    if _SPACY_STOPWORDS is not None:
        return _SPACY_STOPWORDS
    try:
        nlp = spacy.blank("en")
        _SPACY_STOPWORDS = set(nlp.Defaults.stop_words)
        print(f"Loaded {len(_SPACY_STOPWORDS)} spaCy stopwords")
    except Exception as e:
        print(f"Warning: could not load spaCy stopwords: {e}")
        _SPACY_STOPWORDS = set()
    return _SPACY_STOPWORDS


def load_item1a_documents(folder):
    texts, sources = [], []
    for path, dirs, files in os.walk(folder):
        txt_files = [f for f in files if f.lower().endswith(".txt")]
        for fname in sorted(txt_files):
            full_path = os.path.join(path, fname)
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()
            if text:
                texts.append(text)
                sources.append(fname)
    return texts, sources


def build_document_dataframe(report_texts, report_sources):
    df = pd.DataFrame({"file": report_sources, "content": report_texts})
    if df.empty:
        return pd.DataFrame(columns=["file", "content", "year", "cik", "document"])
    pattern = r"(?P<cik>\d+)_(?P<year>\d{4})_item1a\.txt$"
    extracted = df["file"].str.extract(pattern)
    df["cik"] = extracted["cik"]
    df["year"] = pd.to_numeric(extracted["year"], errors="coerce").astype("Int64")
    df = df.sort_values(["year", "cik", "file"]).reset_index(drop=True)
    df["document"] = np.arange(len(df))
    return df


def _basic_token_filter(tokens):
    out = []
    stopwords = _load_spacy_stopwords()
    drop_words = EXTRA_DROP_WORDS | stopwords
    for t in tokens:
        if not isinstance(t, str):
            continue
        t = t.strip().lower()
        if not t or len(t) < TOKEN_MIN_LEN:
            continue
        if not all(part.isalpha() for part in t.split("_")):
            continue
        if t in drop_words:
            continue
        if t.endswith("ly") and len(t) > 4 and t not in ADVERB_EXCEPTIONS:
            continue
        out.append(t)
    return out


def _df_filter_tokens(df, tokens_col, min_df, max_df_ratio=None, year_col=None):
    df = df.copy()
    n_docs = len(df)
    corpus_df = Counter()
    for toks in df[tokens_col]:
        corpus_df.update(set(toks))
    max_df = max(1, int(np.floor(max_df_ratio * n_docs))) if max_df_ratio else None

    if year_col is not None and year_col in df.columns:
        year_df_counters = {}
        for year, group in df.groupby(year_col):
            c = Counter()
            for toks in group[tokens_col]:
                c.update(set(toks))
            year_df_counters[year] = c
        all_words = set(corpus_df.keys())
        peak_df = {w: max(year_df_counters[y].get(w, 0) for y in year_df_counters) for w in all_words}
        allowed = {w for w, peak in peak_df.items() if peak >= min_df and (max_df is None or corpus_df[w] <= max_df)}
    else:
        allowed = {tok for tok, dfi in corpus_df.items() if dfi >= min_df and (max_df is None or dfi <= max_df)}

    print(f"DF filter: keeping {len(allowed)} tokens")
    df[tokens_col] = df[tokens_col].apply(lambda toks: [t for t in toks if t in allowed])
    return df


def _learn_frequent_bigrams(docs_tokens, min_count):
    bigram_counts = Counter()
    for toks in docs_tokens:
        if not toks or len(toks) < 2:
            continue
        for a, b in zip(toks, toks[1:]):
            if "_" in a or "_" in b:
                continue
            bigram_counts[(a, b)] += 1
    return {bg for bg, c in bigram_counts.items() if c >= int(min_count)}


def _augment_with_bigrams(toks, bigram_set):
    if not toks or len(toks) < 2:
        return toks
    out = []
    for a, b in zip(toks, toks[1:]):
        if (a, b) in bigram_set:
            out.append(f"{a}_{b}")
        out.append(a)
    out.append(toks[-1])
    return out


def _rebuild_word_freq_from_tokens(df, tokens_col="tokens"):
    df = df.copy()
    df["word_freq"] = df[tokens_col].apply(lambda toks: Counter(toks) if isinstance(toks, list) else Counter())
    return df


def preprocess_text_and_tokens(df, text_col="content", tokens_col="tokens",
                                min_df=PEAK_YEAR_MIN_DF, max_df_ratio=MAX_DF_RATIO, year_col=None):
    global _CACHED_BIGRAM_SET
    df = df.copy()
    df = clean_and_normalize_text(df, column_name=text_col)
    df = calculate_word_frequencies(df, text_column=text_col)
    if "tokens_raw" not in df.columns:
        df["tokens_raw"] = df[tokens_col].apply(lambda x: list(x) if isinstance(x, list) else [])
    df[tokens_col] = df[tokens_col].apply(_basic_token_filter)

    if USE_BIGRAMS:
        if _CACHED_BIGRAM_SET is None:
            _CACHED_BIGRAM_SET = _learn_frequent_bigrams(df[tokens_col].tolist(), min_count=BIGRAM_MIN_COUNT)
            print(f"Learned {len(_CACHED_BIGRAM_SET)} bigrams with count >= {BIGRAM_MIN_COUNT}")
        df[tokens_col] = df[tokens_col].apply(lambda toks: _augment_with_bigrams(toks, _CACHED_BIGRAM_SET))

    stopwords = _load_spacy_stopwords()
    _all_drop = EXTRA_DROP_WORDS | stopwords

    def _is_boilerplate_bigram(token):
        if "_" not in token:
            return False
        return all(p in _all_drop for p in token.split("_"))

    df[tokens_col] = df[tokens_col].apply(lambda toks: [t for t in toks if not _is_boilerplate_bigram(t)])

    df = _df_filter_tokens(df, tokens_col=tokens_col, min_df=min_df, max_df_ratio=max_df_ratio, year_col=year_col)
    df = _rebuild_word_freq_from_tokens(df, tokens_col=tokens_col)
    return df


def load_cached_vocab():
    df = pd.read_csv(os.path.join(OUT_DIR, "vocab.csv"))
    vocab = df["word"].tolist()
    print(f"Loaded cached vocab: {len(vocab)} words")
    return vocab


def reconstruct_word_cluster_map():
    tw = pd.read_csv(os.path.join(OUT_DIR, "topics_words.csv"))
    print(f"Loaded topics_words.csv: {len(tw)} clusters")

    word_to_cluster = {}
    for _, row in tw.iterrows():
        cluster_id = int(row["topic"])
        dist_str = str(row["topic_distribution"])
        dist_str = dist_str.replace("np.float64(", "").replace(")", "")
        try:
            word_dict = ast.literal_eval(dist_str)
        except (ValueError, SyntaxError):
            words = re.findall(r"'([^']+)':", dist_str)
            word_dict = {w: 0.0 for w in words}
        for word in word_dict.keys():
            word_to_cluster[word] = cluster_id

    print(f"Reconstructed: {len(word_to_cluster)} words across {tw['topic'].nunique()} clusters")
    return word_to_cluster


def build_document_word_data(df_docs, vocab):
    vocab_set = set(vocab)
    rows = []
    for doc_id, word_counts in zip(df_docs["document"], df_docs["word_freq"]):
        for word, count in word_counts.items():
            if word in vocab_set:
                rows.append({"document": doc_id, "ngram": word, "count": int(count)})
    doc_word_df = pd.DataFrame(rows)
    print(f"document_word_data: {doc_word_df.shape[0]:,} rows, {doc_word_df['document'].nunique()} docs")
    return doc_word_df


def build_word_cluster_data(vocab, word_cluster_map):
    rows = [{"ngram": w, "sequential_cluster": word_cluster_map[w]}
            for w in vocab if w in word_cluster_map]
    df = pd.DataFrame(rows)
    print(f"word_cluster_data: {len(df)} words → {df['sequential_cluster'].nunique()} clusters")
    return df


def filter_to_canonical_sample(df_docs, sample_keys_path):
    """
    Restrict df_docs to the (cik, year) pairs in main_regression_sample_keys.csv.
    Returns a new df_docs with CONTIGUOUS document IDs (0..N-1).
    """
    keys = pd.read_csv(sample_keys_path)
    keys["cik"] = keys["cik"].astype(int)
    keys["year"] = keys["loading_year"].astype(int)
    key_set = set(zip(keys["cik"].tolist(), keys["year"].tolist()))
    print(f"Canonical sample: {len(key_set)} unique (cik, year) pairs")

    df_docs = df_docs.copy()
    df_docs["cik_int"] = df_docs["cik"].astype(int)
    df_docs["year_int"] = df_docs["year"].astype(int)
    before = len(df_docs)
    df_docs = df_docs[
        df_docs.apply(lambda r: (r["cik_int"], r["year_int"]) in key_set, axis=1)
    ].reset_index(drop=True)
    print(f"Filtered docs: {before:,} → {len(df_docs):,}")

    df_docs = df_docs.drop(columns=["cik_int", "year_int"])
    df_docs["document"] = np.arange(len(df_docs))
    return df_docs


def main():
    print("=" * 60)
    print("SVD RE-RUN (CANONICAL 7,384 SAMPLE)")
    print("=" * 60)

    # Step 1: Load cached vocab + clusters (unchanged from full run)
    print("\n=== STEP 1: Load cached vocab + cluster map ===")
    vocab = load_cached_vocab()
    word_cluster_map = reconstruct_word_cluster_map()

    # Step 2: Re-parse 10-K text files (same preprocessing as Main_SEB.py)
    print("\n=== STEP 2: Load and preprocess 10-K files ===")
    report_texts, report_sources = load_item1a_documents(item1a_folder)
    print(f"Loaded {len(report_texts)} documents")

    df_docs = build_document_dataframe(report_texts, report_sources)
    print(f"Document dataframe: {len(df_docs)} docs")

    df_docs = preprocess_text_and_tokens(df_docs, text_col="content", tokens_col="tokens")
    df_docs = df_docs[df_docs["tokens"].apply(len) >= 5].copy()
    print(f"After token filter: {len(df_docs)} docs")

    # Step 2b: Restrict to canonical 7,384 sample
    print("\n=== STEP 2b: Filter to canonical sample ===")
    df_docs = filter_to_canonical_sample(df_docs, SAMPLE_KEYS_PATH)

    # Step 3: Build tables using cached vocab + cluster map
    print("\n=== STEP 3: Build document-word + word-cluster tables ===")
    document_word_data = build_document_word_data(df_docs, vocab)
    word_cluster_data = build_word_cluster_data(vocab, word_cluster_map)

    # Step 4: SVD with L2 normalization
    print("\n=== STEP 4: Compute Textual Factors (SVD with L2 norm) ===")
    tf_model = TextualFactors(
        document_word_data=document_word_data,
        word_cluster_data=word_cluster_data,
    )

    (
        first_doc_topics, second_doc_topics,
        first_topics_words, second_topics_words,
        singular_values, topic_importances,
    ) = tf_model.lsa_topics(
        cluster_type="sequential_cluster",
        n_topics=N_TOPICS_PER_CLUSTER,
    )

    first_doc_topics_df = transfer_document_topics(first_doc_topics)
    topics_words_df = transfer_topic_words(first_topics_words)
    singular_values_df = transfer_sigular_values(singular_values)
    topic_importances_df = transfer_topic_importances(topic_importances)

    # Step 5: Save
    print("\n=== STEP 5: Save canonical outputs ===")
    os.makedirs(OUT_DIR, exist_ok=True)

    df_docs[["document", "year", "cik", "file"]].to_csv(
        os.path.join(OUT_DIR, "document_metadata.csv"), index=False)
    first_doc_topics_df.to_csv(
        os.path.join(OUT_DIR, "first_doc_topics.csv"), index=False)
    topics_words_df.to_csv(
        os.path.join(OUT_DIR, "topics_words.csv"), index=False)
    singular_values_df.to_csv(
        os.path.join(OUT_DIR, "singular_values.csv"), index=False)
    topic_importances_df.to_csv(
        os.path.join(OUT_DIR, "topic_importances.csv"), index=False)

    print("\n" + "=" * 60)
    print("CANONICAL SVD RE-RUN COMPLETE")
    print(f"  docs: {len(df_docs):,}")
    print(f"  clusters: {first_doc_topics_df.shape[1] - 1}")
    print(f"  outputs in: {OUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
