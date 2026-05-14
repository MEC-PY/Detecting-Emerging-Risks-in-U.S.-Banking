import os
import sys
import re
import nltk
import spacy
from itertools import chain
from collections import Counter

import numpy as np
import pandas as pd
from openai import OpenAI

# OpenAI setup
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if OPENAI_API_KEY is None:
    raise RuntimeError(
        "Missing OPENAI_API_KEY environment variable. "
        "Set it in Run → Edit Configurations → Environment variables."
    )

OPENAI_API_KEY = OPENAI_API_KEY.strip().strip('"').strip("'")

bad_chars = [c for c in OPENAI_API_KEY if ord(c) > 127]
if bad_chars:
    raise RuntimeError(
        f"OPENAI_API_KEY contains non-ASCII characters: {bad_chars}. "
        f"Current preview: {repr(OPENAI_API_KEY[:50])}"
    )

client = OpenAI(api_key=OPENAI_API_KEY)

# Path setup
BASE = os.path.dirname(os.path.abspath(__file__))
NLTK_DATA_DIR = os.path.join(BASE, "..", "data", "nltk_data")

nltk.data.path = [NLTK_DATA_DIR]

sys.path.insert(0, os.path.join(BASE, "..", "lib"))

from engine import (
    clean_and_normalize_text,
    calculate_word_frequencies,
)

from TextualFactors import (
    EmbeddingCluster,
    NeighborFinder,
    TextualFactors,
    transfer_document_topics,
    transfer_topic_words,
    transfer_sigular_values,
    transfer_topic_importances,
)

# Settings

# Use Item 1A TXT files from v2 corpus (fresh extraction)
item1a_folder = os.path.join(BASE, "..", "data", "10k_item1a_v2", "item1a")
print("Item 1A folder:", item1a_folder)

# Outputs folder — v2 run (old outputs_textual_factors left untouched)
OUT_FOLDER_SEB = os.path.join(BASE, "..", "data", "outputs_textual_factors_v2")
os.makedirs(OUT_FOLDER_SEB, exist_ok=True)

# --- Token / vocab filtering settings ---
TOKEN_MIN_LEN = 3
PEAK_YEAR_MIN_DF = 30 # keep word if it appears in >= this many docs in its peak year
MAX_DF_RATIO = 0.80

# --- Bigram settings ---
USE_BIGRAMS = True
BIGRAM_MIN_COUNT = 300

# Cache learned bigrams so we only learn them once per run
_CACHED_BIGRAM_SET = None

_SPACY_STOPWORDS = None

EXTRA_DROP_WORDS = {
    # --- Original generic boilerplate ---
    "annual", "report", "reports", "group", "page", "pages", "section", "chapter",
    "table", "tables", "figure", "figures", "statement", "statements",
    "introduction", "overview", "note", "notes", "euro",

    # --- Generic SEC / Item 1A boilerplate ---
    "company", "companies", "including", "include", "future",
    "certain", "applicable", "material",

    # --- Legal entities ---
    "limited", "ltd", "inc", "corp", "corporation", "llc", "plc", "ab", "asa", "as",

    # === EXPANDED BOILERPLATE (v2): high-DF generic words identified via ===
    # === corpus-wide document-frequency analysis (all >50% DF, none risk-specific) ===

    # Generic adverbs (DF 60-98%, add no risk content to embeddings)
    "generally", "primarily", "substantially", "significantly", "particularly",
    "previously", "currently", "typically", "frequently", "potentially",
    "separately", "collectively", "increasingly", "directly", "indirectly",
    "effectively", "subsequently", "additionally", "accordingly", "successfully",
    "periodically", "adequately", "heavily", "relatively", "consequently",
    "historically", "furthermore", "recently", "fully", "negatively",
    "materially", "adversely",

    # Generic verbs / verb forms (appear in 60-99% of filings)
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

    # Generic adjectives / determiners (DF 55-96%)
    "additional", "general", "common", "existing", "current",
    "particular", "specific", "similar", "recent", "necessary",
    "appropriate", "effective", "sufficient", "adequate", "actual",
    "possible", "reasonable", "numerous", "various", "different",
    "broad", "wide", "important", "inherent", "overall", "ongoing",
    "prior", "outside", "present", "extensive", "favorable", "difficult",

    # Quantity / measurement / time (not risk-type-specific)
    "time", "period", "date", "year", "quarter", "monthly",
    "december", "january", "quarterly",
    "number", "level", "degree", "size", "range",
    "million", "billion", "thousand", "trillion", "percent", "percentage",
    "average", "approximately", "minimum", "maximum", "total", "excess",

    # Generic process / action words
    "process", "course", "order", "place", "case", "form",
    "need", "meet", "face", "seek", "lack", "turn",
    "attract", "retain", "compete", "prevent", "protect",
    "achieve", "develop", "create", "expand", "grow", "generate",
    "apply", "evaluate", "satisfy", "address", "mitigate",
    "restrict", "monitor", "ensure", "predict",

    # Organizational / administrative filler
    "management", "personnel", "board", "executive",
    "review", "discussion", "assessment", "evaluation",
    "reporting", "disclosure", "accounting",
    "program", "plan", "strategy", "structure", "accordance",
    "agreement", "arrangement", "approval", "authority",
    "oversight", "attention", "confidence",

    # Generic state / quality
    "performance", "quality", "success", "failure",
    "uncertainty", "competitive", "sensitive",
    "business", "item", "value", "cost", "expense", "benefit",
    "payment", "sale", "return", "service", "product", "customer",
    "environment",
    "government", "governmental", "political", "legislative", "legislation",
}

# Words ending in -ly that should NOT be removed (domain-relevant)
ADVERB_EXCEPTIONS = {
    "supply", "rally", "tally", "bully", "ally",         # not adverbs
    "family", "assembly", "anomaly",                      # nouns
    "quarterly", "daily", "monthly", "yearly", "weekly",  # temporal (useful in bigrams)
    "orderly", "disorderly",                              # regulatory context
}

# LSH / clustering settings
N_BITS = 256
N_TABLES = 64
DEFAULT_NEIGHBOR_ALG = "lsh"

# Number of latent topics per cluster
N_TOPICS_PER_CLUSTER = 1


def _load_spacy_stopwords():
    """Load spaCy English stopwords once and cache them."""
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


def load_item1a_documents(item1a_folder):
    texts = []
    sources = []

    print(f"Looking for TXT files in: {item1a_folder}")

    for path, dirs, files in os.walk(item1a_folder):
        txt_files = [f for f in files if f.lower().endswith(".txt")]
        if not txt_files:
            continue

        print("Found TXT files:", txt_files)

        for fname in txt_files:
            full_path = os.path.join(path, fname)
            print(f"Processing {fname}...")

            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read().strip()

            if text:
                texts.append(text)
                sources.append(fname)

    return texts, sources


def build_document_dataframe(report_texts, report_sources):
    """
    Build one document per Item 1A TXT file.

    Expected filename format:
        <cik>_<year>_item1a.txt
    Example:
        0000039060_2006_item1a.txt
    """

    df = pd.DataFrame({
        "file": report_sources,
        "content": report_texts,
    })

    if df.empty:
        return pd.DataFrame(columns=["file", "content", "year", "cik", "document"])

    pattern = r"(?P<cik>\d+)_(?P<year>\d{4})_item1a\.txt$"
    extracted = df["file"].str.extract(pattern)

    df["cik"] = extracted["cik"]
    df["year"] = pd.to_numeric(extracted["year"], errors="coerce").astype("Int64")

    bad = df[df["cik"].isna() | df["year"].isna()]
    if not bad.empty:
        raise ValueError(
            "Could not parse CIK/year from these filenames:\n"
            + "\n".join(bad["file"].tolist())
            + "\nExpected format: <cik>_<year>_item1a.txt"
        )

    df = df.sort_values(["year", "cik", "file"]).reset_index(drop=True)
    df["document"] = np.arange(len(df))

    return df


def _basic_token_filter(tokens):
    """Remove obvious junk tokens before df-based filtering.

    Filters applied:
      1. Length >= TOKEN_MIN_LEN
      2. Alphabetic (or underscore-joined alphabetic bigrams)
      3. Not in spaCy stopwords or EXTRA_DROP_WORDS
      4. Adverb filter: remove words ending in -ly (unless in ADVERB_EXCEPTIONS)
    """
    out = []
    stopwords = _load_spacy_stopwords()
    drop_words = EXTRA_DROP_WORDS | stopwords

    for t in tokens:
        if not isinstance(t, str):
            continue

        t = t.strip().lower()
        if not t:
            continue
        if len(t) < TOKEN_MIN_LEN:
            continue

        # keep alphabetic tokens OR underscore-joined alphabetic bigrams
        if not all(part.isalpha() for part in t.split("_")):
            continue

        if t in drop_words:
            continue

        # Adverb filter: words ending in -ly are almost always generic
        # modifiers that dilute topic specificity (e.g. "respectively",
        # "accordingly", "substantially"). Exceptions preserved for
        # domain-relevant words (see ADVERB_EXCEPTIONS).
        if t.endswith("ly") and len(t) > 4 and t not in ADVERB_EXCEPTIONS:
            continue

        out.append(t)

    return out


def _df_filter_tokens(df, tokens_col, min_df, max_df_ratio=None, year_col=None):
    """Apply document-frequency filtering across the corpus.

    If year_col is provided, uses PEAK-YEAR mode: a word is kept if it appears
    in at least min_df documents in its single most active year. This preserves
    emerging-risk vocabulary that is temporally concentrated (e.g. 'subprime'
    in 2006-08) which a corpus-wide threshold would incorrectly strip out.

    MAX_DF_RATIO is always evaluated corpus-wide to remove omnipresent boilerplate.
    """
    df = df.copy()
    n_docs = len(df)

    # --- corpus-wide counter (always needed for MAX_DF_RATIO) ---
    corpus_df = Counter()
    for toks in df[tokens_col]:
        corpus_df.update(set(toks))

    max_df = max(1, int(np.floor(max_df_ratio * n_docs))) if max_df_ratio else None

    if year_col is not None and year_col in df.columns:
        # Peak-year DF: compute per-year document frequencies
        year_df_counters = {}
        for year, group in df.groupby(year_col):
            c = Counter()
            for toks in group[tokens_col]:
                c.update(set(toks))
            year_df_counters[year] = c

        all_words = set(corpus_df.keys())
        peak_df = {
            w: max(year_df_counters[y].get(w, 0) for y in year_df_counters)
            for w in all_words
        }

        allowed = {
            w for w, peak in peak_df.items()
            if peak >= min_df and (max_df is None or corpus_df[w] <= max_df)
        }
        print(
            f"Peak-year DF filter: keeping {len(allowed)} tokens "
            f"(peak annual DF >= {min_df}, corpus DF <= {max_df or 'n/a'})"
        )

    else:
        # Fallback: original corpus-wide filter
        allowed = {
            tok for tok, dfi in corpus_df.items()
            if dfi >= min_df and (max_df is None or dfi <= max_df)
        }
        print(
            f"Corpus-wide DF filter: keeping {len(allowed)} tokens "
            f"({min_df} <= df <= {max_df or 'n/a'})"
        )

    df[tokens_col] = df[tokens_col].apply(lambda toks: [t for t in toks if t in allowed])
    return df


def _learn_frequent_bigrams(docs_tokens, min_count):
    """Learn frequent bigrams from the corpus using a simple count threshold."""
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
    """
    Insert bigram tokens while keeping unigrams.
    Example:
      ["interest","rate","risk"] -> ["interest_rate","interest","rate","risk"]
    """
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
    """Ensure word_freq matches the final filtered tokens."""
    df = df.copy()
    df["word_freq"] = df[tokens_col].apply(
        lambda toks: Counter(toks) if isinstance(toks, list) else Counter()
    )
    return df


def preprocess_text_and_tokens(
    df,
    text_col="content",
    tokens_col="tokens",
    min_df=PEAK_YEAR_MIN_DF,
    max_df_ratio=MAX_DF_RATIO,
    year_col=None,
):
    """
    Canonical preprocessing step:
    1) clean text
    2) tokenize + count
    3) basic token cleanup
    4) optionally learn + insert frequent bigrams
    5) min/max document-frequency filtering
    6) rebuild word_freq from final tokens
    """

    global _CACHED_BIGRAM_SET

    df = df.copy()

    # 1) Clean / normalize
    df = clean_and_normalize_text(df, column_name=text_col)

    # 2) Tokenize + count
    df = calculate_word_frequencies(df, text_column=text_col)

    if "tokens_raw" not in df.columns:
        df["tokens_raw"] = df[tokens_col].apply(
            lambda x: list(x) if isinstance(x, list) else []
        )

    # 3) Basic cleanup
    df[tokens_col] = df[tokens_col].apply(_basic_token_filter)

    # 4) Learn + insert frequent bigrams
    if USE_BIGRAMS:
        if _CACHED_BIGRAM_SET is None:
            _CACHED_BIGRAM_SET = _learn_frequent_bigrams(
                df[tokens_col].tolist(),
                min_count=BIGRAM_MIN_COUNT,
            )
            print(
                f"Learned {len(_CACHED_BIGRAM_SET)} bigrams "
                f"with count >= {BIGRAM_MIN_COUNT}"
            )

        df[tokens_col] = df[tokens_col].apply(
            lambda toks: _augment_with_bigrams(toks, _CACHED_BIGRAM_SET)
        )

    # 4b) Remove bigrams where BOTH component words are generic boilerplate.
    #     These form clusters like "ability_achieve", "result_current",
    #     "business_result" that carry no risk-specific content.
    stopwords = _load_spacy_stopwords()
    _all_drop = EXTRA_DROP_WORDS | stopwords
    def _is_boilerplate_bigram(token):
        if "_" not in token:
            return False
        parts = token.split("_")
        return all(p in _all_drop for p in parts)

    n_before = sum(len(toks) for toks in df[tokens_col])
    df[tokens_col] = df[tokens_col].apply(
        lambda toks: [t for t in toks if not _is_boilerplate_bigram(t)]
    )
    n_after = sum(len(toks) for toks in df[tokens_col])
    print(f"Boilerplate-bigram filter removed {n_before - n_after:,} token occurrences")

    # 5) DF-based filtering
    df = _df_filter_tokens(
        df,
        tokens_col=tokens_col,
        min_df=min_df,
        max_df_ratio=max_df_ratio,
        year_col=year_col,
    )

    # 6) Rebuild word frequencies
    df = _rebuild_word_freq_from_tokens(df, tokens_col=tokens_col)

    return df


def train_openai_embeddings(df, model_name="text-embedding-3-large"):
    """
    Build word embeddings using OpenAI's embedding API.
    Trains on paragraph-level tokens.
    """
    vocab = sorted(set(chain.from_iterable(df["tokens"].tolist())))
    print(f"Vocabulary size: {len(vocab)} words")

    batch_size = 200
    embeddings = []

    for i in range(0, len(vocab), batch_size):
        batch = vocab[i:i + batch_size]

        response = client.embeddings.create(
            model=model_name,
            input=batch
        )

        if len(response.data) != len(batch):
            raise RuntimeError(
                f"Embedding count mismatch in batch {i // batch_size + 1}: "
                f"got {len(response.data)} embeddings for {len(batch)} inputs"
            )

        batch_embs = [item.embedding for item in response.data]
        embeddings.extend(batch_embs)
        print(f"Processed batch {i // batch_size + 1}")

    if len(embeddings) != len(vocab):
        raise RuntimeError(
            f"Final embedding mismatch: {len(embeddings)} embeddings "
            f"for {len(vocab)} vocab items"
        )

    embedding_matrix = np.array(embeddings, dtype=np.float32)

    out_dir = OUT_FOLDER_SEB
    os.makedirs(out_dir, exist_ok=True)

    np.save(os.path.join(out_dir, "embedding_matrix.npy"), embedding_matrix)
    pd.DataFrame({"word": vocab}).to_csv(
        os.path.join(out_dir, "vocab.csv"),
        index=False
    )

    print("Saved embedding_matrix.npy and vocab.csv to:", out_dir)

    return vocab, embedding_matrix


def cluster_words(
    embedding_matrix,
    target_cluster_size=100,
    neighbor_alg=DEFAULT_NEIGHBOR_ALG,
):
    """
    Cluster word embeddings into semantic groups.
    """

    if neighbor_alg not in {"lsh", "brute"}:
        raise ValueError(f"neighbor_alg must be 'lsh' or 'brute', got: {neighbor_alg}")

    nf = NeighborFinder(
        embedding_matrix,
        random_state=42,
        num_queries=1000,
    )

    if neighbor_alg == "lsh":
        print(
            f"Using FAISS LSH with tuned parameters: "
            f"bits={N_BITS}, tables={N_TABLES}"
        )
        nf.lsh_index = nf.create_lsh_index(N_BITS, N_TABLES)
    else:
        print("Using brute-force neighbor search (no LSH).")

    ec = EmbeddingCluster(nf, neighbor_alg=neighbor_alg)
    clusters = ec.sequentialcluster(cluster_size=target_cluster_size)

    cluster_words_map, word_cluster_map = ec.cluster_word_map(clusters)

    print(f"Number of clusters created: {len(clusters)}")

    return ec, clusters, cluster_words_map, word_cluster_map


def build_document_word_data(df, vocab):
    """
    Create long-format table:
    - document
    - ngram
    - count
    """
    rows = []
    vocab_set = set(vocab)

    for doc_id, word_counts in zip(df["document"], df["word_freq"]):
        for word, count in word_counts.items():
            if word in vocab_set:
                rows.append(
                    {
                        "document": doc_id,
                        "ngram": word,
                        "count": int(count),
                    }
                )

    doc_word_df = pd.DataFrame(rows)

    print(
        f"document_word_data: {doc_word_df.shape[0]} rows, "
        f"{doc_word_df['document'].nunique()} documents"
    )

    return doc_word_df


def build_word_cluster_data(vocab, word_cluster_map):
    """
    Create mapping:
    - ngram
    - sequential_cluster
    """
    cluster_ids = [word_cluster_map[i] for i in range(len(vocab))]

    word_cluster_df = pd.DataFrame(
        {
            "ngram": vocab,
            "sequential_cluster": cluster_ids,
        }
    )

    return word_cluster_df


def compute_textual_factors(document_word_data, word_cluster_data, n_topics=1):
    """
    Compute textual factors using TextualFactors.
    """
    tf_model = TextualFactors(
        document_word_data=document_word_data,
        word_cluster_data=word_cluster_data,
    )

    (
        first_doc_topics,
        second_doc_topics,
        first_topics_words,
        second_topics_words,
        singular_values,
        topic_importances,
    ) = tf_model.lsa_topics(
        cluster_type="sequential_cluster",
        n_topics=n_topics,
    )

    first_doc_topics_df = transfer_document_topics(first_doc_topics)

    if n_topics < 2:
        second_doc_topics_df = pd.DataFrame(
            columns=["cluster_id", "document", "topic_loading"]
        )
    else:
        second_doc_topics_df = transfer_document_topics(second_doc_topics)

    topics_words_df = transfer_topic_words(first_topics_words)
    singular_values_df = transfer_sigular_values(singular_values)
    topic_importances_df = transfer_topic_importances(topic_importances)

    return {
        "first_doc_topics_df": first_doc_topics_df,
        "second_doc_topics_df": second_doc_topics_df,
        "topics_words_df": topics_words_df,
        "singular_values_df": singular_values_df,
        "topic_importances_df": topic_importances_df,
    }


def main():
    print("\n=== STEP 1: Load Item 1A text files ===")
    report_texts, report_sources = load_item1a_documents(item1a_folder)
    print(f"Loaded {len(report_texts)} documents")

    if len(report_texts) == 0:
        raise RuntimeError(
            f"Loaded 0 documents. Check TXT files exist in: {item1a_folder}"
        )

    # Paragraph-level DataFrame for embeddings
    paragraph_rows = []
    for source, text in zip(report_sources, report_texts):
        parts = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not parts:
            parts = [text]

        for part in parts:
            paragraph_rows.append(
                {
                    "content": part,
                    "file": source,
                }
            )

    df_paragraphs = pd.DataFrame(paragraph_rows)
    # Extract year from filename so the peak-year DF filter can use it
    df_paragraphs["year"] = (
        df_paragraphs["file"]
        .str.extract(r"_(\d{4})_item1a")[0]
        .astype(float)
        .astype("Int64")
    )
    df_paragraphs = preprocess_text_and_tokens(
        df_paragraphs,
        text_col="content",
        tokens_col="tokens",
        min_df=PEAK_YEAR_MIN_DF,
        year_col="year",
    )

    print("\n=== STEP 2: Build document-level DataFrame ===")
    df_docs = build_document_dataframe(report_texts, report_sources)
    print(df_docs.head())

    print("\n=== STEP 3: Clean text + tokenize + count words ===")
    df_docs = preprocess_text_and_tokens(
        df_docs,
        text_col="content",
        tokens_col="tokens",
    )

    df_docs = df_docs[df_docs["tokens"].apply(len) >= 5].copy()

    print(f"Documents kept after token+doc filtering: {len(df_docs)}")
    if len(df_docs) == 0:
        raise RuntimeError("No documents left after token filtering.")

    print("Example cleaned document:", df_docs["content"].iloc[0][:200])
    print("Example tokens:", df_docs["tokens"].iloc[0][:20])

    print("\n=== STEP 4: Create OpenAI Embeddings ===")
    vocab, embedding_matrix = train_openai_embeddings(df_paragraphs)
    print(f"Vocabulary size: {len(vocab)}")

    print("\n=== STEP 5: Cluster word embeddings ===")
    ec, clusters, cluster_words_map, word_cluster_map = cluster_words(
        embedding_matrix,
        target_cluster_size=150,
        neighbor_alg="lsh",
    )
    print(f"Number of clusters: {len(clusters)}")

    print("\n=== STEP 6: Build document-word and word-cluster tables ===")
    document_word_data = build_document_word_data(df_docs, vocab)
    word_cluster_data = build_word_cluster_data(vocab, word_cluster_map)

    print(document_word_data.head())
    print(word_cluster_data.head())

    print("\n=== STEP 7: Compute Textual Factors (SVD / LSA) ===")
    tf_results = compute_textual_factors(
        document_word_data,
        word_cluster_data,
        n_topics=N_TOPICS_PER_CLUSTER,
    )

    if N_TOPICS_PER_CLUSTER < 2:
        print("\nNote: N_TOPICS_PER_CLUSTER=1, so TF2 outputs are skipped.")

    out_folder = OUT_FOLDER_SEB
    os.makedirs(out_folder, exist_ok=True)

    # Save metadata
    df_docs[["document", "year", "cik", "file"]].to_csv(
        os.path.join(out_folder, "document_metadata.csv"),
        index=False,
    )

    # Save textual factor outputs
    tf_results["first_doc_topics_df"].to_csv(
        os.path.join(out_folder, "first_doc_topics.csv"),
        index=False,
    )

    if not tf_results["second_doc_topics_df"].empty:
        tf_results["second_doc_topics_df"].to_csv(
            os.path.join(out_folder, "second_doc_topics.csv"),
            index=False,
        )

    tf_results["topics_words_df"].to_csv(
        os.path.join(out_folder, "topics_words.csv"),
        index=False,
    )

    tf_results["singular_values_df"].to_csv(
        os.path.join(out_folder, "singular_values.csv"),
        index=False,
    )

    tf_results["topic_importances_df"].to_csv(
        os.path.join(out_folder, "topic_importances.csv"),
        index=False,
    )

    print("\nPipeline finished.")
    print("Outputs written to:", out_folder)


if __name__ == "__main__":
    main()