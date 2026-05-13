# Replication Package

Code for replicating the empirical analysis in the thesis. Scripts are numbered S0-S6 and should be run in order. Data files (WRDS, CRSP, SEC EDGAR) are not included due to licensing restrictions and must be obtained separately.

## Requirements

- Python 3.10+
- WRDS account (Stage 0)
- OpenAI API key (Stage 2)

```
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Pipeline

**S0 — WRDS extraction.** Pulls Compustat fundamentals, CRSP daily returns, and the PERMNO-CIK link from WRDS.

**S1 — Corpus construction.** Builds the bank-year panel, maps PERMNOs to CIKs, downloads Item 1A text from SEC EDGAR.

**S2 — Embedding, clustering, SVD.** Runs the Cong et al. (2024) textual factors pipeline: OpenAI embeddings, LSH clustering, within-cluster SVD. S2b re-runs SVD on the regression sample only to avoid look-ahead.

**S3 — Theme selection.** Two-track selection: high-CV clusters reviewed manually (Track 1), low-CV clusters matched to a banking risk taxonomy (Track 2). Redundant clusters consolidated. S3c and S3d are Streamlit apps (`streamlit run S3c_review_track1.py`).

**S4 — Loading matrix.** Generates the bank-year loading matrix with cross-sectional standardization.

**S5 — Pairwise regressions.** Builds the pairwise dataset, computes theme products, runs quarterly OLS (Hanley and Hoberg Eq. 3 vs 4), and the leave-one-out theme decomposition.

**S6 — Tables and figures.** Produces all thesis outputs: summary statistics, emerging risk index plots, theme decomposition, determinant regressions, crisis return tests, robustness checks.

## Shared modules

`lib/TextualFactors.py` and `lib/engine.py` implement the core Cong et al. (2024) text processing. Imported by the S2 and S3 scripts.

## Notes

- The SEC downloader (S1c) is rate-limited and takes several hours.
- Embedding generation (S2a) requires OpenAI API calls. Cached embeddings are reused in later stages.
- Scripts write intermediate files to `../data/` and final outputs to `../output/` relative to their own location. These folders are created automatically.
