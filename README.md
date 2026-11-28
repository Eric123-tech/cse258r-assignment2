# CSE258R Assignment 2: Future Artist Listening

## Research question

**Given a user's past listening history, which artists will they listen to in the next 28 days?**

We rank candidate artists and return the top 10 for each user. Previously heard artists are allowed: this task includes repeat listening, not just discovering new artists. The goal is to compare collaborative preference with recent personal listening behavior.

## Current baselines and method

We use exactly **two course-related baselines**:

1. **Jaccard UserCF (main baseline):** compare users by the overlap of their historical artist sets, find similar users, and rank artists using their weighted preferences.
2. **Cosine UserCF (second baseline):** use cosine similarity of binary user–artist histories, with the same neighbor-based prediction procedure.

Both select **20 neighbors** using validation NDCG@10. Neighbor preferences use normalized `log(1 + play count)`.

The supplied `recommendation.pdf` explicitly covers user/item similarities (p.28: Jaccard; pp.29–31: cosine; p.66: user-based and weighting adaptations). These implement similarity-based collaborative filtering from the course's assigned *Personalized Machine Learning*, chapter 4. The [official workbook](https://cseweb.ucsd.edu/~jmcauley/pml/code/chap4.html) demonstrates Jaccard, cosine and similarity-based recommendation. Its example compares items; our implementation adapts the approach to user neighborhoods. See the [course syllabus](https://cseweb.ucsd.edu/classes/fa26/cse258-001/).

Our comparison method is **time-decayed personal preference (Recency)**. Each historical play contributes `2 ** (-age_days / half_life)` to its artist's score. Validation selects a **28-day half-life**. The supplied `temporal.pdf`, pp.12–13, explicitly teaches exponential age weighting. Our day-based implementation applies that course idea to artist plays; it is not a new algorithm or a modification inside UserCF.

We also report two **trivial baselines** with no parameters — global **Popularity** and **Personal counts** (each user's most-played artists, no time decay) — and a **logistic-regression ranker** that scores (user, artist) candidate pairs from 21 history features (play counts, distinct listening days, multi-scale decay, days since last play, CF score, artist popularity). The ranker is trained on the 28 days before each cutoff, so it never sees the evaluation window; C is chosen on validation. These three were added after the original test comparison had been inspected, designed on validation only and evaluated on test once.

| Method | Test NDCG@10 | Test Recall@10 | Test Precision@10 |
|---|---:|---:|---:|
| Popularity — trivial baseline | 0.1677 | 0.0218 | 0.1521 |
| Jaccard UserCF — main baseline | 0.2852 | 0.0354 | 0.2701 |
| Cosine UserCF — second baseline | 0.2883 | 0.0355 | 0.2726 |
| Personal counts — trivial baseline | 0.4887 | 0.0755 | 0.4622 |
| Recency — project method | 0.5445 | 0.0950 | 0.5210 |
| LogReg ranker C=0.01 | **0.5705** | **0.0996** | **0.5419** |

The ranker beats Recency by +0.026 NDCG@10 on test (paired bootstrap 95% interval [0.017, 0.035]; +0.020 on validation), at the cost of lower tail recall (0.055 vs 0.079). Its largest coefficient is the number of distinct days an artist was played; raw play count is negative once days are held fixed. Recency performs better than both UserCF baselines. This does not isolate the effect of time weighting, because Recency uses personal history while UserCF uses neighbor preferences. It also does not establish better new-artist discovery.

## Additions following the detailed assignment slides (local revision)

The notebook answers the questions in `assignment2_fa26.pdf` pages 8 and 10–13 directly, using numerical findings and interpretation after the supporting plots. It does not treat a plot inventory or requirement checklist as the answer. Training-only EDA findings are saved in `reports/eda_findings.csv`. Slides: [shared course folder](https://drive.google.com/drive/folders/1LcoSIMkErp1OHRQ5nUMTokfP_o51OY77).

- **Feature representations:** validation-only Jaccard vote comparison (binary/raw-count/log-count) and a matched uniform-count versus exponential-age-weighting contrast. These diagnostics do not add headline baselines or change the frozen test models.
- **Parameter interpretation:** validation curves for neighbor count and half-life; explicit score equations, model suitability, noise/overfitting discussion, and scoring/evaluation timing and score-matrix storage.
- **Validity and errors:** repeat-target versus novel-target recall, user-level NDCG changes, and deterministic low/median/high examples with artist labels. Already-seen test results are descriptive, not a new blind test.
- **Related work:** links dataset choice to Last.fm 1K versus 360K, implicit-feedback factorization and SASRec, with explicit limits on comparing published scores.

New aggregate outputs include `reports/validation_feature_representations.csv`, `validation_feature_representations.png`, `validation_parameter_choices.png`, `per_user_ndcg_differences.png`, `preprocessing_audit.csv` and `slide_requirements_alignment.csv`. Individual-user examples stay local. The two course baselines and main test protocol are retained.

## Data: download separately

Dataset: [matthewfranglen/lastfm-1k on Hugging Face](https://huggingface.co/datasets/matthewfranglen/lastfm-1k), originally collected by Òscar Celma from Last.fm.

**The dataset is not included in GitHub.** Raw and processed event files are ignored by Git. On the original Linux machine, the raw data are already at:

```text
/home/eric/linux_projects/cse258r-assignment2/data/raw/
```

On your machine, download or copy the files into this directory relative to the repository root:

```text
data/raw/
├── train.gz.parquet
├── valid.gz.parquet
├── test.gz.parquet
├── README.md          # source dataset card
└── manifest.json      # pinned revision, URLs and SHA256 checksums
```

Run `python scripts/download_data.py` after activating the environment to download the approximately **469 MB** dataset automatically. Keep the committed manifest so everyone uses the same source revision. If copying files manually, use the manifest's pinned URLs/files; the downloader verifies their checksums. Generated event audits and split metadata go in `data/processed/`.

Use the data for noncommercial purposes and acknowledge Last.fm and Celma. The [creator's dataset archive](https://zenodo.org/records/6090214) documents the collection and terms.

## How to run

Use **Python 3.13**. Run these commands from the cloned repository root on Linux/macOS:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
python scripts/check_protocol.py
python scripts/build_notebook.py
python scripts/run_notebook.py
```

On Windows, activate with `.venv\Scripts\activate` instead. Internet access is needed for dependencies and the initial dataset download.

- Open **`workbook.html`** in a browser to read the saved analysis without running code.
- Open **`notebooks/assignment2.ipynb`** in VS Code/Jupyter with the `.venv` Python interpreter to work interactively. Install your preferred notebook interface separately if needed.
- Edit **`scripts/build_notebook.py`** to make notebook changes persist when rebuilding. It generates the notebook and resets its outputs; the runner executes it and exports HTML.
- Edit **`src/pipeline.py`** for data handling, model scores, parameter candidates and evaluation.
- Find current metrics and figures in **`reports/`**. Individual-user CSVs, event files, local archives and the supplied assignment PDF are not uploaded.

The protocol checks cover hand-calculated ranking metrics, out-of-catalog targets, temporal weights, self-neighbor exclusion and chronological boundaries.

## Evaluation setup

We combine the dataset's original user-disjoint splits and create a chronological split:

| Split | Period |
|---|---|
| Training | Before March 10, 2009 |
| Validation | March 10–April 6, 2009 |
| Test | April 7–May 4, 2009 |

We remove identical event rows and conservatively exclude timestamps on/after May 5, 2009, based on the documented collection endpoint. Candidate artists require five historical listeners. Evaluated users require 100 history events and observed activity in the future window.

Parameters are selected on validation; models are refit on training plus validation for final test evaluation. All methods rank the same full candidate catalog within each split. Scores are averaged per user. Final evaluation includes **776 users**; the catalog covers **87.71%** of future user–artist targets. Unavailable targets remain in recall denominators.

## Possible next steps

- **Extend the controlled study:** the uniform-versus-decay personal-history comparison is now implemented on validation. A future model could add temporal weighting inside Jaccard UserCF, using separate validation.
- **Try another course method:** add a BPR latent-factor model for implicit feedback.
- **Check robustness:** use additional chronological windows and examine low-activity users, repeat listening and new-artist targets separately.

These are proposals, not implemented improvements. The current test results have already been inspected; do not tune new choices on this test window. Use past-only validation or reserve a new held-out period.

The executed notebook is an initial assignment artifact. We still need to deepen related work and interpretation and prepare the presentation/video required by the assignment.
