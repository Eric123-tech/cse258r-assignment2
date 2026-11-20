# CSE258R Assignment 2: Future Artist Listening

## Research question

**Given a user's past listening history, which artists will they listen to in the next 28 days?**

We rank candidate artists and return the top 10 for each user. Previously heard artists are allowed: this task includes repeat listening, not just discovering new artists. The goal is to compare collaborative preference with recent personal listening behavior.

## Current baselines and method

We use exactly **two course-related baselines**:

1. **Jaccard UserCF (main baseline):** compare users by the overlap of their historical artist sets, find similar users, and rank artists using their weighted preferences.
2. **Cosine UserCF (second baseline):** use cosine similarity of binary user–artist histories, with the same neighbor-based prediction procedure.

Both select **20 neighbors** using validation NDCG@10. Neighbor preferences use normalized `log(1 + play count)`.

These implement similarity-based collaborative filtering from the course's assigned *Personalized Machine Learning*, chapter 4. The [official workbook](https://cseweb.ucsd.edu/~jmcauley/pml/code/chap4.html) demonstrates Jaccard, cosine and similarity-based recommendation. Its example compares items; our implementation adapts the approach to user neighborhoods. See the [course syllabus](https://cseweb.ucsd.edu/classes/fa26/cse258-001/).

Our comparison method is **time-decayed personal preference (Recency)**. Each historical play contributes `2 ** (-age_days / half_life)` to its artist's score. Validation selects a **28-day half-life**. This is a standard temporal heuristic applied to our task, not a new algorithm or a modification inside UserCF.

| Method | Test NDCG@10 | Test Recall@10 | Test Precision@10 |
|---|---:|---:|---:|
| Jaccard UserCF — main baseline | 0.2852 | 0.0354 | 0.2701 |
| Cosine UserCF — second baseline | 0.2883 | 0.0355 | 0.2726 |
| Recency — project method | **0.5445** | **0.0950** | **0.5210** |

Recency performs better in this experiment. This does not isolate the effect of time weighting, because Recency uses personal history while UserCF uses neighbor preferences. It also does not establish better new-artist discovery.

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

- **Isolate the time effect:** compare unweighted and time-weighted versions of the same personal-history method, or add temporal weighting within Jaccard UserCF.
- **Try another course method:** add a BPR latent-factor model for implicit feedback.
- **Check robustness:** use additional chronological windows and examine low-activity users, repeat listening and new-artist targets separately.

These are proposals, not implemented improvements. The current test results have already been inspected; do not tune new choices on this test window. Use past-only validation or reserve a new held-out period.

The executed notebook is an initial assignment artifact. We still need to deepen related work and interpretation and prepare the presentation/video required by the assignment.
