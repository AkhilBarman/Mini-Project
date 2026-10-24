# Detecting Hallucinations in AI-Generated Legal Text

A modular NLP toolkit to detect hallucinations in AI-written summaries of Indian court judgments (English only) using **NLI (SummaC)** and **Self-Consistency Checking (SelfCheckGPT)**.

---

## Project Overview

- **Dataset:** [MILDSum](https://github.com/Law-AI/MILDSum) (Multilingual Indian Legal Document Summarization benchmark - English court judgments and expert-written summaries).
- **Core Approaches Evaluated:**
  - **Approach A (NLI-based):** Compare each summary sentence with the source judgment using Natural Language Inference (SummaC).
  - **Approach B (Self-Consistency):** Sample multiple summaries using an open-source LLM and flag sentences unsupported by alternative versions (SelfCheckGPT).
- **Project Phases:**
  1. **Phase 1: Foundation & Data Pipeline** *(Completed)*
  2. **Phase 2: Labelled Test Set & Evaluation Framework** *(Completed)*
  3. **Phase 3:** Approach A (SummaC NLI pipeline)
  4. **Phase 4:** Approach B (SelfCheckGPT consistency pipeline)
  5. **Phase 5:** Comparative analysis + Streamlit interactive web demo

---

## Directory Structure

```text
hallucination-detector/
├── README.md                      # Project guide & quickstart
├── requirements.txt               # Project dependencies
├── pyproject.toml                 # Package definition & pytest config
├── .gitignore                     # Git exclusions (venv, raw data, caches)
├── AGENTS.md                      # AI agent workflow and project instructions
├── src/
│   └── hd/
│       ├── __init__.py            # Package root exports
│       ├── text_utils.py          # Whitespace normaliser & legal sentence splitter
│       ├── data_loader.py         # Robust MILDSum pair loader and validator
│       ├── mutations.py           # Phase 2: Rule-based sentence mutators (5 error types)
│       └── evaluation.py          # Phase 2: Classification/ranking metrics, Timer, results
├── scripts/
│   ├── download_mildsum.py        # Downloader for available MILDSum English samples
│   ├── prepare_data.py            # Preprocesses, segments, and samples pairs
│   ├── explore_data.py            # CLI exploratory data analysis script
│   ├── make_candidates.py         # Phase 2: Generates candidate mutations for review
│   ├── build_testset.py           # Phase 2: Constructs sentence-level testset.csv
│   ├── evaluate.py                # Phase 2: Evaluates any predictions CSV
│   └── evaluate_baselines.py      # Phase 2: Runs Oracle and Random baseline scorers
├── tests/
│   ├── test_text_utils.py         # Tests sentence splitting, abbreviations, currency
│   ├── test_data_loader.py        # Tests data loader with isolated mock fixtures
│   ├── test_mutations.py          # Tests 5 mutation types and edge cases
│   ├── test_build_testset.py      # Tests testset assembly on isolated fixtures
│   └── test_evaluation.py         # Tests metrics, oracle, random, and timer
├── notebooks/
│   └── 01_explore_data.ipynb      # Interactive EDA & length distributions
├── docs/
│   ├── phase1_notes.md            # Phase 1 technical decisions & findings
│   └── phase2_notes.md            # Phase 2 test set design & baseline results
├── data/
│   ├── raw/                       # Downloaded raw text files (not committed to git)
│   ├── processed/                 # sample_pairs.jsonl and stats.csv
│   └── testset/                   # candidates.csv and ground truth testset.csv
└── results/                       # Baseline outputs and evaluation runs (JSON / CSV)
```

---

## Quickstart & Setup

### Requirements
- **Python 3.10+**
- Git

### 1. Windows Setup (PowerShell)

```powershell
# Navigate into the project folder
cd hallucination-detector

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Install project dependencies
pip install -r requirements.txt
```

*(If script execution is disabled on PowerShell, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

### 2. macOS / Linux Setup

```bash
# Navigate into the project folder
cd hallucination-detector

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install project dependencies
pip install -r requirements.txt
```

---

## Running the Complete Pipeline

### Phase 1: Data Ingestion & Sample Preparation
```bash
# 1. Download available English samples from MILDSum
python scripts/download_mildsum.py

# 2. Prepare, segment, and sample document pairs
python scripts/prepare_data.py --n 30 --seed 42

# 3. Explore dataset statistics
python scripts/explore_data.py
```

### Phase 2: Labelled Test Set & Evaluation Baselines
```bash
# 1. Generate candidate mutations across 5 error categories
python scripts/make_candidates.py --seed 42

# 2. Build the sentence-level testset.csv from reviewed candidates
python scripts/build_testset.py --from-review data/testset/candidates_reviewed.csv

# 3. Run Oracle and Random baselines
python scripts/evaluate_baselines.py

# 4. Evaluate custom prediction scores against testset.csv
python scripts/evaluate.py --predictions results/baseline_oracle_preds.csv
```

### Run All Unit Tests
Run the entire 33-test test suite covering text utilities, data loaders, mutators, test set builders, and evaluation metrics:
```bash
pytest -v
```

---

## Phase 2 Test Set & Baselines Summary

- **Total Testset Sentences:** 462 sentences (across 10 document pairs)
- **Supported Sentences (Label 0):** 446 rows (96.5%)
- **Hallucinated Sentences (Label 1):** 16 rows (3.5%)
- **Error Categories:** `DATE_NUMBER` (3), `PARTY_OUTCOME` (3), `NAME_PLACE` (3), `NEGATION` (2), `FABRICATED` (5)

### Baseline Verification Results

| Metric | Oracle Baseline | Random Baseline (seed=42) |
| :--- | :--- | :--- |
| **Accuracy** | **1.0000** | 0.5108 |
| **Precision** | **1.0000** | 0.0551 |
| **Recall** | **1.0000** | 0.8125 |
| **F1 Score** | **1.0000** | 0.1032 |
| **ROC AUC** | **1.0000** | **0.6326** (Near chance $\approx 0.50$) |
| **Average Precision** | **1.0000** | 0.0476 (Near base rate 3.5%) |

---

## Author & Academic Context
Built as part of B.Tech Mini-Project: *Detecting Hallucinations in AI-Generated Text using NLI and Self-Consistency Checking*.
