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
  2. **Phase 2:** Labelled test set + evaluation framework
  3. **Phase 3:** Approach A (SummaC NLI pipeline)
  4. **Phase 4:** Approach B (SelfCheckGPT consistency pipeline)
  5. **Phase 5:** Comparative analysis + Streamlit interactive web demo

---

## Directory Structure

```text
hallucination-detector/
├── README.md                 # Project guide & quickstart
├── requirements.txt          # Python dependencies (lightweight for Phase 1)
├── pyproject.toml            # Project packaging & pytest config
├── .gitignore                # Git exclusions (venv, raw data, caches)
├── src/
│   └── hd/
│       ├── __init__.py       # Package entry point
│       ├── text_utils.py     # Whitespace normaliser & legal sentence splitter
│       └── data_loader.py    # Robust MILDSum pair loader and validator
├── scripts/
│   ├── download_mildsum.py   # Downloader for available MILDSum English samples
│   ├── prepare_data.py       # Segmenter, sampler, and stats exporter
│   └── explore_data.py       # CLI exploratory data analysis script
├── tests/
│   ├── test_text_utils.py    # Unit tests for abbreviations, currency, cleaning
│   └── test_data_loader.py   # Unit tests with isolated mock fixtures
├── notebooks/
│   └── 01_explore_data.ipynb # Interactive EDA & length distributions
├── docs/
│   └── phase1_notes.md       # Detailed technical decisions & findings
└── data/
    ├── raw/                  # Downloaded raw text files (not committed to git)
    └── processed/            # sample_pairs.jsonl and stats.csv
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

# Install lightweight Phase 1 dependencies
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

# Install lightweight Phase 1 dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

### Step 1: Download Available Raw Samples
Downloads the 10 available English judgment-summary pairs from the official MILDSum repository:
```bash
python scripts/download_mildsum.py
```

### Step 2: Prepare & Sample Processed Data
Cleans text, splits sentences using legal abbreviation rules, and exports outputs:
```bash
python scripts/prepare_data.py --n 30 --seed 42
```
Outputs created:
- `data/processed/sample_pairs.jsonl`: Formatted JSONL records with `id`, `judgment_text`, `summary_text`, `judgment_sentences`, and `summary_sentences`.
- `data/processed/stats.csv`: Character, word, and sentence counts for each document pair.

### Step 3: Run Unit Tests
Run the test suite verifying sentence splitting, currency preservation (`Rs. 5,000`), abbreviations, and encoding safety:
```bash
pytest -v
```

### Step 4: Explore Data & Length Distributions
Run via command line:
```bash
python scripts/explore_data.py
```
Or open Jupyter Notebook:
```bash
jupyter notebook notebooks/01_explore_data.ipynb
```

---

## Key Technical Decisions in Phase 1

1. **Legal Sentence Splitting:**
   Indian legal judgments contain frequent abbreviations (`Ors.`, `vs.`, `Sec.`, `Art.`, `No.`, `Hon.`, `J.`) and currency figures (`Rs. 5,000`). Standard sentence splitters break sentences at `Rs.` and abbreviations. We implemented a custom **NLTK Legal Punkt** tokenizer with custom parameters and paragraph-prefix preservation to eliminate false splits.
2. **Minimal Dependencies:**
   No neural weights or GPU libraries are loaded in Phase 1. Dependencies are restricted to lightweight utilities (`pandas`, `tqdm`, `pytest`, `nltk`, `pysbd`).
3. **Data Safety:**
   Raw files in `data/raw/` are immutable and strictly ignored by git to protect data integrity and avoid committing raw corpora.

---

## Author & Academic Context
Built as part of B.Tech Mini-Project: *Detecting Hallucinations in AI-Generated Text using NLI and Self-Consistency Checking*.
