# AGENTS.md: instructions for any AI coding agent working in this repo

## What this project is
Mini-project (B.Tech Sem 5, Deep Learning + NLP): "Detecting Hallucinations in
AI-Generated Text using NLI and Self-Consistency Checking".
We check an AI-written summary of an Indian court judgment (English only) and
flag summary sentences that the judgment does not support (hallucinations).
Two checkers are built and compared on the same test set:
- Approach A (NLI): each summary sentence is checked against the judgment with
  an NLI model, using the SummaC library.
- Approach B (self-consistency): an open-source LLM (Llama 3 via Ollama) writes
  the summary several times; the SelfCheckGPT library scores how well the other
  versions support each sentence.
Comparison: accuracy, precision, recall, F1, ROC AUC, time per summary, number
of model runs per summary.
Data: MILDSum (English judgments + English expert summaries), academic use only.

## Who you are working with
A student. Keep code simple and readable. Explain decisions in plain, easy
language. If something is unclear or blocked, ask. Do not guess.

## Phase roadmap (update the status at the end of every phase)
1. Foundation and data pipeline: DONE
2. Labelled test set + evaluation framework: DONE
3. Approach A (SummaC NLI): DONE
4. Approach B (LLM sampling + SelfCheckGPT): DONE
5. Comparison + Streamlit demo + final docs: DONE
Work only on the phase you were asked to do.

## Repo map
```text
hallucination-detector/
├── README.md                      # Project overview, benchmark tables, and setup instructions
├── requirements.txt               # Project dependencies including PyTorch, SummaC, Streamlit
├── pyproject.toml                 # Package definition and pytest configuration
├── .gitignore                     # Git ignore rules (venv, raw data, caches)
├── AGENTS.md                      # Instructions for AI agents working in this repository
├── .vscode/                       # VS Code workspace settings
│   └── settings.json
├── data/
│   ├── raw/                       # Immutable raw MILDSum sample downloads (git-ignored)
│   ├── processed/                 # Segmented JSONL records and stats.csv
│   │   ├── sample_pairs.jsonl
│   │   └── stats.csv
│   ├── sampled_summaries/         # Phase 4: Stochastic LLM summary samples (N=3)
│   └── testset/                   # Labelled evaluation test sets (Phase 2)
│       ├── candidates.csv         # Generated mutation candidates
│       ├── candidates_reviewed.csv # Human-reviewed candidates (accept, fixed_sentence, notes)
│       ├── review_report.md       # Review breakdown and acceptance decisions
│       └── testset.csv            # Final ground-truth sentence evaluation dataset (462 sents)
├── demo/
│   └── app.py                     # Phase 5: Interactive Streamlit Web Platform
├── docs/
│   ├── phase1_notes.md            # Ingestion, preprocessing, and exploratory statistics
│   ├── phase2_notes.md            # Test set design, mutations, and baseline evaluation
│   ├── phase3_notes.md            # Approach A (SummaC NLI, threshold calibration, ROC-AUC)
│   ├── phase4_notes.md            # Approach B (SelfCheckGPT, token containment, latency)
│   └── phase5_notes.md            # Final comparative analysis, hybrid ensemble, viva notes
├── notebooks/
│   ├── 01_explore_data.ipynb      # Phase 1 data exploration notebook
│   ├── 02_run_approach_a_colab.ipynb # Phase 3: 1-click Colab GPU inference for SummaC
│   └── 03_run_approach_b_colab.ipynb # Phase 4: 1-click Colab GPU inference for SelfCheckGPT
├── results/                       # Predictions, metrics JSONs, ROC and comparison charts
│   ├── approachA_mnli_base_predictions.csv
│   ├── approachA_mnli_base_predictions_eval.json
│   ├── approachB_selfcheck_predictions.csv
│   ├── approachB_selfcheck_predictions_eval.json
│   ├── approach_hybrid_predictions.csv
│   ├── baseline_oracle_preds.csv
│   ├── baseline_random_preds.csv
│   ├── comparison_summary.csv
│   ├── comparison_summary.md
│   ├── comparison_summary.json
│   ├── comparison_breakdown.png   # Phase 5: 4-panel comparative visualization
│   └── roc_curve_approach_a.png   # Phase 3: Approach A ROC curve
├── scripts/
│   ├── download_mildsum.py        # Automated downloader for English MILDSum samples
│   ├── prepare_data.py            # Preprocesses, segments, and samples pairs
│   ├── explore_data.py            # Terminal-based dataset exploration
│   ├── make_candidates.py         # Phase 2: Generates candidate sentence mutations
│   ├── build_testset.py           # Phase 2: Builds ground truth testset.csv
│   ├── evaluate.py                # Phase 2: Evaluates any predictions CSV
│   ├── evaluate_baselines.py      # Phase 2: Runs Oracle and Random baseline scorers
│   ├── run_approach_a.py          # Phase 3: SummaC NLI evaluation runner
│   ├── sample_summaries.py        # Phase 4: Generates stochastic summary samples
│   ├── run_approach_b.py          # Phase 4: SelfCheckGPT consistency checker
│   └── compare_approaches.py      # Phase 5: Comparative evaluation & ensembling
├── src/
│   └── hd/
│       ├── __init__.py            # Package root exports
│       ├── data_loader.py         # Loads and validates raw MILDSum judgment-summary pairs
│       ├── text_utils.py          # Whitespace normaliser and legal sentence splitter
│       ├── mutations.py           # Phase 2: Deterministic sentence-level rule mutators
│       ├── evaluation.py          # Phase 2: Metric computations, timer, and result exports
│       ├── approach_a.py          # Phase 3: SummaC Cross-Encoder NLI engine
│       ├── sampling.py            # Phase 4: LLM sampling pipeline (Ollama / Transformers)
│       ├── approach_b.py          # Phase 4: SelfCheckGPT consistency checker engine
│       └── comparison.py          # Phase 5: Comparison, calibration, and dashboard visualizer
└── tests/
    ├── test_data_loader.py        # Tests data loader with isolated mock fixtures
    ├── test_text_utils.py         # Tests sentence splitting, legal abbreviations, currency
    ├── test_mutations.py          # Phase 2: Tests all mutation types and edge cases
    ├── test_build_testset.py      # Phase 2: Tests testset creation and row indexing
    ├── test_evaluation.py         # Phase 2: Tests metrics calculation, oracle, and baselines
    ├── test_approach_a.py         # Phase 3: Tests SummaC NLI scoring and caching
    ├── test_approach_b.py         # Phase 4: Tests SelfCheckGPT containment and sampling
    └── test_comparison.py         # Phase 5: Tests calibration, hybrid ensembling, and tables
```

## Commands
```powershell
# 1. Environment Activation (Windows PowerShell)
cd hallucination-detector
.\venv\Scripts\Activate.ps1

# 2. Dependency Installation
pip install -r requirements.txt

# 3. Test Suite Execution (51 passing unit tests)
pytest -v

# 4. Phase 5: Run Full Comparative Evaluation & Generate Visual Dashboard
python scripts/compare_approaches.py

# 5. Phase 5: Launch Interactive Streamlit Web Platform
streamlit run demo/app.py
```

## Conventions every phase must follow
- Unit of prediction = one summary sentence.
- Label: 1 = hallucinated (positive class), 0 = supported.
- Score: every checker outputs hallucination_score in [0, 1], where HIGHER means
  MORE LIKELY hallucinated. SummaC gives a support score (higher = more
  consistent), so Phase 3 must convert it (for example 1 - normalised score)
  and document the conversion.
- Predictions file contract: CSV with columns `row_id`, `hallucination_score` and
  optional `seconds`. Row ids come from `data/testset/testset.csv`.
- Reproducibility: fixed seeds, settings in one config place, no hidden state.
- Results go in `results/` as CSV/JSON with metadata (date, approach, model
  name, threshold).

## Rules
- Code: simple, readable, type hints, docstrings, plain-English comments. No
  clever tricks.
- Data: never edit files in `data/raw/` and never commit them. Original judgment
  and summary text must never be changed, except by the controlled mutation
  script that creates the test set.
- Scope: do not install heavy libraries (torch, transformers, summac,
  selfcheckgpt, ollama) before the phase that needs them.
- Safety: do not change anything outside the project folder or install global
  packages. Use only free, open tools.
- Git: small commits with clear messages. Run the tests before every commit.
- Docs: at the end of each phase, write `docs/phaseN_notes.md` and update this
  file's roadmap, repo map and commands.

## Definition of done for any phase
Setup steps in README work; tests pass; notes written; a short final summary
for the student (what was built, how to run it, decisions and why, problems
found, what the next phase needs).
