# Phase 1 Notes: Foundation and Data Pipeline

**Project:** Detecting Hallucinations in AI-Generated Text using NLI and Self-Consistency Checking  
**Domain:** Indian Court Judgments (English only)  
**Dataset:** MILDSum Benchmark ([Law-AI/MILDSum](https://github.com/Law-AI/MILDSum))  
**Stage:** Phase 1 Complete (Foundation, Data Loader, Text Utils, Sampling, Unit Tests)

---

## 1. Dataset Structure & Sourcing

### 1.1 Where the Data is Hosted
- The MILDSum benchmark is published by the Law-AI research group (accepted at EMNLP 2023).
- **Public Samples:** The GitHub repository hosts 10 complete sample pairs under `Data/MILDSum_Samples/` (`Sample_1` to `Sample_10`).
- **Full Dataset:** Per the official README, the full dataset contains 3,122 judgments. Access requires emailing the authors (`debtanudatta04@gmail.com`).
- **Automated Script:** We created `scripts/download_mildsum.py` which automatically fetches the English files for available samples directly into `data/raw/`.

### 1.2 File Organisation & English-Only Rule
Each sample folder in MILDSum contains three files:
- `EN_Judgment.txt`: The full text of the Indian court judgment.
- `EN_Summary.txt`: The expert-written reference summary in English.
- `HI_Summary.txt`: The Hindi translation of the summary.

**Rule Enforced:** Our pipeline downloads and processes **only** `EN_Judgment.txt` and `EN_Summary.txt`. `HI_Summary.txt` is strictly excluded. Raw data files are kept immutable in `data/raw/` and excluded from git via `.gitignore`.

---

## 2. Technical Decisions & Rationale

### 2.1 Dependency Management
- Requirements are strictly kept minimal for Phase 1: `pandas`, `tqdm`, `pytest`, `nltk`, and `pysbd`.
- Neural model libraries (`torch`, `transformers`, `summac`, `selfcheckgpt`, `ollama`) are deliberately excluded until Phases 3 and 4 to ensure fast, lightweight data foundation work.

### 2.2 Text Utilities (`clean_text`)
- **Strict Content Preservation:** `clean_text()` normalises whitespace only. It removes trailing whitespace, standardises line breaks (`\r\n` to `\n`), collapses multiple horizontal spaces/tabs, and normalises non-breaking spaces.
- **Integrity Guarantee:** It never alters words, casing, punctuation, dates (`20.06.2020`), statutory citations (`Sec. 302 IPC`, `Art. 226`), judge names, or monetary amounts (`Rs. 24,02,380`).

### 2.3 Sentence Splitting Comparison (`split_sentences`)
Indian legal documents feature unique patterns that break generic sentence splitters:
1. Currency amounts: `Rs. 5,000` or `Rs. 24,02,380/-`
2. Legal abbreviations: `Ors.`, `vs.`, `Sec.`, `Art.`, `No.`, `Hon.`, `Adv.`
3. Judicial designations: `Justice Rekha Palli, J.`, `ADJ Dinesh Kumar`
4. Numbered paragraphs: `1. The petitioner...`, `2. Learned senior counsel...`

#### Comparison of Candidates:

| Input Text | NLTK Default `sent_tokenize` | `pysbd` Segmenter | Custom NLTK Legal Punkt (Chosen) |
| :--- | :--- | :--- | :--- |
| `"The petitioner paid Rs. 24,02,380 to the hospital."` | ❌ Breaks into 2: `['The petitioner paid Rs.', '24,02,380 to the hospital.']` | ✅ 1 sentence: `['The petitioner paid Rs. 24,02,380 to the hospital.']` | ✅ 1 sentence: `['The petitioner paid Rs. 24,02,380 to the hospital.']` |
| `"In ABC & Ors. vs. State, the court ruled on Sec. 302 IPC."` | ❌ Breaks into 3: splits at `Ors.`, `vs.`, and `Sec.` | ❌ Breaks into 3: `['In ABC & Ors. ', 'vs. State', ', the court ruled on Sec. 302 IPC.']` | ✅ 1 sentence: preserves entire citation without splits |
| `"1. The petitioner filed a plea. 2. The state replied."` | ❌ Isolates `['1.', 'The petitioner...', '2.', 'The state...']` | ✅ 2 sentences: handles paragraph numbers cleanly | ✅ 2 sentences: numbered prefix placeholder preservation |

#### Selected Engine: **Custom NLTK Legal Punkt**
- **Why chosen:**
  1. Handles legal abbreviations (`Ors.`, `vs.`, `Sec.`, `Art.`, `No.`, `J.`, `Rs.`) without false sentence boundaries.
  2. Runs fully offline without downloading heavy neural models.
  3. Pre-processes paragraph numbers so judgment paragraphs (`1. `, `2. `) remain whole.
  4. Modular: `split_sentences(text, method="pysbd")` is also provided as an alternate option.

---

## 3. Sample Preparation & Statistical Summary

### 3.1 Script Execution
`python scripts/prepare_data.py --n 30 --seed 42`
- Loaded and validated 10 available English pairs from `data/raw/`.
- Segmented into sentences and computed character, word, and sentence statistics.
- Saved processed output:
  - `data/processed/sample_pairs.jsonl`
  - `data/processed/stats.csv`

### 3.2 Key Dataset Statistics (10 Sample Pairs)

| Metric | Mean | Median | Min | Max |
| :--- | :--- | :--- | :--- | :--- |
| **Judgment Characters** | 18,683.9 | 14,443.5 | 6,116.0 | 36,200.0 |
| **Judgment Words** | 3,078.7 | 2,436.5 | 995.0 | 5,959.0 |
| **Judgment Sentences** | 102.4 | 82.5 | 32.0 | 270.0 |
| **Summary Characters** | 4,383.2 | 4,371.0 | 2,043.0 | 5,829.0 |
| **Summary Words** | 726.1 | 745.0 | 367.0 | 964.0 |
| **Summary Sentences** | 23.5 | 22.5 | 12.0 | 35.0 |
| **Word Compression Ratio** | 29.8% | 26.2% | 12.5% | 59.6% |
| **Sentence Compression Ratio**| 34.2% | 30.8% | 9.4% | 78.1% |

---

## 4. Problems Identified & Downstream Implications

### Problem 1: Extreme Document Length vs. NLI Input Limits (Phase 3)
- **Finding:** Court judgments average **~3,078 words (~4,000+ tokens)** and reach up to **5,959 words (~8,000 tokens)**.
- **NLI Limit:** Cross-encoder NLI models used by SummaC (e.g., `roberta-large-mnli`, `deberta-v3-base`) have a hard maximum token limit of **512 tokens**.
- **Action for Phase 3:** We cannot evaluate `(Judgment, Summary_Sentence)` as a single pair. We must implement:
  - Paragraph-level chunking of the judgment.
  - Sentence-level similarity retrieval (e.g. BM25 or embedding cosine similarity) to select the top-K relevant judgment sentences before running SummaC NLI.

### Problem 2: Factual Density in Summaries (Phase 4)
- **Finding:** Summaries contain high concentrations of named entities, dates, specific figures (`Rs. 16,93,880/-`), circular references (`20.06.2020`), and hospital names (`PSRI Hospital`).
- **Hallucination Risk:** Generative LLMs frequently hallucinate or alter specific digits and section numbers during summarization.
- **Action for Phase 4:** SelfCheckGPT consistency checks must specifically monitor factual assertions and entity consistency across multiple sampled summaries.

### Problem 3: Public Dataset Size
- **Finding:** The GitHub repository provides 10 sample pairs. For full statistical power in later evaluation, having 30 to 50 pairs is desirable.
- **Action:** The student may email `debtanudatta04@gmail.com` to request the full dataset archive. Once received, dropping the sample folders into `data/raw/` will automatically integrate them into the pipeline.

---

## 5. Open Questions for User / Supervisor

1. **Full Dataset Request:** Would you like to email the authors for the full 3,122 cases, or continue developing Phases 2-4 using the 10 available pairs (with synthetic perturbations for evaluation)?
2. **LLM Selection for Approach B:** Which local open-source LLM do you prefer for generating sample summaries in Phase 4? (e.g. `llama3:8b`, `mistral:7b`, or `gemma2:9b` via Ollama).
