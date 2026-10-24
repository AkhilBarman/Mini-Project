# Phase 2 Notes: Labelled Test Set and Evaluation Framework

**Project:** Detecting Hallucinations in AI-Generated Text using NLI and Self-Consistency Checking  
**Domain:** Indian Court Judgments (English only, MILDSum benchmark)  
**Stage:** Phase 2 Complete (Mutations Engine, Candidate Generation, Test Set Builder, Evaluation Framework, Baselines)

---

## 1. Overview and Problem Context

In real-world deployment, we need an objective, repeatable way to measure how well our two checkers (**Approach A: SummaC NLI** and **Approach B: SelfCheckGPT**) detect hallucinated summary sentences. 

Because the reference summaries in the MILDSum benchmark were drafted by legal practitioners, they are assumed to be factual and do not contain annotated hallucination labels. To build a ground-truth evaluation benchmark without training-data contamination, we inject controlled, realistic factual mutations into real summary sentences.

---

## 2. Mutation Engine & Error Taxonomy

We implemented five deterministic, rule-based mutation operators in `src/hd/mutations.py` that perturb factual statements while maintaining grammatical fluency:

| Error Type | Description | Original Example | Mutated Example | Testset Count |
| :--- | :--- | :--- | :--- | :--- |
| **`DATE_NUMBER`** | Alters monetary sums (`Rs.`), dates, years, statutory section numbers, or durations. | `"...directed to pay over Rs. 16 lakhs..."` | `"...directed to pay over Rs. 31 lakhs..."` | 3 |
| **`PARTY_OUTCOME`** | Swaps legal antonyms (`allowed`/`dismissed`, `petitioner`/`respondent`, `granted`/`denied`). | `"Refusing anticipatory bail to a man..."` | `"Granting anticipatory bail to a man..."` | 4 |
| **`NAME_PLACE`** | Swaps court names, cities, medical institutions, or judge names. | `"The Jammu and Kashmir and Ladakh High Court..."` | `"The Allahabad High Court..."` | 5 |
| **`NEGATION`** | Flips semantic polarity (adds or removes negation, inverts holding). | `"...a wife expressing a desire to work does not amount to cruelty..."` | `"...a wife expressing a desire to work amounts to cruelty..."` | 3 |
| **`FABRICATED`** | Injects plausible yet completely unsupported Indian legal directive templates. | *(Original case context)* | `"The High Court directed the petitioner to deposit an exemplary security sum of Rs. 15,00,000 with the Prime Minister's National Relief Fund..."` | 5 |

---

## 3. Human Review Protocol (`candidates.csv`)

Candidate mutations were automatically generated using `scripts/make_candidates.py --seed 42` into `data/testset/candidates.csv`. 

### Reviewer Checklist for Acceptance:
1. **Naturalness:** Does the edited sentence read smoothly like natural legal English without awkward phrasing?
2. **True Factual Contradiction:** Does the edit genuinely alter the factual claim so that the source judgment no longer supports it?
3. **Accidental Ground Truth:** Could the mutated fact accidentally still be true somewhere else in the judgment? (e.g. if the judgment discusses both Rs. 16 lakhs and Rs. 31 lakhs, changing one to the other might not be unsupported).

### Review Workflow:
- Open `data/testset/candidates.csv` in Excel or VS Code.
- Set `accept` to `Y` (keep) or `N` (reject).
- (Optional) If an edit is mostly good but needs tweaking, enter the refined sentence in `fixed_sentence`.
- Build the final test set using:
  ```bash
  python scripts/build_testset.py --from-review data/testset/candidates.csv
  ```
- *Note:* For automated pipelines and test suites, `data/testset/candidates_UNREVIEWED.csv` is provided as an auto-accepted development reference.

---

## 4. Test Set Composition & Core Assumption

### 4.1 Test Set Statistics (`testset.csv`)

| Metric | Count | Percentage |
| :--- | :--- | :--- |
| **Total Document Pairs** | 10 | 100.0% |
| **Total Evaluated Sentences** | 462 | 100.0% |
| **Negative Rows (Label 0: Supported)** | 442 | 95.7% |
| **Positive Rows (Label 1: Hallucinated)**| 20 | 4.3% |

For each document pair, `testset.csv` includes:
1. **Original Summary sentences:** All labeled `0` (Supported).
2. **Edited Summary sentences:** Mutated sentences labeled `1` (Hallucinated); unedited sentences labeled `0`.

### 4.2 Known Limitation: The Ground-Truth Assumption
- **Assumption:** We assume that all original sentences in the expert-written MILDSum summaries are 100% supported by the court judgment.
- **Limitation:** In practice, human legal summarizers may occasionally introduce slight extrapolations, background legal context, or interpretations not verbatim in the text of the ruling. While expert summaries are highly reliable, this assumption is an approximation standard across academic summarization benchmarks (e.g. Frank et al., Pagnoni et al.).

---

## 5. Standardized Evaluation Framework (`src/hd/evaluation.py`)

### 5.1 Metrics Contract
- **Positive Class:** Label `1` = Hallucination (the phenomenon we want to detect).
- **Hallucination Score:** Continuous score in `[0, 1]`, where **higher** indicates a higher probability of hallucination.
- **Classification Metrics (at threshold 0.5):**
  - Accuracy, Precision, Recall, F1 Score.
  - Confusion Matrix (`TP`, `FP`, `TN`, `FN`).
- **Threshold-Independent Metrics:**
  - **ROC AUC:** Receiver Operating Characteristic Area Under Curve (discrimination ability across all operating thresholds).
  - **Average Precision (PR AUC):** Area under the Precision-Recall curve, essential given class imbalance (4.3% positive).
- **Optimistic Threshold Sweep:** `best_f1_threshold()` sweeps thresholds in `[0.01, 0.99]`. It is explicitly labeled `OPTIMISTIC` in metadata because the threshold is selected on the evaluation data itself.
- **Fine-Grained Analysis:** `per_error_type_recall()` reports detection rates broken down by the 5 error categories.

---

## 6. Baseline Verification Results

We verified the evaluation engine using two sanity baselines via `python scripts/evaluate_baselines.py`:

| Metric | Oracle Baseline (`score = label`) | Random Baseline (`score ~ U(0,1)`) |
| :--- | :--- | :--- |
| **Accuracy** | **1.0000** (100.0%) | 0.5065 (50.6%) |
| **Precision** | **1.0000** (100.0%) | 0.0593 (5.9%) |
| **Recall** | **1.0000** (100.0%) | 0.7000 (70.0%) |
| **F1 Score** | **1.0000** (100.0%) | 0.1094 (10.9%) |
| **ROC AUC** | **1.0000** (Perfect) | **0.5807** (Near chance $\approx 0.50$) |
| **Average Precision** | **1.0000** (Perfect) | 0.0511 ($\approx$ base rate 4.3%) |
| **Confusion Matrix** | `TP: 20, FP: 0, TN: 442, FN: 0` | `TP: 14, FP: 222, TN: 220, FN: 6` |

**Conclusion:** The evaluation scripts correctly compute metrics, handle class imbalance, and produce expected theoretical bounds (Oracle = 1.0, Random $\approx$ chance).

---

## 7. Implications for Phase 3 (Approach A: SummaC NLI)

When building Approach A in Phase 3, we must follow these conventions:
1. **Predictions File Contract:** The checker must output a CSV with columns:
   `row_id, hallucination_score, [seconds]`
   where `row_id` matches `data/testset/testset.csv`.
2. **Score Polarity Conversion:** SummaC produces a *consistency/support score* (where higher = more consistent). We must invert this to conform to the project convention:
   $$\text{hallucination\_score} = 1.0 - \text{normalised\_summac\_score}$$
3. **Long Context Chunking:** As discovered in Phase 1, judgments average 3,078 words (~4,000+ tokens). In Phase 3, we must implement sentence retrieval / chunking before passing candidate context to SummaC's 512-token cross-encoder models.
