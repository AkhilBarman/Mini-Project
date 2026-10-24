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

| Error Type | Description | Original Example | Mutated Example | Reviewed Testset Count |
| :--- | :--- | :--- | :--- | :--- |
| **`DATE_NUMBER`** | Alters monetary sums (`Rs.`), dates, years, statutory section numbers, or durations. | `"...directed to pay over Rs. 16 lakhs..."` | `"...directed to pay over Rs. 31 lakhs..."` | 3 |
| **`PARTY_OUTCOME`** | Swaps legal antonyms (`allowed`/`dismissed`, `petitioner`/`respondent`, `granted`/`denied`). | `"Refusing anticipatory bail to a man..."` | `"Granting anticipatory bail to a man..."` | 3 |
| **`NAME_PLACE`** | Swaps court names, cities, medical institutions, or judge names. | `"The Jammu and Kashmir and Ladakh High Court..."` | `"The Allahabad High Court..."` | 3 |
| **`NEGATION`** | Flips semantic polarity (adds or removes negation, inverts holding). | `"...a wife expressing a desire to work does not amount to cruelty..."` | `"...a wife expressing a desire to work amounts to cruelty..."` | 2 |
| **`FABRICATED`** | Injects plausible yet completely unsupported Indian legal directive templates. | *(Original case context)* | `"The High Court directed the petitioner to deposit an exemplary security sum of Rs. 15,00,000 with the Prime Minister's National Relief Fund..."` | 5 |

---

## 3. Human Review Protocol & Findings

Candidate mutations were automatically generated using `scripts/make_candidates.py --seed 42` into `data/testset/candidates.csv`. 

### Review Workflow & Artifacts:
- **Independent Review:** Evaluated every candidate against source judgment text in `sample_pairs.jsonl`.
- **Outputs Created:**
  - `data/testset/candidates_reviewed.csv`: 20 candidates evaluated (15 accepted as-is, 1 fixed, 4 rejected).
  - `data/testset/review_report.md`: Detailed breakdown of acceptance rates, reasons for rejection, and label noise cases.
  - `data/testset/spot_check.csv`: 15-row sample (seed 42) mixing accepted and rejected rows, with borderline/rejected rows marked `NEEDS HUMAN CHECK`.
- **Command to build final dataset:**
  ```bash
  python scripts/build_testset.py --from-review data/testset/candidates_reviewed.csv
  ```

### Key Review Findings & Rejections:
1. **Coram/Caption Stripping in Dataset:** In raw MILDSum judgments, the formal court header/coram (listing presiding judges) is occasionally missing. Original summaries naming judges cannot be verified against the judgment text itself (`cand_004`, `cand_018`). These were rejected to avoid label noise.
2. **External Legal Exposition in Summaries:** In `cand_015`, the summary writer explained the statutory twin conditions of Section 37 NDPS Act from background knowledge, but the judgment never quoted those conditions. Rejected under Rule 2d.
3. **Semantic Incoherence in Automated Negation:** In `cand_014`, inserting `not` before `argued` created an illogical scenario (counsel withdrawing an unargued petition to avoid an adverse order). Rejected under Rule 2a.
4. **Awkward Phrasing Fixed:** In `cand_006`, the edit *"refusal to grant divorce on the ground that cruelty is clearly proven"* was rewritten in `fixed_sentence` to *"appeal against family court's grant of divorce to the husband on the ground that cruelty was proven"*, making it natural while directly contradicted by the judgment.

---

## 4. Test Set Composition & Core Assumption

### 4.1 Test Set Statistics (`testset.csv`)

| Metric | Count | Percentage |
| :--- | :--- | :--- |
| **Total Document Pairs** | 10 | 100.0% |
| **Total Evaluated Sentences** | 462 | 100.0% |
| **Negative Rows (Label 0: Supported)** | 446 | 96.5% |
| **Positive Rows (Label 1: Hallucinated)**| 16 | 3.5% |

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
  - **Average Precision (PR AUC):** Area under the Precision-Recall curve, essential given class imbalance (3.5% positive).
- **Optimistic Threshold Sweep:** `best_f1_threshold()` sweeps thresholds in `[0.01, 0.99]`. It is explicitly labeled `OPTIMISTIC` in metadata because the threshold is selected on the evaluation data itself.
- **Fine-Grained Analysis:** `per_error_type_recall()` reports detection rates broken down by the 5 error categories.

---

## 6. Baseline Verification Results

We verified the evaluation engine on the reviewed test set using two sanity baselines via `python scripts/evaluate_baselines.py`:

| Metric | Oracle Baseline (`score = label`) | Random Baseline (`score ~ U(0,1)`) |
| :--- | :--- | :--- |
| **Accuracy** | **1.0000** (100.0%) | 0.5108 (51.1%) |
| **Precision** | **1.0000** (100.0%) | 0.0551 (5.5%) |
| **Recall** | **1.0000** (100.0%) | 0.8125 (81.3%) |
| **F1 Score** | **1.0000** (100.0%) | 0.1032 (10.3%) |
| **ROC AUC** | **1.0000** (Perfect) | **0.6326** (Near chance $\approx 0.50$) |
| **Average Precision** | **1.0000** (Perfect) | 0.0476 ($\approx$ base rate 3.5%) |
| **Confusion Matrix** | `TP: 16, FP: 0, TN: 446, FN: 0` | `TP: 13, FP: 223, TN: 223, FN: 3` |

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
