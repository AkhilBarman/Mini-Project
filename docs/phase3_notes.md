# Phase 3 Notes: Approach A (SummaC NLI Hallucination Detection)

**Project:** Detecting Hallucinations in AI-Generated Text using NLI and Self-Consistency Checking  
**Domain:** Indian Court Judgments (English only, MILDSum benchmark)  
**Stage:** Phase 3 Complete (SummaC NLI Integration, GPU Execution, Evaluation & Analysis)

---

## 1. Overview & Theoretical Framework

In Phase 3, we implemented **Approach A: Natural Language Inference (NLI) Hallucination Checking** using the `summac` library and `microsoft/deberta-base-mnli` as the underlying cross-encoder model.

### 1.1 Mathematical Formulation & Score Conversion
SummaC operates by segmenting the source judgment into sentences or paragraphs and computing pairwise NLI probabilities against each summary sentence. 

For each source judgment chunk $d_i$ and summary sentence $s_j$:
- $P(\text{Entailment}_{i,j})$
- $P(\text{Contradiction}_{i,j})$
- $P(\text{Neutral}_{i,j})$

SummaC defines the **Support Score** $S(s_j)$ as the maximum support across all document chunks:
$$S(s_j) = \max_i \left( P(\text{Entailment}_{i,j}) - P(\text{Contradiction}_{i,j}) \right) \in [-1.0, +1.0]$$

Because our project standard requires a **Hallucination Score** $H(s_j) \in [0.0, 1.0]$ where **higher values indicate higher likelihood of hallucination**, we apply the linear inversion:
$$H(s_j) = \frac{1.0 - S(s_j)}{2.0}$$

**Theoretical Bounds:**
- $S = +1.0$ (Maximum Entailment) $\to H = 0.0$ (Supported / Factually Consistent)
- $S = 0.0$ (Neutral / Uninformative) $\to H = 0.5$ (Uncertain)
- $S = -1.0$ (Maximum Contradiction) $\to H = 1.0$ (Hallucinated)

---

## 2. Execution Setup & Hardware Performance

Evaluating 462 sentence pairs against up to 100 judgment chunks per document involves over 35,000 NLI cross-encoder forward passes.

| Environment | Hardware | Total Run Time | Avg Time per Summary | Avg Time per Sentence |
| :--- | :--- | :--- | :--- | :--- |
| **Local Machine** | Intel/AMD CPU (8 threads) | ~42 to 45 mins (estimated) | ~120.0s | ~5.50s |
| **Google Colab** | NVIDIA Tesla T4 GPU (16 GB) | **107.01 seconds (~1.78 mins)** | **~5.35s** | **~0.23s** |

**Speedup:** Using the GPU runtime via `notebooks/02_run_approach_a_colab.ipynb` achieved a **~24x speedup**, finishing the entire test suite in under 2 minutes.

---

## 3. Evaluation Results & Baseline Comparison

We evaluated Approach A against the ground-truth test set (`data/testset/testset.csv`, 462 sentences: 16 hallucinated, 446 supported) and compared it with the Phase 2 baselines:

### 3.1 Comparative Benchmark Table

| Model / Approach | Accuracy | Precision | Recall | F1 Score | ROC AUC | PR AUC (Avg Prec) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Oracle Baseline** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Random Baseline** | 0.5108 | 0.0551 | 0.8125 | 0.1032 | 0.6326 | 0.0476 |
| **Approach A (SummaC MNLI)** | **0.2727** | **0.0455** | **1.0000** | **0.0870** | **0.8585** | **0.1382** |

*(Default decision threshold = 0.50)*

### 3.2 Breakdown by Mutation Error Type

At the default threshold (0.50), Approach A successfully detected **100% of all injected hallucinations**:

| Error Category | Detected / Total | Recall | Analysis |
| :--- | :---: | :---: | :--- |
| **`DATE_NUMBER`** | 3 / 3 | **100.0%** | Detected altered amounts (Rs. 31 lakhs vs Rs. 16 lakhs) with very high contradiction scores. |
| **`FABRICATED`** | 5 / 5 | **100.0%** | Fictitious relief fund deposit orders showed low entailment across all judgment chunks. |
| **`NAME_PLACE`** | 3 / 3 | **100.0%** | Altered High Courts (e.g., Allahabad vs J&K) triggered clear contradiction against territorial recitals. |
| **`NEGATION`** | 2 / 2 | **100.0%** | Polar reversals (cruelty established vs not established) directly triggered contradiction. |
| **`PARTY_OUTCOME`**| 3 / 3 | **100.0%** | Inverted legal rulings (granting vs refusing anticipatory bail) were flagged reliably. |
| **Overall Positive Recall** | **16 / 16** | **100.0%** | Zero false negatives at threshold 0.50. |

---

## 4. In-Depth Analysis: Strengths and Limitations

### 4.1 Strong Discriminative Capacity (ROC-AUC = 0.8585)
The **ROC-AUC of 0.8585** demonstrates that `microsoft/deberta-base-mnli` possesses strong ranking ability: hallucinated sentences systematically receive higher hallucination scores than factually supported sentences.

### 4.2 High False Positive Rate at Threshold 0.50 (Precision = 0.0455)
At the standard 0.50 threshold, the confusion matrix shows:
$$\text{TP} = 16, \quad \text{FP} = 336, \quad \text{TN} = 110, \quad \text{FN} = 0$$

**Why does this happen?**
1. **Domain Shift & Stylistic Mismatch:** `deberta-base-mnli` was trained on standard English sentence pairs (MNLI/SNLI). Legal summaries use complex, multi-clause syntactic structures and legal phrasing (e.g., *"The Division bench while dealing with a case, where the hospital had charged over and above package rates, held..."*). General NLI models struggle to find direct semantic entailment between long, specialized legal text and the summary sentence, assigning low entailment ($< 0.10$).
2. **Extreme Class Imbalance:** Real-world legal summaries are predominantly factual (446 supported vs 16 hallucinated, ~3.4% base rate). Even a modest false positive rate severely degrades precision.
3. **Threshold Calibration:** When sweeping the decision threshold, the **optimal F1 score rises to 0.2609 at threshold $\theta = 0.94$**, filtering out the majority of neutral/complex legal sentences while retaining true factual contradictions.

---

## 5. Technical Issues Solved in Phase 3

1. **NLTK Resource Availability (`LookupError: punkt_tab`):**
   - In modern NLTK (v3.9+), sentence tokenization requires `punkt_tab`. Added automatic runtime verification in `src/hd/approach_a.py` to ensure `punkt` and `punkt_tab` are downloaded without manual user intervention.
2. **Transformers 4.45+ Breaking Change (`batch_encode_plus`):**
   - Hugging Face phased out `.batch_encode_plus()` on tokenizer wrappers in favor of direct callable syntax `tokenizer(prems, hypos)`.
   - Added an automatic backward-compatibility adapter in `src/hd/approach_a.py` that wraps tokenizer calls, ensuring compatibility across all old and new `transformers` versions.
3. **Persistent Result Caching:**
   - Implemented MD5-keyed JSON disk caching (`results/cache_approach_a_mnli_base.json`), ensuring interrupted runs or re-evaluations resume instantaneously without redundant compute.

---

## 6. Readiness for Phase 4

Phase 3 is complete. The project now has:
- Working Approach A inference script (`scripts/run_approach_a.py`)
- Colab GPU workflow (`notebooks/02_run_approach_a_colab.ipynb`)
- Benchmark predictions (`results/approachA_mnli_base_predictions.csv`)
- Full evaluation metrics (`results/approachA_mnli_base_predictions_eval.json`)

**Next Step (Phase 4):** Implement **Approach B (LLM Sampling via Llama 3 + SelfCheckGPT)**, and evaluate how self-consistency compares against NLI on this legal judgment benchmark.
