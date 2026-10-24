# Comparative Benchmark Results

**Generated:** 2026-10-24 19:02:26
**Benchmark Dataset:** `data/testset/testset.csv` (462 sentences)

## 1. Overall Performance Metrics

| Model / Approach                     | Threshold | Accuracy | Precision | Recall | F1     | ROC-AUC | Avg Precision | TP | FP  | TN  | FN | Runtime (s) |
| ------------------------------------ | --------- | -------- | --------- | ------ | ------ | ------- | ------------- | -- | --- | --- | -- | ----------- |
| Oracle Baseline                      | 0.5       | 1.0      | 1.0       | 1.0    | 1.0    | 1.0     | 1.0           | 16 | 0   | 446 | 0  | 0.01        |
| Random Baseline                      | 0.5       | 0.5108   | 0.0551    | 0.8125 | 0.1032 | 0.6326  | 0.0476        | 13 | 223 | 223 | 3  | 0.01        |
| Approach A (SummaC NLI - Default)    | 0.5       | 0.2727   | 0.0455    | 1.0    | 0.087  | 0.8585  | 0.1382        | 16 | 336 | 110 | 0  | 107.01      |
| Approach A (SummaC NLI - Calibrated) | 0.85      | 0.7229   | 0.1111    | 1.0    | 0.2    | 0.8585  | 0.1382        | 16 | 128 | 318 | 0  | 107.01      |
| Approach B (SelfCheckGPT)            | 0.5       | 0.9242   | 0.1724    | 0.3125 | 0.2222 | 0.3715  | 0.249         | 5  | 24  | 422 | 11 | 0.5         |
| Hybrid (NLI + SelfCheck)             | 0.5       | 0.6883   | 0.1       | 1.0    | 0.1818 | 0.8521  | 0.1361        | 16 | 144 | 302 | 0  | 107.51      |

## 2. Per-Error-Type Recall Breakdown

| Model / Approach                     | DATE_NUMBER | FABRICATED | NAME_PLACE | NEGATION | PARTY_OUTCOME |
| ------------------------------------ | ----------- | ---------- | ---------- | -------- | ------------- |
| Oracle Baseline                      | 1.0         | 1.0        | 1.0        | 1.0      | 1.0           |
| Random Baseline                      | 0.6667      | 0.8        | 1.0        | 1.0      | 0.6667        |
| Approach A (SummaC NLI - Default)    | 1.0         | 1.0        | 1.0        | 1.0      | 1.0           |
| Approach A (SummaC NLI - Calibrated) | 1.0         | 1.0        | 1.0        | 1.0      | 1.0           |
| Approach B (SelfCheckGPT)            | 0.0         | 1.0        | 0.0        | 0.0      | 0.0           |
| Hybrid (NLI + SelfCheck)             | 1.0         | 1.0        | 1.0        | 1.0      | 1.0           |
