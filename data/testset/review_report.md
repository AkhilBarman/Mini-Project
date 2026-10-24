# Human Review Report: Candidate Mutations Review

**Date:** 2026-10-24  
**Dataset:** MILDSum Sample Pairs (`data/processed/sample_pairs.jsonl`)  
**Input:** `data/testset/candidates.csv` (20 candidate rows generated from 10 sample judgment-summary pairs, seed 42)  
**Review Standard:** Strict independent reviewer evaluation under Phase 2, Part B Step 3 instructions.

---

## 1. Summary of Decisions per Error Type

| Error Type | Total Candidates | Accepted (As-Is) | Fixed & Accepted | Rejected | Acceptance Rate |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **DATE_NUMBER** | 3 | 3 | 0 | 0 | 100.0% |
| **FABRICATED** | 5 | 5 | 0 | 0 | 100.0% |
| **NAME_PLACE** | 5 | 3 | 0 | 2 | 60.0% |
| **NEGATION** | 3 | 1 | 1 | 1 | 66.7% |
| **PARTY_OUTCOME** | 4 | 3 | 0 | 1 | 75.0% |
| **TOTAL** | **20** | **15** | **1** | **4** | **80.0%** |

- **Total Accepted for Evaluation Test Set:** 16 / 20 (80.0%)
- **Total Rejected:** 4 / 20 (20.0%)

---

## 2. Main Reasons for Rejection

1. **Label Noise: Original Sentence Unsupported by Judgment Text (3 rows rejected)**
   - Under Rule 2d (*"confirm the original sentence is actually supported by the judgment. If the ORIGINAL looks unsupported, reject the row and mark 'original seems unsupported' in notes"*), rows where the original summary sentence relies on external background knowledge or stripped header metadata must be rejected.
   - In raw MILDSum judgments, the formal court coram (caption listing presiding judges) is occasionally stripped. As a result, original sentences citing specific authoring judges cannot be verified against the judgment text itself, creating false positives/neutrals for downstream NLI models.
   - Similarly, summary sentences that recite statutory sub-clauses (e.g. Section 37 twin conditions) not quoted in the judgment text constitute external exposition.

2. **Semantic Incoherence in Automated Mutation (1 row rejected)**
   - Under Rule 2a (*"Natural: grammatical and reads like a real court summary"*) and Rule 2b (*"Really wrong: change meaning so judgment contradicts it, reject harmless or cosmetic edits"*), injecting negation into a subordinate clause caused logical contradiction in counsel withdrawal behavior.

---

## 3. Detailed Review of Rejected and Fixed Rows

### A. Label-Noise Cases (Originals Unsupported by Judgment)
- **`cand_004` (Sample_2, NAME_PLACE): REJECTED (N)**
  - *Original Sentence:* `A bench comprising Justice Waseem Sadiq Nargal observed, "As every Government/executive order by virtue of a policy has prospective operation..."`
  - *Edited Sentence:* `A bench comprising Justice Ali Mohammad Magrey observed, "As every Government/executive order..."`
  - *Evidence:* In `Sample_2/EN_Judgment.txt`, the judgment text begins directly at paragraph 1 without the formal coram heading. Neither "Justice Waseem Sadiq Nargal" nor "Justice Ali Mohammad Magrey" appears anywhere in the judgment text. Evaluating this mutation against the text would evaluate a fact unmentioned in the source document.
- **`cand_015` (Sample_8, PARTY_OUTCOME): REJECTED (N)**
  - *Original Sentence:* `Section 37 states that bail should not be granted to an accused unless the accused is able to satisfy twin conditions i.e. reasonable ground for believing that the accused is not guilty of such an offence and that the accused would not commit an offence or is not likely to commit an offence, if granted bail.`
  - *Edited Sentence:* `Section 37 states that bail should not be denied to an accused unless the accused is able to satisfy twin conditions...`
  - *Evidence:* The judgment in `Sample_8` only states: *"Since the collection of sample itself is faulty, the rigours of Section 37 of the NDPS Act will not be applicable."* The judgment never quotes or defines the twin conditions. The original summary sentence was added by the annotator from general statutory knowledge.
- **`cand_018` (Sample_9, NAME_PLACE): REJECTED (N)**
  - *Original Sentence:* `Justice Dinesh Kumar Sharma made the said observation while expressing displeasure over the conduct of a Family judge who had shared his personal mobile number with both the parties...`
  - *Edited Sentence:* `Justice Prathiba M. Singh made the said observation while expressing displeasure over the conduct of a Family judge...`
  - *Evidence:* The High Court judge's name "Justice Dinesh Kumar Sharma" does not appear in `Sample_9` judgment text. Furthermore, "Singh" appears twice in the judgment referring to the Family Court judge (`Sh. Sanjeev Kumar Singh`), making the substitution confusing and the original unsupported.

### B. Awkward / Illogical Automated Mutation
- **`cand_014` (Sample_7, NEGATION): REJECTED (N)**
  - *Original Sentence:* `It also voiced concern over the unfortunate trend being adopted by "unscrupulous litigants" in which anticipatory bail is argued and when the Court is about to dismiss the petition, in order to avoid a detailed adverse order, the counsel seeks to withdraw the petition...`
  - *Edited Sentence:* `It also voiced concern over the unfortunate trend being adopted by "unscrupulous litigants" in which anticipatory bail is not argued and when the Court is about to dismiss the petition...`
  - *Evidence:* Inserting `not` before `argued` produces an ungrammatical and nonsensical procedural scenario (counsel withdrawing an unargued petition to avoid an adverse order).

### C. Awkward Phrasing Rewritten via `fixed_sentence`
- **`cand_006` (Sample_3, NEGATION): ACCEPTED WITH FIX (Y)**
  - *Original Sentence:* `Division bench of Justices Atul Chandurkar and Urmila Joshi-Phalke was dealing with an appeal against family court's refusal to grant divorce to the husband on the ground that cruelty is not proven.`
  - *Edited Sentence:* `Division bench of Justices Atul Chandurkar and Urmila Joshi-Phalke was dealing with an appeal against family court's refusal to grant divorce to the husband on the ground that cruelty is clearly proven.`
  - *Issue:* Stating that the family court *refused* divorce *because* cruelty was proven is legally illogical.
  - *Fixed Sentence:* `Division bench of Justices Atul Chandurkar and Urmila Joshi-Phalke was dealing with an appeal against family court's grant of divorce to the husband on the ground that cruelty was proven.`
  - *Evidence:* In `Sample_3` judgment paragraph 2, the Family Court dismissed the husband's petition (refused divorce). Stating that the Family Court granted divorce is natural, grammatical, and directly contradicted by the judgment.

---

## 4. Sentence-Split Analysis

- **Total summary sentences reviewed across all candidate pairs:** 20 candidate sentences out of 462 total sentences.
- **Split Quality Check:** All 20 sentences exhibited proper start-of-sentence capitalization and proper terminal punctuation (`.` or `."`).
- **Legal Abbreviations:** The legal abbreviation-aware splitter in `src/hd/text_utils.py` correctly avoided splitting on abbreviations present in the candidates:
  - `ADJ` (`cand_002`): Correctly kept with `Dinesh Kumar`.
  - `Rs.` (`cand_001`, `cand_009`, `cand_012`): Correctly kept with amount tokens.
  - `u/s` / `No.` / `dated`: Correctly kept intact.
- **Sentence-Split Bugs Found:** **0**. No orphan fragments or severed citations were detected.

---

## 5. Test Set Balance & Next Steps

After strict review, the 16 accepted candidates provide solid representation across all 5 legal error categories:
- **FABRICATED:** 5 accepted
- **DATE_NUMBER:** 3 accepted
- **NAME_PLACE:** 3 accepted
- **PARTY_OUTCOME:** 3 accepted
- **NEGATION:** 2 accepted

All 5 error categories have sufficient representation to proceed to testset construction (`build_testset.py --from-review data/testset/candidates_reviewed.csv`), yielding a clean, noise-free ground truth evaluation set.
