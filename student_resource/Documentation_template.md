# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Arete  
**Team Members:** Joe, Abhinav, Chanchal, Immanuel  
**Submission Date:** 27/09/2026

---

## 1. Executive Summary
Our solution follows a two-stage entity-resolution pipeline. It generates candidates by indexing normalized business names and addresses, then scores each candidate pair with a class-balanced logistic-regression classifier. The current implementation uses seven token-overlap, edit-similarity, and country features. The default decision threshold is 0.5; it is configurable and should be assessed on a held-out validation set rather than described as calibrated.

---

## 2. Methodology

### 2.1 Problem Analysis
The data is highly noisy and heterogeneous across sources. Business names frequently vary due to abbreviations, legal suffixes, punctuation, transliteration, and formatting inconsistencies, while addresses may differ in road/street naming, missing components, landmark references, or incomplete postal information. There are also country-level differences and partial matches that are not always identical at the string level. Because the challenge scoring is precision-heavy, a purely “loose” blocking strategy would create too many false candidates, while a strict strategy would risk missing valid matches.

### 2.2 Solution Strategy
We used a two-stage pipeline:

1. Candidate generation through normalized name + address blocking
2. Match scoring with a logistic-regression classifier trained on positive and sampled negative pairs

This is a hybrid blocking + classifier approach, designed to balance recall in the first stage with precision in the second stage.

**Approach Type:** Hybrid (Blocking + Classifier)  
**Core Approach:** Normalized-name and normalized-address indexes reduce the comparison space; a compact seven-feature model ranks the remaining pairs. Unicode letters and numbers are preserved during NFKC-based normalization.

---

## 3. Candidate Generation (Blocking)
The blocking stage creates a candidate set for each Source 1 entity by indexing records from Sources 2 and 3 using normalized text forms. We did not rely on exact raw string equality; instead, we normalized name and address fields to reduce common noise patterns such as abbreviations, punctuation, legal suffixes, and address shorthand.

- **Blocking keys used:** normalized business name variants, normalized address strings, source-country consistency checks
- **Candidate pairs generated:** `generate_candidate_pairs.py` writes one row per Source 1 record to `output/candidate_pairs.tsv`, including empty candidate lists. This output is generated when the test data is available; it is not included in the current checkout.
- **How we ensured true matches were not lost:**
  - business names are NFKC-normalized, case-folded to lowercase, cleaned of punctuation, and normalized for common legal suffixes and domain-style tokens
  - address strings were normalized to a comparable representation that removes noisy markers like landmarks, country labels, and repeated apartment/unit fragments
  - candidate generation combined both name and address evidence, and pairs were filtered by country compatibility when both country values were available
  - Unicode letters and numbers are preserved; candidate generation matches exact normalized keys, not fuzzy similarities

The generated candidate file is the set passed to inference. The blocker admits a pair when either its normalized name key or normalized address key matches, provided known country values do not conflict. Candidate recall must be measured on training data; it is not guaranteed by the rules themselves.

---

## 4. Matching Model

**Features used:**
- `name_jaccard`: Jaccard similarity of normalized business-name tokens
- `name_token_overlap`: intersection size divided by the larger name-token set
- `name_edit_similarity`: `difflib.SequenceMatcher` ratio on normalized names
- `address_jaccard`: Jaccard similarity of normalized address tokens
- `address_token_overlap`: intersection size divided by the larger address-token set
- `address_edit_similarity`: `difflib.SequenceMatcher` ratio on normalized addresses
- `country_match`: equality of lowercased country strings

**Model type:** Logistic Regression (`max_iter=2000`, `class_weight="balanced"`)

**Threshold:** 0.5 by default, configurable at prediction and evaluation time. It has not been tuned or calibrated in the checked-in code.

The training pipeline labels blocked pairs from ground truth and samples additional non-blocked negative pairs. The negative sample count varies by Source 1 entity (up to ten). `ml/evaluate.py` creates a deterministic holdout split by Source 1 ID and reports candidate recall and macro F0.5, including correct singleton predictions.

---

## 5. Results & Error Analysis

- **F0.5 Score (macro):** not yet recorded. Run `python -m ml.evaluate` with the training dataset and report its measured result here.
- **Candidate recall:** not yet recorded for the current blocker. The same validation command reports it.
- **Error analysis:** no labeled validation predictions are included in this checkout, so false-positive and false-negative patterns have not been measured. Likely failure cases to inspect include shared business names, partial addresses, and records whose normalized keys do not collide.

The main limitation is the exact-key blocking rule: fuzzy similarities are applied only after blocking, so a true match with neither a shared normalized name nor a shared normalized address cannot be scored. Prediction currently loads the source tables and candidate map into memory; full-scale resource use should be measured on the challenge data.

---

## 6. Conclusion
The repository contains an executable blocking and matching pipeline, a holdout evaluator, and a submission validator. Its quality and full-scale behavior remain to be established by running the evaluator and generating the required outputs with the challenge dataset.

---

## Appendix

### A. Code Artefacts
The runnable pipeline is organized as follows:

- `src/blocking.py` — normalization helpers, indexing logic, candidate generation, and evaluation utilities
- `ml/features.py` — feature construction for candidate pairs and model training logic
- `ml/train.py` — train and save the logistic-regression model and feature list
- `ml/evaluate.py` — deterministic Source 1 holdout evaluation using macro F0.5
- `ml/predict.py` — score candidates and write `matching_results.tsv`, reusing or creating the saved model
- `generate_candidate_pairs.py` — end-to-end candidate generation for the test data
- `student_resource/utils/validate_submission.py` — validate submission formatting and ID coverage
- `output/` — generated artifacts; not present until the pipeline is run

To reproduce the outputs:

1. Install dependencies from `requirements.txt` and place the provided data under `student_resource/dataset/{train,test}/`.
2. Run `python -m ml.evaluate --train-dir student_resource/dataset/train` and record the measured validation results above.
3. Run `python -m ml.train --train-dir student_resource/dataset/train --output-dir output`.
4. Generate test candidates with `python generate_candidate_pairs.py`.
5. Run `python -m ml.predict` and validate the two TSV files with `student_resource/utils/validate_submission.py`.

Commands and arguments are documented in the repository-root `README.md`.

### B. Additional Results
No candidate or matching output files are included in the current checkout. The validator checks the generated TSV headers, required Source 1 coverage, duplicate IDs, allowed ID prefixes, and (optionally) ID existence and candidate-subset consistency. 
