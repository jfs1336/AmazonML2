# Business Entity Resolution

This repository implements a two-stage matcher for the ML Challenge 2026: normalized name/address blocking followed by a class-balanced logistic-regression model. It includes a Source 1 holdout evaluator and a submission-format validator.

## Environment and Data

Use Python 3.10 or newer. Install the pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```

Place the challenge files in this layout (the data is not included in this repository):

```text
student_resource/dataset/
  train/train_source1.tsv
  train/train_source2.tsv
  train/train_source3.tsv
  train/train_ground_truth.tsv
  test/test_source1.tsv
  test/test_source2.tsv
  test/test_source3.tsv
```

## Run the Pipeline

Evaluate a deterministic holdout split grouped by Source 1 entity. This reports candidate recall and macro F0.5, counting correctly predicted singletons as correct:

```powershell
python -m ml.evaluate --train-dir student_resource/dataset/train --validation-size 0.2 --random-state 42 --threshold 0.5
```

Train and save the model and its feature-column list:

```powershell
python -m ml.train --train-dir student_resource/dataset/train --output-dir output
```

Generate the candidate set that will be scored:

```powershell
python generate_candidate_pairs.py `
  --source1 student_resource/dataset/test/test_source1.tsv `
  --source2 student_resource/dataset/test/test_source2.tsv `
  --source3 student_resource/dataset/test/test_source3.tsv `
  --output output/candidate_pairs.tsv
```

Score those candidates and write one result row for every test Source 1 entity. The saved model is reused when present; otherwise this command trains and saves it:

```powershell
python -m ml.predict `
  --train-dir student_resource/dataset/train `
  --source1 student_resource/dataset/test/test_source1.tsv `
  --source2 student_resource/dataset/test/test_source2.tsv `
  --source3 student_resource/dataset/test/test_source3.tsv `
  --candidate output/candidate_pairs.tsv `
  --output output/matching_results.tsv `
  --model output/matching_model.joblib `
  --threshold 0.5
```

Validate output headers, Source 1 coverage, duplicate IDs, prefixes, and candidate-subset consistency:

```powershell
python student_resource/utils/validate_submission.py `
  --matching output/matching_results.tsv `
  --candidate output/candidate_pairs.tsv `
  --test-dir student_resource/dataset/test
```

Add `--check-ids` to check candidate and match IDs against the test source files; this optional check can use substantial memory on the full dataset.

## Implementation Notes

- `src/blocking.py` normalizes names and addresses and generates candidates from matching normalized keys, with country compatibility filtering.
- `ml/features.py` computes seven features: name/address Jaccard, token overlap and edit similarity, plus country equality.
- `ml/train.py` writes `matching_model.joblib` and `matching_model_features.txt` under the selected output directory.
- `ml/evaluate.py` holds out Source 1 IDs before training and reports macro F0.5 and candidate recall. Use measured output in the methodology; do not substitute unverified metrics.
- `ml/predict.py` produces the challenge-format matching TSV from the candidate file.

The candidate and matching output files, challenge datasets, trained model, and validation metrics are not present in the current checkout. They must be generated using the supplied challenge data before submission.