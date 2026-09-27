from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Mapping, Sequence

from sklearn.linear_model import LogisticRegression

from ml.features import (
    DEFAULT_FEATURE_COLUMNS,
    generate_training_examples,
    predict_matches_for_candidate_map,
)
from src.blocking import EnhancedNameAddressBlocker, load_source_file, parse_ground_truth


def macro_fbeta_score(
    source1_ids: Sequence[str],
    true_matches: Mapping[str, set[str]],
    predictions: Mapping[str, Sequence[str]],
    beta: float = 0.5,
) -> float:
    if beta <= 0:
        raise ValueError("beta must be greater than zero")
    beta_squared = beta**2
    scores: list[float] = []

    for source1_id in source1_ids:
        expected = set(true_matches.get(source1_id, set()))
        predicted = set(predictions.get(source1_id, []))
        if not expected and not predicted:
            scores.append(1.0)
            continue
        if not expected or not predicted:
            scores.append(0.0)
            continue

        true_positives = len(expected & predicted)
        if not true_positives:
            scores.append(0.0)
            continue
        precision = true_positives / len(predicted)
        recall = true_positives / len(expected)
        denominator = beta_squared * precision + recall
        scores.append((1 + beta_squared) * precision * recall / denominator)

    return sum(scores) / len(scores) if scores else 0.0


def evaluate_holdout(
    train_dir: Path,
    validation_size: float = 0.2,
    random_state: int = 42,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")

    source1 = load_source_file(train_dir / "train_source1.tsv")
    source2 = load_source_file(train_dir / "train_source2.tsv")
    source3 = load_source_file(train_dir / "train_source3.tsv")
    ground_truth = parse_ground_truth(train_dir / "train_ground_truth.tsv")
    source1_ids = source1["entity_id"].astype(str).tolist()
    if len(source1_ids) < 2:
        raise ValueError("At least two Source 1 entities are required for holdout evaluation.")

    rng = random.Random(random_state)
    shuffled_ids = source1_ids.copy()
    rng.shuffle(shuffled_ids)
    validation_count = min(max(1, round(len(shuffled_ids) * validation_size)), len(shuffled_ids) - 1)
    validation_ids = set(shuffled_ids[:validation_count])
    training_ids = set(shuffled_ids[validation_count:])

    training_examples = generate_training_examples(train_dir, source1_ids=training_ids)
    feature_columns = [column for column in DEFAULT_FEATURE_COLUMNS if column in training_examples.columns]
    if training_examples.empty or training_examples["label"].nunique() < 2:
        raise ValueError("The training split must contain both positive and negative candidate pairs.")

    model = LogisticRegression(max_iter=2000, class_weight="balanced")
    model.fit(
        training_examples[feature_columns].fillna(0.0).astype(float),
        training_examples["label"].astype(int),
    )

    candidate_rows = EnhancedNameAddressBlocker().generate_candidates(source1, source2, source3)
    candidate_map: dict[str, list[str]] = {source1_id: [] for source1_id in validation_ids}
    for source1_id, candidate_id, _, _ in candidate_rows:
        if source1_id in validation_ids:
            candidate_map[source1_id].append(candidate_id)

    validation_source1 = source1[source1["entity_id"].astype(str).isin(validation_ids)]
    predictions = predict_matches_for_candidate_map(
        model,
        feature_columns,
        validation_source1,
        source2,
        source3,
        candidate_map,
        threshold,
    )
    validation_truth = {source1_id: ground_truth.get(source1_id, set()) for source1_id in validation_ids}
    total_true = sum(len(matches) for matches in validation_truth.values())
    total_recalled = sum(
        len(validation_truth[source1_id] & set(candidate_map[source1_id]))
        for source1_id in validation_ids
    )

    return {
        "training_entities": len(training_ids),
        "validation_entities": len(validation_ids),
        "candidate_recall": total_recalled / total_true if total_true else 1.0,
        "macro_f0_5": macro_fbeta_score(validation_ids, validation_truth, predictions, beta=0.5),
        "threshold": threshold,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the matcher on a Source-1 holdout split.")
    parser.add_argument("--train-dir", type=Path, default=Path("student_resource/dataset/train"))
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    metrics = evaluate_holdout(args.train_dir, args.validation_size, args.random_state, args.threshold)
    print(f"Training Source 1 entities: {metrics['training_entities']}")
    print(f"Validation Source 1 entities: {metrics['validation_entities']}")
    print(f"Candidate recall: {metrics['candidate_recall']:.6f}")
    print(f"Macro F0.5: {metrics['macro_f0_5']:.6f}")
    print(f"Threshold: {metrics['threshold']:.3f}")


if __name__ == "__main__":
    main()