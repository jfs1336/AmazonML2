from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd

from generate_candidate_pairs import write_candidate_rows
from ml.features import load_candidate_pairs, predict_matches_for_candidate_map, train_model
from src.blocking import load_source_file


def write_matching_results(
    train_dir: Path,
    s1_path: Path,
    s2_path: Path,
    s3_path: Path,
    output_path: Path,
    candidate_path: Path,
    threshold: float = 0.5,
    model_path: Path | None = None,
) -> None:
    if not candidate_path.exists():
        write_candidate_rows(s1_path, s2_path, s3_path, candidate_path)

    candidate_map = load_candidate_pairs(candidate_path)
    s1_df = load_source_file(s1_path)
    s2_df = load_source_file(s2_path)
    s3_df = load_source_file(s3_path)

    import joblib

    if model_path is not None and model_path.exists():
        feature_path = model_path.with_name(f"{model_path.stem}_features.txt")
        if not feature_path.exists():
            raise FileNotFoundError(f"Model feature list not found: {feature_path}")
        model = joblib.load(model_path)
        feature_cols = feature_path.read_text(encoding="utf-8").splitlines()
    else:
        model, feature_cols = train_model(train_dir)
        if model_path is not None:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, model_path)
            feature_path = model_path.with_name(f"{model_path.stem}_features.txt")
            feature_path.write_text("\n".join(feature_cols), encoding="utf-8")

    predicted = predict_matches_for_candidate_map(model, feature_cols, s1_df, s2_df, s3_df, candidate_map, threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for item in s1_df["entity_id"].astype(str).tolist():
            matches = predicted.get(item, [])
            writer.writerow([item, ",".join(matches)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the matching model over blocking candidates.")
    parser.add_argument("--train-dir", type=Path, default=Path("student_resource/dataset/train"))
    parser.add_argument("--source1", type=Path, default=Path("student_resource/dataset/test/test_source1.tsv"))
    parser.add_argument("--source2", type=Path, default=Path("student_resource/dataset/test/test_source2.tsv"))
    parser.add_argument("--source3", type=Path, default=Path("student_resource/dataset/test/test_source3.tsv"))
    parser.add_argument("--candidate", type=Path, default=Path("output/candidate_pairs.tsv"))
    parser.add_argument("--output", type=Path, default=Path("output/matching_results.tsv"))
    parser.add_argument("--model", type=Path, default=Path("output/matching_model.joblib"))
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    write_matching_results(
        train_dir=args.train_dir,
        s1_path=args.source1,
        s2_path=args.source2,
        s3_path=args.source3,
        output_path=args.output,
        candidate_path=args.candidate,
        threshold=args.threshold,
        model_path=args.model,
    )
    print(f"Wrote matching results to {args.output}")


if __name__ == "__main__":
    main()
