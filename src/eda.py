from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "student_resource" / "dataset" / "train"


def inspect_file(filename):
    path = TRAIN_DIR / filename

    print("\n" + "=" * 60)
    print(filename)
    print("=" * 60)

    df = pd.read_csv(
        path,
        sep="\t",
        encoding="utf-8-sig"
    )

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isna().sum())

    print("\nDuplicate rows:", df.duplicated().sum())

    for column in df.columns:
        print(f"\n--- {column} ---")
        print("Unique:", df[column].nunique(dropna=True))
        print("Sample:")
        print(df[column].dropna().head(5).tolist())

    return df


if __name__ == "__main__":

    print("BUSINESS ENTITY RESOLUTION - INITIAL EDA")

    source1 = inspect_file("train_source1.tsv")
    source2 = inspect_file("train_source2.tsv")
    source3 = inspect_file("train_source3.tsv")

    print("\n" + "=" * 60)
    print("GROUND TRUTH")
    print("=" * 60)

    ground_truth = pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t",
        encoding="utf-8-sig"
    )

    print("Shape:", ground_truth.shape)
    print("\nColumns:")
    print(ground_truth.columns.tolist())

    print("\nMissing values:")
    print(ground_truth.isna().sum())

    print("\nSample:")
    print(ground_truth.head(10).to_string(index=False))