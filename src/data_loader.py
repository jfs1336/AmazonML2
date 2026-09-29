from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_DIR = PROJECT_ROOT / "student_resource" / "dataset" / "train"
TEST_DIR = PROJECT_ROOT / "student_resource" / "dataset" / "test"


def load_training_data():
    source1 = pd.read_csv(
        TRAIN_DIR / "train_source1.tsv",
        sep="\t",
        encoding="utf-8-sig"
    )

    source2 = pd.read_csv(
        TRAIN_DIR / "train_source2.tsv",
        sep="\t",
        encoding="utf-8-sig"
    )

    source3 = pd.read_csv(
        TRAIN_DIR / "train_source3.tsv",
        sep="\t",
        encoding="utf-8-sig"
    )

    ground_truth = pd.read_csv(
        TRAIN_DIR / "train_ground_truth.tsv",
        sep="\t",
        encoding="utf-8-sig"
    )

    return source1, source2, source3, ground_truth


def load_test_data():
    source1 = pd.read_csv(
        TEST_DIR / "test_source1.tsv",
        sep="\t",
        encoding="utf-8-sig"
    )

    source2 = pd.read_csv(
        TEST_DIR / "test_source2.tsv",
        sep="\t",
        encoding="utf-8-sig"
    )

    source3 = pd.read_csv(
        TEST_DIR / "test_source3.tsv",
        sep="\t",
        encoding="utf-8-sig"
    )

    return source1, source2, source3


if __name__ == "__main__":

    print("Loading training data...")

    source1, source2, source3, ground_truth = load_training_data()

    print()
    print("TRAINING DATA")
    print("----------------------------")
    print("Source 1:", source1.shape)
    print("Source 2:", source2.shape)
    print("Source 3:", source3.shape)
    print("Ground Truth:", ground_truth.shape)

    print()
    print("Source 1 columns:")
    print(source1.columns.tolist())

    print()
    print("Source 2 columns:")
    print(source2.columns.tolist())

    print()
    print("Source 3 columns:")
    print(source3.columns.tolist())

    print()
    print("Ground truth columns:")
    print(ground_truth.columns.tolist())