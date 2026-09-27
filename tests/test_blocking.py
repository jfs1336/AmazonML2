import unittest

import pandas as pd

from ml.features import generate_training_examples
from ml.evaluate import evaluate_holdout, macro_fbeta_score
from src.blocking import EnhancedNameAddressBlocker, normalize_address, normalize_business_name


class TestBlocking(unittest.TestCase):
    def test_normalization_preserves_unicode_letters_and_numbers(self):
        self.assertEqual(normalize_business_name("株式会社青木"), "株式会社青木")
        self.assertEqual(normalize_address("東京都千代田区1-2"), "東京都千代田区1 2")

    def test_macro_fbeta_includes_correct_singletons(self):
        score = macro_fbeta_score(
            ["S1-1", "S1-2"],
            {"S1-1": set(), "S1-2": {"S2-2"}},
            {"S1-1": [], "S1-2": ["S2-2"]},
        )

        self.assertEqual(score, 1.0)

    def test_macro_fbeta_scores_disjoint_nonempty_sets_as_zero(self):
        score = macro_fbeta_score(
            ["S1-1"],
            {"S1-1": {"S2-1"}},
            {"S1-1": ["S2-2"]},
        )

        self.assertEqual(score, 0.0)

    def test_holdout_evaluation_runs_on_synthetic_training_data(self):
        source1 = pd.DataFrame([
            {
                "entity_id": f"S1-{index}",
                "business_name": "Acme",
                "business_address": f"{index} Main Street",
                "country": "US",
            }
            for index in range(8)
        ])
        source2 = pd.DataFrame([
            {
                "entity_id": f"S2-{index}",
                "business_name": "Acme",
                "business_address": f"{index} Main St",
                "country": "US",
            }
            for index in range(8)
        ])
        source3 = pd.DataFrame([
            {
                "entity_id": f"S3-{index}",
                "business_name": f"Other {index}",
                "business_address": f"{100 + index} Oak Road",
                "country": "US",
            }
            for index in range(8)
        ])
        ground_truth = pd.DataFrame([
            {"source1_entity_id": f"S1-{index}", "matched_entity_ids": f"S2-{index}"}
            for index in range(8)
        ])

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = Path(tmpdir)
            source1.to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)
            source2.to_csv(train_dir / "train_source2.tsv", sep="\t", index=False)
            source3.to_csv(train_dir / "train_source3.tsv", sep="\t", index=False)
            ground_truth.to_csv(train_dir / "train_ground_truth.tsv", sep="\t", index=False)

            metrics = evaluate_holdout(train_dir, validation_size=0.25, random_state=42)

        self.assertEqual(metrics["training_entities"], 6)
        self.assertEqual(metrics["validation_entities"], 2)
        self.assertEqual(metrics["candidate_recall"], 1.0)

    def test_blocker_finds_name_and_address_variant_matches(self):
        s1 = pd.DataFrame([
            {
                "entity_id": "S1-100",
                "business_name": "Acme Pvt Ltd",
                "business_address": "45 Main Street, New York",
                "country": "US",
            }
        ])
        s2 = pd.DataFrame([
            {
                "entity_id": "S2-200",
                "business_name": "Acme Corporation",
                "business_address": "45 Main St, New York",
                "country": "US",
            }
        ])
        s3 = pd.DataFrame([
            {
                "entity_id": "S3-300",
                "business_name": "Other LLC",
                "business_address": "100 Oak Ave",
                "country": "US",
            }
        ])

        blocker = EnhancedNameAddressBlocker()
        rows = blocker.generate_candidates(s1, s2, s3)

        self.assertIn(("S1-100", "S2-200", "source2", "name_address"), rows)

    def test_blocker_handles_website_and_unit_variants(self):
        s1 = pd.DataFrame([
            {
                "entity_id": "S1-900",
                "business_name": "George Saul Inc",
                "business_address": "Unit UNIT 367, 1400 Great Wolf Drive, Village Of Lake Delton, WI",
                "country": "US",
            }
        ])
        s2 = pd.DataFrame([
            {
                "entity_id": "S2-901",
                "business_name": "georgesaul.com",
                "business_address": "1400 Great Wolf Dr, Unit UNIT 367, Baraboo, Wisconsin",
                "country": "US",
            }
        ])
        s3 = pd.DataFrame([
            {
                "entity_id": "S3-902",
                "business_name": "Other LLC",
                "business_address": "100 Oak Ave",
                "country": "US",
            }
        ])

        blocker = EnhancedNameAddressBlocker()
        rows = blocker.generate_candidates(s1, s2, s3)

        self.assertIn(("S1-900", "S2-901", "source2", "name_address"), rows)

    def test_training_examples_include_both_positive_and_negative_pairs(self):
        s1 = pd.DataFrame([
            {
                "entity_id": "S1-100",
                "business_name": "Acme Pvt Ltd",
                "business_address": "45 Main Street, New York",
                "country": "US",
            },
            {
                "entity_id": "S1-101",
                "business_name": "Beta Labs",
                "business_address": "1 Elm St",
                "country": "US",
            },
        ])
        s2 = pd.DataFrame([
            {
                "entity_id": "S2-200",
                "business_name": "Acme Corporation",
                "business_address": "45 Main St, New York",
                "country": "US",
            },
            {
                "entity_id": "S2-201",
                "business_name": "Gamma Industries",
                "business_address": "20 Pine Rd",
                "country": "US",
            },
        ])
        s3 = pd.DataFrame([
            {
                "entity_id": "S3-300",
                "business_name": "Other LLC",
                "business_address": "100 Oak Ave",
                "country": "US",
            },
            {
                "entity_id": "S3-301",
                "business_name": "Beta Consulting",
                "business_address": "1 Elm Street",
                "country": "US",
            },
        ])

        gt = pd.DataFrame([
            {"source1_entity_id": "S1-100", "matched_entity_ids": "S2-200"},
            {"source1_entity_id": "S1-101", "matched_entity_ids": "S3-301"},
        ])

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            s1.to_csv(base / "train_source1.tsv", sep="\t", index=False)
            s2.to_csv(base / "train_source2.tsv", sep="\t", index=False)
            s3.to_csv(base / "train_source3.tsv", sep="\t", index=False)
            gt.to_csv(base / "train_ground_truth.tsv", sep="\t", index=False)

            df = generate_training_examples(base)

            self.assertTrue((df["label"] == 1).any())
            self.assertTrue((df["label"] == 0).any())
            self.assertIn("S2-200", set(df["candidate_entity_id"]))
            self.assertIn("S2-201", set(df["candidate_entity_id"]))
