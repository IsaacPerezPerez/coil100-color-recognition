"""Verifica artefactos reales guardados; nunca ejecuta nuevas predicciones."""

import csv
import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from src.coil100.features import COLOR_SPACES

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts/phase3b"


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


class SavedEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not (ARTIFACTS / "summary.csv").exists():
            raise unittest.SkipTest("Fase 3B aún no ejecutada: faltan artefactos reales.")
        cls.predictions, cls.matrices, cls.metrics = {}, {}, {}
        cls.summary = {row["color_space"]: row for row in read_csv(ARTIFACTS / "summary.csv")}
        for space in COLOR_SPACES:
            suffix = space.lower()
            cls.predictions[space] = read_csv(ARTIFACTS / f"predictions_{suffix}.csv")
            with (ARTIFACTS / f"confusion_{suffix}.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.reader(stream))
            cls.assert_class_order(rows)
            cls.matrices[space] = np.array([[int(value) for value in row[1:]] for row in rows[1:]])
            cls.metrics[space] = read_csv(ARTIFACTS / f"metrics_{suffix}.csv")

    @staticmethod
    def assert_class_order(rows: list[list[str]]) -> None:
        expected = [f"obj{obj}" for obj in range(1, 21)]
        if rows[0][1:] != expected or [row[0] for row in rows[1:]] != expected:
            raise AssertionError("El orden real/predicho de clases no es obj1 a obj20.")

    def test_01_each_matrix_and_prediction_file_has_1120_observations(self) -> None:
        for space in COLOR_SPACES:
            self.assertEqual(self.matrices[space].shape, (20, 20))
            self.assertEqual(int(self.matrices[space].sum()), 1120)
            self.assertEqual(len(self.predictions[space]), 1120)
            reconstructed = np.zeros((20, 20), dtype=int)
            for row in self.predictions[space]:
                reconstructed[int(row["true_object_id"]) - 1, int(row["predicted_object_id"]) - 1] += 1
            np.testing.assert_array_equal(self.matrices[space], reconstructed)

    def test_02_each_class_has_56_real_samples_and_support(self) -> None:
        for space in COLOR_SPACES:
            np.testing.assert_array_equal(self.matrices[space].sum(axis=1), np.full(20, 56))
            self.assertEqual([int(row["object_id"]) for row in self.metrics[space]], list(range(1, 21)))
            self.assertEqual([int(row["support"]) for row in self.metrics[space]], [56] * 20)

    def test_03_diagonal_matches_correct_predictions(self) -> None:
        for space in COLOR_SPACES:
            correct = sum(row["true_object_id"] == row["predicted_object_id"] for row in self.predictions[space])
            self.assertEqual(int(np.trace(self.matrices[space])), correct)

    def test_04_global_accuracy_and_error_use_correct_over_1120(self) -> None:
        for space in COLOR_SPACES:
            correct = sum(row["true_object_id"] == row["predicted_object_id"] for row in self.predictions[space])
            self.assertEqual(int(self.summary[space]["total"]), 1120)
            self.assertEqual(int(self.summary[space]["correct"]), correct)
            self.assertAlmostEqual(float(self.summary[space]["accuracy"]), correct / 1120)
            self.assertAlmostEqual(float(self.summary[space]["error"]), 1 - correct / 1120)

    def test_05_sum_tp_matches_correct_predictions(self) -> None:
        for space in COLOR_SPACES:
            self.assertEqual(sum(int(row["TP"]) for row in self.metrics[space]), int(np.trace(self.matrices[space])))

    def test_06_sum_fp_equals_sum_fn(self) -> None:
        for space in COLOR_SPACES:
            self.assertEqual(sum(int(row["FP"]) for row in self.metrics[space]), sum(int(row["FN"]) for row in self.metrics[space]))

    def test_07_one_vs_rest_counts_sum_1120_for_each_class(self) -> None:
        for space in COLOR_SPACES:
            matrix = self.matrices[space]
            for index, row in enumerate(self.metrics[space]):
                counts = {key: int(row[key]) for key in ("TP", "FP", "TN", "FN")}
                self.assertEqual(sum(counts.values()), 1120)
                self.assertEqual(counts["TP"], int(matrix[index, index]))
                self.assertEqual(counts["FN"], int(matrix[index].sum()) - counts["TP"])
                self.assertEqual(counts["FP"], int(matrix[:, index].sum()) - counts["TP"])

    def test_08_metrics_are_in_unit_interval_and_macro_matches_classes(self) -> None:
        for space in COLOR_SPACES:
            for row in self.metrics[space]:
                for field in ("precision", "recall", "f1"):
                    self.assertTrue(0 <= float(row[field]) <= 1)
            for field in ("accuracy", "error", "precision_macro", "recall_macro", "f1_macro"):
                self.assertTrue(0 <= float(self.summary[space][field]) <= 1)
            for field, macro in (("precision", "precision_macro"), ("recall", "recall_macro"), ("f1", "f1_macro")):
                self.assertAlmostEqual(float(self.summary[space][macro]), sum(float(row[field]) for row in self.metrics[space]) / 20)

    def test_09_no_nan_or_infinite_numeric_results(self) -> None:
        for space in COLOR_SPACES:
            for row in self.metrics[space]:
                self.assertTrue(all(np.isfinite(float(value)) for value in row.values()))
            self.assertTrue(all(np.isfinite(float(value)) for field, value in self.summary[space].items() if field != "color_space"))
            self.assertTrue(all(np.isfinite(float(row["distance"])) and 0 <= float(row["distance"]) <= 1 for row in self.predictions[space]))

    def test_10_train_test_are_disjoint_and_all_neighbors_are_train(self) -> None:
        entries = read_csv(ROOT / "data/processed/inventory.csv")
        train = {row["image_path"] for row in entries if row["split"] == "train"}
        test = {row["image_path"] for row in entries if row["split"] == "test"}
        self.assertEqual(len(train), 320)
        self.assertEqual(len(test), 1120)
        self.assertFalse(train & test)
        for space in COLOR_SPACES:
            self.assertEqual({row["image_path"] for row in self.predictions[space]}, test)
            self.assertTrue(all(row["nearest_image"] in train and row["color_space"] == space for row in self.predictions[space]))

    def test_11_inventory_and_method_hashes_match_frozen_metadata(self) -> None:
        metadata = json.loads((ARTIFACTS / "evaluation_metadata.json").read_text())
        actual = hashlib.sha256((ROOT / "data/processed/inventory.csv").read_bytes()).hexdigest()
        self.assertEqual(actual, metadata["inventory_sha256_before"])
        self.assertEqual(actual, metadata["inventory_sha256_after"])
        self.assertEqual(metadata["method_sha256_before"], metadata["method_sha256_after"])
        for name, expected in metadata["method_sha256_before"].items():
            self.assertEqual(hashlib.sha256((ROOT / "src/coil100" / name).read_bytes()).hexdigest(), expected)
        self.assertEqual(metadata["prediction_count"], 3360)
        self.assertEqual(metadata["predictions_per_space"], {space: 1120 for space in COLOR_SPACES})
