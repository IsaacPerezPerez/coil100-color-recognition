import unittest

import numpy as np

from src.coil100.metrics import confusion_matrix, recognition_metrics


class MetricFormulaTests(unittest.TestCase):
    def test_known_confusion_matrix_counts_and_formulas(self) -> None:
        rows, summary = recognition_metrics(np.array([[3, 1], [2, 4]]), (1, 2))
        self.assertEqual([rows[0][key] for key in ("TP", "FP", "TN", "FN", "support")], [3, 2, 4, 1, 4])
        self.assertEqual([rows[1][key] for key in ("TP", "FP", "TN", "FN", "support")], [4, 1, 3, 2, 6])
        self.assertAlmostEqual(rows[0]["precision"], 3 / 5)
        self.assertAlmostEqual(rows[0]["recall"], 3 / 4)
        self.assertAlmostEqual(rows[0]["f1"], 6 / 9)
        self.assertAlmostEqual(summary["accuracy"], 7 / 10)
        self.assertAlmostEqual(summary["error"], 3 / 10)
        self.assertAlmostEqual(summary["precision_macro"], (3 / 5 + 4 / 5) / 2)
        self.assertAlmostEqual(summary["recall_macro"], (3 / 4 + 4 / 6) / 2)
        self.assertAlmostEqual(summary["f1_macro"], (6 / 9 + 8 / 11) / 2)

    def test_rows_are_real_and_columns_are_predicted_in_fixed_order(self) -> None:
        actual = confusion_matrix([2, 1, 1], [1, 1, 2], (1, 2, 3))
        np.testing.assert_array_equal(actual, [[1, 1, 0], [1, 0, 0], [0, 0, 0]])
        with self.assertRaises(ValueError):
            confusion_matrix([1], [4], (1, 2, 3))

    def test_zero_divisions_and_missing_classes_are_included_in_macro(self) -> None:
        rows, summary = recognition_metrics(np.array([[2, 0, 0], [0, 0, 0], [0, 0, 0]]), (1, 2, 3))
        for row in rows[1:]:
            self.assertEqual([row[key] for key in ("precision", "recall", "f1")], [0, 0, 0])
        self.assertAlmostEqual(summary["f1_macro"], 1 / 3)
        rows, summary = recognition_metrics(np.zeros((3, 3), dtype=int), (1, 2, 3))
        self.assertEqual(summary["accuracy"], 0)
        self.assertTrue(all(np.isfinite(value) for row in rows for value in row.values()))

    def test_global_accuracy_is_not_average_one_vs_rest_accuracy(self) -> None:
        rows, summary = recognition_metrics(np.array([[8, 1, 1], [0, 0, 0], [0, 0, 0]]), (1, 2, 3))
        self.assertAlmostEqual(summary["accuracy"], 0.8)
        ovr_average = sum((row["TP"] + row["TN"]) / 10 for row in rows) / 3
        self.assertNotAlmostEqual(summary["accuracy"], ovr_average)

    def test_invalid_counts_are_rejected(self) -> None:
        for matrix in (np.array([[1, -1], [0, 2]]), np.ones((2, 3)), np.array([[1, np.nan], [0, 1]]), np.full((2, 2), 0.5)):
            with self.assertRaises(ValueError):
                recognition_metrics(matrix, (1, 2))
