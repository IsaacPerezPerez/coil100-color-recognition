import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from src.coil100.audit_training import audit_training, leave_one_out
from src.coil100.classifier import TrainingReference
from src.coil100.dataset import build_inventory, write_inventory_csv
from src.coil100.features import COLOR_SPACES
from src.coil100.inventory import InventoryEntry


class LeaveOneOutTests(unittest.TestCase):
    def test_audit_serializes_results_and_opens_only_train_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = root / "inventory.csv"
            views = {obj: [(angle, root / f"obj{obj}__{angle}.png") for angle in range(0, 360, 5)] for obj in range(1, 21)}
            rows = build_inventory(views, root)
            write_inventory_csv(rows, inventory)
            before = inventory.read_bytes()
            allowed = {(root / row["image_path"]).resolve() for row in rows if row["split"] == "train"}

            def training_only(path: Path) -> np.ndarray:
                self.assertIn(path.resolve(), allowed)
                return np.full((8, 8, 3), (0, 0, 255), dtype=np.uint8)

            def stub_loo(references: list[TrainingReference]) -> list[dict]:
                return [{"color_space": space, "correct": True} for ref in references for space in COLOR_SPACES]

            output = root / "audit"
            with patch("src.coil100.audit_training.load_bgr", side_effect=training_only), patch("src.coil100.audit_training.leave_one_out", side_effect=stub_loo):
                result = audit_training(inventory, root, output)
            self.assertEqual(result, json.loads((output / "audit_summary.json").read_text(encoding="utf-8")))
            self.assertEqual(result["train_count"], 320)
            self.assertEqual(result["test_images_loaded"], 0)
            self.assertEqual(result["empty_masks"], 0)
            self.assertTrue((output / "train_mask_contact_sheet.png").is_file())
            self.assertEqual(inventory.read_bytes(), before)

    def test_query_is_excluded_and_remaining_reference_determines_prediction(self) -> None:
        references = []
        for object_id, bin_index in ((1, 0), (2, 1)):
            histogram = np.zeros(512, np.float32)
            histogram[bin_index] = 1
            entry = InventoryEntry(object_id, 0, Path(f"obj{object_id}__0.png"), "train")
            references.append(TrainingReference(entry, {space: histogram for space in COLOR_SPACES}))
        rows = leave_one_out(references)
        self.assertEqual(len(rows), 6)
        for row in rows:
            self.assertNotEqual(row["query_image"], row["nearest_image"])
            self.assertNotEqual(row["true_object_id"], row["predicted_object_id"])
            self.assertEqual(row["reference_count"], 1)
            self.assertAlmostEqual(row["distance"], 1)
            self.assertFalse(row["correct"])

    def test_singleton_and_duplicate_paths_are_rejected(self) -> None:
        histogram = np.full(512, 1 / 512, np.float32)
        entry = InventoryEntry(1, 0, Path("obj1__0.png"), "train")
        reference = TrainingReference(entry, {space: histogram for space in COLOR_SPACES})
        for references in ([reference], [reference, reference]):
            with self.assertRaises(ValueError):
                leave_one_out(references)
