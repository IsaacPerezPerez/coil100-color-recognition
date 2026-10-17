from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from src.coil100.dataset import (
    OBJECT_IDS,
    TRAIN_VIEW_INDICES,
    build_inventory,
    validate_dataset,
    write_inventory_csv,
)


def make_synthetic_dataset(root: Path) -> None:
    """Crea PNG mínimos válidos para pruebas, nunca usados como dataset experimental."""
    from PIL import Image

    for object_id in OBJECT_IDS:
        for angle in range(0, 360, 5):
            path = root / f"obj{object_id}__{angle}.png"
            Image.new("RGB", (2, 2), (object_id, angle % 256, 0)).save(path)


class DatasetPreparationTests(unittest.TestCase):
    def test_split_has_16_train_and_56_test_for_each_of_20_classes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "coil"
            root.mkdir()
            make_synthetic_dataset(root)
            views = validate_dataset(root)
            rows = build_inventory(views, root)

            self.assertEqual(len(rows), 20 * 72)
            self.assertEqual({row["object_id"] for row in rows}, set(OBJECT_IDS))
            for object_id in OBJECT_IDS:
                class_rows = [row for row in rows if row["object_id"] == object_id]
                self.assertEqual(len(class_rows), 72)
                self.assertEqual(sum(row["split"] == "train" for row in class_rows), 16)
                self.assertEqual(sum(row["split"] == "test" for row in class_rows), 56)
                self.assertEqual(
                    [row["angle_degrees"] for row in class_rows if row["split"] == "train"],
                    [5 * index for index in TRAIN_VIEW_INDICES],
                )

    def test_no_image_is_duplicated_between_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "coil"
            root.mkdir()
            make_synthetic_dataset(root)
            rows = build_inventory(validate_dataset(root), root)
            train = {row["image_path"] for row in rows if row["split"] == "train"}
            test = {row["image_path"] for row in rows if row["split"] == "test"}
            self.assertTrue(train.isdisjoint(test))
            self.assertEqual(len(train), 320)
            self.assertEqual(len(test), 1120)

    def test_csv_inventory_has_relative_paths_and_stable_columns(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "coil"
            root.mkdir()
            make_synthetic_dataset(root)
            rows = build_inventory(validate_dataset(root), root)
            csv_path = Path(temp_dir) / "out" / "inventory.csv"
            write_inventory_csv(rows, csv_path)
            with csv_path.open(encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                self.assertEqual(reader.fieldnames, ["object_id", "angle_degrees", "image_path", "split"])
                first = next(reader)
            self.assertEqual(first["image_path"], "obj1__0.png")


if __name__ == "__main__":
    unittest.main()
