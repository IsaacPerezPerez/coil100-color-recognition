"""Pruebas sintéticas rápidas: no requieren COIL-100 ni consultan imágenes test."""

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.coil100.classifier import Histogram1NN, TrainingReference
from src.coil100.dataset import CSV_FIELDS, TRAIN_VIEW_INDICES
from src.coil100.features import COLOR_SPACES, convert_color, extract_all, extract_histogram
from src.coil100.inventory import InventoryEntry, read_inventory
from src.coil100.preprocessing import load_bgr, segment_object


def color_image(bgr: tuple[int, int, int]) -> np.ndarray:
    return np.full((8, 8, 3), bgr, dtype=np.uint8)


def reference(object_id: int, bgr: tuple[int, int, int], split: str = "train") -> TrainingReference:
    image = color_image(bgr)
    entry = InventoryEntry(object_id, 0, Path(f"obj{object_id}__0.png"), split)
    return TrainingReference(entry, extract_all(image, segment_object(image).mask))


def synthetic_inventory(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for object_id in range(1, 21):
            for view in range(72):
                writer.writerow({
                    "object_id": object_id, "angle_degrees": view * 5,
                    "image_path": f"obj{object_id}__{view * 5}.png",
                    "split": "train" if view in TRAIN_VIEW_INDICES else "test",
                })


class PreprocessingTests(unittest.TestCase):
    def test_read_preserves_dimensions_bgr_and_unicode_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "imagen_á.png"
            image = np.full((12, 17, 3), (7, 20, 200), np.uint8)
            cv2.imencode(".png", image)[1].tofile(path)
            actual = load_bgr(path)
            self.assertEqual(actual.shape, (12, 17, 3))
            np.testing.assert_array_equal(actual, image)

    def test_invalid_and_empty_files_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.png"
            for contents in (b"not an image", b""):
                path.write_bytes(contents)
                with self.assertRaises(ValueError):
                    load_bgr(path)

    def test_dark_object_is_not_removed_and_dark_interior_is_filled(self) -> None:
        dark = np.zeros((12, 12, 3), np.uint8)
        dark[3:9, 3:9] = 6
        segmentation = segment_object(dark)
        self.assertTrue(segmentation.used_fallback)
        self.assertEqual(np.count_nonzero(segmentation.mask), 36)
        dark[3:9, 3:9] = 60
        dark[4:8, 4:8] = 2
        segmentation = segment_object(dark)
        self.assertFalse(segmentation.used_fallback)
        self.assertEqual(np.count_nonzero(segmentation.mask), 36)

    def test_black_image_is_rejected_instead_of_creating_empty_histogram(self) -> None:
        with self.assertRaises(ValueError):
            segment_object(np.zeros((8, 8, 3), np.uint8))

    def test_largest_component_discards_isolated_noise(self) -> None:
        image = np.zeros((12, 12, 3), np.uint8)
        image[3:9, 3:9] = 80
        image[0, 0] = 255
        self.assertEqual(np.count_nonzero(segment_object(image).mask), 36)


class FeatureTests(unittest.TestCase):
    def test_known_color_conversions(self) -> None:
        red = color_image((0, 0, 255))
        np.testing.assert_array_equal(convert_color(red, "RGB")[0, 0], [255, 0, 0])
        np.testing.assert_array_equal(convert_color(red, "HSV")[0, 0], [0, 255, 255])
        np.testing.assert_allclose(convert_color(red, "CIELAB")[0, 0], [136, 208, 195], atol=1)
        np.testing.assert_array_equal(convert_color(color_image((255, 0, 0)), "HSV")[0, 0], [120, 255, 255])

    def test_histogram_dimensions_normalization_and_hue_range(self) -> None:
        blue = color_image((255, 0, 0))
        mask = np.full((8, 8), 255, np.uint8)
        for space, histogram in extract_all(blue, mask).items():
            self.assertEqual(histogram.shape, (512,))
            self.assertAlmostEqual(float(histogram.sum()), 1.0)
            self.assertTrue(np.isfinite(histogram).all())
            self.assertTrue((histogram >= 0).all())
        # H=120 cae en bin floor(120*8/180)=5; S=V=255 caen en bin 7.
        histogram = extract_histogram(blue, mask, "HSV").reshape(8, 8, 8)
        self.assertEqual(histogram[5, 7, 7], 1.0)
        rgb = extract_histogram(blue, mask, "RGB").reshape(8, 8, 8)
        self.assertEqual(rgb[0, 0, 7], 1.0)

    def test_all_extractors_receive_same_mask_without_modification(self) -> None:
        image = color_image((20, 150, 200))
        mask = segment_object(image).mask
        before = mask.copy()
        with patch("src.coil100.features.extract_histogram", wraps=extract_histogram) as mocked:
            extract_all(image, mask)
        self.assertEqual(mocked.call_count, 3)
        self.assertTrue(all(call.args[1] is mask for call in mocked.call_args_list))
        np.testing.assert_array_equal(mask, before)

    def test_mask_excludes_background_and_counts_mixture(self) -> None:
        image = np.zeros((8, 8, 3), np.uint8)
        image[:2] = (0, 0, 255)
        image[2:4] = (255, 0, 0)
        mask = np.zeros((8, 8), np.uint8)
        mask[:4] = 255
        histogram = extract_histogram(image, mask, "RGB").reshape(8, 8, 8)
        self.assertEqual(histogram[7, 0, 0], 0.5)
        self.assertEqual(histogram[0, 0, 7], 0.5)
        self.assertEqual(histogram[0, 0, 0], 0.0)

    def test_empty_or_wrong_size_mask_rejected(self) -> None:
        for mask in (np.zeros((8, 8), np.uint8), np.ones((4, 4), np.uint8)):
            with self.assertRaises(ValueError):
                extract_histogram(color_image((0, 0, 255)), mask, "RGB")


class ClassifierTests(unittest.TestCase):
    def test_controlled_examples_reproducibility_and_one_segmentation(self) -> None:
        model = Histogram1NN([reference(1, (0, 0, 255)), reference(2, (255, 0, 0))])
        query = color_image((255, 0, 0))
        with patch("src.coil100.classifier.segment_object", wraps=segment_object) as mocked:
            predictions = model.recognize(query)
        self.assertEqual(mocked.call_count, 1)
        self.assertEqual(predictions, model.recognize(query))
        self.assertEqual([item.color_space for item in predictions], list(COLOR_SPACES))
        for item in predictions:
            self.assertEqual(item.object_id, 2)
            self.assertEqual(item.distance, 0)
            self.assertEqual(item.nearest_image, "obj2__0.png")
            self.assertEqual(model.recognize(query, item.color_space), [item])

    def test_bhattacharyya_has_known_distance_for_partial_overlap(self) -> None:
        model = Histogram1NN([reference(1, (0, 0, 255))])
        query = np.zeros(512, np.float32)
        query[7 * 64] = 0.5
        query[7] = 0.5
        self.assertAlmostEqual(model.predict(query, "RGB").distance, np.sqrt(1 - np.sqrt(0.5)))

    def test_ties_resolve_by_class_and_angle_independent_of_input_order(self) -> None:
        a, b = reference(1, (0, 0, 255)), reference(2, (0, 0, 255))
        query = color_image((0, 0, 255))
        self.assertEqual(Histogram1NN([b, a]).recognize(query), Histogram1NN([a, b]).recognize(query))
        self.assertEqual(Histogram1NN([b, a]).recognize(query)[0].object_id, 1)

    def test_rejects_test_references_and_invalid_histograms(self) -> None:
        with self.assertRaises(ValueError):
            Histogram1NN([reference(1, (0, 0, 255), "test")])
        model = Histogram1NN([reference(1, (0, 0, 255))])
        for histogram in (np.zeros(512), np.full(512, np.nan), np.ones(8)):
            with self.assertRaises(ValueError):
                model.predict(histogram, "RGB")

    def test_from_inventory_reads_exactly_320_train_images_and_never_test(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = root / "inventory.csv"
            synthetic_inventory(inventory)
            before = inventory.read_bytes()
            entries = read_inventory(inventory, root)
            allowed = {entry.image_path for entry in entries if entry.split == "train"}

            def training_only_loader(path: Path) -> np.ndarray:
                self.assertIn(path, allowed)
                return color_image((0, 0, 255))

            with patch("src.coil100.classifier.load_bgr", side_effect=training_only_loader) as loader:
                model = Histogram1NN.from_inventory(inventory, root)
            self.assertEqual(loader.call_count, 320)
            self.assertEqual(len(model.references), 320)
            self.assertTrue(all(entry.split == "train" for entry in model.references))
            self.assertEqual({entry.image_path for entry in model.references}, allowed)
            self.assertEqual(inventory.read_bytes(), before)

    def test_invalid_inventory_duplicates_and_path_escape_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inventory = root / "inventory.csv"
            synthetic_inventory(inventory)
            contents = inventory.read_text(encoding="utf-8")
            for invalid in (
                contents + contents.splitlines()[1] + "\n",
                contents.replace("obj1__0.png", "../outside.png"),
                contents.replace("1,0,obj1__0.png,train", "1,0,obj1__0.png,test"),
            ):
                inventory.write_text(invalid, encoding="utf-8")
                with self.assertRaises(ValueError):
                    read_inventory(inventory, root)


if __name__ == "__main__":
    unittest.main()
