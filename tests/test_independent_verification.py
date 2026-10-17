import shutil
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.coil100.classifier import Histogram1NN, TrainingReference
from src.coil100.features import COLOR_SPACES
from src.coil100.inventory import InventoryEntry
from src.coil100.preprocessing import load_bgr
from src.coil100.verify_independent import SEED, permute_training_labels


def controlled_references() -> list[TrainingReference]:
    references = []
    for obj in range(1, 21):
        histogram = np.zeros(512, np.float32)
        histogram[obj - 1] = 1
        entry = InventoryEntry(obj, 0, Path(f"obj{obj}__0.png"), "train")
        references.append(TrainingReference(entry, {space: histogram for space in COLOR_SPACES}))
    return references


class IndependentControlTests(unittest.TestCase):
    def test_permutation_is_reproducible_balanced_and_does_not_mutate_references(self) -> None:
        original = controlled_references()
        first = permute_training_labels(original, SEED)
        second = permute_training_labels(original, SEED)
        self.assertEqual([ref.entry.object_id for ref in first], [ref.entry.object_id for ref in second])
        self.assertEqual(Counter(ref.entry.object_id for ref in original), Counter(ref.entry.object_id for ref in first))
        self.assertEqual([ref.entry.object_id for ref in original], list(range(1, 21)))
        for source, changed in zip(original, first):
            self.assertIs(source.histograms, changed.histograms)
            self.assertEqual(source.entry.image_path, changed.entry.image_path)

    def test_negative_control_loses_perfect_accuracy_with_controlled_histograms(self) -> None:
        references = controlled_references()
        model = Histogram1NN(permute_training_labels(references, SEED))
        correct = sum(model.predict(ref.histograms["RGB"], "RGB").object_id == ref.entry.object_id for ref in references)
        self.assertLess(correct, 20)

    def test_query_filename_does_not_change_pixel_based_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory) / "obj1__0.png"
            renamed = Path(directory) / "obj20__355.png"
            image = np.full((8, 8, 3), (0, 0, 255), dtype=np.uint8)
            cv2.imencode(".png", image)[1].tofile(original)
            shutil.copyfile(original, renamed)
            from src.coil100.features import extract_all
            from src.coil100.preprocessing import segment_object

            entry = InventoryEntry(1, 0, Path("reference.png"), "train")
            model = Histogram1NN([TrainingReference(entry, extract_all(image, segment_object(image).mask))])
            self.assertEqual(model.recognize(load_bgr(original)), model.recognize(load_bgr(renamed)))
