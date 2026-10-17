"""Un solo clasificador 1-NN de histogramas con distancia de Bhattacharyya."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .features import COLOR_SPACES, extract_all
from .inventory import InventoryEntry, read_inventory
from .preprocessing import SegmentationConfig, load_bgr, segment_object


def validate_histogram(histogram: np.ndarray) -> np.ndarray:
    histogram = np.asarray(histogram, dtype=np.float32)
    if (
        histogram.shape != (512,)
        or not np.isfinite(histogram).all()
        or (histogram < 0).any()
        or not np.isclose(histogram.sum(), 1, atol=1e-5)
    ):
        raise ValueError("El histograma debe tener 512 valores finitos no negativos y sumar 1.")
    return histogram


@dataclass(frozen=True)
class TrainingReference:
    entry: InventoryEntry
    histograms: dict[str, np.ndarray]


@dataclass(frozen=True)
class Prediction:
    object_id: int
    distance: float
    nearest_image: str
    color_space: str


class Histogram1NN:
    def __init__(
        self,
        references: list[TrainingReference],
        config: SegmentationConfig = SegmentationConfig(),
    ) -> None:
        if not references or any(ref.entry.split != "train" for ref in references):
            raise ValueError("Las referencias deben ser no vacías y proceder exclusivamente de train.")
        # El orden de desempate es explícito, independiente del orden del CSV.
        ordered = sorted(references, key=lambda ref: (ref.entry.object_id, ref.entry.angle_degrees, str(ref.entry.image_path)))
        if len({ref.entry.image_path.resolve() for ref in ordered}) != len(ordered):
            raise ValueError("Hay referencias duplicadas.")
        self.config = config
        self.references = tuple(ref.entry for ref in ordered)
        self._histograms = {}
        for space in COLOR_SPACES:
            histograms = np.stack([validate_histogram(ref.histograms[space]) for ref in ordered])
            histograms.setflags(write=False)
            self._histograms[space] = histograms

    @classmethod
    def from_inventory(
        cls,
        inventory_path: Path | str,
        dataset_root: Path | str,
        config: SegmentationConfig = SegmentationConfig(),
    ) -> "Histogram1NN":
        entries = read_inventory(inventory_path, dataset_root)
        references = []
        for entry in entries:
            if entry.split != "train":
                continue
            image = load_bgr(entry.image_path)
            segmentation = segment_object(image, config)
            references.append(TrainingReference(entry, extract_all(image, segmentation.mask)))
        return cls(references, config)

    def predict(self, histogram: np.ndarray, color_space: str) -> Prediction:
        if color_space not in COLOR_SPACES:
            raise ValueError(f"Espacio no válido: {color_space}.")
        query = validate_histogram(histogram)
        distances = [
            cv2.compareHist(query, reference, cv2.HISTCMP_BHATTACHARYYA)
            for reference in self._histograms[color_space]
        ]
        nearest = int(np.argmin(distances))
        entry = self.references[nearest]
        return Prediction(entry.object_id, float(distances[nearest]), str(entry.image_path), color_space)

    def recognize(self, image: np.ndarray, color_space: str = "all") -> list[Prediction]:
        if color_space != "all" and color_space not in COLOR_SPACES:
            raise ValueError(f"Espacio no válido: {color_space}.")
        segmentation = segment_object(image, self.config)
        # Una sola máscara original para los tres histogramas de la consulta.
        histograms = extract_all(image, segmentation.mask)
        spaces = COLOR_SPACES if color_space == "all" else (color_space,)
        return [self.predict(histograms[space], space) for space in spaces]
