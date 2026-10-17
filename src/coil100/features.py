"""Histogramas conjuntos de 8 x 8 x 8 bins con normalización L1."""

import cv2
import numpy as np

from .preprocessing import validate_bgr

COLOR_SPACES = ("RGB", "HSV", "CIELAB")
CONVERSIONS = {"RGB": cv2.COLOR_BGR2RGB, "HSV": cv2.COLOR_BGR2HSV, "CIELAB": cv2.COLOR_BGR2LAB}
RANGES = {
    "RGB": [0, 256, 0, 256, 0, 256],
    "HSV": [0, 180, 0, 256, 0, 256],
    "CIELAB": [0, 256, 0, 256, 0, 256],
}


def convert_color(image: np.ndarray, color_space: str) -> np.ndarray:
    validate_bgr(image)
    if color_space not in COLOR_SPACES:
        raise ValueError(f"Espacio no válido: {color_space}. Use {COLOR_SPACES}.")
    return cv2.cvtColor(image, CONVERSIONS[color_space])


def extract_histogram(image: np.ndarray, mask: np.ndarray, color_space: str) -> np.ndarray:
    """Usa la máscara recibida sin modificarla ni volver a segmentar."""
    converted = convert_color(image, color_space)
    if mask.dtype != np.uint8 or mask.shape != image.shape[:2] or not np.any(mask):
        raise ValueError("La máscara debe ser uint8, no vacía y del tamaño H x W de la imagen.")
    histogram = cv2.calcHist([converted], [0, 1, 2], mask, [8, 8, 8], RANGES[color_space])
    histogram = histogram.reshape(512)
    histogram /= histogram.sum()
    return histogram


def extract_all(image: np.ndarray, mask: np.ndarray) -> dict[str, np.ndarray]:
    return {space: extract_histogram(image, mask, space) for space in COLOR_SPACES}
