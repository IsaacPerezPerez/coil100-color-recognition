"""Lectura BGR y una sola máscara independiente de la clase y del espacio de color."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class SegmentationConfig:
    background_threshold: int = 40

    def __post_init__(self) -> None:
        if not 0 <= self.background_threshold <= 254:
            raise ValueError("El umbral debe estar entre 0 y 254.")


@dataclass(frozen=True)
class Segmentation:
    mask: np.ndarray
    threshold_used: float
    used_fallback: bool


def validate_bgr(image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("Se requiere una imagen uint8 BGR no vacía de dimensiones H x W x 3.")


def load_bgr(path: Path | str) -> np.ndarray:
    """Decodifica con OpenCV; fromfile admite rutas Unicode de Windows."""
    encoded = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {path}")
    validate_bgr(image)
    return image


def segment_object(
    image: np.ndarray, config: SegmentationConfig = SegmentationConfig()
) -> Segmentation:
    """Umbral sobre max(B,G,R), componente mayor y relleno de sus huecos internos.

    Si el umbral elimina todos los píxeles, se reduce a la mitad del máximo
    observado. Una imagen completamente negra produce un error explícito.
    """
    validate_bgr(image)
    brightness = image.max(axis=2)
    threshold = float(config.background_threshold)
    foreground = brightness > threshold
    fallback = not bool(foreground.any())
    if fallback:
        peak = int(brightness.max())
        if peak == 0:
            raise ValueError("Imagen completamente negra: no hay objeto segmentable.")
        threshold = peak / 2
        foreground = brightness > threshold
    binary = foreground.astype(np.uint8) * 255
    # El conteo de píxeles funciona también para componentes de un solo píxel.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    largest = 1 + int(np.argmax(stats[1:count, cv2.CC_STAT_AREA]))
    component = (labels == largest).astype(np.uint8) * 255
    contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(component)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    mask.setflags(write=False)
    return Segmentation(mask, threshold, fallback)
