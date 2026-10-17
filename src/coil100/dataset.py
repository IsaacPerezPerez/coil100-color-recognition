"""Validación e inventario reproducible de COIL-100."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

from PIL import Image

OBJECT_IDS = tuple(range(1, 21))
VIEWS_PER_OBJECT = 72
ANGLE_STEP_DEGREES = 5
# 16 posiciones repartidas sobre 72 vistas; los intervalos alternan 4 y 5 vistas.
TRAIN_VIEW_INDICES = tuple(i * VIEWS_PER_OBJECT // 16 for i in range(16))
CSV_FIELDS = ("object_id", "angle_degrees", "image_path", "split")
FILENAME_RE = re.compile(r"^obj(?P<object>\d+)__(?P<angle>\d+)\.png$", re.IGNORECASE)


class DatasetValidationError(ValueError):
    """El directorio no cumple el formato esperado del COIL-100 original."""


def _parse_filename(path: Path) -> tuple[int, int] | None:
    match = FILENAME_RE.fullmatch(path.name)
    if match is None:
        return None
    return int(match.group("object")), int(match.group("angle"))


def _is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, SyntaxError, ValueError):
        return False


def validate_dataset(dataset_root: Path | str) -> dict[int, list[tuple[int, Path]]]:
    """Valida 72 PNG legibles por objeto y devuelve vistas ordenadas por ángulo."""
    root = Path(dataset_root)
    if not root.is_dir():
        raise DatasetValidationError(f"No existe el directorio del dataset: {root}")

    found: dict[int, dict[int, Path]] = {obj: {} for obj in OBJECT_IDS}
    problems: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        parsed = _parse_filename(path)
        if parsed is None:
            continue
        object_id, angle = parsed
        if object_id not in found:
            continue
        if angle < 0 or angle >= 360 or angle % ANGLE_STEP_DEGREES != 0:
            problems.append(f"Ángulo inesperado en {path.relative_to(root)}: {angle}")
            continue
        if angle in found[object_id]:
            problems.append(f"Ángulo duplicado para obj{object_id}: {angle}")
            continue
        if not _is_valid_image(path):
            problems.append(f"Imagen PNG inválida o ilegible: {path.relative_to(root)}")
            continue
        found[object_id][angle] = path

    expected_angles = set(range(0, 360, ANGLE_STEP_DEGREES))
    for object_id in OBJECT_IDS:
        actual = set(found[object_id])
        missing = sorted(expected_angles - actual)
        extra_count = len(actual - expected_angles)
        if missing:
            problems.append(
                f"obj{object_id}: {len(actual)}/72 imágenes válidas; faltan ángulos {missing}"
            )
        if extra_count:
            problems.append(f"obj{object_id}: {extra_count} ángulos inesperados")

    if problems:
        raise DatasetValidationError("Validación COIL-100 fallida:\n- " + "\n- ".join(problems))

    return {
        object_id: [(angle, found[object_id][angle]) for angle in sorted(found[object_id])]
        for object_id in OBJECT_IDS
    }


def build_inventory(
    views_by_object: dict[int, Iterable[tuple[int, Path]]],
    dataset_root: Path | str,
) -> list[dict[str, str | int]]:
    """Crea filas CSV con la división fija de 16 entrenamiento y 56 prueba."""
    root = Path(dataset_root).resolve()
    rows: list[dict[str, str | int]] = []
    for object_id in OBJECT_IDS:
        views = sorted(views_by_object[object_id], key=lambda item: item[0])
        if len(views) != VIEWS_PER_OBJECT:
            raise DatasetValidationError(
                f"obj{object_id} requiere 72 vistas para crear el inventario; recibió {len(views)}"
            )
        angles = [angle for angle, _ in views]
        if angles != list(range(0, 360, ANGLE_STEP_DEGREES)):
            raise DatasetValidationError(f"obj{object_id} no contiene los 72 ángulos esperados")
        for view_index, (angle, path) in enumerate(views):
            rows.append(
                {
                    "object_id": object_id,
                    "angle_degrees": angle,
                    "image_path": path.resolve().relative_to(root).as_posix(),
                    "split": "train" if view_index in TRAIN_VIEW_INDICES else "test",
                }
            )
    return rows


def write_inventory_csv(rows: Iterable[dict[str, str | int]], output_path: Path | str) -> None:
    """Escribe el inventario CSV en UTF-8 con orden de columnas estable."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
