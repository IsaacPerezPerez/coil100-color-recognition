"""Lectura del inventario existente sin volver a escribirlo ni cambiar su división."""

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .dataset import CSV_FIELDS, OBJECT_IDS


@dataclass(frozen=True)
class InventoryEntry:
    object_id: int
    angle_degrees: int
    image_path: Path
    split: str


def read_inventory(csv_path: Path | str, dataset_root: Path | str) -> tuple[InventoryEntry, ...]:
    root = Path(dataset_root).resolve()
    entries = []
    paths: set[Path] = set()
    object_angles: set[tuple[int, int]] = set()
    with Path(csv_path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(CSV_FIELDS):
            raise ValueError(f"Columnas de inventario incorrectas: se requiere {CSV_FIELDS}.")
        for row in reader:
            object_id, angle = int(row["object_id"]), int(row["angle_degrees"])
            relative_path = Path(row["image_path"])
            path = (root / relative_path).resolve()
            if relative_path.is_absolute() or not path.is_relative_to(root):
                raise ValueError("La ruta de imagen debe ser relativa y permanecer dentro del dataset.")
            if object_id not in OBJECT_IDS or angle not in range(0, 360, 5):
                raise ValueError("Clase o ángulo fuera de COIL-100 obj1 a obj20.")
            if row["split"] not in ("train", "test"):
                raise ValueError("El conjunto debe ser train o test.")
            if path in paths or (object_id, angle) in object_angles:
                raise ValueError("Imagen o combinación de objeto/ángulo duplicada en el inventario.")
            paths.add(path)
            object_angles.add((object_id, angle))
            entries.append(InventoryEntry(object_id, angle, path, row["split"]))
    counts = Counter((entry.object_id, entry.split) for entry in entries)
    if any(counts[obj, "train"] != 16 or counts[obj, "test"] != 56 for obj in OBJECT_IDS):
        raise ValueError("Se requieren exactamente 16 train y 56 test para cada uno de los 20 objetos.")
    return tuple(entries)
