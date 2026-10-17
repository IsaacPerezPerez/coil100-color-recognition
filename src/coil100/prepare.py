"""CLI para validar COIL-100 y generar el inventario con su división."""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import DatasetValidationError, build_inventory, validate_dataset, write_inventory_csv


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/raw/coil-100"),
        help="Directorio con las imágenes originales (por defecto: data/raw/coil-100)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/inventory.csv"),
        help="Ruta del CSV de salida (por defecto: data/processed/inventory.csv)",
    )
    args = parser.parse_args()
    try:
        views = validate_dataset(args.dataset_root)
        rows = build_inventory(views, args.dataset_root)
        write_inventory_csv(rows, args.output)
    except (DatasetValidationError, OSError, ValueError) as error:
        parser.error(str(error))

    print(f"Validación correcta: 20 objetos, 1440 imágenes PNG válidas.")
    print(f"División: 320 train y 1120 test. Inventario: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
