"""Predicción individual desde consola; reutiliza el CSV de la Fase 1."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .classifier import Histogram1NN
from .features import COLOR_SPACES
from .preprocessing import SegmentationConfig, load_bgr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/coil-100"))
    parser.add_argument("--inventory", type=Path, default=Path("data/processed/inventory.csv"))
    parser.add_argument("--color-space", choices=(*COLOR_SPACES, "all"), default="all")
    parser.add_argument("--background-threshold", type=int, default=40)
    args = parser.parse_args()
    try:
        config = SegmentationConfig(args.background_threshold)
        model = Histogram1NN.from_inventory(args.inventory, args.dataset_root, config)
        predictions = model.recognize(load_bgr(args.image), args.color_space)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps([asdict(prediction) for prediction in predictions], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
