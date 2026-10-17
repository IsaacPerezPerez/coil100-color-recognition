"""Evidencia visual y comprobación básica usando exclusivamente imágenes train."""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .classifier import Histogram1NN, TrainingReference
from .features import extract_all
from .inventory import read_inventory
from .preprocessing import SegmentationConfig, load_bgr, segment_object


def save_png(path: Path, image: np.ndarray) -> None:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError(f"No se pudo generar {path}")
    encoded.tofile(path)


def inspect_training(inventory: Path, dataset_root: Path, output_dir: Path, threshold: int) -> dict:
    inventory_hash = hashlib.sha256(inventory.read_bytes()).hexdigest()
    entries = read_inventory(inventory, dataset_root)
    config = SegmentationConfig(threshold)
    references, checks = [], []
    panels: dict[int, list[np.ndarray]] = {0: [], 180: []}
    first_image = None
    for entry in entries:
        if entry.split != "train":
            continue
        image = load_bgr(entry.image_path)
        segmentation = segment_object(image, config)
        histograms = extract_all(image, segmentation.mask)
        if any(hist.shape != (512,) or not np.isclose(hist.sum(), 1) for hist in histograms.values()):
            raise ValueError(f"Histograma incorrecto: {entry.image_path}")
        references.append(TrainingReference(entry, histograms))
        if first_image is None:
            first_image = image
        pixels = int(np.count_nonzero(segmentation.mask))
        checks.append({
            "object_id": entry.object_id,
            "angle_degrees": entry.angle_degrees,
            "image_path": entry.image_path.relative_to(dataset_root.resolve()).as_posix(),
            "height": image.shape[0], "width": image.shape[1],
            "mask_pixels": pixels,
            "mask_fraction": pixels / segmentation.mask.size,
            "threshold_used": segmentation.threshold_used,
            "used_fallback": segmentation.used_fallback,
        })
        if entry.angle_degrees in panels:
            masked = cv2.bitwise_and(image, image, mask=segmentation.mask)
            views = (image, cv2.cvtColor(segmentation.mask, cv2.COLOR_GRAY2BGR), masked)
            strip = np.hstack([cv2.resize(view, (128, 128), interpolation=cv2.INTER_NEAREST) for view in views])
            panel = np.full((152, 384, 3), 240, np.uint8)
            panel[24:] = strip
            cv2.putText(panel, f"obj{entry.object_id} {entry.angle_degrees}deg: original / mask / object", (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
            panels[entry.angle_degrees].append(panel)

    model = Histogram1NN(references, config)
    predictions = model.recognize(first_image)
    if predictions != model.recognize(first_image):
        raise ValueError("Las predicciones repetidas difieren.")
    if hashlib.sha256(inventory.read_bytes()).hexdigest() != inventory_hash:
        raise ValueError("El inventario cambió durante la comprobación.")
    summary = {
        "scope": "training_only_implementation_check_not_final_evaluation",
        "inventory_sha256": inventory_hash,
        "training_references": len(model.references),
        "config": asdict(config),
        "image_dimensions": sorted({(item["height"], item["width"]) for item in checks}),
        "empty_masks": sum(item["mask_pixels"] == 0 for item in checks),
        "fallback_count": sum(item["used_fallback"] for item in checks),
        "min_mask_fraction": min(item["mask_fraction"] for item in checks),
        "max_mask_fraction": max(item["mask_fraction"] for item in checks),
        "single_training_query": str(model.references[0].image_path),
        "predictions_reproducible": True,
        "predictions": [asdict(prediction) for prediction in predictions],
        "versions": {"opencv": cv2.__version__, "numpy": np.__version__},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "segmentation_train.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(checks[0]))
        writer.writeheader()
        writer.writerows(checks)
    for angle, items in panels.items():
        if items:
            blank = np.full_like(items[0], 240)
            items += [blank] * ((-len(items)) % 4)
            sheet = np.vstack([np.hstack(items[start:start + 4]) for start in range(0, len(items), 4)])
            save_png(output_dir / f"segmentation_{angle:03d}.png", sheet)
    (output_dir / "training_check.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/coil-100"))
    parser.add_argument("--inventory", type=Path, default=Path("data/processed/inventory.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase2"))
    parser.add_argument("--background-threshold", type=int, default=40)
    args = parser.parse_args()
    try:
        summary = inspect_training(args.inventory, args.dataset_root, args.output_dir, args.background_threshold)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
