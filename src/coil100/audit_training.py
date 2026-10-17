"""Auditoría de máscaras y Leave-One-Out exclusivamente sobre entrenamiento."""

import argparse
import csv
import hashlib
import json
import platform
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from .classifier import Histogram1NN, TrainingReference, validate_histogram
from .dataset import OBJECT_IDS, TRAIN_VIEW_INDICES
from .features import COLOR_SPACES, extract_all
from .inspect_training import save_png
from .inventory import read_inventory
from .preprocessing import SegmentationConfig, load_bgr, segment_object


def leave_one_out(references: list[TrainingReference]) -> list[dict]:
    if len(references) < 2:
        raise ValueError("Leave-One-Out requiere al menos dos referencias train.")
    paths = [ref.entry.image_path.resolve() for ref in references]
    if len(set(paths)) != len(paths):
        raise ValueError("Leave-One-Out no admite rutas duplicadas.")
    results = []
    for index, query in enumerate(references):
        model = Histogram1NN(references[:index] + references[index + 1:])
        if query.entry.image_path.resolve() in {entry.image_path.resolve() for entry in model.references}:
            raise ValueError("La consulta sigue presente entre las referencias.")
        for space in COLOR_SPACES:
            prediction = model.predict(query.histograms[space], space)
            results.append({
                "query_image": query.entry.image_path.name,
                "true_object_id": query.entry.object_id,
                "predicted_object_id": prediction.object_id,
                "color_space": space,
                "distance": prediction.distance,
                "nearest_image": Path(prediction.nearest_image).name,
                "reference_count": len(model.references),
                "correct": prediction.object_id == query.entry.object_id,
            })
    return results


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def mask_panel(image: np.ndarray, mask: np.ndarray, caption: str) -> np.ndarray:
    masked = cv2.bitwise_and(image, image, mask=mask)
    views = (image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), masked)
    panel = np.full((152, 384, 3), 240, np.uint8)
    panel[24:] = np.hstack([cv2.resize(view, (128, 128), interpolation=cv2.INTER_NEAREST) for view in views])
    cv2.putText(panel, caption, (3, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 0, 0), 1)
    return panel


def audit_training(inventory: Path, dataset_root: Path, output_dir: Path) -> dict:
    before = hashlib.sha256(inventory.read_bytes()).hexdigest()
    entries = read_inventory(inventory, dataset_root)
    expected_train = {index * 5 for index in TRAIN_VIEW_INDICES}
    if any(entry.split != ("train" if entry.angle_degrees in expected_train else "test") for entry in entries):
        raise ValueError("La división difiere de la regla original de Fase 1.")
    train = sorted((entry for entry in entries if entry.split == "train"), key=lambda entry: (entry.object_id, entry.angle_degrees))
    selected = {}
    for obj in OBJECT_IDS:
        views = [entry for entry in train if entry.object_id == obj]
        selected[obj] = (views[0].angle_degrees, views[len(views) // 2].angle_degrees)

    config = SegmentationConfig()
    references, mask_rows, tiles = [], [], {obj: [] for obj in OBJECT_IDS}
    maximum_sum_error = 0.0
    for entry in train:
        image = load_bgr(entry.image_path)
        segmentation = segment_object(image, config)
        histograms = extract_all(image, segmentation.mask)
        for histogram in histograms.values():
            validate_histogram(histogram)
            maximum_sum_error = max(maximum_sum_error, abs(float(histogram.sum()) - 1))
        references.append(TrainingReference(entry, histograms))
        percent = 100 * int(np.count_nonzero(segmentation.mask)) / segmentation.mask.size
        mask_rows.append({
            "object_id": entry.object_id, "angle_degrees": entry.angle_degrees,
            "image_path": entry.image_path.relative_to(dataset_root.resolve()).as_posix(),
            "retained_percent": percent, "used_fallback": segmentation.used_fallback,
            "in_contact_sheet": entry.angle_degrees in selected[entry.object_id],
        })
        if entry.angle_degrees in selected[entry.object_id]:
            caption = f"obj{entry.object_id} {entry.angle_degrees}deg {percent:.1f}% | original / mask / object"
            tiles[entry.object_id].append(mask_panel(image, segmentation.mask, caption))

    object_summary = []
    for obj in OBJECT_IDS:
        rows = [row for row in mask_rows if row["object_id"] == obj]
        percentages = [row["retained_percent"] for row in rows]
        chosen = [row for row in rows if row["in_contact_sheet"]]
        object_summary.append({
            "object_id": obj, "training_views": len(rows),
            "min_percent": min(percentages), "mean_percent": float(np.mean(percentages)),
            "max_percent": max(percentages),
            "view_1_angle": chosen[0]["angle_degrees"], "view_1_percent": chosen[0]["retained_percent"],
            "view_2_angle": chosen[1]["angle_degrees"], "view_2_percent": chosen[1]["retained_percent"],
        })
    extremes = []
    varied_objects = sorted(object_summary, key=lambda row: row["max_percent"] - row["min_percent"], reverse=True)[:6]
    for obj in varied_objects:
        row = max((item for item in mask_rows if item["object_id"] == obj["object_id"]), key=lambda item: item["retained_percent"])
        image = load_bgr(dataset_root / row["image_path"])
        mask = segment_object(image, config).mask
        caption = f"{row['image_path']} {row['retained_percent']:.1f}% | original / mask / object"
        extremes.append(mask_panel(image, mask, caption))
    predictions = leave_one_out(references)
    loo = {}
    for space in COLOR_SPACES:
        rows = [row for row in predictions if row["color_space"] == space]
        correct = sum(row["correct"] for row in rows)
        loo[space] = {"correct": correct, "total": len(rows), "accuracy": correct / len(rows)}
    after = hashlib.sha256(inventory.read_bytes()).hexdigest()
    if before != after:
        raise ValueError("El inventario cambió durante la auditoría.")
    summary = {
        "scope": "training_only_leave_one_out_diagnostic_not_final_generalization",
        "inventory_sha256_before": before, "inventory_sha256_after": after,
        "inventory_unchanged": True, "original_split_rule_verified": True,
        "train_count": len(train), "test_count_metadata_only": len(entries) - len(train),
        "test_images_loaded": 0, "config": asdict(config),
        "selection_rule": "sort train by angle; select positions 0 and n//2 per object",
        "empty_masks": sum(row["retained_percent"] == 0 for row in mask_rows),
        "fallback_count": sum(row["used_fallback"] for row in mask_rows),
        "maximum_histogram_sum_error": maximum_sum_error,
        "references_per_loo_query": len(train) - 1, "loo": loo,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv2.__version__},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    object_tiles = [np.vstack(tiles[obj]) for obj in OBJECT_IDS]
    sheet = np.vstack([np.hstack(object_tiles[start:start + 4]) for start in range(0, 20, 4)])
    save_png(output_dir / "train_mask_contact_sheet.png", sheet)
    save_png(output_dir / "train_mask_extremes.png", np.vstack([np.hstack(extremes[start:start + 2]) for start in range(0, 6, 2)]))
    write_csv(output_dir / "train_mask_details.csv", mask_rows)
    write_csv(output_dir / "train_mask_summary.csv", object_summary)
    write_csv(output_dir / "loo_predictions.csv", predictions)
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/coil-100"))
    parser.add_argument("--inventory", type=Path, default=Path("data/processed/inventory.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase3a"))
    args = parser.parse_args()
    try:
        result = audit_training(args.inventory, args.dataset_root, args.output_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
