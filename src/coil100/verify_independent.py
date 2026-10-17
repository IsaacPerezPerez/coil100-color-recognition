"""Controles diagnósticos separados de los resultados oficiales de Fase 3B."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .audit_training import write_csv
from .classifier import Histogram1NN, TrainingReference
from .evaluation_plots import plot_comparison, plot_confusion
from .features import COLOR_SPACES, extract_all
from .inventory import read_inventory
from .preprocessing import load_bgr, segment_object

SEED = 20261017


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def permute_training_labels(references: list[TrainingReference], seed: int) -> list[TrainingReference]:
    labels = np.random.default_rng(seed).permutation([ref.entry.object_id for ref in references])
    return [TrainingReference(replace(ref.entry, object_id=int(label)), ref.histograms) for ref, label in zip(references, labels)]


def verify_saved_results(root: Path) -> dict:
    official = root / "artifacts/phase3b"
    metadata = json.loads((official / "evaluation_metadata.json").read_text())
    if sha256(root / "data/processed/inventory.csv") != metadata["inventory_sha256_before"]:
        raise ValueError("Cambió el inventario oficial.")
    for name, expected in {**metadata["method_sha256_before"], **metadata["evaluation_source_sha256"]}.items():
        if sha256(root / "src/coil100" / name) != expected:
            raise ValueError(f"Cambió el módulo original: {name}")
    entries = csv_rows(root / "data/processed/inventory.csv")
    train = {row["image_path"] for row in entries if row["split"] == "train"}
    test = {row["image_path"]: int(row["object_id"]) for row in entries if row["split"] == "test"}
    if len(train) != 320 or len(test) != 1120 or train.intersection(test):
        raise ValueError("La separación de conjuntos no es válida.")
    summaries = csv_rows(official / "summary.csv")
    checks = {}
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        for space in COLOR_SPACES:
            suffix = space.lower()
            rows = csv_rows(official / f"predictions_{suffix}.csv")
            if len(rows) != 1120 or {row["image_path"] for row in rows} != set(test):
                raise ValueError(f"Las predicciones no cubren test exactamente en {space}.")
            matrix = np.zeros((20, 20), dtype=np.int64)
            for row in rows:
                if int(row["true_object_id"]) != test[row["image_path"]] or row["nearest_image"] not in train or row["color_space"] != space:
                    raise ValueError("Etiqueta real, vecino o espacio inconsistente.")
                matrix[int(row["true_object_id"]) - 1, int(row["predicted_object_id"]) - 1] += 1
            with (official / f"confusion_{suffix}.csv").open(newline="", encoding="utf-8") as stream:
                saved = list(csv.reader(stream))
            if saved[0][1:] != [f"obj{i}" for i in range(1, 21)] or [row[0] for row in saved[1:]] != [f"obj{i}" for i in range(1, 21)]:
                raise ValueError("Orden incorrecto en la matriz de confusión.")
            if not np.array_equal(matrix, np.array([[int(value) for value in row[1:]] for row in saved[1:]])):
                raise ValueError("La matriz no corresponde a las predicciones.")
            if not np.array_equal(np.diag(matrix), np.full(20, 56)) or matrix.sum() != 1120:
                raise ValueError("La matriz no contiene los 56 aciertos reportados por clase.")
            metrics = csv_rows(official / f"metrics_{suffix}.csv")
            for index, row in enumerate(metrics):
                tp = int(matrix[index, index])
                fp = int(matrix[:, index].sum()) - tp
                fn = int(matrix[index].sum()) - tp
                tn = 1120 - tp - fp - fn
                if [int(row[key]) for key in ("object_id", "TP", "FP", "TN", "FN", "support")] != [index + 1, tp, fp, tn, fn, 56]:
                    raise ValueError("Conteos por objeto inconsistentes.")
                expected = (tp / (tp + fp) if tp + fp else 0, tp / (tp + fn) if tp + fn else 0, 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)
                if not np.allclose([float(row[key]) for key in ("precision", "recall", "f1")], expected, rtol=0, atol=1e-12):
                    raise ValueError("Métricas por objeto inconsistentes.")
            summary = next(row for row in summaries if row["color_space"] == space)
            expected = [np.trace(matrix) / 1120, 1 - np.trace(matrix) / 1120] + [np.mean([float(row[key]) for row in metrics]) for key in ("precision", "recall", "f1")]
            if not np.allclose([float(summary[key]) for key in ("accuracy", "error", "precision_macro", "recall_macro", "f1_macro")], expected, rtol=0, atol=1e-12):
                raise ValueError("Resumen global inconsistente.")
            regenerated = temporary / f"confusion_{suffix}.png"
            plot_confusion(matrix, space, regenerated)
            with Image.open(regenerated) as image, Image.open(official / regenerated.name) as original:
                identical = np.array_equal(np.asarray(image), np.asarray(original))
            if not identical:
                raise ValueError(f"La figura oficial difiere de la regenerada: {space}.")
            checks[space] = {"predictions": len(rows), "diagonal_per_class": 56, "metrics_consistent": True, "confusion_figure_pixel_identical": True}
        numeric = [{key: value if key == "color_space" else float(value) for key, value in row.items()} for row in summaries]
        regenerated = temporary / "comparison.png"
        plot_comparison(numeric, regenerated)
        with Image.open(regenerated) as image, Image.open(official / "comparison.png") as original:
            if not np.array_equal(np.asarray(image), np.asarray(original)):
                raise ValueError("La comparación oficial no coincide con el resumen CSV.")
    return {"spaces": checks, "comparison_figure_pixel_identical": True, "original_source_hashes_verified": True}


def cli_predict(root: Path, dataset: Path, image: Path) -> list[dict]:
    command = [sys.executable, "-m", "src.coil100.recognize", "--dataset-root", str(dataset), "--inventory", str(root / "data/processed/inventory.csv"), "--image", str(image), "--color-space", "all"]
    result = subprocess.run(command, cwd=root, capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(result.stdout)


def verify(root: Path, dataset: Path, output: Path, seed: int = SEED) -> dict:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use un directorio vacío para preservar diagnósticos anteriores.")
    protected = [root / "data/processed/inventory.csv", *(root / "src/coil100").glob("*.py"), *(root / "artifacts/phase3b").glob("*")]
    before = {path.relative_to(root).as_posix(): sha256(path) for path in protected if path.is_file()}
    saved_check = verify_saved_results(root)
    entries = read_inventory(root / "data/processed/inventory.csv", dataset)
    train = sorted((entry for entry in entries if entry.split == "train"), key=lambda entry: (entry.object_id, entry.angle_degrees))
    test = sorted((entry for entry in entries if entry.split == "test"), key=lambda entry: (entry.object_id, entry.angle_degrees))
    references = []
    for entry in train:
        image = load_bgr(entry.image_path)
        references.append(TrainingReference(entry, extract_all(image, segment_object(image).mask)))
    permuted = permute_training_labels(references, seed)
    negative_model = Histogram1NN(permuted)
    rename_rows = []
    with tempfile.TemporaryDirectory() as directory:
        for obj in (1, 4, 6, 11, 15, 19):
            entry = next(entry for entry in test if entry.object_id == obj)
            misleading_name = f"obj{21 - obj}__355.png"
            copied = Path(directory) / misleading_name
            shutil.copyfile(entry.image_path, copied)
            original = cli_predict(root, dataset, entry.image_path)
            renamed = cli_predict(root, dataset, copied)
            if original != renamed or sha256(entry.image_path) != sha256(copied):
                raise ValueError("Posible dependencia del nombre de consulta: control de renombrado fallido.")
            for prediction in original:
                rename_rows.append({"original_image": entry.image_path.name, "misleading_name": misleading_name, "color_space": prediction["color_space"], "predicted_object_id": prediction["object_id"], "distance": prediction["distance"], "nearest_image": Path(prediction["nearest_image"]).name, "identical_prediction": True})
    official = {space: {row["image_path"]: row for row in csv_rows(root / f"artifacts/phase3b/predictions_{space.lower()}.csv")} for space in COLOR_SPACES}
    negative_rows = []
    ties = {space: 0 for space in COLOR_SPACES}
    minimum_gap = {space: 1.0 for space in COLOR_SPACES}
    crosscheck_count = 0
    for entry in test:
        image = load_bgr(entry.image_path)
        histograms = extract_all(image, segment_object(image).mask)
        for space in COLOR_SPACES:
            distances = np.array([cv2.compareHist(histograms[space], ref.histograms[space], cv2.HISTCMP_BHATTACHARYYA) for ref in references])
            order = np.argsort(distances, kind="stable")
            gap = float(distances[order[1]] - distances[order[0]])
            minimum_gap[space] = min(minimum_gap[space], gap)
            ties[space] += int(gap <= 1e-12)
            nearest = references[int(order[0])].entry
            recorded = official[space][entry.image_path.relative_to(dataset).as_posix()]
            if nearest.image_path.name != recorded["nearest_image"] or nearest.object_id != int(recorded["predicted_object_id"]) or not np.isclose(distances[order[0]], float(recorded["distance"]), rtol=0, atol=1e-12):
                raise ValueError("La distancia calculada desde los píxeles no reproduce el vecino oficial.")
            crosscheck_count += 1
            prediction = negative_model.predict(histograms[space], space)
            negative_rows.append({"image_path": entry.image_path.name, "true_object_id": entry.object_id, "predicted_object_id": prediction.object_id, "color_space": space, "distance": prediction.distance, "nearest_image": Path(prediction.nearest_image).name, "correct": prediction.object_id == entry.object_id})
    negative_summary = {}
    for space in COLOR_SPACES:
        rows = [row for row in negative_rows if row["color_space"] == space]
        correct = sum(row["correct"] for row in rows)
        if correct == len(rows):
            raise ValueError("El control negativo sigue perfecto: requiere investigación antes de continuar.")
        negative_summary[space] = {"correct": correct, "total": len(rows), "accuracy": correct / len(rows)}
    for relative, expected in before.items():
        if sha256(root / relative) != expected:
            raise ValueError(f"Cambió un archivo protegido: {relative}")
    result = {
        "scope": "phase4_diagnostics_do_not_replace_phase3b",
        "seed": seed, "training_references": len(references), "renamed_images": 6,
        "renamed_comparisons": len(rename_rows), "rename_predictions_identical": True,
        "official_neighbors_verified_from_pixels": crosscheck_count,
        "minimum_distance_gaps": minimum_gap, "ties_within_1e_minus_12": ties,
        "negative_control": negative_summary, "saved_results_check": saved_check,
        "protected_files_sha256": before, "protected_files_unchanged": True,
        "tie_caveat": "Reference labels, angles and paths order exact ties; query metadata is not passed to inference.",
        "historical_limit": "Source inspection, prior artifacts and controls support no test-based tuning; this is not a complete external history audit.",
    }
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "renamed_queries.csv", rename_rows)
    write_csv(output / "negative_predictions.csv", negative_rows)
    write_csv(output / "permuted_training_labels.csv", [{"image_path": ref.entry.image_path.name, "original_object_id": original.entry.object_id, "permuted_object_id": ref.entry.object_id} for original, ref in zip(references, permuted)])
    (output / "verification_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/coil-100"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase4"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        result = verify(root, args.dataset_root.resolve(), args.output_dir.resolve(), args.seed)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(json.dumps({key: result[key] for key in ("seed", "renamed_comparisons", "official_neighbors_verified_from_pixels", "ties_within_1e_minus_12", "negative_control", "protected_files_unchanged")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
