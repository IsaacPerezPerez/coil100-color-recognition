"""Evaluación final del método fijo sobre las 1120 vistas test del inventario."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .audit_training import write_csv
from .classifier import Histogram1NN
from .dataset import OBJECT_IDS, TRAIN_VIEW_INDICES
from .evaluation_plots import plot_comparison, plot_confusion
from .features import COLOR_SPACES, RANGES
from .inventory import read_inventory
from .metrics import confusion_matrix, recognition_metrics
from .preprocessing import SegmentationConfig, load_bgr


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def method_hashes() -> dict[str, str]:
    folder = Path(__file__).parent
    return {name: file_hash(folder / name) for name in ("dataset.py", "inventory.py", "preprocessing.py", "features.py", "classifier.py")}


def write_confusion(path: Path, matrix) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["real/predicted", *(f"obj{obj}" for obj in OBJECT_IDS)])
        for object_id, row in zip(OBJECT_IDS, matrix):
            writer.writerow([f"obj{object_id}", *(int(value) for value in row)])


def write_notes(output: Path, summaries: list[dict], class_metrics: dict[str, list[dict]], metadata: dict) -> None:
    lines = [
        "# Evaluación final — Fase 3B", "",
        "Los resultados corresponden a 1120 vistas no utilizadas como referencia de los mismos 20 objetos físicos de COIL-100. No evalúan reconocimiento de objetos arbitrarios ni generalización a otras condiciones de captura.", "",
        "## Metodología congelada", "",
        "Se mantuvieron la división original, el umbral 40 y su respaldo ya definido, el componente mayor con relleno de huecos, una máscara compartida por imagen, los histogramas 8×8×8 normalizados L1 y el 1-NN con distancia de Bhattacharyya de OpenCV. Las únicas referencias fueron las 320 filas train. No se ajustó el método usando test.", "",
        "Se ejecutaron exactamente 1120 predicciones por espacio: 3360 en total. Las tres evaluaciones compartieron imágenes, etiquetas y segmentación. Los archivos individuales conservan todos los aciertos y errores.", "",
        "## Resultados", "",
        "| Espacio | Aciertos / 1120 | Accuracy | Error | Precision macro | Recall macro | F1 macro |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        values = " | ".join(f"{row[field]:.6f}" for field in ("accuracy", "error", "precision_macro", "recall_macro", "f1_macro"))
        lines.append(f"| {row['color_space']} | {row['correct']} / {row['total']} | {values} |")
    lines += ["", "## Objetos con más errores", ""]
    for space in COLOR_SPACES:
        rows = class_metrics[space]
        maximum = max(row["FN"] for row in rows)
        affected = [f"obj{row['object_id']} ({row['FN']} errores de 56)" for row in rows if row["FN"] == maximum]
        lines.append(f"- {space}: " + (", ".join(affected) if maximum else "ningún objeto presentó errores.") )
    lines += [
        "", "## Definición y verificación de métricas", "",
        "Las filas de las matrices son clases reales y las columnas predichas; el orden fijo es obj1 a obj20. TP es la diagonal, FP la suma de columna menos TP, FN la suma de fila menos TP y TN el total menos TP, FP y FN. Precision=TP/(TP+FP), Recall=TP/(TP+FN), F1=2TP/(2TP+FP+FN). Toda división entre cero devuelve 0. Los promedios macro incluyen las 20 clases.", "",
        "Accuracy global es la suma de la diagonal dividida entre 1120; Error=1−Accuracy. No se utiliza la media de exactitudes uno contra el resto. Las pruebas verifican conteos, identidades matemáticas y ausencia de NaN/infinito, tanto con matrices conocidas como con los archivos reales guardados.", "",
        "## Integridad y reproducción", "",
        f"SHA-256 del inventario antes y después: `{metadata['inventory_sha256_before']}`.", "",
        f"Revisión Git: {metadata['git_revision'] or 'sin commit válido; se registran hashes de archivos en evaluation_metadata.json'}. Los módulos del método conservaron sus hashes durante toda la ejecución.", "",
        "```powershell", ".\\.venv\\Scripts\\python.exe -m src.coil100.evaluate", ".\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v", "```", "",
        "El comando rechaza sobrescribir un directorio que ya contiene resultados de evaluación. Para otra reproducción usa `--output-dir artifacts/phase3b_reproduction`. Las pruebas de resultados reales leen los CSV guardados y no ejecutan predicciones adicionales.", "",
        "Versiones: " + ", ".join(f"{name} {version}" for name, version in metadata["versions"].items()) + ".", "",
        "## Limitaciones", "",
        "Las máscaras pueden perder ruedas y bordes oscuros o incluir fondo dentro de asas. Los histogramas no conservan estructura espacial, el clasificador asigna siempre una clase y la distancia no es una confianza calibrada. La auditoría Leave-One-Out previa es un diagnóstico de entrenamiento independiente de esta evaluación. No se buscaron hiperparámetros ni se filtraron resultados desfavorables.", "",
    ]
    (output / "evaluation_notes.md").write_text("\n".join(lines), encoding="utf-8")


def evaluate(inventory: Path, dataset_root: Path, output: Path) -> list[dict]:
    if output.exists() and any(output.iterdir()):
        raise ValueError("El directorio de salida no está vacío; use otro para preservar resultados existentes.")
    inventory_before = file_hash(inventory)
    code_before = method_hashes()
    config = SegmentationConfig()
    if config.background_threshold != 40:
        raise ValueError("La segmentación difiere de la configuración congelada de Fase 3A.")
    entries = read_inventory(inventory, dataset_root)
    train_angles = {index * 5 for index in TRAIN_VIEW_INDICES}
    if any(entry.split != ("train" if entry.angle_degrees in train_angles else "test") for entry in entries):
        raise ValueError("La división difiere de la regla original de Fase 1.")
    train = [entry for entry in entries if entry.split == "train"]
    test = sorted((entry for entry in entries if entry.split == "test"), key=lambda entry: (entry.object_id, entry.angle_degrees))
    train_paths = {entry.image_path for entry in train}
    if len(train) != 320 or len(test) != 1120 or train_paths.intersection(entry.image_path for entry in test):
        raise ValueError("La división no cumple los conteos o contiene rutas compartidas.")
    revision = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], capture_output=True, text=True, check=False)
    metadata = {
        "scope": "held_out_views_of_the_same_20_physical_objects",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "inventory_sha256_before": inventory_before, "method_sha256_before": code_before,
        "git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
        "versions": {"python": platform.python_version(), **{name: importlib.metadata.version(name) for name in ("numpy", "opencv-python-headless", "Pillow", "matplotlib")}},
        "config": {"segmentation": asdict(config), "bins_per_channel": 8, "feature_length": 512, "normalization": "L1", "histogram_ranges": RANGES, "classifier": "1-NN", "distance": "cv2.HISTCMP_BHATTACHARYYA", "class_order": list(OBJECT_IDS), "zero_division": 0},
        "train_count": len(train), "test_count": len(test),
        "evaluation_source_sha256": {name: file_hash(Path(__file__).parent / name) for name in ("evaluate.py", "metrics.py", "evaluation_plots.py")},
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    start = time.perf_counter()
    model = Histogram1NN.from_inventory(inventory, dataset_root, config)
    if {entry.image_path for entry in model.references} != train_paths:
        raise ValueError("Las referencias no coinciden exactamente con train.")
    predictions = {space: [] for space in COLOR_SPACES}
    for entry in test:
        for prediction in model.recognize(load_bgr(entry.image_path)):
            nearest = Path(prediction.nearest_image)
            if nearest not in train_paths:
                raise ValueError("Se seleccionó un vecino fuera de train.")
            predictions[prediction.color_space].append({
                "image_path": entry.image_path.relative_to(dataset_root.resolve()).as_posix(),
                "true_object_id": entry.object_id, "predicted_object_id": prediction.object_id,
                "color_space": prediction.color_space, "distance": prediction.distance,
                "nearest_image": nearest.relative_to(dataset_root.resolve()).as_posix(),
            })
    metadata["recognition_seconds"] = time.perf_counter() - start
    metadata["inventory_sha256_after"] = file_hash(inventory)
    metadata["method_sha256_after"] = method_hashes()
    if metadata["inventory_sha256_after"] != inventory_before or metadata["method_sha256_after"] != code_before:
        raise ValueError("El inventario o el método cambió durante la evaluación.")
    metadata["predictions_per_space"] = {space: len(rows) for space, rows in predictions.items()}
    metadata["prediction_count"] = sum(metadata["predictions_per_space"].values())
    summaries, class_metrics, matrices = [], {}, {}
    for space in COLOR_SPACES:
        rows = predictions[space]
        if len(rows) != 1120:
            raise ValueError(f"Número inesperado de predicciones en {space}.")
        matrix = confusion_matrix([row["true_object_id"] for row in rows], [row["predicted_object_id"] for row in rows])
        metrics, summary = recognition_metrics(matrix)
        matrices[space] = matrix
        class_metrics[space] = metrics
        summaries.append({"color_space": space, **summary})
        suffix = space.lower()
        write_csv(output / f"predictions_{suffix}.csv", rows)
        write_csv(output / f"metrics_{suffix}.csv", metrics)
        write_confusion(output / f"confusion_{suffix}.csv", matrix)
    write_csv(output / "summary.csv", summaries)
    (output / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for space, matrix in matrices.items():
        plot_confusion(matrix, space, output / f"confusion_{space.lower()}.png")
    plot_comparison(summaries, output / "comparison.png")
    metadata["completed_utc"] = datetime.now(timezone.utc).isoformat()
    (output / "evaluation_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_notes(output, summaries, class_metrics, metadata)
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/coil-100"))
    parser.add_argument("--inventory", type=Path, default=Path("data/processed/inventory.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/phase3b"))
    args = parser.parse_args()
    try:
        summaries = evaluate(args.inventory, args.dataset_root, args.output_dir)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summaries, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
