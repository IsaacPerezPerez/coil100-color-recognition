"""Métricas multiclase con filas reales y columnas predichas."""

from collections.abc import Sequence

import numpy as np

from .dataset import OBJECT_IDS


def confusion_matrix(
    actual: Sequence[int], predicted: Sequence[int], class_ids: Sequence[int] = OBJECT_IDS
) -> np.ndarray:
    if len(actual) != len(predicted):
        raise ValueError("Las listas de clases reales y predichas deben tener igual longitud.")
    indices = {object_id: index for index, object_id in enumerate(class_ids)}
    if len(indices) != len(class_ids):
        raise ValueError("El orden de clases no debe contener duplicados.")
    matrix = np.zeros((len(class_ids), len(class_ids)), dtype=np.int64)
    for real, prediction in zip(actual, predicted):
        if real not in indices or prediction not in indices:
            raise ValueError("Clase real o predicha fuera del orden de clases.")
        matrix[indices[real], indices[prediction]] += 1
    return matrix


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def recognition_metrics(
    matrix: np.ndarray, class_ids: Sequence[int] = OBJECT_IDS
) -> tuple[list[dict], dict]:
    matrix = np.asarray(matrix)
    if (
        matrix.shape != (len(class_ids), len(class_ids))
        or not len(class_ids)
        or not np.isfinite(matrix).all()
        or (matrix < 0).any()
        or not np.equal(matrix, np.floor(matrix)).all()
    ):
        raise ValueError("La matriz debe ser cuadrada con conteos enteros no negativos y clases conocidas.")
    total = int(matrix.sum())
    rows = []
    for index, object_id in enumerate(class_ids):
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum()) - tp
        fn = int(matrix[index, :].sum()) - tp
        tn = total - tp - fp - fn
        rows.append({
            "object_id": object_id, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "precision": safe_ratio(tp, tp + fp),
            "recall": safe_ratio(tp, tp + fn),
            "f1": safe_ratio(2 * tp, 2 * tp + fp + fn), "support": tp + fn,
        })
    correct = int(np.trace(matrix))
    accuracy = safe_ratio(correct, total)
    summary = {
        "correct": correct, "total": total, "accuracy": accuracy, "error": 1 - accuracy,
        "precision_macro": sum(row["precision"] for row in rows) / len(rows),
        "recall_macro": sum(row["recall"] for row in rows) / len(rows),
        "f1_macro": sum(row["f1"] for row in rows) / len(rows),
    }
    return rows, summary
