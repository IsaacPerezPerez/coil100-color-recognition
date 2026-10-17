"""Figuras estáticas para los resultados de reconocimiento."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .dataset import OBJECT_IDS


def plot_confusion(matrix: np.ndarray, space: str, output: Path) -> None:
    labels = [f"obj{obj}" for obj in OBJECT_IDS]
    fig, ax = plt.subplots(figsize=(10, 9), layout="constrained")
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=56)
    ax.set_xticks(range(20), labels, rotation=60, ha="right", fontsize=9)
    ax.set_yticks(range(20), labels, fontsize=9)
    ax.set_xlabel("Clase predicha")
    ax.set_ylabel("Clase real")
    ax.set_title(f"Matriz de confusión — {space}\n1120 vistas de prueba; 56 por objeto")
    for row in range(20):
        for column in range(20):
            value = int(matrix[row, column])
            if value:
                ax.text(column, row, str(value), ha="center", va="center", fontsize=8, color="white" if value > 28 else "black")
    fig.colorbar(image, ax=ax, label="Número de imágenes", shrink=0.85)
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_comparison(summaries: list[dict], output: Path) -> None:
    fields = ("accuracy", "precision_macro", "recall_macro", "f1_macro")
    labels = ("Accuracy", "Precision macro", "Recall macro", "F1 macro")
    x = np.arange(len(fields))
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    for index, summary in enumerate(summaries):
        values = [summary[field] for field in fields]
        bars = ax.bar(x + (index - 1) * 0.24, values, 0.24, label=summary["color_space"])
        ax.bar_label(bars, labels=[f"{value:.4f}" for value in values], fontsize=8, padding=3)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.12)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_ylabel("Valor de la métrica")
    ax.set_title("Reconocimiento de vistas de prueba — COIL-100, obj1 a obj20")
    ax.legend(loc="upper center", ncols=3)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    fig.savefig(output, dpi=200)
    plt.close(fig)
