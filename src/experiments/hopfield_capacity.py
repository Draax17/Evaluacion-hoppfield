"""Evaluate Hopfield storage capacity for alphabet datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.hopfield import HopfieldNetwork

ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT / "dataset"
OUTPUT_PATH = ROOT / "results" / "tables" / "hopfield_capacity.csv"
FIGURES_DIR = ROOT / "results" / "figures"


def load_dataset(size: int, dataset_dir: Path = DATASET_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Load labels and vector patterns from alphabet_{size}x{size}.npz."""
    path = dataset_dir / f"alphabet_{size}x{size}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = np.load(path, allow_pickle=False)
    return data["labels"], data["vectors"].astype(np.int8)


def evaluate_size(size: int, max_patterns: int = 26, max_iter: int = 100) -> list[dict[str, int | float | str]]:
    """Evaluate recovery accuracy when storing 1..max_patterns letters."""
    labels, vectors = load_dataset(size)
    rows: list[dict[str, int | float | str]] = []

    for n_patterns in range(1, min(max_patterns, len(vectors)) + 1):
        patterns = vectors[:n_patterns]
        used_labels = labels[:n_patterns].astype(str)
        net = HopfieldNetwork(n_neurons=patterns.shape[1])
        net.train(patterns)

        correct = 0
        iterations: list[int] = []
        for pattern in patterns:
            recovered = net.recall(pattern, max_iter=max_iter)
            correct += int(np.array_equal(recovered, pattern))
            iterations.append(net.last_iterations)

        rows.append(
            {
                "size": size,
                "n_neurons": size * size,
                "n_patterns": n_patterns,
                "stored_labels": "".join(used_labels),
                "correct": correct,
                "total": len(patterns),
                "accuracy": correct / len(patterns),
                "mean_iterations": float(np.mean(iterations)),
            }
        )
    return rows


def run_capacity_experiment(
    sizes: Iterable[int] = (7, 10, 15),
    output_path: Path = OUTPUT_PATH,
    max_patterns: int = 26,
    max_iter: int = 100,
) -> pd.DataFrame:
    """Run capacity experiments for every configured dataset size and save CSV."""
    rows: list[dict[str, int | float | str]] = []
    for size in sizes:
        rows.extend(evaluate_size(size=size, max_patterns=max_patterns, max_iter=max_iter))

    results = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


def save_capacity_plots(results: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> list[Path]:
    """Save one capacity figure per grid size."""

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for size, subset in results.groupby("size"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(subset["n_patterns"], subset["accuracy"], marker="o", color="#4C72B0")
        ax.set_title(f"Hopfield capacity - {size}x{size}")
        ax.set_xlabel("Stored patterns")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        figure_path = output_dir / f"hopfield_capacity_{size}x{size}.png"
        fig.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(figure_path)

    return saved_paths


def main() -> None:
    """Command-line entry point."""
    results = run_capacity_experiment()
    figure_paths = save_capacity_plots(results)
    print(f"Saved: {OUTPUT_PATH}")
    for figure_path in figure_paths:
        print(f"Saved figure: {figure_path}")
    print(results)


if __name__ == "__main__":
    main()
