"""Capacity experiment for the Hebbian autoassociative memory.

The experiment evaluates how many alphabet patterns the network recalls
correctly when trained with the first 1..26 stored letters for each available
grid size in the dataset.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.dataset_generator import load_dataset
from src.models.hebbian import HebbianNetwork


DATASET_PATH = Path(__file__).resolve().parents[1] / "data" / "dataset" / "alphabet.npz"
OUTPUT_CSV = Path(__file__).resolve().parents[2] / "results" / "tables" / "hebb_capacity.csv"
OUTPUT_FIGURE = Path(__file__).resolve().parents[2] / "results" / "figures" / "hebb_capacity.png"


def evaluate_capacity(dataset_path: str | Path = DATASET_PATH) -> pd.DataFrame:
    """Evaluate recall capacity for 1..26 stored patterns across all sizes."""

    dataset = load_dataset(dataset_path)
    rows: list[dict[str, float | int]] = []

    for size in sorted(dataset):
        bundle = dataset[size]
        patterns = bundle.matrices
        total_patterns = int(patterns.shape[0])

        for n_patterns in range(1, total_patterns + 1):
            network = HebbianNetwork().train(patterns[:n_patterns])
            recalled = np.asarray([network.predict(pattern) for pattern in patterns[:n_patterns]])
            matches = np.all(recalled == patterns[:n_patterns], axis=tuple(range(1, recalled.ndim)))
            correct = int(np.sum(matches))
            accuracy = float(correct / n_patterns)

            rows.append(
                {
                    "size": int(size),
                    "n_patterns": int(n_patterns),
                    "correct": correct,
                    "total": int(n_patterns),
                    "accuracy": accuracy,
                }
            )

    return pd.DataFrame(rows)


def save_capacity_plot(results: pd.DataFrame, output_path: str | Path = OUTPUT_FIGURE) -> Path:
    """Save a line plot of recall accuracy versus stored patterns."""

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    for size, subset in results.groupby("size"):
        ax.plot(subset["n_patterns"], subset["accuracy"], marker="o", label=f"{size}x{size}")

    ax.set_title("Hebbian capacity")
    ax.set_xlabel("Stored patterns")
    ax.set_ylabel("Exact recall accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Grid size")
    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_file


def main() -> None:
    """Run the capacity experiment and write CSV and figure outputs."""

    results = evaluate_capacity()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)
    save_capacity_plot(results)
    print(f"Capacity results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
