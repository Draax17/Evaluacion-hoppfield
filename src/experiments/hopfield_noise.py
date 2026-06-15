"""Evaluate Hopfield recovery under 5%, 10% and 15% noise."""

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
OUTPUT_PATH = ROOT / "results" / "tables" / "hopfield_noise.csv"
FIGURES_DIR = ROOT / "results" / "figures"


def load_dataset(size: int, dataset_dir: Path = DATASET_DIR) -> tuple[np.ndarray, np.ndarray]:
    """Load labels and vector patterns from alphabet_{size}x{size}.npz."""
    path = dataset_dir / f"alphabet_{size}x{size}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    data = np.load(path, allow_pickle=False)
    return data["labels"], data["vectors"].astype(np.int8)


def add_noise(pattern: np.ndarray, noise_level: float, rng: np.random.Generator) -> np.ndarray:
    """Flip noise_level percent of bits in a bipolar vector."""
    noisy = np.asarray(pattern, dtype=np.int8).copy()
    n_flip = int(round(noise_level * noisy.size))
    if n_flip > 0:
        idx = rng.choice(noisy.size, size=n_flip, replace=False)
        noisy[idx] *= -1
    return noisy


def similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return bitwise similarity between two vectors."""
    return float(np.mean(np.asarray(a) == np.asarray(b)))


def evaluate_size(
    size: int,
    noise_levels: Iterable[float] = (0.05, 0.10, 0.15),
    repetitions: int = 10,
    max_patterns: int = 26,
    max_iter: int = 100,
    random_state: int = 42,
) -> list[dict[str, int | float | str]]:
    """Evaluate noisy recovery for 1..max_patterns stored letters."""
    rng = np.random.default_rng(random_state)
    labels, vectors = load_dataset(size)
    rows: list[dict[str, int | float | str]] = []

    for n_patterns in range(1, min(max_patterns, len(vectors)) + 1):
        patterns = vectors[:n_patterns]
        used_labels = labels[:n_patterns].astype(str)
        net = HopfieldNetwork(n_neurons=patterns.shape[1], random_state=random_state)
        net.train(patterns)

        for noise_level in noise_levels:
            correct = 0
            recovered_sims: list[float] = []
            noisy_sims: list[float] = []

            for _ in range(repetitions):
                for pattern in patterns:
                    noisy = add_noise(pattern, noise_level, rng)
                    recovered = net.recall(noisy, max_iter=max_iter)
                    correct += int(np.array_equal(recovered, pattern))
                    noisy_sims.append(similarity(noisy, pattern))
                    recovered_sims.append(similarity(recovered, pattern))

            total = repetitions * len(patterns)
            rows.append(
                {
                    "size": size,
                    "n_neurons": size * size,
                    "n_patterns": n_patterns,
                    "stored_labels": "".join(used_labels),
                    "noise_level": noise_level,
                    "noise_percent": int(noise_level * 100),
                    "repetitions": repetitions,
                    "correct": correct,
                    "total_trials": total,
                    "accuracy": correct / total,
                    "mean_noisy_similarity": float(np.mean(noisy_sims)),
                    "mean_recovered_similarity": float(np.mean(recovered_sims)),
                    "std_recovered_similarity": float(np.std(recovered_sims)),
                }
            )
    return rows


def run_noise_experiment(
    sizes: Iterable[int] = (7, 10, 15),
    output_path: Path = OUTPUT_PATH,
    noise_levels: Iterable[float] = (0.05, 0.10, 0.15),
    repetitions: int = 10,
    max_patterns: int = 26,
    max_iter: int = 100,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run noise experiments for every configured dataset size and save CSV."""
    rows: list[dict[str, int | float | str]] = []
    for size in sizes:
        rows.extend(
            evaluate_size(
                size=size,
                noise_levels=noise_levels,
                repetitions=repetitions,
                max_patterns=max_patterns,
                max_iter=max_iter,
                random_state=random_state,
            )
        )
    results = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    return results


def save_noise_plots(results: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> list[Path]:
    """Save one grouped bar chart per grid size."""

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for size, subset in results.groupby("size"):
        fig, ax = plt.subplots(figsize=(8, 5))
        pivot = subset.pivot(index="n_patterns", columns="noise_level", values="accuracy").sort_index()
        x_positions = np.arange(len(pivot.index))
        noise_levels = list(pivot.columns)
        bar_width = 0.8 / max(len(noise_levels), 1)

        for index, noise_level in enumerate(noise_levels):
            offsets = x_positions + (index - (len(noise_levels) - 1) / 2) * bar_width
            ax.bar(
                offsets,
                pivot[noise_level].to_numpy(),
                width=bar_width,
                label=f"noise = {noise_level:.2f}",
            )

        ax.set_title(f"Hopfield noise recovery - {size}x{size}")
        ax.set_xlabel("Stored patterns")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(pivot.index.astype(int))
        ax.grid(True, alpha=0.3)
        ax.legend(title="Noise level")
        fig.tight_layout()

        figure_path = output_dir / f"hopfield_noise_{size}x{size}.png"
        fig.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(figure_path)

    return saved_paths


def main() -> None:
    """Command-line entry point."""
    results = run_noise_experiment()
    figure_paths = save_noise_plots(results)
    print(f"Saved: {OUTPUT_PATH}")
    for figure_path in figure_paths:
        print(f"Saved figure: {figure_path}")
    print(results)


if __name__ == "__main__":
    main()
