"""Noise robustness experiment for the Hebbian autoassociative memory.

The experiment corrupts the stored alphabet patterns with 5%, 10%, and 15%
bit flips, then evaluates whether the Hebbian network recovers the original
pattern in one direct step.
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
OUTPUT_CSV = Path(__file__).resolve().parents[2] / "results" / "tables" / "hebb_noise.csv"
OUTPUT_FIGURES_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"
NOISE_LEVELS: tuple[float, ...] = (0.05, 0.10, 0.15)


def _add_bipolar_noise(pattern: np.ndarray, noise_level: float, rng: np.random.Generator) -> np.ndarray:
    """Flip an exact proportion of bits in a bipolar pattern."""

    flat = np.asarray(pattern, dtype=np.int8).reshape(-1)
    n_flips = int(round(noise_level * flat.size))
    noisy = flat.copy()

    if n_flips > 0:
        indices = rng.choice(flat.size, size=n_flips, replace=False)
        noisy[indices] *= -1

    return noisy.reshape(pattern.shape)


def evaluate_noise(dataset_path: str | Path = DATASET_PATH, seed: int = 42) -> pd.DataFrame:
    """Evaluate recall under controlled bit-flip noise for all sizes."""

    dataset = load_dataset(dataset_path)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float | int]] = []

    for size in sorted(dataset):
        bundle = dataset[size]
        patterns = bundle.matrices
        network = HebbianNetwork().train(patterns)

        for noise_level in NOISE_LEVELS:
            exact_matches = 0
            bit_accuracies: list[float] = []

            for original_pattern in patterns:
                noisy_pattern = _add_bipolar_noise(original_pattern, noise_level, rng)
                recalled_pattern = network.predict(noisy_pattern)
                exact_matches += int(np.array_equal(recalled_pattern, original_pattern))
                bit_accuracies.append(float(np.mean(recalled_pattern == original_pattern)))

            rows.append(
                {
                    "size": int(size),
                    "noise_level": float(noise_level),
                    "exact_recovery_rate": float(exact_matches / len(patterns)),
                    "mean_bit_accuracy": float(np.mean(bit_accuracies)),
                    "total_patterns": int(len(patterns)),
                }
            )

    return pd.DataFrame(rows)


def save_noise_plots(results: pd.DataFrame, output_dir: str | Path = OUTPUT_FIGURES_DIR) -> list[Path]:
    """Save one grouped bar chart per grid size for the noise recovery results."""

    figures_dir = Path(output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: list[Path] = []
    for size, subset in results.groupby("size"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(
            subset["noise_level"].astype(str),
            subset["exact_recovery_rate"],
            color="#4C72B0",
        )

        ax.set_title(f"Hebbian recovery under noise - {size}x{size}")
        ax.set_xlabel("Noise level")
        ax.set_ylabel("Exact recovery rate")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()

        output_file = figures_dir / f"hebb_noise_{size}x{size}.png"
        fig.savefig(output_file, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(output_file)

    return saved_paths


def main() -> None:
    """Run the noise experiment and write CSV and figure outputs."""

    results = evaluate_noise()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_CSV, index=False)
    figure_paths = save_noise_plots(results)
    print(f"Noise results saved to {OUTPUT_CSV}")
    for figure_path in figure_paths:
        print(f"Noise figure saved to {figure_path}")


if __name__ == "__main__":
    main()
