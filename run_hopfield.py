"""Run the full Hopfield experiment pipeline from the project root.

This entrypoint prepares the alphabet datasets expected by the Hopfield
experiments and then executes the capacity and noise evaluations in sequence.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.dataset_generator import generate_alphabet
from src.experiments.hopfield_capacity import main as run_capacity_experiment
from src.experiments.hopfield_noise import main as run_noise_experiment


ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
GRID_SIZES: tuple[int, ...] = (7, 10, 15)


def prepare_datasets() -> None:
    """Generate the `.npz` files required by the Hopfield experiments."""

    datasets = generate_alphabet(sizes=GRID_SIZES)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for size, bundle in datasets.items():
        output_path = DATASET_DIR / f"alphabet_{size}x{size}.npz"
        np.savez_compressed(
            output_path,
            labels=np.array(bundle.letters, dtype="U1"),
            vectors=bundle.vectors.astype(np.int8, copy=False),
            matrices=bundle.matrices.astype(np.int8, copy=False),
        )
        print(f"Prepared dataset: {output_path}")


def main() -> None:
    """Prepare the datasets and run every Hopfield experiment."""

    print("Preparing Hopfield datasets...")
    prepare_datasets()

    print("Running Hopfield capacity experiment...")
    run_capacity_experiment()

    print("Running Hopfield noise experiment...")
    run_noise_experiment()

    print("Hopfield pipeline completed.")


if __name__ == "__main__":
    main()