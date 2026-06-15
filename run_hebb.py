"""Run the full Hebbian experiment pipeline from the project root.

This entrypoint executes the capacity and noise experiments in sequence so the
CSV outputs and per-size figures are generated with a single command.
"""

from __future__ import annotations

from src.experiments.hebb_capacity import main as run_capacity_experiment
from src.experiments.hebb_noise import main as run_noise_experiment


def main() -> None:
    """Run every Hebbian experiment in order."""

    print("Running Hebbian capacity experiment...")
    run_capacity_experiment()

    print("Running Hebbian noise experiment...")
    run_noise_experiment()

    print("Hebbian pipeline completed.")


if __name__ == "__main__":
    main()