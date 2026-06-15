"""Run the Hopfield vs Hebbian comparison pipeline from the project root."""

from __future__ import annotations

from src.experiments.compare_hopfield_hebb import main as run_comparison


def main() -> None:
    """Execute the comparison workflow."""

    print("Running Hopfield vs Hebbian comparison...")
    run_comparison()
    print("Comparison completed.")


if __name__ == "__main__":
    main()