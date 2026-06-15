"""Compare Hopfield and Hebbian networks using generated experiment results.

The comparison focuses on two aspects:
- recovery / storage, using the capacity CSVs;
- tolerance to noise, using the noise CSVs at the maximum storage level.

The module reads the existing CSV outputs, produces comparison tables, and
generates one figure per grid size for each comparison.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = ROOT / "results" / "tables"
FIGURES_DIR = ROOT / "results" / "figures"

HOPFIELD_CAPACITY_CSV = TABLES_DIR / "hopfield_capacity.csv"
HEBB_CAPACITY_CSV = TABLES_DIR / "hebb_capacity.csv"
HOPFIELD_NOISE_CSV = TABLES_DIR / "hopfield_noise.csv"
HEBB_NOISE_CSV = TABLES_DIR / "hebb_noise.csv"

OUTPUT_CAPACITY_CSV = TABLES_DIR / "comparison_capacity.csv"
OUTPUT_NOISE_CSV = TABLES_DIR / "comparison_noise.csv"
OUTPUT_SUMMARY_CSV = TABLES_DIR / "comparison_summary.csv"
OUTPUT_NOISE_BY_LEVEL_CSV = TABLES_DIR / "comparison_noise_by_level.csv"


def _require_files(paths: list[Path]) -> None:
    """Raise a helpful error if any required result file is missing."""

    missing = [path for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(
            "Missing experiment outputs. Run the Hopfield and Hebbian pipelines first:\n"
            f"{formatted}"
        )


def load_capacity_results() -> pd.DataFrame:
    """Load and normalize the Hopfield and Hebbian capacity results."""

    _require_files([HOPFIELD_CAPACITY_CSV, HEBB_CAPACITY_CSV])

    hopfield = pd.read_csv(HOPFIELD_CAPACITY_CSV).copy()
    hopfield["network"] = "Hopfield"
    hopfield = hopfield[["size", "n_patterns", "accuracy", "network"]]

    hebbian = pd.read_csv(HEBB_CAPACITY_CSV).copy()
    hebbian["network"] = "Hebbian"
    hebbian = hebbian[["size", "n_patterns", "accuracy", "network"]]

    return pd.concat([hopfield, hebbian], ignore_index=True)


def load_noise_results() -> pd.DataFrame:
    """Load and normalize the Hopfield and Hebbian noise results.

    Both networks are compared across the full 1..26 stored-pattern sweep.
    """

    _require_files([HOPFIELD_NOISE_CSV, HEBB_NOISE_CSV])

    hopfield = pd.read_csv(HOPFIELD_NOISE_CSV).copy()
    hopfield["network"] = "Hopfield"
    hopfield = hopfield[["size", "n_patterns", "noise_level", "accuracy", "network"]]

    hebbian = pd.read_csv(HEBB_NOISE_CSV).copy()
    hebbian["network"] = "Hebbian"
    hebbian = hebbian[["size", "n_patterns", "noise_level", "accuracy", "network"]]

    return pd.concat([hopfield, hebbian], ignore_index=True)


def save_capacity_plots(results: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> list[Path]:
    """Save one capacity comparison figure per grid size."""

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for size, subset in results.groupby("size"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for network, network_subset in subset.groupby("network"):
            ordered = network_subset.sort_values("n_patterns")
            ax.plot(
                ordered["n_patterns"],
                ordered["accuracy"],
                marker="o",
                linewidth=2,
                label=network,
            )

        ax.set_title(f"Recovery and storage comparison - {size}x{size}")
        ax.set_xlabel("Stored patterns")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Network")
        fig.tight_layout()

        figure_path = output_dir / f"comparison_capacity_{size}x{size}.png"
        fig.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(figure_path)

    return saved_paths


def save_noise_plots(results: pd.DataFrame, output_dir: Path = FIGURES_DIR) -> list[Path]:
    """Save one grouped-bar comparison figure per grid size."""

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for size, subset in results.groupby("size"):
        fig, ax = plt.subplots(figsize=(8, 5))
        pivot = (
            subset.pivot_table(index="n_patterns", columns=["noise_level", "network"], values="accuracy")
            .sort_index()
        )
        x_positions = np.arange(len(pivot.index))
        columns = list(pivot.columns)
        bar_width = 0.8 / max(len(columns), 1)

        for index, column in enumerate(columns):
            noise_level, network = column
            offsets = x_positions + (index - (len(columns) - 1) / 2) * bar_width
            ax.bar(
                offsets,
                pivot[column].to_numpy(),
                width=bar_width,
                label=f"{network} {noise_level:.2f}",
            )

        ax.set_title(f"Noise tolerance comparison - {size}x{size}")
        ax.set_xlabel("Stored patterns")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.0, 1.05)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(pivot.index.astype(int))
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(title="Network / noise", ncol=2)
        fig.tight_layout()

        figure_path = output_dir / f"comparison_noise_{size}x{size}.png"
        fig.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(figure_path)

    return saved_paths


def build_summary(capacity: pd.DataFrame, noise: pd.DataFrame) -> pd.DataFrame:
    """Build a compact summary table for quick comparison."""

    capacity = capacity.sort_values(["size", "network", "n_patterns"])
    noise = noise.sort_values(["size", "network", "noise_level", "n_patterns"])

    capacity_auc_rows: list[dict[str, float | int | str]] = []
    for (size, network), group in capacity.groupby(["size", "network"]):
        capacity_auc_rows.append(
            {
                "size": int(size),
                "network": network,
                "capacity_auc": float(
                    np.trapz(group["accuracy"].to_numpy(), group["n_patterns"].to_numpy())
                    / group["n_patterns"].max()
                ),
            }
        )
    capacity_auc = pd.DataFrame(capacity_auc_rows)

    capacity_summary = (
        capacity.groupby(["size", "network"], as_index=False)
        .agg(
            mean_capacity_accuracy=("accuracy", "mean"),
            std_capacity_accuracy=("accuracy", "std"),
            min_capacity_accuracy=("accuracy", "min"),
            max_capacity_accuracy=("accuracy", "max"),
            final_capacity_accuracy=("accuracy", "last"),
            best_capacity_accuracy=("accuracy", "max"),
        )
    )
    noise_summary = (
        noise.groupby(["size", "network"], as_index=False)
        .agg(
            mean_noise_accuracy=("accuracy", "mean"),
            std_noise_accuracy=("accuracy", "std"),
            min_noise_accuracy=("accuracy", "min"),
            max_noise_accuracy=("accuracy", "max"),
        )
    )

    noise_level_table = build_noise_by_level_summary(noise)
    noise_level_summary = (
        noise_level_table.groupby(["size", "network"], as_index=False)
        .agg(
            mean_noise_level_accuracy=("mean_noise_accuracy", "mean"),
            std_noise_level_accuracy=("mean_noise_accuracy", "std"),
            min_noise_level_accuracy=("min_noise_accuracy", "min"),
            max_noise_level_accuracy=("max_noise_accuracy", "max"),
            mean_noise_level_auc=("noise_auc", "mean"),
            std_noise_level_auc=("noise_auc", "std"),
        )
    )

    summary = capacity_summary.merge(capacity_auc, on=["size", "network"], how="outer")
    summary = summary.merge(noise_summary, on=["size", "network"], how="outer")
    summary = summary.merge(noise_level_summary, on=["size", "network"], how="outer")
    return summary


def build_noise_by_level_summary(noise: pd.DataFrame) -> pd.DataFrame:
    """Summarize noise results by grid size, network, and noise level."""

    grouped = (
        noise.sort_values(["size", "network", "noise_level", "n_patterns"])
        .groupby(["size", "network", "noise_level"], as_index=False)
        .agg(
            mean_noise_accuracy=("accuracy", "mean"),
            std_noise_accuracy=("accuracy", "std"),
            min_noise_accuracy=("accuracy", "min"),
            max_noise_accuracy=("accuracy", "max"),
            final_noise_accuracy=("accuracy", "last"),
        )
    )

    auc_rows: list[dict[str, float | int | str]] = []
    ordered_noise = noise.sort_values(["size", "network", "noise_level", "n_patterns"])
    for (size, network, noise_level), group in ordered_noise.groupby(["size", "network", "noise_level"]):
        auc_rows.append(
            {
                "size": int(size),
                "network": network,
                "noise_level": float(noise_level),
                "noise_auc": float(
                    np.trapz(group["accuracy"].to_numpy(), group["n_patterns"].to_numpy())
                    / group["n_patterns"].max()
                ),
            }
        )

    auc_frame = pd.DataFrame(auc_rows)
    return grouped.merge(auc_frame, on=["size", "network", "noise_level"], how="left")


def run_comparison() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, compare, and save the Hopfield vs Hebbian results."""

    capacity = load_capacity_results()
    noise = load_noise_results()
    summary = build_summary(capacity, noise)
    noise_by_level = build_noise_by_level_summary(noise)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    capacity.to_csv(OUTPUT_CAPACITY_CSV, index=False)
    noise.to_csv(OUTPUT_NOISE_CSV, index=False)
    summary.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    noise_by_level.to_csv(OUTPUT_NOISE_BY_LEVEL_CSV, index=False)

    capacity_figures = save_capacity_plots(capacity)
    noise_figures = save_noise_plots(noise)

    return capacity, noise, summary


def main() -> None:
    """Command-line entry point."""

    capacity, noise, summary = run_comparison()
    print(f"Saved: {OUTPUT_CAPACITY_CSV}")
    print(f"Saved: {OUTPUT_NOISE_CSV}")
    print(f"Saved: {OUTPUT_SUMMARY_CSV}")
    print(f"Saved: {OUTPUT_NOISE_BY_LEVEL_CSV}")
    print(capacity.head())
    print(noise.head())
    print(summary)


if __name__ == "__main__":
    main()