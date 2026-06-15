"""Dataset generation utilities for letter patterns.

This module renders the alphabet A-Z with a TrueType font, converts each
letter into bipolar matrices in {-1, +1}, and stores both matrix and vector
representations for later comparison between Hopfield and Hebbian models.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from matplotlib import font_manager, pyplot as plt
from PIL import Image, ImageDraw, ImageFont

DEFAULT_ALPHABET: tuple[str, ...] = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DEFAULT_SIZES: tuple[int, ...] = (7, 10, 15)


@dataclass(frozen=True)
class LetterPattern:
    """Single letter pattern in matrix and vector form."""

    letter: str
    size: int
    matrix: np.ndarray
    vector: np.ndarray


@dataclass(frozen=True)
class AlphabetDataset:
    """Dataset bundle for one grid size."""

    size: int
    letters: tuple[str, ...]
    matrices: np.ndarray
    vectors: np.ndarray


def _resolve_font_path(font_path: str | Path | None = None) -> str:
    """Resolve a TrueType font path.

    The function prefers an explicit path when provided and otherwise uses a
    well-supported DejaVu Sans font shipped with Matplotlib.
    """

    if font_path is not None:
        candidate = Path(font_path)
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"Font file not found: {candidate}")

    preferred_fonts = [
        font_manager.FontProperties(family="DejaVu Sans", weight="bold"),
        font_manager.FontProperties(family="DejaVu Sans"),
    ]

    for properties in preferred_fonts:
        resolved = Path(font_manager.findfont(properties, fallback_to_default=True))
        if resolved.is_file() and resolved.suffix.lower() in {".ttf", ".otf", ".ttc"}:
            return str(resolved)

    raise RuntimeError("No TrueType font could be resolved for dataset generation.")


def _render_letter(letter: str, size: int, font_path: str | Path | None = None) -> Image.Image:
    """Render a letter on a high-resolution grayscale canvas.

    The canvas is intentionally larger than the final target grid so that the
    downsampling stage preserves the shape of the glyph.
    """

    if len(letter) != 1 or not letter.isalpha():
        raise ValueError("letter must be a single alphabetical character")

    canvas_size = max(size * 16, 128)
    image = Image.new("L", (canvas_size, canvas_size), color=255)
    drawer = ImageDraw.Draw(image)
    resolved_font_path = _resolve_font_path(font_path)
    font = ImageFont.truetype(resolved_font_path, size=int(canvas_size * 0.78))

    bbox = drawer.textbbox((0, 0), letter.upper(), font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (canvas_size - text_width) / 2 - bbox[0]
    y = (canvas_size - text_height) / 2 - bbox[1]
    drawer.text((x, y), letter.upper(), fill=0, font=font)
    return image


def _image_to_bipolar_matrix(image: Image.Image, size: int, threshold: int = 128) -> np.ndarray:
    """Convert a grayscale PIL image into a bipolar matrix."""

    resized = image.resize((size, size), Image.Resampling.LANCZOS)
    pixels = np.asarray(resized, dtype=np.uint8)
    matrix = np.where(pixels < threshold, 1, -1).astype(np.int8)
    return matrix


def generate_letter(
    letter: str,
    size: int,
    font_path: str | Path | None = None,
    threshold: int = 128,
) -> LetterPattern:
    """Generate a bipolar pattern for a single letter.

    Parameters
    ----------
    letter:
        Character to render.
    size:
        Final grid size, for example 7, 10 or 15.
    font_path:
        Optional path to a TrueType font file.
    threshold:
        Grayscale threshold used to binarize the image.
    """

    rendered = _render_letter(letter, size=size, font_path=font_path)
    matrix = _image_to_bipolar_matrix(rendered, size=size, threshold=threshold)
    vector = matrix.reshape(-1)
    return LetterPattern(letter=letter.upper(), size=size, matrix=matrix, vector=vector)


def generate_alphabet(
    sizes: Sequence[int] = DEFAULT_SIZES,
    letters: Sequence[str] = DEFAULT_ALPHABET,
    font_path: str | Path | None = None,
    threshold: int = 128,
) -> dict[int, AlphabetDataset]:
    """Generate the complete alphabet for one or more grid sizes."""

    dataset: dict[int, AlphabetDataset] = {}
    normalized_letters = tuple(letter.upper() for letter in letters)

    for size in sizes:
        matrices = []
        vectors = []
        for letter in normalized_letters:
            pattern = generate_letter(letter=letter, size=size, font_path=font_path, threshold=threshold)
            matrices.append(pattern.matrix)
            vectors.append(pattern.vector)

        dataset[size] = AlphabetDataset(
            size=size,
            letters=normalized_letters,
            matrices=np.stack(matrices, axis=0),
            vectors=np.stack(vectors, axis=0),
        )

    return dataset


def save_dataset(dataset: dict[int, AlphabetDataset] | AlphabetDataset, path: str | Path) -> Path:
    """Save one or many datasets to a compressed `.npz` file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(dataset, AlphabetDataset):
        datasets = {dataset.size: dataset}
    else:
        datasets = dict(dataset)

    payload: dict[str, object] = {
        "sizes": np.array(sorted(datasets.keys()), dtype=np.int16),
    }

    reference_letters: tuple[str, ...] | None = None
    for size, bundle in datasets.items():
        payload[f"size_{size}_letters"] = np.array(bundle.letters, dtype="U1")
        payload[f"size_{size}_matrices"] = bundle.matrices
        payload[f"size_{size}_vectors"] = bundle.vectors
        if reference_letters is None:
            reference_letters = bundle.letters

    if reference_letters is not None:
        payload["alphabet"] = np.array(reference_letters, dtype="U1")

    np.savez_compressed(output_path, **payload)
    return output_path


def load_dataset(path: str | Path) -> dict[int, AlphabetDataset]:
    """Load datasets previously saved with :func:`save_dataset`."""

    input_path = Path(path)
    with np.load(input_path, allow_pickle=False) as archive:
        sizes = archive["sizes"].tolist()
        loaded: dict[int, AlphabetDataset] = {}

        for size in sizes:
            size_int = int(size)
            letters = tuple(str(value) for value in archive[f"size_{size_int}_letters"].tolist())
            matrices = archive[f"size_{size_int}_matrices"]
            vectors = archive[f"size_{size_int}_vectors"]
            loaded[size_int] = AlphabetDataset(
                size=size_int,
                letters=letters,
                matrices=matrices,
                vectors=vectors,
            )

    return loaded


def visualize_pattern(
    pattern: LetterPattern | np.ndarray,
    ax: plt.Axes | None = None,
    title: str | None = None,
    cmap: str = "gray_r",
) -> plt.Axes:
    """Visualize a letter pattern in matrix or vector form."""

    if isinstance(pattern, LetterPattern):
        matrix = pattern.matrix
        default_title = f"{pattern.letter} ({pattern.size}x{pattern.size})"
    else:
        array = np.asarray(pattern)
        if array.ndim == 1:
            side = int(np.sqrt(array.size))
            if side * side != array.size:
                raise ValueError("1D patterns must have a square length.")
            matrix = array.reshape(side, side)
        elif array.ndim == 2:
            matrix = array
        else:
            raise ValueError("pattern must be a 1D vector or 2D matrix.")
        default_title = f"Pattern {matrix.shape[0]}x{matrix.shape[1]}"

    if ax is None:
        _, ax = plt.subplots(figsize=(3, 3))

    ax.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title or default_title)
    return ax
