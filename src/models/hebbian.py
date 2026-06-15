"""Autoassociative Hebbian memory for bipolar patterns.

The implementation follows the classical Hebbian rule for stored bipolar
patterns in {-1, +1}. Retrieval is direct: a single matrix-vector product
followed by a bipolar sign threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


def _as_pattern_array(pattern: np.ndarray | Sequence[int]) -> np.ndarray:
    """Convert an input pattern to a one-dimensional NumPy array."""

    array = np.asarray(pattern)
    if array.ndim == 0:
        raise ValueError("pattern must contain at least one element")
    return array.reshape(-1)


def _stack_patterns(patterns: np.ndarray | Sequence[np.ndarray | Sequence[int]]) -> np.ndarray:
    """Normalize a collection of bipolar patterns to a 2D array.

    The returned array has shape ``(n_patterns, n_neurons)``.
    """

    array = np.asarray(patterns)

    if array.ndim == 1:
        return array.reshape(1, -1)

    if array.ndim == 2:
        return array.reshape(array.shape[0], -1)

    if array.ndim >= 3:
        return array.reshape(array.shape[0], -1)

    raise ValueError("patterns must be a 1D, 2D, or 3D array-like object")


def _validate_bipolar(patterns: np.ndarray) -> None:
    """Ensure that the array contains only bipolar values."""

    unique_values = np.unique(patterns)
    if not np.all(np.isin(unique_values, (-1, 1))):
        raise ValueError("patterns must contain only values in {-1, +1}")


def _bipolar_sign(values: np.ndarray) -> np.ndarray:
    """Map values to {-1, +1} using the bipolar sign convention."""

    return np.where(values >= 0, 1, -1).astype(np.int8, copy=False)


@dataclass
class HebbianNetwork:
    """Autoassociative Hebbian memory for bipolar patterns.

    Attributes
    ----------
    weights:
        Symmetric weight matrix learned from stored patterns.
    n_neurons:
        Number of neurons in the pattern representation.
    n_patterns:
        Number of patterns used during training.
    """

    weights: np.ndarray | None = field(default=None, init=False)
    n_neurons: int | None = field(default=None, init=False)
    n_patterns: int = field(default=0, init=False)

    def train(self, patterns: np.ndarray | Sequence[np.ndarray | Sequence[int]]) -> "HebbianNetwork":
        """Train the network with bipolar patterns using the Hebbian rule.

        Parameters
        ----------
        patterns:
            Collection of stored patterns. Each pattern must use values in
            ``{-1, +1}`` and may be provided as a vector or as a matrix.
        """

        stacked_patterns = _stack_patterns(patterns).astype(np.int8, copy=False)
        _validate_bipolar(stacked_patterns)

        n_patterns, n_neurons = stacked_patterns.shape
        weights = stacked_patterns.T @ stacked_patterns
        weights = weights.astype(np.float64, copy=False) / float(n_neurons)
        np.fill_diagonal(weights, 0.0)

        self.weights = weights
        self.n_neurons = n_neurons
        self.n_patterns = n_patterns
        return self

    def recall(self, pattern: np.ndarray | Sequence[int]) -> np.ndarray:
        """Recall a stored pattern from a cue in one direct step."""

        if self.weights is None or self.n_neurons is None:
            raise RuntimeError("The network must be trained before calling recall().")

        cue = _as_pattern_array(pattern).astype(np.int8, copy=False)
        if cue.size != self.n_neurons:
            raise ValueError(
                f"pattern has {cue.size} neurons, expected {self.n_neurons}"
            )
        _validate_bipolar(cue)

        response = self.weights @ cue.astype(np.float64, copy=False)
        return _bipolar_sign(response).reshape(np.asarray(pattern).shape)

    def predict(self, pattern: np.ndarray | Sequence[int]) -> np.ndarray:
        """Alias for :meth:`recall` for compatibility with experiment code."""

        return self.recall(pattern)
