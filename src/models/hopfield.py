"""Hopfield network for bipolar patterns {-1, +1}."""

from __future__ import annotations
from typing import Optional
import numpy as np


class HopfieldNetwork:
    """Discrete Hopfield memory with symmetric Hebbian weights."""

    def __init__(self, n_neurons: Optional[int] = None, random_state: Optional[int] = None) -> None:
        self.n_neurons = n_neurons
        self.weights: Optional[np.ndarray] = None
        self.rng = np.random.default_rng(random_state)
        self.last_iterations = 0
        self.converged = False

    @staticmethod
    def _sign(x: np.ndarray) -> np.ndarray:
        """Bipolar sign; zero activations become +1."""
        return np.where(x >= 0, 1, -1).astype(np.int8)

    @staticmethod
    def _validate(x: np.ndarray, name: str) -> None:
        """Validate bipolar {-1, +1} data."""
        if not np.all(np.isin(np.unique(x), [-1, 1])):
            raise ValueError(f"{name} must contain only -1 and +1.")

    def train(self, patterns: np.ndarray) -> None:
        """Train with symmetric Hebbian learning and zero weight diagonal."""
        p = np.asarray(patterns, dtype=np.int8)
        if p.ndim != 2 or len(p) == 0:
            raise ValueError("patterns must have shape (n_patterns, n_neurons).")
        self._validate(p, "patterns")
        if self.n_neurons is None:
            self.n_neurons = p.shape[1]
        if p.shape[1] != self.n_neurons:
            raise ValueError(f"Expected {self.n_neurons} neurons, got {p.shape[1]}.")
        self.weights = (p.T @ p).astype(float) / self.n_neurons
        np.fill_diagonal(self.weights, 0.0)

    def recall(self, pattern: np.ndarray, max_iter: int = 100) -> np.ndarray:
        """Iteratively recover a stable state from a bipolar input pattern."""
        if self.weights is None or self.n_neurons is None:
            raise RuntimeError("Call train() before recall().")
        state = np.asarray(pattern, dtype=np.int8).reshape(-1).copy()
        if state.size != self.n_neurons:
            raise ValueError(f"Expected {self.n_neurons} neurons, got {state.size}.")
        self._validate(state, "pattern")
        self.converged = False
        for i in range(1, max_iter + 1):
            new_state = self._sign(self.weights @ state)
            self.last_iterations = i
            if np.array_equal(new_state, state):
                self.converged = True
                break
            state = new_state
        return state.astype(np.int8)

    def predict(self, pattern: np.ndarray) -> np.ndarray:
        """Recover a pattern using recall()."""
        return self.recall(pattern)

    def energy(self, state: np.ndarray) -> float:
        """Compute Hopfield energy: -0.5 * s.T @ W @ s."""
        if self.weights is None:
            raise RuntimeError("Call train() before energy().")
        s = np.asarray(state, dtype=np.int8).reshape(-1)
        self._validate(s, "state")
        return float(-0.5 * s.T @ self.weights @ s)
