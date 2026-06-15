"""Hopfield network implementation for bipolar pattern recovery.

The network stores patterns encoded with values {-1, +1}. Training uses the
symmetric Hebbian learning rule and removes self-connections by setting the
weight diagonal to zero.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class HopfieldNetwork:
    """Discrete Hopfield network for autoassociative memory.

    Parameters
    ----------
    n_neurons : int or None, optional
        Number of neurons in the network. If None, it is inferred during train().
    random_state : int or None, optional
        Seed used by asynchronous updates.

    Notes
    -----
    Patterns