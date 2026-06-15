"""Hopfield network implementation for bipolar pattern recovery.

The network stores patterns encoded with values {-1, +1}. Training uses the
symmetric Hebbian learning rule and removes self-connections by setting the
weight diagonal to zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RecallResult