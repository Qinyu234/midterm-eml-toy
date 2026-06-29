"""Bind Feynman sample columns to leaf D slot arrays."""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from tree.coords import BindingMode, input_slot_indices


def bind_d_args(
    n_eml: int,
    x_slot: int,
    n_vars: int,
    X: np.ndarray,
    learnable: Sequence[float],
    *,
    binding: BindingMode = 'bottom_d',
) -> Tuple[np.ndarray, ...]:
    """Feynman cols → input slots; other slots ← learnable constants (broadcast)."""
    n_slots = n_eml + 1
    n = len(X) if np.ndim(X) >= 1 else 1
    d: List[np.ndarray] = [
        np.full(n, learnable[i], dtype=np.complex128)
        for i in range(n_slots)
    ]
    if X.ndim == 1:
        cols = [X]
    else:
        cols = [X[:, i] for i in range(X.shape[1])]
    slots = input_slot_indices(n_eml, x_slot, n_vars, binding=binding)
    for col_i, slot in enumerate(slots):
        d[slot] = np.asarray(cols[col_i], dtype=np.complex128)
    return tuple(d)
