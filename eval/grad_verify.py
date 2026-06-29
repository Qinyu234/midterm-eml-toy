"""
Symbolic vs finite-difference gradient checks on leaf D.
符号梯度 vs 数值差分（叶槽 D）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from compile.lambdify_exprs import compile_node_exprs
from numerics.d_binding import bind_d_args
from symbolic.pipeline import compute_node_exprs
from tree.coords import learnable_slot_indices


@dataclass(frozen=True)
class GradVerifyResult:
    byte: int
    max_fd_error: float
    mean_dz_bar_norm: float
    finite_rate: float
    n_points: int

    @property
    def ok(self) -> bool:
        return (
            np.isfinite(self.max_fd_error)
            and self.max_fd_error < 0.05
            and self.finite_rate >= 0.9
        )


def value_wirtinger_grads(
    byte: int,
    d_args: Tuple[np.ndarray, ...],
    *,
    point: int = 0,
) -> Tuple[complex, List[complex], List[complex]]:
    """Value and ∂z/∂z̄ at one batch index."""
    exprs = compute_node_exprs(byte)
    value_fn, grad_fns = compile_node_exprs(exprs)
    scalars = tuple(
        np.asarray(a[point] if np.ndim(a) else a, dtype=np.complex128)
        for a in d_args
    )
    z = complex(value_fn(*scalars))
    dz = [complex(g.dz_fn(*scalars)) for g in grad_fns]
    dzb = [complex(g.dz_bar_fn(*scalars)) for g in grad_fns]
    return z, dz, dzb


def grad_fd_error_at(
    byte: int,
    d_args: Tuple[np.ndarray, ...],
    param_index: int,
    *,
    point: int = 0,
    eps: float = 1e-7,
) -> float:
    """|sym ∂z/∂D - FD| for one learnable D (real axis bump)."""
    exprs = compute_node_exprs(byte)
    value_fn, grad_fns = compile_node_exprs(exprs)
    scalars = list(
        float(np.asarray(a[point] if np.ndim(a) else a).real)
        if np.iscomplexobj(a) or np.asarray(a).dtype == np.complex128
        else float(np.asarray(a[point] if np.ndim(a) else a))
        for a in d_args
    )
    # rebuild as complex scalars
    scalars_c = [
        complex(np.asarray(a[point] if np.ndim(a) else a))
        for a in d_args
    ]
    sym_g = complex(grad_fns[param_index].dz_fn(*scalars_c))
    base = complex(value_fn(*scalars_c))
    bumped = list(scalars_c)
    bumped[param_index] = scalars_c[param_index] + eps
    fd = (complex(value_fn(*bumped)) - base) / eps
    return float(abs(sym_g - fd))


def verify_gradients(
    byte: int,
    *,
    n_vars: int = 1,
    n_points: int = 5,
    learnable: Sequence[float] | None = None,
    lo: float = 0.5,
    hi: float = 1.5,
) -> GradVerifyResult:
    from codec.node import decode_node

    node = decode_node(byte)
    learn_slots = learnable_slot_indices(node.n_eml, node.x_slot, n_vars)
    eps = 1e-7
    if learnable is None:
        learnable_vals = [0.3 + 0.1 * i for i in range(node.n_eml + 1)]
    else:
        learnable_vals = list(learnable)
    if len(learnable_vals) != node.n_eml + 1:
        raise ValueError('learnable must have n_eml+1 entries')

    xs = np.linspace(lo, hi, n_points)
    if n_vars == 1:
        X = xs
    else:
        X = np.stack([xs, xs + 0.2], axis=-1)

    d_args = bind_d_args(
        node.n_eml, node.x_slot, n_vars, X, learnable_vals,
    )
    exprs = compute_node_exprs(byte)
    value_fn, grad_fns = compile_node_exprs(exprs)

    finite = 0
    fd_errors: List[float] = []
    dz_bar_norms: List[float] = []

    for pt in range(n_points):
        scalars = tuple(
            complex(np.asarray(a[pt] if np.ndim(a) else a))
            for a in d_args
        )
        z = value_fn(*scalars)
        if np.isfinite(complex(z)):
            finite += 1
        for g in grad_fns:
            dzb = complex(g.dz_bar_fn(*scalars))
            dz_bar_norms.append(abs(dzb))
        for slot in learn_slots:
            fd_errors.append(
                grad_fd_error_at(byte, d_args, slot, point=pt, eps=eps)
            )

    return GradVerifyResult(
        byte=byte,
        max_fd_error=float(max(fd_errors)) if fd_errors else float('inf'),
        mean_dz_bar_norm=float(np.mean(dz_bar_norms)) if dz_bar_norms else 0.0,
        finite_rate=finite / max(n_points, 1),
        n_points=n_points,
    )
