"""
Log-sum gap threshold from floating precision, and T-domain ln-sum stabilization.

ln(T(e,n,a)+T(e,n,b)) = logsum_pair(T(e,n-1,a), T(e,n-1,b))  [T(e,0,x)=x]
logsum_pair(u,v) = max(u,v) + gap(|u-v|)
gap(d) = 0           if d > threshold
       = ln(1+exp(-d)) if d <= threshold

threshold: exp(-G) < epsilon  =>  G = ceil(-ln(epsilon)); float64 epsilon=1e-16 => G=37, use 40.
"""
from __future__ import annotations

import math
from typing import Union

import numpy as np
import sympy as sp

from symbolic.tetration import E_TAG, Tetration

DEFAULT_EPSILON = 1e-16
DEFAULT_GAP_THRESHOLD = 40.0


def gap_threshold_from_epsilon(epsilon: float = DEFAULT_EPSILON) -> float:
    """|a-b| above this makes ln(1+exp(-|a-b|)) negligible at given epsilon."""
    if epsilon <= 0:
        raise ValueError('epsilon must be positive')
    return float(math.ceil(-math.log(epsilon)))


def gap_threshold_for_dtype(dtype: Union[np.dtype, type] = np.float64) -> float:
    eps = float(np.finfo(dtype).eps)
    return max(gap_threshold_from_epsilon(eps), gap_threshold_from_epsilon(DEFAULT_EPSILON))


def gap_correction_symbolic(
    d: sp.Expr,
    threshold: float = DEFAULT_GAP_THRESHOLD,
) -> sp.Expr:
    d_abs = sp.Abs(d)
    return sp.Piecewise(
        (sp.S.Zero, d_abs > threshold),
        (sp.log(1 + sp.exp(-d_abs, evaluate=False), evaluate=False), True),
    )


def logsum_pair_symbolic(
    a: sp.Expr,
    b: sp.Expr,
    threshold: float = DEFAULT_GAP_THRESHOLD,
) -> sp.Expr:
    """ln(exp(a)+exp(b)) stable; a,b in exponent chart."""
    m = sp.Max(a, b)
    return m + gap_correction_symbolic(a - b, threshold)


def t_preimage_symbolic(n: int, x: sp.Expr) -> sp.Expr:
    """Inner argument to outermost exp in T(e,n,x); T(e,0,x)=x."""
    if n <= 1:
        return x
    return Tetration(E_TAG, n - 1, x)


class LogsumT(sp.Function):
    """ln(T(e,n,a)+T(e,n,b)) with gap stabilization; n positive integer."""


def ln_t_sum_symbolic(n: int, a: sp.Expr, b: sp.Expr, threshold: float = DEFAULT_GAP_THRESHOLD) -> sp.Expr:
    if n < 1:
        raise ValueError(f'n must be >= 1, got {n}')
    ua = t_preimage_symbolic(n, a)
    ub = t_preimage_symbolic(n, b)
    return logsum_pair_symbolic(ua, ub, threshold)


def merge_t_exp_add(expr: sp.Expr) -> sp.Expr:
    """T(e,1,a+b) = T(e,1,a)*T(e,1,b) for binary Add."""
    if isinstance(expr, Tetration):
        e, n, z = expr.args
        z = merge_t_exp_add(z)
        try:
            n_i = int(n)
        except (TypeError, ValueError):
            return Tetration(e, n, z)
        if n_i == 1 and isinstance(z, sp.Add) and len(z.args) == 2:
            return sp.Mul(
                *[Tetration(e, 1, merge_t_exp_add(t)) for t in z.args],
                evaluate=False,
            )
        return Tetration(e, n, z)
    if not expr.args:
        return expr
    return expr.func(*[merge_t_exp_add(a) for a in expr.args])


def _try_stabilize_ln_t_sum(t: Tetration) -> sp.Expr | None:
    e, n_outer, z = t.args
    try:
        if int(n_outer) != -1:
            return None
    except (TypeError, ValueError):
        return None
    if not isinstance(z, sp.Add) or len(z.args) != 2:
        return None
    n_inner: int | None = None
    inner_args: list[sp.Expr] = []
    for term in z.args:
        if not isinstance(term, Tetration) or term.args[0] != e:
            return None
        try:
            ni = int(term.args[1])
        except (TypeError, ValueError):
            return None
        if n_inner is None:
            n_inner = ni
        elif n_inner != ni:
            return None
        inner_args.append(term.args[2])
    if n_inner is None or n_inner < 1:
        return None
    return LogsumT(n_inner, inner_args[0], inner_args[1])


def stabilize_ln_t_sum(expr: sp.Expr) -> sp.Expr:
    """T(e,-1,T(e,n,a)+T(e,n,b)) → LogsumT(n,a,b)."""
    if isinstance(expr, Tetration):
        z = stabilize_ln_t_sum(expr.args[2])
        rebuilt = Tetration(expr.args[0], expr.args[1], z)
        hit = _try_stabilize_ln_t_sum(rebuilt)
        if hit is not None:
            return hit
        return rebuilt
    if not expr.args:
        return expr
    return expr.func(*[stabilize_ln_t_sum(a) for a in expr.args])
