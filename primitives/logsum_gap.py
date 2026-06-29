"""Numeric log-sum gap for ln(T(e,n,a)+T(e,n,b))."""
from __future__ import annotations

import numpy as np

from primitives.ops import soft_exp_numpy

DEFAULT_GAP_THRESHOLD = 40.0

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def gap_correction_numpy(
    d: np.ndarray,
    threshold: float = DEFAULT_GAP_THRESHOLD,
) -> np.ndarray:
    d = np.asarray(d, dtype=np.complex128)
    ad = np.abs(d)
    out = np.zeros_like(ad, dtype=np.complex128)
    small = ad <= threshold
    if np.any(small):
        out[small] = np.log1p(np.exp(-ad[small]))
    return out


def logsum_pair_numpy(
    u: np.ndarray,
    v: np.ndarray,
    threshold: float = DEFAULT_GAP_THRESHOLD,
) -> np.ndarray:
    u = np.asarray(u, dtype=np.complex128)
    v = np.asarray(v, dtype=np.complex128)
    ru, rv = np.real(u), np.real(v)
    m = np.where(ru >= rv, u, v)
    return m + gap_correction_numpy(u - v, threshold)


def t_forward_numpy(n: int, x: np.ndarray) -> np.ndarray:
    """T(e,n,x) value for n>=0; n=0 is identity."""
    out = np.asarray(x, dtype=np.complex128)
    for _ in range(n):
        out = soft_exp_numpy(out)
    return out


def t_preimage_numpy(n: int, x: np.ndarray) -> np.ndarray:
    if n <= 1:
        return np.asarray(x, dtype=np.complex128)
    return t_forward_numpy(n - 1, x)


def ln_t_sum_numpy(
    n: int,
    a: np.ndarray,
    b: np.ndarray,
    threshold: float = DEFAULT_GAP_THRESHOLD,
) -> np.ndarray:
    """ln(T(e,n,a)+T(e,n,b)) with gap stabilization."""
    if n < 1:
        raise ValueError(f'n must be >= 1, got {n}')
    ua = t_preimage_numpy(n, a)
    ub = t_preimage_numpy(n, b)
    return logsum_pair_numpy(ua, ub, threshold)


if torch is not None:

    def gap_correction_torch(
        d: "torch.Tensor",
        threshold: float = DEFAULT_GAP_THRESHOLD,
    ) -> "torch.Tensor":
        d = d.to(torch.complex128)
        ad = d.abs()
        out = torch.zeros_like(ad, dtype=torch.complex128)
        small = ad <= threshold
        if small.any():
            out[small] = torch.log1p(torch.exp(-ad[small]))
        return out

    def logsum_pair_torch(
        u: "torch.Tensor",
        v: "torch.Tensor",
        threshold: float = DEFAULT_GAP_THRESHOLD,
    ) -> "torch.Tensor":
        u = u.to(torch.complex128)
        v = v.to(torch.complex128)
        m = torch.where(u.real >= v.real, u, v)
        return m + gap_correction_torch(u - v, threshold)

    def t_forward_torch(n: int, x: "torch.Tensor") -> "torch.Tensor":
        from primitives.ops import soft_exp_torch

        out = x.to(torch.complex128)
        for _ in range(n):
            out = soft_exp_torch(out)
        return out

    def t_preimage_torch(n: int, x: "torch.Tensor") -> "torch.Tensor":
        if n <= 1:
            return x.to(torch.complex128)
        return t_forward_torch(n - 1, x)

    def ln_t_sum_torch(
        n: int,
        a: "torch.Tensor",
        b: "torch.Tensor",
        threshold: float = DEFAULT_GAP_THRESHOLD,
    ) -> "torch.Tensor":
        ua = t_preimage_torch(n, a)
        ub = t_preimage_torch(n, b)
        return logsum_pair_torch(ua, ub, threshold)

else:  # pragma: no cover
    gap_correction_torch = None  # type: ignore
    logsum_pair_torch = None  # type: ignore
    ln_t_sum_torch = None  # type: ignore
