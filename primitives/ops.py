"""
Numeric operators: safe_ln / soft_exp (numpy + torch).
数值原语：safe_ln(log1p) 与 soft_exp。
"""
from __future__ import annotations

import math

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

LN_EPS = 1e-8
C_PLUS = 20.0
C_MINUS = 20.0
L_PLUS = 1e300
L_MINUS = 1e-300


def _ln_arg(w: np.ndarray) -> np.ndarray:
    """log1p argument: w-1+εi ≡ (w+εi)-1 / log1p 自变量。"""
    w = np.asarray(w, dtype=np.complex128)
    return w - 1.0 + 1j * LN_EPS


def safe_ln_numpy(w: np.ndarray) -> np.ndarray:
    return np.log1p(_ln_arg(w))


def _soft_exp_real(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128)
    re = np.real(x)
    im = np.imag(x)
    out_re = np.empty_like(re)
    mask_lo = re < -C_MINUS
    mask_hi = re > C_PLUS
    mask_mid = ~(mask_lo | mask_hi)

    em = math.exp(-C_MINUS)
    km = em / (em - L_MINUS)
    out_re[mask_lo] = L_MINUS + (em - L_MINUS) * np.exp(km * (re[mask_lo] + C_MINUS))

    ep = math.exp(C_PLUS)
    kp = ep / (L_PLUS - ep)
    out_re[mask_hi] = L_PLUS - (L_PLUS - ep) * np.exp(-kp * (re[mask_hi] - C_PLUS))

    out_re[mask_mid] = np.exp(re[mask_mid])

    scale = np.ones_like(re)
    nz = np.abs(re) > 1e-30
    scale[nz] = out_re[nz] / re[nz]
    return out_re + 1j * im * scale


def soft_exp_numpy(x: np.ndarray) -> np.ndarray:
    return _soft_exp_real(np.asarray(x, dtype=np.complex128))


def eml_numpy(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    return soft_exp_numpy(z) - safe_ln_numpy(w)


if torch is not None:

    def safe_ln_torch(w: "torch.Tensor") -> "torch.Tensor":
        w = w.to(torch.complex128)
        return torch.log1p(w - 1.0 + 1j * LN_EPS)

    def soft_exp_torch(x: "torch.Tensor") -> "torch.Tensor":
        re = x.real
        im = x.imag
        em = math.exp(-C_MINUS)
        km = em / (em - L_MINUS)
        ep = math.exp(C_PLUS)
        kp = ep / (L_PLUS - ep)

        lo = L_MINUS + (em - L_MINUS) * torch.exp(km * (re + C_MINUS))
        hi = L_PLUS - (L_PLUS - ep) * torch.exp(-kp * (re - C_PLUS))
        mid = torch.exp(re)

        out_re = torch.where(
            re < -C_MINUS, lo, torch.where(re > C_PLUS, hi, mid)
        )
        scale = torch.where(re.abs() > 1e-30, out_re / re.clamp_min(1e-30), torch.ones_like(re))
        return out_re + 1j * im * scale

    def eml_torch(z: "torch.Tensor", w: "torch.Tensor") -> "torch.Tensor":
        return soft_exp_torch(z) - safe_ln_torch(w)

else:  # pragma: no cover
    safe_ln_torch = None  # type: ignore
    soft_exp_torch = None  # type: ignore
    eml_torch = None  # type: ignore
