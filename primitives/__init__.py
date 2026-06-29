"""Numeric primitives for lambdify (§1.5–1.6). / 数值原语。"""
from .ops import (
    LN_EPS,
    eml_numpy,
    eml_torch,
    safe_ln_numpy,
    safe_ln_torch,
    soft_exp_numpy,
    soft_exp_torch,
)

__all__ = [
    'LN_EPS', 'eml_numpy', 'eml_torch',
    'safe_ln_numpy', 'safe_ln_torch', 'soft_exp_numpy', 'soft_exp_torch',
]
