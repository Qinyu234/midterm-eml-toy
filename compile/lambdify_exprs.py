"""
Compile value and Wirtinger Jacobian (∂z, ∂z̄) to numpy callables.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import sympy as sp

from primitives.logsum_gap import (
    gap_correction_numpy,
    ln_t_sum_numpy,
    logsum_pair_numpy,
)
from primitives.ops import safe_ln_numpy, soft_exp_numpy
from symbolic.logsum_gap import LogsumT
from symbolic.pipeline import NodeSymbolicExprs, ParamGradient

_NUMPY_MODULES = {
    'exp': soft_exp_numpy,
    'log': safe_ln_numpy,
    'ln': safe_ln_numpy,
    'log1p': np.log1p,
    'expm1': np.expm1,
    'LogsumT': ln_t_sum_numpy,
    'logsum_pair': logsum_pair_numpy,
    'gap_correction': gap_correction_numpy,
}


@dataclass(frozen=True)
class CompiledWirtingerGrad:
    param: sp.Expr
    dz_fn: Callable
    dz_bar_fn: Callable


def _lambdify(syms, expr):
    return sp.lambdify(syms, expr, modules=[_NUMPY_MODULES, 'numpy'])


def compile_param_gradient(g: ParamGradient, syms) -> CompiledWirtingerGrad:
    return CompiledWirtingerGrad(
        param=g.param,
        dz_fn=_lambdify(syms, g.dz),
        dz_bar_fn=_lambdify(syms, g.dz_bar),
    )


def compile_node_exprs(
    exprs: NodeSymbolicExprs,
) -> Tuple[Callable, Tuple[CompiledWirtingerGrad, ...]]:
    syms = [g.param for g in exprs.param_grads]
    value_fn = _lambdify(syms, exprs.value_expr)
    grad_fns = tuple(compile_param_gradient(g, syms) for g in exprs.param_grads)
    return value_fn, grad_fns
