"""
Feynman function subset (n_vars ∈ {1,2}) and sampling.
Feynman 函数子集与采样。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import sympy as sp


@dataclass(frozen=True)
class FeynmanFunction:
    eq_id: str
    n_vars: int
    formula_sympy: sp.Expr
    var_names: Tuple[str, ...]

    def numpy_fn(self) -> Callable:
        syms = [sp.Symbol(v) for v in self.var_names]
        fn = sp.lambdify(syms, self.formula_sympy, modules=['numpy'])
        return fn


def _fn(eq_id: str, n_vars: int, expr_str: str, names: Tuple[str, ...]) -> FeynmanFunction:
    expr = sp.sympify(expr_str)
    return FeynmanFunction(eq_id, n_vars, expr, names)


FEYNMAN_FUNCTIONS: Tuple[FeynmanFunction, ...] = (
    _fn('I.6.2', 1, 'x0**3', ('x0',)),
    _fn('I.8.4', 1, 'x0**2/2', ('x0',)),
    _fn('I.12.1', 2, 'x0*x1', ('x0', 'x1')),
    _fn('I.12.5', 2, 'x0*x1', ('x0', 'x1')),
    _fn('I.14.3', 2, 'x0*x1/2', ('x0', 'x1')),
    _fn('I.15.3x', 2, 'x0/(1-x1)', ('x0', 'x1')),
    _fn('I.26.2', 1, 'sin(x0)**2', ('x0',)),
    _fn('I.34.1', 2, 'x0*x1', ('x0', 'x1')),
)


def list_feynman_functions(n_vars: int | None = None) -> List[FeynmanFunction]:
    if n_vars is None:
        return list(FEYNMAN_FUNCTIONS)
    return [f for f in FEYNMAN_FUNCTIONS if f.n_vars == n_vars]


def get_feynman_function(eq_id: str) -> FeynmanFunction:
    for f in FEYNMAN_FUNCTIONS:
        if f.eq_id == eq_id:
            return f
    raise KeyError(eq_id)


def grid_sample(n_vars: int, n_per_dim: int = 7, lo: float = -3.0, hi: float = 3.0) -> np.ndarray:
    """Grid on [-3,3]^n, shape (N, n_vars) / [-3,3]^n 网格采样。"""
    axes = [np.linspace(lo, hi, n_per_dim) for _ in range(n_vars)]
    grids = np.meshgrid(*axes, indexing='ij')
    return np.stack([g.ravel() for g in grids], axis=-1)


def random_sample(n_vars: int, n_points: int, lo: float = -3.0, hi: float = 3.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(lo, hi, size=(n_points, n_vars))


def apply_function(fn: FeynmanFunction, X: np.ndarray) -> np.ndarray:
    f = fn.numpy_fn()
    if fn.n_vars == 1:
        return np.asarray(f(X[:, 0]), dtype=np.float64)
    cols = [X[:, i] for i in range(fn.n_vars)]
    return np.asarray(f(*cols), dtype=np.float64)


@dataclass
class DataBatch:
    X: np.ndarray
    y: np.ndarray
    function: FeynmanFunction
