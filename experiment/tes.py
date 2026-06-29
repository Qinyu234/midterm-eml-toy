"""Tower Efficiency Score and related metrics."""
from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from experiment.metrics import rmse


def fit_quality(in_rmse: float) -> float:
    return 1.0 / (1.0 + in_rmse)


def per_step_cost(method: str, *, n_params: int, n_eml: int = 0, n_learnable: int = 0) -> float:
    if method == 'eml':
        return math.exp(n_eml) + math.log(1 + n_learnable)
    return float(1 + n_params)


def eml_complexity(n_eml: int, n_learnable: int) -> float:
    return math.exp(n_eml) + math.log(1 + n_learnable)


def steps_to_threshold(rmse_curve: List[float], eps: float) -> Optional[int]:
    for i, v in enumerate(rmse_curve):
        if np.isfinite(v) and v < eps:
            return i + 1
    return None


def total_compute(
    method: str,
    steps: int,
    *,
    n_params: int,
    n_eml: int = 0,
    n_learnable: int = 0,
    steps_to_threshold: Optional[int] = None,
    max_steps: int | None = None,
) -> float:
    used = steps_to_threshold if steps_to_threshold is not None else (max_steps or steps)
    return used * per_step_cost(
        method, n_params=n_params, n_eml=n_eml, n_learnable=n_learnable,
    )


def tower_efficiency_score(
    in_rmse: float,
    compute: float,
) -> float:
    if compute <= 0 or not np.isfinite(in_rmse):
        return 0.0
    return fit_quality(in_rmse) / compute


def evaluate_rmse(predict_fn, x: np.ndarray, y: np.ndarray) -> float:
    pred = np.asarray(predict_fn(x), dtype=np.float64)
    return rmse(y, pred)
