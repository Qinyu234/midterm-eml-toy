"""Two-phase training: Search (+ Harden placeholder). / 两阶段训练。"""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import torch

from baselines.base import FitResult
from codec.node import decode_node, is_evaluable
from data.feynman import FeynmanFunction, apply_function, grid_sample
from data.targets import BenchmarkTarget
from experiment.metrics import SearchRun, rmse
from experiment.tes import (
    eml_complexity,
    steps_to_threshold,
    total_compute,
    tower_efficiency_score,
)
from experiment.train_loop import TrainStopConfig, lr_decay, training_should_stop
from export.workplace import iter_4bit_node_bytes
from numerics.symbolic_model import CompiledEMLModel


def _train_eml_curve(
    model: CompiledEMLModel,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_steps: int,
    lr: float,
    loss_mode: str = 'complex',
    stop: TrainStopConfig | None = None,
) -> List[float]:
    cfg = stop or TrainStopConfig(max_steps=max_steps)
    curve: List[float] = []
    step = 0
    while step < cfg.max_steps:
        step_lr = lr_decay(lr, step)
        loss, grads_d, ga, gb = model.loss_and_symbolic_grads(
            x_train, y_train, loss_mode=loss_mode,
        )
        if not np.isfinite(loss):
            curve.append(float('inf'))
            break
        model.apply_symbolic_grads(grads_d, ga, gb, step_lr)
        with torch.no_grad():
            pred = model(torch.tensor(x_train, dtype=torch.float64)).numpy()
        curve.append(rmse(y_train, pred))
        step += 1
        if training_should_stop(curve, cfg):
            break
    return curve


def fit_eml(
    target: BenchmarkTarget,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_ood: np.ndarray,
    y_ood: np.ndarray,
    *,
    byte: int,
    max_steps: int = 500,
    lr: float = 0.01,
    eps: float = 1e-2,
    binding: str = 'bottom_d',
    loss_mode: str = 'complex',
    stop: TrainStopConfig | None = None,
) -> FitResult:
    model = CompiledEMLModel(byte, n_vars=1, binding=binding)  # type: ignore[arg-type]
    cfg = stop or TrainStopConfig(max_steps=max_steps)
    curve = _train_eml_curve(
        model, x_train, y_train,
        max_steps=cfg.max_steps, lr=lr, loss_mode=loss_mode, stop=cfg,
    )

    with torch.no_grad():
        x_t = torch.tensor(x_train, dtype=torch.float64)
        x_o = torch.tensor(x_ood, dtype=torch.float64)
        in_rmse = rmse(y_train, model(x_t).numpy())
        ood_rmse = rmse(y_ood, model(x_o).numpy())

    node = decode_node(byte)
    n_learnable = len(model.d_params)
    n_params = n_learnable + 2  # alpha, beta
    complexity = eml_complexity(node.n_eml, n_learnable)
    st = steps_to_threshold(curve, eps)
    compute = total_compute(
        'eml', len(curve),
        n_params=n_params,
        n_eml=node.n_eml,
        n_learnable=n_learnable,
        steps_to_threshold=st,
        max_steps=cfg.max_steps,
    )
    return FitResult(
        method='eml',
        target_id=target.target_id,
        in_rmse=in_rmse,
        ood_rmse=ood_rmse,
        n_params=n_params,
        complexity=complexity,
        steps_to_threshold=st,
        total_steps=len(curve),
        compute=compute,
        tes=tower_efficiency_score(in_rmse, compute),
        meta={
            'byte': byte,
            'n_eml': node.n_eml,
            'n_learnable': n_learnable,
            'x_slot': node.x_slot,
            'binding': binding,
            'loss_mode': loss_mode,
        },
        rmse_curve=curve,
    )


def scan_eml(
    target: BenchmarkTarget,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_ood: np.ndarray,
    y_ood: np.ndarray,
    *,
    max_steps: int = 500,
    lr: float = 0.01,
    eps: float = 1e-2,
    bytes: Optional[Iterable[int]] = None,
) -> FitResult:
    pool = list(bytes) if bytes is not None else list(iter_4bit_node_bytes())
    pool = [b for b in pool if is_evaluable(b)]

    best_rmse: Optional[FitResult] = None
    passing: List[FitResult] = []

    for byte in pool:
        r = fit_eml(
            target, x_train, y_train, x_ood, y_ood,
            byte=byte, max_steps=max_steps, lr=lr, eps=eps,
        )
        if not np.isfinite(r.in_rmse):
            continue
        if best_rmse is None or r.in_rmse < best_rmse.in_rmse:
            best_rmse = r
        if r.steps_to_threshold is not None:
            passing.append(r)

    if passing:
        passing.sort(key=lambda r: (r.meta['n_eml'], r.meta['n_learnable'], r.in_rmse))
        return passing[0]
    if best_rmse is not None:
        return best_rmse
    return FitResult(
        method='eml',
        target_id=target.target_id,
        in_rmse=float('inf'),
        ood_rmse=float('inf'),
        n_params=0,
        complexity=float('inf'),
        steps_to_threshold=None,
        total_steps=0,
        compute=float('inf'),
        tes=0.0,
        meta={},
    )


def search_run(
    byte: int,
    fn: FeynmanFunction,
    *,
    steps: int = 50,
    lr: float = 0.01,
    n_grid: int = 7,
    recovery_threshold: float = 1e-3,
) -> SearchRun:
    X = grid_sample(fn.n_vars, n_per_dim=n_grid)
    y = apply_function(fn, X)

    model = CompiledEMLModel(byte, n_vars=fn.n_vars)
    run = SearchRun(byte=byte, eq_id=fn.eq_id)

    for step in range(steps):
        loss, grads_d, ga, gb = model.loss_and_symbolic_grads(X, y)
        if not np.isfinite(loss):
            run.final_rmse = float('inf')
            return run
        model.apply_symbolic_grads(grads_d, ga, gb, lr)
        run.add_record(step, loss)

    with torch.no_grad():
        x_t = torch.tensor(X, dtype=torch.float64)
        pred = model(x_t).numpy()
    run.final_rmse = rmse(y, pred)
    run.recovered = run.final_rmse < recovery_threshold
    return run
