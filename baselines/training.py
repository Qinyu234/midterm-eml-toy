"""Train PyTorch regressors until convergence or oscillation."""
from __future__ import annotations

from typing import Callable, List, Literal, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import optimize

from experiment.metrics import rmse
from experiment.train_loop import TrainStopConfig, lr_decay, training_should_stop

OptimizerName = Literal['adamw', 'bfgs']


def _params_to_vector(model: nn.Module) -> np.ndarray:
    parts = [p.detach().cpu().numpy().ravel() for p in model.parameters()]
    return np.concatenate(parts) if parts else np.array([], dtype=np.float64)


def _vector_to_params(model: nn.Module, vec: np.ndarray) -> None:
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            chunk = vec[offset:offset + n].reshape(p.shape)
            p.copy_(torch.tensor(chunk, dtype=p.dtype, device=p.device))
            offset += n


def _params_grad_vector(model: nn.Module) -> np.ndarray:
    parts = []
    for p in model.parameters():
        if p.grad is None:
            parts.append(np.zeros(p.numel(), dtype=np.float64))
        else:
            parts.append(p.grad.detach().cpu().numpy().ravel())
    return np.concatenate(parts) if parts else np.array([], dtype=np.float64)


def train_regressor(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    optimizer: OptimizerName = 'adamw',
    lr: float = 0.01,
    stop: TrainStopConfig | None = None,
) -> List[float]:
    cfg = stop or TrainStopConfig()
    if optimizer == 'bfgs':
        return _train_bfgs(model, x_train, y_train, stop=cfg)
    return _train_adamw(model, x_train, y_train, lr=lr, stop=cfg)


def _train_adamw(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    lr: float,
    stop: TrainStopConfig,
) -> List[float]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    x_t = torch.tensor(x_train, dtype=torch.float64)
    y_t = torch.tensor(y_train, dtype=torch.float64)
    curve: List[float] = []
    step = 0
    while step < stop.max_steps:
        step_lr = lr_decay(lr, step)
        for g in opt.param_groups:
            g['lr'] = step_lr
        opt.zero_grad()
        pred = model(x_t)
        loss = torch.mean((pred - y_t) ** 2)
        if not torch.isfinite(loss):
            curve.append(float('inf'))
            break
        loss.backward()
        opt.step()
        with torch.no_grad():
            curve.append(rmse(y_train, model(x_t).numpy()))
        step += 1
        if training_should_stop(curve, stop):
            break
    return curve


def _train_bfgs(
    model: nn.Module,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    stop: TrainStopConfig,
) -> List[float]:
    x_t = torch.tensor(x_train, dtype=torch.float64)
    y_t = torch.tensor(y_train, dtype=torch.float64)
    curve: List[float] = []
    # Cap BFGS iterations; each scipy step is expensive even with analytic grad.
    maxiter = min(stop.max_steps, 150)

    def objective(vec: np.ndarray) -> float:
        _vector_to_params(model, vec)
        model.zero_grad()
        pred = model(x_t)
        if not torch.isfinite(pred).all():
            curve.append(float('inf'))
            return float('inf')
        loss = torch.mean((pred - y_t) ** 2)
        if not torch.isfinite(loss):
            curve.append(float('inf'))
            return float('inf')
        loss.backward()
        with torch.no_grad():
            curve.append(rmse(y_train, pred.numpy()))
        return float(loss.item())

    def gradient(vec: np.ndarray) -> np.ndarray:
        _vector_to_params(model, vec)
        model.zero_grad()
        pred = model(x_t)
        loss = torch.mean((pred - y_t) ** 2)
        loss.backward()
        return _params_grad_vector(model)

    x0 = _params_to_vector(model)
    optimize.minimize(
        objective,
        x0,
        method='BFGS',
        jac=gradient,
        options={'maxiter': maxiter, 'gtol': 1e-6},
    )
    return curve
