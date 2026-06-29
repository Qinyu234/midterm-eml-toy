"""Fourier feature baseline for 1D regression."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from baselines.base import FitResult
from data.targets import BenchmarkTarget
from experiment.metrics import rmse
from experiment.tes import (
    evaluate_rmse,
    steps_to_threshold,
    total_compute,
    tower_efficiency_score,
)

FREQ_COUNTS = (2, 4, 8, 16, 32)


class FourierFeatures(nn.Module):
    def __init__(self, n_freq: int):
        super().__init__()
        self.n_freq = n_freq
        self.linear = nn.Linear(2 * n_freq + 1, 1, dtype=torch.float64)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        parts = [torch.ones_like(x)]
        for k in range(1, self.n_freq + 1):
            parts.append(torch.sin(k * x))
            parts.append(torch.cos(k * x))
        return torch.stack(parts, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.features(x)).squeeze(-1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def _train_fourier(
    model: FourierFeatures,
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_steps: int,
    lr: float,
) -> List[float]:
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_t = torch.tensor(x_train, dtype=torch.float64)
    y_t = torch.tensor(y_train, dtype=torch.float64)
    curve: List[float] = []
    for _ in range(max_steps):
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
    return curve


def fit_fourier(
    target: BenchmarkTarget,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_ood: np.ndarray,
    y_ood: np.ndarray,
    *,
    n_freq: int = 8,
    max_steps: int = 500,
    lr: float = 0.01,
    eps: float = 1e-2,
) -> FitResult:
    model = FourierFeatures(n_freq)
    curve = _train_fourier(model, x_train, y_train, max_steps=max_steps, lr=lr)

    def predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64)
            return model(xt).numpy()

    in_rmse = evaluate_rmse(predict, x_train, y_train)
    ood_rmse = evaluate_rmse(predict, x_ood, y_ood)
    n_params = model.n_params
    st = steps_to_threshold(curve, eps)
    compute = total_compute(
        'fourier', max_steps, n_params=n_params,
        steps_to_threshold=st, max_steps=max_steps,
    )
    return FitResult(
        method='fourier',
        target_id=target.target_id,
        in_rmse=in_rmse,
        ood_rmse=ood_rmse,
        n_params=n_params,
        complexity=float(n_params),
        steps_to_threshold=st,
        total_steps=len(curve),
        compute=compute,
        tes=tower_efficiency_score(in_rmse, compute),
        meta={'n_freq': n_freq},
        rmse_curve=curve,
    )


def scan_fourier(
    target: BenchmarkTarget,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_ood: np.ndarray,
    y_ood: np.ndarray,
    *,
    max_steps: int = 500,
    lr: float = 0.01,
    eps: float = 1e-2,
    freq_counts: Tuple[int, ...] = FREQ_COUNTS,
) -> FitResult:
    best: Optional[FitResult] = None
    for k in freq_counts:
        r = fit_fourier(
            target, x_train, y_train, x_ood, y_ood,
            n_freq=k, max_steps=max_steps, lr=lr, eps=eps,
        )
        if best is None or r.in_rmse < best.in_rmse:
            best = r
    minimal: Optional[FitResult] = None
    for k in freq_counts:
        r = fit_fourier(
            target, x_train, y_train, x_ood, y_ood,
            n_freq=k, max_steps=max_steps, lr=lr, eps=eps,
        )
        if r.steps_to_threshold is not None:
            if minimal is None or k < minimal.meta['n_freq']:
                minimal = r
    if minimal is not None:
        return minimal
    assert best is not None
    return best
