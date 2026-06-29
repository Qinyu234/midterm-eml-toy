"""MLP baseline for 1D regression."""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from baselines.base import FitResult
from baselines.training import OptimizerName, train_regressor
from data.targets import BenchmarkTarget
from experiment.tes import (
    evaluate_rmse,
    steps_to_threshold,
    total_compute,
    tower_efficiency_score,
)
from experiment.train_loop import TrainStopConfig

HIDDEN_SIZES = (4, 8, 16, 32, 64, 128)
MLP_OPTIMIZERS: Tuple[OptimizerName, ...] = ('adamw', 'bfgs')


class MLP1D(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(hidden, 1, dtype=torch.float64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.unsqueeze(-1)).squeeze(-1)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def fit_mlp(
    target: BenchmarkTarget,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_ood: np.ndarray,
    y_ood: np.ndarray,
    *,
    hidden: int = 32,
    max_steps: int = 5000,
    lr: float = 0.01,
    eps: float = 1e-2,
    optimizer: OptimizerName = 'adamw',
    stop: TrainStopConfig | None = None,
) -> FitResult:
    model = MLP1D(hidden)
    cfg = stop or TrainStopConfig(max_steps=max_steps)
    curve = train_regressor(
        model, x_train, y_train, optimizer=optimizer, lr=lr, stop=cfg,
    )

    def predict(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64)
            return model(xt).numpy()

    in_rmse = evaluate_rmse(predict, x_train, y_train)
    ood_rmse = evaluate_rmse(predict, x_ood, y_ood)
    n_params = model.n_params
    st = steps_to_threshold(curve, eps)
    compute = total_compute(
        'mlp', len(curve), n_params=n_params,
        steps_to_threshold=st, max_steps=cfg.max_steps,
    )
    return FitResult(
        method='mlp',
        target_id=target.target_id,
        in_rmse=in_rmse,
        ood_rmse=ood_rmse,
        n_params=n_params,
        complexity=float(n_params),
        steps_to_threshold=st,
        total_steps=len(curve),
        compute=compute,
        tes=tower_efficiency_score(in_rmse, compute),
        meta={'hidden': hidden, 'optimizer': optimizer},
        rmse_curve=curve,
    )


def scan_mlp(
    target: BenchmarkTarget,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_ood: np.ndarray,
    y_ood: np.ndarray,
    *,
    max_steps: int = 5000,
    lr: float = 0.01,
    eps: float = 1e-2,
    hidden_sizes: Tuple[int, ...] = HIDDEN_SIZES,
    optimizers: Tuple[OptimizerName, ...] = MLP_OPTIMIZERS,
    stop: TrainStopConfig | None = None,
) -> FitResult:
    cfg = stop or TrainStopConfig(max_steps=max_steps)
    results: List[FitResult] = []
    for opt_name in optimizers:
        for h in hidden_sizes:
            results.append(fit_mlp(
                target, x_train, y_train, x_ood, y_ood,
                hidden=h, max_steps=cfg.max_steps, lr=lr, eps=eps,
                optimizer=opt_name, stop=cfg,
            ))

    best = min(results, key=lambda r: r.in_rmse)
    passing = [r for r in results if r.steps_to_threshold is not None]
    if passing:
        passing.sort(key=lambda r: (r.meta['hidden'], r.in_rmse))
        return passing[0]
    return best
