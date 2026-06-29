"""Experiment metrics: AE, recovery rate, step log. / 实验指标。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class StepRecord:
    step: int
    loss: float
    compute: int = 1

    @property
    def ae(self) -> float:
        return self.loss  # per-step Δloss filled by caller / 单步 Δloss 由调用方填充


@dataclass
class SearchRun:
    byte: int
    eq_id: str
    records: List[StepRecord] = field(default_factory=list)
    final_rmse: float = float('inf')
    recovered: bool = False

    def add_record(self, step: int, loss: float) -> None:
        self.records.append(StepRecord(step=step, loss=loss))

    @property
    def total_compute(self) -> int:
        return len(self.records)

    @property
    def mean_ae(self) -> float:
        if len(self.records) < 2:
            return 0.0
        losses = [r.loss for r in self.records]
        delta = losses[0] - losses[-1]
        return delta / max(self.total_compute, 1)


@dataclass(frozen=True)
class PropertyMetrics:
    """Gradient / numeric property checks at a node."""
    byte: int
    eq_id: str
    max_fd_error: float
    mean_dz_bar_norm: float
    finite_rate: float
    final_rmse: float
    recovered: bool


def recovery_rate(runs: List[SearchRun]) -> float:
    if not runs:
        return 0.0
    return sum(1 for r in runs if r.recovered) / len(runs)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
