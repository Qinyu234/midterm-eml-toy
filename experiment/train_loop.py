"""Shared training stop criteria and learning-rate schedule."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True)
class TrainStopConfig:
    max_steps: int = 5000
    patience: int = 40
    min_delta: float = 1e-7
    osc_window: int = 25
    osc_rel_range: float = 0.005


def lr_decay(initial_lr: float, step: int, *, every: int = 50, factor: float = 0.9) -> float:
    if step <= 0:
        return initial_lr
    return initial_lr * (factor ** (step // every))


def training_should_stop(curve: List[float], cfg: TrainStopConfig) -> bool:
    if not curve:
        return False
    if not np.isfinite(curve[-1]):
        return True
    if len(curve) >= cfg.max_steps:
        return True
    if len(curve) < cfg.patience + 2:
        return False

    best = float(np.min(curve))
    recent = curve[-cfg.patience:]
    recent_best = float(np.min(recent))
    if recent_best <= best + cfg.min_delta:
        return False

    tail = curve[-cfg.osc_window:] if len(curve) >= cfg.osc_window else curve
    rel_range = (max(tail) - min(tail)) / (best + 1e-12)
    return rel_range <= cfg.osc_rel_range or recent_best > best + cfg.min_delta
