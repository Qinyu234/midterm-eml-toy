"""Linear optimal projection scheme A (§3.2). / 线性 Optimal Projection（方案 A）。"""
from __future__ import annotations

import torch
import torch.nn as nn


class LinearProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.beta = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.alpha * z.real + self.beta * z.imag
