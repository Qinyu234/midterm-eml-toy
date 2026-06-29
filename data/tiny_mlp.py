"""
Tiny 1D MLP with fixed public weights (reproducible reference net).
用固定公开权重的小 MLP，供 eML 拟合其中一列参数/激活曲线。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import torch
import torch.nn as nn

ColumnKind = Literal['w1', 'b1', 'w2', 'neuron_act']


@dataclass(frozen=True)
class TinyPublicMLP:
    """4-hidden tanh MLP on [-2,2]; weights chosen by hand (documented below)."""
    hidden: int = 4

    # W1 (4×1), b1 (4), W2 (1×4), b2 (1) — public reference
    w1: Tuple[float, ...] = (0.85, -1.20, 0.55, 1.40)
    b1: Tuple[float, ...] = (0.10, -0.30, 0.25, -0.15)
    w2: Tuple[float, ...] = (0.50, -0.80, 0.35, 0.65)
    b2: float = 0.05

    def module(self) -> nn.Module:
        m = nn.Sequential(
            nn.Linear(1, self.hidden, dtype=torch.float64),
            nn.Tanh(),
            nn.Linear(self.hidden, 1, dtype=torch.float64),
        )
        with torch.no_grad():
            m[0].weight.copy_(torch.tensor(self.w1, dtype=torch.float64).unsqueeze(1))
            m[0].bias.copy_(torch.tensor(self.b1, dtype=torch.float64))
            m[2].weight.copy_(torch.tensor(self.w2, dtype=torch.float64).unsqueeze(0))
            m[2].bias.copy_(torch.tensor([self.b2], dtype=torch.float64))
        return m

    def forward(self, x: np.ndarray) -> np.ndarray:
        xt = torch.tensor(np.asarray(x, dtype=np.float64), dtype=torch.float64).unsqueeze(-1)
        with torch.no_grad():
            return self.module()(xt).squeeze(-1).numpy()

    def column_samples(
        self,
        kind: ColumnKind,
        *,
        neuron: int = 0,
        n: int = 200,
        x_lo: float = -2.0,
        x_hi: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Build (x, y) for eML fit.

        w1/b1/w2: x = normalized hidden index, y = weight/bias value.
        neuron_act: x = input grid, y = tanh(w1[i]*x + b1[i]).
        """
        if kind in ('w1', 'b1', 'w2'):
            idx = np.arange(self.hidden, dtype=np.float64)
            x = -1.0 + 2.0 * idx / max(1, self.hidden - 1)
            if kind == 'w1':
                y = np.array(self.w1, dtype=np.float64)
                label = f'w1_col (input weights, H={self.hidden})'
            elif kind == 'b1':
                y = np.array(self.b1, dtype=np.float64)
                label = f'b1_col (hidden biases, H={self.hidden})'
            else:
                y = np.array(self.w2, dtype=np.float64)
                label = f'w2_row (output weights, H={self.hidden})'
            return x, y, label

        if kind == 'neuron_act':
            i = int(neuron) % self.hidden
            x = np.linspace(x_lo, x_hi, n)
            w, b = self.w1[i], self.b1[i]
            y = np.tanh(w * x + b)
            label = f'neuron_act i={i} tanh({w:.4f}*x + {b:.4f})'
            return x, y, label

        raise ValueError(kind)

    def public_params_table(self) -> str:
        lines = [
            f'TinyPublicMLP hidden={self.hidden}',
            f'  W1 = {list(self.w1)}',
            f'  b1 = {list(self.b1)}',
            f'  W2 = {list(self.w2)}',
            f'  b2 = {self.b2}',
        ]
        return '\n'.join(lines)
