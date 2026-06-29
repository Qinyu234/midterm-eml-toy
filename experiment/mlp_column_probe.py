"""Fit eML to one column/curve from TinyPublicMLP; report recovered structure."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import sympy as sp

from codec.sweep import describe_node, list_16_nodes
from data.targets import BenchmarkTarget
from data.tiny_mlp import ColumnKind, TinyPublicMLP
from experiment.search import fit_eml
from symbolic.pipeline import compute_node_exprs


@dataclass
class MlpColumnProbeResult:
    column_kind: ColumnKind
    label: str
    best_byte: int
    best_in_rmse: float
    best_n_eml: int
    node_label: str
    value_expr: str
    public_params: str
    top_rows: List[dict]

    def format_report(self) -> str:
        lines = [
            '# MLP column probe — eML structure recovery',
            '',
            '## Public reference MLP',
            self.public_params,
            '',
            f'## Target: {self.label}',
            f'best_byte={self.best_byte}  in_rmse={self.best_in_rmse:.6e}  n_eml={self.best_n_eml}',
            f'node: {self.node_label}',
            '',
            '## Recovered symbolic value (before numeric D bind)',
            self.value_expr,
            '',
            '## Top bytes by in_rmse',
            '| byte | n_eml | x_slot | bind | in_rmse |',
            '|------|-------|--------|------|---------|',
        ]
        for r in self.top_rows:
            lines.append(
                f'| {r["byte"]} | {r["n_eml"]} | {r["x_slot"]} | {r["bind"]} | '
                f'{r["in_rmse"]:.6e} |'
            )
        return '\n'.join(lines)


def _upsample_column(x: np.ndarray, y: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate sparse weight-index samples to dense grid for smoother fit."""
    if len(x) >= n:
        return x, y
    x_dense = np.linspace(float(x.min()), float(x.max()), n)
    y_dense = np.interp(x_dense, x, y)
    return x_dense, y_dense


def run_mlp_column_probe(
    *,
    kind: ColumnKind = 'w1',
    neuron: int = 0,
    max_steps: int = 800,
    lr: float = 0.01,
    eps: float = 1e-2,
    n_train: int = 200,
    upsample_weights: bool = True,
    top_k: int = 5,
) -> MlpColumnProbeResult:
    ref = TinyPublicMLP()
    x, y, label = ref.column_samples(kind, neuron=neuron, n=n_train)
    if kind in ('w1', 'b1', 'w2') and upsample_weights:
        x, y = _upsample_column(x, y, n_train)

    target = BenchmarkTarget(
        f'mlp_col_{kind}',
        float(x.min()),
        float(x.max()),
        ((float(x.min()) - 0.5, float(x.min())), (float(x.max()), float(x.max()) + 0.5)),
        lambda t: np.interp(t, x, y),
    )
    x_train, y_train = x, y
    x_ood = np.linspace(float(x.min()) - 0.3, float(x.max()) + 0.3, 40)
    y_ood = np.interp(x_ood, x, y)

    rows: List[dict] = []
    best_rmse = float('inf')
    best_byte = 1
    for node in list_16_nodes(evaluable_only=True):
        r = fit_eml(
            target, x_train, y_train, x_ood, y_ood,
            byte=node.byte, max_steps=max_steps, lr=lr, eps=eps,
        )
        if not np.isfinite(r.in_rmse):
            continue
        rows.append({
            'byte': node.byte,
            'n_eml': r.meta.get('n_eml'),
            'x_slot': 'L' if r.meta.get('x_slot') == 0 else 'R',
            'bind': node.input_bind_slot,
            'in_rmse': r.in_rmse,
        })
        if r.in_rmse < best_rmse:
            best_rmse = r.in_rmse
            best_byte = node.byte
    rows.sort(key=lambda r: r['in_rmse'])
    top = rows[:top_k]

    exprs = compute_node_exprs(best_byte)
    node = describe_node(best_byte)
    return MlpColumnProbeResult(
        column_kind=kind,
        label=label,
        best_byte=best_byte,
        best_in_rmse=best_rmse,
        best_n_eml=int(node.n_eml),
        node_label=node.label,
        value_expr=sp.pretty(exprs.value_expr),
        public_params=ref.public_params_table(),
        top_rows=top,
    )
