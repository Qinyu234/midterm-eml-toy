"""Node × function adaptability evaluation (DESIGN §5). / 适配性评估。"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Optional

import numpy as np

from codec.node import is_evaluable
from data.feynman import FeynmanFunction, list_feynman_functions
from experiment.metrics import SearchRun
from experiment.search import search_run


@dataclass
class NodeEvaluation:
    byte: int
    eq_id: str
    final_rmse: float
    converged: bool
    run: Optional[SearchRun] = None


@dataclass
class AdaptabilityMatrix:
    rows: List[int] = field(default_factory=list)
    cols: List[str] = field(default_factory=list)
    rmse: np.ndarray = field(default_factory=lambda: np.array([]))
    converged: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> dict:
        return {
            'rows': self.rows,
            'cols': self.cols,
            'rmse': self.rmse.tolist(),
            'converged': self.converged.tolist(),
        }


class NodeEvaluator:
    def __init__(self, *, steps: int = 30, functions: Optional[List[FeynmanFunction]] = None):
        self.steps = steps
        self.functions = functions or list_feynman_functions(n_vars=1)

    def evaluate_single_node(self, byte: int, fn: FeynmanFunction) -> NodeEvaluation:
        if not is_evaluable(byte):
            return NodeEvaluation(byte, fn.eq_id, float('inf'), False)
        run = search_run(byte, fn, steps=self.steps)
        converged = np.isfinite(run.final_rmse) and run.final_rmse < 1.0
        return NodeEvaluation(byte, fn.eq_id, run.final_rmse, converged, run)

    def evaluate(self, bytes: Optional[List[int]] = None) -> List[NodeEvaluation]:
        if bytes is None:
            bytes = [b for b in range(256) if is_evaluable(b)]
        results: List[NodeEvaluation] = []
        for byte in bytes:
            for fn in self.functions:
                results.append(self.evaluate_single_node(byte, fn))
        return results

    def adaptability_matrix(self, bytes: Optional[List[int]] = None) -> AdaptabilityMatrix:
        evals = self.evaluate(bytes)
        fn_ids = [f.eq_id for f in self.functions]
        node_bytes = sorted({e.byte for e in evals})
        rmse = np.full((len(node_bytes), len(fn_ids)), np.inf)
        conv = np.zeros((len(node_bytes), len(fn_ids)), dtype=bool)
        idx_b = {b: i for i, b in enumerate(node_bytes)}
        idx_f = {f: j for j, f in enumerate(fn_ids)}
        for e in evals:
            rmse[idx_b[e.byte], idx_f[e.eq_id]] = e.final_rmse
            conv[idx_b[e.byte], idx_f[e.eq_id]] = e.converged
        return AdaptabilityMatrix(node_bytes, fn_ids, rmse, conv)


def generate_report(matrix: AdaptabilityMatrix) -> str:
    lines = ['# Node Adaptability Report', '']
    lines.append(f'Nodes: {len(matrix.rows)}, Functions: {len(matrix.cols)}')
    if matrix.rmse.size:
        best = np.nanmin(matrix.rmse, axis=0)
        lines.append(f'Mean best RMSE per function: {float(np.mean(best)):.4e}')
    return '\n'.join(lines)
