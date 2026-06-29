"""Unified fit benchmark: MLP / Fourier / eML comparison."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from baselines.base import FitResult
from baselines.fourier import scan_fourier
from baselines.mlp import scan_mlp
from data.targets import BenchmarkTarget, resolve_targets
from experiment.search import scan_eml


def _load_splits(
    target: BenchmarkTarget,
    *,
    n_train: int = 200,
    n_ood: int = 100,
    seed: int = 0,
) -> tuple:
    x_train, y_train = target.sample_train(n_train, seed=seed)
    x_ood, y_ood = target.sample_ood(n_ood, seed=seed + 1)
    return x_train, y_train, x_ood, y_ood


def run_single(
    target: BenchmarkTarget,
    method: str,
    *,
    max_steps: int = 500,
    lr: float = 0.01,
    eps: float = 1e-2,
    bytes: Optional[Iterable[int]] = None,
    n_train: int = 200,
    n_ood: int = 100,
    seed: int = 0,
) -> FitResult:
    x_train, y_train, x_ood, y_ood = _load_splits(
        target, n_train=n_train, n_ood=n_ood, seed=seed,
    )
    if method == 'mlp':
        return scan_mlp(
            target, x_train, y_train, x_ood, y_ood,
            max_steps=max_steps, lr=lr, eps=eps,
        )
    if method == 'fourier':
        return scan_fourier(
            target, x_train, y_train, x_ood, y_ood,
            max_steps=max_steps, lr=lr, eps=eps,
        )
    if method == 'eml':
        return scan_eml(
            target, x_train, y_train, x_ood, y_ood,
            max_steps=max_steps, lr=lr, eps=eps, bytes=bytes,
        )
    raise ValueError(f'unknown method: {method}')


def run_comparison(
    target: BenchmarkTarget,
    *,
    methods: Sequence[str] = ('mlp', 'fourier', 'eml'),
    max_steps: int = 500,
    lr: float = 0.01,
    eps: float = 1e-2,
    bytes: Optional[Iterable[int]] = None,
    n_train: int = 200,
    n_ood: int = 100,
    seed: int = 0,
) -> List[FitResult]:
    return [
        run_single(
            target, m,
            max_steps=max_steps, lr=lr, eps=eps, bytes=bytes,
            n_train=n_train, n_ood=n_ood, seed=seed,
        )
        for m in methods
    ]


def summarize_table(results: List[FitResult]) -> str:
    lines = [
        '| target | method | in_rmse | ood_rmse | ood_ratio | n_params | complexity | steps | compute | TES |',
        '|--------|--------|---------|----------|-----------|----------|------------|-------|---------|-----|',
    ]
    for r in results:
        steps = r.steps_to_threshold if r.steps_to_threshold is not None else 'FAIL'
        lines.append(
            f'| {r.target_id} | {r.method} | {r.in_rmse:.4e} | {r.ood_rmse:.4e} | '
            f'{r.ood_ratio:.2f} | {r.n_params} | {r.complexity:.2f} | {steps} | '
            f'{r.compute:.2e} | {r.tes:.4e} |'
        )
    return '\n'.join(lines)


def run_benchmark(
    targets: Sequence[str] | str = 'all',
    *,
    methods: Sequence[str] = ('mlp', 'fourier', 'eml'),
    max_steps: int = 500,
    lr: float = 0.01,
    eps: float = 1e-2,
    bytes: Optional[Iterable[int]] = None,
) -> List[FitResult]:
    all_results: List[FitResult] = []
    for target in resolve_targets(targets):
        all_results.extend(
            run_comparison(
                target, methods=methods,
                max_steps=max_steps, lr=lr, eps=eps, bytes=bytes,
            )
        )
    return all_results


def results_to_json(results: List[FitResult]) -> str:
    payload = []
    for r in results:
        d = asdict(r)
        payload.append(d)
    return json.dumps(payload, indent=2)


def write_results(results: List[FitResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(results_to_json(results), encoding='utf-8')
