"""
Benchmark targets for eML vs baseline comparison.
拟合基准目标（解析函数 + 规则/语言友好组合 + 模式代理）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Literal, Sequence, Tuple

import numpy as np

TargetCategory = Literal['classic', 'rule']


@dataclass(frozen=True)
class BenchmarkTarget:
    target_id: str
    train_lo: float
    train_hi: float
    ood_segments: Tuple[Tuple[float, float], ...]
    fn: Callable[[np.ndarray], np.ndarray]
    category: TargetCategory = 'classic'

    def apply(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        y = self.fn(x)
        return np.asarray(y, dtype=np.float64)

    def sample_train(self, n: int = 200, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = rng.uniform(self.train_lo, self.train_hi, size=n)
        return x, self.apply(x)

    def sample_ood(self, n: int = 100, seed: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        segs = self.ood_segments
        per = max(1, n // len(segs))
        xs: List[np.ndarray] = []
        for lo, hi in segs:
            xs.append(rng.uniform(lo, hi, size=per))
        x = np.concatenate(xs)[:n]
        return x, self.apply(x)


def _stock_pattern(x: np.ndarray) -> np.ndarray:
    return x * (1.0 + 0.3 * np.sin(7.0 * x)) * np.exp(-0.05 * x ** 2)


def _info_pattern(x: np.ndarray) -> np.ndarray:
    return np.sin(x) * np.sin(13.0 * x) * np.exp(-0.05 * np.abs(x))


def _compose_sin_sq(x: np.ndarray) -> np.ndarray:
    return np.sin(x ** 2)


def _compose_log1p(x: np.ndarray) -> np.ndarray:
    return np.log1p(x ** 2)


def _rational(x: np.ndarray) -> np.ndarray:
    return (x ** 2 + 1.0) / (x + 2.0)


def _gauss_sin(x: np.ndarray) -> np.ndarray:
    return np.exp(-x ** 2) * np.sin(x)


def _relu_sq(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x) ** 2


def _interfere_sin(x: np.ndarray) -> np.ndarray:
    return np.sin(3.0 * x) * np.sin(11.0 * x)


def _sqrt_abs(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.abs(x) + 0.1)


def _nested_exp_sin(x: np.ndarray) -> np.ndarray:
    return np.exp(np.sin(x))


CLASSIC_TARGETS: Tuple[BenchmarkTarget, ...] = (
    BenchmarkTarget('sin', -2.0, 2.0, ((-4.0, -2.0), (2.0, 4.0)), np.sin),
    BenchmarkTarget('exp', -1.0, 1.0, ((-2.0, -1.0), (1.0, 2.0)), np.exp),
    BenchmarkTarget('log', 0.5, 3.0, ((0.1, 0.5), (3.0, 5.0)), np.log),
    BenchmarkTarget('sin_exp', -1.0, 1.0, ((-2.0, -1.0), (1.0, 2.0)), lambda x: np.sin(np.exp(x))),
    BenchmarkTarget('exp_sin', -2.0, 2.0, ((-4.0, -2.0), (2.0, 4.0)), lambda x: np.exp(np.sin(x))),
    BenchmarkTarget('stock_pattern', -3.0, 3.0, ((-5.0, -3.0), (3.0, 5.0)), _stock_pattern),
    BenchmarkTarget(
        'info_pattern',
        -math.pi,
        math.pi,
        ((-2 * math.pi, -math.pi), (math.pi, 2 * math.pi)),
        _info_pattern,
    ),
)

RULE_TARGETS: Tuple[BenchmarkTarget, ...] = (
    BenchmarkTarget(
        'compose_sin_sq', -2.0, 2.0, ((-3.0, -2.0), (2.0, 3.0)),
        _compose_sin_sq, category='rule',
    ),
    BenchmarkTarget(
        'compose_log1p', -2.0, 2.0, ((-3.0, -2.0), (2.0, 3.0)),
        _compose_log1p, category='rule',
    ),
    BenchmarkTarget(
        'rational', -1.5, 1.5, ((-2.5, -1.5), (1.5, 2.5)),
        _rational, category='rule',
    ),
    BenchmarkTarget(
        'gauss_sin', -2.5, 2.5, ((-4.0, -2.5), (2.5, 4.0)),
        _gauss_sin, category='rule',
    ),
    BenchmarkTarget(
        'relu_sq', -2.0, 2.0, ((-3.0, -2.0), (2.0, 3.0)),
        _relu_sq, category='rule',
    ),
    BenchmarkTarget(
        'interfere_sin', -math.pi, math.pi,
        ((-2 * math.pi, -math.pi), (math.pi, 2 * math.pi)),
        _interfere_sin, category='rule',
    ),
    BenchmarkTarget(
        'sqrt_abs', -2.0, 2.0, ((-3.0, -2.0), (2.0, 3.0)),
        _sqrt_abs, category='rule',
    ),
    BenchmarkTarget(
        'nested_exp_sin', -1.5, 1.5, ((-2.5, -1.5), (1.5, 2.5)),
        _nested_exp_sin, category='rule',
    ),
)

BENCHMARK_TARGETS: Tuple[BenchmarkTarget, ...] = CLASSIC_TARGETS + RULE_TARGETS


def list_benchmark_targets(*, category: TargetCategory | None = None) -> List[BenchmarkTarget]:
    if category is None:
        return list(BENCHMARK_TARGETS)
    return [t for t in BENCHMARK_TARGETS if t.category == category]


def get_benchmark_target(target_id: str) -> BenchmarkTarget:
    for t in BENCHMARK_TARGETS:
        if t.target_id == target_id:
            return t
    raise KeyError(target_id)


def resolve_targets(spec: Sequence[str] | str) -> List[BenchmarkTarget]:
    if spec == 'all' or spec == ['all']:
        return list_benchmark_targets()
    if spec == 'classic' or spec == ['classic']:
        return list_benchmark_targets(category='classic')
    if spec == 'rule' or spec == ['rule']:
        return list_benchmark_targets(category='rule')
    if isinstance(spec, str):
        spec = [s.strip() for s in spec.split(',') if s.strip()]
    return [get_benchmark_target(s) for s in spec]
