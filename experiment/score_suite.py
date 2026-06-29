"""
Full score suite: benchmark targets × methods × 16-node sweep.
y @ tree top; x binds bottom_d (min |d|, center tie); topology sweeps x_slot × n_eml.
"""
from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from baselines.base import FitResult
from baselines.fourier import scan_fourier
from baselines.mlp import scan_mlp
from codec.sweep import NodeSweepEntry, list_16_nodes
from data.targets import BenchmarkTarget, list_benchmark_targets, resolve_targets
from experiment.fit_benchmark import _load_splits
from experiment.search import fit_eml


@dataclass
class ScoreRow:
    target_id: str
    method: str
    byte: Optional[int]
    n_eml: Optional[int]
    x_slot: Optional[int]
    x_leaf: Optional[int]
    input_bind_slot: Optional[int]
    in_rmse: float
    ood_rmse: float
    ood_ratio: float
    n_params: int
    complexity: float
    steps_to_threshold: Optional[int]
    total_steps: int
    compute: float
    tes: float
    passed: bool = False
    node_label: str = ''
    meta: dict = field(default_factory=dict)

    @classmethod
    def from_fit(
        cls,
        r: FitResult,
        node: Optional[NodeSweepEntry] = None,
        *,
        eps: float = 1e-2,
    ) -> ScoreRow:
        byte = r.meta.get('byte') if r.meta else None
        return cls(
            target_id=r.target_id,
            method=r.method,
            byte=byte,
            n_eml=r.meta.get('n_eml') if r.meta else None,
            x_slot=r.meta.get('x_slot') if r.meta else None,
            x_leaf=node.x_leaf if node else r.meta.get('x_leaf'),
            input_bind_slot=(
                node.input_bind_slot if node else r.meta.get('input_bind_slot')
            ),
            in_rmse=r.in_rmse,
            ood_rmse=r.ood_rmse,
            ood_ratio=r.ood_ratio,
            n_params=r.n_params,
            complexity=r.complexity,
            steps_to_threshold=r.steps_to_threshold,
            total_steps=r.total_steps,
            compute=r.compute,
            tes=r.tes,
            passed=bool(r.in_rmse < eps),
            node_label=node.label if node else '',
            meta=dict(r.meta),
        )


def _fit_to_row(
    r: FitResult,
    node: Optional[NodeSweepEntry] = None,
    *,
    eps: float = 1e-2,
) -> ScoreRow:
    if node is not None:
        r.meta['x_leaf'] = node.x_leaf
        r.meta['input_bind_slot'] = node.input_bind_slot
    return ScoreRow.from_fit(r, node, eps=eps)


def run_score_for_target(
    target: BenchmarkTarget,
    *,
    max_steps: int = 500,
    lr: float = 0.01,
    eps: float = 1e-2,
    methods: Sequence[str] = ('mlp', 'fourier', 'eml'),
    nodes: Optional[List[NodeSweepEntry]] = None,
    n_train: int = 200,
    n_ood: int = 100,
    seed: int = 0,
) -> List[ScoreRow]:
    x_train, y_train, x_ood, y_ood = _load_splits(
        target, n_train=n_train, n_ood=n_ood, seed=seed,
    )
    rows: List[ScoreRow] = []
    if 'mlp' in methods:
        print(f'  [{target.target_id}] mlp scan...', file=sys.stderr, flush=True)
        rows.append(_fit_to_row(scan_mlp(
            target, x_train, y_train, x_ood, y_ood,
            max_steps=max_steps, lr=lr, eps=eps,
        ), eps=eps))
    if 'fourier' in methods:
        print(f'  [{target.target_id}] fourier scan...', file=sys.stderr, flush=True)
        rows.append(_fit_to_row(scan_fourier(
            target, x_train, y_train, x_ood, y_ood,
            max_steps=max_steps, lr=lr, eps=eps,
        ), eps=eps))
    if 'eml' in methods:
        node_list = nodes or list_16_nodes(evaluable_only=True)
        for i, node in enumerate(node_list, 1):
            print(
                f'  [{target.target_id}] eml {i}/{len(node_list)} byte={node.byte}...',
                file=sys.stderr, flush=True,
            )
            r = fit_eml(
                target, x_train, y_train, x_ood, y_ood,
                byte=node.byte,
                max_steps=max_steps,
                lr=lr,
                eps=eps,
                binding='bottom_d',
            )
            rows.append(_fit_to_row(r, node, eps=eps))
    return rows


def run_score_suite(
    targets: Sequence[str] | str = 'all',
    *,
    max_steps: int = 500,
    lr: float = 0.01,
    eps: float = 1e-2,
    methods: Sequence[str] = ('mlp', 'fourier', 'eml'),
    on_target_done: Optional[Callable[[List[ScoreRow], str], None]] = None,
) -> List[ScoreRow]:
    nodes = list_16_nodes(evaluable_only=True)
    all_rows: List[ScoreRow] = []
    for target in resolve_targets(targets):
        batch = run_score_for_target(
            target,
            max_steps=max_steps,
            lr=lr,
            eps=eps,
            methods=methods,
            nodes=nodes,
        )
        all_rows.extend(batch)
        if on_target_done is not None:
            on_target_done(all_rows, target.target_id)
    return all_rows


def append_score_row(
    row: ScoreRow,
    *,
    json_path: Path = Path('results/SCORE.json'),
) -> None:
    """Append one row to JSON lines file (crash-safe incremental log)."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(asdict(row), ensure_ascii=False) + '\n')


def format_score_summary(rows: List[ScoreRow], *, eps: float = 1e-2) -> str:
    """Per-target best row per method + TES winner."""
    lines = [
        f'# Summary (eps={eps:.0e})',
        '',
        '| target | best_mlp_in | best_fourier_in | best_eml_in | best_eml_byte | '
        'eml_pass_n | winner_tes | winner_in_rmse |',
        '|--------|-------------|-----------------|-------------|---------------|'
        '-----------|------------|----------------|',
    ]
    targets = sorted({r.target_id for r in rows})
    for tid in targets:
        sub = [r for r in rows if r.target_id == tid]
        by_m = {m: [r for r in sub if r.method == m] for m in ('mlp', 'fourier', 'eml')}
        best = {m: (min(rs, key=lambda r: r.in_rmse) if rs else None) for m, rs in by_m.items()}
        eml_rs = by_m.get('eml', [])
        eml_pass = sum(1 for r in eml_rs if r.passed)
        best_eml = best.get('eml')
        winner_tes = max(sub, key=lambda r: r.tes)
        winner_rmse = min(sub, key=lambda r: r.in_rmse)
        mlp_in = f'{best["mlp"].in_rmse:.4e}' if best.get('mlp') else '—'
        fou_in = f'{best["fourier"].in_rmse:.4e}' if best.get('fourier') else '—'
        eml_in = f'{best_eml.in_rmse:.4e}' if best_eml else '—'
        eml_byte = str(best_eml.byte) if best_eml and best_eml.byte is not None else '—'
        w_tes = f'{winner_tes.method}({winner_tes.byte or ""})'.strip('()')
        if winner_tes.byte is None:
            w_tes = winner_tes.method
        else:
            w_tes = f'{winner_tes.method} b={winner_tes.byte}'
        w_rmse = f'{winner_rmse.method}'
        if winner_rmse.byte is not None:
            w_rmse += f' b={winner_rmse.byte}'
        lines.append(
            f'| {tid} | {mlp_in} | {fou_in} | {eml_in} | {eml_byte} | '
            f'{eml_pass}/{len(eml_rs)} | {w_tes} | {w_rmse} |'
        )
    return '\n'.join(lines)


def format_score_table(rows: List[ScoreRow], *, eps: float = 1e-2) -> str:
    lines = [
        '# SCORE: target × method × node (y @ tree top, bind bottom_d)',
        f'# nodes: 16 low-nibble sweep; {len(list_16_nodes(evaluable_only=True))} evaluable',
        f'# eps={eps:.0e}',
        '',
        format_score_summary(rows, eps=eps),
        '',
        '# Detail',
        '',
        '| target | method | byte | n_eml | x_slot | bind_leaf | in_rmse | ood_rmse | '
        'ood_ratio | n_params | complexity | compute | steps | pass | TES |',
        '|--------|--------|------|-------|--------|-----------|---------|----------|'
        '-----------|----------|------------|---------|-------|------|-----|',
    ]
    for r in rows:
        steps = r.steps_to_threshold if r.steps_to_threshold is not None else 'FAIL'
        byte_s = '' if r.byte is None else str(r.byte)
        n_eml_s = '' if r.n_eml is None else str(r.n_eml)
        xsl = '' if r.x_slot is None else ('L' if r.x_slot == 0 else 'R')
        xleaf = '' if r.input_bind_slot is None else str(r.input_bind_slot)
        pass_s = 'Y' if r.passed else 'N'
        lines.append(
            f'| {r.target_id} | {r.method} | {byte_s} | {n_eml_s} | {xsl} | {xleaf} | '
            f'{r.in_rmse:.4e} | {r.ood_rmse:.4e} | {r.ood_ratio:.2f} | '
            f'{r.n_params} | {r.complexity:.2f} | {r.compute:.2e} | {steps} | {pass_s} | '
            f'{r.tes:.4e} |'
        )
    return '\n'.join(lines)


def write_score_files(
    rows: List[ScoreRow],
    *,
    txt_path: Path = Path('results/SCORE.txt'),
    json_path: Path = Path('results/SCORE.json'),
    eps: float = 1e-2,
) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text(format_score_table(rows, eps=eps), encoding='utf-8')
    payload = {
        'eps': eps,
        'nodes_16': [asdict(n) for n in list_16_nodes()],
        'targets': [t.target_id for t in list_benchmark_targets()],
        'summary': format_score_summary(rows, eps=eps),
        'rows': [asdict(r) for r in rows],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
