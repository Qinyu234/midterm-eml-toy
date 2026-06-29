"""Run node × Feynman adaptability matrix with gradient property checks."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from codec.node import decode_node, is_evaluable
from data.feynman import FeynmanFunction, list_feynman_functions
from eval.grad_verify import verify_gradients
from eval.node_eval import NodeEvaluator, generate_report
from experiment.metrics import PropertyMetrics
from experiment.search import search_run


def _parse_bytes(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(b for b in out if is_evaluable(b)))


def run_property(
    byte: int,
    fn: FeynmanFunction,
    *,
    steps: int,
) -> PropertyMetrics:
    gv = verify_gradients(byte, n_vars=fn.n_vars, n_points=3)
    run = search_run(byte, fn, steps=steps)
    return PropertyMetrics(
        byte=byte,
        eq_id=fn.eq_id,
        max_fd_error=gv.max_fd_error,
        mean_dz_bar_norm=gv.mean_dz_bar_norm,
        finite_rate=gv.finite_rate,
        final_rmse=run.final_rmse,
        recovered=run.recovered,
    )


def main() -> int:
    p = argparse.ArgumentParser(description='Node × Feynman matrix + grad checks')
    p.add_argument('--bytes', default='1-15', help='byte list or range')
    p.add_argument('--steps', type=int, default=25)
    p.add_argument('--n-vars', type=int, default=None, help='filter Feynman n_vars')
    p.add_argument('--json', type=Path, default=None)
    p.add_argument('--grad-only', action='store_true')
    args = p.parse_args()

    bytes_ = _parse_bytes(args.bytes)
    fns = list_feynman_functions(args.n_vars)

    rows: list[dict] = []
    for byte in bytes_:
        node = decode_node(byte)
        for fn in fns:
            if fn.n_vars > node.n_eml + 1:
                continue
            if args.grad_only:
                gv = verify_gradients(byte, n_vars=fn.n_vars, n_points=3)
                rows.append({
                    'byte': byte,
                    'eq_id': fn.eq_id,
                    'max_fd_error': gv.max_fd_error,
                    'finite_rate': gv.finite_rate,
                    'dz_bar_norm': gv.mean_dz_bar_norm,
                    'grad_ok': gv.ok,
                })
            else:
                pm = run_property(byte, fn, steps=args.steps)
                rows.append({
                    'byte': pm.byte,
                    'eq_id': pm.eq_id,
                    'max_fd_error': pm.max_fd_error,
                    'finite_rate': pm.finite_rate,
                    'dz_bar_norm': pm.mean_dz_bar_norm,
                    'final_rmse': pm.final_rmse,
                    'recovered': pm.recovered,
                })

    ev = NodeEvaluator(steps=args.steps, functions=fns)
    matrix = ev.adaptability_matrix(bytes=bytes_)
    report = generate_report(matrix)

    print(report)
    print(f'\n# property rows: {len(rows)}')
    ok_grad = sum(1 for r in rows if r.get('max_fd_error', 1) < 0.05)
    print(f'grad_fd_ok: {ok_grad}/{len(rows)}')
    if not args.grad_only:
        rec = sum(1 for r in rows if r.get('recovered'))
        print(f'recovered: {rec}/{len(rows)}')

    if args.json:
        args.json.write_text(
            json.dumps({'report': report, 'rows': rows}, indent=2),
            encoding='utf-8',
        )
        print(f'wrote {args.json}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
