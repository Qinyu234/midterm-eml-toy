"""Probe: eML fit one column from TinyPublicMLP public weights."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from experiment.mlp_column_probe import run_mlp_column_probe


def main() -> int:
    p = argparse.ArgumentParser(
        description='Fit eML to one tiny-MLP weight column / neuron curve',
    )
    p.add_argument(
        '--kind',
        choices=('w1', 'b1', 'w2', 'neuron_act'),
        default='w1',
        help='w1/b1/w2 = weight column; neuron_act = tanh(w*x+b) curve',
    )
    p.add_argument('--neuron', type=int, default=0)
    p.add_argument('--max-steps', type=int, default=800)
    p.add_argument('--eps', type=float, default=1e-2)
    p.add_argument('--out', type=Path, default=Path('results/MLP_COLUMN.txt'))
    args = p.parse_args()

    print(f'probe kind={args.kind} neuron={args.neuron}', flush=True)
    result = run_mlp_column_probe(
        kind=args.kind,
        neuron=args.neuron,
        max_steps=args.max_steps,
        eps=args.eps,
    )
    text = result.format_report()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding='utf-8')
    print(text, flush=True)
    print(f'\n→ {args.out}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
