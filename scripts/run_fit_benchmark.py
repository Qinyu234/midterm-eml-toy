"""Run eML vs MLP vs Fourier fit benchmark with TES."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from experiment.fit_benchmark import run_benchmark, summarize_table, write_results


def _parse_bytes(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def main() -> int:
    p = argparse.ArgumentParser(description='eML fit benchmark vs MLP/Fourier')
    p.add_argument('--targets', default='all', help='all or comma ids')
    p.add_argument('--methods', default='mlp,fourier,eml')
    p.add_argument('--eps', type=float, default=1e-2)
    p.add_argument('--max-steps', type=int, default=500)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--bytes', default='1-15', help='eML byte sweep range')
    p.add_argument('--out', type=Path, default=Path('results/benchmark.json'))
    args = p.parse_args()

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    bytes_ = _parse_bytes(args.bytes) if 'eml' in methods else None

    results = run_benchmark(
        args.targets,
        methods=methods,
        max_steps=args.max_steps,
        lr=args.lr,
        eps=args.eps,
        bytes=bytes_,
    )
    print(summarize_table(results))
    write_results(results, args.out)
    print(f'\nwrote {args.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
