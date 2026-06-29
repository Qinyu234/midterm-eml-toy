"""Incremental FORMULA_WORKPLACE export (write after each byte)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from codec.node import is_evaluable
from export.workplace import format_workplace_block, iter_4bit_node_bytes
from symbolic.pipeline import compute_node_formula_comparison


def main() -> int:
    out = Path('FORMULA_WORKPLACE.txt')
    header = (
        '# FORMULA_WORKPLACE: T value + Wirtinger D grad (∂z, ∂z̄ a+bi)\n'
        '# all leaf D_1..D_{n+1}; high→low chain rewrite on ∂z\n\n'
    )
    out.write_text(header, encoding='utf-8')
    n = 0
    for byte in iter_4bit_node_bytes():
        if not is_evaluable(byte):
            continue
        print(f'byte {byte}...', file=sys.stderr, flush=True)
        block = format_workplace_block(
            compute_node_formula_comparison(byte, include_raw_grads=False)
        ) + '\n'
        with out.open('a', encoding='utf-8') as f:
            f.write(block)
        n += 1
    print(f'done {n} nodes', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
