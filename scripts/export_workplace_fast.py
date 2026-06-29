"""Export FORMULA_WORKPLACE one byte at a time (progress to stderr)."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from export.workplace import iter_workplace_blocks


def main() -> int:
    out = Path('FORMULA_WORKPLACE.txt')
    header = (
        '# FORMULA_WORKPLACE: T value + Wirtinger D grad (∂z, ∂z̄ a+bi)\n'
        '# all leaf D_1..D_{n+1}; high→low chain rewrite on ∂z\n\n'
    )
    blocks = list(iter_workplace_blocks())
    out.write_text(header + '\n'.join(blocks), encoding='utf-8')
    print(f'Wrote {len(blocks)} nodes to {out}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
