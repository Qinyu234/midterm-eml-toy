"""
CLI: export FORMULA_WORKPLACE.txt (raw vs optimized value / D_gradient).
CLI：导出 4-bit node 优化前后公式到 FORMULA_WORKPLACE.txt。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from export.workplace import export_formula_workplace, write_formula_workplace


def _assert_symbolic_only_deps() -> None:
    blocked = ('compile.lambdify_exprs', 'numerics.symbolic_model')
    for name in blocked:
        if name in sys.modules:
            raise RuntimeError(
                f'export_workplace is symbolic-only; unload numeric module: {name}'
            )


def main(argv: list[str] | None = None) -> int:
    _assert_symbolic_only_deps()
    p = argparse.ArgumentParser(description='Export FORMULA_WORKPLACE.txt')
    p.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('FORMULA_WORKPLACE.txt'),
        help='Output path (default FORMULA_WORKPLACE.txt)',
    )
    p.add_argument('--byte', type=int, nargs='*', metavar='B', help='Subset of bytes')
    p.add_argument('--stdout', action='store_true')
    args = p.parse_args(argv)

    if args.stdout:
        import io
        buf = io.StringIO()
        n = write_formula_workplace(buf, bytes=args.byte)
        print(buf.getvalue(), end='')
        print(f'# wrote {n} nodes', file=sys.stderr)
        return 0

    if args.byte:
        n = write_formula_workplace(args.output, bytes=args.byte)
    else:
        n = write_formula_workplace(args.output)
    print(f'Wrote {n} nodes to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
