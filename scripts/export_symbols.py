"""
CLI: export symbolic value_expr and grad_expr per node (numeric-free).
CLI：按 node 导出 value / grad 符号公式，拒绝数值。

Usage / 用法:
  python scripts/export_symbols.py
  python scripts/export_symbols.py --byte 1 9
  python scripts/export_symbols.py --per-node -o out/nodes
  python scripts/export_symbols.py -o formulas.txt --no-pretty
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow running as script from py/eml / 允许在 py/eml 下直接运行
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from export.formulas import (
    iter_node_formulas,
    write_all_formulas,
    write_per_node_files,
)


def _assert_symbolic_only_deps() -> None:
    """Block compile/numerics if already loaded / 禁止已加载数值编译模块。"""
    blocked = ('compile.lambdify_exprs', 'numerics.symbolic_model')
    for name in blocked:
        if name in sys.modules:
            raise RuntimeError(
                f'export_symbols is symbolic-only; unload numeric module: {name}'
            )


def main(argv: list[str] | None = None) -> int:
    _assert_symbolic_only_deps()
    p = argparse.ArgumentParser(
        description='Export symbolic value/grad formulas per node (no numeric eval).'
    )
    p.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('phase1_symbols.txt'),
        help='Output file or directory (with --per-node) / 输出路径',
    )
    p.add_argument(
        '--byte', type=int, nargs='*', metavar='B',
        help='Only these bytes (0-255); default all evaluable / 指定 byte',
    )
    p.add_argument(
        '--per-node', action='store_true',
        help='Write one file per node into output dir / 每 node 一文件',
    )
    p.add_argument(
        '--no-pretty', action='store_true',
        help='Use str() instead of sympy pretty / 不用 pretty 打印',
    )
    p.add_argument(
        '--stdout', action='store_true',
        help='Print to stdout instead of file / 打印到标准输出',
    )
    args = p.parse_args(argv)
    pretty = not args.no_pretty
    byte_list = args.byte

    if args.stdout:
        for block in iter_node_formulas(byte_list, pretty=pretty):
            print(block)
        return 0

    if args.per_node:
        n = write_per_node_files(args.output, pretty=pretty)
        print(f'Wrote {n} symbolic node files to {args.output}')
        return 0

    n = write_all_formulas(args.output, bytes=byte_list, pretty=pretty)
    print(f'Wrote {n} nodes to {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
