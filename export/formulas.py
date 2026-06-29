"""
Pure-symbolic formula export (no lambdify, no numeric eval).
纯符号公式导出（拒绝数值求值）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, TextIO

import sympy as sp

from codec.node import NodeCode, decode_node, is_evaluable
from symbolic.pipeline import NodeSymbolicExprs, compute_node_exprs


def format_node_header(byte: int, node: NodeCode) -> str:
    return (
        f'=== byte {byte} (low={node.low_nibble:#04x}) '
        f'n_eml={node.n_eml} x_slot={node.x_slot} reserved={node.reserved} ==='
    )


def format_node_formulas(exprs: NodeSymbolicExprs, *, pretty: bool = True) -> str:
    """Return value + grad as sympy strings only / 仅输出符号公式。"""
    printer = sp.pretty if pretty else str
    lines = [
        format_node_header(exprs.byte, exprs.node),
        f'value_expr = {printer(exprs.value_expr)}',
        '',
        'grad_expr:',
    ]
    for g in exprs.param_grads:
        lines.append(f'  ∂/∂{g.param} = {printer(g.grad)}')
    lines.append('')
    return '\n'.join(lines)


def iter_node_formulas(
    bytes: Iterable[int] | None = None,
    *,
    pretty: bool = True,
) -> Iterator[str]:
    """Yield formatted symbolic blocks; skips n_eml=0 / 遍历有效 node 符号块。"""
    codes = range(256) if bytes is None else bytes
    for byte in codes:
        if not is_evaluable(byte):
            continue
        exprs = compute_node_exprs(byte)
        yield format_node_formulas(exprs, pretty=pretty)


def write_all_formulas(
    out: Path | TextIO,
    *,
    bytes: Iterable[int] | None = None,
    pretty: bool = True,
) -> int:
    """Write formulas to file or stream; returns count / 写入文件，返回 node 数。"""
    blocks = list(iter_node_formulas(bytes, pretty=pretty))
    text = '\n'.join(blocks)
    if isinstance(out, Path):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding='utf-8')
    else:
        out.write(text)
    return len(blocks)


def write_per_node_files(out_dir: Path, *, pretty: bool = True) -> int:
    """One file per byte under out_dir/node_XXX.txt / 每个 node 单独文件。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for byte in range(256):
        if not is_evaluable(byte):
            continue
        exprs = compute_node_exprs(byte)
        path = out_dir / f'node_{byte:03d}.txt'
        path.write_text(format_node_formulas(exprs, pretty=pretty), encoding='utf-8')
        count += 1
    return count


def export_phase1_txt(out_path: Path | None = None) -> Path:
    """Write all evaluable node formulas to one file / 导出全部 node 符号公式。"""
    out_path = out_path or Path('phase1_symbols.txt')
    write_all_formulas(out_path)
    return out_path


def run_phase1_default_output() -> Path:
    return export_phase1_txt(Path('phase1_symbols.txt'))
