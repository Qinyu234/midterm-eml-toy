"""
FORMULA_WORKPLACE export: T-domain value and D_gradient per 4-bit node.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, TextIO

from codec.node import NodeCode, decode_node, is_evaluable
from export.grad_display import format_grad_lines
from symbolic.pipeline import NodeFormulaComparison, compute_node_formula_comparison
from symbolic.tetration import format_t_expr


def iter_4bit_node_bytes() -> range:
    return range(1, 16)


def format_workplace_block(cmp: NodeFormulaComparison) -> str:
    node = cmp.node
    lines = [
        format_node_header(cmp.byte, node),
        '',
        'value [before]:',
        f'  {format_t_expr(cmp.value_t_raw)}',
        '',
        'value [after]:',
        f'  {format_t_expr(cmp.value_t_opt)}',
        '',
        'D_gradient:',
        *format_grad_lines(cmp.d_grads_opt),
        '',
    ]
    return '\n'.join(lines)


def format_node_header(byte: int, node: NodeCode) -> str:
    return (
        f'=== byte {byte} n_eml={node.n_eml} x_slot={node.x_slot} ==='
    )


def iter_workplace_blocks(
    bytes: Iterable[int] | None = None,
) -> Iterator[str]:
    codes = iter_4bit_node_bytes() if bytes is None else bytes
    for byte in codes:
        if not is_evaluable(byte):
            continue
        yield format_workplace_block(
            compute_node_formula_comparison(byte, include_raw_grads=False)
        )


def write_formula_workplace(
    out: Path | TextIO,
    *,
    bytes: Iterable[int] | None = None,
) -> int:
    header = (
        '# FORMULA_WORKPLACE: T value + Wirtinger D grad (∂z, ∂z̄ a+bi)\n'
        '# all leaf D_1..D_{n+1}; high→low chain rewrite on ∂z\n\n'
    )
    blocks = list(iter_workplace_blocks(bytes))
    text = header + '\n'.join(blocks)
    if isinstance(out, Path):
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding='utf-8')
    else:
        out.write(text)
    return len(blocks)


def export_formula_workplace(out_path: Path | None = None) -> Path:
    out_path = out_path or Path('FORMULA_WORKPLACE.txt')
    write_formula_workplace(out_path)
    return out_path
