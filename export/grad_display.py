"""
Gradient display: Wirtinger ∂z/∂z̄ (a+bi) + adjacent-chain rewrite on ∂z.
"""
from __future__ import annotations

from typing import Sequence

import sympy as sp

from symbolic.pipeline import ParamGradient
from symbolic.tetration import format_t_expr, simplify_in_t
from symbolic.wirtinger import format_abi


def grad_display_param(param: sp.Expr) -> str:
    """Label d/D_n matches internal symbol D_n (no offset)."""
    return str(param)


def _ratio_factor(lo: sp.Expr, hi: sp.Expr) -> sp.Expr | None:
    if sp.simplify(hi) == 0:
        return None
    if sp.simplify(hi - 1) == 0:
        return None
    ratio = sp.simplify(lo / hi)
    if sp.simplify(ratio * hi - lo) != 0:
        return None
    return simplify_in_t(ratio)


def _format_dz_line(
    param: sp.Expr,
    dz: sp.Expr,
    *,
    ref: sp.Expr | None = None,
) -> str:
    lo_l = grad_display_param(param)
    if ref is None:
        return f'  d/{lo_l}/∂z = {format_abi(dz)}'
    hi_l = grad_display_param(ref)
    rs = format_t_expr(dz)
    if rs == '1':
        dz_part = f'd/{hi_l}'
    elif rs == '-1':
        dz_part = f'-d/{hi_l}'
    else:
        dz_part = f'{rs} * d/{hi_l}'
    return f'  d/{lo_l}/∂z = {dz_part}'


def format_grad_lines(grads: Sequence[ParamGradient]) -> list[str]:
    if not grads:
        return ['  (none)']

    hi_to_lo = list(reversed(grads))
    by_param = {g.param: g for g in grads}
    lines: list[str] = []

    for i, g in enumerate(hi_to_lo):
        lo_l = grad_display_param(g.param)
        if i == 0:
            lines.append(_format_dz_line(g.param, by_param[g.param].dz))
            lines.append(f'  d/{lo_l}/∂z̄ = {format_abi(by_param[g.param].dz_bar)}')
            continue
        g_hi = hi_to_lo[i - 1]
        hi_g = by_param[g_hi.param]
        lo_g = by_param[g.param]
        ratio = _ratio_factor(lo_g.dz, hi_g.dz)
        if ratio is not None:
            lines.append(_format_dz_line(g.param, ratio, ref=g_hi.param))
        else:
            lines.append(_format_dz_line(g.param, lo_g.dz))
        lines.append(f'  d/{lo_l}/∂z̄ = {format_abi(lo_g.dz_bar)}')

    return lines
