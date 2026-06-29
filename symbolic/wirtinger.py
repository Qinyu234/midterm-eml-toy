"""
Wirtinger derivatives ∂f/∂z and ∂f/∂z̄ for complex chain rule.
Wirtinger 导数：传导数需 z 与 z̄ 两分量，存 a+bi。
"""
from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from symbolic.tetration import Tetration, format_t_expr, simplify_in_t


@dataclass(frozen=True)
class WirtingerJacobian:
    """∂f/∂z and ∂f/∂z̄ as complex sympy expressions (a+bi)."""
    dz: sp.Expr
    dz_bar: sp.Expr


def _subst_back(expr: sp.Expr, z: sp.Expr, re: sp.Symbol, im: sp.Symbol) -> sp.Expr:
    """Replace temporary re/im split with atomic z."""
    expr = expr.subs(re + sp.I * im, z)
    if expr.has(re) or expr.has(im):
        expr = expr.subs(
            {
                re: (z + sp.conjugate(z)) / 2,
                im: (z - sp.conjugate(z)) / (2 * sp.I),
            }
        )
    return sp.simplify(expr)


def wirtinger_grad(expr: sp.Expr, z: sp.Expr) -> WirtingerJacobian:
    """∂f/∂z, ∂f/∂z̄; split z=re+I*im then substitute back as atomic z."""
    re = sp.Symbol(f'__{z.name}_re', real=True)
    im = sp.Symbol(f'__{z.name}_im', real=True)
    zc = re + sp.I * im
    fs = expr.subs(z, zc)
    d_re = sp.diff(fs, re)
    d_im = sp.diff(fs, im)
    dz = sp.simplify((d_re - sp.I * d_im) / 2)
    dz_bar = sp.simplify((d_re + sp.I * d_im) / 2)
    return WirtingerJacobian(
        dz=_subst_back(dz, z, re, im),
        dz_bar=_subst_back(dz_bar, z, re, im),
    )


def format_abi(expr: sp.Expr) -> str:
    """Format complex expr as a+b*I (keep D_n atomic)."""
    expr = sp.simplify(expr)
    if expr.has(Tetration):
        return format_t_expr(expr)
    c = sp.expand_complex(expr)
    if sp.simplify(c) == 0:
        return '0'
    re = sp.expand(sp.re(c))
    im = sp.expand(sp.im(c))
    if sp.simplify(im) == 0:
        return format_t_expr(re)
    if sp.simplify(re) == 0:
        im_s = format_t_expr(im)
        if im_s == '1':
            return 'I'
        if im_s == '-1':
            return '-I'
        return f'{im_s}*I'
    re_s = format_t_expr(re)
    im_s = format_t_expr(im)
    if im_s == '1':
        return f'{re_s}+I'
    if im_s == '-1':
        return f'{re_s}-I'
    if im_s.startswith('-'):
        return f'{re_s}{im_s}*I'
    return f'{re_s}+{im_s}*I'


def simplify_wirtinger(w: WirtingerJacobian) -> WirtingerJacobian:
    dz, dz_bar = sp.simplify(w.dz), sp.simplify(w.dz_bar)
    if dz.has(Tetration) or dz_bar.has(Tetration):
        dz = simplify_in_t(dz)
        dz_bar = simplify_in_t(dz_bar)
    return WirtingerJacobian(dz=dz, dz_bar=dz_bar)
