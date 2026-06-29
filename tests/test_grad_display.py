"""Gradient display rewrite tests."""
import sympy as sp

from export.grad_display import format_grad_lines, grad_display_param
from symbolic.pipeline import ParamGradient, compute_node_formula_comparison


def test_grad_display_param_identity():
    assert grad_display_param(sp.Symbol('D_7')) == 'D_7'


def test_byte7_d6_rewritten_with_d7():
    cmp = compute_node_formula_comparison(7)
    lines = format_grad_lines(cmp.d_grads_opt)
    text = '\n'.join(lines)
    assert 'd/D_8/∂z = 1' in text
    assert 'd/D_7/∂z = T(e, 1, D_7)/' in text
    assert 'd/D_1/∂z' in text
    assert 'd/D_3/∂z' in text
    assert '* d/D_7' in text
    assert '∂z̄' in text
