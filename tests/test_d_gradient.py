"""D-space formula pipeline: C-tree → D=T(e,d,C) → ∂/∂D."""
import sympy as sp

from symbolic.pipeline import compute_node_formula_comparison, simplify_expr
from symbolic.tetration import format_t_expr
from tree.coords import c_to_d, d_to_c


def test_c_to_d_inverts_d_to_c():
    c = sp.Symbol('c', positive=True)
    d = sp.Symbol('D_1', positive=True)
    for cx, cy in [(1, 0), (2, 0), (0, 0), (0, -1), (0, -2)]:
        assert simplify_expr(
            c_to_d(d_to_c(d, cx, cy), cx, cy) - d
        ) == 0
        assert simplify_expr(
            d_to_c(c_to_d(c, cx, cy), cx, cy) - c
        ) == 0


def test_byte1_value_in_d_only():
    cmp = compute_node_formula_comparison(1)
    syms = cmp.value_t_opt.free_symbols
    assert sp.Symbol('x') not in syms
    assert all(str(s).startswith('D_') for s in syms)


def test_byte1_no_x_after_optimize():
    cmp = compute_node_formula_comparison(1)
    assert sp.Symbol('x') not in cmp.value_t_opt.free_symbols
    assert str(cmp.value_t_opt) == '-D_1 + D_2'


def test_byte1_d_gradient_count():
    cmp = compute_node_formula_comparison(1)
    assert len(cmp.d_grads_opt) == 2
    params = {g.param for g in cmp.d_grads_opt}
    assert params == {sp.Symbol('D_1'), sp.Symbol('D_2')}
