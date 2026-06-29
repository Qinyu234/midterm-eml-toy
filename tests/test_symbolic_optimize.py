# expand_ln_exp_algebra: off main path (log(T) opt disabled); unit tests only
import sympy as sp

from symbolic.optimize import (
    expand_ln_exp_algebra,
    fold_coeff_into_exp,
    log_div_exponent,
    log_div_value,
    merge_nested_tetration,
    stabilize_exp_quotient,
)
from symbolic.pipeline import simplify_expr
from symbolic.tetration import E_TAG, Tetration, compressed_exp, expand_t_to_log_exp, merge_opposite_tetration, simplify_in_t


def test_merge_nested_tetration():
    x = sp.Symbol('x')
    inner = Tetration(E_TAG, 2, x)
    outer = Tetration(E_TAG, 3, inner)
    assert merge_nested_tetration(outer) == Tetration(E_TAG, 5, x)


def test_merge_nested_tetration_negative():
    x = sp.Symbol('x')
    inner = Tetration(E_TAG, -1, x)
    outer = Tetration(E_TAG, -2, inner)
    assert merge_nested_tetration(outer) == Tetration(E_TAG, -3, x)


def test_merge_opposite_tetration_mixed_sign_cancels():
    x = sp.Symbol('x')
    inner = Tetration(E_TAG, -1, x)
    outer = Tetration(E_TAG, 1, inner)
    assert merge_opposite_tetration(outer) == x


def test_merge_opposite_tetration_same_sign_no_merge():
    x = sp.Symbol('x')
    inner = Tetration(E_TAG, 3, x)
    outer = Tetration(E_TAG, 2, inner)
    assert merge_opposite_tetration(outer) == Tetration(E_TAG, 2, Tetration(E_TAG, 3, x))


def test_stabilize_exp_quotient_n_ge_m():
    n, m = sp.symbols('n m', real=True)
    raw = sp.exp(n) / (sp.exp(n) + sp.exp(m))
    stable = stabilize_exp_quotient(raw)
    assert stable == sp.exp(n - sp.Max(m, n)) / (
        sp.exp(n - sp.Max(m, n)) + sp.exp(m - sp.Max(m, n))
    )
    concrete = stable.subs({n: 3, m: 3})
    assert sp.simplify(concrete - sp.Rational(1, 2)) == 0


def test_stabilize_exp_quotient_m_gt_n():
    n, m = sp.symbols('n m', real=True)
    raw = sp.exp(n) / (sp.exp(n) + sp.exp(m))
    stable = stabilize_exp_quotient(raw)
    concrete = stable.subs({n: 1, m: 3})
    expected = sp.exp(-2) / (1 + sp.exp(-2))
    assert sp.simplify(concrete - expected) == 0


def test_stabilize_all_exp_exponents_non_positive():
    n, m = sp.symbols('n m', real=True)
    raw = sp.exp(n) / (sp.exp(n) + sp.exp(m))
    stable = simplify_expr(raw)
    for node in stable.atoms(sp.exp):
        expn = node.args[0]
        assert not expn.is_positive, f'positive exponent {expn} in {stable}'


def test_compressed_exp_chain_merges_via_simplify():
    x = sp.Symbol('x')
    expr = compressed_exp(2, compressed_exp(3, x))
    assert simplify_in_t(expr) == Tetration(E_TAG, 2, Tetration(E_TAG, 3, x))
    assert simplify_expr(expr) == expand_t_to_log_exp(expr)


def test_fold_coeff_into_exp():
    z = sp.Symbol('z')
    D = sp.Symbol('D', positive=True)
    folded = fold_coeff_into_exp(D * sp.exp(z))
    assert sp.simplify(folded - sp.exp(sp.log(D) + z)) == 0


def test_ln_exp_quotient_value_domain():
    z = sp.Symbol('z')
    D = sp.Symbol('D', positive=True)
    raw = sp.log(sp.exp(z) / D)
    assert expand_ln_exp_algebra(raw) == log_div_value(z, D)


def test_ln_exp_quotient_exp_domain():
    z, d = sp.symbols('z d', real=True)
    raw = sp.log(sp.exp(z) / sp.exp(d))
    assert expand_ln_exp_algebra(raw) == log_div_exponent(z, d)


def test_ln_exp_difference_stabilized():
    z = sp.Symbol('z')
    D = sp.Symbol('D', positive=True)
    raw = sp.log(sp.exp(z) - D)
    out = expand_ln_exp_algebra(raw)
    expected = z + sp.log(1 - sp.exp(sp.log(D) - z))
    assert sp.simplify(out - expected) == 0
    assert sp.simplify(out.subs({z: 3, D: 2}) - sp.log(sp.exp(3) - 2)) == 0


def test_ln_exp_difference_exp_subtrahend():
    z, d = sp.symbols('z d', real=True)
    raw = sp.log(sp.exp(z) - sp.exp(d))
    out = expand_ln_exp_algebra(raw)
    expected = z + sp.log(1 - sp.exp(d - z))
    assert sp.simplify(out - expected) == 0
    assert sp.simplify(out.subs({z: 3, d: 1}) - sp.log(sp.exp(3) - sp.exp(1))) == 0


def test_merge_exp_products_d_symbols():
    d1, d2 = sp.symbols('D_1 D_2', real=True)
    from symbolic.optimize import merge_exp_products
    assert merge_exp_products(sp.exp(d1) * sp.exp(d2)) == sp.exp(d1 + d2)

