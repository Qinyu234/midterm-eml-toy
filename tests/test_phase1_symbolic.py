import sympy as sp

from tree.coords import (
    EXP_OFFSET,
    LN_OFFSET,
    OUTPUT_COORD,
    layer_is_left,
    normalize_param,
    slot_bindings,
    slot_coords,
    x_slot_index,
)
from codec.node import decode_node
from symbolic.pipeline import build_value_tree, compute_node_exprs, eml, simplify_expr
from symbolic.tetration import compressed_exp, compressed_ln
from export.formulas import export_phase1_txt, run_phase1_default_output


def test_layer_is_left():
    assert layer_is_left(1, x_slot=0) is True
    assert layer_is_left(2, x_slot=0) is False
    assert layer_is_left(1, x_slot=1) is False
    assert layer_is_left(2, x_slot=1) is True


def test_slot_coords_n1():
    coords = slot_coords(1, x_slot=0)
    assert len(coords) == 2
    assert coords[0] == (0, -1)
    assert coords[1] == (1, 0)


def test_x_slot_index():
    assert x_slot_index(1, 0) == 1
    assert x_slot_index(1, 1) == 1


def test_normalize_param_identity():
    B = sp.Symbol('B')
    assert normalize_param(B, 0, 0) == B


def test_normalize_param_ln():
    B = sp.Symbol('B')
    assert normalize_param(B, 1, 0) == compressed_exp(1, B)


def test_normalize_param_exp():
    B = sp.Symbol('B')
    assert normalize_param(B, 0, -1) == compressed_ln(1, B)


def test_n_eml_zero_skipped():
    node = decode_node(0)
    assert node.n_eml == 0


def test_value_only_d_symbols():
    e = compute_node_exprs(1)
    assert sp.Symbol('x') not in e.value_expr.free_symbols
    assert all(str(s).startswith('D_') for s in e.value_expr.free_symbols)


def test_n_eml_one_x_slot_left():
    e = compute_node_exprs(0b0001)
    assert e.node.x_slot == 0
    assert e.node.n_eml == 1
    assert len(e.param_grads) == 2


def test_n_eml_one_x_slot_right():
    e = compute_node_exprs(0b1001)
    assert e.node.x_slot == 1
    assert len(e.param_grads) == 2


def test_param_count_matches_n_eml():
    for low in range(1, 8):
        e = compute_node_exprs(low)
        assert len(e.param_grads) == low + 1


def test_8bit_same_structure_different_reserved():
    e1 = compute_node_exprs(0x01)
    e2 = compute_node_exprs(0xF1)
    assert e1.value_expr == e2.value_expr
    assert e2.node.reserved == 0xF


def test_build_value_tree_n1_left():
    node = decode_node(1)
    bindings = slot_bindings(1, 0)
    tree = build_value_tree(node, bindings)
    assert tree == eml(bindings[1].normalized, bindings[0].normalized)


def test_gradient_wirtinger_dz_matches_diff():
    e = compute_node_exprs(2)
    for g in e.param_grads:
        expected = sp.diff(e.value_expr, g.param)
        assert simplify_expr(g.dz - expected) == 0


def test_simplify_log_exp_collapses_via_t():
    """log(exp(x)) → T(e,-1,T(e,1,x)) → T(e,0,x) → x after T merge + expand."""
    x = sp.Symbol('x')
    assert simplify_expr(sp.log(sp.exp(x))) == x


def test_all_low_nibble_nodes_have_exprs():
    for low in range(1, 16):
        e = compute_node_exprs(low)
        assert e.value_expr is not None


def test_export_phase1_txt(tmp_path):
    p = export_phase1_txt(tmp_path / 'out.txt')
    text = p.read_text(encoding='utf-8')
    assert 'byte 1' in text
    assert 'x_slot' in text


def test_run_phase1_default_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    p = run_phase1_default_output()
    assert p.exists()
