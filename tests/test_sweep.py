"""16-node sweep and score suite tests."""
from codec.sweep import describe_node, iter_16_node_bytes, list_16_nodes


def test_16_nodes_count():
    assert len(list(iter_16_node_bytes())) == 16


def test_evaluable_count():
    assert len(list_16_nodes(evaluable_only=True)) == 14


def test_byte1_n_eml1_x_left():
    e = describe_node(1)
    assert e.n_eml == 1 and e.x_slot == 0 and e.evaluable


def test_byte9_n_eml1_x_right():
    e = describe_node(9)
    assert e.n_eml == 1 and e.x_slot == 1 and e.evaluable


def test_non_evaluable_n_eml0():
    e = describe_node(0)
    assert e.n_eml == 0 and not e.evaluable
