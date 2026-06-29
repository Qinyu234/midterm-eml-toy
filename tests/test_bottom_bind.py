"""Bottom_d input binding: min |d|, center tie-break."""
from codec.node import decode_node, is_evaluable
from codec.sweep import describe_node, iter_16_node_bytes
from tree.coords import bottom_input_slot, input_slot_indices, slot_coords


def _slot_d(coord):
    return abs(coord[0] + coord[1])


def test_bottom_d_matches_min_abs_d_for_all_evaluable_nodes():
    for byte in iter_16_node_bytes():
        if not is_evaluable(byte):
            continue
        node = decode_node(byte)
        coords = slot_coords(node.n_eml, node.x_slot)
        bind = bottom_input_slot(node.n_eml, node.x_slot)
        d_vals = [_slot_d(coords[i]) for i in range(node.n_eml + 1)]
        min_d = min(d_vals)
        assert d_vals[bind] == min_d
        candidates = [i for i, d in enumerate(d_vals) if d == min_d]
        assert bind == min(candidates)


def test_describe_node_reports_bind_slot():
    entry = describe_node(1)
    assert entry.input_bind_slot == input_slot_indices(1, 0, 1, binding='bottom_d')[0]
    assert entry.x_leaf != entry.input_bind_slot or entry.n_eml == 1
