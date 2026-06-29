from codec.node import NodeCode, decode_node, encode_node, is_evaluable, iter_node_bytes


def test_x_slot_is_bit3():
    assert decode_node(0b0000).x_slot == 0
    assert decode_node(0b1000).x_slot == 1
    assert decode_node(0b1001).x_slot == 1


def test_n_eml_is_lower_3_bits():
    assert decode_node(0b0111).n_eml == 7
    assert decode_node(0b0000).n_eml == 0


def test_8bit_low_nibble_only():
    """Structure from low 4 bits; high 4 bits are reserved / 结构由低 4 bit 决定。"""
    a = decode_node(0x01)
    b = decode_node(0xF1)
    assert a.n_eml == b.n_eml == 1
    assert a.x_slot == b.x_slot == 0
    assert a.reserved == 0
    assert b.reserved == 0xF


def test_n_params_equals_n_eml():
    for b in range(256):
        n = decode_node(b)
        assert n.n_params == n.n_eml


def test_properties():
    n = NodeCode(x_slot=1, n_eml=3, reserved=2)
    assert n.n_params == 3
    assert n.to_byte() == (2 << 4) | 0b1011


def test_roundtrip_8bit():
    for reserved in range(16):
        for x_slot in (0, 1):
            for n_eml in range(8):
                nc = NodeCode(x_slot, n_eml, reserved)
                assert decode_node(encode_node(nc)) == nc


def test_roundtrip_low_nibble():
    for b in range(16):
        assert decode_node(b).low_nibble == b


def test_iter_and_evaluable():
    assert len(list(iter_node_bytes())) == 256
    assert not is_evaluable(0)
    assert not is_evaluable(0x10)
    assert is_evaluable(1)


def test_x_slot_and_n_eml():
    left = decode_node(0b0001)
    right = decode_node(0b1001)
    assert left.x_slot == 0 and left.n_eml == 1
    assert right.x_slot == 1 and right.n_eml == 1
