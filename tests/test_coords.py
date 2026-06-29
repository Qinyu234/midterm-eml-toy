"""C leaf coordinate trace: value at (0,0), top→bottom, left x++, right y--."""
from tree.coords import LEFT_OFFSET, OUTPUT_COORD, RIGHT_OFFSET, slot_coords


def test_value_ref_is_origin():
    for n in range(1, 5):
        for x_slot in (0, 1):
            coords = slot_coords(n, x_slot)
            assert len(coords) == n + 1


def test_n1_x_slot_0():
    """eml(x,C): x left (1,0), C right (0,-1) from value ref."""
    assert slot_coords(1, 0) == ((0, -1), (1, 0))


def test_n1_x_slot_1():
    """eml(C,x): C left (1,0), x right (0,-1)."""
    assert slot_coords(1, 1) == ((1, 0), (0, -1))


def test_n2_x_slot_0():
    """layer2 right eml(inner,C2), layer1 left eml(x,C1)."""
    assert slot_coords(2, 0) == ((1, -1), (2, 0), (0, -1))


def test_n2_x_slot_1():
    assert slot_coords(2, 1) == ((1, -1), (0, -2), (1, 0))


def test_offsets_match_rule():
    assert LEFT_OFFSET == (1, 0)
    assert RIGHT_OFFSET == (0, -1)
    assert OUTPUT_COORD == (0, 0)
