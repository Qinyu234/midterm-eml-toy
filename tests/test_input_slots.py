"""Feynman input → leaf D slot mapping (bottom |d| first)."""
from tree.coords import input_slot_indices, learnable_slot_indices


def test_n1_one_var_picks_smallest_abs_d():
    # both leaves have |d|=1; tie → lower slot index
    assert input_slot_indices(1, 0, 1) == (0,)
    assert input_slot_indices(1, 1, 1) == (0,)


def test_n1_two_vars_both_leaves():
    assert input_slot_indices(1, 0, 2) == (0, 1)


def test_n2_two_vars_prefers_d_zero_then_negative():
    # coords: (1,-1)d=0, (2,0)d=2, (0,-1)d=-1
    assert input_slot_indices(2, 0, 2) == (0, 2)


def test_learnable_complement():
    ins = input_slot_indices(2, 0, 1)
    learn = learnable_slot_indices(2, 0, 1)
    assert set(ins) | set(learn) == {0, 1, 2}
    assert set(ins) & set(learn) == set()
