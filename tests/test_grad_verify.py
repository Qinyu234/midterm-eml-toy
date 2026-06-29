"""Gradient FD verification on leaf D."""
from eval.grad_verify import verify_gradients


def test_byte1_grad_fd_small():
    r = verify_gradients(1, n_vars=1, n_points=3, lo=0.8, hi=1.2)
    assert r.finite_rate >= 1.0
    assert r.max_fd_error < 0.05


def test_byte1_two_input_slots_no_learnable_still_finite():
    r = verify_gradients(1, n_vars=2, n_points=2, lo=0.9, hi=1.1)
    assert r.finite_rate >= 1.0
