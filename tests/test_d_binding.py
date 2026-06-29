"""Multi-D Feynman binding on CompiledEMLModel."""
import numpy as np
import torch

from data.feynman import apply_function, get_feynman_function, grid_sample
from numerics.d_binding import bind_d_args
from numerics.symbolic_model import CompiledEMLModel
from tree.coords import input_slot_indices


def test_bind_two_vars_byte2():
    X = np.array([[1.0, 2.0], [1.5, 2.5]])
    d = bind_d_args(2, 0, 2, X, [0.1, 0.2, 0.3])
    ins = input_slot_indices(2, 0, 2)
    assert d[ins[0]][0] == 1.0 + 0j
    assert d[ins[1]][0] == 2.0 + 0j


def test_model_two_var_forward():
    fn = get_feynman_function('I.12.1')
    X = grid_sample(2, n_per_dim=3)
    y = apply_function(fn, X)
    model = CompiledEMLModel(2, n_vars=2)
    out = model.forward(torch.tensor(X, dtype=torch.float64))
    assert out.shape == (len(X),)
    assert np.all(np.isfinite(out.detach().numpy()))


def test_model_two_var_training_step():
    fn = get_feynman_function('I.12.1')
    X = grid_sample(2, n_per_dim=4)
    y = apply_function(fn, X)
    model = CompiledEMLModel(2, n_vars=2)
    loss, gd, ga, gb = model.loss_and_symbolic_grads(X, y)
    assert np.isfinite(loss)
    model.apply_symbolic_grads(gd, ga, gb, lr=0.01)
