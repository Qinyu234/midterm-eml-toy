"""Tiny MLP column probe smoke."""
import numpy as np

from data.tiny_mlp import TinyPublicMLP
from experiment.mlp_column_probe import run_mlp_column_probe


def test_tiny_mlp_forward_finite():
    ref = TinyPublicMLP()
    y = ref.forward(np.linspace(-1, 1, 10))
    assert np.all(np.isfinite(y))


def test_column_w1_shape():
    ref = TinyPublicMLP()
    x, y, label = ref.column_samples('w1')
    assert len(x) == ref.hidden
    assert len(y) == ref.hidden
    assert 'w1' in label


def test_probe_neuron_act_smoke():
    r = run_mlp_column_probe(kind='neuron_act', neuron=0, max_steps=20, eps=1.0)
    assert np.isfinite(r.best_in_rmse)
    assert r.best_byte >= 1
