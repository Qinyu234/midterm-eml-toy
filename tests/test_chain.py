import numpy as np
import torch

from numerics.symbolic_model import CompiledEMLModel, EMLModel


def test_chain_forward_shape():
    model = CompiledEMLModel(1)
    x = torch.linspace(-2, 2, 10, dtype=torch.float64)
    out = model(x)
    assert out.shape == x.shape


def test_chain_no_nan():
    model = EMLModel(1)
    x = torch.linspace(-3, 3, 20, dtype=torch.float64)
    y = model(x)
    assert torch.all(torch.isfinite(y))


def test_chain_trainable():
    model = CompiledEMLModel(1)
    x = np.linspace(-1, 1, 8)
    y = np.zeros_like(x)
    losses = []
    for _ in range(20):
        loss, gc, ga, gb = model.loss_and_symbolic_grads(x, y)
        if not np.isfinite(loss):
            break
        model.apply_symbolic_grads(gc, ga, gb, lr=0.05)
        losses.append(loss)
    assert len(losses) >= 2
    assert losses[-1] <= losses[0]
