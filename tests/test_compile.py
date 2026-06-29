import numpy as np

from compile.lambdify_exprs import compile_node_exprs
from symbolic.pipeline import compute_node_exprs


def test_grad_wirtinger_compiled_finite():
    e = compute_node_exprs(1)
    value_fn, grad_fns = compile_node_exprs(e)
    d1, d2 = 0.5, 1.0
    z = value_fn(d1, d2)
    dz1 = grad_fns[0].dz_fn(d1, d2)
    dzb1 = grad_fns[0].dz_bar_fn(d1, d2)
    assert np.isfinite(z)
    assert np.isfinite(dz1)
    assert np.isfinite(dzb1)
    assert len(grad_fns) == 2
