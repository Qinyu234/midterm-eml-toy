import numpy as np
import torch

from compile.lambdify_exprs import compile_node_exprs
from primitives.ops import eml_numpy, eml_torch, safe_ln_numpy, soft_exp_numpy
from symbolic.pipeline import compute_node_exprs


def test_safe_ln_no_nan():
    w = np.array([-1.0 + 0.5j, 0.0 + 0.0j, 1e-20 + 1e-20j], dtype=np.complex128)
    out = safe_ln_numpy(w)
    assert np.all(np.isfinite(out))


def test_safe_ln_log1p_near_one():
    """log1p(w-1+εi) ≡ ln(w+εi); stable near w≈1 / 恒等且 w≈1 更稳。"""
    w = np.array([1.0 + 1e-14 + 1e-15j], dtype=np.complex128)
    via_log1p = safe_ln_numpy(w)[0]
    via_log = np.log(w + 1j * 1e-8)[0]
    assert np.isfinite(via_log1p)
    assert abs(via_log1p - via_log) < 1e-6


def test_soft_exp_no_nan():
    x = np.array([-100.0, 0.0, 100.0], dtype=np.complex128)
    out = soft_exp_numpy(x)
    assert np.all(np.isfinite(out.real))
    assert np.all(out.real >= 0)


def test_soft_exp_c1():
    x = np.linspace(-25, 25, 500)
    g = soft_exp_numpy(x.astype(np.complex128)).real
    dg = np.gradient(g, x)
    d2g = np.gradient(dg, x)
    assert np.all(np.isfinite(d2g))


def test_eml_op_shape():
    z = torch.tensor([1.0 + 0.5j, -2.0], dtype=torch.complex128)
    w = torch.tensor([2.0 + 0.1j, 0.5 + 0.0j], dtype=torch.complex128)
    out = eml_torch(z, w)
    assert out.shape == z.shape


def test_compile_value_and_symbolic_grad():
    exprs = compute_node_exprs(1)
    value_fn, grad_fns = compile_node_exprs(exprs)
    d1, d2 = 0.5, 1.0
    z = value_fn(d1, d2)
    gz1 = grad_fns[0].dz_fn(d1, d2)
    assert np.all(np.isfinite(z))
    assert np.all(np.isfinite(gz1))
    eps = 1e-7
    z_p = value_fn(d1 + eps, d2)
    fd = (z_p - z) / eps
    assert np.allclose(np.real(gz1), np.real(fd), rtol=0.1, atol=0.5)
