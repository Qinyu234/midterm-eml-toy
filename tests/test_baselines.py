"""MLP and Fourier baseline tests."""
import numpy as np

from baselines.fourier import fit_fourier
from baselines.mlp import fit_mlp
from data.targets import BenchmarkTarget


def _x_squared(x: np.ndarray) -> np.ndarray:
    return x ** 2


def test_mlp_fits_x_squared():
    t = BenchmarkTarget('x2', -1.0, 1.0, ((-2.0, -1.0), (1.0, 2.0)), _x_squared)
    x, y = t.sample_train(n=80, seed=0)
    xo, yo = t.sample_ood(n=20, seed=1)
    r0 = fit_mlp(t, x, y, xo, yo, hidden=4, max_steps=30, eps=1e-2, optimizer='adamw')
    r1 = fit_mlp(
        t, x, y, xo, yo, hidden=32, max_steps=200, eps=1e-2, optimizer='adamw',
    )
    assert r1.in_rmse <= r0.in_rmse
    assert r1.meta['optimizer'] == 'adamw'


def test_mlp_bfgs_runs():
    t = BenchmarkTarget('x2', -1.0, 1.0, ((-2.0, -1.0), (1.0, 2.0)), _x_squared)
    x, y = t.sample_train(n=40, seed=0)
    xo, yo = t.sample_ood(n=10, seed=1)
    r = fit_mlp(t, x, y, xo, yo, hidden=4, max_steps=50, eps=1e-1, optimizer='bfgs')
    assert np.isfinite(r.in_rmse)
    assert r.meta['optimizer'] == 'bfgs'


def test_fourier_fits_x_squared():
    t = BenchmarkTarget('x2', -1.0, 1.0, ((-2.0, -1.0), (1.0, 2.0)), _x_squared)
    x, y = t.sample_train(n=80, seed=0)
    xo, yo = t.sample_ood(n=20, seed=1)
    r = fit_fourier(t, x, y, xo, yo, n_freq=8, max_steps=200, eps=1e-1)
    assert np.isfinite(r.in_rmse)
    assert r.in_rmse < 0.5
