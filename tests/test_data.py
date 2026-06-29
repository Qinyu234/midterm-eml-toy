import numpy as np

from data.feynman import (
    DataBatch,
    apply_function,
    get_feynman_function,
    grid_sample,
    list_feynman_functions,
    random_sample,
)


def test_list_feynman_functions():
    fns = list_feynman_functions()
    assert len(fns) >= 5


def test_get_feynman_function():
    fn = get_feynman_function('I.6.2')
    assert fn.n_vars == 1


def test_grid_sample():
    X = grid_sample(2, n_per_dim=5)
    assert X.shape == (25, 2)
    assert X.min() >= -3 and X.max() <= 3


def test_random_sample():
    X = random_sample(1, 50, seed=42)
    assert X.shape == (50, 1)


def test_apply_function():
    fn = get_feynman_function('I.6.2')
    X = grid_sample(1, n_per_dim=3)
    y = apply_function(fn, X)
    assert y.shape == (3,)


def test_function_target():
    fn = get_feynman_function('I.8.4')
    X = grid_sample(1, n_per_dim=4)
    y = apply_function(fn, X)
    expected = X[:, 0] ** 2 / 2
    np.testing.assert_allclose(y, expected, rtol=1e-10)


def test_data_batch():
    fn = get_feynman_function('I.6.2')
    X = grid_sample(1, n_per_dim=4)
    y = apply_function(fn, X)
    batch = DataBatch(X=X, y=y, function=fn)
    assert batch.X.shape[0] == batch.y.shape[0]
