"""TES metric tests."""
from experiment.tes import (
    fit_quality,
    steps_to_threshold,
    total_compute,
    tower_efficiency_score,
)


def test_fit_quality_monotone():
    assert fit_quality(0.0) > fit_quality(0.1)


def test_steps_to_threshold():
    curve = [1.0, 0.5, 0.05, 0.01]
    assert steps_to_threshold(curve, 0.1) == 3


def test_tes_better_rmse_and_fewer_steps_higher():
    q = fit_quality(0.01)
    c_fast = total_compute('mlp', 100, n_params=10, steps_to_threshold=20, max_steps=100)
    c_slow = total_compute('mlp', 100, n_params=10, steps_to_threshold=80, max_steps=100)
    tes_fast = tower_efficiency_score(0.01, c_fast)
    tes_slow = tower_efficiency_score(0.01, c_slow)
    assert tes_fast > tes_slow
