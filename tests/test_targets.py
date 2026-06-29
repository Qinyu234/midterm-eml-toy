"""Benchmark target tests."""
import numpy as np

from data.targets import get_benchmark_target, list_benchmark_targets, resolve_targets


def test_list_fifteen_targets():
    assert len(list_benchmark_targets()) == 15


def test_resolve_all():
    assert len(resolve_targets('all')) == 15


def test_resolve_rule_group():
    rule = resolve_targets('rule')
    assert len(rule) == 8
    assert all(t.category == 'rule' for t in rule)


def test_resolve_classic_group():
    assert len(resolve_targets('classic')) == 7


def test_sin_finite_on_train():
    t = get_benchmark_target('sin')
    x, y = t.sample_train(n=50, seed=0)
    assert len(x) == 50
    assert np.all(np.isfinite(y))


def test_log_train_domain_positive():
    t = get_benchmark_target('log')
    x, y = t.sample_train(n=20, seed=0)
    assert np.all(x > 0)
    assert np.all(np.isfinite(y))


def test_ood_outside_train():
    t = get_benchmark_target('sin')
    x, _ = t.sample_ood(n=40, seed=0)
    assert np.all(x < -2) or np.all(x > 2) or (np.any(x < -2) and np.any(x > 2))
