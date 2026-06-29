"""Log-sum gap threshold and T-domain ln-sum rules."""
import math

import numpy as np
import sympy as sp

from symbolic.logsum_gap import (
    DEFAULT_GAP_THRESHOLD,
    LogsumT,
    gap_threshold_from_epsilon,
    gap_threshold_for_dtype,
    ln_t_sum_symbolic,
    logsum_pair_symbolic,
    merge_t_exp_add,
    stabilize_ln_t_sum,
    t_preimage_symbolic,
)
from symbolic.tetration import E_TAG, Tetration, expand_t_to_log_exp, simplify_in_t
from primitives.logsum_gap import ln_t_sum_numpy, logsum_pair_numpy


def test_gap_threshold_float64():
    g = gap_threshold_from_epsilon(1e-16)
    assert g == 37
    assert gap_threshold_for_dtype(np.float64) >= 37


def test_merge_t_exp_add():
    a, b = sp.symbols('a b')
    expr = Tetration(E_TAG, 1, a + b)
    out = merge_t_exp_add(expr)
    assert out == Tetration(E_TAG, 1, a) * Tetration(E_TAG, 1, b)


def test_stabilize_ln_t_sum_n1():
    a, b = sp.symbols('a b')
    expr = Tetration(E_TAG, -1, Tetration(E_TAG, 1, a) + Tetration(E_TAG, 1, b))
    out = stabilize_ln_t_sum(expr)
    assert isinstance(out, LogsumT)
    assert out.args == (1, a, b)


def test_stabilize_ln_t_sum_n2():
    a, b = sp.symbols('a b')
    expr = Tetration(E_TAG, -1, Tetration(E_TAG, 2, a) + Tetration(E_TAG, 2, b))
    out = stabilize_ln_t_sum(expr)
    assert isinstance(out, LogsumT)
    assert out.args[0] == 2


def test_expand_logsum_no_raw_exp_sum():
    a, b = sp.symbols('a b')
    expr = LogsumT(1, a, b)
    expanded = expand_t_to_log_exp(expr)
    assert not expanded.has(sp.Add, sp.exp(a), sp.exp(b)) or expanded.has(sp.Max)


def test_logsum_pair_numeric_matches_log():
    a = np.array([1.0, 10.0, 100.0])
    b = np.array([2.0, 5.0, 80.0])
    got = logsum_pair_numpy(a, b)
    ref = np.log(np.exp(a) + np.exp(b))
    assert np.allclose(got.real, ref.real, rtol=1e-10)


def test_logsum_pair_large_gap_uses_max():
    a, b = 100.0, 50.0
    got = logsum_pair_numpy(np.array(a), np.array(b))
    assert abs(got - 100.0) < 1e-10


def test_ln_t_sum_n2_finite():
    a = np.array([1.0 + 0.1j])
    b = np.array([2.0 - 0.2j])
    got = ln_t_sum_numpy(2, a, b)
    assert np.isfinite(got)


def test_simplify_in_t_merges_ln_sum():
    a, b = sp.symbols('a b')
    raw = Tetration(E_TAG, -1, Tetration(E_TAG, 1, a) + Tetration(E_TAG, 1, b))
    opt = simplify_in_t(raw)
    assert isinstance(opt, LogsumT)
