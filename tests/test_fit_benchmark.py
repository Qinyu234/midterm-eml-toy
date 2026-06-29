"""End-to-end fit benchmark smoke."""
from experiment.fit_benchmark import run_comparison, summarize_table
from data.targets import get_benchmark_target


def test_smoke_sin_mlp():
    t = get_benchmark_target('sin')
    results = run_comparison(t, methods=('mlp',), max_steps=30, eps=1.0)
    assert len(results) == 1
    assert results[0].method == 'mlp'
    assert 'sin' in summarize_table(results)


def test_smoke_sin_eml_small_bytes():
    t = get_benchmark_target('sin')
    r = run_comparison(
        t, methods=('eml',), max_steps=20, eps=1.0, bytes=[1, 2, 3],
    )
    assert len(r) == 1
    assert r[0].method == 'eml'
    assert 'byte' in r[0].meta
