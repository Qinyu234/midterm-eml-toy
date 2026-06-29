"""Score suite smoke."""
from experiment.score_suite import format_score_table, run_score_for_target
from data.targets import get_benchmark_target


def test_score_smoke_sin_mlp_only():
    t = get_benchmark_target('sin')
    rows = run_score_for_target(t, methods=('mlp',), max_steps=15, eps=1.0)
    assert len(rows) == 1
    assert 'sin' in format_score_table(rows)


def test_score_smoke_sin_eml_two_nodes():
    from codec.sweep import describe_node

    t = get_benchmark_target('sin')
    nodes = [describe_node(1), describe_node(9)]
    rows = run_score_for_target(
        t, methods=('eml',), max_steps=10, eps=1.0, nodes=nodes,
    )
    assert len(rows) == 2
    assert rows[0].byte == 1
    assert rows[1].x_slot == 1
