"""ln(T+T) workspace visibility (DESIGN §1.8, incomplete)."""
from research.ln_t_plus_t import ON_MAIN_PATH, STATUS, workspace_summary


def test_workspace_not_on_main_path():
    assert ON_MAIN_PATH is False
    assert STATUS.name == 'INCOMPLETE'


def test_workspace_summary_points_to_file():
    s = workspace_summary()
    assert 'ln(T+T)' in s
    assert 'simplify_expr: NO' in s
    assert 'symbolic/pipeline.py' in s
