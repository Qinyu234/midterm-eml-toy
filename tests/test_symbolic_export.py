import pytest
from pathlib import Path

from symbolic.pipeline import compute_node_exprs
from export.formulas import format_node_formulas, write_all_formulas, write_per_node_files


def test_symbolic_export_module_has_no_lambdify():
    src = (Path(__file__).resolve().parents[1] / 'export' / 'formulas.py').read_text(
        encoding='utf-8'
    )
    assert 'import numpy' not in src
    assert 'sp.lambdify' not in src
    assert 'from compile' not in src


def test_format_contains_value_and_grad():
    exprs = compute_node_exprs(1)
    text = format_node_formulas(exprs, pretty=False)
    assert 'value_expr' in text
    assert 'grad_expr' in text
    assert 'c_1' in text


def test_write_all_formulas(tmp_path: Path):
    out = tmp_path / 'all.txt'
    n = write_all_formulas(out, bytes=[1, 2])
    assert n == 2
    body = out.read_text(encoding='utf-8')
    assert 'value_expr' in body
    assert 'byte 1' in body


def test_write_per_node_files(tmp_path: Path):
    n = write_per_node_files(tmp_path, pretty=False)
    assert n > 0
    assert (tmp_path / 'node_001.txt').exists()
    assert 'value_expr' in (tmp_path / 'node_001.txt').read_text(encoding='utf-8')


def test_cli_blocks_lambdify_if_preloaded(monkeypatch):
    import sys
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        'export_symbols_cli',
        Path(__file__).resolve().parents[1] / 'scripts' / 'export_symbols.py',
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    sys.modules['compile.lambdify_exprs'] = object()  # type: ignore
    with pytest.raises(RuntimeError, match='symbolic-only'):
        mod._assert_symbolic_only_deps()
    del sys.modules['compile.lambdify_exprs']
