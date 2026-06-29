"""FORMULA_WORKPLACE export tests."""
from pathlib import Path

from export.workplace import iter_4bit_node_bytes, write_formula_workplace


def test_iter_4bit_node_bytes():
    assert list(iter_4bit_node_bytes()) == list(range(1, 16))


def test_write_formula_workplace(tmp_path: Path):
    out = tmp_path / 'FORMULA_WORKPLACE.txt'
    n = write_formula_workplace(out, bytes=[1])
    assert n == 1
    text = out.read_text(encoding='utf-8')
    assert 'value [before]:' in text
    assert 'value [after]:' in text
    assert 'D_gradient:' in text
    assert 'value [after]:' in text
    grad_section = text.split('D_gradient:')[1]
    assert 'd/D_1/∂z' in grad_section
    assert 'd/D_2/∂z' in grad_section
    assert '∂z̄' in grad_section
    assert 'c_1' not in text.split('value [after]')[1].split('D_gradient')[0]
