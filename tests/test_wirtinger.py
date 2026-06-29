"""Wirtinger Jacobian tests."""
import sympy as sp

from symbolic.wirtinger import WirtingerJacobian, format_abi, wirtinger_grad


def test_holomorphic_z():
    z = sp.Symbol('z', complex=True)
    w = wirtinger_grad(z**2, z)
    assert sp.simplify(w.dz - 2 * z) == 0
    assert sp.simplify(w.dz_bar) == 0


def test_conjugate_bar():
    z = sp.Symbol('z', complex=True)
    w = wirtinger_grad(sp.conjugate(z), z)
    assert sp.simplify(w.dz) == 0
    assert sp.simplify(w.dz_bar - 1) == 0


def test_format_abi():
    z = sp.Symbol('z', complex=True)
    assert format_abi(1 + 2 * sp.I) == '1+2*I'
    assert format_abi(sp.I) == 'I'


def test_abs_squared():
    z = sp.Symbol('z', complex=True)
    w = wirtinger_grad(z * sp.conjugate(z), z)
    assert sp.simplify(w.dz - sp.conjugate(z)) == 0
    assert sp.simplify(w.dz_bar - z) == 0
