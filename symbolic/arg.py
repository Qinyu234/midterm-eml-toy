"""
Complex Arg symbolic function: rational fdiff, no 2π wrap in symbolic layer.
复数 Arg：有理式导数；符号层不 2π 取余。
"""
from __future__ import annotations

import sympy as sp


class ComplexArg(sp.Function):
    """Arg(x+I*y); rational partials w.r.t. Re/Im / 对实部虚部有理式导数。"""

    @classmethod
    def eval(cls, w):  # type: ignore[override]
        return None

    def fdiff(self, argindex=1):
        if argindex != 1:
            raise ArgumentIndexError(self, argindex)
        w = self.args[0]
        x, y = sp.re(w), sp.im(w)
        denom = x**2 + y**2
        return sp.I * x / denom - y / denom


def complex_log_symbolic(w: sp.Expr) -> sp.Expr:
    """ln|w| + i·Arg(w) for symbolic layer / 符号层复数对数。"""
    return sp.log(sp.Abs(w)) + sp.I * ComplexArg(w)
