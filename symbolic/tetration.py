"""
Tetration T(e,n,z) and compressed_exp / compressed_ln aliases.
Tetration 与 compressed_exp / compressed_ln 别名。

Symbolic layer uses standard exp/ln; fdiff hard-coded.
符号层标准 exp/ln；fdiff 硬编码。
"""
from __future__ import annotations

import sympy as sp

E_TAG = sp.Symbol('e', positive=True)


class Tetration(sp.Function):
    """T(e,n,z): n>0 n-fold exp; n=0 identity; n<0 |n|-fold ln."""

    @classmethod
    def eval(cls, e, n, z):  # type: ignore[override]
        # Keep unevaluated in symbolic layer; expand via expand_t_to_log_exp only.
        return None

    def fdiff(self, argindex=3):
        if argindex != 3:
            raise ArgumentIndexError(self, argindex)
        n = self.args[1]
        z = self.args[2]
        try:
            n_int = int(n)
        except (TypeError, ValueError):
            return sp.Derivative(self, z)
        if n_int == 0:
            return sp.S.One
        if n_int > 0:
            factors = [Tetration(E_TAG, i, z) for i in range(1, n_int + 1)]
            return sp.Mul(*factors)
        m = -n_int
        factors = [sp.Pow(Tetration(E_TAG, -(i - 1), z), -1) for i in range(1, m + 1)]
        return sp.Mul(*factors)


def compressed_exp(n: int, z: sp.Expr) -> sp.Expr:
    return Tetration(E_TAG, n, z)


def compressed_ln(n: int, z: sp.Expr) -> sp.Expr:
    return Tetration(E_TAG, -n, z)


def _exp_depth(expr: sp.Expr) -> tuple[int, sp.Expr] | None:
    if not isinstance(expr, sp.exp):
        return None
    inner = expr.args[0]
    sub = _exp_depth(inner)
    if sub is None:
        return (1, inner)
    return (sub[0] + 1, sub[1])


def _log_depth(expr: sp.Expr) -> tuple[int, sp.Expr] | None:
    if not (expr.func == sp.log or isinstance(expr, sp.log)):
        return None
    inner = expr.args[0]
    sub = _log_depth(inner)
    if sub is None:
        return (1, inner)
    return (sub[0] + 1, sub[1])


def merge_tetration(expr: sp.Expr) -> sp.Expr:
    """Merge nested exp/ln chains into Tetration / 合并连续 exp/ln。"""
    if expr.is_Atom:
        return expr

    exp_info = _exp_depth(expr)
    if exp_info is not None:
        n, base = exp_info
        return compressed_exp(n, merge_tetration(base))

    log_info = _log_depth(expr)
    if log_info is not None:
        n, base = log_info
        return compressed_ln(n, merge_tetration(base))

    return expr.func(*[merge_tetration(a) for a in expr.args])


to_t_form = merge_tetration


def expand_t_to_log_exp(expr: sp.Expr) -> sp.Expr:
    """Expand T(e,n,z) → nested exp/ln for output / 交付前 T 展开。"""
    from symbolic.logsum_gap import LogsumT, ln_t_sum_symbolic

    if isinstance(expr, LogsumT):
        n, a, b = expr.args
        try:
            n_i = int(n)
        except (TypeError, ValueError):
            return LogsumT(n, a, b)
        return ln_t_sum_symbolic(
            n_i,
            expand_t_to_log_exp(a),
            expand_t_to_log_exp(b),
        )
    if isinstance(expr, Tetration):
        e, n, z = expr.args
        z = expand_t_to_log_exp(z)
        try:
            n_int = int(n)
        except (TypeError, ValueError):
            return Tetration(e, n, z)
        if n_int == 0:
            return z
        out = z
        if n_int > 0:
            for _ in range(n_int):
                out = sp.exp(out, evaluate=False)
            return out
        for _ in range(-n_int):
            out = sp.log(out, evaluate=False)
        return out
    if not expr.args:
        return expr
    return expr.func(*[expand_t_to_log_exp(a) for a in expr.args])


def format_t_expr(expr: sp.Expr) -> str:
    """Plain T(e,n,z) string (no pretty-print)."""
    return str(expr).replace('Tetration', 'T')


def merge_opposite_tetration(expr: sp.Expr) -> sp.Expr:
    """T(e,m,T(e,n,z)) → T(e,m+n,z) only when m*n<0; T(e,0,z) → z."""
    if isinstance(expr, Tetration):
        e, n, z = expr.args
        z = merge_opposite_tetration(z)
        if isinstance(z, Tetration) and z.args[0] == e:
            m = z.args[1]
            inner = z.args[2]
            try:
                n_i, m_i = int(n), int(m)
            except (TypeError, ValueError):
                return Tetration(e, n, z)
            if n_i * m_i < 0:
                total = n_i + m_i
                if total == 0:
                    return merge_opposite_tetration(inner)
                return merge_opposite_tetration(Tetration(e, total, inner))
        try:
            if int(n) == 0:
                return merge_opposite_tetration(z)
        except (TypeError, ValueError):
            pass
        return Tetration(e, n, z)
    if not expr.args:
        return expr
    return expr.func(*[merge_opposite_tetration(a) for a in expr.args])


def simplify_in_t(expr: sp.Expr) -> sp.Expr:
    """to_t_form + T merges + ln-sum gap stabilization until fixed point."""
    from symbolic.logsum_gap import merge_t_exp_add, stabilize_ln_t_sum

    expr = to_t_form(expr)
    expr = merge_t_exp_add(expr)
    prev = None
    while prev != expr:
        prev = expr
        expr = merge_opposite_tetration(expr)
        expr = stabilize_ln_t_sum(expr)
        expr = merge_t_exp_add(expr)
    return expr

