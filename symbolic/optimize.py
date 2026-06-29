"""
Symbolic rewrite rules for simplify_expr (DESIGN §1.2).
符号化简规则：Tetration 嵌套、exp 比值稳定、exp/ln 代数折叠。
"""
from __future__ import annotations

import sympy as sp

from symbolic.logsum_gap import merge_t_exp_add, stabilize_ln_t_sum
from symbolic.tetration import Tetration


def log_div_exponent(a: sp.Expr, b: sp.Expr) -> sp.Expr:
    """a ./ exp(b) := a − b — divide exp-values ⇔ subtract exponents."""
    return a - b


def log_div_value(a: sp.Expr, b: sp.Expr) -> sp.Expr:
    """a ./ b := a − ln(b) — divide by positive value b in exponent chart."""
    return a - sp.log(b)


def _try_fold_coeff_into_exp(expr: sp.Expr) -> sp.Expr | None:
    """D*exp(z) → exp(ln(D)+z)."""
    if not isinstance(expr, sp.Mul):
        return None
    exp_terms = [a for a in expr.args if isinstance(a, sp.exp)]
    if len(exp_terms) != 1:
        return None
    coeff = sp.Mul(*[a for a in expr.args if a not in exp_terms])
    if coeff == 1:
        return None
    z = exp_terms[0].args[0]
    return sp.exp(sp.log(coeff, evaluate=False) + z, evaluate=False)


def _is_folded_exp_form(expr: sp.Expr) -> bool:
    """exp(ln(D)+z) — keep folded; sympy may rewrite back to D*exp(z)."""
    if not isinstance(expr, sp.exp):
        return False
    inner = expr.args[0]
    return inner.is_Add and any(a.func == sp.log for a in inner.args)


def fold_coeff_into_exp(expr: sp.Expr) -> sp.Expr:
    """D*exp(z) → exp(ln(D)+z)."""
    if _is_folded_exp_form(expr):
        return expr
    if expr.is_Atom:
        return expr
    expr = expr.func(*[fold_coeff_into_exp(a) for a in expr.args])
    if _is_folded_exp_form(expr):
        return expr
    rewritten = _try_fold_coeff_into_exp(expr)
    if rewritten is not None:
        return rewritten
    return expr


def _try_expand_ln_exp_quotient(arg: sp.Expr) -> sp.Expr | None:
    """ln(exp(z)/D) → z ./ D; ln(exp(z)/exp(d)) → z ./ exp(d) = z − d."""
    numer, denom = arg.as_numer_denom()
    if not isinstance(numer, sp.exp):
        return None
    z = numer.args[0]
    if isinstance(denom, sp.exp):
        return log_div_exponent(z, denom.args[0])
    if denom == 1:
        return z
    return log_div_value(z, denom)


def _subtrahend_from_add_term(term: sp.Expr) -> sp.Expr | None:
    """Extract positive subtrahend from (-D) or plain D when used as exp(z) − D."""
    if term.is_Mul and term.could_extract_minus_sign():
        coeff, rest = term.as_coeff_Mul()
        if coeff == -1:
            return rest
        return None
    if term.is_positive or (term.is_real and not term.is_negative):
        return term
    return None


def _try_expand_ln_exp_difference(arg: sp.Expr) -> sp.Expr | None:
    """
    ln(exp(z) − D) → z + ln(1 − exp(ln(D)−z)); inner exp exponent ≤ 0 when z dominates.

    When D = exp(d): ln(exp(z) − exp(d)) → z + ln(1 − exp(d−z)).
    """
    if not isinstance(arg, sp.Add) or len(arg.args) != 2:
        return None
    exp_terms = [a for a in arg.args if isinstance(a, sp.exp)]
    if len(exp_terms) != 1:
        return None
    z = exp_terms[0].args[0]
    others = [a for a in arg.args if a not in exp_terms]
    if len(others) != 1:
        return None
    sub = _subtrahend_from_add_term(others[0])
    if sub is None:
        return None
    if isinstance(sub, sp.exp):
        shift = sub.args[0] - z
    else:
        shift = sp.log(sub) - z
    if shift.has(sp.Max):
        return None
    return z + sp.log(1 - sp.exp(shift, evaluate=False), evaluate=False)


def expand_ln_exp_algebra(expr: sp.Expr) -> sp.Expr:
    """
    ln(exp(z)/D) = z ./ D; ln(exp(z)/exp(d)) = z ./ exp(d) = z − d.
    ln(exp(z) − D) = z + ln(1 − exp(ln(D)−z)) (stable difference; not quotient).
    """
    if expr.is_Atom:
        return expr

    expr = expr.func(*[expand_ln_exp_algebra(a) for a in expr.args])

    if expr.func == sp.log:
        arg = expr.args[0]
        quotient = _try_expand_ln_exp_quotient(arg)
        if quotient is not None:
            return expand_ln_exp_algebra(quotient)
        difference = _try_expand_ln_exp_difference(arg)
        if difference is not None:
            return expand_ln_exp_algebra(difference)
    return expr


def merge_exp_products(expr: sp.Expr) -> sp.Expr:
    """exp(a)*exp(b)*… → exp(a+b+…); includes exp(D_1)*exp(D_2) = exp(D_1+D_2)."""
    if expr.is_Atom:
        return expr

    expr = expr.func(*[merge_exp_products(a) for a in expr.args])

    if not isinstance(expr, sp.Mul):
        return expr

    exponents: list[sp.Expr] = []
    rest: list[sp.Expr] = []
    for factor in expr.args:
        if isinstance(factor, sp.exp):
            exponents.append(factor.args[0])
        else:
            rest.append(factor)

    if len(exponents) < 2:
        return expr

    merged = sp.exp(sp.Add(*exponents, evaluate=False), evaluate=False)
    if not rest:
        return merged
    return sp.Mul(*rest, merged, evaluate=False)


def merge_nested_tetration(expr: sp.Expr) -> sp.Expr:
    """T(e,n,T(e,m,z)) → T(e,n+m,z) when base tag e matches."""
    if isinstance(expr, Tetration):
        e, n, z = expr.args
        z = merge_nested_tetration(z)
        if isinstance(z, Tetration) and z.args[0] == e:
            m = z.args[1]
            inner = merge_nested_tetration(z.args[2])
            try:
                n_i, m_i = int(n), int(m)
            except (TypeError, ValueError):
                return Tetration(e, n, z)
            return Tetration(e, n_i + m_i, inner)
        return Tetration(e, n, z)
    if not expr.args:
        return expr
    return expr.func(*[merge_nested_tetration(a) for a in expr.args])


def _rewrite_exp_over_exp_sum(numer: sp.Expr, denom: sp.Expr) -> sp.Expr | None:
    """exp(a) / Σ exp(bᵢ) → exp(a−M) / Σ exp(bᵢ−M), M = max exponents; all shifts ≤ 0."""
    if not isinstance(numer, sp.exp):
        return None
    if not isinstance(denom, sp.Add):
        return None

    exp_exponents: list[sp.Expr] = []
    other_terms: list[sp.Expr] = []
    for term in denom.args:
        if isinstance(term, sp.exp):
            exp_exponents.append(term.args[0])
        else:
            other_terms.append(term)
    if not exp_exponents:
        return None

    a = numer.args[0]
    all_exps = [a, *exp_exponents]
    if any(e.has(sp.Max) for e in all_exps):
        return None

    M = sp.Max(*all_exps)
    new_num = sp.exp(a - M, evaluate=False)
    new_den = sp.Add(
        *[sp.exp(b - M, evaluate=False) for b in exp_exponents],
        *other_terms,
        evaluate=False,
    )
    return new_num / new_den

def stabilize_exp_quotient(expr: sp.Expr) -> sp.Expr:
    """
    Stabilize exp(a)/(exp(a)+exp(b)+…) using max-shift so exp exponents are ≤ 0.

    exp(n)/(exp(n)+exp(m)) = 1/(1+exp(m−n))           when n ≥ m
                           = exp(n−m)/(1+exp(n−m))     when m > n
    Unified: exp(a−Max(a,b,…)) / Σ exp(·−Max).
    """
    if expr.is_Atom:
        return expr

    expr = expr.func(*[stabilize_exp_quotient(a) for a in expr.args])

    if expr.is_Mul or expr.is_Pow:
        numer, denom = expr.as_numer_denom()
        rewritten = _rewrite_exp_over_exp_sum(numer, denom)
        if rewritten is not None:
            return stabilize_exp_quotient(rewritten)
    return expr


def optimize_symbolic(expr: sp.Expr) -> sp.Expr:
    """Apply symbolic rewrite rules once (no log(T) / ln∘exp algebra)."""
    expr = fold_coeff_into_exp(expr)
    expr = merge_t_exp_add(expr)
    expr = merge_nested_tetration(expr)
    expr = merge_exp_products(expr)
    expr = stabilize_ln_t_sum(expr)
    expr = stabilize_exp_quotient(expr)
    return merge_nested_tetration(expr)
