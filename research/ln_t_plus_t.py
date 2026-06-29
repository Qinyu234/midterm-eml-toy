"""
ln(T+T) optimization workspace (DESIGN §1.8) — INCOMPLETE, off main path.
ln(T+T) 优化工作区 — 未完成，未接入主路径。

This module documents the research direction and stub approximations.
It is NOT imported by symbolic/pipeline.simplify_expr or compile/lambdify_exprs.

展示位置 / where to look:
  - This file: `py/eml/research/ln_t_plus_t.py`
  - Index:     `py/eml/research/README.md`
  - Design:    `DESIGN.md` §1.8
  - Explicitly excluded in PURPOSE.md (out of scope for main pipeline)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import sympy as sp

from symbolic.tetration import Tetration, E_TAG


class WorkspaceStatus(Enum):
    INCOMPLETE = auto()
    OFF_MAIN_PATH = auto()


STATUS = WorkspaceStatus.INCOMPLETE
ON_MAIN_PATH = False  # must stay False until exact identity is proven / 恒等式未证前保持 False


@dataclass(frozen=True)
class LnTPlusTIdentity:
    """
    Target identity (research, unproven):
    目标恒等式（研究中，未证明）:

        ln(T(e,n,z) + T(e,n,w)) = T(e, n-1, f(z,w))   # f(·,·) exact form TBD
    """
    n: int
    z: sp.Expr
    w: sp.Expr
    f_zw: Optional[sp.Expr] = None  # unknown / 待推导

    @property
    def lhs(self) -> sp.Expr:
        Tn = lambda expr: Tetration(E_TAG, self.n, expr)
        return sp.log(Tn(self.z) + Tn(self.w))

    @property
    def rhs_goal(self) -> sp.Expr:
        if self.f_zw is None:
            return sp.Symbol('f(z,w)')  # placeholder / 占位
        return Tetration(E_TAG, self.n - 1, self.f_zw)


def logsumexp_upper_bound(a: sp.Expr, b: sp.Expr) -> sp.Expr:
    """
    Current rough approximation (DESIGN §1.8):
    当前粗糙上界近似:

        ln(e^a + e^b) ≈ max(a, b)     valid only when a ≈ b
        ln(e^a + e^b) ≈ max(a, b)     仅 a≈b 时逼近
    """
    return sp.Max(a, b)


def approximate_ln_t_plus_t(n: int, z: sp.Expr, w: sp.Expr) -> sp.Expr:
    """
    Stub: map ln(T_n(z)+T_n(w)) → logsumexp on inner logs (research only).
    桩：不用于生产化简，仅供实验对比。
    """
    Tn = lambda expr: Tetration(E_TAG, n, expr)
    # ln(T+T) ≠ ln(e^a+e^b) directly; documents the intended LSE direction only
    # ln(T+T) 方向记录，未接入 simplify_expr
    a, b = sp.log(Tn(z)), sp.log(Tn(w))
    return logsumexp_upper_bound(a, b)


def workspace_summary() -> str:
    """Human-readable status for CLI / docs / 状态摘要。"""
    return '\n'.join([
        'ln(T+T) optimization workspace (DESIGN §1.8)',
        f'  status: {STATUS.name}',
        f'  on_main_path: {ON_MAIN_PATH}',
        '  target: ln(T(e,n,z)+T(e,n,w)) = T(e,n-1,f(z,w))  [f TBD]',
        '  current stub: ln(e^a+e^b) ≈ max(a,b)  (a≈b only)',
        '  wired into simplify_expr: NO',
        '  see: py/eml/research/ln_t_plus_t.py',
        '  symbolic pipeline: py/eml/symbolic/pipeline.py (does not import research)',
    ])
