"""
Symbolic pipeline (DESIGN §1.2, §2.3).
符号 pipeline：T 域化简 → value/grad → 展开为 log/exp。

Phases / 阶段:
  1. to_t_form        — exp/ln → T(e,n,·)
  2. simplify_in_t    — T 合并（嵌套、T(e,1,·) 积）
  3. diff             — value 与 grad（T 域）
  4. expand_t_to_log_exp — 交付 log/exp 式
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import sympy as sp

from codec.node import NodeCode, decode_node
from symbolic.tetration import (
    compressed_exp,
    compressed_ln,
    expand_t_to_log_exp,
    simplify_in_t,
    to_t_form,
)
from symbolic.wirtinger import WirtingerJacobian, simplify_wirtinger, wirtinger_grad
from tree.coords import (
    SlotBinding,
    all_d_params,
    all_leaf_d_substitutions,
    d_symbol_for_slot,
    layer_is_left,
    slot_bindings,
)


def eml(z: sp.Expr, w: sp.Expr) -> sp.Expr:
    """eml(z,w) in T form: T(e,1,z) − T(e,−1,w)."""
    return compressed_exp(1, z) - compressed_ln(1, w)


def build_value_tree(
    node: NodeCode,
    bindings: Sequence[SlotBinding],
    *,
    use_raw_c: bool = False,
) -> sp.Expr:
    expected = node.n_eml + 1
    if len(bindings) != expected:
        raise ValueError(f'expected {expected} slots, got {len(bindings)}')

    def leaf(b: SlotBinding) -> sp.Expr:
        if use_raw_c:
            return b.raw
        return b.normalized

    h = leaf(bindings[0])
    if node.n_eml == 0:
        return h

    for i, b in enumerate(bindings[1:], start=1):
        slot = leaf(b)
        if layer_is_left(i, node.x_slot):
            h = eml(slot, h)
        else:
            h = eml(h, slot)
    return h


def express_in_d(tree_c: sp.Expr, bindings: Sequence[SlotBinding]) -> sp.Expr:
    """Substitute every leaf (C or x) → T(e,-(x+y),D); only D symbols / 仅 D。"""
    return tree_c.subs(all_leaf_d_substitutions(bindings))


def simplify_expr(expr: sp.Expr) -> sp.Expr:
    """Full pipeline: T simplify then expand to log/exp."""
    return expand_t_to_log_exp(simplify_in_t(expr))


@dataclass(frozen=True)
class ParamGradient:
    param: sp.Expr
    grad: sp.Expr
    dz: sp.Expr
    dz_bar: sp.Expr

    @classmethod
    def from_wirtinger(cls, param: sp.Expr, w: WirtingerJacobian) -> ParamGradient:
        w = simplify_wirtinger(w)
        return cls(param=param, grad=w.dz, dz=w.dz, dz_bar=w.dz_bar)

    def expanded(self, expand_fn) -> ParamGradient:
        return ParamGradient(
            param=self.param,
            grad=expand_fn(self.grad),
            dz=expand_fn(self.dz),
            dz_bar=expand_fn(self.dz_bar),
        )


@dataclass(frozen=True)
class NodeSymbolicExprs:
    byte: int
    node: NodeCode
    bindings: Tuple[SlotBinding, ...]
    value_expr: sp.Expr
    param_grads: Tuple[ParamGradient, ...]


@dataclass(frozen=True)
class NodeFormulaComparison:
    """T-domain value and ∂value/∂D."""
    byte: int
    node: NodeCode
    bindings: Tuple[SlotBinding, ...]
    value_t_raw: sp.Expr
    value_t_opt: sp.Expr
    d_grads_raw: Tuple[ParamGradient, ...]
    d_grads_opt: Tuple[ParamGradient, ...]


def compute_node_formula_comparison(
    byte: int,
    *,
    include_raw_grads: bool = True,
) -> NodeFormulaComparison:
    """D-space tree → T → simplify_in_t → Wirtinger grad in T."""
    node = decode_node(byte)
    bindings = slot_bindings(node.n_eml, node.x_slot)
    tree_c = build_value_tree(node, bindings, use_raw_c=True)
    value_d = express_in_d(tree_c, bindings)

    value_t_raw = to_t_form(value_d)
    value_t_opt = simplify_in_t(value_d)

    d_params = all_d_params(bindings)
    if include_raw_grads:
        d_grads_raw = tuple(
            ParamGradient.from_wirtinger(d, wirtinger_grad(value_t_raw, d))
            for d in d_params
        )
    else:
        d_grads_raw = ()
    d_grads_opt = tuple(
        ParamGradient.from_wirtinger(d, wirtinger_grad(value_t_opt, d))
        for d in d_params
    )

    return NodeFormulaComparison(
        byte=byte,
        node=node,
        bindings=bindings,
        value_t_raw=value_t_raw,
        value_t_opt=value_t_opt,
        d_grads_raw=d_grads_raw,
        d_grads_opt=d_grads_opt,
    )


def compute_node_exprs(byte: int) -> NodeSymbolicExprs:
    node = decode_node(byte)
    bindings = slot_bindings(node.n_eml, node.x_slot)
    tree_c = build_value_tree(node, bindings, use_raw_c=True)
    value_d = express_in_d(tree_c, bindings)
    value_t = simplify_in_t(value_d)
    value = expand_t_to_log_exp(value_t)

    param_grads = tuple(
        ParamGradient.from_wirtinger(
            d,
            wirtinger_grad(value_t, d),
        ).expanded(expand_t_to_log_exp)
        for d in all_d_params(bindings)
    )

    return NodeSymbolicExprs(
        byte=byte,
        node=node,
        bindings=bindings,
        value_expr=value,
        param_grads=param_grads,
    )
