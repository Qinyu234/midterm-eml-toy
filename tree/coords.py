"""
Linear-chain slot coordinates and param normalization (DESIGN §1.7, §2.1).
线性链叶槽坐标与参数归一化。

C 坐标规则 / C coordinate rule:
  - value 参考点 (0,0)
  - 从外到内（layer n_eml → 1，自上而下）每层 eml(z,w)：
      左子 z（exp）→ (x,y) += (1,0)
      右子 w（ln） → (x,y) += (0,-1)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence, Tuple

import sympy as sp

BindingMode = Literal['bottom_d', 'x_slot']

OUTPUT_COORD: Tuple[int, int] = (0, 0)
LEFT_OFFSET: Tuple[int, int] = (1, 0)    # 左 / exp: x++
RIGHT_OFFSET: Tuple[int, int] = (0, -1)  # 右 / ln: y--
# Legacy aliases / 旧名
EXP_OFFSET = LEFT_OFFSET
LN_OFFSET = RIGHT_OFFSET

X_SYMBOL = sp.Symbol('x')


def _add(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[int, int]:
    return (a[0] + b[0], a[1] + b[1])


def layer_is_left(layer_index: int, x_slot: int) -> bool:
    """Layer i (1=outermost): eml(left,right)? / 第 i 层是否 eml(左,右)。"""
    if x_slot == 0:
        return layer_index % 2 == 1
    return layer_index % 2 == 0


def slot_coords(n_eml: int, x_slot: int) -> Tuple[Tuple[int, int], ...]:
    """
    Leaf coords for slots 0..n_eml (slot 0 innermost, slot n_eml outermost).

    Walk top→bottom from value ref (0,0) at inner[n_eml].
    """
    if n_eml == 0:
        return (OUTPUT_COORD,)

    coords: list[Tuple[int, int] | None] = [None] * (n_eml + 1)
    inner: list[Tuple[int, int] | None] = [None] * (n_eml + 1)
    inner[n_eml] = OUTPUT_COORD

    for layer in range(n_eml, 0, -1):
        ref = inner[layer]
        assert ref is not None
        if layer_is_left(layer, x_slot):
            coords[layer] = _add(ref, LEFT_OFFSET)
            inner[layer - 1] = _add(ref, RIGHT_OFFSET)
        else:
            inner[layer - 1] = _add(ref, LEFT_OFFSET)
            coords[layer] = _add(ref, RIGHT_OFFSET)

    coords[0] = inner[0]
    assert all(c is not None for c in coords)
    return tuple(coords)  # type: ignore[return-value]


def x_slot_index(n_eml: int, x_slot: int) -> int:
    """Topology x leaf (x_slot sweep metadata only; not default Feynman bind)."""
    left = layer_is_left(1, x_slot)
    if x_slot == 0:
        return 1 if left else 0
    return 1 if not left else 0


def bottom_input_slot(n_eml: int, x_slot: int, n_vars: int = 1) -> int:
    """Canonical bind: innermost / min |d|, tie → lower slot (center-biased)."""
    return input_slot_indices(n_eml, x_slot, n_vars, binding='bottom_d')[0]


def _slot_d(coord: Tuple[int, int]) -> int:
    return coord[0] + coord[1]


def input_slot_indices(
    n_eml: int,
    x_slot: int,
    n_vars: int,
    *,
    binding: BindingMode = 'bottom_d',
) -> Tuple[int, ...]:
    """Map inputs → leaf slots. x_slot mode: x at topology x leaf (16-node sweep)."""
    if n_vars < 1:
        raise ValueError(f'n_vars must be >= 1, got {n_vars}')
    n_slots = n_eml + 1
    if n_vars > n_slots:
        raise ValueError(f'need {n_vars} input slots but node has {n_slots}')
    if binding == 'x_slot' and n_vars == 1:
        return (x_slot_index(n_eml, x_slot),)
    coords = slot_coords(n_eml, x_slot)
    ranked = sorted(
        range(n_slots),
        key=lambda i: (abs(_slot_d(coords[i])), i),
    )
    return tuple(ranked[:n_vars])


def learnable_slot_indices(
    n_eml: int,
    x_slot: int,
    n_vars: int,
    *,
    binding: BindingMode = 'bottom_d',
) -> Tuple[int, ...]:
    """Leaf slots treated as learnable constants (not Feynman inputs)."""
    ins = set(input_slot_indices(n_eml, x_slot, n_vars, binding=binding))
    return tuple(i for i in range(n_eml + 1) if i not in ins)


def c_to_d(c_expr: sp.Expr, cx: int, cy: int, *, evaluate: bool = False) -> sp.Expr:
    """D = T(e, x+y, C) / 坐标变换：C → D。"""
    from symbolic.tetration import compressed_exp, compressed_ln

    offset = cx + cy
    if offset == 0:
        return c_expr
    if offset > 0:
        return compressed_exp(offset, c_expr)
    return compressed_ln(-offset, c_expr)


def d_to_c(d_expr: sp.Expr, cx: int, cy: int, *, evaluate: bool = False) -> sp.Expr:
    """C = T(e, -(x+y), D); inverse of c_to_d when D = T(e, x+y, C)."""
    from symbolic.tetration import compressed_exp, compressed_ln

    offset = cx + cy
    if offset == 0:
        return d_expr
    if offset > 0:
        return compressed_ln(offset, d_expr)
    return compressed_exp(-offset, d_expr)


def normalize_param(b: sp.Expr, cx: int, cy: int, *, evaluate: bool = False) -> sp.Expr:
    """Alias: D = T(e, x+y, C) for tree leaf / 叶槽归一化 D。"""
    return c_to_d(b, cx, cy, evaluate=evaluate)


def param_symbol(index: int) -> sp.Symbol:
    return sp.Symbol(f'c_{index}')


def d_param_symbol(index: int) -> sp.Symbol:
    """D coordinate symbol (complex a+bi) / 坐标 D。"""
    return sp.Symbol(f'D_{index}', complex=True)


def d_symbol_for_slot(slot_index: int) -> sp.Symbol:
    """D for leaf slot (0..n_eml); includes input x slot / 每个叶槽一个 D。"""
    return d_param_symbol(slot_index + 1)


def all_leaf_d_substitutions(bindings: Sequence['SlotBinding']) -> dict[sp.Expr, sp.Expr]:
    """Every leaf C or x ← T(e, -(x+y), D_slot); only D symbols remain."""
    subs: dict[sp.Expr, sp.Expr] = {}
    for b in bindings:
        d_sym = d_symbol_for_slot(b.index)
        subs[b.raw] = d_to_c(d_sym, b.coord[0], b.coord[1])
    return subs


def c_to_d_substitutions(bindings: Sequence['SlotBinding']) -> dict[sp.Expr, sp.Expr]:
    """Alias for all_leaf_d_substitutions / 全叶槽 D 代入。"""
    return all_leaf_d_substitutions(bindings)


def all_d_params(bindings: Sequence['SlotBinding']) -> Tuple[sp.Symbol, ...]:
    """Every leaf slot D_{i+1}; all get gradients / 全叶槽求导。"""
    return tuple(d_symbol_for_slot(b.index) for b in bindings)


# backward-compatible alias
learnable_d_params = all_d_params


@dataclass(frozen=True)
class SlotBinding:
    index: int
    coord: Tuple[int, int]
    raw: sp.Expr
    is_input: bool

    @property
    def d(self) -> int:
        return self.coord[0] + self.coord[1]

    @property
    def normalized(self) -> sp.Expr:
        return normalize_param(self.raw, self.coord[0], self.coord[1], evaluate=False)


def slot_bindings(n_eml: int, x_slot: int) -> Tuple[SlotBinding, ...]:
    """One param per leaf slot; x_slot only affects tree topology, not symbols."""
    coords = slot_coords(n_eml, x_slot)
    return tuple(
        SlotBinding(i, coords[i], param_symbol(i + 1), is_input=False)
        for i in range(n_eml + 1)
    )
