"""16-node sweep: all low-nibble (x_slot × n_eml) combinations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List

from codec.node import NodeCode, decode_node, encode_node, is_evaluable
from tree.coords import bottom_input_slot, x_slot_index


@dataclass(frozen=True)
class NodeSweepEntry:
    """One of 16 low-nibble node codes; y at outermost (top) by tree design."""
    byte: int
    n_eml: int
    x_slot: int
    evaluable: bool
    x_leaf: int  # topology x leaf (x_slot mode)
    input_bind_slot: int  # bottom_d canonical bind (min |d|, center tie)

    @property
    def label(self) -> str:
        side = 'L' if self.x_slot == 0 else 'R'
        return (
            f'byte={self.byte} n_eml={self.n_eml} x_slot={side} '
            f'x@leaf{self.x_leaf} bind@leaf{self.input_bind_slot}'
        )


def iter_16_node_bytes() -> range:
    """All 16 low-nibble codes 0..15 (x_slot × n_eml)."""
    return range(16)


def iter_evaluable_16_node_bytes() -> Iterator[int]:
    for byte in iter_16_node_bytes():
        if is_evaluable(byte):
            yield byte


def describe_node(byte: int) -> NodeSweepEntry:
    node = decode_node(byte)
    if node.n_eml > 0:
        x_leaf = x_slot_index(node.n_eml, node.x_slot)
        bind = bottom_input_slot(node.n_eml, node.x_slot)
    else:
        x_leaf = -1
        bind = -1
    return NodeSweepEntry(
        byte=byte,
        n_eml=node.n_eml,
        x_slot=node.x_slot,
        evaluable=is_evaluable(byte),
        x_leaf=x_leaf,
        input_bind_slot=bind,
    )


def list_16_nodes(*, evaluable_only: bool = False) -> List[NodeSweepEntry]:
    it = iter_evaluable_16_node_bytes() if evaluable_only else iter_16_node_bytes()
    return [describe_node(b) for b in it]


def encode_sweep_node(n_eml: int, x_slot: int) -> int:
    return encode_node(NodeCode(x_slot=x_slot, n_eml=n_eml, reserved=0))
