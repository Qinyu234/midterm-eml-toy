"""
Node encoding (DESIGN §2.2, §5.1).
节点编码。

8-bit sweep; low 4 bits effective, high 4 bits reserved:
  bit 3   : x_slot — x in left(0) or right(1) slot / x 进左槽(0)或右槽(1)
  bit 2-0 : n_eml  — count of eml nodes ∈ [0,7] = #learnable params / eml 节点数
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeCode:
    x_slot: int   # 0=left, 1=right / 左槽、右槽
    n_eml: int    # nesting depth 0–7 / 嵌套层数
    reserved: int = 0  # high nibble / 高 4 bit

    @property
    def n_params(self) -> int:
        return self.n_eml

    @property
    def low_nibble(self) -> int:
        return ((self.x_slot & 0x1) << 3) | (self.n_eml & 0x7)

    def to_byte(self) -> int:
        return ((self.reserved & 0xF) << 4) | self.low_nibble


def decode_node(byte: int) -> NodeCode:
    assert 0 <= byte <= 255, f'expected 8-bit byte, got {byte}'
    low = byte & 0x0F
    return NodeCode(
        x_slot=(low >> 3) & 0x1,
        n_eml=low & 0x7,
        reserved=(byte >> 4) & 0xF,
    )


def encode_node(node: NodeCode) -> int:
    return node.to_byte()


def iter_node_bytes() -> range:
    """DESIGN §5.1: sweep all 256 8-bit codes / 遍历 256 个编码。"""
    return range(256)


def is_evaluable(byte: int) -> bool:
    """Skip when n_eml=0 (empty chain) / n_eml=0 空链跳过。"""
    return decode_node(byte).n_eml > 0
