"""Node metadata cache. / Node 元信息缓存。"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from codec.node import decode_node, is_evaluable
from symbolic.pipeline import compute_node_exprs


@dataclass
class NodeInfo:
    byte: int
    n_eml: int
    x_slot: int
    reserved: int
    n_params: int
    value_str: str
    stable: bool = True

    @property
    def structure_description(self) -> str:
        slot = 'L' if self.x_slot == 0 else 'R'
        return f'n_eml={self.n_eml}, x_slot={slot}, reserved={self.reserved}'


def build_node_info(byte: int) -> NodeInfo:
    node = decode_node(byte)
    if node.n_eml == 0:
        return NodeInfo(byte, 0, node.x_slot, node.reserved, 0, 'empty', stable=True)
    try:
        exprs = compute_node_exprs(byte)
        val = str(exprs.value_expr)
        stable = node.n_eml <= 3
    except Exception:
        val = ''
        stable = False
    return NodeInfo(
        byte, node.n_eml, node.x_slot, node.reserved,
        node.n_params, val, stable,
    )


def build_cache() -> Dict[int, NodeInfo]:
    return {b: build_node_info(b) for b in range(256)}


_GLOBAL_CACHE: Optional[Dict[int, NodeInfo]] = None


def get_global_cache() -> Dict[int, NodeInfo]:
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE is None:
        _GLOBAL_CACHE = build_cache()
    return _GLOBAL_CACHE


def get_node_info(byte: int) -> NodeInfo:
    return get_global_cache()[byte]


def get_nodes_by_complexity(n_eml: int) -> List[NodeInfo]:
    return [i for i in get_global_cache().values() if i.n_eml == n_eml]


def get_stable_nodes() -> List[NodeInfo]:
    return [i for i in get_global_cache().values() if i.stable and i.n_eml > 0]


def save_cache(path: Path) -> None:
    cache = get_global_cache()
    data = {str(k): asdict(v) for k, v in cache.items()}
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def load_cache(path: Path) -> Dict[int, NodeInfo]:
    raw = json.loads(path.read_text(encoding='utf-8'))
    return {int(k): NodeInfo(**v) for k, v in raw.items()}
