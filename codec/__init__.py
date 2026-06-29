"""8-bit node codec (DESIGN §2.2). / Node 编解码。"""
from .node import NodeCode, decode_node, encode_node, is_evaluable, iter_node_bytes

__all__ = ['NodeCode', 'decode_node', 'encode_node', 'is_evaluable', 'iter_node_bytes']
