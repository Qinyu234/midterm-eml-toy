"""strategy package / strategy 子包。"""
from .base import G0Random, G1BeamSearch, G2Adaptive, GenerationStrategy

__all__ = ['GenerationStrategy', 'G0Random', 'G1BeamSearch', 'G2Adaptive']
