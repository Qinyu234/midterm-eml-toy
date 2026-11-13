"""Configuration types shared by the byte-level model and training pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    n_layers: int
    n_heads: int
    d_ff: int
    context_length: int
    position_encoding: Literal["nope", "rope"] = "nope"
    lut_seed: int = 0
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_layers <= 0 or self.n_heads <= 0:
            raise ValueError("d_model, n_layers, and n_heads must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_ff <= 0 or self.context_length <= 0:
            raise ValueError("d_ff and context_length must be positive")
        if self.position_encoding not in ("nope", "rope"):
            raise ValueError("position_encoding must be 'nope' or 'rope'")
        if self.position_encoding == "rope" and (self.d_model // self.n_heads) % 2:
            raise ValueError("RoPE requires an even head dimension")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
