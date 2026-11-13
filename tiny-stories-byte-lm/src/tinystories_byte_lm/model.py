"""Fixed-LUT byte Transformer; dimensions stay config-driven until budget review.

Inputs and outputs use exactly 256 byte IDs. The input LUT is a seeded,
isotropic N(0, 1/d_model) table with gradients disabled. Attention is full
causal scaled-dot-product attention. The output projection is independent of
the LUT. No BOS/EOS token or weight tying is used.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig

VOCAB_SIZE = 256


class TeLU(nn.Module):
    """TeLU: x * tanh(exp(x)), with a stable large-positive FP32 path."""

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise TypeError(f"TeLU supports FP16, BF16, and FP32, not {inputs.dtype}")
        work = inputs.float() if inputs.dtype != torch.float32 else inputs
        # Above 20, tanh(exp(x)) rounds to 1 in FP32; clamping avoids exp overflow.
        output = work * torch.tanh(torch.exp(torch.clamp(work, max=20.0)))
        return output.to(dtype=inputs.dtype)


def _apply_rope(query: Tensor, key: Tensor, theta: float = 10_000.0) -> tuple[Tensor, Tensor]:
    head_dim = query.shape[-1]
    positions = torch.arange(query.shape[-2], device=query.device, dtype=torch.float32)
    dimensions = torch.arange(0, head_dim, 2, device=query.device, dtype=torch.float32)
    inverse_frequencies = torch.pow(theta, -dimensions / head_dim)
    angles = torch.outer(positions, inverse_frequencies)
    cosine = angles.cos()[None, None, :, :]
    sine = angles.sin()[None, None, :, :]

    def rotate(values: Tensor) -> Tensor:
        values_fp32 = values.float()
        even = values_fp32[..., 0::2]
        odd = values_fp32[..., 1::2]
        rotated = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine),
            dim=-1,
        ).flatten(-2)
        return rotated.to(dtype=values.dtype)

    return rotate(query), rotate(key)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.position_encoding = config.position_encoding
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.output = nn.Linear(config.d_model, config.d_model)

    def forward(self, inputs: Tensor) -> Tensor:
        batch, sequence, width = inputs.shape
        qkv = self.qkv(inputs).reshape(
            batch, sequence, 3, self.n_heads, self.head_dim
        )
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
        if self.position_encoding == "rope":
            query, key = _apply_rope(query, key)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, sequence, width)
        return self.output(attended)


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.feed_forward_norm = nn.LayerNorm(config.d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            TeLU(),
            nn.Linear(config.d_ff, config.d_model),
        )
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        inputs = inputs + self.residual_dropout(self.attention(self.attention_norm(inputs)))
        return inputs + self.residual_dropout(
            self.feed_forward(self.feed_forward_norm(inputs))
        )


class ByteTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        generator = torch.Generator(device="cpu").manual_seed(config.lut_seed)
        lut = torch.randn(VOCAB_SIZE, config.d_model, generator=generator)
        lut.mul_(1.0 / math.sqrt(config.d_model))
        self.byte_lut = nn.Embedding.from_pretrained(lut, freeze=True)
        self.blocks = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.n_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, VOCAB_SIZE)

    def forward(self, byte_ids: Tensor) -> Tensor:
        if byte_ids.ndim != 2:
            raise ValueError("byte_ids must have shape [batch, sequence]")
        if byte_ids.shape[1] > self.config.context_length:
            raise ValueError(
                f"sequence length {byte_ids.shape[1]} exceeds "
                f"context_length {self.config.context_length}"
            )
        hidden = self.byte_lut(byte_ids)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output_projection(self.final_norm(hidden))


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
