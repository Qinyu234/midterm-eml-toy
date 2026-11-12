"""Unconstrained byte generation and strict UTF-8 validity measurement."""

from __future__ import annotations

import torch
from torch import nn


@torch.no_grad()
def generate_bytes(
    model: nn.Module,
    prompt: bytes,
    max_new_bytes: int,
    temperature: float = 1.0,
    seed: int = 0,
) -> bytes:
    if not prompt:
        raise ValueError("A byte prompt is required; this model has no BOS token")
    if max_new_bytes < 0:
        raise ValueError("max_new_bytes must be non-negative")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    parameter = next(model.parameters())
    device = parameter.device
    generator = torch.Generator(device=device).manual_seed(seed)
    output = list(prompt)
    model.eval()
    for _ in range(max_new_bytes):
        context_length = model.config.context_length
        context = torch.tensor(
            output[-context_length:], dtype=torch.long, device=device
        ).unsqueeze(0)
        logits = model(context)[0, -1].float() / temperature
        probabilities = torch.softmax(logits, dim=-1)
        next_byte = int(torch.multinomial(probabilities, 1, generator=generator).item())
        output.append(next_byte)
    return bytes(output)


def is_valid_utf8(data: bytes) -> bool:
    try:
        data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    return True
