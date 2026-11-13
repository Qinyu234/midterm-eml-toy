"""Compatibility-only SPSA/RDSA smoke test using the optional torchzero library.

This is not part of TinyStories baseline training and does not implement either
zeroth-order algorithm locally.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from typing import Any

import torch
from torch.nn import functional as F

import torchzero as tz
from tinystories_byte_lm.config import ModelConfig
from tinystories_byte_lm.model import ByteTransformer


def probe_method(method_name: str, seed: int) -> dict[str, Any]:
    if method_name not in ("SPSA", "RDSA"):
        raise ValueError(f"Unsupported probe method: {method_name}")

    torch.manual_seed(seed)
    model = ByteTransformer(
        ModelConfig(
            d_model=16,
            n_layers=1,
            n_heads=4,
            d_ff=32,
            context_length=8,
            position_encoding="nope",
            lut_seed=seed,
        )
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    before = [parameter.detach().clone() for parameter in parameters]
    inputs = torch.tensor([[65, 66, 67, 68]], dtype=torch.long)
    targets = torch.tensor([[66, 67, 68, 69]], dtype=torch.long)

    approximation = getattr(tz.m, method_name)(
        h=1e-3,
        n_samples=1,
        formula="central2",
        seed=seed,
    )
    optimizer = tz.Optimizer(
        parameters,
        approximation,
        tz.m.Identity(),
        tz.m.LR(1e-4),
    )

    def closure(backward: bool = True) -> torch.Tensor:
        if backward:
            optimizer.zero_grad()
        loss = F.cross_entropy(
            model(inputs).reshape(-1, 256),
            targets.reshape(-1),
        )
        if backward:
            loss.backward()
        return loss

    loss = optimizer.step(closure)
    loss_value = float(loss.detach().float().item())
    changed = any(
        not torch.equal(old, current.detach())
        for old, current in zip(before, parameters, strict=True)
    )
    finite_parameters = all(
        bool(torch.isfinite(parameter).all()) for parameter in parameters
    )
    return {
        "method": method_name,
        "finite_loss": bool(torch.isfinite(torch.tensor(loss_value))),
        "finite_parameters": finite_parameters,
        "parameters_updated": changed,
        "loss": loss_value,
        "library_api_compatible": finite_parameters and changed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()
    report = {
        "library": "torchzero",
        "version": version("torchzero"),
        "purpose": "API compatibility only; no TinyStories training or optimizer comparison",
        "results": [
            probe_method("SPSA", args.seed),
            probe_method("RDSA", args.seed + 1),
        ],
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
