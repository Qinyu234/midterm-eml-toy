"""Probe CPU training support, finite values, and throughput for FP32/FP16/BF16."""

from __future__ import annotations

import json
import platform
import time
from dataclasses import asdict
from typing import Any

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .model import ByteTransformer, TeLU
from .training import train_step

DTYPES: tuple[tuple[str, torch.dtype], ...] = (
    ("fp32", torch.float32),
    ("fp16", torch.float16),
    ("bf16", torch.bfloat16),
)


def _finite_gradients(model: nn.Module) -> bool:
    return all(
        bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
        if parameter.grad is not None
    )


def probe_telu(dtype: torch.dtype) -> dict[str, Any]:
    values = torch.tensor(
        [-100.0, -20.0, -5.0, -1.0, 0.0, 1.0, 5.0, 20.0, 100.0],
        dtype=dtype,
        requires_grad=True,
    )
    output = TeLU()(values)
    output.float().sum().backward()
    return {
        "supported": True,
        "finite_forward": bool(torch.isfinite(output).all()),
        "finite_backward": values.grad is not None
        and bool(torch.isfinite(values.grad).all()),
        "output_dtype": str(output.dtype),
    }


def probe_dtype(
    dtype_name: str,
    dtype: torch.dtype,
    config: ModelConfig,
    inputs: Tensor,
    targets: Tensor,
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, Any]:
    torch.manual_seed(config.lut_seed)
    try:
        model = ByteTransformer(config).to(device="cpu", dtype=dtype)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=1e-3,
        )
        for _ in range(warmup_steps):
            train_step(model, optimizer, inputs, targets)
        start = time.perf_counter()
        metrics = None
        for _ in range(measured_steps):
            metrics = train_step(model, optimizer, inputs, targets)
        elapsed = time.perf_counter() - start
        finite = (
            metrics is not None
            and torch.isfinite(torch.tensor(metrics.loss))
            and torch.isfinite(torch.tensor(metrics.gradient_norm))
            and _finite_gradients(model)
        )
        input_bytes = int((targets >= 0).sum().item()) * measured_steps
        return {
            "dtype": dtype_name,
            "supported": True,
            "finite_forward_loss_and_gradients": bool(finite),
            "measured_steps": measured_steps,
            "elapsed_seconds": elapsed,
            "input_bytes_per_second": input_bytes / elapsed,
            "last_loss": metrics.loss if metrics else None,
            "last_gradient_norm": metrics.gradient_norm if metrics else None,
            "telu": probe_telu(dtype),
        }
    except (NotImplementedError, RuntimeError) as error:
        return {
            "dtype": dtype_name,
            "supported": False,
            "finite_forward_loss_and_gradients": False,
            "error": f"{type(error).__name__}: {error}",
            "telu": probe_telu(dtype),
        }


def run_cpu_precision_probe(
    batch_size: int = 4,
    sequence_length: int = 32,
    warmup_steps: int = 1,
    measured_steps: int = 3,
    num_threads: int | None = None,
) -> dict[str, Any]:
    if min(batch_size, sequence_length, measured_steps) <= 0 or warmup_steps < 0:
        raise ValueError("batch_size, sequence_length, and measured_steps must be positive")
    if num_threads is not None:
        if num_threads <= 0:
            raise ValueError("num_threads must be positive")
        torch.set_num_threads(num_threads)

    config = ModelConfig(
        d_model=32,
        n_layers=1,
        n_heads=4,
        d_ff=64,
        context_length=sequence_length,
        position_encoding="nope",
        lut_seed=1729,
    )
    generator = torch.Generator(device="cpu").manual_seed(2718)
    inputs = torch.randint(
        0, 256, (batch_size, sequence_length), generator=generator, dtype=torch.long
    )
    targets = torch.roll(inputs, shifts=-1, dims=1)
    results = [
        probe_dtype(
            dtype_name,
            dtype,
            config,
            inputs,
            targets,
            warmup_steps,
            measured_steps,
        )
        for dtype_name, dtype in DTYPES
    ]
    return {
        "device": "cpu",
        "cpu": platform.processor(),
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "probe_config": asdict(config),
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "results": results,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = run_cpu_precision_probe(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        warmup_steps=args.warmup_steps,
        measured_steps=args.steps,
        num_threads=args.threads,
    )
    result = json.dumps(report, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
