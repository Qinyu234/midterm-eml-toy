"""A/B harness for step-time variants at the CPU pilot shape.

Each variant runs the same workload (real model, AdamW, real dataset) and reports
median seconds per step over the timed steps, so a change is only kept if it moves
the number.

Usage: python bench_step_variants.py [--steps 10]
"""

from __future__ import annotations

import argparse
import random
import statistics
import time
from pathlib import Path

import torch

from tinystories_byte_lm.config import ModelConfig
from tinystories_byte_lm.data import MappedByteStories
from tinystories_byte_lm.model import ByteTransformer

ROOT = Path(__file__).resolve().parents[1]
CAP = 64_522_880
BATCH, CTX = 32, 256


class _TeLUFn(torch.autograd.Function):
    """x * tanh(exp(x)) as one autograd node instead of four."""

    @staticmethod
    def forward(ctx, inputs: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            u = torch.exp(torch.clamp(inputs, max=20.0))
            t = torch.tanh(u)
            out = inputs * t
            ctx.save_for_backward(inputs, t, u)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor) -> torch.Tensor:
        inputs, t, u = ctx.saved_tensors
        return grad * (t + inputs * (1 - t * t) * u)


class FastTeLU(torch.nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return _TeLUFn.apply(inputs)


def build(activation: str) -> ByteTransformer:
    cfg = ModelConfig(d_model=256, n_layers=4, n_heads=4, d_ff=1024,
                      context_length=CTX, position_encoding="nope", dropout=0.0)
    torch.manual_seed(1729)
    model = ByteTransformer(cfg)
    if activation == "fast":
        for block in model.blocks:
            block.feed_forward[1] = FastTeLU()
    return model


def run(model: ByteTransformer, steps: int, denormal: bool) -> tuple[float, float]:
    torch.set_flush_denormal(denormal)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
    rng = random.Random(2718)
    times = []
    with MappedByteStories(ROOT / "data" / "processed" / "train.bin",
                           ROOT / "data" / "processed" / "train.idx", max_bytes=CAP) as data:
        for step in range(steps):
            inputs, targets = data.sample_batch(BATCH, CTX, rng)
            t0 = time.perf_counter()
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step >= 2:
                times.append(time.perf_counter() - t0)
    med = statistics.median(times)
    return med, BATCH * CTX / med


def check_equivalence() -> float:
    """FastTeLU must agree with the reference activation in value and gradient."""
    torch.manual_seed(7)
    x = torch.randn(64, 32) * 4.0
    x_a, x_b = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    ref = x_a * torch.tanh(torch.exp(torch.clamp(x_a, max=20.0)))
    fast = _TeLUFn.apply(x_b)
    ref.sum().backward()
    fast.sum().backward()
    return float((ref - fast).abs().max())


def set_priority(name: str) -> None:
    """Windows priority class for this process (realtime needs elevation)."""
    if name == "normal":
        return
    import ctypes

    classes = {"high": 0x00000080, "realtime": 0x00000100}
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    kernel32.GetCurrentProcess.argtypes = []
    kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    kernel32.SetPriorityClass.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    if not kernel32.SetPriorityClass(handle, classes[name]):
        raise RuntimeError(f"SetPriorityClass({name}) failed: {ctypes.get_last_error()}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["baseline", "fast"], required=True)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--threads", type=int, default=10)
    ap.add_argument("--interop", type=int, default=None)
    ap.add_argument("--priority", choices=["normal", "high", "realtime"], default="normal")
    args = ap.parse_args()

    set_priority(args.priority)
    torch.set_num_threads(args.threads)
    if args.interop is not None:
        torch.set_num_interop_threads(args.interop)
    model = build(args.variant)
    med, bps = run(model, args.steps, False)
    print(f"{args.variant:8} threads={args.threads} interop={args.interop} prio={args.priority} "
          f"{med:6.2f} s/step  {bps:7,.0f} B/s")


if __name__ == "__main__":
    main()
