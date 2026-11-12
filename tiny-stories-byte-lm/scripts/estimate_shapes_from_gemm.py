"""Estimate per-step time for candidate shapes from matrix-multiply throughput.

Phase 1 of the shape search: no full model, no training loop. We time square
matmuls at the sizes each shape actually uses, then convert the analytic FLOP
count of one training step into seconds.

Phase 2 (separate script) measures the top-k shapes for real.

Usage: python estimate_shapes_from_gemm.py
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
TARGET_PARAMS = 95_041_105
BATCH = 32
CTX = 256
THREADS = 10
GEMM_SIZES = [256, 512, 576, 768, 1024, 1280, 1536, 1600, 2048, 2304, 3072, 4608, 6912]


def bench_gemm(n: int, reps: int = 5) -> float:
    """GFLOPS for an n x n x n float32 matmul."""
    a = torch.randn(n, n)
    b = torch.randn(n, n)
    torch.mm(a, b)  # warmup
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        torch.mm(a, b)
        times.append(time.perf_counter() - t0)
    t = statistics.median(times)
    return 2 * n**3 / t / 1e9


def params_for(d: int, layers: int, ff: int) -> int:
    return layers * (4 * d * d + 2 * d * ff + 9 * d + ff) + 258 * d + 256


def solve_layers(d: int, ff: int) -> int:
    per_layer = 4 * d * d + 2 * d * ff + 9 * d + ff
    return max(1, round((TARGET_PARAMS - 258 * d - 256) / per_layer))


def rate_at(rates: dict[int, float], n: int) -> float:
    """Nearest measured GEMM size, so a shape is charged the rate it will really see."""
    return rates[min(rates, key=lambda s: abs(s - n))]


def estimate_step_seconds(d: int, layers: int, ff: int, rates: dict[int, float]) -> tuple[float, float]:
    """Seconds per training step, splitting linear work by d- and ff-sized matmuls."""
    tokens = BATCH * CTX
    # Per layer linear FLOPs per token (fwd) and the backward factor of 2x.
    d_part = 4 * d * d + 9 * d        # attention projections + norms/residuals
    ff_part = 2 * d * ff + ff          # MLP up/down
    linear_flops = 3 * tokens * layers * (d_part + ff_part)
    # Attention score/value products: 4 * T * d per layer per token, times 3 for bwd.
    attn_flops = 3 * layers * 4 * BATCH * CTX * CTX * d

    t_d = linear_flops * d_part / (d_part + ff_part) / (rate_at(rates, d) * 1e9)
    t_ff = linear_flops * ff_part / (d_part + ff_part) / (rate_at(rates, ff) * 1e9)
    t_attn = attn_flops / (rate_at(rates, d) * 1e9)
    return t_d + t_ff + t_attn, linear_flops + attn_flops


def main() -> None:
    torch.set_num_threads(THREADS)
    print(f"GEMM throughput, float32, {THREADS} threads\n")
    rates = {}
    for n in GEMM_SIZES:
        rates[n] = bench_gemm(n)
        print(f"  n={n:>5}  {rates[n]:>7.1f} GFLOPS")

    # Squares that are also multiples of 64 (k divisible by 8), plus the calibration shape.
    shapes = []
    for k in (16, 24, 32, 40, 48):
        d = k * k
        ff = 3 * d
        shapes.append((d, solve_layers(d, ff), ff))
    shapes.append((1089, 8, 3267))  # measured anchor: 38.95 s/step at batch 32

    print(f"\ntarget params {TARGET_PARAMS:,} | batch {BATCH} | ctx {CTX}\n")
    header = f"{'d_model':>8} {'layers':>7} {'ff':>6} {'params':>11} {'TFLOPs/step':>12} {'est s/step':>11} {'est B/s':>8}"
    print(header)
    print("-" * len(header))
    rows = []
    for d, layers, ff in shapes:
        s_est, flops = estimate_step_seconds(d, layers, ff, rates)
        n_params = params_for(d, layers, ff)
        print(f"{d:>8} {layers:>7} {ff:>6} {n_params:>11,} {flops / 1e12:>12.2f} {s_est:>11.2f} {BATCH * CTX / s_est:>8.1f}")
        rows.append({"d_model": d, "n_layers": layers, "d_ff": ff, "params": n_params,
                     "tflops_per_step": round(flops / 1e12, 2),
                     "estimated_seconds_per_step": round(s_est, 2)})

    anchor = next(r for r in rows if r["d_model"] == 1089)
    correction = 38.95 / anchor["estimated_seconds_per_step"]
    print(f"\ncalibration: d1089 measured 38.95 s/step, estimate {anchor['estimated_seconds_per_step']} s "
          f"-> scale all estimates by {correction:.2f}x")
    for r in rows:
        r["calibrated_seconds_per_step"] = round(r["estimated_seconds_per_step"] * correction, 2)

    print("\nranking (calibrated):")
    for r in sorted(rows, key=lambda r: r["calibrated_seconds_per_step"]):
        print(f"  d{r['d_model']:<5} L{r['n_layers']:<3} {r['calibrated_seconds_per_step']:>7.2f} s/step")

    out = ROOT / "reports" / "shape-estimate-from-gemm.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gemm_gflops": rates, "batch_size": BATCH, "context_length": CTX,
                               "threads": THREADS, "calibration_scale": round(correction, 3),
                               "shapes": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
