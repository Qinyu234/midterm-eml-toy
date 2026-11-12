"""Sweep square model widths at a fixed ~95M parameter budget and time each shape.

The question: at equal parameter count, which square width d = k^2 is fastest on
this CPU? Layer count is solved from the architecture formula so every shape
lands within ~1% of the same budget.

Usage: python sweep_square_widths.py [--batch 32] [--timed 3]
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from tinystories_byte_lm.config import ModelConfig
from tinystories_byte_lm.model import ByteTransformer

ROOT = Path(__file__).resolve().parents[1]
TARGET_PARAMS = 95_041_105
CTX = 256
VOCAB = 256


def params_for(d: int, layers: int, ff: int) -> int:
    return layers * (4 * d * d + 2 * d * ff + 9 * d + ff) + 258 * d + 256


def solve_layers(d: int, ff: int) -> int:
    per_layer = 4 * d * d + 2 * d * ff + 9 * d + ff
    return max(1, round((TARGET_PARAMS - 258 * d - 256) / per_layer))


def pick_heads(d: int, ff: int) -> int:
    for heads in (12, 11, 9, 8, 6, 4):
        if d % heads == 0 and d // heads >= 32:
            return heads
    return 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--timed", type=int, default=3)
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)

    widths = [k * k for k in range(24, 35)]
    rows = []
    print(f"target params {TARGET_PARAMS:,} | batch {args.batch} | ctx {CTX} | {args.timed} timed steps\n")
    header = f"{'d_model':>8} {'k':>3} {'layers':>7} {'ff':>6} {'heads':>6} {'params':>11} {'dev%':>6} {'s/step':>8} {'B/s':>7}"
    print(header)
    print("-" * len(header))

    for d in widths:
        ff = 3 * d  # the 3x ratio all three committed candidates use
        layers = solve_layers(d, ff)
        heads = pick_heads(d, ff)
        cfg = ModelConfig(d_model=d, n_layers=layers, n_heads=heads, d_ff=ff,
                          context_length=CTX, position_encoding="nope", dropout=0.0)
        model = ByteTransformer(cfg)
        n_params = sum(p.numel() for p in model.parameters())
        opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
        g = torch.Generator().manual_seed(2718)
        inputs = torch.randint(0, VOCAB, (args.batch, CTX), generator=g)
        targets = torch.randint(0, VOCAB, (args.batch, CTX), generator=g)

        model.train()
        for _ in range(1):  # warmup allocator / thread pool
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        times = []
        for _ in range(args.timed):
            t0 = time.perf_counter()
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            times.append(time.perf_counter() - t0)

        s_step = statistics.median(times)
        bps = args.batch * CTX / s_step
        dev = (n_params - TARGET_PARAMS) / TARGET_PARAMS * 100
        print(f"{d:>8} {round(d ** 0.5):>3} {layers:>7} {ff:>6} {heads:>6} {n_params:>11,} {dev:>6.2f} {s_step:>8.2f} {bps:>7.1f}")
        rows.append({"d_model": d, "k": round(d ** 0.5), "n_layers": layers, "d_ff": ff,
                     "n_heads": heads, "params": n_params, "param_deviation_pct": round(dev, 2),
                     "seconds_per_step": round(s_step, 2), "bytes_per_second": round(bps, 1)})
        del model, opt, logits, loss

    best = min(rows, key=lambda r: r["seconds_per_step"])
    print(f"\nfastest: d_model={best['d_model']} (k={best['k']}, {best['n_layers']} layers) "
          f"{best['seconds_per_step']} s/step, {best['bytes_per_second']} B/s")
    for r in sorted(rows, key=lambda r: r["seconds_per_step"]):
        print(f"  d{r['d_model']:<5} L{r['n_layers']:<3} {r['seconds_per_step']:>7.2f} s/step   {r['seconds_per_step'] / best['seconds_per_step']:>5.2f}x")

    out = ROOT / "reports" / "square-width-sweep.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"target_params": TARGET_PARAMS, "batch_size": args.batch,
                               "context_length": CTX, "threads": args.threads,
                               "results": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
