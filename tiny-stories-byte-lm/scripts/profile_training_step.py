"""Profile one training step: where does the time actually go?

Runs the real model + optimizer + dataset at the CPU pilot shape and reports the
top kernels from torch.profiler, plus a wall-clock split between data loading,
forward, backward and the optimizer step.

Usage: python profile_training_step.py [--steps 5]
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from tinystories_byte_lm.config import ModelConfig
from tinystories_byte_lm.data import MappedByteStories
from tinystories_byte_lm.model import ByteTransformer

ROOT = Path(__file__).resolve().parents[1]
CAP = 64_522_880
BATCH, CTX = 32, 256


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    cfg = ModelConfig(d_model=256, n_layers=4, n_heads=4, d_ff=1024,
                      context_length=CTX, position_encoding="nope", dropout=0.0)
    model = ByteTransformer(cfg)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
    rng = random.Random(2718)

    split = {"load": 0.0, "zero": 0.0, "forward": 0.0, "backward": 0.0, "step": 0.0}
    with MappedByteStories(ROOT / "data" / "processed" / "train.bin",
                           ROOT / "data" / "processed" / "train.idx", max_bytes=CAP) as data:
        prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
        for step in range(args.steps + 1):
            active = step > 0
            t0 = time.perf_counter()
            inputs, targets = data.sample_batch(BATCH, CTX, rng)
            t1 = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            t2 = time.perf_counter()
            logits = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 256), targets.reshape(-1))
            t3 = time.perf_counter()
            loss.backward()
            t4 = time.perf_counter()
            opt.step()
            t5 = time.perf_counter()
            if step == 1:
                prof.start()
            if active:
                split["load"] += t1 - t0
                split["zero"] += t2 - t1
                split["forward"] += t3 - t2
                split["backward"] += t4 - t3
                split["step"] += t5 - t4
            if step == args.steps - 1:
                prof.stop()

    total = sum(split.values())
    print(f"\nwall clock over {args.steps} steps: {total:.2f}s ({total / args.steps:.2f}s/step)")
    for key, value in split.items():
        print(f"  {key:9} {value / args.steps:.3f}s/step  {value / total * 100:5.1f}%")

    table = prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=15)
    print("\ntop kernels by self CPU time:")
    print(table)

    out = ROOT / "reports" / "profile-training-step.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "steps": args.steps, "threads": args.threads, "batch": BATCH, "ctx": CTX,
        "split_seconds_per_step": {k: round(v / args.steps, 3) for k, v in split.items()},
        "top_kernels": [
            {"name": e.key, "self_cpu_ms": round(e.self_cpu_time_total / 1000, 1),
             "calls": e.count}
            for e in prof.key_averages()[:15]
        ],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
