"""Phase 2: measure real training-step time for the top-k shapes from the GEMM estimate.

Reads reports/shape-estimate-from-gemm.json, keeps shapes within 2% of the 95M
budget, takes the k fastest by calibrated estimate, and times one warmup plus
three real AdamW steps on random bytes at batch 32.

Usage: python measure_shape_topk.py [--k 3]
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
BATCH = 32
CTX = 256
VOCAB = 256
PARAM_TOLERANCE = 0.02


def pick_heads(d: int) -> int:
    """Largest head count that keeps head_dim >= 32 and divisible, RoPE-friendly when possible."""
    for heads in (16, 12, 11, 9, 8, 6, 4):
        if d % heads == 0 and d // heads >= 32:
            return heads
    return 4


def measure(d: int, layers: int, ff: int, timed: int, threads: int) -> dict:
    cfg = ModelConfig(d_model=d, n_layers=layers, n_heads=pick_heads(d), d_ff=ff,
                      context_length=CTX, position_encoding="nope", dropout=0.0)
    model = ByteTransformer(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), eps=1e-8)
    g = torch.Generator().manual_seed(2718)
    inputs = torch.randint(0, VOCAB, (BATCH, CTX), generator=g)
    targets = torch.randint(0, VOCAB, (BATCH, CTX), generator=g)

    model.train()
    peak_mib = 0.0
    times = []
    for i in range(timed + 1):
        t0 = time.perf_counter()
        logits = model(inputs)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        dt = time.perf_counter() - t0
        if i:  # skip the allocator warmup step
            times.append(dt)
        try:
            import ctypes

            peak_mib = max(peak_mib, ctypes.windll.psapi.GetProcessMemoryInfo)
        except Exception:
            pass

    return {
        "params": n_params,
        "seconds_per_step": round(statistics.median(times), 2),
        "bytes_per_second": round(BATCH * CTX / statistics.median(times), 1),
        "step_times": [round(t, 2) for t in times],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--timed", type=int, default=3)
    ap.add_argument("--threads", type=int, default=10)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    est_path = ROOT / "reports" / "shape-estimate-from-gemm.json"
    estimates = json.loads(est_path.read_text(encoding="utf-8"))

    on_budget = [s for s in estimates["shapes"]
                 if abs(s["params"] - TARGET_PARAMS) / TARGET_PARAMS <= PARAM_TOLERANCE]
    on_budget.sort(key=lambda s: s["calibrated_seconds_per_step"])
    top = on_budget[: args.k]
    print(f"on-budget shapes: {[(s['d_model'], s['n_layers']) for s in on_budget]}")
    print(f"top-{args.k} by estimate: {[(s['d_model'], round(s['calibrated_seconds_per_step'], 2)) for s in top]}\n")

    results = []
    for s in top:
        m = measure(s["d_model"], s["n_layers"], s["d_ff"], args.timed, args.threads)
        entry = {**{k: s[k] for k in ("d_model", "n_layers", "d_ff", "params", "calibrated_seconds_per_step")}, **m}
        entry["estimate_error_pct"] = round(
            (m["seconds_per_step"] - s["calibrated_seconds_per_step"]) / m["seconds_per_step"] * 100, 1)
        print(f"d{s['d_model']:<5} L{s['n_layers']:<3} measured {m['seconds_per_step']:>6.2f} s/step "
              f"(est {s['calibrated_seconds_per_step']:>6.2f})  {m['bytes_per_second']:>6.1f} B/s")
        results.append(entry)

    best = min(results, key=lambda r: r["seconds_per_step"])
    total_bytes = 1_900_822_118
    steps = total_bytes / (BATCH * CTX)
    print(f"\nfastest measured: d{best['d_model']}/L{best['n_layers']} {best['seconds_per_step']} s/step, "
          f"{best['bytes_per_second']} B/s")
    print(f"full-budget epoch: {steps:,.0f} steps -> {steps * best['seconds_per_step'] / 86400:.1f} days")

    out = ROOT / "reports" / "shape-topk-measured.json"
    out.write_text(json.dumps({"batch_size": BATCH, "context_length": CTX, "threads": args.threads,
                               "total_training_bytes": total_bytes,
                               "results": results}, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
