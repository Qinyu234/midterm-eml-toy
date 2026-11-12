"""Condense the three candidate throughput logs into per-candidate time estimates.

Usage: python analyze_candidates.py
Reads checkpoints/cand-*/metrics.jsonl, reports steady-state throughput and
extrapolates to the full training budget.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOTAL_BYTES = 1_900_822_118  # next_byte_input_bytes from data/stats.json
BATCH = 32
CTX = 256

CANDIDATES = {
    "cand-d1089": (1089, 8, 3267, 11, 95_259_442),
    "cand-d1024": (1024, 9, 3072, 16, 94_746_880),
    "cand-d576": (576, 24, 2304, 9, 95_880_064),
}


def steady(rows: list[dict]) -> tuple[float, float, float]:
    """Drop the first two steps (allocator warmup), return median s/step, B/s, peak MiB."""
    body = rows[2:] if len(rows) > 3 else rows
    s_step = statistics.median(r["elapsed_seconds"] for r in body)
    bps = statistics.median(r["input_bytes_per_second"] for r in body)
    peak = max(r["peak_process_memory_bytes"] for r in rows) / 2**20
    return s_step, bps, peak


def main() -> None:
    print(f"total training bytes: {TOTAL_BYTES:,} | batch {BATCH} x ctx {CTX}\n")
    header = f"{'candidate':12} {'params':>10} {'s/step':>8} {'B/s':>7} {'peak GiB':>9} {'steps':>8} {'epoch h':>9} {'epoch d':>8}"
    print(header)
    print("-" * len(header))
    results = {}
    for name, (d, layers, ff, heads, params) in CANDIDATES.items():
        path = ROOT / "checkpoints" / name / "metrics.jsonl"
        if not path.exists():
            print(f"{name:12} log missing")
            continue
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(rows) < 2:
            print(f"{name:12} log too short ({len(rows)} steps)")
            continue
        s_step, bps, peak_mib = steady(rows)
        steps = TOTAL_BYTES / (BATCH * CTX)
        epoch_s = steps * s_step
        print(
            f"{name:12} {params:>10,} {s_step:>8.2f} {bps:>7.0f} {peak_mib / 1024:>9.2f} "
            f"{steps:>8.0f} {epoch_s / 3600:>9.0f} {epoch_s / 86400:>8.1f}"
        )
        results[name] = {
            "params": params,
            "shape": {"d_model": d, "n_layers": layers, "d_ff": ff, "n_heads": heads},
            "steps_logged": len(rows),
            "seconds_per_step": round(s_step, 2),
            "bytes_per_second": round(bps),
            "peak_process_memory_mib": round(peak_mib),
            "epoch_steps": round(steps),
            "epoch_hours": round(epoch_s / 3600, 1),
            "epoch_days": round(epoch_s / 86400, 1),
            "final_train_loss": round(rows[-1]["train_loss"], 4),
        }

    if len(results) >= 2:
        print("\nrelative cost (fastest = 1.00):")
        fastest = max(r["bytes_per_second"] for r in results.values())
        for name, r in sorted(results.items(), key=lambda kv: -kv[1]["bytes_per_second"]):
            print(f"  {name:12} {fastest / r['bytes_per_second']:.2f}x   (epoch {r['epoch_days']} d)")

    out = ROOT / "reports" / "candidate-throughput.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"total_training_bytes": TOTAL_BYTES, "batch_size": BATCH,
                               "context_length": CTX, "candidates": results}, indent=2) + "\n",
                   encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
