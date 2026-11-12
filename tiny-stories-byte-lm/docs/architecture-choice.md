# Architecture choice at the 95M budget

**Decision: `d_model=1024` (32²), `n_layers=9`, `d_ff=3072`, `n_heads=16` — 94,746,880 parameters.**

## How it was chosen

Two phases, because a full model sweep on CPU is too slow to iterate:

1. **Matrix estimate** (`scripts/estimate_shapes_from_gemm.py`): time square matmuls
   at every size the candidate would touch (256…6912), then convert each shape's
   analytic per-step FLOP count into seconds. This ranks shapes before any model is
   built.
2. **Top-k measurement** (`scripts/measure_shape_topk.py`): run one warmup plus three
   real AdamW steps on the four on-budget shapes the estimate liked, at batch 32,
   fp32, 10 threads.

## Phase 1 — GEMM throughput (float32, 10 threads, this machine)

| n | GFLOPS | n | GFLOPS |
|---:|---:|---:|---:|
| 256 | 174.9 | 1280 | 145.4 |
| 512 | 211.3 | 1536 | 151.5 |
| 576 | 223.3 | 1600 | 157.2 |
| 768 | 175.6 | 2048 | 186.7 |
| 1024 | 163.8 | 2304 | 179.5 |
| 3072 | 185.3 | 4608 | 196.5 |

Per-step FLOPs at 95M: **d1024 = 2.55 TFLOPs**, d1089 = 2.55, d576 = 2.79, d1600 = 2.68,
d256 = 3.26. The wide-shallow shapes win because attention cost scales with
`n_layers × d_model`, and at a fixed parameter budget `n_layers ≈ P / (10 d²)`, so
attention FLOPs fall as `1 / d_model`. Narrowing from 1024 to 256 adds 28% FLOPs
that buy nothing.

Calibration against the one shape already measured end to end (d1089/L8, 38.95 s/step)
gives a scale factor of 2.67× — the gap between pure GEMM and a real step is optimizer
update, layernorm, activation traffic and the byte-embedding lookup.

## Phase 2 — measured (batch 32, ctx 256, fp32, 10 threads, high priority)

| shape | params | s/step | bytes/s | vs best |
|---|---:|---:|---:|---:|
| **d1024 / L9 / ff3072** | **94,746,880** | **41.50** | **197.4** | **1.00×** |
| d1089 / L8 / ff3267 | 95,259,442 | 45.34 | 180.7 | 1.09× |
| d576 / L29 / ff2304 | 96,564,352 | 51.95 | 157.7 | 1.25× |
| d256 / L144 / ff768 | 94,880,512 | ~50 (est) | ~410 est-shot | worst by estimate |

Estimate error on the measured shapes: +0% to +8% — good enough to rank, not good
enough to quote a single number.

## Why d1024 specifically

- **Fastest measured at equal parameters** — 197 B/s, 9% ahead of d1089 and 25% ahead
  of d576. Nothing on budget beat it.
- **Power of two and a multiple of 64**, so every matmul lands on a shape the BLAS
  kernels like; `d_ff=3072` keeps the 3× ratio and is also a multiple of 64.
- **head_dim = 64**, so the RoPE arm is legal on the same shape — d1089 has head_dim 99
  (odd) and cannot run RoPE at all without changing the head count.
- 0.31% under the parameter budget, so the 20:1 token-to-parameter rule still holds.

## What this costs

Full training budget (1,900,822,118 bytes, 232,034 steps at batch 32):

| machine | time |
|---|---|
| this CPU (197 B/s) | **111 days** |
| single H100 (analytic, 6·P·bytes) | ~0.8 h |
| single 4090 | ~2.6 h |
| single T4 | ~13.3 h |

The CPU cannot train this architecture on the full corpus. The 8-hour CPU arm stays
at `d256/L4/ff1024` (3.23M parameters) — that experiment tests the pipeline and
learnability, not the architecture choice. d1024 is the shape to carry to a GPU.
