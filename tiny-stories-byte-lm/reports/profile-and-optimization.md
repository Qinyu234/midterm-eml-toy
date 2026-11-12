# Profiling and what actually moved the step time

## Where the time goes

`scripts/profile_training_step.py`, CPU pilot shape (d256/L4/ff1024, batch 32,
ctx 256, fp32, 10 threads), 6 steps:

| phase | s/step | share |
|---|---:|---:|
| data loading | 0.008 | 0.4% |
| zero_grad | 0.003 | 0.2% |
| forward | 0.573 | 32.0% |
| backward | 1.183 | 66.0% |
| optimizer step | 0.024 | 1.3% |

Top kernels by self CPU time: `aten::mm` 32.4%, `aten::addmm` 16.9%, `aten::mul` 7.5%,
CPU flash attention 6.8% (+2.4% in its backward node), `aten::copy_` 4.0%, `aten::add_`
2.1%, `aten::sum` 1.8%, then the TeLU pieces — `tanh`, `exp`, `clamp`, `where` at
1.5–1.7% each.

Two conclusions worth stating plainly:

- **It is a matmul-bound workload.** 49% of self time sits in `mm`/`addmm`; nothing
  outside the linear algebra can pay for much. Data loading, the optimizer, and the
  activation are together under 12%.
- **Batching already paid its dividend.** The loader costs 0.4% at batch 32, versus
  the 50% share it had at batch 8 (1,440 B/s), which is why batch 32 is the config.

## What was tried

| change | result | kept? |
|---|---|---|
| TeLU fused into one autograd node (exact: value+grad max diff 0.0) | 1.70 / 1.85 s vs baseline 1.96 / 1.62 s — inside the noise band | no |
| `torch.set_flush_denormal(True)` | 1.90 s vs 1.88 s | no |
| `set_num_interop_threads(1)` | 1.68–1.93 s, no consistent direction | no |
| process priority HIGH | **1.55 s** (1.56, 1.54) vs normal 1.73 s (1.85, 1.61) | already in config |
| process priority REALTIME | 1.56 s (1.56, 1.56) — no gain over HIGH | no |
| power plan: 平衡 → 高性能 | part of the improvement above; measured only after the switch | yes, machine-level |

Honest caveat: repeated runs of the *same* configuration vary 1.54–1.96 s/step
(±12%), so anything below ~10% cannot be resolved on this machine without many more
repetitions. That is why the activation fusion was rejected rather than adopted on a
favourable single reading.

Realtime priority does work here (the process can raise itself to 0x100 as
Administrator) — it simply buys nothing, because the process already gets the cores
it asks for at HIGH.

## Resulting numbers

Real training loop (`checkpoints/perf-check/metrics.jsonl`, 60 steps, HIGH priority,
high-performance plan): **median 1.72 s/step, 4,076 B/s**.

That is a correction: the earlier 2.76 s/step figure came from a run that shared the
machine with concurrent benchmarks, and it inflated the wall-clock estimate by ~75%.

| step | NoPE | RoPE (+12%) |
|---:|---:|---:|
| 1,000 | 0.48 h | 0.54 h |
| 2,000 | 0.96 h | 1.07 h |
| 4,000 | 1.91 h | 2.14 h |
| 6,000 | 2.87 h | 3.21 h |
| 7,876 (full 64.5 MB) | **3.77 h** | **4.22 h** |

Both arms now finish well inside the 8-hour budget, with 3.8–4.3 h of slack.

## What is left, if more speed is ever needed

- Fewer FLOPs per step is the only large lever left: attention cost scales with
  `n_layers × d_model`, which is why the 95M search picked the widest on-budget shape.
- Bigger batch would cut per-step overhead from 1.72 s toward the 1.55 s compute floor,
  but the loader is already 0.4% and the optimizer 1.3%, so the ceiling is ~10%.
- Half precision is off the table on this CPU: measured 0.4–0.6 GFLOPS for half matmuls
  versus 150–197 for fp32.
