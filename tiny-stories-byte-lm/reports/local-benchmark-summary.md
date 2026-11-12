# Local CPU benchmark

Two machines are being compared with the same script, the same reference shape,
and the same torch version. Raw output lives next to this file as
`precision-benchmark-<host>.json`.

## Reproduce on a machine

```powershell
uv sync
uv run python scripts/benchmark_local_cpu.py --threads 1,2,4,8,12 `
    --probe-steps 10 --probe-warmup 3 --warmup-steps 1 --steps 3 `
    --output reports/precision-benchmark-<HOSTNAME>.json
```

The script needs only `torch`; it does not read `data/` and does not depend on
the OS. `--skip-throughput` skips the reference-shape part for a quick sweep.

## Machines

| | WIN-1OHFIPSVANM (measured) | baseline in `precision-benchmark.json` |
|---|---|---|
| CPU | AMD Ryzen 5 5500U (Zen 2, 6C/12T) | Intel 12th gen (i5-1240P per `docs/experiment-design.md`) |
| Logical CPUs | 12 | 16 |
| RAM | 15.34 GiB | 15.68 GiB (from the design doc, not recorded in that JSON) |
| torch | 2.15.0+cpu | 2.15.0+cpu |

The committed `precision-benchmark.json` was produced on the Intel machine and
records only `platform.processor()`, so its RAM and core count are not
reproducible from the file. The new report records host, CPU model string,
logical CPU count, total RAM, and torch version for each machine.

## Precision and thread sweep (tiny probe: d_model 32, 1 layer, ctx 32, batch 4)

Input bytes per second on WIN-1OHFIPSVANM:

| threads | fp32 | fp16 | bf16 |
|---:|---:|---:|---:|
| 1 | 23,202 | 5,756 (NaN loss) | 6,326 |
| 2 | 23,418 | 7,066 (NaN loss) | 7,585 |
| 4 | 22,585 | 7,225 (NaN loss) | 8,305 |
| 8 | 19,723 | 7,809 (NaN loss) | 8,079 |
| 12 | 16,216 | 7,005 (NaN loss) | 7,464 |

Readings:

- fp32 is fastest and peaks at 1-2 threads; the model is too small for the
  16 KiB-per-thread intra-op overhead to pay off, so more threads make it
  slower. This is a probe artifact, not a statement about the real model.
- FP16 is supported: forward, loss, and backward all produce finite values and
  the gradient norm matches fp32 (0.9169 vs 0.9169). The NaN comes from
  `AdamW(eps=1e-8)`, whose epsilon is below the FP16 smallest normal (6.1e-5)
  and rounds to zero there, so `0 / 0` in the update produces NaN on the first
  `optimizer.step()`. Measured on the same tiny model:

  | variant | result |
  |---|---|
  | fp16 + AdamW eps 1e-8 | params non-finite after step 1, loss NaN from step 2 |
  | fp16 + AdamW eps 1e-3 | finite, loss 5.6641 -> 5.3086 over 4 steps |
  | fp16 + SGD | finite, loss 5.6641 -> 5.6445 over 4 steps |
  | fp16 + AdamW eps 1e-8 + loss scale 1024 | still NaN, so this is not gradient underflow |
  | bf16 + AdamW eps 1e-8 | finite, loss 5.6875 -> 5.4375 |
  | fp32 + AdamW eps 1e-8 | finite, loss 5.6646 -> 5.4312 |

  So "FP16 works" is true for compute and for optimizers whose epsilon survives
  the format; the baseline stays fp32 because this CPU has no fast FP16 path
  (7.8k B/s vs 19.7k B/s at 8 threads) and because bf16 needs no epsilon fix.
- bf16 stays finite but is ~2.7x slower than fp32 at every thread count.


## Reference-shape training throughput

Shape: `d_model 384`, 8 layers, 6 heads, `d_ff 1536`, context 256, batch 8,
14,295,040 trainable parameters, batch reused across the measured steps so the
number is a compute ceiling rather than a data-loader measurement.

| threads | bytes/s | s/step | peak RSS |
|---:|---:|---:|---:|
| 1 | 405 | 5.06 | 1,070 MiB |
| 2 | 609 | 3.36 | 1,092 MiB |
| 4 | 789 | 2.59 | 1,092 MiB |
| 8 | 957 | 2.14 | 1,103 MiB |
| 12 | 834 | 2.46 | 1,133 MiB |

8 threads is the plateau; 12 threads regresses (SMT and throttling).

## Plain GEMM benchmark (random matrices, no model)

`torch.mm` on random square matrices, 8 threads, 2 warmup + 5 timed ops. This
isolates arithmetic throughput from the model and from the optimizer:

| size | fp32 | bf16 | fp16 |
|---|---|---|---|
| 512x512 | 1.5 ms (173 GFLOPS) | 468 ms (0.6 GFLOPS) | 456 ms (0.6 GFLOPS) |
| 1024x1024 | 14.2 ms (151 GFLOPS) | 4,919 ms (0.4 GFLOPS) | 4,900 ms (0.4 GFLOPS) |
| 2048x2048 | 87.1 ms (197 GFLOPS) | not measured | not measured |

fp32 gets ~150-200 GFLOPS from the oneDNN/MKL path. fp16 and bf16 get 0.4-0.6
GFLOPS, i.e. **300-400x slower per operation**, because this build has no fast
CPU kernel for half precision and falls back to a slow path. The 2048 runs were
dropped for the same reason: at 0.4 GFLOPS one op takes ~43 s.

This is the real explanation for why every model-level fp16 timing looked like
a freeze: the matmuls themselves are three orders of magnitude slower, so a
handful of steps at the reference shape takes many minutes with no output.

Which means: fp16 on this CPU is *correct* but not *fast*. Half precision at
the 95M-parameter budget would be slower than fp32, not faster.

## Feasibility note

Parameter budget, stated explicitly: `target trainable parameters =
train.next_byte_input_bytes / 20`, i.e. 20 next-byte input positions per
trainable parameter. With the measured `next_byte_input_bytes` of 1,900,822,118
that is 95,041,105 trainable parameters.

At 957 B/s a full pass over the measured `train.next_byte_input_bytes`
(1,900,822,118) takes about `1.9e9 / 957 ≈ 2.0e6 s ≈ 23 days`. Scaling by the
parameter ratio to the 95,041,105-parameter budget (~6.6x) gives an order-of-
magnitude estimate above 100 days per epoch on this CPU. A CPU-only
single-epoch run over the whole dataset is therefore out of reach; the
architecture/budget decision has to account for a byte subset or a different
compute target. This is an estimate from one reference shape, not a measured
full run.

## Smoke run

`python -m tinystories_byte_lm.train` with `configs/smoke.json` over
`data/processed` completed on 2 steps (loss 5.5748 → 5.5381, validation
perplexity 271.5 → 240.2), so the materialized splits are wired correctly.
