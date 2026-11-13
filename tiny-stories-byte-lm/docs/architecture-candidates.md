# Architecture candidates at the measured parameter budget

Target: `target trainable parameters = train.next_byte_input_bytes / 20`
= 1,900,822,118 / 20 = **95,041,105**.

Widths are restricted to perfect squares (`d_model = k^2`), per the user's
constraint. Parameter counts use the contract formula in
`docs/experiment-design.md` (fixed LUT excluded, biases included):

```text
P = n_layers * (4*d_model^2 + 2*d_model*d_ff + 9*d_model + d_ff) + 258*d_model + 256
```

Every candidate was instantiated with `ByteTransformer` and the counted
trainable parameters matched the formula exactly.

## Feasible square-width shapes

`d = k^2`, `d_ff` in {3d, 4d}, depth solved to land near the target:

| d_model | k | d_ff | d_ff/d | layers | params | error |
|---:|---:|---:|---:|---:|---:|---:|
| 576 | 24 | 2,304 | 4 | 24 | 95,880,064 | +0.88% |
| 625 | 25 | 2,500 | 4 | 20 | 94,074,006 | -1.02% |
| 676 | 26 | 2,704 | 4 | 17 | 93,547,164 | -1.57% |
| 729 | 27 | 2,916 | 4 | 15 | 95,989,873 | +1.00% |
| 784 | 28 | 3,136 | 4 | 13 | 96,221,360 | +1.24% |
| 841 | 29 | 3,364 | 4 | 11 | 93,698,589 | -1.41% |
| 900 | 30 | 3,600 | 4 | 10 | 97,549,456 | +2.64% |
| 961 | 31 | 2,883 | 3 | 10 | 92,715,614 | -2.45% |
| **1024** | 32 | 3,072 | 3 | 9 | 94,746,880 | **-0.31%** |
| **1089** | 33 | 3,267 | 3 | 8 | 95,259,442 | **+0.23%** |
| 1156 | 34 | 3,468 | 3 | 7 | 93,939,128 | -1.16% |

## Measured cost on WIN-1OHFIPSVANM (AMD Ryzen 5 5500U, 8 threads, CPU fp32)

One training step, context 256, batch 8 (2,048 bytes per step), one warmup step
before timing, single timed step:

| candidate | d | layers | d_ff | heads | params | s/step | bytes/s | CPU epoch days |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| d1089 / L8 / ff3267 | 1089 | 8 | 3267 | 11 | 95,259,442 | 10.58 | 194 | 114 |
| d1024 / L9 / ff3072 | 1024 | 9 | 3072 | 16 | 94,746,880 | 11.53 | 178 | 124 |
| d576 / L24 / ff2304 | 576 | 24 | 2304 | 9 | 95,880,064 | 12.85 | 159 | 138 |

All three pay ~6·P FLOPs per byte, so the shape spread is 20%, not a different
asymptotic class: on CPU, fewer and wider layers win.

Note `d1089` only divides evenly by 11 heads (head_dim 99); if head_dim should
stay a multiple of 64, `d1024` with 16 heads is the cleaner choice at -0.31%.

## GPU time estimate

Per-epoch compute is `6·P·bytes ≈ 1.08e18` FLOPs (attention excluded; at
context 256 it adds under 10%). Hours below are `1.08e18 / (peak·MFU)`, single
GPU, no data-loading overhead:

| GPU | assumed MFU | effective TFLOPS | hours/epoch |
|---|---:|---:|---:|
| H100 SXM | 40% | 396 | 0.8 |
| A100 80GB | 45% | 140 | 2.1 |
| RTX 4090 | 35% | 115 | 2.6 |
| L4 | 35% | 42 | 7.1 |
| RTX 3090 | 40% | 28 | 10.6 |
| T4 | 35% | 23 | 13.3 |

All candidates take the same time within 2%, because the cost is set by
parameters and bytes, not by shape.

These are analytic estimates, not measurements: this host has the CPU-only
torch build, so no GPU was run. The MFU assumptions are the optimistic part.
At batch 8 and context 256 a step is only 2,048 tokens, which leaves a GPU
starved; expect the real number to land at 1.5-2.5x the table until batch is
raised (gradient accumulation to 32-64 sequences/step). Report the GPU model
and the figure can be pinned with a measured step instead.

## What this changes

On CPU the full dataset is 114-138 days per epoch for any candidate, so the
CPU path has to run on a byte subset. On a single 4090 the same epoch is
~2.6 hours (~6 hours at the pessimistic MFU), so the full 1.9 GB / 95M-parameter
budget is entirely feasible there. The architecture choice stops being a
compute-constrained decision and becomes an actual depth-versus-width
experiment:

- `d1024 / L9 / ff3072`: cleanest (16 heads, head_dim 64), -0.31% of budget.
- `d1089 / L8 / ff3267`: closest to budget (+0.23%), fastest measured, but
  head_dim 99.
- `d576 / L24 / ff2304`: depth-heavy control at the same budget, for testing
  whether depth beats width once compute stops mattering.
