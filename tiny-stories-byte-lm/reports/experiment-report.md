# TinyStories Byte-Level Experiment Report

**Status:** In progress. This is a running report, not a claim that training
has started or that language structure has been learned.

## 1. Environment

| Item | Observed |
|---|---|
| OS | Windows 11 Home |
| CPU | Intel Core i5-1240P, 12 cores / 16 logical processors |
| Memory | 15.68 GiB |
| Accelerator | Intel Iris Xe; no CUDA device |
| Python | CPython 3.12.14 project runtime; system Python 3.14.7 |
| Dependency manager | uv 0.12.17 |
| Training precision | CPU FP32 baseline; directly half-cast FP16 produced NaNs |
| PyTorch | 2.15.0+cpu, installed through Tsinghua PyPI mirror |
| Optional optimizer probe | torchzero 0.4.4; SciPy 1.18.1 |

The local proxy is `localhost:10077`. A 1 MiB ranged transfer test returned
about 25 KB/s; this is a point-in-time measurement, not a training throughput
estimate.

## 2. Dataset and byte statistics

**Pinned source:** [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories),
revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.

| Statistic | Result |
|---|---:|
| Official raw train rows (viewer metadata) | 2,119,719 |
| Official raw validation rows (viewer metadata) | 21,990 |
| Clean train rows | Pending full scan |
| Clean validation rows | Pending full scan and deduplication |
| Clean test rows | Pending deterministic held-out split |
| Raw UTF-8 bytes | Pending full scan |
| Train input bytes used for parameter ratio | Pending full scan |
| Validation/test byte totals | Pending full scan |
| Mean clean bytes per story | Pending full scan |
| Byte-length and structural distributions | Pending full scan |

The viewer's mean string lengths (train 896.33258; validation 872.68386) are
not UTF-8 byte totals. They are not used in parameter-budget calculations.
The dataset exposes no genre labels; structural categories will be computed
from the text and labeled as such.

The official source has no independent test split. The official validation
split is deterministically divided into validation/test by SHA-256 of exact
text, with exact train overlaps and repeated validation stories removed from
held-out data. The split seed/rule is fixed in
[`experiment-design.md`](../docs/experiment-design.md).

The local proxy download is active. The downloader validates the expected size
of each pinned shard before the UTF-8 byte scan starts.

## 3. Parameter budget and architecture

```text
Target parameters = clean train input bytes / 20
```

### Provisional 1 GB sizing scenario

Assumption supplied for this estimate: **1 GB means 1,000,000,000 actual
UTF-8 next-byte training input positions**, not compressed Parquet size and not
Unicode characters. The 1:20 target is therefore **50,000,000 trainable
parameters**. If the intended unit is 1 GiB, the target would be 53,687,091.
This is a sizing scenario only; the downloaded TinyStories statistics remain
the source of the final budget.

For the implemented pre-LN blocks (all linear biases and LayerNorm parameters
counted; fixed LUT excluded), the count is:

```text
P = L * (4*d^2 + 2*d*d_ff + 9*d + d_ff) + 258*d + 256
```

Exact counts below were verified against `trainable_parameter_count` in the
implemented model. Compute assumes context length 256 and one pass over 1B
bytes; it is an approximate training-FLOP estimate (`6P` per byte plus
causal-attention QK/AV overhead). The FP32 training-state floor includes
weights, gradients, and two Adam moments (`16P` bytes), excluding activations,
temporary buffers, allocator overhead, and the fixed LUT.

| Candidate | d_model | Layers | Heads | Head dim | d_ff | Trainable params | Bytes/param | FP32 weights + Adam state/grad floor | Approx. FLOPs / 1B bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A — balanced/deeper | 512 | 16 | 8 | 64 | 2,048 | 50,570,496 | 19.77 | 809 MB | 328.6 PFLOP |
| B — wider/shallower | 640 | 10 | 10 | 64 | 2,560 | 49,400,576 | 20.24 | 790 MB | 316.1 PFLOP |
| C — wide/shallower | 768 | 7 | 12 | 64 | 3,072 | 49,813,504 | 20.08 | 797 MB | 315.4 PFLOP |
| D — narrow/deep | 384 | 28 | 6 | 64 | 1,536 | 49,784,320 | 20.09 | 797 MB | 331.7 PFLOP |

All candidates use `d_ff = 4*d_model`, and NoPE/RoPE have the same parameter
count. Candidate A is a reasonable depth-preserving reference; B is a
conservative width/depth alternative. C is wider but only seven layers; D's
extra depth raises attention cost and execution overhead. Keep all four as
candidates until actual byte count, context-length requirements, and CPU
throughput are known; these are not approved training configurations.

The more aggressive input-dependent bilinear-mixing idea is documented as a
separate follow-up in
[`operator-mixing-hypothesis.md`](../docs/operator-mixing-hypothesis.md). It
does not replace this standard Transformer baseline or enter its parameter
comparison.

## 4. Model and configuration

| Setting | Planned value |
|---|---|
| Inputs / outputs | UTF-8 byte IDs / 256-way distribution |
| Input LUT | Fixed random `[256, d_model]`; excluded from trainable parameter count |
| Attention | Full causal |
| FFN activation | TeLU, `x * tanh(exp(x))`; finite forward/backward verified in FP32, FP16, and BF16 |
| Normalization / residual | Standard Transformer; exact variant pending candidate design |
| Output projection | Independent from input LUT; no weight tying |
| Position encoding | Explicit NoPE/RoPE config variable |
| BOS/EOS | No extra byte/token IDs; each story remains an independent sequence |
| Loss | Next-byte cross entropy |
| Optimizer | Fixed in reproducible config before training; not used as an experiment variable |
| Seed | To be fixed in the run config |

## 5. Training and evaluation results

No training run has started. Results below will be recorded at initialization
and at multiple checkpoints. UTF-8 and word measurements are evaluation-only,
not reward or training signals.

| Checkpoint / input bytes | Train loss | Validation loss | Byte PPL | LR | Grad norm | Input bytes/s | Time | Peak RAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Initial | — | — | — | — | — | — | — | — |
| Checkpoint 1 | — | — | — | — | — | — | — | — |
| Checkpoint 2 | — | — | — | — | — | — | — | — |

### Requested measurements

| Checkpoint | Strictly valid UTF-8 generations / total (95% CI) | Mean UTF-8-legal byte mass | Validation word mean log P | Validation word mean log P/byte | Validation empirical word-frequency summary |
|---|---:|---:|---:|---:|---|
| Initial | — | — | — | — | — |
| Checkpoint 1 | — | — | — | — | — |
| Checkpoint 2 | — | — | — | — | — |

Generated samples and their strict UTF-8 decode results will be included
separately. A lower byte loss does not by itself establish UTF-8 validity or
TinyStories-style story generation.

## 6. Reproduction, smoke tests, and checkpoints

| Check | Result |
|---|---|
| Dataset scanner unit tests (UTF-8 bytes, cleaning, deterministic split/dedup) | 5 passed |
| Fixed-seed initialization repeatability | Pending |
| Forward/backward finite values | Pending |
| CPU memory / throughput smoke test | Pending |
| Checkpoint save/load equivalence | Pending |
| UTF-8 strict decoding test | Pending |
| Resume from checkpoint | Pending |
| Final generated samples | Pending |

Exact commands, dependency versions, configuration, statistics JSON, metric
curves, checkpoints, and generated samples will be linked here after the
corresponding stage completes.

### CPU precision and library probes

The CPU probe used a one-layer `d_model=32`, `d_ff=64` model, batch 4, sequence
length 32, one warmup step, and three measured steps. These microbenchmark
throughputs compare local operator support only; they do not predict the final
architecture's performance.

| Threads | Precision | Full train step supported | Finite loss/gradients | Input bytes/s | TeLU finite forward/backward |
|---:|---|---|---|---:|---|
| 1 | FP32 | Yes | Yes | 12,856 | Yes |
| 1 | FP16 | Yes | **No (NaN)** | 3,317 | Yes |
| 1 | BF16 | Yes | Yes | 3,879 | Yes |
| 4 | FP32 | Yes | Yes | 1,635 | Yes |
| 4 | FP16 | Yes | **No (NaN)** | 2,073 | Yes |
| 4 | BF16 | Yes | Yes | 2,069 | Yes |

Use FP32 for the baseline. Do not use the directly half-cast FP16 model:
although the CPU kernels execute, loss and gradients become NaN. TeLU itself
produced finite forward and backward values in every tested dtype. The small
four-thread result is slower and is not evidence that higher thread counts
improve this workload. Raw probe output:
[`precision-benchmark.json`](precision-benchmark.json).

The installed `torchzero` 0.4.4 SPSA and RDSA modules each completed one
synthetic CPU step with finite loss/parameters and updated parameters. This is
API compatibility only, not a baseline optimizer or quality comparison:
[`zeroth-order-library-probe.json`](zeroth-order-library-probe.json).

| Check | Result |
|---|---|
| Dataset scanner/materialization tests | 6 passed |
| Model, causality, TeLU, and generation tests | Passed |
| Deterministic checkpoint/resume equivalence | Passed |
| Full test suite | 14 passed |
| TinyStories byte scan / exact budget | Pending completed download |
