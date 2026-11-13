# CPU one-pass experiment (8-hour budget)

## Why this shape

CPU throughput is `~9.0e3 * P(M)^-0.84` bytes/s. Solving the 8-hour budget together
with the contract rule `parameters = bytes / 20` puts the feasible corner near
P = 3.2M / 64.5 MB, i.e. one pass over a 3.4% slice of the 1.9 GB corpus.

Measured serial throughput at that shape (8 threads, ctx 256, fp32):

| batch | s/step | bytes/s |
|---:|---:|---:|
| 8 | 0.79 | 2,606 (compute only) |
| 16 | 1.60 | 2,563 (compute only) |
| 32 | 3.01 | 2,721 (compute only) |
| 64 | 5.96 | 2,750 (compute only) |

Batch size barely moves compute throughput, but raising it from 8 to 32 cut the
data-loading share: the real training loop ran at 1,440 B/s with batch 8 and at
**2,868 B/s with batch 32** (7,221 valid bytes per 2.46 s step).

## The run

`configs/cpu-8h.json`

| field | value |
|---|---|
| d_model / layers / heads / d_ff | 256 / 4 / 4 / 1,024 (d_model = 16²) |
| trainable parameters | 3,226,144 |
| training bytes | 3,226,144 x 20 = 64,522,880 (61.5 MiB, 3.4% of the corpus) |
| steps | 7,876 at batch 32 (8,192 bytes/step); there is no wall-clock stop, the step count is the horizon |
| optimizer | AdamW, β=(0.9, 0.95), eps 1e-8, wd 0.1 on matrices only |
| lr | 6e-4 peak, 160-step linear warmup, cosine to 0.1× |
| grad clip | global norm 1.0 |
| precision | fp32 (CPU) |

Projected wall clock at the measured 2,868 B/s: `64.5e6 / 2868 ≈ 5.4e4 s ≈ 6.2 h`,
inside the 8-hour bound with margin. Checkpoints every 1,000 steps, eval every 500.

## Running on the full corpus with a byte cap

If the full processed data is on hand, no slicing step is needed. `--train-bytes`
(and the `training.train_bytes` config field) caps how much of the byte file is usable;
the loader drops stories that would straddle the cut instead of truncating them.

```
python -m tinystories_byte_lm.train --config configs/cpu-8h.json \
  --train-data data/processed/train.bin --train-index data/processed/train.idx \
  --validation-data data/processed/validation.bin --validation-index data/processed/validation.idx \
  --output-dir checkpoints/cpu-8h --train-bytes 64522880
```

`configs/cpu-8h.json` already carries `train_bytes: 64522880` and
`validation_bytes: 6491512`, so the CLI flags are optional. Verified: the capped view
of the full corpus gives exactly the same 72,409 stories and 64,449,976 next-byte
positions as a pre-cut copy, so both routes train identically.

### If the data has to move by hand

The same slice can also be shipped as one archive — useful when the other machine
cannot download from the mirror. `scripts/pack_cpu_pilot.py` writes:

```
data/cpu-pilot-data.zip          22.1 MB (31% of the 72.3 MB source)
data/cpu-pilot-data.zip.sha256   418c347baa4cab19cc3808b8b8712f9e52c03c26de116a1983ddd6d5173c651c
```

Unpack so `cpu-pilot/` sits under `data/`, then point `--train-data` at
`data/cpu-pilot/train.bin` and drop the `--train-bytes` flag. Verified with a five
step run: loss 3.67, validation byte-perplexity 34.7, 2,866 B/s. The archives are
gitignored — they move by file transfer, not by `git push`.

## Expected wall clock

Measured at this shape (batch 32, ctx 256, fp32, 10 threads, high priority,
high-performance power plan — see `reports/profile-and-optimization.md`):

| measurement | NoPE | RoPE |
|---|---:|---:|
| compute only, s/step | 1.55 | 1.74 (+12%) |
| compute only, B/s | 5,300 | 4,730 |
| real training loop, s/step | 1.72 | 1.93 |
| real training loop, B/s | 4,076 | 3,640 |

Cumulative milestones from the real-loop rate (eval adds ~0.04 h total):

| step | bytes seen | NoPE | RoPE |
|---:|---:|---:|---:|
| 1,000 | 8.2 MB | 0.48 h | 0.54 h |
| 2,000 | 16.4 MB | 0.96 h | 1.07 h |
| 4,000 | 32.8 MB | 1.91 h | 2.14 h |
| 6,000 | 49.2 MB | 2.87 h | 3.21 h |
| 7,876 (full 64.5 MB) | 64.5 MB | **3.77 h** | **4.22 h** |

Both arms fit inside the 8-hour budget with ~3.8–4.3 h of slack. An earlier estimate
of 6.0 h came from a run that shared the machine with concurrent benchmarks; treat
only the numbers above as current. Same-configuration repeat runs vary ±12% on this
laptop, so plan with that margin.

## What counts as a visible result

Reference points, all measured or computed on this machine:

| reference | byte loss (nats) | byte perplexity |
|---|---|---|
| uniform 256-way init | ln 256 = 5.545 | 256 |
| unigram baseline on the first 80 MB | 3.0718 | 21.6 |
| this model at step 31 (2.5 min in) | 2.497 | 12.1 |

The model is already 0.575 nats below the unigram floor after 31 steps. A
completed pass should land near 1.5-2.0 nats (perplexity 4.5-7.4) for this much
TinyStories data, i.e. 1.1-1.6 nats under unigram, far outside the noise of a
409,600-byte evaluation (standard error on loss at that sample size is ~0.01 nats).

Reporting rule: quote train loss, the validation byte-perplexity from
`metrics.jsonl`, and the gap against both the unigram baseline and the uniform
prior. The gap, not the raw loss, is the significance claim.

## Reproduce

```powershell
uv run python -m tinystories_byte_lm.train --config configs/cpu-8h.json `
    --train-data data/processed/train.bin --train-index data/processed/train.idx `
    --validation-data data/processed/validation.bin --validation-index data/processed/validation.idx `
    --output-dir checkpoints/cpu-8h
```

Resume with `--resume-from checkpoints/cpu-8h/checkpoint-XXXXXXXX.pt`.

## Thread count, affinity, and priority

Same 3.23M shape, batch 32, ctx 256, fp32, one warmup + 4 timed steps:

| setup | threads | s/step | bytes/s |
|---|---:|---:|---:|
| unpinned, normal | 12 | 2.52 | 3,256 |
| unpinned, normal | 8 | 2.43 | 3,372 |
| unpinned, normal | 9 | 2.35 | 3,485 |
| unpinned, normal | 10 | 2.42 | 3,391 |
| unpinned, **high** | **10** | **2.18** | **3,750** |
| unpinned, high | 8 | 2.23 | 3,675 |
| pinned to 6 physical cores | 6 | 3.03 | 2,706 |
| pinned to 6 physical, high | 6 | 2.82 | 2,904 |
| pinned to 6 physical, above-normal | 6 | 2.90 | 2,821 |
| pinned to 4 physical cores | 4 | 4.02 | 2,038 |
| pinned to 3 physical cores | 3 | 4.72 | 1,735 |
| pinned to 8 logical (6 physical + 2 SMT) | 8 | 2.68 | 3,059 |

Two conclusions:

1. **Do not pin.** Every affinity mask loses to leaving the scheduler free, because
   Zen 2 SMT siblings add ~15-20% on this model and a 6-core mask throws them away.
   Pinning to the 6 physical cores costs 28% versus unpinned 8 threads.
2. **Raise priority and use 10 threads.** High priority on 10 unpinned threads is the
   fastest combination measured (3,750 B/s, +11% over 8 threads at normal priority).
   Both are now config fields: `num_threads`, `process_priority`, `cpu_affinity_mask`
   (Windows only, applied through `apply_process_tuning`).

Measured in the real training loop (loader included): 2,751 B/s with 10 threads at
high priority, versus 1,440 B/s at batch 8 / 8 threads. That is the number that
matters for the 8-hour projection.

## The other machine

The branch carries everything needed: `git fetch origin && git checkout
add-tinystories-byte-lm`, then `uv sync`, then download the dataset
(`HF_ENDPOINT=https://hf-mirror.com`, `--connections 8`) and run `prepare_dataset`.
Configs are committed: `configs/cpu-8h.json` (NoPE) and `configs/cpu-8h-rope.json`
(RoPE, same shape and recipe, differs only in `position_encoding`).

Only run the RoPE arm on the second machine if its measured throughput differs
clearly from this one; if the two machines land within a few percent, the RoPE
comparison is cheaper to run back-to-back on one machine where the throughput
number is already characterized.

## Scope caveat

The run does not sample from the whole pool. It takes a **prefix cut** of the
concatenated clean stream, keeping only stories that end inside the first
64,522,880 bytes: 72,409 stories, 3.4% of the corpus, with no shuffle and no
length stratification (the exact rule is in
[`data-selection.md`](data-selection.md)). So "one pass" means one pass over that
64.5 MB prefix, not an epoch of the whole corpus. Anything claimed about the
95M-parameter budget must come from a run with a config at that scale, which this
machine cannot do in reasonable time.
