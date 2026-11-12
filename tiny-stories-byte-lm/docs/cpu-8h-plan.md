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
| training bytes | 3,226,144 × 20 = 64,522,880 (64.5 MB, 3.4% of the corpus) |
| steps | 7,876 at batch 32 (8,192 bytes/step) |
| optimizer | AdamW, β=(0.9, 0.95), eps 1e-8, wd 0.1 on matrices only |
| lr | 6e-4 peak, 160-step linear warmup, cosine to 0.1× |
| grad clip | global norm 1.0 |
| precision | fp32 (CPU) |

Projected wall clock at the measured 2,868 B/s: `64.5e6 / 2868 ≈ 5.4e4 s ≈ 6.2 h`,
inside the 8-hour bound with margin. Checkpoints every 1,000 steps, eval every 500.

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

## Scope caveat

The run consumes 64.5 MB of byte positions sampled from the full training pool; it
does not materialize a separate slice file. So "one pass" means one pass over
64.5 MB worth of tokens drawn from the 1.9 GB pool, not a single epoch of the whole
corpus. Anything claimed about the 95M-parameter budget must come from a run with
`configs/` at that scale, which this machine cannot do in reasonable time.
