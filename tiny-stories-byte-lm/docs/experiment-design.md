# TinyStories Byte-Level Experiment Design

## Status and execution gate

This document records the experiment contract, not experimental conclusions.
The full source-data download and exact byte scan are in progress. Architecture
candidates and the final parameter budget must be reported before a full training
run; no hidden-size or depth is preselected.

## Scope

- Train a standard autoregressive Transformer over UTF-8 bytes with 256 outputs.
- Use a fixed random `[256, d_model]` lookup table; it is not trainable and is
  excluded from the trainable-parameter budget.
- Use full causal attention, TeLU feed-forward activation, standard residuals
  and normalization, and an independent 256-way output projection.
- Keep NoPE and RoPE as explicit, separately configurable position-encoding
  experiments; do not silently select one as a conclusion.
- Use next-byte cross entropy and a standard optimizer. Evaluation metrics do
  not enter the loss, optimizer, or any reward.
- Exclude diffusion, flow matching, SNN, zeroth-order optimization, JEPA,
  dynamic tokenization, VQ/latent tokens, MoE, personalization, and data
  augmentation from this phase.
- Keep input-dependent bilinear mixing as a separate follow-up hypothesis.
  Its math questions and staged evaluation proposal are recorded in
  [`operator-mixing-hypothesis.md`](operator-mixing-hypothesis.md); it is not
  part of the baseline model or its training run.

## Environment (initial audit)

- Windows 11 Home; Intel Core i5-1240P (12 cores / 16 logical processors);
  15.68 GiB RAM; Intel Iris Xe integrated graphics; no CUDA device.
- System Python is 3.14.7. `uv` 0.12.17 is available, and CPython 3.12.14 is
  installed for the project environment.
- PyTorch was not installed at the initial audit. Training will use CPU and
  FP32 unless a later local capability check establishes a safe alternative.
  CPU FP16 performance is not assumed.

## Data source, cleaning, and split

- Source: [`roneneldan/TinyStories`](https://huggingface.co/datasets/roneneldan/TinyStories),
  pinned to revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`, converted
  Parquet configuration `default`.
- The source provides official `train` and `validation` splits, but no
  independent `test` split. The official training split remains training.
  The official validation split is partitioned deterministically by SHA-256
  digest of the exact UTF-8 text: even first digest byte goes to validation,
  odd goes to test.
- Text is preserved verbatim. Exclude only whitespace-only samples and rows
  that fail strict UTF-8 encoding. Remove exact train overlaps and repeated
  validation stories from held-out data so identical stories cannot cross
  train/validation/test boundaries. The script records all excluded counts.
- Keep each story as an independent sequence; never concatenate adjacent
  stories. No BOS/EOS bytes or extra token IDs are added. Vocabulary and output
  remain exactly 256 byte values.
- For a story of `n` bytes, teacher forcing uses `x[0:n-1]` to predict
  `x[1:n]`; therefore the parameter-budget denominator is the sum of actual
  input bytes, `max(n - 1, 0)`, across clean training stories. The first byte
  has no preceding byte context and is not included in this next-byte loss.
  Generation/evaluation will use documented byte prompts, not an unannounced
  BOS token.
- The source has only a `text` column, not genre or text-type labels. The
  statistics script therefore reports observable structure (length bins,
  single/multiline text, and presence of dialogue quotation marks), not
  inferred genres.

### Upstream metadata (not byte totals)

The Hugging Face dataset viewer reported 2,119,719 train rows and 21,990
validation rows. Its string-length statistics are not UTF-8 byte statistics
and are included only as a cross-check:

| Split | Reported mean length | Median | Min | Max |
|---|---:|---:|---:|---:|
| Train | 896.33258 | 787 | 0 | 5,499 |
| Validation | 872.68386 | 777.5 | 61 | 4,272 |

Compressed Parquet file sizes at the pinned revision:

| File | Bytes |
|---|---:|
| train shard 1 | 248,731,111 |
| train shard 2 | 248,171,980 |
| train shard 3 | 245,894,874 |
| train shard 4 | 247,988,350 |
| validation shard | 9,989,127 |

Exact raw/clean sample counts, UTF-8 byte totals, post-dedup split totals,
average byte length, and byte-length distributions are pending the local scan.
Do not infer them from compressed sizes or character lengths.

## Model and parameter-budget decisions

The target is computed only after the scan:

```text
Target trainable parameters = clean training input bytes / 20
```

For the provisional 1 GB sizing scenario (1,000,000,000 actual UTF-8
next-byte input positions), the target is 50,000,000 trainable parameters.
This is a user-selected scenario, not the measured TinyStories total; the final
budget must use `data/stats.json`. If "1 GB" means 1 GiB, the corresponding
target is 53,687,091 parameters.

For the current pre-LN block implementation, with biases enabled and no
learned position parameters:

```text
P = n_layers * (4*d_model^2 + 2*d_model*d_ff + 9*d_model + d_ff)
    + 258*d_model + 256
```

The fixed LUT is excluded. NoPE and RoPE have identical trainable parameter
counts in this implementation.

Trainable output, attention, feed-forward, normalization, and any learned
position-related parameters count toward the budget. The fixed LUT does not.
The exact model parameter count is to be calculated from each candidate
configuration, then compared with this target.

Candidate architectures must include `d_model`, layer/head counts, `head_dim`,
`d_ff`, exact parameter count, CPU RAM/activation estimate, context length, and
compute cost. Context length is an explicit design input because full causal
attention cost and memory depend on it. Provisional 1 GB candidates are recorded
in `reports/experiment-report.md`; none is approved until the actual byte
budget is available.

LUT initialization remains independently configurable. The intended starting
family is a zero-mean isotropic Gaussian with element standard deviation
`1/sqrt(d_model)` (expected row norm near one), seeded and stored in config;
this is a proposal pending measured norm/scale and numerical-stability checks.
It must not encode byte values, bits, or character structure.

TeLU is `f(x) = x * tanh(exp(x))`. Its implementation is checked for
finite forward/backward values in FP32 before training. Position encoding,
LUT initialization, model size, and activation are not to be varied together.

## Training, evaluation, and reproducibility

- Loss: standard next-byte cross entropy over each story's shifted byte
  sequence; no cross-story targets.
- The optimizer and all training hyperparameters will be written to a
  versioned config before the run. The same seed/config is used for
  reproducibility; checkpoints and resume behavior are tested.
- Zeroth-order optimization remains outside the baseline. A separate
  optional-library compatibility probe may exercise SPSA/RDSA on a tiny
  synthetic model, but it does not change the baseline optimizer or enter
  training results. The probe must use the library API rather than a local
  reimplementation.
- Record train/validation loss, byte-level perplexity, learning rate, gradient
  norm, input bytes/s, elapsed time, and peak process memory. Save multiple
  checkpoints. Report CPU throughput rather than substituting an accelerator
  or silently shrinking the requested experiment.
- Keep an untouched held-out test split for final evaluation and generated
  examples. Any validation/test distinction and any data cleaning exclusion
  is recorded in the statistics JSON.
- UTF-8 and word metrics below are measurement-only. They never influence
  gradients, loss, optimizer state, or reward.

### Additional requested measurements

At initialization and each selected checkpoint, use the same fixed generation
prompts, sample count, maximum byte count, sampling parameters, and seed:

1. **Strict UTF-8 sequence validity:** valid generated continuations divided by
   all generated continuations, using strict UTF-8 decode. Also report
   Wilson 95% confidence intervals so a small sample is not presented as a
   precise probability.
2. **UTF-8-legal next-byte probability mass:** at each generated prefix, sum
   the model's 256-way probability over bytes allowed by the UTF-8 decoder
   state; report the mean across positions. This is distinct from empirical
   sequence validity and from constrained decoding.
3. **Validation-file word probability:** tokenize words for evaluation only
   (Unicode-aware regex; case-folded for frequency counts). For each word
   span, compute the model's conditional byte-sequence log probability given
   preceding bytes from that validation story. Report mean log probability
   per word, mean log probability per byte, and byte-normalized perplexity.
   This evaluation tokenizer does not alter model inputs or the 256-byte
   vocabulary.
4. **Validation-file empirical word probability:** count each case-folded word
   in validation and divide by the total validation word count. Report common
   word frequencies and distribution summaries separately from model
   likelihood; corpus frequency is not a training objective.

All metrics are plotted against training input bytes/steps, with initial and
checkpoint values retained in machine-readable output. UTF-8 sequence
validity, byte prediction quality, and story-distribution quality are reported
as separate capabilities.

### Optional zeroth-order library capability probe

The candidate probe library is `torchzero`, which exports SPSA and RDSA
gradient-approximation modules through its PyTorch optimizer interface. Verify
the installed version and run one finite, tiny-model closure step for each
method. This only records whether the library API integrates with this model
and CPU precision; it does not compare optimizer quality or use either method
in the TinyStories baseline. The project still excludes zeroth-order
optimization from the experiment itself.

## Current implementation

- `scripts/inspect_dataset.py` performs the exact byte scan, deterministic
  validation/test assignment, held-out duplicate filtering, and JSON report.
- `scripts/prepare_dataset.py` materializes the splits as story-indexed byte
  files and checks materialized row/byte counts against the statistics.
- `tests/test_inspect_dataset.py` covers UTF-8 byte counting, whitespace
  filtering, bucket boundaries, deterministic assignment, and materialization.
- `src/tinystories_byte_lm/` contains the configurable Transformer, CPU training
  loop, checkpoint/resume, generation, and FP32/FP16/BF16 probe. The training
  entry point is `python -m tinystories_byte_lm.train`.
- `scripts/probe_zeroth_order_library.py` checks the optional `torchzero`
  SPSA/RDSA APIs on a synthetic model only; neither optimizer is part of the
  baseline.
- `data/stats.json` will be generated after all pinned files are present.
- Final experiment outputs belong under `reports/` and checkpoints under
  `checkpoints/`; raw downloaded data is kept under `data/raw/` and is not a
  source-code artifact.
