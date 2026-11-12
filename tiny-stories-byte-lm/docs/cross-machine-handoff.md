# Cross-machine handoff: NoPE here, RoPE there

Two machines run the same 3.23M byte model on the same 64.5 MB slice. The only
difference is the positional treatment: **this machine (Ryzen 5 5500U) runs NoPE, the
other machine runs RoPE.** Everything else — steps, batch, schedule, checkpoints, eval
set — is identical on purpose, so the two runs are comparable.

## Exact parameters of the current run (NoPE arm)

| item | value |
|---|---|
| model file | `configs/cpu-continuous.json` |
| parameters | 3,226,144 |
| d_model / layers / heads / d_ff | 256 / 4 / 4 / 1,024 |
| context length | 256 |
| positional encoding | `nope` |
| activation | TeLU (repo default), `dropout: 0.0` |
| batch size | 32 (one step consumes 8,192 byte positions) |
| steps | 15,752 (two passes over the same slice) |
| learning rate | 6e-4, linear warmup 200, cosine to 0.1× |
| AdamW | β = (0.9, 0.95), eps 1e-8, weight decay 0.1 on ndim ≥ 2 only |
| gradient clipping | 1.0 |
| eval | every 500 steps, 20 validation batches |
| checkpoints | every 250 steps |
| seed | 2718 |
| precision | fp32 |
| threads / priority / affinity | 10 / high / none |
| train bytes | 64,522,880 (capped by `train_bytes`, 3.4% of the 1.9 GB corpus) |
| validation bytes | 6,491,512 (unchanged, never trained on) |

Measured on this machine: **1.6–1.9 s/step, 3,470–4,300 B/s**, ~3.5 h for the 7,876-step
first pass.

## The RoPE arm

`configs/cpu-8h-rope.json` is byte-for-byte the same recipe with
`"position_encoding": "rope"` and `d_model / n_heads = 64` (even head dimension, so
RoPE is legal). On this machine RoPE costs about **+12%** per step; plan for that on
matching hardware.

## Commands on the other machine

```bash
git fetch origin
git checkout add-tinystories-byte-lm
git pull
cd tiny-stories-byte-lm
# if uv rewrote the lock file's package source, undo it:
git checkout -- uv.lock

python scripts/download_dataset.py --help   # only if the parquet files are absent
python scripts/pack_dataset_splits.py --help  # not needed: train/validation already split

python -m tinystories_byte_lm.train \
    --config configs/cpu-8h-rope.json \
    --train-data data/processed/train.bin --train-index data/processed/train.idx \
    --validation-data data/processed/validation.bin --validation-index data/processed/validation.idx \
    --output-dir checkpoints/cpu-cont-rope
```

Add `--train-bytes 64522880` if the config is not picked up. Then, whenever a new checkpoint
appears:

```bash
python scripts/eval_generation_quality.py checkpoints/cpu-cont-rope \
    --jsonl checkpoints/cpu-cont-rope/gen-quality.jsonl --skip-samples
python scripts/sample_checkpoint.py checkpoints/cpu-cont-rope/checkpoint-00001000.pt
```

## What to send back

1. `checkpoints/cpu-cont-rope/metrics.jsonl` — loss, learning rate, gradient norm, s/step
2. `checkpoints/cpu-cont-rope/gen-quality.jsonl` — the two quality metrics per checkpoint
3. The benchmark JSON from `python scripts/benchmark_local_cpu.py` — so per-step speed
   can be compared across machines rather than guessed

## How to read the comparison

Speed is machine-bound; **quality metrics are not**. `invalid_utf8_rate`, `illegal_byte_share`
and `oov_word_rate` at the same step number answer "does RoPE help this model learn
language faster", while `s/step` answers "which machine is quicker". Keep those two
questions apart — with one arm per machine, a joint conclusion would confound them.

## Things that broke on this machine

- `huggingface.co` is blocked here; `hf-mirror.com` works, hence `--endpoint`.
- numpy is absent from this venv; torch warns on every start, harmless.
- `uv run` / `uv sync` rewrite `uv.lock` to a different package index; restore it with
  `git checkout -- uv.lock` before committing.
- A process started from an interactive shell dies with that shell. Training and the
  quality monitor run as Windows scheduled tasks instead.
