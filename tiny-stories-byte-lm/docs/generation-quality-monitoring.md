# Tracking generation quality while training

Loss tells you the model fits the data; it does not tell you whether the samples
read as English. Two measurements answer that directly, both computed from samples
generated with fixed seeds so checkpoints stay comparable.

## Metrics

| metric | meaning | what "good" looks like |
|---|---|---|
| `invalid_utf8_rate` | share of samples that fail a strict UTF-8 decode | → 0: the model stops emitting byte sequences no text file could contain |
| `illegal_byte_share` | share of emitted bytes that are neither printable ASCII, whitespace, nor part of a valid UTF-8 character | → 0: control bytes and broken multi-byte tails disappear first |
| `oov_word_rate` | share of generated words absent from the training slice's vocabulary (17,067 distinct words in the first 64.5 MB) | falls toward 0: the model invents spelling instead of reusing words it has seen |
| `words` | word tokens in the sample set, a sanity check that the text is not all punctuation | rises then stabilises |

Invention is the interesting one. A byte model that has learned the corpus
reproduces its spelling; one that is still guessing produces plausible-looking but
unseen words, which `oov_examples` lists verbatim.

## Usage

```bash
# one checkpoint, with the generated text
python scripts/sample_checkpoint.py checkpoints/cpu-cont/checkpoint-00000500.pt

# every new checkpoint, one JSON row each, skipping steps already recorded
python scripts/eval_generation_quality.py checkpoints/cpu-cont \
    --jsonl checkpoints/cpu-cont/gen-quality.jsonl --skip-samples
```

The eval is incremental: it reads the steps already present in `--jsonl` and only
measures checkpoints that are new, so it can be re-run on a schedule while training
continues.

## First observation

At step 250 (loss ≈ 2.1) on the 3.23M model: one third of samples fail strict UTF-8,
0.08% of bytes are illegal, and 57.8% of the 237 generated words are spelled in ways
the corpus never uses. That is the expected shape for a model this early — it has
learned word-length byte clusters but not the vocabulary. The curve to watch is
`oov_word_rate` falling and `invalid_utf8_rate` reaching zero.

## How the run is scheduled

The training and the monitor both run as Windows scheduled tasks, because a process
started from an interactive shell is reaped when that shell's session ends:

- `bitter-lesson-cpu-cont` — the training loop, `configs/cpu-continuous.json`
- `bitter-lesson-quality-monitor` — the incremental eval, every 15 minutes

Check `metrics.jsonl` for loss and `gen-quality.jsonl` for the two quality metrics.
