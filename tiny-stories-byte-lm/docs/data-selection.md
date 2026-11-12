# How the CPU training slice was cut from the full corpus

The short version: **a prefix cut, not a sample** — take the first 64,522,880 bytes of
the concatenated clean training stream, keep only stories that end inside that prefix,
and sample windows from it randomly.

## 1. Cleaning (whole corpus)

Source: `roneneldan/TinyStories`, revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`.
Text is preserved verbatim; the only rows dropped are the ones that carry no text.

| step | count |
|---|---:|
| official train rows | 2,119,719 |
| invalid UTF-8 rows | 0 |
| whitespace-only rows | 230 |
| **clean stories** | **2,119,489** |
| clean story bytes | 1,902,941,607 |
| usable next-byte positions | 1,900,822,118 |

Each story's last byte is not a training input, because targets are the input shifted
by one and stories never cross boundaries. That is the difference between
"story bytes" and "next-byte positions".

## 2. Packed into a flat byte stream

- clean stories are concatenated **in parquet row order** into `data/processed/train.bin`
- `data/processed/train.idx` stores `<offset, length>` per story (`QQ`), which is how
  a byte position is later resolved back to a story boundary
- no BOS/EOS token, no shuffling, no length stratification

## 3. The CPU slice

```
keep the story iff  offset + length ≤ 64,522,880
```

A prefix of the stream. When the boundary falls inside a story, the whole story is
dropped rather than truncated, so no half story is ever trained on.

| item | value |
|---|---:|
| stories kept | **72,409** |
| stories dropped | 2,047,080 |
| usable input bytes | **64,449,976** |
| share of corpus | 3.4% |
| next-byte positions per parameter | 64,449,976 / 3,225,344 = **19.98** |

## 4. Window sampling during training

- a byte position is drawn uniformly from `[0, 64,449,976)` **with replacement**
- that position resolves to a story; the window runs from it for at most 256 bytes and
  stops at the story's end
- the 32 windows of a batch have different lengths and are padded to the longest,
  padding carries the ignore target and contributes no loss

Measured consequence: effective input is **6,568–7,325 bytes/step**, not the nominal
`32 × 256 = 8,192`, because short stories end windows early.

## 5. What is deliberately *not* done

- no shuffle of story order — the slice is a contiguous prefix of the corpus
- no length stratification — story lengths span 0–127 to 2048+ bytes naturally
- no cross-split deduplication for training; deduplication only applies when building
  the held-out validation/test split
- no guarantee of equal exposure — sampling is with replacement, so within two passes
  a story may be seen zero times or three times

## 6. Evaluation data is separate

`data/processed/validation.bin` — 7,715 stories, 6,491,512 bytes, used in full, never
trained on. `test.bin` holds 7,674 stories / 6,430,285 bytes and is untouched by the CPU
experiment.
