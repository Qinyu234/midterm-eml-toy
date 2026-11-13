# What the 3.2M model learned, and what it did not

Probes run on `checkpoints/cpu-cont/checkpoint-00015752.pt` (15,752 steps, val 0.801
nats). Script: `scripts/audit_learned.py`. Every number below is a rate that can be
reproduced; none of it is a reading of sample text.

## The numbers

| probe | result | what it means |
|---|---|---|
| next-byte accuracy, overall | **74.9%** (256-way, 140,371 held-out positions) | 3 of 4 bytes predicted exactly |
| — inside a word | **76.1%** | it finishes words it has begun |
| — at a word start | **70.7%** | choosing *which* word comes next is the weak spot |
| word completion, first byte | **75.0%** (200 real words, 3-byte suffix) | orthography is solid |
| word completion, whole 3-byte suffix | **63.0%** | it knows how words are spelled, not just their first letter |
| illegal UTF-8 in generation | **0%** (was 33% at step 250) | byte distribution converged onto valid text |
| OOV word rate | 6.2% at step 2,750, falling | it reuses corpus vocabulary instead of inventing |
| self-repetition (5-grams repeated inside one sample) | **12.1%** | mild looping, not a collapse |
| sentence length | **10.6 words generated vs 9.3 corpus** | story grammar and sentence scale reproduced |
| 12-gram overlap with training slice | 88.4% generated vs **91.4% unseen validation** | composes rather than recites |
| 16-gram overlap | 65.3% vs **73.3%** | same |
| 24-gram overlap | 25.2% vs **33.3%** | same |

## Do not trust the 6-gram copy rate

The first version of the copy probe reported 99.4% and looked like wholesale
memorisation. Its own control killed it: at n=6, *held-out* validation text scores
99.8% against the same reference set. English at 6 bytes is so redundant that the
metric cannot separate copying from generalising, so it was replaced by n=12/16/24.

At those lengths the model overlaps the training slice **less** than natural
unseen text does. It is not reciting stories; it is recombining phrases, and its
failures are not plagiarism but incoherence.

## Learned

1. **Orthography.** 63% exact 3-byte word-suffix completion. It knows that
   *happ* goes to *y*, that *beau* goes to *tiful*.
2. **Local syntax.** 76.1% inside-word accuracy, 0% illegal bytes, sentence lengths
   within 1.3 words of the corpus. Subject-verb-object shapes, quotation marks,
   `Once upon a time` openers and `The X was so happy` closers all reproduce.
3. **Phrase collocation.** Real multi-word templates: "so happy that", "he had
   found", "the little girl was". This is what makes the samples read as English
   for two or three sentences.
4. **Composition, not memorisation.** Proven by the gram controls above.

## Not learned

1. **Word selection.** The 5.4-point gap (76.1% inside word vs 70.7% at word start)
   is the whole story in one number: it can finish any word it has started, but
   nearly 30% of the time it starts the wrong one. That is exactly the layer where
   meaning lives — `delicious brush`, `the subway started with everyone`,
   `no one coat cans` are all failures of selection, not of spelling.
2. **Entity and causal tracking.** Names drift (Tweetie becomes "the bird"), objects
   teleport (a boat appears in a jungle story), a character's stated emotion never
   motivates anything that follows. Nothing in the model carries "who is doing what
   to whom" past one sentence.
3. **Long-range structure.** 256-byte context, and the run shows no evidence it uses
   more than roughly one sentence of it.
4. **Code.** Zero. Its inputs are code-shaped, where indentation and newlines are
   distribution outliers; at T=0.2 the model degenerates into letter soup
   (`theat and the stairs and and that she he`).

## Reading

The model learned the **surface statistics of the language** — spelling, morphology,
local syntax, collocations, sentence rhythm — and did not learn the **situation model**
behind the text. At 3.2M parameters on 64.5 MB that split is the expected outcome:
surface statistics are cheap to store (a few million parameters hold a great deal of
English orthography and phrase structure), while a situation model needs capacity that
this scale does not have.

The 0.801 nat loss is consistent with this: it corresponds to roughly 2.2 effective
choices per byte, which is exactly what fluent-but-vacuous prose looks like. The
remaining distance to the 0.722 nat floor would buy the model slightly better
collocations, not coherence — which is why the run was stopped here rather than
extended.
