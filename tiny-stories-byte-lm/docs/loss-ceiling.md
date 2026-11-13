# Where the loss stops falling, and what to run next

The NoPE run finished its full 15,752-step horizon. This note records what the
validation curve says about further training, and what the other machine should run.

## Result

| step | validation loss | train loss |
|---:|---:|---:|
| 500 | 1.929 | 1.930 |
| 2,000 | 1.135 | 1.106 |
| 5,000 | 0.949 | 0.958 |
| 8,000 | 0.886 | 0.873 |
| 10,000 | 0.845 | 0.845 |
| 12,000 | 0.830 | 0.786 |
| **15,752 (end)** | **0.801** | **0.782** |

Validation never diverged from training, so at this size the 64.5 MB slice is not
being overfit in any damaging way; the run is data-bound, not capacity-bound.

## Projection

Fit over all 32 evaluation points, `L = c + a · step^-b` by grid search:

```
L = 0.722 + 125.7 · step^-0.75        (rmse 0.014 nats)
```

| steps | predicted val | gain vs now |
|---:|---:|---:|
| 15,752 (measured 0.801) | 0.812 | — |
| 31,504 (4 passes) | 0.775 | −0.03 |
| 63,008 (8 passes) | 0.754 | −0.05 |
| 126,016 (16 passes) | 0.741 | −0.06 |

So the curve is still monotone but has flattened: the last 5,000 steps bought 0.037,
the next 5,000 are worth roughly 0.013.

**Empirical floor: 0.722 nats ≈ 1.042 bits/byte.** For reference, uniform over 256
bytes is 5.545 nats and the unigram baseline on this corpus is 3.072; the model is at
0.801 (1.156 bits/byte). Natural English runs near 1.0–1.3 bits/byte, and TinyStories
uses a deliberately simple vocabulary, so ~1.04 bits/byte is close to the entropy of
this data — the 3.2M model has already extracted most of what 64.5 MB contains.

Two caveats worth stating rather than hiding:

- the fit assumes more steps keep buying information, but repeating the same slice
  does not add information, so the real floor appears earlier than 0.722
- moving below it requires a different variable: more unique data (96% of the 1.9 GB
  corpus is still unused) or more parameters — not more steps on 64.5 MB

Decision: **do not extend the NoPE run.** Spend the remaining budget on the metrics
`docs/experiment-design.md` requires and on the second machine's arm.

## What the second machine runs

Per `docs/cross-machine-handoff.md`: `configs/cpu-8h-rope.json`, 15,752 steps, batch
32, context 256, checkpoint every 250, eval every 500, seed 2718, same 64.5 MB cap.
The only difference from this machine's finished run is `position_encoding: rope`.

Observing that the RoPE arm's loss falls faster is a legitimate signal, because loss
per step is machine-independent, but only if the rest of the recipe matches. Check it
before drawing a conclusion:

```bash
python scripts/compare_runs.py checkpoints/cpu-cont checkpoints/cpu-cont-rope \
    --labels nope rope
```

The first line of output is the config check: batch, steps, cosine horizon, warmup,
seed, eval cadence and data cap must all agree, because those determine the learning
rate at every step. A shorter cosine horizon alone produces a "faster falling" curve
that has nothing to do with positional encoding. Then it prints validation loss at
matching step numbers and the gap between the arms.

What to send back: `metrics.jsonl`, `gen-quality.jsonl`, and the benchmark JSON. Do not
send checkpoints — they are 39 MB each and the numbers are what matters.
