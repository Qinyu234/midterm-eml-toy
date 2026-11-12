# Model specification: what every number means

This is the authoritative definition set for both machines. Anything quoted elsewhere
that disagrees is a bookkeeping error; the rule is "measure the live model, not the formula".

## Architecture (byte-level, no learned embedding)

Verified by instantiating `configs/cpu-continuous.json` and summing trainable tensors:

| module | tensors | shape | params |
|---|---:|---|---:|
| `blocks.*` (4 layers) | 48 | see below | **3,159,040** |
| `final_norm` | 2 | weight+bias (256) | 512 |
| `output_projection` | 2 | weight 256×256, bias 256 | 65,792 |
| **total trainable** | **52** | | **3,225,344** |

Per layer (12 tensors, 789,760 params):

| tensor | shape | params |
|---|---|---:|
| attention_norm.weight / .bias | 256 | 256 + 256 |
| attention.qkv.weight / .bias | 768×256 / 768 | 196,608 + 768 |
| attention.output.weight / .bias | 256×256 / 256 | 65,536 + 256 |
| mlp_norm.weight / .bias | 256 | 256 + 256 |
| mlp up-project.weight / .bias | 1024×256 / 1024 | 262,144 + 1,024 |
| mlp down-project.weight / .bias | 256×1024 / 256 | 262,144 + 256 |

**There is no embedding table.** Input bytes are mixed into `d_model` by a fixed
random lookup table (`lut_seed`, default 0, this run 1729), which is why the only
byte-facing trainable tensor is the 256×256 `output_projection`. Consequences worth
keeping straight when comparing against a normal transformer:

- parameter counts are not comparable one-to-one with GPT-style models of the same
  `d_model` (no `256 × d_model` embedding, no tied head)
- changing `lut_seed` changes the model's initial representation without changing any
  parameter count, so seed must be recorded alongside the config
- `attention.qkv` is fused: one 768×256 projection produces query, key and value

Architecture knobs, all in `model`:

| key | this run | meaning |
|---|---:|---|
| `d_model` | 256 | residual width, must equal `n_heads × head_dim` |
| `n_layers` | 4 | transformer blocks |
| `n_heads` | 4 | attention heads, `head_dim = 64` |
| `d_ff` | 1,024 | MLP hidden width, `4 × d_model` |
| `context_length` | 256 | maximum bytes visible at once, also the causal window |
| `position_encoding` | `nope` | `nope` = causal mask only; `rope` needs even `head_dim` |
| `lut_seed` | 1729 | seed of the fixed byte-mixing table |
| `dropout` | 0.0 | no dropout at this size |

## What one "step" is

```
1 step = 1 optimizer update over 1 batch
batch  = 32 independent windows
window = up to 256 byte positions, cut at a story boundary
⇒ nominal 32 × 256 = 8,192 next-byte predictions per step
```

Three definitions that must not drift between machines:

1. **Nominal bytes per step** = `batch_size × context_length` = 8,192. This is what
   `steps × 8,192` uses to convert steps into bytes consumed.
2. **Effective bytes per step** is lower, because each window stops at the end of the
   story it starts in and the batch is padded with an ignore target: measured
   **6,568–7,325** input bytes per step (80–89% of nominal). This is why logged
   `input_bytes_per_second` is below `8,192 / s_per_step`.
3. **Steps are the only stopping condition.** There is no wall-clock limit and no
   epoch counter; sampling is uniform with replacement over the capped byte range, so
   running past `steps` counted once is "another pass", not an error.

## Data

| item | value |
|---|---|
| corpus | TinyStories, revision `f54c09fd23315a6f9c86f9dc80f725de7d8f9c64` |
| full train stream | `data/processed/train.bin`, 1,902,941,607 B, 2,119,489 stories |
| train index | `data/processed/train.idx`, `<QQ` (offset, length), 33,911,824 B |
| validation | `data/processed/validation.bin`, 6,491,512 B, 7,715 stories — never trained on |
| this run's cap | `train_bytes: 64,522,880` (stories whose end ≤ 64,522,880; ~72,409 stories) |

`train_bytes` is a **read cap, not a normalization**: under it the loader simply never
sees later stories. `64,522,880 / 3,225,344 = 20.01` next-byte positions per parameter —
the 20:1 convention, expressed in **bytes** (byte-level model, so 1 byte = 1 token).
The convention governs data volume; it does not cap steps.

## Training hyperparameters (`configs/cpu-continuous.json`)

| key | value | note |
|---|---:|---|
| `steps` | 15,752 | 2 passes of the 64.5 MB cap |
| `batch_size` | 32 | |
| `learning_rate` | 6e-4 | peak, reached after warmup |
| `warmup_steps` | 200 | linear 0 → 6e-4 |
| `lr_schedule` | cosine | decays to `min_lr_ratio × peak` |
| `min_lr_ratio` | 0.1 | floor 6e-5 |
| `weight_decay` | 0.1 | applied to ndim ≥ 2 tensors only |
| `grad_clip` | 1.0 | global norm |
| optimizer | AdamW | β = (0.9, 0.95), eps = 1e-8 |
| `eval_interval` | 500 | validation loss, 20 batches |
| `checkpoint_interval` | 250 | 63 checkpoints over the run |
| `seed` | 2718 | data order, dropout, init jitter |
| `precision` | fp32 | fp16 is usable but useless here: half matmuls run 0.4–0.6 GFLOPS vs 150+ for fp32 |
| `num_threads` | 10 | 6 physical cores; 12 threads is 4% slower |
| `process_priority` | high | worth ~11% over normal; realtime adds nothing |
| `cpu_affinity_mask` | null | pinning to physical cores costs 28% |
| `validation_bytes` | 6,491,512 | the whole validation split |

## Arms

| arm | config | position encoding | machine |
|---|---|---|---|
| NoPE | `configs/cpu-continuous.json` | `nope` | this machine (Ryzen 5 5500U) |
| RoPE | `configs/cpu-8h-rope.json` | `rope` | second machine |

Both: 3,225,344 parameters, 15,752 steps, batch 32, ctx 256, checkpoint every 250, eval
every 500, same seed, same data cap. The only difference is `position_encoding`,
so any quality difference at the same step is attributable to it.

## Known bookkeeping error

Docs written before this measurement quote **3,226,144** parameters. The live count is
**3,225,344** — 800 more than reality in the earlier figure, from counting the fused
`qkv` bias twice. The gap is 0.025%; it changes the 20:1 byte budget by 16 KB and no
step count. Numbers here supersede the older ones.
