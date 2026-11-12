# TinyStories Byte-Level Experiment

This project is being built in phases. Do not start a full training run until
the exact UTF-8 byte statistics, 1:20 parameter budget, and candidate
architectures are recorded in [`reports/experiment-report.md`](reports/experiment-report.md).

## Environment

Python is pinned with `uv` using `.python-version`. Install project
dependencies and run the focused tests with:

```powershell
uv sync
uv run python -m unittest discover -s tests -v
```

The initial environment audit found an Intel i5-1240P, 15.68 GiB RAM, Intel
Iris Xe graphics, and no CUDA device. CPU FP32 is planned; no assumption is
made that CPU FP16 is efficient.

## Dataset statistics

The source is pinned to Hugging Face revision
`f54c09fd23315a6f9c86f9dc80f725de7d8f9c64`. Place the four official train
Parquet shards and official validation Parquet shard in `data/raw/` with the
resumable downloader, then run:

```powershell
# Direct huggingface.co is blocked on this host; HF_ENDPOINT selects a mirror.
$env:HF_ENDPOINT = "https://hf-mirror.com"
uv run python scripts/download_dataset.py --connections 8
uv run python scripts/inspect_dataset.py --data-dir data/raw --output data/stats.json
```

`--endpoint` overrides `HF_ENDPOINT`, which overrides the default
`https://huggingface.co`. `--connections 1` keeps the original single resumable
curl stream; a larger value splits each file into fixed byte ranges fetched
concurrently, resumes per completed chunk, then assembles and verifies the
pinned total size. A chunk that stalls below 2 KB/s for 60 s is aborted and
retried.

The downloader verifies each pinned file size and resumes partial transfers;
if a transfer is interrupted, rerun the same command. After the download and
byte scan finish, materialize the boundary-preserving split files:

```powershell
uv run python -m scripts.prepare_dataset --data-dir data/raw --stats data/stats.json --output-dir data/processed
```

The scanner counts actual strict UTF-8 bytes, excludes only whitespace-only
and invalid UTF-8 samples, prevents exact held-out duplicates, and creates a
deterministic validation/test split from the official validation data. No
special token is added and stories are not concatenated.

See [`docs/experiment-design.md`](docs/experiment-design.md) for the frozen
design constraints and [`reports/experiment-report.md`](reports/experiment-report.md)
for current results and pending gates.

## Training pipeline

The model, CPU training loop, evaluation, checkpoint/resume, generation, and
precision probe live in `src/tinystories_byte_lm/`. The smoke configuration is
for pipeline tests only; it is not a parameter-budget candidate. Do not start a
full run before the byte scan and architecture candidates are recorded.

```powershell
uv run python -m unittest discover -s tests -v
uv run python -m tinystories_byte_lm.precision_probe --threads 1 --output reports/precision-benchmark.json
uv run python -m tinystories_byte_lm.train --config configs/smoke.json --train-data data/processed/train.bin --train-index data/processed/train.idx --validation-data data/processed/validation.bin --validation-index data/processed/validation.idx --output-dir checkpoints/smoke
```

The optional SPSA/RDSA library probe is isolated from baseline training:

```powershell
uv sync --extra optimizer-probe
uv run python scripts/probe_zeroth_order_library.py --output reports/zeroth-order-library-probe.json
```

See [`reports/experiment-report.md`](reports/experiment-report.md) for
measured CPU precision behavior and remaining data/parameter-budget gates.
