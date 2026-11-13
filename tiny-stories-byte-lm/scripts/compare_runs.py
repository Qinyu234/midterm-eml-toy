"""Compare two training runs at matching steps.

Loss per step is machine-independent, so this is the honest way to compare an arm run
on a faster or slower computer: line up the step numbers, not the wall clock.

    python scripts/compare_runs.py checkpoints/cpu-cont checkpoints/cpu-cont-rope \
        --labels nope rope

Checks that both runs share the parameters that must match for the comparison to mean
anything (batch, context, data cap, seed, steps-per-eval), then prints the validation
loss at each common step plus the gap.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MATCH_KEYS = (
    "batch_size",
    "steps",
    "seed",
    "eval_interval",
    "eval_batches",
    "train_bytes",
    "validation_bytes",
    "learning_rate",
    "warmup_steps",
    "lr_schedule",
    "min_lr_ratio",
    "weight_decay",
    "grad_clip",
)


def load(run: Path) -> tuple[dict, list[dict]]:
    metrics = [json.loads(line) for line in (run / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    config_path = run / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    return config, metrics


def training_of(config: dict) -> dict:
    return config.get("training", config)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--labels", nargs="+", default=None)
    args = ap.parse_args()

    labels = args.labels or [p.name for p in args.runs]
    loaded = [load(p) for p in args.runs]

    reference = training_of(loaded[0][0])
    for label, (config, _) in zip(labels[1:], loaded[1:]):
        other = training_of(config)
        mismatched = [
            f"{key}: {reference.get(key)} vs {other.get(key)}"
            for key in MATCH_KEYS
            if reference.get(key) != other.get(key)
        ]
        print(f"config check {labels[0]} vs {label}: " + ("identical" if not mismatched else "; ".join(mismatched)))

    val: dict[str, dict[int, float]] = {}
    for label, (_, metrics) in zip(labels, loaded):
        val[label] = {
            row["step"]: row["validation_loss"]
            for row in metrics
            if "validation_loss" in row
        }

    steps = sorted(set.intersection(*(set(v) for v in val.values())))
    header = f"{'step':>7} " + " ".join(f"{label:>9}" for label in labels)
    if len(labels) == 2:
        header += f" {'gap':>9}"
    print(header)
    print("-" * len(header))
    for step in steps:
        line = f"{step:>7} " + " ".join(f"{val[label][step]:>9.3f}" for label in labels)
        if len(labels) == 2:
            line += f" {val[labels[0]][step] - val[labels[1]][step]:>+9.3f}"
        print(line)
    if not steps:
        print("no common validation steps yet")


if __name__ == "__main__":
    main()
