"""CPU next-byte training entry point; use configs/smoke.json for local testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ModelConfig
from .training import TrainConfig, run_training


def load_config(path: Path) -> tuple[ModelConfig, TrainConfig, int | None, int | None]:
    with path.open(encoding="utf-8") as config_file:
        raw = json.load(config_file)
    model_config = ModelConfig(**raw["model"])
    training = raw["training"]
    train_config = TrainConfig(
        steps=training["steps"],
        batch_size=training.get("batch_size", 4),
        learning_rate=training["learning_rate"],
        eval_interval=training.get("eval_interval", 1),
        eval_batches=training.get("eval_batches", 1),
        checkpoint_interval=training.get("checkpoint_interval", 1),
        seed=training["seed"],
        precision=training.get("precision", "fp32"),
        num_threads=training.get("num_threads"),
        warmup_steps=training.get("warmup_steps", 0),
        lr_schedule=training.get("lr_schedule", "constant"),
        min_lr_ratio=training.get("min_lr_ratio", 0.1),
        weight_decay=training.get("weight_decay", 0.0),
        grad_clip=training.get("grad_clip"),
        cpu_affinity_mask=training.get("cpu_affinity_mask"),
        process_priority=training.get("process_priority", "normal"),
    )
    return (
        model_config,
        train_config,
        training.get("train_bytes"),
        training.get("validation_bytes"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--validation-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--train-bytes",
        type=int,
        help="Use only the first N bytes of the training split (subset of a full corpus).",
    )
    parser.add_argument(
        "--validation-bytes",
        type=int,
        help="Use only the first N bytes of the validation split.",
    )
    args = parser.parse_args()

    model_config, train_config, config_train_bytes, config_validation_bytes = load_config(args.config)
    train_bytes = args.train_bytes if args.train_bytes is not None else config_train_bytes
    validation_bytes = (
        args.validation_bytes if args.validation_bytes is not None else config_validation_bytes
    )
    records = run_training(
        model_config,
        train_config,
        args.train_data,
        args.train_index,
        args.validation_data,
        args.validation_index,
        args.output_dir,
        args.resume_from,
        train_max_bytes=train_bytes,
        validation_max_bytes=validation_bytes,
    )
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
