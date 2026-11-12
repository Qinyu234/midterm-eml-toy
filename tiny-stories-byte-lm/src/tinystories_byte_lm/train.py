"""CPU next-byte training entry point; use configs/smoke.json for local testing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ModelConfig
from .training import TrainConfig, run_training


def load_config(path: Path) -> tuple[ModelConfig, TrainConfig]:
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
    )
    return model_config, train_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--train-index", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--validation-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()

    model_config, train_config = load_config(args.config)
    records = run_training(
        model_config,
        train_config,
        args.train_data,
        args.train_index,
        args.validation_data,
        args.validation_index,
        args.output_dir,
        args.resume_from,
    )
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
