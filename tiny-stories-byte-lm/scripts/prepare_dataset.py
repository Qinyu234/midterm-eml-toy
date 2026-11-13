"""Materialize the pinned TinyStories splits as boundary-safe byte datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

from scripts.inspect_dataset import (
    SOURCE_REVISION,
    TRAIN_FILES,
    VALIDATION_FILE,
    SplitStats,
    iter_parquet_texts,
    validation_fold,
)
from tinystories_byte_lm.data import write_byte_dataset


def _check_written_counts(
    split_name: str,
    written: dict[str, int],
    expected: dict[str, Any],
) -> None:
    actual = {
        "clean_rows": written["stories"],
        "clean_story_bytes": written["story_bytes"],
        "next_byte_input_bytes": written["next_byte_input_bytes"],
    }
    if any(actual[key] != expected[key] for key in actual):
        raise ValueError(
            f"Materialized {split_name} data disagrees with statistics: "
            f"expected {expected}, got {actual}"
        )


def prepare_dataset(
    data_dir: Path,
    stats_path: Path,
    output_dir: Path,
    batch_size: int = 65536,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    if stats.get("revision") != SOURCE_REVISION:
        raise ValueError("Statistics do not match the pinned TinyStories revision")
    if stats.get("data_dir") != str(data_dir.resolve()):
        raise ValueError("Statistics were generated from a different data directory")

    train_hashes: set[bytes] = set()

    def clean_train_stories() -> Iterator[bytes]:
        split_stats = SplitStats()
        for text in iter_parquet_texts(
            [data_dir / name for name in TRAIN_FILES], batch_size
        ):
            encoded = split_stats.add(text)
            if encoded is not None:
                train_hashes.add(hashlib.sha256(encoded).digest())
                yield encoded

    output_dir.mkdir(parents=True, exist_ok=True)
    train_written = write_byte_dataset(
        clean_train_stories(), output_dir / "train.bin", output_dir / "train.idx"
    )
    _check_written_counts("train", train_written, stats["train"])

    validation_stats = SplitStats()
    validation_hashes: set[bytes] = set()
    heldout: dict[str, list[bytes]] = {"validation": [], "test": []}
    for text in iter_parquet_texts([data_dir / VALIDATION_FILE], batch_size):
        encoded = validation_stats.add(text)
        if encoded is None:
            continue
        digest = hashlib.sha256(encoded).digest()
        if digest in train_hashes or digest in validation_hashes:
            continue
        validation_hashes.add(digest)
        heldout[validation_fold(encoded)].append(encoded)

    materialized: dict[str, dict[str, int]] = {"train": train_written}
    for split_name in ("validation", "test"):
        written = write_byte_dataset(
            heldout[split_name],
            output_dir / f"{split_name}.bin",
            output_dir / f"{split_name}.idx",
        )
        _check_written_counts(split_name, written, stats[split_name])
        materialized[split_name] = written

    manifest = {
        "dataset": stats["dataset"],
        "revision": SOURCE_REVISION,
        "encoding": "UTF-8 bytes, one byte per token",
        "special_tokens": "none",
        "story_boundaries": "preserved by one index record per story",
        "statistics_file": str(stats_path.resolve()),
        "splits": {
            split_name: {
                "data_file": f"{split_name}.bin",
                "index_file": f"{split_name}.idx",
                **materialized[split_name],
            }
            for split_name in ("train", "validation", "test")
        },
    }
    manifest_path = output_dir / "manifest.json"
    temporary_path = manifest_path.with_name(manifest_path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_path, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--stats", type=Path, default=Path("data/stats.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--batch-size", type=int, default=65536)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_dataset(args.data_dir, args.stats, args.output_dir, args.batch_size),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
