"""TinyStories byte-counting and leakage-aware split audit.

The source is pinned to the Hugging Face revision recorded in this module.
Text is preserved verbatim; only empty/whitespace-only and invalid UTF-8
samples are excluded. The official validation split is deterministically
partitioned into validation/test, and exact duplicates of training stories
are excluded from both held-out splits. No special tokens are added.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

SOURCE_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
TRAIN_FILES = tuple(f"train-{i:05d}-of-00004.parquet" for i in range(4))
VALIDATION_FILE = "validation-00000-of-00001.parquet"
BUCKETS = (
    (0, 127, "0-127"),
    (128, 255, "128-255"),
    (256, 511, "256-511"),
    (512, 1023, "512-1023"),
    (1024, 2047, "1024-2047"),
    (2048, None, "2048+"),
)


def length_bucket(length: int) -> str:
    for lower, upper, label in BUCKETS:
        if length >= lower and (upper is None or length <= upper):
            return label
    raise ValueError(f"Unsupported byte length: {length}")


def validation_fold(text_bytes: bytes) -> str:
    """Assign one story consistently; identical text always lands in one fold."""
    digest = hashlib.sha256(text_bytes).digest()
    return "validation" if digest[0] % 2 == 0 else "test"


@dataclass
class SplitStats:
    rows: int = 0
    valid_utf8_rows: int = 0
    invalid_utf8_rows: int = 0
    blank_rows: int = 0
    clean_rows: int = 0
    valid_source_bytes: int = 0
    clean_story_bytes: int = 0
    next_byte_input_bytes: int = 0
    byte_length_histogram: Counter[str] = field(default_factory=Counter)
    format_counts: Counter[str] = field(default_factory=Counter)

    def add(self, text: str) -> bytes | None:
        self.rows += 1
        try:
            encoded = text.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            self.invalid_utf8_rows += 1
            return None

        self.valid_utf8_rows += 1
        self.valid_source_bytes += len(encoded)
        if not text.strip():
            self.blank_rows += 1
            return None

        self.clean_rows += 1
        self.clean_story_bytes += len(encoded)
        self.next_byte_input_bytes += max(len(encoded) - 1, 0)
        self.byte_length_histogram[length_bucket(len(encoded))] += 1
        self.format_counts["multiline"] += int("\n" in text or "\r" in text)
        self.format_counts["single_line"] += int("\n" not in text and "\r" not in text)
        self.format_counts["contains_dialogue_quotes"] += int(
            any(quote in text for quote in ('"', "\u201c", "\u201d"))
        )
        return encoded

    def report(self) -> dict[str, object]:
        data = asdict(self)
        data["byte_length_histogram"] = dict(sorted(self.byte_length_histogram.items()))
        data["format_counts"] = dict(sorted(self.format_counts.items()))
        data["mean_clean_bytes_per_story"] = (
            self.clean_story_bytes / self.clean_rows if self.clean_rows else 0.0
        )
        return data


def iter_parquet_texts(paths: Iterable[Path], batch_size: int = 65536) -> Iterable[str]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install project dependencies with `uv sync` first.") from exc

    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required TinyStories parquet file not found: {path}")
        parquet = pq.ParquetFile(path)
        if "text" not in parquet.schema.names:
            raise ValueError(f"Expected a `text` column in {path}")
        for batch in parquet.iter_batches(columns=["text"], batch_size=batch_size):
            for value in batch.column(0).to_pylist():
                if not isinstance(value, str):
                    raise TypeError(f"Expected text strings in {path}, got {type(value).__name__}")
                yield value


def collect_statistics(data_dir: Path, batch_size: int = 65536) -> dict[str, object]:
    train_stats = SplitStats()
    train_hashes: set[bytes] = set()
    train_paths = [data_dir / name for name in TRAIN_FILES]

    for text in iter_parquet_texts(train_paths, batch_size):
        encoded = train_stats.add(text)
        if encoded is not None:
            train_hashes.add(hashlib.sha256(encoded).digest())

    official_validation = SplitStats()
    validation_stats = SplitStats()
    test_stats = SplitStats()
    validation_hashes: set[bytes] = set()
    dropped_train_duplicates = 0
    dropped_validation_duplicates = 0

    for text in iter_parquet_texts([data_dir / VALIDATION_FILE], batch_size):
        encoded = official_validation.add(text)
        if encoded is None:
            continue
        digest = hashlib.sha256(encoded).digest()
        if digest in train_hashes:
            dropped_train_duplicates += 1
            continue
        if digest in validation_hashes:
            dropped_validation_duplicates += 1
            continue
        validation_hashes.add(digest)
        target = validation_stats if validation_fold(encoded) == "validation" else test_stats
        target.add(text)

    return {
        "dataset": "roneneldan/TinyStories",
        "revision": SOURCE_REVISION,
        "data_dir": str(data_dir.resolve()),
        "encoding": "UTF-8, strict",
        "cleaning": "Preserve text verbatim; exclude only whitespace-only and invalid UTF-8 rows.",
        "special_tokens": "None; no BOS/EOS token is added to the 256-byte vocabulary.",
        "byte_budget_definition": (
            "next_byte_input_bytes counts each clean story's UTF-8 bytes except its final "
            "byte, because targets are shifted by one and stories never cross boundaries."
        ),
        "validation_test_split": (
            "Hash each clean official-validation story with SHA-256; even first digest "
            "byte goes to validation, odd goes to test. Exact train overlaps and repeated "
            "validation stories are excluded from held-out splits."
        ),
        "official_train": train_stats.report(),
        "official_validation_before_dedup": official_validation.report(),
        "train": train_stats.report(),
        "validation": validation_stats.report(),
        "test": test_stats.report(),
        "heldout_deduplication": {
            "dropped_exact_train_duplicates": dropped_train_duplicates,
            "dropped_repeated_validation_stories": dropped_validation_duplicates,
        },
        "target_trainable_parameters": train_stats.next_byte_input_bytes // 20,
        "target_parameter_formula": "train.next_byte_input_bytes / 20",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/stats.json"))
    parser.add_argument("--batch-size", type=int, default=65536)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    report = collect_statistics(args.data_dir, args.batch_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
