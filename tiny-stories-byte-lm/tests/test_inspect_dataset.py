import tempfile
import unittest
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.inspect_dataset import (
    TRAIN_FILES,
    VALIDATION_FILE,
    SplitStats,
    collect_statistics,
    length_bucket,
    validation_fold,
)
from scripts.prepare_dataset import prepare_dataset
from tinystories_byte_lm.data import MappedByteStories


class DatasetStatisticsTests(unittest.TestCase):
    def test_counts_utf8_bytes_not_unicode_characters(self) -> None:
        stats = SplitStats()

        encoded = stats.add("café")

        self.assertEqual(encoded, b"caf\xc3\xa9")
        self.assertEqual(stats.clean_story_bytes, 5)
        self.assertEqual(stats.next_byte_input_bytes, 4)
        self.assertEqual(stats.clean_rows, 1)

    def test_blank_rows_are_excluded_without_losing_source_accounting(self) -> None:
        stats = SplitStats()

        stats.add(" \n ")

        self.assertEqual(stats.rows, 1)
        self.assertEqual(stats.valid_source_bytes, 3)
        self.assertEqual(stats.clean_rows, 0)
        self.assertEqual(stats.clean_story_bytes, 0)

    def test_length_bins_have_inclusive_boundaries(self) -> None:
        self.assertEqual(length_bucket(0), "0-127")
        self.assertEqual(length_bucket(127), "0-127")
        self.assertEqual(length_bucket(128), "128-255")
        self.assertEqual(length_bucket(2048), "2048+")

    def test_validation_assignment_is_stable_for_identical_stories(self) -> None:
        self.assertEqual(validation_fold(b"story"), validation_fold(b"story"))

    def test_heldout_duplicates_are_removed_before_deterministic_split(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            for index, name in enumerate(TRAIN_FILES):
                texts = ["training story"] if index == 0 else []
                pq.write_table(
                    pa.table({"text": pa.array(texts, type=pa.string())}),
                    data_dir / name,
                )
            pq.write_table(
                pa.table(
                    {
                        "text": [
                            "training story",
                            "heldout story",
                            "heldout story",
                            "another heldout story",
                        ]
                    }
                ),
                data_dir / VALIDATION_FILE,
            )

            report = collect_statistics(data_dir, batch_size=2)

        self.assertEqual(report["heldout_deduplication"], {
            "dropped_exact_train_duplicates": 1,
            "dropped_repeated_validation_stories": 1,
        })
        self.assertEqual(report["validation"]["clean_rows"] + report["test"]["clean_rows"], 2)

    def test_prepared_files_preserve_splits_and_exclude_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "raw"
            output_dir = root / "processed"
            data_dir.mkdir()
            for index, name in enumerate(TRAIN_FILES):
                texts = ["training story"] if index == 0 else []
                pq.write_table(
                    pa.table({"text": pa.array(texts, type=pa.string())}),
                    data_dir / name,
                )
            validation_story = next(
                f"validation {index}"
                for index in range(100)
                if validation_fold(f"validation {index}".encode("utf-8"))
                == "validation"
            )
            test_story = next(
                f"test {index}"
                for index in range(100)
                if validation_fold(f"test {index}".encode("utf-8")) == "test"
            )
            pq.write_table(
                pa.table(
                    {
                        "text": [
                            "training story",
                            validation_story,
                            validation_story,
                            test_story,
                        ]
                    }
                ),
                data_dir / VALIDATION_FILE,
            )
            stats = collect_statistics(data_dir, batch_size=2)
            stats_path = root / "stats.json"
            stats_path.write_text(json.dumps(stats), encoding="utf-8")

            manifest = prepare_dataset(data_dir, stats_path, output_dir, batch_size=2)

            self.assertEqual(manifest["splits"]["train"]["stories"], 1)
            self.assertEqual(manifest["splits"]["validation"]["stories"], 1)
            self.assertEqual(manifest["splits"]["test"]["stories"], 1)
            with MappedByteStories(
                output_dir / "validation.bin", output_dir / "validation.idx"
            ) as validation:
                inputs, targets = validation.window_at(0, 100)

        self.assertEqual(bytes(inputs.tolist()), validation_story.encode("utf-8")[:-1])
        self.assertEqual(bytes(targets.tolist()), validation_story.encode("utf-8")[1:])


if __name__ == "__main__":
    unittest.main()
