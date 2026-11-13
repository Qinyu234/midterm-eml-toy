"""Subset training: cap the usable byte extent of a full corpus instead of copying it."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tinystories_byte_lm.data import MappedByteStories, write_byte_dataset


class ByteSubsetTest(unittest.TestCase):
    def test_stories_crossing_the_cut_are_dropped_not_truncated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "stories.bin"
            index_path = root / "stories.idx"
            # offsets: 0 (len 3), 3 (len 4), 7 (len 2) -> 9 bytes total
            stats = write_byte_dataset([b"aaa", b"bbbb", b"cc"], data_path, index_path)
            self.assertEqual(stats["story_bytes"], 9)

            with MappedByteStories(data_path, index_path, max_bytes=7) as subset:
                self.assertEqual(subset.total_input_bytes, 2 + 3)
                self.assertEqual(subset.skipped_stories, 1)
                self.assertEqual(subset.usable_bytes, 7)
                inputs, targets = subset.window_at(2, 4)
                self.assertEqual(bytes(inputs.tolist()), b"bbb")
                self.assertEqual(bytes(targets.tolist()), b"bbb")

    def test_cap_that_lands_inside_a_story_keeps_only_whole_stories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "stories.bin"
            index_path = root / "stories.idx"
            write_byte_dataset([b"aaa", b"bbbb", b"cc"], data_path, index_path)

            with MappedByteStories(data_path, index_path, max_bytes=6) as subset:
                self.assertEqual(subset.total_input_bytes, 2)  # only "aaa" survives
                self.assertEqual(subset.skipped_stories, 2)

    def test_without_a_cap_every_story_stays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_path = root / "stories.bin"
            index_path = root / "stories.idx"
            write_byte_dataset([b"aaa", b"bbbb", b"cc"], data_path, index_path)

            with MappedByteStories(data_path, index_path) as full:
                self.assertEqual(full.total_input_bytes, 2 + 3 + 1)
                self.assertEqual(full.skipped_stories, 0)


if __name__ == "__main__":
    unittest.main()
