"""Story-preserving byte storage and next-byte window sampling."""

from __future__ import annotations

import bisect
import mmap
import os
import random
import struct
import tempfile
from array import array
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import Tensor

IGNORE_TARGET = -100
INDEX_RECORD = struct.Struct("<QQ")


def write_byte_dataset(
    stories: Iterable[bytes], data_path: Path, index_path: Path
) -> dict[str, int]:
    """Write concatenated bytes plus per-story offsets; no boundary byte is invented."""
    data_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    data_handle, data_temp_name = tempfile.mkstemp(
        prefix=f".{data_path.name}.", suffix=".tmp", dir=data_path.parent
    )
    index_handle, index_temp_name = tempfile.mkstemp(
        prefix=f".{index_path.name}.", suffix=".tmp", dir=index_path.parent
    )
    rows = 0
    byte_count = 0
    input_byte_count = 0
    try:
        with os.fdopen(data_handle, "wb") as data_file, os.fdopen(
            index_handle, "wb"
        ) as index_file:
            for story in stories:
                if not isinstance(story, bytes):
                    raise TypeError(f"stories must contain bytes, got {type(story).__name__}")
                index_file.write(INDEX_RECORD.pack(byte_count, len(story)))
                data_file.write(story)
                rows += 1
                byte_count += len(story)
                input_byte_count += max(len(story) - 1, 0)
            data_file.flush()
            index_file.flush()
            os.fsync(data_file.fileno())
            os.fsync(index_file.fileno())
        os.replace(data_temp_name, data_path)
        os.replace(index_temp_name, index_path)
    finally:
        for temp_name in (data_temp_name, index_temp_name):
            if os.path.exists(temp_name):
                os.unlink(temp_name)
    return {
        "stories": rows,
        "story_bytes": byte_count,
        "next_byte_input_bytes": input_byte_count,
    }


class MappedByteStories:
    """Memory-mapped stories sampled uniformly by eligible next-byte input."""

    def __init__(self, data_path: Path, index_path: Path) -> None:
        self._data_file = data_path.open("rb")
        self._index_file = index_path.open("rb")
        index_size = os.fstat(self._index_file.fileno()).st_size
        if index_size % INDEX_RECORD.size:
            self.close()
            raise ValueError(f"Invalid story index size: {index_size}")
        self.offsets: array[int] = array("Q")
        self.lengths: array[int] = array("Q")
        self.cumulative_inputs: array[int] = array("Q")
        total_inputs = 0
        data_size = os.fstat(self._data_file.fileno()).st_size
        for offset, length in INDEX_RECORD.iter_unpack(self._index_file.read()):
            if offset + length > data_size:
                self.close()
                raise ValueError("Story index points outside the byte data file")
            self.offsets.append(offset)
            self.lengths.append(length)
            total_inputs += max(length - 1, 0)
            self.cumulative_inputs.append(total_inputs)
        if data_size == 0:
            self.close()
            raise ValueError("Byte data file is empty")
        self._data = mmap.mmap(self._data_file.fileno(), 0, access=mmap.ACCESS_READ)
        self.total_input_bytes = total_inputs
        if not self.total_input_bytes:
            self.close()
            raise ValueError("Dataset contains no next-byte training positions")

    def close(self) -> None:
        mapped = getattr(self, "_data", None)
        if mapped is not None:
            mapped.close()
        data_file = getattr(self, "_data_file", None)
        if data_file is not None and not data_file.closed:
            data_file.close()
        index_file = getattr(self, "_index_file", None)
        if index_file is not None and not index_file.closed:
            index_file.close()

    def __enter__(self) -> MappedByteStories:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def window_at(self, global_input_position: int, context_length: int) -> tuple[Tensor, Tensor]:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if not 0 <= global_input_position < self.total_input_bytes:
            raise IndexError("global input position is outside the dataset")
        story_index = bisect.bisect_right(self.cumulative_inputs, global_input_position)
        previous_end = (
            self.cumulative_inputs[story_index - 1] if story_index else 0
        )
        story_start = global_input_position - previous_end
        input_length = self.lengths[story_index] - 1
        window_length = min(context_length, input_length - story_start)
        byte_start = self.offsets[story_index] + story_start
        values = self._data[byte_start : byte_start + window_length + 1]
        inputs = torch.tensor(list(values[:-1]), dtype=torch.long)
        targets = torch.tensor(list(values[1:]), dtype=torch.long)
        return inputs, targets

    def sample_batch(
        self,
        batch_size: int,
        context_length: int,
        rng: random.Random,
    ) -> tuple[Tensor, Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        windows = [
            self.window_at(rng.randrange(self.total_input_bytes), context_length)
            for _ in range(batch_size)
        ]
        max_length = max(inputs.numel() for inputs, _ in windows)
        inputs = torch.zeros((batch_size, max_length), dtype=torch.long)
        targets = torch.full(
            (batch_size, max_length), IGNORE_TARGET, dtype=torch.long
        )
        for row, (window_inputs, window_targets) in enumerate(windows):
            length = window_inputs.numel()
            inputs[row, :length] = window_inputs
            targets[row, :length] = window_targets
        return inputs, targets
