"""Download the pinned TinyStories Parquet files with resumable curl transfers."""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

from inspect_dataset import SOURCE_REVISION

DATASET = "roneneldan/TinyStories"
BASE_URL = f"https://huggingface.co/datasets/{DATASET}/resolve/{SOURCE_REVISION}/data"


@dataclass(frozen=True)
class DatasetFile:
    source_name: str
    local_name: str
    size_bytes: int


FILES = (
    DatasetFile(
        "train-00000-of-00004-2d5a1467fff1081b.parquet",
        "train-00000-of-00004.parquet",
        248_731_111,
    ),
    DatasetFile(
        "train-00001-of-00004-5852b56a2bd28fd9.parquet",
        "train-00001-of-00004.parquet",
        248_171_980,
    ),
    DatasetFile(
        "train-00002-of-00004-a26307300439e943.parquet",
        "train-00002-of-00004.parquet",
        245_894_874,
    ),
    DatasetFile(
        "train-00003-of-00004-d243063613e5a057.parquet",
        "train-00003-of-00004.parquet",
        247_988_350,
    ),
    DatasetFile(
        "validation-00000-of-00001-869c898b519ad725.parquet",
        "validation-00000-of-00001.parquet",
        9_989_127,
    ),
)


def download_file(dataset_file: DatasetFile, output_dir: Path, proxy: str | None) -> None:
    destination = output_dir / dataset_file.local_name
    if destination.exists() and destination.stat().st_size == dataset_file.size_bytes:
        print(f"Verified; skipping {destination.name}")
        return

    url = f"{BASE_URL}/{dataset_file.source_name}"
    command = [
        "curl.exe",
        "--fail",
        "--location",
        "--retry",
        "10",
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "60",
        "--continue-at",
        "-",
        "--output",
        str(destination),
    ]
    configured_proxy = proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if configured_proxy:
        command.extend(["--proxy", configured_proxy])
    command.append(url)

    print(f"Downloading {dataset_file.source_name} to {destination}")
    subprocess.run(command, check=True)
    actual_size = destination.stat().st_size
    if actual_size != dataset_file.size_bytes:
        raise RuntimeError(
            f"Incomplete download for {destination.name}: "
            f"expected {dataset_file.size_bytes} bytes, got {actual_size}; "
            "rerun to resume."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--proxy",
        help="HTTP proxy URL; otherwise HTTPS_PROXY/HTTP_PROXY is used if set.",
    )
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    for dataset_file in FILES:
        download_file(dataset_file, args.data_dir, args.proxy)


if __name__ == "__main__":
    main()
