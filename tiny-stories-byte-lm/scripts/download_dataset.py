"""Download the pinned TinyStories Parquet files with resumable curl transfers."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from inspect_dataset import SOURCE_REVISION

DATASET = "roneneldan/TinyStories"
DEFAULT_ENDPOINT = "https://huggingface.co"


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


def curl_executable() -> str:
    """Locate curl even when the shell PATH omits the Windows system directory."""
    candidates = [
        shutil.which("curl.exe"),
        shutil.which("curl"),
        str(Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "curl.exe"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise RuntimeError("curl executable not found; install curl or add it to PATH")


def resolve_endpoint(argument: str | None) -> str:
    """Resolve the dataset host; HF_ENDPOINT allows a reachable mirror."""
    configured = argument or os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT
    return configured.rstrip("/")


def file_url(source_name: str, endpoint: str) -> str:
    return f"{endpoint}/datasets/{DATASET}/resolve/{SOURCE_REVISION}/data/{source_name}"


def transfer(url: str, output: Path, proxy: str | None, extra: list[str] | None = None) -> None:
    command = [
        curl_executable(),
        "--silent",
        "--show-error",
        "--fail",
        "--location",
        "--retry",
        "10",
        "--retry-all-errors",
        "--retry-delay",
        "5",
        "--connect-timeout",
        "60",
        "--speed-limit",
        "2000",
        "--speed-time",
        "60",
    ]
    configured_proxy = proxy or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if configured_proxy:
        command.extend(["--proxy", configured_proxy])
    if extra:
        command.extend(extra)
    command.extend(["--output", str(output), url])
    subprocess.run(command, check=True)


def download_single(dataset_file: DatasetFile, output_dir: Path, proxy: str | None, url: str) -> None:
    destination = output_dir / dataset_file.local_name
    transfer(url, destination, proxy, ["--continue-at", "-"])


def download_chunked(
    dataset_file: DatasetFile,
    output_dir: Path,
    proxy: str | None,
    url: str,
    connections: int,
) -> None:
    """Fetch one file as fixed byte ranges so interrupted transfers resume per chunk."""
    part_dir = output_dir / f".{dataset_file.local_name}.parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    chunk = dataset_file.size_bytes // connections
    parts: list[tuple[Path, int]] = []
    pending: list[tuple[Path, int, int, int]] = []
    for index in range(connections):
        start = index * chunk
        end = dataset_file.size_bytes - 1 if index == connections - 1 else start + chunk - 1
        expected = end - start + 1
        part_path = part_dir / f"{index:03d}.part"
        if part_path.exists() and part_path.stat().st_size == expected:
            print(f"Verified part {part_path.name}; skipping")
        else:
            if part_path.exists():
                part_path.unlink()
            pending.append((part_path, start, end, expected))
        parts.append((part_path, expected))

    if pending:
        print(
            f"Downloading {dataset_file.source_name} in {len(pending)} ranges "
            f"({connections} connections)"
        )

        def fetch(item: tuple[Path, int, int, int]) -> None:
            part_path, start, end, expected = item
            transfer(url, part_path, proxy, ["--range", f"{start}-{end}"])
            actual = part_path.stat().st_size
            if actual != expected:
                raise RuntimeError(
                    f"Chunk {part_path.name} of {dataset_file.local_name} is incomplete: "
                    f"expected {expected} bytes, got {actual}; rerun to resume."
                )

        with ThreadPoolExecutor(max_workers=connections) as pool:
            futures = [pool.submit(fetch, item) for item in pending]
            for future in as_completed(futures):
                future.result()
    temporary = output_dir / f"{dataset_file.local_name}.assembling"
    destination = output_dir / dataset_file.local_name
    with temporary.open("wb") as assembled:
        for part_path, _ in parts:
            with part_path.open("rb") as part:
                shutil.copyfileobj(part, assembled, length=1024 * 1024)
    temporary.replace(destination)
    shutil.rmtree(part_dir)


def download_file(
    dataset_file: DatasetFile,
    output_dir: Path,
    proxy: str | None,
    endpoint: str,
    connections: int,
) -> None:
    destination = output_dir / dataset_file.local_name
    if destination.exists() and destination.stat().st_size == dataset_file.size_bytes:
        print(f"Verified; skipping {destination.name}")
        return

    url = file_url(dataset_file.source_name, endpoint)
    if connections > 1:
        download_chunked(dataset_file, output_dir, proxy, url, connections)
    else:
        print(f"Downloading {dataset_file.source_name} to {destination}")
        download_single(dataset_file, output_dir, proxy, url)

    actual_size = destination.stat().st_size
    if actual_size != dataset_file.size_bytes:
        raise RuntimeError(
            f"Incomplete download for {destination.name}: "
            f"expected {dataset_file.size_bytes} bytes, got {actual_size}; "
            "rerun to resume."
        )
    print(f"Verified {destination.name} ({actual_size} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--proxy",
        help="HTTP proxy URL; otherwise HTTPS_PROXY/HTTP_PROXY is used if set.",
    )
    parser.add_argument(
        "--endpoint",
        help="Dataset host; otherwise $HF_ENDPOINT, then https://huggingface.co.",
    )
    parser.add_argument(
        "--connections",
        type=int,
        default=1,
        help="Parallel byte-range connections per file; 1 uses a single resumable stream.",
    )
    args = parser.parse_args()
    if args.connections < 1:
        raise SystemExit("--connections must be at least 1")
    endpoint = resolve_endpoint(args.endpoint)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using dataset endpoint {endpoint}")
    for dataset_file in FILES:
        download_file(dataset_file, args.data_dir, args.proxy, endpoint, args.connections)


if __name__ == "__main__":
    main()
