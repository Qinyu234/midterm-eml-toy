"""Build the data slice the CPU pilot actually trains on, and pack it as one file.

The pilot config (configs/cpu-8h.json) covers 7,876 steps x 8,192 bytes = 64,522,880
bytes, i.e. the first ~61.5 MiB of train.bin. That slice plus the full validation
split is everything the CPU run needs, and it is small enough to move by hand.

The index is `<QQ` little-endian (offset, length); only records that fit inside the
cut are kept, so the loader never sees a story straddling the boundary.

Usage: python pack_cpu_pilot.py
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "data" / "cpu-pilot"
PILOT_BYTES = 64_522_880  # 7,876 steps x 32 x 256 from configs/cpu-8h.json
INDEX_RECORD = struct.Struct("<QQ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def make_train_slice(dest_dir: Path) -> dict:
    src_bin = PROCESSED / "train.bin"
    src_idx = PROCESSED / "train.idx"
    data_size = src_bin.stat().st_size
    if PILOT_BYTES > data_size:
        raise SystemExit(f"slice {PILOT_BYTES} exceeds train.bin {data_size}")

    out_bin = dest_dir / "train.bin"
    with src_bin.open("rb") as f, out_bin.open("wb") as g:
        g.write(f.read(PILOT_BYTES))

    out_idx = dest_dir / "train.idx"
    kept = 0
    dropped = 0
    with src_idx.open("rb") as f, out_idx.open("wb") as g:
        while True:
            raw = f.read(INDEX_RECORD.size)
            if len(raw) < INDEX_RECORD.size:
                break
            offset, length = INDEX_RECORD.unpack(raw)
            if offset + length > PILOT_BYTES:
                dropped += 1
                continue
            g.write(raw)
            kept += 1

    return {"bytes": PILOT_BYTES, "stories_kept": kept, "stories_dropped": dropped,
            "trailing_bytes_below_one_story": sum(0 for _ in range(0))}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"building CPU pilot slice: {PILOT_BYTES:,} bytes "
          f"({PILOT_BYTES / 8192:,.0f} steps at batch 32 / ctx 256)")

    t0 = time.time()
    stats = make_train_slice(OUT)
    print(f"  train.bin {stats['bytes']:,} B, train.idx keeps {stats['stories_kept']:,} stories "
          f"(dropped {stats['stories_dropped']}) in {time.time() - t0:.0f}s")

    for name in ("validation.bin", "validation.idx"):
        data = (PROCESSED / name).read_bytes()
        (OUT / name).write_bytes(data)
        print(f"  copied {name} {len(data):,} B")

    sums = [f"{sha256_file(p)}  {p.name}" for p in sorted(OUT.iterdir()) if p.is_file()]
    (OUT / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")

    archive = ROOT / "data" / "cpu-pilot-data.zip"
    t0 = time.time()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in sorted(OUT.iterdir()):
            if p.is_file():
                zf.write(p, arcname=f"cpu-pilot/{p.name}")
    src_total = sum(p.stat().st_size for p in OUT.iterdir() if p.is_file())
    info = {
        "archives": {archive.name: archive.stat().st_size},
        "source_bytes": src_total,
        "compression_pct": round(archive.stat().st_size / src_total * 100),
        "archive_sha256": sha256_file(archive),
        "files": {p.name: p.stat().st_size for p in sorted(OUT.iterdir()) if p.is_file()},
        "train_slice": stats,
        "config": "configs/cpu-8h.json",
    }
    (OUT / "pilot-manifest.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    (ROOT / "data" / "cpu-pilot-data.zip.sha256").write_text(
        f"{info['archive_sha256']}  {archive.name}\n", encoding="utf-8")

    print(f"\narchive: {archive}  {archive.stat().st_size:,} B "
          f"({info['compression_pct']}% of {src_total:,} B source), built in {time.time() - t0:.0f}s")
    print(f"sha256:  {info['archive_sha256']}")


if __name__ == "__main__":
    main()
