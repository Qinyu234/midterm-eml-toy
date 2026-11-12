"""CPU benchmark: precision/thread sweep plus fixed-shape training throughput.

Measurement only. `REFERENCE` is a benchmark fixture so two machines can be
compared, not a parameter-budget candidate; the approved architecture still has
to come from `data/stats.json` and the recorded candidates.

Both Windows and Linux are supported, and the run only needs `torch`, so the
same command can be repeated on a second machine.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from dataclasses import asdict
from pathlib import Path

import torch

from tinystories_byte_lm.config import ModelConfig
from tinystories_byte_lm.model import ByteTransformer
from tinystories_byte_lm.precision_probe import run_cpu_precision_probe
from tinystories_byte_lm.training import peak_process_memory_bytes, train_step

REFERENCE = {
    "d_model": 384,
    "n_layers": 8,
    "n_heads": 6,
    "d_ff": 1536,
    "context_length": 256,
    "lut_seed": 1729,
}


def total_memory_bytes() -> int | None:
    """Total physical memory without requiring a third-party dependency."""
    try:
        if os.name == "nt":
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.dwLength = ctypes.sizeof(MemoryStatus)
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
            kernel32.GlobalMemoryStatusEx.restype = ctypes.c_int
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.ullTotalPhys)
            return None
        with open("/proc/meminfo", encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, AttributeError, TypeError):
        return None
    return None


def cpu_name() -> str:
    """Model string; Windows `platform.processor()` only reports the binary arch."""
    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            ) as key:
                return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def measure_training_throughput(
    threads: int,
    batch_size: int,
    sequence_length: int,
    warmup_steps: int,
    measured_steps: int,
) -> dict[str, object]:
    torch.set_num_threads(threads)
    config = ModelConfig(**{**REFERENCE, "context_length": sequence_length})
    torch.manual_seed(2718)
    model = ByteTransformer(config)
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-3,
    )

    generator = torch.Generator().manual_seed(2718)
    batches = [
        torch.randint(0, 256, (batch_size, sequence_length), generator=generator, dtype=torch.long)
        for _ in range(warmup_steps + measured_steps)
    ]
    prepared = [(batch, torch.roll(batch, shifts=-1, dims=1)) for batch in batches]

    for inputs, targets in prepared[:warmup_steps]:
        train_step(model, optimizer, inputs, targets)
    start = time.perf_counter()
    last_loss = None
    for inputs, targets in prepared[warmup_steps:]:
        metrics = train_step(model, optimizer, inputs, targets)
        last_loss = metrics.loss
    elapsed = time.perf_counter() - start

    return {
        "threads": threads,
        "table": dict(REFERENCE, context_length=sequence_length),
        "trainable_parameters": trainable_parameters,
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "measured_steps": measured_steps,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / measured_steps,
        "input_bytes_per_second": batch_size * sequence_length * measured_steps / elapsed,
        "peak_process_memory_bytes": peak_process_memory_bytes(),
        "last_loss": last_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", default="1,2,4,8", help="Comma-separated thread counts.")
    parser.add_argument("--probe-steps", type=int, default=10)
    parser.add_argument("--probe-warmup", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--skip-throughput", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    thread_counts = [int(item) for item in args.threads.split(",") if item.strip()]
    if not thread_counts or min(thread_counts) <= 0:
        raise SystemExit("--threads must list positive integers")

    report: dict[str, object] = {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cpu": cpu_name(),
        "logical_cpus": os.cpu_count(),
        "total_memory_bytes": total_memory_bytes(),
        "torch_version": torch.__version__,
        "precision_sweep": [
            run_cpu_precision_probe(
                warmup_steps=args.probe_warmup,
                measured_steps=args.probe_steps,
                num_threads=threads,
            )
            for threads in thread_counts
        ],
        "reference_shape": REFERENCE,
    }
    if not args.skip_throughput:
        report["throughput"] = [
            measure_training_throughput(
                threads,
                args.batch_size,
                args.sequence_length,
                args.warmup_steps,
                args.steps,
            )
            for threads in thread_counts
        ]

    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    print(text)


if __name__ == "__main__":
    main()
