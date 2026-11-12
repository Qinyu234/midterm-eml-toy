"""Small reusable next-byte train/evaluation steps and resumable checkpoints."""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .config import ModelConfig
from .data import IGNORE_TARGET, MappedByteStories
from .model import ByteTransformer
from .model import VOCAB_SIZE


@dataclass(frozen=True)
class StepMetrics:
    loss: float
    gradient_norm: float
    input_bytes: int


@dataclass(frozen=True)
class TrainConfig:
    steps: int
    batch_size: int
    learning_rate: float
    eval_interval: int
    eval_batches: int
    checkpoint_interval: int
    seed: int
    precision: str = "fp32"
    num_threads: int | None = None

    def __post_init__(self) -> None:
        if min(
            self.steps,
            self.batch_size,
            self.eval_interval,
            self.eval_batches,
            self.checkpoint_interval,
        ) <= 0:
            raise ValueError("step, batch, evaluation, and checkpoint intervals must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.precision not in ("fp32", "fp16", "bf16"):
            raise ValueError("precision must be one of: fp32, fp16, bf16")
        if self.num_threads is not None and self.num_threads <= 0:
            raise ValueError("num_threads must be positive")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _validate_batch(inputs: Tensor, targets: Tensor, logits: Tensor) -> int:
    if targets.shape != logits.shape[:-1]:
        raise ValueError("targets must match the model's [batch, sequence] output shape")
    valid_targets = int((targets != IGNORE_TARGET).sum().item())
    if valid_targets == 0:
        raise ValueError("batch has no next-byte targets")
    return valid_targets


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
) -> StepMetrics:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    valid_targets = _validate_batch(inputs, targets, logits)
    loss = F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE),
        targets.reshape(-1),
        ignore_index=IGNORE_TARGET,
    )
    loss.backward()
    squared_norm = sum(
        float(parameter.grad.detach().float().square().sum().item())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    gradient_norm = math.sqrt(squared_norm)
    optimizer.step()
    return StepMetrics(
        loss=float(loss.detach().float().item()),
        gradient_norm=gradient_norm,
        input_bytes=valid_targets,
    )


@torch.no_grad()
def evaluate_loss(model: nn.Module, batches: Iterable[tuple[Tensor, Tensor]]) -> dict[str, float]:
    model.eval()
    weighted_loss = 0.0
    total_targets = 0
    for inputs, targets in batches:
        logits = model(inputs)
        valid_targets = _validate_batch(inputs, targets, logits)
        loss = F.cross_entropy(
            logits.reshape(-1, VOCAB_SIZE),
            targets.reshape(-1),
            ignore_index=IGNORE_TARGET,
            reduction="sum",
        )
        weighted_loss += float(loss.float().item())
        total_targets += valid_targets
    if total_targets == 0:
        raise ValueError("evaluation received no next-byte targets")
    mean_loss = weighted_loss / total_targets
    return {"loss": mean_loss, "byte_perplexity": math.exp(mean_loss)}


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    config: dict[str, object],
    python_rng_state: object | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "config": config,
            "torch_rng_state": torch.get_rng_state(),
            "python_rng_state": python_rng_state or random.getstate(),
        },
        temporary_path,
    )
    os.replace(temporary_path, path)


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, dict[str, object]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required_keys = {
        "model_state",
        "optimizer_state",
        "step",
        "config",
        "torch_rng_state",
        "python_rng_state",
    }
    if not required_keys.issubset(checkpoint):
        raise ValueError(f"Checkpoint is missing keys: {sorted(required_keys - set(checkpoint))}")
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    random.setstate(checkpoint["python_rng_state"])
    return int(checkpoint["step"]), dict(checkpoint["config"])


def append_metrics(path: Path, step: int, values: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"step": step, **values}
    with path.open("a", encoding="utf-8") as metrics_file:
        metrics_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def peak_process_memory_bytes() -> int:
    if os.name == "nt":
        import ctypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("Kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("Psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        get_memory_info = psapi.GetProcessMemoryInfo
        get_memory_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ProcessMemoryCounters),
            ctypes.c_ulong,
        ]
        get_memory_info.restype = ctypes.c_int
        if not get_memory_info(
            kernel32.GetCurrentProcess(),
            ctypes.byref(counters),
            counters.cb,
        ):
            error = ctypes.get_last_error()
            raise OSError(error, "GetProcessMemoryInfo failed")
        return int(counters.PeakWorkingSetSize)

    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def run_training(
    model_config: ModelConfig,
    train_config: TrainConfig,
    train_data_path: Path,
    train_index_path: Path,
    validation_data_path: Path,
    validation_index_path: Path,
    output_dir: Path,
    resume_from: Path | None = None,
) -> list[dict[str, object]]:
    if model_config.context_length <= 0:
        raise ValueError("model context_length must be positive")
    if train_config.num_threads is not None:
        torch.set_num_threads(train_config.num_threads)
    seed_everything(train_config.seed)
    dtype_by_name = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    model = ByteTransformer(model_config).to(
        device="cpu", dtype=dtype_by_name[train_config.precision]
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=train_config.learning_rate,
    )
    run_config = {
        "model": asdict(model_config),
        "training": asdict(train_config),
        "device": "cpu",
        "optimizer": "AdamW",
        "loss": "next-byte cross entropy",
    }
    start_step = 0
    if resume_from is not None:
        start_step, saved_config = load_checkpoint(resume_from, model, optimizer)
        if saved_config != run_config:
            raise ValueError("Checkpoint config does not match the requested run config")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    records: list[dict[str, object]] = []
    with MappedByteStories(train_data_path, train_index_path) as train_data, MappedByteStories(
        validation_data_path, validation_index_path
    ) as validation_data:
        for step in range(start_step + 1, train_config.steps + 1):
            step_start = time.perf_counter()
            inputs, targets = train_data.sample_batch(
                train_config.batch_size,
                model_config.context_length,
                random,
            )
            metrics = train_step(model, optimizer, inputs, targets)
            elapsed = time.perf_counter() - step_start
            record: dict[str, object] = {
                "train_loss": metrics.loss,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "gradient_norm": metrics.gradient_norm,
                "input_bytes": metrics.input_bytes,
                "input_bytes_per_second": metrics.input_bytes / elapsed,
                "elapsed_seconds": elapsed,
                "peak_process_memory_bytes": peak_process_memory_bytes(),
            }
            if step % train_config.eval_interval == 0 or step == train_config.steps:
                validation_rng = random.Random(train_config.seed + step)
                validation_batches = [
                    validation_data.sample_batch(
                        train_config.batch_size,
                        model_config.context_length,
                        validation_rng,
                    )
                    for _ in range(train_config.eval_batches)
                ]
                record.update(
                    {
                        f"validation_{key}": value
                        for key, value in evaluate_loss(model, validation_batches).items()
                    }
                )
            append_metrics(metrics_path, step, record)
            records.append({"step": step, **record})

            if step % train_config.checkpoint_interval == 0 or step == train_config.steps:
                checkpoint_path = output_dir / f"checkpoint-{step:08d}.pt"
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    step,
                    run_config,
                    python_rng_state=random.getstate(),
                )
    return records
