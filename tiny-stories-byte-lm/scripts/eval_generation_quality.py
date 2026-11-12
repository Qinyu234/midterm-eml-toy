"""Measure how far a checkpoint has moved from byte soup toward English.

Two independent signals, both computed on samples generated with fixed seeds so
checkpoints are comparable:

- illegal characters: bytes that are not printable ASCII, not whitespace, and not
  part of a valid UTF-8 multibyte sequence. Reported both as a strict-decode
  failure rate per sample and as the share of emitted bytes.
- invented spelling: words in the generated text that never occur in the training
  slice. A model that has learned the corpus vocabulary invents few new words; one
  that is still guessing invents many.

    python scripts/eval_generation_quality.py checkpoints/cpu-8h \
        --train-data data/processed/train.bin --limit-bytes 64522880
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

from tinystories_byte_lm.generation import generate_bytes
from tinystories_byte_lm.config import ModelConfig
from tinystories_byte_lm.model import ByteTransformer

WORD = re.compile(rb"[A-Za-z]+(?:'[A-Za-z]+)?")
PROMPTS = (b"Once upon a time", b"The little dog", b"Lily wanted to")
ALLOWED_CONTROL = {0x09, 0x0A, 0x0D}


def corpus_words(train_data: Path, limit_bytes: int | None) -> set[str]:
    with train_data.open("rb") as handle:
        data = handle.read(limit_bytes) if limit_bytes else handle.read()
    return {match.group(0).lower().decode("ascii") for match in WORD.finditer(data)}


def illegal_byte_share(data: bytes) -> tuple[float, int]:
    """Share of bytes outside printable ASCII / whitespace / valid UTF-8.

    A byte is legal if it is printable ASCII, allowed whitespace, or part of a
    well-formed UTF-8 multibyte character. Anything else (lone continuation bytes,
    truncated sequences, control bytes) is counted illegal.
    """
    illegal = 0
    index = 0
    while index < len(data):
        byte = data[index]
        if 0x20 <= byte <= 0x7E or byte in ALLOWED_CONTROL:
            index += 1
            continue
        length = 0
        if 0xC2 <= byte <= 0xDF:
            length = 2
        elif 0xE0 <= byte <= 0xEF:
            length = 3
        elif 0xF0 <= byte <= 0xF4:
            length = 4
        if length and index + length <= len(data):
            tail = data[index + 1 : index + length]
            if all(0x80 <= b <= 0xBF for b in tail):
                index += length
                continue
        illegal += 1
        index += 1
    return illegal / max(len(data), 1), illegal


def valid_utf8(data: bytes) -> bool:
    try:
        data.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    return True


def evaluate(
    checkpoint: Path,
    vocab: set[str],
    prompts: tuple[bytes, ...],
    new_bytes: int,
    temperature: float,
) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state.get("config") or {}
    model = ByteTransformer(ModelConfig(**config.get("model", config)))
    model.load_state_dict(state["model_state"])
    model.eval()

    words_total = 0
    words_oov = 0
    byte_total = 0
    byte_illegal = 0
    samples_invalid = 0
    oov_examples: list[str] = []
    texts = []
    for index, prompt in enumerate(prompts):
        text = generate_bytes(
            model, prompt, new_bytes, temperature=temperature, seed=1000 + index
        )
        texts.append(text.decode("utf-8", errors="replace"))
        samples_invalid += 0 if valid_utf8(text) else 1
        share, count = illegal_byte_share(text)
        byte_total += len(text)
        byte_illegal += count
        for match in WORD.finditer(text):
            words_total += 1
            word = match.group(0).lower().decode("ascii")
            if word not in vocab:
                words_oov += 1
                if len(oov_examples) < 20:
                    oov_examples.append(word)

    return {
        "checkpoint": str(checkpoint),
        "step": int(state.get("step", -1)),
        "temperature": temperature,
        "invalid_utf8_samples": samples_invalid,
        "sample_count": len(prompts),
        "invalid_utf8_rate": samples_invalid / len(prompts),
        "illegal_byte_share": byte_illegal / max(byte_total, 1),
        "illegal_bytes": byte_illegal,
        "bytes_generated": byte_total,
        "words": words_total,
        "words_outside_corpus": words_oov,
        "oov_word_rate": words_oov / max(words_total, 1),
        "oov_examples": oov_examples,
        "samples": texts,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", type=Path, help="directory or a single .pt file")
    ap.add_argument("--train-data", type=Path, default=Path("data/processed/train.bin"))
    ap.add_argument("--limit-bytes", type=int, default=64_522_880)
    ap.add_argument("--new-bytes", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument(
        "--jsonl",
        type=Path,
        default=None,
        help="append one row per checkpoint, skipping steps already recorded",
    )
    ap.add_argument("--prompts", type=int, default=3)
    ap.add_argument("--skip-samples", action="store_true")
    args = ap.parse_args()

    vocab = corpus_words(args.train_data, args.limit_bytes)
    if args.checkpoints.is_dir():
        paths = sorted(args.checkpoints.glob("checkpoint-*.pt"))
    else:
        paths = [args.checkpoints]
    if not paths:
        sys.exit(f"no checkpoints under {args.checkpoints}")

    prompts = PROMPTS[: args.prompts]
    done: set[int] = set()
    if args.jsonl and args.jsonl.exists():
        for line in args.jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["step"])

    rows = []
    for path in paths:
        step = int(torch.load(path, map_location="cpu", weights_only=True)["step"])
        if step in done:
            continue
        row = evaluate(path, vocab, prompts, args.new_bytes, args.temperature)
        if args.skip_samples:
            row.pop("samples")
        rows.append(row)
        if args.jsonl:
            with args.jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        print(
            f"{row['step']:>7} {row['invalid_utf8_rate']:>13.2%} "
            f"{row['illegal_byte_share']:>14.2%} {row['oov_word_rate']:>9.2%} "
            f"{row['words']:>7}",
            flush=True,
        )
    if not rows:
        print("nothing new", flush=True)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
