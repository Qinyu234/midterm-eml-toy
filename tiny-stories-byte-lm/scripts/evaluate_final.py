"""The measurements docs/experiment-design.md requires, all in one run.

Four things the design calls for and that were still missing:

  strict UTF-8 validity   with a Wilson 95% interval, because 3 samples cannot
                          support a "0%" claim
  legal-byte probability  the share of the model's own 256-way probability that
                          falls on bytes the UTF-8 decoder still accepts, at every
                          generated position -- distinct from empirical validity
  word probability        conditional log probability of each validation word given
                          its story's preceding bytes: per word, per byte, and the
                          byte-normalised perplexity
  word frequency          the corpus's own case-folded word distribution, reported
                          separately so model likelihood is never confused with it

Plus the untouched test split, which the design reserves for final evaluation.

    python scripts/evaluate_final.py checkpoints/cpu-cont/checkpoint-00015752.pt
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

import torch

from tinystories_byte_lm.config import ModelConfig
from tinystories_byte_lm.model import ByteTransformer

WORD = re.compile(rb"[A-Za-z]+(?:'[A-Za-z]+)?")
CONTINUATION = range(0x80, 0xC0)


def load(checkpoint: Path) -> tuple[ByteTransformer, dict]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state.get("config") or {}
    model = ByteTransformer(ModelConfig(**config.get("model", config)))
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, state


def wilson(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(centre - half, 0.0), min(centre + half, 1.0)


# --- UTF-8 decoder state, as (bytes still needed, low, high) for the next byte ---

START = (0, 0, 0)


def allowed_mask(state: tuple[int, int, int]) -> list[bool]:
    need, low, high = state
    mask = [False] * 256
    if need == 0:
        for byte in range(0x80):
            mask[byte] = True
        for byte in range(0xC2, 0xF5):
            mask[byte] = True
    else:
        for byte in range(low, high + 1):
            mask[byte] = True
    return mask


def advance(state: tuple[int, int, int], byte: int) -> tuple[int, int, int]:
    need, _, _ = state
    if need > 0:
        return (need - 1, 0x80, 0xBF) if need - 1 > 0 else START
    if byte < 0x80:
        return START
    if byte <= 0xDF:
        return (1, 0x80, 0xBF)
    if byte == 0xE0:
        return (2, 0xA0, 0xBF)
    if byte <= 0xEC:
        return (2, 0x80, 0xBF)
    if byte == 0xED:
        return (2, 0x80, 0x9F)
    if byte <= 0xEF:
        return (2, 0x80, 0xBF)
    if byte == 0xF0:
        return (3, 0x90, 0xBF)
    if byte <= 0xF3:
        return (3, 0x80, 0xBF)
    return (3, 0x80, 0x8F)


def generate_batch(
    model: ByteTransformer, prompts: list[bytes], new_bytes: int, temperature: float, seed: int
) -> list[dict]:
    """Sample many continuations in parallel, recording the legal-byte mass per position.

    Sequences are generated as one batch per distinct prompt length, because batching
    is what makes this affordable: 40 sequences share every forward pass, so the cost
    is `new_bytes` forwards per prompt length instead of `new_bytes * samples`.
    """
    generator = torch.Generator().manual_seed(seed)
    context_length = model.config.context_length

    grouped: dict[int, list[int]] = {}
    for index, prompt in enumerate(prompts):
        grouped.setdefault(len(prompt), []).append(index)

    results: list[dict] = [{} for _ in prompts]
    for prompt_length, indices in grouped.items():
        batch = len(indices)
        contexts = [list(prompts[index]) for index in indices]
        states = [START for _ in indices]
        for position in range(prompt_length):
            for row, index in enumerate(indices):
                states[row] = advance(states[row], prompts[index][position])
        produced = [[] for _ in indices]
        masses = [[] for _ in indices]
        for _ in range(new_bytes):
            window = torch.tensor([c[-context_length:] for c in contexts], dtype=torch.long)
            with torch.no_grad():
                logits = model(window)[:, -1, :]
            probabilities = torch.softmax(logits, dim=-1)
            mask = torch.tensor([allowed_mask(state) for state in states])
            legal = (probabilities * mask).sum(dim=-1)
            choices = torch.multinomial(
                torch.softmax(logits / temperature, dim=-1), 1, generator=generator
            ).squeeze(-1)
            for row in range(batch):
                choice = int(choices[row])
                masses[row].append(float(legal[row]))
                produced[row].append(choice)
                contexts[row].append(choice)
                states[row] = advance(states[row], choice)
        for row, index in enumerate(indices):
            text = prompts[index] + bytes(produced[row])
            try:
                text.decode("utf-8", errors="strict")
                valid = True
            except UnicodeDecodeError:
                valid = False
            results[index] = {
                "bytes": len(produced[row]),
                "valid": valid,
                "mean_legal_mass": sum(masses[row]) / len(masses[row]) if masses[row] else 0.0,
                "text": text,
            }
    return results


def story_logprobs(model: ByteTransformer, raw: bytes, tiles: int = 16) -> list[float]:
    """Conditional log p for every byte after the first, using at most ctx bytes of context.

    Full 256-byte tiles are batched -- `tiles` windows share each forward pass -- and the
    trailing short tile (if any) is run alone, since a ragged batch is not a tensor.
    """
    context_length = model.config.context_length
    starts = list(range(0, len(raw) - 1, context_length))
    logprobs: list[float] = []
    full = [start for start in starts if start + context_length <= len(raw)]
    remainder = [start for start in starts if start + context_length > len(raw)]

    for batch_start in range(0, len(full), tiles):
        batch = full[batch_start : batch_start + tiles]
        windows = [raw[start : start + context_length] for start in batch]
        with torch.no_grad():
            logits = model(torch.tensor([list(w) for w in windows], dtype=torch.long))
        log_softmax = torch.log_softmax(logits, dim=-1)
        # window[i] predicts raw[start + i + 1]
        targets = torch.tensor(
            [list(raw[start + 1 : start + context_length + 1]) for start in batch]
        )
        picked = log_softmax.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        logprobs.extend(float(value) for value in picked.reshape(-1))

    for start in remainder:
        window = raw[start : len(raw) - 1]
        with torch.no_grad():
            logits = model(torch.tensor(list(window), dtype=torch.long).unsqueeze(0))[0]
        log_softmax = torch.log_softmax(logits, dim=-1)
        for offset, target in enumerate(raw[start + 1 :]):
            logprobs.append(float(log_softmax[offset, target]))
    return logprobs


def word_metrics(model: ByteTransformer, raw: bytes, limit: int) -> dict:
    """Word-level conditional log probability over the first `limit` bytes."""
    raw = raw[:limit]
    logprobs = story_logprobs(model, raw)
    # logprobs[i] is the log probability of raw[i + 1]
    word_logprobs: list[float] = []
    for match in WORD.finditer(raw):
        start, end = match.span()
        if end <= 1:
            continue
        values = logprobs[start - 1 : end - 1] if start >= 1 else logprobs[0 : end - 1]
        if values:
            word_logprobs.append(sum(values))
    total_logprob = sum(logprobs)
    per_byte = total_logprob / len(logprobs)
    words = len(word_logprobs)
    return {
        "bytes": len(logprobs),
        "words": words,
        "mean_logprob_per_word": sum(word_logprobs) / words if words else 0.0,
        "mean_logprob_per_byte": per_byte,
        "byte_perplexity": math.exp(-per_byte),
        "byte_cross_entropy_nats": -per_byte,
    }


def word_frequency(raw: bytes) -> dict:
    counts = Counter(match.group(0).lower().decode("ascii") for match in WORD.finditer(raw))
    total = sum(counts.values())
    hapax = sum(1 for count in counts.values() if count == 1)
    return {
        "total_words": total,
        "distinct_words": len(counts),
        "hapax_share": hapax / len(counts) if counts else 0.0,
        "most_common": counts.most_common(20),
        "words_to_cover_90pct": _coverage(counts, 0.9),
    }


def _coverage(counts: Counter, target: float) -> int:
    total = sum(counts.values())
    running = 0
    for index, (_, count) in enumerate(counts.most_common(), start=1):
        running += count
        if running / total >= target:
            return index
    return len(counts)


def split_loss(model: ByteTransformer, data_path: Path, index_path: Path, batches: int) -> dict:
    """Story-aware next-byte loss, the same sampling path training used for validation."""
    from tinystories_byte_lm.data import IGNORE_TARGET, MappedByteStories

    correct = total = 0
    logprob_sum = 0.0
    rng = random.Random(4242)
    with MappedByteStories(data_path, index_path) as split:
        for _ in range(batches):
            inputs, targets = split.sample_batch(32, model.config.context_length, rng)
            with torch.no_grad():
                logits = model(inputs)
            mask = targets != IGNORE_TARGET
            log_softmax = torch.log_softmax(logits, dim=-1)
            picked = log_softmax.gather(2, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
            logprob_sum += float((picked * mask).sum())
            correct += int(((logits.argmax(dim=-1) == targets) & mask).sum())
            total += int(mask.sum())
    return {
        "windows": batches * 32,
        "bytes": total,
        "cross_entropy_nats": -logprob_sum / total,
        "bits_per_byte": -logprob_sum / total / math.log(2),
        "accuracy": correct / total,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--validation-data", type=Path, default=Path("data/processed/validation.bin"))
    ap.add_argument("--validation-index", type=Path, default=Path("data/processed/validation.idx"))
    ap.add_argument("--test-data", type=Path, default=Path("data/processed/test.bin"))
    ap.add_argument("--test-index", type=Path, default=Path("data/processed/test.idx"))
    ap.add_argument("--limit-bytes", type=int, default=64_522_880)
    ap.add_argument("--loss-batches", type=int, default=200)
    ap.add_argument("--validity-samples", type=int, default=120)
    ap.add_argument("--validity-bytes", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--word-limit", type=int, default=2_000_000)
    ap.add_argument("--out", type=Path, default=Path("reports/final-eval.json"))
    args = ap.parse_args()

    model, state = load(args.checkpoint)
    step = int(state.get("step", -1))
    prompts = (b"Once upon a time", b"The little dog", b"Lily wanted to")

    samples = generate_batch(
        model,
        [prompts[index % len(prompts)] for index in range(args.validity_samples)],
        args.validity_bytes,
        args.temperature,
        seed=4000,
    )
    valid = sum(1 for sample in samples if sample["valid"])
    low, high = wilson(valid, len(samples))
    legal_mass = sum(sample["mean_legal_mass"] for sample in samples) / len(samples)

    validation_raw = args.validation_data.read_bytes()
    test_raw = args.test_data.read_bytes()

    result = {
        "checkpoint": str(args.checkpoint),
        "step": step,
        "utf8_validity": {
            "samples": len(samples),
            "valid": valid,
            "rate": valid / len(samples),
            "wilson95": [low, high],
            "bytes_per_sample": args.validity_bytes,
            "temperature": args.temperature,
        },
        "legal_probability_mass": {
            "mean": legal_mass,
            "note": "sum of the model's raw 256-way probability over bytes the decoder accepts",
        },
        "validation_word_probability": word_metrics(model, validation_raw, args.word_limit),
    }

    result["validation_loss"] = split_loss(
        model, args.validation_data, args.validation_index, args.loss_batches
    )
    result["test_loss"] = split_loss(
        model, args.test_data, args.test_index, args.loss_batches
    )
    result["test_word_probability"] = word_metrics(model, test_raw, args.word_limit)
    result["validation_word_frequency"] = word_frequency(validation_raw)
    result["generated_from_test"] = []
    rng = random.Random(31)
    offsets = [rng.randrange(len(test_raw) - 400) for _ in range(6)]
    test_prompts = []
    for offset in offsets:
        prompt = test_raw[offset : offset + 40]
        prompt = prompt[prompt.rfind(b".") + 1 :][:40] if b"." in prompt else prompt
        test_prompts.append(prompt)
    test_samples = generate_batch(model, test_prompts, 240, args.temperature, seed=5000)
    for prompt, sample in zip(test_prompts, test_samples):
        result["generated_from_test"].append(
            {"prompt": prompt.decode("utf-8", "replace"),
             "text": sample["text"].decode("utf-8", "replace"),
             "valid": sample["valid"],
             "mean_legal_mass": sample["mean_legal_mass"]}
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    v = result["utf8_validity"]
    print(f"step {step}")
    print(f"strict UTF-8 validity: {v['valid']}/{v['samples']} = {v['rate']:.1%} "
          f"(Wilson 95% {v['wilson95'][0]:.1%}-{v['wilson95'][1]:.1%})")
    print(f"legal-byte probability mass: mean {legal_mass:.3%}")
    w = result["validation_word_probability"]
    print(f"validation words: {w['words']}, mean logprob/word {w['mean_logprob_per_word']:.3f}, "
          f"/byte {w['mean_logprob_per_byte']:.3f}, byte ppl {w['byte_perplexity']:.2f}")
    print(f"validation loss {result['validation_loss']['cross_entropy_nats']:.3f} nats "
          f"({result['validation_loss']['bits_per_byte']:.3f} bits/byte, "
          f"acc {result['validation_loss']['accuracy']:.1%})")
    print(f"test loss {result['test_loss']['cross_entropy_nats']:.3f} nats "
          f"({result['test_loss']['bits_per_byte']:.3f} bits/byte, "
          f"acc {result['test_loss']['accuracy']:.1%})")
    f = result["validation_word_frequency"]
    print(f"validation words: {f['total_words']} tokens, {f['distinct_words']} types, "
          f"hapax {f['hapax_share']:.1%}, {f['words_to_cover_90pct']} types cover 90%")
    print(f"written to {args.out}")

if __name__ == "__main__":
    main()
