"""Audit what the model actually learned, with numbers instead of impressions.

Each probe targets one capability and reports a rate, so "learned" and "not learned"
stop being a judgement call:

  next-byte accuracy     argmax accuracy on held-out validation bytes, and split by
                         context (inside a word vs at a word boundary)
  word completion        given a real word's prefix, does top-1 produce its suffix
  copy rate              share of generated 6-grams that already exist verbatim in
                         the training slice -- separarates generalising from reciting
  self-repetition        share of 5-grams that repeat inside one generated sample,
                         the signature of a model that has run out of context ideas
  sentence shape         words per sentence in generations vs the corpus

    python scripts/audit_learned.py checkpoints/cpu-cont/checkpoint-00015752.pt
"""

from __future__ import annotations

import argparse
import random
import re
import statistics
from pathlib import Path

import torch

from tinystories_byte_lm.config import ModelConfig
from tinystories_byte_lm.data import MappedByteStories
from tinystories_byte_lm.generation import generate_bytes
from tinystories_byte_lm.model import ByteTransformer

WORD = re.compile(rb"[A-Za-z]{4,}")
SENTENCE = re.compile(rb"[^.!?]+[.!?]")
GRAM = 6
LOOP_GRAM = 5


def load(checkpoint: Path) -> tuple[ByteTransformer, dict]:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = state.get("config") or {}
    model = ByteTransformer(ModelConfig(**config.get("model", config)))
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, config


def next_byte_accuracy(model: ByteTransformer, data: MappedByteStories, batches: int) -> dict:
    rng = random.Random(1234)
    correct = total = 0
    correct_inside = total_inside = 0
    correct_boundary = total_boundary = 0
    with torch.no_grad():
        for _ in range(batches):
            inputs, targets = data.sample_batch(32, model.config.context_length, rng)
            logits = model(inputs)
            predicted = logits.argmax(dim=-1)
            mask = targets != -100
            hits = (predicted == targets) & mask
            correct += int(hits.sum())
            total += int(mask.sum())
            # byte position i predicts byte i+1; boundary means the target starts a word
            # (previous input byte was a space or a sentence end)
            previous = inputs
            boundary = (previous == 32) | (previous == 46) | (previous == 10)
            boundary = torch.cat([torch.zeros_like(boundary[:, :1]), boundary[:, :-1]], dim=1)
            inside = mask & ~boundary
            boundary = mask & boundary
            correct_inside += int((hits & inside).sum())
            total_inside += int(inside.sum())
            correct_boundary += int((hits & boundary).sum())
            total_boundary += int(boundary.sum())
    return {
        "overall": correct / total,
        "inside_word": correct_inside / max(total_inside, 1),
        "word_start": correct_boundary / max(total_boundary, 1),
        "positions": total,
    }


def word_completion(model: ByteTransformer, raw: bytes, samples: int) -> dict:
    rng = random.Random(99)
    words: list[bytes] = []
    for match in WORD.finditer(raw):
        word = match.group(0).lower()
        if len(word) >= 6 and word.isalpha():
            words.append(word)
        if len(words) >= samples * 20:
            break
    rng.shuffle(words)
    words = words[:samples]

    first_hit = full_hit = 0
    with torch.no_grad():
        for word in words:
            prefix = b" " + word[: len(word) - 3]
            suffix = word[len(word) - 3 :]
            context = torch.tensor(list(prefix), dtype=torch.long).unsqueeze(0)
            logits = model(context)[0, -1]
            predicted = int(logits.argmax())
            first_hit += int(predicted == suffix[0])
            generated = bytes([predicted])
            for _ in range(len(suffix) - 1):
                context = torch.tensor(
                    list(prefix) + list(generated), dtype=torch.long
                ).unsqueeze(0)
                logits = model(context)[0, -1]
                byte = int(logits.argmax())
                generated += bytes([byte])
            full_hit += int(generated == suffix)
    return {
        "words": len(words),
        "first_byte": first_hit / len(words),
        "whole_suffix": full_hit / len(words),
    }


def gram_set(raw: bytes, gram: int, limit: int = 8_000_000) -> set[bytes]:
    end = min(len(raw) - gram, limit)
    return {raw[index : index + gram] for index in range(end)}


def copy_and_loop(
    model: ByteTransformer, train_grams: set[bytes], samples: int, length: int
) -> dict:
    prompts = (b"Once upon a time", b"The little dog", b"One day,")
    copied = grams = 0
    looped = loop_total = 0
    for index in range(samples):
        prompt = prompts[index % len(prompts)]
        text = generate_bytes(model, prompt, length, temperature=0.7, seed=100 + index)
        grams_in_text = [text[i : i + GRAM] for i in range(len(text) - GRAM)]
        copied += sum(1 for gram in grams_in_text if gram in train_grams)
        grams += len(grams_in_text)
        loops = [text[i : i + LOOP_GRAM] for i in range(len(text) - LOOP_GRAM)]
        looped += len(loops) - len(set(loops))
        loop_total += len(loops)
    return {"copy_rate": copied / max(grams, 1), "self_repeat": looped / max(loop_total, 1)}


def sentence_shape(model: ByteTransformer, corpus: bytes, samples: int, length: int) -> dict:
    generated = []
    prompts = (b"Once upon a time", b"The little dog", b"One day,")
    for index in range(samples):
        text = generate_bytes(
            model, prompts[index % len(prompts)], length, temperature=0.7, seed=200 + index
        )
        for match in SENTENCE.finditer(text):
            words = len(match.group(0).split())
            if 1 <= words <= 60:
                generated.append(words)
    corpus_words = [len(m.group(0).split()) for m in SENTENCE.finditer(corpus[:2_000_000])]
    corpus_words = [w for w in corpus_words if 1 <= w <= 60]
    return {
        "generated_sentences": len(generated),
        "generated_mean_words": statistics.fmean(generated) if generated else 0,
        "corpus_mean_words": statistics.fmean(corpus_words) if corpus_words else 0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--train-data", type=Path, default=Path("data/processed/train.bin"))
    ap.add_argument("--train-index", type=Path, default=Path("data/processed/train.idx"))
    ap.add_argument("--validation-data", type=Path, default=Path("data/processed/validation.bin"))
    ap.add_argument("--validation-index", type=Path, default=Path("data/processed/validation.idx"))
    ap.add_argument("--limit-bytes", type=int, default=64_522_880)
    ap.add_argument("--accuracy-batches", type=int, default=20)
    ap.add_argument("--word-samples", type=int, default=200)
    ap.add_argument("--gen-samples", type=int, default=8)
    ap.add_argument("--gen-length", type=int, default=240)
    args = ap.parse_args()

    model, config = load(args.checkpoint)
    raw = open(args.train_data, "rb").read()[: args.limit_bytes]
    train_grams = gram_set(raw, GRAM)
    with MappedByteStories(args.train_data, args.train_index, max_bytes=args.limit_bytes) as train:
        with MappedByteStories(args.validation_data, args.validation_index) as validation:
            accuracy = next_byte_accuracy(model, validation, args.accuracy_batches)
    del train
    completion = word_completion(model, raw, args.word_samples)
    copy = copy_and_loop(model, train_grams, args.gen_samples, args.gen_length)
    shape = sentence_shape(model, raw, args.gen_samples, args.gen_length)

    print(f"checkpoint {args.checkpoint.name}")
    print(f"next-byte accuracy: overall {accuracy['overall']:.1%} "
          f"| inside word {accuracy['inside_word']:.1%} "
          f"| word start {accuracy['word_start']:.1%} "
          f"({accuracy['positions']:,} positions)")
    print(f"word completion ({completion['words']} words, 3-byte suffix): "
          f"first byte {completion['first_byte']:.1%} | whole suffix {completion['whole_suffix']:.1%}")
    print(f"copy rate (share of 6-grams present in training slice): {copy['copy_rate']:.1%}")
    print(f"self repetition (5-grams repeated inside one sample): {copy['self_repeat']:.1%}")
    print(f"sentence shape: generated {shape['generated_mean_words']:.1f} words "
          f"vs corpus {shape['corpus_mean_words']:.1f} words "
          f"({shape['generated_sentences']} sentences)")


if __name__ == "__main__":
    main()
