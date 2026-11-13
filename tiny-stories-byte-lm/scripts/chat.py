"""Interactive continuation with the trained byte model.

    python scripts/chat.py checkpoints/cpu-cont/checkpoint-00015752.pt

It is not a chatbot: the model has no instruction or turn format, it only continues
the bytes you give it. The script keeps the continuation in context so the next line
builds on the previous one; `/reset` clears it, `/temp 0.9` changes sampling, `/quit`
exits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tinystories_byte_lm.config import ModelConfig
from tinystories_byte_lm.generation import generate_bytes
from tinystories_byte_lm.model import ByteTransformer


def load(path: Path) -> tuple[ByteTransformer, int]:
    state = torch.load(path, map_location="cpu", weights_only=True)
    config = state.get("config") or {}
    model = ByteTransformer(ModelConfig(**config.get("model", config)))
    model.load_state_dict(state["model_state"])
    model.eval()
    return model, int(config.get("model", {}).get("context_length", 256))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--bytes", type=int, default=200, help="continuation length per turn")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    model, context_length = load(args.checkpoint)
    history = b""
    temperature = args.temperature
    turn = 0
    print(f"loaded {args.checkpoint.name}; context {context_length} bytes; "
          f"temperature {temperature}. /reset /temp X /quit")
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line == "/quit":
            break
        if line == "/reset":
            history = b""
            print("(context cleared)")
            continue
        if line.startswith("/temp"):
            try:
                temperature = float(line.split()[1])
                print(f"(temperature {temperature})")
            except (IndexError, ValueError):
                print("(usage: /temp 0.9)")
            continue

        turn += 1
        seed = args.seed + turn
        prompt = line.encode("utf-8")
        continuation = generate_bytes(
            model, prompt, args.bytes, temperature=temperature, seed=seed
        )[len(prompt):]
        history = (history + prompt + continuation)[-context_length:]
        print(continuation.decode("utf-8", errors="replace"))
        print()


if __name__ == "__main__":
    main()
