"""Sample text from a saved checkpoint, so a run can be judged qualitatively.

    python scripts/sample_checkpoint.py checkpoints/cpu-8h/last.pt \
        --prompt "Once upon a time" --bytes 600

Prints the continuation plus whether the bytes survive a strict UTF-8 decode
(byte-level models can emit invalid sequences early in training, which reads as
mojibake rather than a model error).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tinystories_byte_lm.generation import generate_bytes, is_valid_utf8
from tinystories_byte_lm.model import ByteLanguageModel

PROMPTS = (
    "Once upon a time",
    "The little dog",
    "Lily wanted to",
)


def load(path: Path) -> tuple[ByteLanguageModel, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config") or {}
    model_config = config.get("model", config)
    model = ByteLanguageModel(model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--bytes", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    model, config = load(args.checkpoint)
    prompts = (args.prompt,) if args.prompt else PROMPTS
    results = []
    for index, prompt in enumerate(prompts):
        text = generate_bytes(
            model,
            prompt.encode("utf-8"),
            args.bytes,
            temperature=args.temperature,
            seed=args.seed + index,
        )
        valid = is_valid_utf8(text)
        print(f"--- {args.checkpoint} | prompt={prompt!r} | valid_utf8={valid} ---")
        print(text.decode("utf-8", errors="replace"))
        print()
        results.append(
            {
                "prompt": prompt,
                "valid_utf8": valid,
                "text": text.decode("utf-8", errors="replace"),
            }
        )
    if args.json:
        args.json.write_text(
            json.dumps(
                {"checkpoint": str(args.checkpoint), "config": config, "samples": results},
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
