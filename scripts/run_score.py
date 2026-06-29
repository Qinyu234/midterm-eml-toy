"""Run full SCORE suite: functions × 16 nodes × all methods."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from codec.sweep import list_16_nodes
from experiment.score_suite import format_score_table, run_score_suite, write_score_files


def main() -> int:
    p = argparse.ArgumentParser(
        description='SCORE: targets × 16-node x_slot sweep × mlp/fourier/eml',
    )
    p.add_argument('--targets', default='all')
    p.add_argument('--methods', default='mlp,fourier,eml')
    p.add_argument('--max-steps', type=int, default=500)
    p.add_argument('--eps', type=float, default=1e-2)
    p.add_argument('--lr', type=float, default=0.01)
    p.add_argument('--txt', type=Path, default=Path('results/SCORE.txt'))
    p.add_argument('--json', type=Path, default=Path('results/SCORE.json'))
    args = p.parse_args()

    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    print(f'# 16 nodes ({len(list_16_nodes(evaluable_only=True))} evaluable), y @ tree top', flush=True)
    for n in list_16_nodes():
        flag = 'ok' if n.evaluable else 'skip'
        print(f'  [{flag}] {n.label}', flush=True)

    args.txt.parent.mkdir(parents=True, exist_ok=True)
    args.json.unlink(missing_ok=True)

    def _after_target(rows, target_id: str) -> None:
        args.txt.write_text(format_score_table(rows, eps=args.eps), encoding='utf-8')
        print(f'done target {target_id} ({len(rows)} rows)', file=sys.stderr, flush=True)

    rows = run_score_suite(
        args.targets,
        max_steps=args.max_steps,
        lr=args.lr,
        eps=args.eps,
        methods=methods,
        on_target_done=_after_target,
    )
    write_score_files(rows, txt_path=args.txt, json_path=args.json, eps=args.eps)
    print(f'\n{len(rows)} score rows → {args.txt} , {args.json}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
