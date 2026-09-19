"""Aggregate a per-kernel census CSV into the planner's per-task cost table.

The eager task timer (profile_llm_tasks.py) includes host launch overhead,
which dominates decode-phase numbers and hides real inter-GPU differences
once the runtime replays CUDA graphs. The kernel census
(profile_llm_kernels.py) records pure device time, split by phase and block;
this tool spreads each block total uniformly across layers to produce the
name,prefill_us,decode_us table the schedulers consume.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    args = parse_args()
    totals: dict[tuple[str, str], float] = defaultdict(float)
    for row in csv.DictReader(args.census.open()):
        totals[(row["phase"], row["block"])] += float(row["total_us"])

    def per_call(phase: str, block: str, splits: int = 1) -> float:
        return totals[(phase, block)] / args.repeat / splits

    rows: list[dict[str, object]] = [
        {
            "name": "embed",
            "prefill_us": f"{per_call('prefill', 'embed'):.3f}",
            "decode_us": f"{per_call('decode', 'embed'):.3f}",
        }
    ]
    for layer in range(args.num_layers):
        rows.append(
            {
                "name": f"layer_{layer}_attn",
                "prefill_us": f"{per_call('prefill', 'attn', args.num_layers):.3f}",
                "decode_us": f"{per_call('decode', 'attn', args.num_layers):.3f}",
            }
        )
        rows.append(
            {
                "name": f"layer_{layer}_mlp",
                "prefill_us": f"{per_call('prefill', 'ffn', args.num_layers):.3f}",
                "decode_us": f"{per_call('decode', 'ffn', args.num_layers):.3f}",
            }
        )
    rows.append(
        {
            "name": "norm_lm_head",
            "prefill_us": f"{per_call('prefill', 'head'):.3f}",
            "decode_us": f"{per_call('decode', 'head'):.3f}",
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "prefill_us", "decode_us"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"aggregated {args.census} -> {args.output} ({len(rows)} tasks)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a per-kernel census CSV into a planner task cost table"
    )
    parser.add_argument("--census", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument(
        "--repeat",
        type=int,
        default=5,
        help="census --repeat value (totals are per-N-iterations)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
