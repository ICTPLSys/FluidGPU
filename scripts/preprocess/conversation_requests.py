from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def main() -> None:
    args = parse_args()
    assert args.splitwise.exists(), f"missing Splitwise-style source: {args.splitwise}"
    assert args.azure.exists(), f"missing Azure trace source: {args.azure}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    with args.output.open("w") as f:
        for index in range(args.count):
            input_tokens = jittered_tokens(rng, args.median_input_tokens)
            output_tokens = jittered_tokens(rng, args.median_output_tokens)
            f.write(
                json.dumps(
                    {
                        "request_id": f"conversation_{index}",
                        "arrival_s": round(index / args.rps, 6),
                        "prompt": synthetic_prompt(input_tokens),
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "source_splitwise": str(args.splitwise),
                        "source_azure": str(args.azure),
                    }
                )
                + "\n"
            )
    print(
        json.dumps(
            {
                "status": "ok",
                "output": str(args.output),
                "request_count": args.count,
                "median_input_tokens": args.median_input_tokens,
                "median_output_tokens": args.median_output_tokens,
            },
            indent=2,
        )
    )


def jittered_tokens(rng: random.Random, median: int) -> int:
    return max(1, int(rng.lognormvariate(0.0, 0.25) * median))


def synthetic_prompt(token_count: int) -> str:
    return " ".join(f"tok{i % 997}" for i in range(token_count))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare conversation benchmark request manifest")
    parser.add_argument("--splitwise", type=Path, required=True)
    parser.add_argument("--azure", type=Path, required=True)
    parser.add_argument("--median-input-tokens", type=int, default=1020)
    parser.add_argument("--median-output-tokens", type=int, default=129)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--rps", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.count > 0, "--count must be positive"
    assert args.rps > 0.0, "--rps must be positive"
    return args


if __name__ == "__main__":
    main()
