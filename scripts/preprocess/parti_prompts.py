from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    assert args.input.exists(), f"missing PartiPrompts input: {args.input}"
    prompts = load_prompts(args.input)
    assert prompts, f"no prompts found under {args.input}"
    if args.limit is not None:
        prompts = prompts[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for index, prompt in enumerate(prompts):
            f.write(
                json.dumps(
                    {
                        "request_id": f"parti_{index}",
                        "prompt": prompt,
                        "resolution": args.resolution,
                        "denoising_steps": args.denoising_steps,
                    }
                )
                + "\n"
            )
    print(
        json.dumps(
            {
                "status": "ok",
                "input": str(args.input),
                "output": str(args.output),
                "prompt_count": len(prompts),
                "resolution": args.resolution,
                "denoising_steps": args.denoising_steps,
            },
            indent=2,
        )
    )


def load_prompts(root: Path) -> list[str]:
    files = [root] if root.is_file() else sorted(path for path in root.rglob("*") if path.is_file())
    prompts: list[str] = []
    for path in files:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            prompts.extend(prompts_from_jsonl(path))
        elif suffix == ".json":
            prompts.extend(prompts_from_json(json.loads(path.read_text())))
        elif suffix in (".csv", ".tsv"):
            prompts.extend(prompts_from_table(path, delimiter="\t" if suffix == ".tsv" else ","))
        elif suffix == ".txt":
            prompts.extend(line.strip() for line in path.read_text().splitlines() if line.strip())
    return [prompt for prompt in prompts if prompt]


def prompts_from_jsonl(path: Path) -> list[str]:
    prompts: list[str] = []
    for line in path.read_text().splitlines():
        if line.strip():
            prompts.extend(prompts_from_json(json.loads(line)))
    return prompts


def prompts_from_json(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        prompts: list[str] = []
        for item in value:
            prompts.extend(prompts_from_json(item))
        return prompts
    if isinstance(value, dict):
        for key in ("prompt", "Prompt", "text", "caption"):
            if key in value and isinstance(value[key], str):
                return [value[key]]
    return []


def prompts_from_table(path: Path, *, delimiter: str) -> list[str]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        prompts: list[str] = []
        for row in reader:
            for key in ("prompt", "Prompt", "text", "caption"):
                if row.get(key):
                    prompts.append(row[key])
                    break
        return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare PartiPrompts diffusion prompt manifest")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--denoising-steps", type=int, default=28)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    main()
