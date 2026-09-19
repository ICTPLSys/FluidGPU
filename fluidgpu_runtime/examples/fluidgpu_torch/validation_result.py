from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any

import torch


ERROR_PATTERNS = (
    "Traceback",
    "AssertionError",
    "RuntimeError",
    "DistBackendError",
    "ncclUnhandledCudaError",
    "CUDA error",
    "shape mismatch",
)


def main() -> None:
    args = parse_args()
    rank0_text = read_text(args.rank0_log)
    rank1_text = read_text(args.rank1_log)
    combined = "\n".join([rank0_text, rank1_text])
    tokens = parse_generated_tokens(rank0_text)
    manifest: dict[str, Any] = {
        "name": args.name,
        "rank0_log": str(args.rank0_log),
        "rank1_log": str(args.rank1_log) if args.rank1_log is not None else None,
        "diagnostics": str(args.diagnostics) if args.diagnostics is not None else None,
        "parity_log": str(args.parity_log) if args.parity_log is not None else None,
        "generated_token_ids": tokens,
        "generated_token_count": len(tokens),
        "first_logits_argmax": parse_first_logits_argmax(rank0_text),
        "request_indices": parse_request_indices(rank0_text),
        "gdrdma_seen": "GDRDMA" in combined or "GDR 1" in combined,
        "errors": [pattern for pattern in ERROR_PATTERNS if pattern in combined],
    }
    if args.diagnostics is not None and args.diagnostics.exists():
        manifest["diagnostics_summary"] = diagnostics_summary(args.diagnostics)
    if args.parity_log is not None and args.parity_log.exists():
        manifest["parity_summary"] = parity_summary(args.parity_log)

    assert not manifest["errors"], f"validation log contains errors: {manifest['errors']}"
    if args.require_generated_tokens is not None:
        assert len(tokens) == args.require_generated_tokens, (
            f"generated token count {len(tokens)} != {args.require_generated_tokens}"
        )
    if args.require_request_count is not None:
        assert len(manifest["request_indices"]) == args.require_request_count, (
            f"request count {len(manifest['request_indices'])} != {args.require_request_count}"
        )
    if args.require_gdr:
        assert manifest["gdrdma_seen"], "GDRDMA/GDR 1 was not found in logs"

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"validation_result name={args.name} output={args.output_json}")


def read_text(path: Path | None) -> str:
    if path is None:
        return ""
    return path.read_text(errors="replace")


def parse_generated_tokens(text: str) -> list[int]:
    match = re.search(r"generated_token_ids:\s*(\[[^\n]*\])", text)
    if match is None:
        return []
    value = ast.literal_eval(match.group(1))
    return [int(token) for token in value]


def parse_first_logits_argmax(text: str) -> int | None:
    match = re.search(r"first_logits_argmax:\s*(-?\d+)", text)
    if match is None:
        return None
    return int(match.group(1))


def parse_request_indices(text: str) -> list[int]:
    return [int(value) for value in re.findall(r"request_index:\s*(\d+)", text)]


def diagnostics_summary(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    tokens = [int(token) for token in payload.get("tokens", [])]
    first_logits = payload.get("first_logits")
    return {
        "model": payload.get("model"),
        "prompt": payload.get("prompt"),
        "max_new_tokens": payload.get("max_new_tokens"),
        "seed": payload.get("seed"),
        "token_count": len(tokens),
        "tokens": tokens,
        "first_logits_argmax": int(first_logits.argmax(dim=-1).item())
        if first_logits is not None
        else None,
    }


def parity_summary(path: Path) -> dict[str, float]:
    text = path.read_text(errors="replace")
    return {
        "token_match_ratio": parse_float(text, r"token_match_ratio@\d+:\s*([0-9.]+)"),
        "first_logits_cosine": parse_float(text, r"first_logits_cosine:\s*([0-9.]+)"),
        "first_logits_max_abs": parse_float(text, r"first_logits_max_abs:\s*([0-9.]+)"),
    }


def parse_float(text: str, pattern: str) -> float | None:
    match = re.search(pattern, text)
    if match is None:
        return None
    return float(match.group(1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a reproducible FluidGPU validation result manifest"
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--rank0-log", type=Path, required=True)
    parser.add_argument("--rank1-log", type=Path)
    parser.add_argument("--diagnostics", type=Path)
    parser.add_argument("--parity-log", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--require-generated-tokens", type=int)
    parser.add_argument("--require-request-count", type=int)
    parser.add_argument("--require-gdr", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
