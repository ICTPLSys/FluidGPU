from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F


def main() -> None:
    args = parse_args()
    baseline = torch.load(args.baseline, map_location="cpu")
    candidate = torch.load(args.candidate, map_location="cpu")

    baseline_tokens = [int(token) for token in baseline["tokens"]]
    candidate_tokens = [int(token) for token in candidate["tokens"]]
    compare_count = min(args.token_count, len(baseline_tokens), len(candidate_tokens))
    assert compare_count > 0, "no generated tokens to compare"

    baseline_prefix = baseline_tokens[:compare_count]
    candidate_prefix = candidate_tokens[:compare_count]
    matches = sum(1 for a, b in zip(baseline_prefix, candidate_prefix) if a == b)
    token_match_ratio = matches / compare_count

    baseline_logits = baseline["first_logits"].float().flatten()
    candidate_logits = candidate["first_logits"].float().flatten()
    assert baseline_logits.shape == candidate_logits.shape, (
        f"logit shape mismatch: {baseline_logits.shape} vs {candidate_logits.shape}"
    )
    cosine = F.cosine_similarity(baseline_logits, candidate_logits, dim=0).item()
    max_abs = (baseline_logits - candidate_logits).abs().max().item()

    print("baseline_tokens:", baseline_tokens)
    print("candidate_tokens:", candidate_tokens)
    print(f"token_match_ratio@{compare_count}: {token_match_ratio:.6f}")
    print(f"first_logits_cosine: {cosine:.9f}")
    print(f"first_logits_max_abs: {max_abs:.9f}")

    assert token_match_ratio >= args.min_token_match_ratio, (
        f"token match ratio {token_match_ratio:.6f} < {args.min_token_match_ratio}"
    )
    assert cosine >= args.min_cosine, f"logits cosine {cosine:.9f} < {args.min_cosine}"
    if args.max_abs is not None:
        assert max_abs <= args.max_abs, f"logits max_abs {max_abs:.9f} > {args.max_abs}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare FluidGPU output against single-GPU baseline")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--token-count", type=int, default=8)
    parser.add_argument("--min-token-match-ratio", type=float, default=1.0)
    parser.add_argument("--min-cosine", type=float, default=0.999)
    parser.add_argument("--max-abs", type=float)
    return parser.parse_args()


if __name__ == "__main__":
    main()
