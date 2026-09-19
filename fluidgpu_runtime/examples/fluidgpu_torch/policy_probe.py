from __future__ import annotations

import argparse
from pathlib import Path

from fluidgpu_torch.policy import (
    PolicyCandidate,
    PolicyConfig,
    choose_policy,
    read_policy_candidates,
    write_policy_decision,
)


def main() -> None:
    args = parse_args()
    candidates = demo_candidates() if args.demo else read_policy_candidates(args.input_csv)
    config = PolicyConfig(
        objective=args.objective,
        decode_weight=args.decode_weight,
        latency_weight=args.latency_weight,
        throughput_weight=args.throughput_weight,
        runtime_compatible_only=not args.allow_offline_profiles,
        max_decode_p95_us=args.max_decode_p95_us,
        max_error_rate=args.max_error_rate,
        min_tokens_per_s=args.min_tokens_per_s,
    )
    decision = choose_policy(candidates, config)
    write_policy_decision(args.output_json, decision)
    print(
        "policy_decision "
        f"objective={decision.objective} winner={decision.winner.candidate.name} "
        f"score={decision.winner.score:.6f} considered={len(decision.considered)} "
        f"rejected={len(decision.rejected)} output={args.output_json}"
    )


def demo_candidates() -> list[PolicyCandidate]:
    return [
        PolicyCandidate(
            name="low_latency_profile",
            profile_path="/tmp/profiles/low_latency.json",
            prefill_us=10000.0,
            decode_us=100.0,
            tokens_per_s=700.0,
            decode_p95_us=150.0,
            runtime_compatible=True,
        ),
        PolicyCandidate(
            name="high_throughput_profile",
            profile_path="/tmp/profiles/high_throughput.json",
            prefill_us=16000.0,
            decode_us=160.0,
            tokens_per_s=1400.0,
            decode_p95_us=240.0,
            runtime_compatible=True,
        ),
        PolicyCandidate(
            name="balanced_profile",
            profile_path="/tmp/profiles/balanced.json",
            prefill_us=13500.0,
            decode_us=110.0,
            tokens_per_s=1200.0,
            decode_p95_us=180.0,
            runtime_compatible=True,
        ),
        PolicyCandidate(
            name="offline_fine_profile",
            profile_path="/tmp/profiles/fine_profile.json",
            prefill_us=9000.0,
            decode_us=90.0,
            tokens_per_s=1500.0,
            decode_p95_us=130.0,
            runtime_compatible=False,
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone online policy selection probe")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--objective",
        choices=("latency", "throughput", "balanced"),
        default="balanced",
    )
    parser.add_argument("--decode-weight", type=float, default=32.0)
    parser.add_argument("--latency-weight", type=float, default=0.5)
    parser.add_argument("--throughput-weight", type=float, default=0.5)
    parser.add_argument("--max-decode-p95-us", type=float)
    parser.add_argument("--max-error-rate", type=float)
    parser.add_argument("--min-tokens-per-s", type=float)
    parser.add_argument("--allow-offline-profiles", action="store_true")
    args = parser.parse_args()
    assert args.demo or args.input_csv is not None, "provide --demo or --input-csv"
    return args


if __name__ == "__main__":
    main()
