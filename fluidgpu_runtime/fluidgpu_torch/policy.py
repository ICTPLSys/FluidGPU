from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


PolicyObjective = Literal["latency", "throughput", "balanced"]


@dataclass(frozen=True)
class PolicyCandidate:
    name: str
    prefill_us: float
    decode_us: float
    tokens_per_s: float
    decode_p95_us: float | None = None
    error_rate: float = 0.0
    runtime_compatible: bool = True
    profile_path: str | None = None

    def latency_objective_us(self, decode_weight: float) -> float:
        return self.prefill_us + decode_weight * self.decode_us


@dataclass(frozen=True)
class PolicyConfig:
    objective: PolicyObjective = "balanced"
    decode_weight: float = 32.0
    latency_weight: float = 0.5
    throughput_weight: float = 0.5
    runtime_compatible_only: bool = True
    max_decode_p95_us: float | None = None
    max_error_rate: float | None = None
    min_tokens_per_s: float | None = None


@dataclass(frozen=True)
class CandidateScore:
    candidate: PolicyCandidate
    score: float
    latency_objective_us: float
    throughput_penalty: float
    reason: str


@dataclass(frozen=True)
class RejectedCandidate:
    candidate: PolicyCandidate
    reason: str


@dataclass(frozen=True)
class PolicyDecision:
    objective: PolicyObjective
    winner: CandidateScore
    considered: tuple[CandidateScore, ...]
    rejected: tuple[RejectedCandidate, ...]


def choose_policy(
    candidates: list[PolicyCandidate],
    config: PolicyConfig,
) -> PolicyDecision:
    accepted: list[PolicyCandidate] = []
    rejected: list[RejectedCandidate] = []
    for candidate in candidates:
        reason = rejection_reason(candidate, config)
        if reason is None:
            accepted.append(candidate)
        else:
            rejected.append(RejectedCandidate(candidate, reason))
    assert accepted, "no policy candidates passed constraints"

    min_latency = min(candidate.latency_objective_us(config.decode_weight) for candidate in accepted)
    max_throughput = max(candidate.tokens_per_s for candidate in accepted)
    assert min_latency > 0.0, "latency objective must be positive"
    assert max_throughput > 0.0, "tokens_per_s must be positive"

    scores = tuple(
        sorted(
            (score_candidate(candidate, config, min_latency, max_throughput) for candidate in accepted),
            key=lambda item: (item.score, item.candidate.name),
        )
    )
    return PolicyDecision(
        objective=config.objective,
        winner=scores[0],
        considered=scores,
        rejected=tuple(rejected),
    )


def score_candidate(
    candidate: PolicyCandidate,
    config: PolicyConfig,
    min_latency: float,
    max_throughput: float,
) -> CandidateScore:
    latency = candidate.latency_objective_us(config.decode_weight)
    throughput_penalty = max_throughput / candidate.tokens_per_s
    if config.objective == "latency":
        score = latency
        reason = "min_latency_objective"
    elif config.objective == "throughput":
        score = throughput_penalty
        reason = "max_tokens_per_s"
    else:
        latency_penalty = latency / min_latency
        score = config.latency_weight * latency_penalty + config.throughput_weight * throughput_penalty
        reason = "weighted_latency_throughput"
    return CandidateScore(
        candidate=candidate,
        score=score,
        latency_objective_us=latency,
        throughput_penalty=throughput_penalty,
        reason=reason,
    )


def rejection_reason(candidate: PolicyCandidate, config: PolicyConfig) -> str | None:
    if config.runtime_compatible_only and not candidate.runtime_compatible:
        return "not_runtime_compatible"
    if config.max_decode_p95_us is not None and candidate.decode_p95_us is not None:
        if candidate.decode_p95_us > config.max_decode_p95_us:
            return "decode_p95_over_limit"
    if config.max_error_rate is not None and candidate.error_rate > config.max_error_rate:
        return "error_rate_over_limit"
    if config.min_tokens_per_s is not None and candidate.tokens_per_s < config.min_tokens_per_s:
        return "tokens_per_s_under_limit"
    return None


def read_policy_candidates(path: str | Path) -> list[PolicyCandidate]:
    with Path(path).open(newline="") as f:
        rows = csv.DictReader(f)
        return [candidate_from_row(row) for row in rows]


def candidate_from_row(row: dict[str, str]) -> PolicyCandidate:
    return PolicyCandidate(
        name=row["name"],
        prefill_us=float(row["prefill_us"]),
        decode_us=float(row["decode_us"]),
        tokens_per_s=float(row["tokens_per_s"]),
        decode_p95_us=optional_float(row.get("decode_p95_us")),
        error_rate=float(row.get("error_rate") or 0.0),
        runtime_compatible=parse_bool(row.get("runtime_compatible"), default=True),
        profile_path=row.get("profile_path") or None,
    )


def write_policy_decision(path: str | Path, decision: PolicyDecision) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(policy_decision_to_dict(decision), indent=2) + "\n")


def policy_decision_to_dict(decision: PolicyDecision) -> dict[str, Any]:
    return {
        "objective": decision.objective,
        "winner": score_to_dict(decision.winner),
        "considered": [score_to_dict(score) for score in decision.considered],
        "rejected": [
            {
                "candidate": candidate_to_dict(item.candidate),
                "reason": item.reason,
            }
            for item in decision.rejected
        ],
        "policy_probe": {
            "runtime_compatible": False,
            "note": "Offline policy decision only; not connected to runtime.",
        },
    }


def score_to_dict(score: CandidateScore) -> dict[str, Any]:
    return {
        "candidate": candidate_to_dict(score.candidate),
        "score": score.score,
        "latency_objective_us": score.latency_objective_us,
        "throughput_penalty": score.throughput_penalty,
        "reason": score.reason,
    }


def candidate_to_dict(candidate: PolicyCandidate) -> dict[str, Any]:
    return {
        "name": candidate.name,
        "profile_path": candidate.profile_path,
        "prefill_us": candidate.prefill_us,
        "decode_us": candidate.decode_us,
        "tokens_per_s": candidate.tokens_per_s,
        "decode_p95_us": candidate.decode_p95_us,
        "error_rate": candidate.error_rate,
        "runtime_compatible": candidate.runtime_compatible,
    }


def optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def parse_bool(value: str | None, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "y"):
        return True
    if normalized in ("0", "false", "no", "n"):
        return False
    raise AssertionError(f"invalid bool value {value!r}")
