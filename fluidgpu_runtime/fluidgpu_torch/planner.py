from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


RUNTIME_LAYER_SUFFIXES = {
    "attn",
    "qkv",
    "sdpa",
    "o_proj",
    "mlp",
    "mlp_gate_up",
    "mlp_down_proj",
    "moe_router",
    "moe_experts",
    "moe_combine",
}
LAYER_SUFFIX_ORDER = {
    "attn": 0,
    "qkv": 0,
    "sdpa": 1,
    "o_proj": 2,
    "mlp": 3,
    "mlp_gate_up": 3,
    "mlp_down_proj": 4,
    "moe_router": 3,
    "moe_experts": 4,
    "moe_combine": 5,
}


@dataclass(frozen=True)
class CsvTask:
    name: str
    prefill_us: float
    decode_us: float


@dataclass(frozen=True)
class PlanConfig:
    hidden_size: int
    dtype_bytes: int = 2
    prompt_len: int = 512
    comm_bw_gbps: float = 25.0
    decode_weight: float = 32.0
    # Decode hand-offs are ~KB-sized and latency-dominated, so comm_bw_gbps is
    # calibrated as a latency-equivalent bandwidth for them. Prefill hand-offs
    # are MB-sized and run near the fabric's line rate; pricing them at the
    # decode-calibrated bandwidth overestimates their cost ~50x and forbids
    # profitable prefill splits. Set this to the measured large-message
    # bandwidth to price the two phases separately (None keeps legacy
    # single-bandwidth behavior).
    prefill_comm_bw_gbps: float | None = None

    @property
    def prefill_transfer_us(self) -> float:
        return comm_us(
            bytes_count=self.hidden_size * self.dtype_bytes * self.prompt_len,
            bw_gbps=self.prefill_comm_bw_gbps or self.comm_bw_gbps,
        )

    @property
    def decode_transfer_us(self) -> float:
        return comm_us(
            bytes_count=self.hidden_size * self.dtype_bytes,
            bw_gbps=self.comm_bw_gbps,
        )


@dataclass(frozen=True)
class PlanState:
    objective_us: float
    prefill_us: float
    decode_us: float
    task_assignments: tuple[tuple[str, int], ...]

    @property
    def rank_switches(self) -> int:
        ranks = [rank for _, rank in self.task_assignments]
        ranks = [0, *ranks, 0]
        return sum(1 for left, right in zip(ranks, ranks[1:]) if left != right)

    @property
    def rank_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for _, rank in self.task_assignments:
            counts[rank] = counts.get(rank, 0) + 1
        return counts


def read_csv(path: Path) -> dict[str, CsvTask]:
    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        return {
            row["name"]: CsvTask(
                name=row["name"],
                prefill_us=float(row["prefill_us"]),
                decode_us=float(row["decode_us"]),
            )
            for row in rows
        }


def choose_plan(
    *,
    tasks: list[str],
    rank0: dict[str, CsvTask],
    rank1: dict[str, CsvTask],
    hidden_size: int,
    dtype_bytes: int,
    prompt_len: int,
    comm_bw_gbps: float,
    decode_weight: float,
) -> PlanState:
    config = PlanConfig(
        hidden_size=hidden_size,
        dtype_bytes=dtype_bytes,
        prompt_len=prompt_len,
        comm_bw_gbps=comm_bw_gbps,
        decode_weight=decode_weight,
    )
    return choose_plan_with_config(tasks=tasks, rank0=rank0, rank1=rank1, config=config)


def choose_plan_with_config(
    *,
    tasks: list[str],
    rank0: dict[str, CsvTask],
    rank1: dict[str, CsvTask],
    config: PlanConfig,
) -> PlanState:
    assert "embed" in rank0 and "norm_lm_head" in rank0, "rank0 CSV must include fixed tasks"
    assert "embed" in rank1 and "norm_lm_head" in rank1, "rank1 CSV must include fixed tasks"
    prefill_transfer_us = config.prefill_transfer_us
    decode_transfer_us = config.decode_transfer_us

    dp: dict[int, PlanState] = {
        0: PlanState(
            objective_us=rank0["embed"].prefill_us
            + config.decode_weight * rank0["embed"].decode_us,
            prefill_us=rank0["embed"].prefill_us,
            decode_us=rank0["embed"].decode_us,
            task_assignments=(),
        )
    }
    sources = (rank0, rank1)
    for name in tasks:
        next_dp: dict[int, PlanState] = {}
        for rank in (0, 1):
            source = sources[rank]
            assert name in source, f"{source_name(rank)} CSV missing task {name}"
            best: PlanState | None = None
            for prev_rank, state in dp.items():
                transfer_prefill = prefill_transfer_us if prev_rank != rank else 0.0
                transfer_decode = decode_transfer_us if prev_rank != rank else 0.0
                prefill = state.prefill_us + transfer_prefill + source[name].prefill_us
                decode = state.decode_us + transfer_decode + source[name].decode_us
                objective = prefill + config.decode_weight * decode
                candidate = PlanState(
                    objective_us=objective,
                    prefill_us=prefill,
                    decode_us=decode,
                    task_assignments=(*state.task_assignments, (name, rank)),
                )
                if best is None or candidate.objective_us < best.objective_us:
                    best = candidate
            assert best is not None
            next_dp[rank] = best
        dp = next_dp

    best_final: PlanState | None = None
    for prev_rank, state in dp.items():
        transfer_prefill = prefill_transfer_us if prev_rank != 0 else 0.0
        transfer_decode = decode_transfer_us if prev_rank != 0 else 0.0
        prefill = state.prefill_us + transfer_prefill + rank0["norm_lm_head"].prefill_us
        decode = state.decode_us + transfer_decode + rank0["norm_lm_head"].decode_us
        objective = prefill + config.decode_weight * decode
        candidate = PlanState(
            objective_us=objective,
            prefill_us=prefill,
            decode_us=decode,
            task_assignments=state.task_assignments,
        )
        if best_final is None or candidate.objective_us < best_final.objective_us:
            best_final = candidate
    assert best_final is not None
    return best_final


def comm_us(*, bytes_count: int, bw_gbps: float) -> float:
    assert bw_gbps > 0, f"comm_bw_gbps must be positive, got {bw_gbps}"
    return bytes_count / (bw_gbps * 1e9) * 1e6


def task_sort_key(name: str) -> tuple[int, int]:
    parts = name.split("_")
    assert len(parts) >= 3, f"invalid task name {name}"
    assert parts[0] == "layer" and parts[1].isdigit(), f"invalid task name {name}"
    suffix = "_".join(parts[2:])
    assert suffix in LAYER_SUFFIX_ORDER, f"invalid task name {name}"
    return int(parts[1]), LAYER_SUFFIX_ORDER[suffix]


def is_runtime_compatible_task_set(tasks: list[str]) -> bool:
    return all(layer_suffix(name) in RUNTIME_LAYER_SUFFIXES for name in tasks)


def layer_suffix(name: str) -> str:
    parts = name.split("_")
    assert len(parts) >= 3, f"invalid task name {name}"
    return "_".join(parts[2:])


def source_name(rank: int) -> str:
    return f"rank{rank}"
