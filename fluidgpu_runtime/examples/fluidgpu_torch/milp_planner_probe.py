from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from fluidgpu_torch.milp_planner import (
    build_multi_rank_milp,
    solve_multi_rank_with_gurobi,
    write_milp_json,
    write_plan_json,
)
from fluidgpu_torch.planner import CsvTask, PlanConfig, read_csv, task_sort_key


def main() -> None:
    args = parse_args()
    tasks, rank_costs = load_inputs(args)
    config = PlanConfig(
        hidden_size=args.hidden_size,
        dtype_bytes=args.dtype_bytes,
        prompt_len=args.prompt_len,
        comm_bw_gbps=args.comm_bw_gbps,
        decode_weight=args.decode_weight if args.decode_weight is not None else mode_decode_weight(args.mode),
    )

    start_ns = time.perf_counter_ns()
    model = build_multi_rank_milp(tasks=tasks, rank_costs=rank_costs, config=config)
    build_ms = (time.perf_counter_ns() - start_ns) / 1e6
    if args.output_model_json is not None:
        write_milp_json(args.output_model_json, model)

    print(
        "milp_model "
        f"tasks={len(tasks)} gpus={args.gpus} variables={len(model.binary_variables)} "
        f"constraints={len(model.constraints)} objective_constant={model.objective.constant:.3f}"
    )

    summary: dict[str, object] = {
        "kernels": len(tasks),
        "gpus": args.gpus,
        "mode": args.mode,
        "variables": len(model.binary_variables),
        "constraints": len(model.constraints),
        "objective_constant": model.objective.constant,
        "build_ms": build_ms,
    }

    if args.solve_exact:
        solve_start_ns = time.perf_counter_ns()
        solution = solve_multi_rank_with_gurobi(
            tasks=tasks,
            rank_costs=rank_costs,
            config=config,
        )
        solver_ms = (time.perf_counter_ns() - solve_start_ns) / 1e6
        if args.output_plan_json is not None:
            write_plan_json(args.output_plan_json, solution)
        plan = solution.plan
        rank_counts = plan.rank_counts
        summary.update(
            {
                "solver": solution.solver,
                "gurobi_status": solution.gurobi_status,
                "solver_ms": solver_ms,
                "objective_us": plan.objective_us,
                "prefill_us": plan.prefill_us,
                "decode_us": plan.decode_us,
                "rank_switches": plan.rank_switches,
                "rank_counts": rank_counts,
            }
        )
        rank_text = " ".join(
            f"rank{rank}={count}" for rank, count in sorted(rank_counts.items())
        )
        print(
            "milp_exact_plan "
            f"tasks={len(plan.task_assignments)} gpus={args.gpus} {rank_text} "
            f"switches={plan.rank_switches} objective_us={plan.objective_us:.3f} "
            f"prefill_us={plan.prefill_us:.3f} decode_us={plan.decode_us:.3f} "
            f"solver={solution.solver} gurobi_status={solution.gurobi_status}"
        )

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")


def load_inputs(args: argparse.Namespace) -> tuple[list[str], list[dict[str, CsvTask]]]:
    if args.kernels is not None:
        return synthetic_inputs(args.kernels, gpus=args.gpus, seed=args.seed)
    if args.demo:
        assert args.gpus == 2, "--demo currently targets the two-rank formulation only"
        return demo_inputs()

    assert args.rank0_csv is not None and args.rank1_csv is not None, (
        "provide --demo or both --rank0-csv and --rank1-csv"
    )
    assert args.gpus == 2, "--rank0-csv/--rank1-csv currently support --gpus 2 only"
    rank0 = read_csv(args.rank0_csv)
    rank1 = read_csv(args.rank1_csv)
    tasks = sorted(
        [name for name in rank0 if name.startswith("layer_")],
        key=task_sort_key,
    )
    assert set(tasks) == {name for name in rank1 if name.startswith("layer_")}
    return tasks, [rank0, rank1]


def demo_inputs() -> tuple[list[str], list[dict[str, CsvTask]]]:
    tasks = ["layer_0_attn", "layer_0_mlp", "layer_1_attn", "layer_1_mlp"]
    rank0 = {
        "embed": CsvTask("embed", 1.0, 1.0),
        "norm_lm_head": CsvTask("norm_lm_head", 1.0, 1.0),
        "layer_0_attn": CsvTask("layer_0_attn", 1.0, 1.0),
        "layer_0_mlp": CsvTask("layer_0_mlp", 10.0, 10.0),
        "layer_1_attn": CsvTask("layer_1_attn", 1.0, 1.0),
        "layer_1_mlp": CsvTask("layer_1_mlp", 10.0, 10.0),
    }
    rank1 = {
        "embed": CsvTask("embed", 1.0, 1.0),
        "norm_lm_head": CsvTask("norm_lm_head", 1.0, 1.0),
        "layer_0_attn": CsvTask("layer_0_attn", 10.0, 10.0),
        "layer_0_mlp": CsvTask("layer_0_mlp", 1.0, 1.0),
        "layer_1_attn": CsvTask("layer_1_attn", 10.0, 10.0),
        "layer_1_mlp": CsvTask("layer_1_mlp", 1.0, 1.0),
    }
    return tasks, [rank0, rank1]


def synthetic_inputs(
    kernels: int,
    *,
    gpus: int,
    seed: int,
) -> tuple[list[str], list[dict[str, CsvTask]]]:
    assert kernels > 0, f"kernels must be positive, got {kernels}"
    assert gpus >= 2, f"gpus must be >= 2, got {gpus}"
    rng = random.Random(seed)
    tasks = [f"layer_{i}_attn" for i in range(kernels)]
    rank_costs = [
        {
            "embed": CsvTask("embed", 12.0, 1.0),
            "norm_lm_head": CsvTask("norm_lm_head", 16.0, 1.5),
        }
        for _ in range(gpus)
    ]
    for index, name in enumerate(tasks):
        base = 20.0 + rng.random() * 8.0
        preferred_rank = index % gpus
        for rank in range(gpus):
            distance = min(
                (rank - preferred_rank) % gpus,
                (preferred_rank - rank) % gpus,
            )
            factor = 1.0 + 0.18 * distance
            if rank == preferred_rank:
                factor = 1.0
            prefill = base * factor
            decode = base * 0.08 * factor
            rank_costs[rank][name] = CsvTask(name, prefill, decode)
    return tasks, rank_costs


def mode_decode_weight(mode: str) -> float:
    if mode == "latency":
        return 1.0
    if mode == "throughput":
        return 64.0
    raise AssertionError(f"invalid mode {mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone MILP planner formulation probe")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--kernels", type=int)
    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--mode", choices=("throughput", "latency"), default="throughput")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rank0-csv", type=Path)
    parser.add_argument("--rank1-csv", type=Path)
    parser.add_argument("--output-model-json", type=Path)
    parser.add_argument("--output-plan-json", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--hidden-size", type=int, default=8)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--prompt-len", type=int, default=1)
    parser.add_argument("--comm-bw-gbps", type=float, default=1e9)
    parser.add_argument("--decode-weight", type=float)
    parser.add_argument(
        "--solve-exact",
        action="store_true",
        help="Solve the MILP with Gurobi (requires gurobipy + license).",
    )
    args = parser.parse_args()
    assert args.kernels is not None or args.demo or (
        args.rank0_csv is not None and args.rank1_csv is not None
    ), (
        "provide --kernels, --demo, or both --rank0-csv and --rank1-csv"
    )
    assert (
        args.output_model_json is not None
        or args.output_plan_json is not None
        or args.summary_json is not None
    ), (
        "provide --output-model-json, --output-plan-json, or --summary-json"
    )
    if args.output_plan_json is not None:
        assert args.solve_exact, "--output-plan-json requires --solve-exact"
    return args


if __name__ == "__main__":
    main()
