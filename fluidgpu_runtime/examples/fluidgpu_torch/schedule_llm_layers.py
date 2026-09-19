from __future__ import annotations

import argparse
from pathlib import Path

import torch

from fluidgpu_torch.planner import (
    PlanConfig,
    choose_plan,
    choose_plan_with_config,
    is_runtime_compatible_task_set,
    read_csv,
    task_sort_key,
)
from fluidgpu_torch.profile import make_ranked_task_profile, write_profile


def main() -> None:
    args = parse_args()
    rank0 = read_csv(args.rank0_csv)
    rank1 = read_csv(args.rank1_csv)
    tasks = sorted(
        [name for name in rank0 if name.startswith("layer_")],
        key=task_sort_key,
    )
    assert set(tasks) == {name for name in rank1 if name.startswith("layer_")}
    runtime_compatible = is_runtime_compatible_task_set(tasks)
    assert runtime_compatible, (
        "task set is not runtime-compatible; provide attn/mlp tasks or supported "
        "qkv/sdpa/o_proj/mlp fine-grained tasks"
    )

    config = PlanConfig(
        hidden_size=args.hidden_size,
        dtype_bytes=2,
        prompt_len=args.prompt_len,
        comm_bw_gbps=args.comm_bw_gbps,
        decode_weight=args.decode_weight,
        prefill_comm_bw_gbps=args.prefill_comm_bw_gbps,
    )
    gurobi_status: str | None = None
    per_phase = None
    if args.per_phase:
        from fluidgpu_torch.milp_planner import solve_per_phase_with_gurobi

        assert args.solver == "milp", "--per-phase requires --solver milp"
        per_phase = solve_per_phase_with_gurobi(
            tasks=tasks, rank0=rank0, rank1=rank1, config=config
        )
        gurobi_status = per_phase.gurobi_status
    elif args.solver == "milp":
        from fluidgpu_torch.milp_planner import solve_with_gurobi

        solution = solve_with_gurobi(tasks=tasks, rank0=rank0, rank1=rank1, config=config)
        plan = solution.plan
        gurobi_status = solution.gurobi_status
        dp_plan = choose_plan_with_config(tasks=tasks, rank0=rank0, rank1=rank1, config=config)
        assert abs(plan.objective_us - dp_plan.objective_us) <= max(
            1e-6 * dp_plan.objective_us, 1e-3
        ), (
            "MILP objective diverges from DP optimum: "
            f"milp={plan.objective_us:.6f} dp={dp_plan.objective_us:.6f}"
        )
    else:
        plan = choose_plan_with_config(tasks=tasks, rank0=rank0, rank1=rank1, config=config)

    if per_phase is not None:
        assignments = list(zip(tasks, per_phase.prefill_ranks))
        objective_us, prefill_us, decode_us = (
            per_phase.objective_us,
            per_phase.prefill_us,
            per_phase.decode_us,
        )
        switches = sum(
            1 for a, b in zip((0, *per_phase.decode_ranks), (*per_phase.decode_ranks, 0)) if a != b
        )
    else:
        assignments = list(plan.task_assignments)
        objective_us, prefill_us, decode_us = plan.objective_us, plan.prefill_us, plan.decode_us
        switches = plan.rank_switches
    profile = make_ranked_task_profile(
        model=args.model,
        task_assignments=assignments,
        hidden_size=args.hidden_size,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        max_seq_len=args.max_seq_len,
        dtype=torch.bfloat16,
    )
    if per_phase is not None:
        # Stateless FFN groups may decode on a different rank than prefill.
        decode_by_name = dict(zip(tasks, per_phase.decode_ranks))
        for task in profile["tasks"]:
            name = task["name"]
            if name in decode_by_name and decode_by_name[name] != task["rank"]:
                task["decode_rank"] = decode_by_name[name]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    profile["decode_weight"] = args.decode_weight
    profile["predicted_cost_us"] = {
        "objective": objective_us,
        "prefill_critical_path": prefill_us,
        "decode_critical_path": decode_us,
        "rank_switches": switches,
    }
    profile["planner_config"] = {
        "prompt_len": args.prompt_len,
        "comm_bw_gbps": args.comm_bw_gbps,
        "decode_weight": args.decode_weight,
        "runtime_compatible": runtime_compatible,
        "solver": args.solver,
    }
    if args.prefill_comm_bw_gbps is not None:
        profile["planner_config"]["prefill_comm_bw_gbps"] = args.prefill_comm_bw_gbps
    if gurobi_status is not None:
        profile["planner_config"]["gurobi_status"] = gurobi_status
    if per_phase is not None:
        profile["planner_config"]["per_phase"] = True
    write_profile(args.output, profile)
    rank0_count = sum(1 for _, rank in assignments if rank == 0)
    rank1_count = len(assignments) - rank0_count
    tag = "milp-per-phase" if per_phase is not None else args.solver
    print(
        f"planned solver={tag} "
        f"tasks={len(assignments)} rank0={rank0_count} rank1={rank1_count} "
        f"switches={switches} objective_us={objective_us:.3f} "
        f"prefill_us={prefill_us:.3f} decode_us={decode_us:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a two-rank FluidGPU layer split")
    parser.add_argument("--rank0-csv", type=Path, required=True)
    parser.add_argument("--rank1-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--num-kv-heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--comm-bw-gbps", type=float, default=25.0)
    parser.add_argument(
        "--prefill-comm-bw-gbps",
        type=float,
        default=None,
        help="price prefill (MB-sized) hand-offs at this measured large-message "
        "bandwidth; --comm-bw-gbps then only prices the latency-dominated "
        "decode hand-offs (default: one bandwidth for both phases)",
    )
    parser.add_argument("--decode-weight", type=float, default=32.0)
    parser.add_argument(
        "--per-phase",
        action="store_true",
        help="jointly place prefill and decode; stateless FFN groups may decode "
        "on a different rank (requires --solver milp)",
    )
    parser.add_argument(
        "--solver",
        choices=["dp", "milp"],
        default="dp",
        help="dp (default) uses the built-in dynamic program; milp solves the same "
        "assignment with Gurobi and emits an identical runtime profile",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
