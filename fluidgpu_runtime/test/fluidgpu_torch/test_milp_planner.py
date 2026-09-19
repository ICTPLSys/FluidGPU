import importlib.util
import json
import sys
from pathlib import Path

import pytest

from fluidgpu_torch.milp_planner import (
    build_multi_rank_milp,
    build_two_rank_milp,
    milp_to_dict,
    solve_multi_rank_with_gurobi,
    solve_with_gurobi,
    task_variable,
)
from fluidgpu_torch.planner import CsvTask, PlanConfig, choose_plan_with_config


def test_build_two_rank_milp_shape_and_objective():
    tasks, rank0, rank1 = synthetic_inputs()
    config = PlanConfig(hidden_size=8, prompt_len=1, comm_bw_gbps=1e9, decode_weight=1.0)

    model = build_two_rank_milp(tasks=tasks, rank0=rank0, rank1=rank1, config=config)

    assert len(model.binary_variables) == len(tasks) + len(tasks) + 1
    assert len(model.constraints) == 4 * (len(tasks) + 1)
    assert model.metadata["runtime_compatible"] is False
    objective = {term.variable: term.coefficient for term in model.objective.terms}
    assert objective[task_variable("layer_0_attn")] > 0.0
    assert objective[task_variable("layer_0_mlp")] < 0.0
    assert model.objective.constant > 0.0


def test_solve_with_gurobi_matches_dp_planner():
    pytest.importorskip("gurobipy")
    tasks, rank0, rank1 = synthetic_inputs()
    config = PlanConfig(hidden_size=8, prompt_len=1, comm_bw_gbps=1e9, decode_weight=1.0)

    solution = solve_with_gurobi(tasks=tasks, rank0=rank0, rank1=rank1, config=config)
    dp_plan = choose_plan_with_config(tasks=tasks, rank0=rank0, rank1=rank1, config=config)

    assert solution.solver == "gurobi"
    assert solution.gurobi_status == "OPTIMAL"
    assert solution.plan.task_assignments == dp_plan.task_assignments
    assert abs(solution.plan.objective_us - dp_plan.objective_us) < 1e-9
    assert solution.plan.rank_counts == {0: 2, 1: 2}


def test_milp_model_serializes_to_json():
    tasks, rank0, rank1 = synthetic_inputs()
    config = PlanConfig(hidden_size=8, prompt_len=1, comm_bw_gbps=1e9, decode_weight=1.0)

    raw = milp_to_dict(build_two_rank_milp(tasks=tasks, rank0=rank0, rank1=rank1, config=config))

    assert raw["metadata"]["task_count"] == 4
    assert raw["objective"]["sense"] == "min"
    json.dumps(raw)


def test_build_multi_rank_milp_shape_for_three_gpus():
    tasks, rank_costs = synthetic_inputs_multi(gpus=3)
    config = PlanConfig(hidden_size=8, prompt_len=1, comm_bw_gbps=1e9, decode_weight=1.0)

    model = build_multi_rank_milp(tasks=tasks, rank_costs=rank_costs, config=config)

    assert model.metadata["rank_count"] == 3
    assert len(model.binary_variables) == len(tasks) * 3 + (len(tasks) + 1) * 9
    assert len(model.constraints) == len(tasks) + (len(tasks) + 1) * 3 * 2


def test_solve_multi_rank_with_gurobi_returns_three_rank_solution():
    pytest.importorskip("gurobipy")
    tasks, rank_costs = synthetic_inputs_multi(gpus=3)
    config = PlanConfig(hidden_size=8, prompt_len=1, comm_bw_gbps=1e9, decode_weight=1.0)

    solution = solve_multi_rank_with_gurobi(tasks=tasks, rank_costs=rank_costs, config=config)

    assert solution.solver == "gurobi"
    assert solution.gurobi_status == "OPTIMAL"
    assert set(solution.plan.rank_counts) <= {0, 1, 2}
    assert len(solution.plan.task_assignments) == len(tasks)


def test_milp_planner_probe_demo_cli(tmp_path):
    pytest.importorskip("gurobipy")
    module = load_cli()
    model_path = tmp_path / "milp.json"
    plan_path = tmp_path / "plan.json"

    run_main(
        module,
        "--demo",
        "--solve-exact",
        "--output-model-json",
        str(model_path),
        "--output-plan-json",
        str(plan_path),
    )

    model = json.loads(model_path.read_text())
    plan = json.loads(plan_path.read_text())
    assert model["metadata"]["runtime_compatible"] is False
    assert model["metadata"]["task_count"] == 4
    assert plan["rank_counts"] == {"0": 2, "1": 2}
    assert plan["rank_switches"] == 4
    assert plan["milp_probe"]["solver"] == "gurobi"
    assert plan["milp_probe"]["gurobi_status"] == "OPTIMAL"
    assert plan["milp_probe"]["runtime_compatible"] is False


def synthetic_inputs():
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
    return tasks, rank0, rank1


def synthetic_inputs_multi(*, gpus: int):
    assert gpus >= 3
    tasks = ["layer_0_attn", "layer_1_attn", "layer_2_attn", "layer_3_attn"]
    rank_costs = [
        {
            "embed": CsvTask("embed", 1.0, 1.0),
            "norm_lm_head": CsvTask("norm_lm_head", 1.0, 1.0),
        }
        for _ in range(gpus)
    ]
    for index, name in enumerate(tasks):
        preferred = index % gpus
        for rank in range(gpus):
            factor = 1.0 if rank == preferred else 1.4 + 0.1 * abs(rank - preferred)
            rank_costs[rank][name] = CsvTask(name, factor, factor)
    return tasks, rank_costs


def load_cli():
    path = Path(__file__).parents[2] / "examples" / "fluidgpu_torch" / "milp_planner_probe.py"
    spec = importlib.util.spec_from_file_location("fluidgpu_milp_planner_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_main(module, *args: str) -> None:
    old_argv = sys.argv
    sys.argv = ["milp_planner_probe.py", *args]
    try:
        module.main()
    finally:
        sys.argv = old_argv


def test_per_phase_ties_fine_grained_tasks_to_one_rank():
    # Runtime contract (profile.py): decode_rank is only supported on the
    # coarse whole-FFN "mlp" group. The per-phase solver must not emit a
    # phase-crossing placement on attention pieces or fine-grained FFN
    # sub-groups, even when the cost matrix rewards it — the profile loader
    # would reject the plan at load time.
    pytest.importorskip("gurobipy")
    from fluidgpu_torch.milp_planner import solve_per_phase_with_gurobi

    tasks = [
        "layer_0_qkv",
        "layer_0_sdpa",
        "layer_0_o_proj",
        "layer_0_mlp_gate_up",
        "layer_0_mlp_down_proj",
        "layer_1_attn",
        "layer_1_mlp",
    ]
    names = [*tasks, "embed", "norm_lm_head"]
    # rank0 wins every prefill, rank1 wins every decode by a wide margin:
    # without the contract constraint every task would cross ranks.
    rank0 = {name: CsvTask(name, 1.0, 100.0) for name in names}
    rank1 = {name: CsvTask(name, 100.0, 1.0) for name in names}
    config = PlanConfig(
        hidden_size=8, prompt_len=1, comm_bw_gbps=1e9, decode_weight=1.0
    )

    solution = solve_per_phase_with_gurobi(
        tasks=tasks, rank0=rank0, rank1=rank1, config=config
    )

    moved = {
        task
        for task, pre, dec in zip(
            tasks, solution.prefill_ranks, solution.decode_ranks
        )
        if pre != dec
    }
    assert moved <= {"layer_1_mlp"}, f"contract-violating phase moves: {moved}"
    assert "layer_1_mlp" in moved  # coarse-mlp mobility itself still works
