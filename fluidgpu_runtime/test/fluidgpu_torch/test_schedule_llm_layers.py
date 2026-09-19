import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from fluidgpu_torch.planner import CsvTask
from fluidgpu_torch.profile import make_ranked_task_profile, profile_from_dict, validate_profile


def gurobi_solver_available() -> bool:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError:
        return False
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        try:
            model = gp.Model("license_probe", env=env)
            model.addVar(vtype=GRB.BINARY)
            model.optimize()
        finally:
            env.dispose()
    except gp.GurobiError:
        return False
    return True


def test_dp_planner_assigns_kernel_groups_by_cost():
    planner = load_planner()
    tasks = ["layer_0_attn", "layer_0_mlp", "layer_1_attn", "layer_1_mlp"]
    rank0, rank1 = synthetic_csv(planner)

    plan = planner.choose_plan(
        tasks=tasks,
        rank0=rank0,
        rank1=rank1,
        hidden_size=4096,
        dtype_bytes=2,
        prompt_len=1,
        comm_bw_gbps=1e9,
        decode_weight=1.0,
    )

    assert plan.task_assignments == (
        ("layer_0_attn", 0),
        ("layer_0_mlp", 1),
        ("layer_1_attn", 0),
        ("layer_1_mlp", 1),
    )
    profile = make_ranked_task_profile(
        model="tiny",
        task_assignments=list(plan.task_assignments),
        hidden_size=4096,
        num_kv_heads=8,
        head_dim=128,
    )
    runtime_profile = profile_from_dict(profile, model_name="tiny")
    validate_profile(runtime_profile, model_name="tiny")


def test_dp_planner_accounts_for_communication_cost():
    planner = load_planner()
    tasks = ["layer_0_attn", "layer_0_mlp", "layer_1_attn", "layer_1_mlp"]
    rank0, rank1 = synthetic_csv(planner)

    plan = planner.choose_plan(
        tasks=tasks,
        rank0=rank0,
        rank1=rank1,
        hidden_size=4096,
        dtype_bytes=2,
        prompt_len=2048,
        comm_bw_gbps=0.001,
        decode_weight=1.0,
    )

    assert plan.task_assignments == (
        ("layer_0_attn", 0),
        ("layer_0_mlp", 0),
        ("layer_1_attn", 0),
        ("layer_1_mlp", 0),
    )
    assert plan.rank_switches == 0


def test_fine_grained_task_sort_and_gate():
    planner = load_planner()
    tasks = [
        "layer_0_mlp",
        "layer_0_o_proj",
        "layer_0_qkv",
        "layer_0_sdpa",
        "layer_1_mlp",
        "layer_1_qkv",
    ]

    assert sorted(tasks, key=planner.task_sort_key) == [
        "layer_0_qkv",
        "layer_0_sdpa",
        "layer_0_o_proj",
        "layer_0_mlp",
        "layer_1_qkv",
        "layer_1_mlp",
    ]
    assert planner.is_runtime_compatible_task_set(tasks)
    assert planner.is_runtime_compatible_task_set(["layer_0_attn", "layer_0_mlp"])


def test_fine_grained_writes_runtime_profile(tmp_path):
    planner = load_planner()
    rank0_csv = tmp_path / "rank0.csv"
    rank1_csv = tmp_path / "rank1.csv"
    output = tmp_path / "nested" / "fine_plan.json"
    write_fine_csv(rank0_csv, qkv=1.0, sdpa=50.0, o_proj=1.0, mlp=50.0)
    write_fine_csv(rank1_csv, qkv=50.0, sdpa=1.0, o_proj=50.0, mlp=1.0)

    run_main(
        planner,
        "--rank0-csv",
        str(rank0_csv),
        "--rank1-csv",
        str(rank1_csv),
        "--output",
        str(output),
        "--model",
        "tiny",
        "--hidden-size",
        "8",
        "--num-kv-heads",
        "1",
        "--head-dim",
        "4",
        "--prompt-len",
        "1",
        "--comm-bw-gbps",
        "1000000",
        "--decode-weight",
        "1",
    )

    raw = json.loads(output.read_text())
    assert "fine_grained" not in raw
    assert raw["planner_config"]["runtime_compatible"] is True
    assert [task["name"] for task in raw["tasks"][1:-1]] == [
        "layer_0_qkv",
        "layer_0_sdpa",
        "layer_0_o_proj",
        "layer_0_mlp",
    ]
    assert [task["rank"] for task in raw["tasks"][1:-1]] == [0, 1, 0, 1]
    validate_profile(profile_from_dict(raw, model_name="tiny"), model_name="tiny")


def test_fine_grained_writes_runtime_profile_without_extra_flag(tmp_path):
    planner = load_planner()
    rank0_csv = tmp_path / "rank0.csv"
    rank1_csv = tmp_path / "rank1.csv"
    output = tmp_path / "fine_plan.json"
    write_fine_csv(rank0_csv, qkv=1.0, sdpa=1.0, o_proj=1.0, mlp=1.0)
    write_fine_csv(rank1_csv, qkv=1.0, sdpa=1.0, o_proj=1.0, mlp=1.0)

    run_main(
        planner,
        "--rank0-csv",
        str(rank0_csv),
        "--rank1-csv",
        str(rank1_csv),
        "--output",
        str(output),
        "--model",
        "tiny",
        "--hidden-size",
        "8",
        "--num-kv-heads",
        "1",
        "--head-dim",
        "4",
    )

    raw = json.loads(output.read_text())
    assert "fine_grained" not in raw
    validate_profile(profile_from_dict(raw, model_name="tiny"), model_name="tiny")


def test_milp_solver_writes_runtime_profile(tmp_path):
    if not gurobi_solver_available():
        pytest.skip("gurobipy or a working Gurobi license is unavailable")
    planner = load_planner()
    rank0_csv = tmp_path / "rank0.csv"
    rank1_csv = tmp_path / "rank1.csv"
    dp_output = tmp_path / "dp_plan.json"
    milp_output = tmp_path / "milp_plan.json"
    write_fine_csv(rank0_csv, qkv=1.0, sdpa=50.0, o_proj=1.0, mlp=50.0)
    write_fine_csv(rank1_csv, qkv=50.0, sdpa=1.0, o_proj=50.0, mlp=1.0)

    common_args = [
        "--rank0-csv",
        str(rank0_csv),
        "--rank1-csv",
        str(rank1_csv),
        "--model",
        "tiny",
        "--hidden-size",
        "8",
        "--num-kv-heads",
        "1",
        "--head-dim",
        "4",
        "--prompt-len",
        "1",
        "--comm-bw-gbps",
        "1000000",
        "--decode-weight",
        "1",
    ]
    run_main(planner, *common_args, "--output", str(dp_output), "--solver", "dp")
    run_main(planner, *common_args, "--output", str(milp_output), "--solver", "milp")

    dp_raw = json.loads(dp_output.read_text())
    milp_raw = json.loads(milp_output.read_text())
    assert milp_raw["planner_config"]["solver"] == "milp"
    assert milp_raw["planner_config"]["gurobi_status"] == "OPTIMAL"
    assert milp_raw["planner_config"]["runtime_compatible"] is True
    assert (
        milp_raw["predicted_cost_us"]["objective"]
        == pytest.approx(dp_raw["predicted_cost_us"]["objective"], rel=1e-6)
    )
    assert [task["rank"] for task in milp_raw["tasks"]] == [
        task["rank"] for task in dp_raw["tasks"]
    ]
    validate_profile(profile_from_dict(milp_raw, model_name="tiny"), model_name="tiny")


def synthetic_csv(planner):
    rank0 = {
        "embed": CsvTask("embed", 1, 1),
        "norm_lm_head": CsvTask("norm_lm_head", 1, 1),
        "layer_0_attn": CsvTask("layer_0_attn", 1, 10),
        "layer_0_mlp": CsvTask("layer_0_mlp", 100, 100),
        "layer_1_attn": CsvTask("layer_1_attn", 1, 10),
        "layer_1_mlp": CsvTask("layer_1_mlp", 100, 100),
    }
    rank1 = {
        "embed": CsvTask("embed", 1, 1),
        "norm_lm_head": CsvTask("norm_lm_head", 1, 1),
        "layer_0_attn": CsvTask("layer_0_attn", 100, 100),
        "layer_0_mlp": CsvTask("layer_0_mlp", 1, 10),
        "layer_1_attn": CsvTask("layer_1_attn", 100, 100),
        "layer_1_mlp": CsvTask("layer_1_mlp", 1, 10),
    }
    return rank0, rank1


def write_fine_csv(
    path: Path,
    *,
    qkv: float,
    sdpa: float,
    o_proj: float,
    mlp: float,
) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "prefill_us", "decode_us"])
        writer.writeheader()
        writer.writerow({"name": "embed", "prefill_us": "1.0", "decode_us": "1.0"})
        writer.writerow({"name": "layer_0_qkv", "prefill_us": qkv, "decode_us": qkv})
        writer.writerow({"name": "layer_0_sdpa", "prefill_us": sdpa, "decode_us": sdpa})
        writer.writerow(
            {"name": "layer_0_o_proj", "prefill_us": o_proj, "decode_us": o_proj}
        )
        writer.writerow({"name": "layer_0_mlp", "prefill_us": mlp, "decode_us": mlp})
        writer.writerow(
            {"name": "norm_lm_head", "prefill_us": "1.0", "decode_us": "1.0"}
        )


def run_main(planner, *args: str) -> None:
    old_argv = sys.argv
    sys.argv = ["schedule_llm_layers.py", *args]
    try:
        planner.main()
    finally:
        sys.argv = old_argv


def load_planner():
    path = Path(__file__).parents[2] / "examples" / "fluidgpu_torch" / "schedule_llm_layers.py"
    spec = importlib.util.spec_from_file_location("fluidgpu_schedule_llm_layers", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
