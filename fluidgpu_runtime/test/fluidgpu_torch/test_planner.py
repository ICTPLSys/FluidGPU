import csv
from pathlib import Path

from fluidgpu_torch.planner import (
    CsvTask,
    PlanConfig,
    choose_plan,
    choose_plan_with_config,
    is_runtime_compatible_task_set,
    read_csv,
    task_sort_key,
)


def test_planner_core_assigns_fast_rank_per_task():
    rank0, rank1 = synthetic_costs()

    plan = choose_plan(
        tasks=["layer_0_attn", "layer_0_mlp", "layer_1_attn", "layer_1_mlp"],
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
    assert plan.rank_counts == {0: 2, 1: 2}
    assert plan.rank_switches == 4


def test_planner_core_accounts_for_communication_cost():
    rank0, rank1 = synthetic_costs()
    config = PlanConfig(
        hidden_size=4096,
        prompt_len=2048,
        comm_bw_gbps=0.001,
        decode_weight=1.0,
    )

    plan = choose_plan_with_config(
        tasks=["layer_0_attn", "layer_0_mlp", "layer_1_attn", "layer_1_mlp"],
        rank0=rank0,
        rank1=rank1,
        config=config,
    )

    assert plan.task_assignments == (
        ("layer_0_attn", 0),
        ("layer_0_mlp", 0),
        ("layer_1_attn", 0),
        ("layer_1_mlp", 0),
    )
    assert plan.rank_switches == 0


def test_planner_task_sort_key_supports_runtime_and_fine_tasks():
    tasks = [
        "layer_0_mlp",
        "layer_0_moe_combine",
        "layer_0_moe_experts",
        "layer_0_moe_router",
        "layer_0_o_proj",
        "layer_0_qkv",
        "layer_0_sdpa",
        "layer_1_attn",
    ]

    assert sorted(tasks, key=task_sort_key) == [
        "layer_0_qkv",
        "layer_0_sdpa",
        "layer_0_o_proj",
        "layer_0_mlp",
        "layer_0_moe_router",
        "layer_0_moe_experts",
        "layer_0_moe_combine",
        "layer_1_attn",
    ]
    assert is_runtime_compatible_task_set(tasks)
    assert is_runtime_compatible_task_set(["layer_0_attn", "layer_0_mlp"])
    assert is_runtime_compatible_task_set(
        [
            "layer_0_qkv",
            "layer_0_sdpa",
            "layer_0_o_proj",
            "layer_0_mlp_gate_up",
            "layer_0_mlp_down_proj",
        ]
    )
    assert is_runtime_compatible_task_set(
        [
            "layer_0_attn",
            "layer_0_moe_router",
            "layer_0_moe_experts",
            "layer_0_moe_combine",
        ]
    )


def test_planner_read_csv(tmp_path):
    path = tmp_path / "costs.csv"
    write_cost_csv(path)

    costs = read_csv(path)

    assert costs["layer_0_attn"] == CsvTask("layer_0_attn", 3.0, 4.0)
    assert costs["norm_lm_head"].decode_us == 2.0


def synthetic_costs():
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


def write_cost_csv(path: Path) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "prefill_us", "decode_us"])
        writer.writeheader()
        writer.writerow({"name": "embed", "prefill_us": "1.0", "decode_us": "1.0"})
        writer.writerow(
            {"name": "layer_0_attn", "prefill_us": "3.0", "decode_us": "4.0"}
        )
        writer.writerow(
            {"name": "norm_lm_head", "prefill_us": "2.0", "decode_us": "2.0"}
        )


def test_prefill_comm_bw_override():
    from fluidgpu_torch.planner import PlanConfig

    base = PlanConfig(hidden_size=1024, prompt_len=512, comm_bw_gbps=1.5)
    split = PlanConfig(
        hidden_size=1024, prompt_len=512, comm_bw_gbps=1.5, prefill_comm_bw_gbps=60.0
    )
    # decode pricing is untouched by the override
    assert split.decode_transfer_us == base.decode_transfer_us
    # prefill pricing scales with the dedicated bandwidth (40x cheaper here)
    assert split.prefill_transfer_us < base.prefill_transfer_us / 39
    assert split.prefill_transfer_us > base.prefill_transfer_us / 41
