import json
from pathlib import Path

import torch

from fluidgpu_torch.profile import (
    load_profile,
    make_handwritten_profile,
    make_ranked_task_profile,
    make_task_profile,
    owned_layer_ids,
)


def test_paper_workload_profiles_load():
    repo = Path(__file__).resolve().parents[3]
    cases = {
        "Qwen/Qwen2.5-VL-7B-Instruct": repo / "fluidgpu_runtime/profiles/qwen25_vl_7b_kernel_group_split.json",
        "mistralai/Mamba-Codestral-7B-v0.1": repo / "fluidgpu_runtime/profiles/mamba_codestral_7b_layers.json",
        "stabilityai/stable-diffusion-3.5-medium": repo / "fluidgpu_runtime/profiles/sd35_medium_layers.json",
    }

    for model_name, path in cases.items():
        profile = load_profile(path, model_name=model_name)
        assert profile.model == model_name
        assert profile.task_count == len(profile.tasks)
        assert profile.rank_count == 2


def test_handwritten_profile_validates(tmp_path):
    raw = make_handwritten_profile(
        model="Qwen/Qwen2.5-1.5B",
        n_layers=4,
        split_at=2,
        hidden_size=1536,
        num_kv_heads=2,
        head_dim=128,
        max_seq_len=2048,
        dtype=torch.bfloat16,
    )
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(raw))

    profile = load_profile(path, model_name="Qwen/Qwen2.5-1.5B")

    assert profile.task_count == 10
    assert owned_layer_ids(profile, 0) == [0, 1]
    assert owned_layer_ids(profile, 1) == [2, 3]


def test_profile_rejects_non_adjacent_kernel_groups(tmp_path):
    raw = make_task_profile(
        model="m",
        task_names=["layer_0_attn", "layer_1_attn", "layer_0_mlp", "layer_1_mlp"],
        split_at=4,
        hidden_size=8,
        num_kv_heads=1,
        head_dim=4,
    )
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw))

    try:
        load_profile(path, model_name="m")
    except AssertionError as exc:
        assert "adjacent" in str(exc)
    else:
        raise AssertionError("expected profile validation to fail")


def test_profile_accepts_moe_fine_grained_tasks(tmp_path):
    raw = make_task_profile(
        model="m",
        task_names=[
            "layer_0_attn",
            "layer_0_moe_router",
            "layer_0_moe_experts",
            "layer_0_moe_combine",
        ],
        split_at=4,
        hidden_size=8,
        num_kv_heads=1,
        head_dim=4,
    )
    path = tmp_path / "moe.json"
    path.write_text(json.dumps(raw))

    profile = load_profile(path, model_name="m")

    assert profile.task_count == 6
    assert owned_layer_ids(profile, 0) == [0]


def test_profile_accepts_moe_subtasks_on_different_ranks(tmp_path):
    raw = make_task_profile(
        model="m",
        task_names=[
            "layer_0_attn",
            "layer_0_moe_router",
            "layer_0_moe_experts",
            "layer_0_moe_combine",
        ],
        split_at=4,
        hidden_size=8,
        num_kv_heads=1,
        head_dim=4,
    )
    raw["tasks"][2]["rank"] = 0
    raw["tasks"][3]["rank"] = 1
    raw["tasks"][4]["rank"] = 0
    path = tmp_path / "split_moe.json"
    path.write_text(json.dumps(raw))

    profile = load_profile(path, model_name="m")

    assert profile.task_by_name["layer_0_moe_router"].rank == 0
    assert profile.task_by_name["layer_0_moe_experts"].rank == 1
    assert profile.task_by_name["layer_0_moe_combine"].rank == 0


def test_profile_accepts_dense_attention_and_mlp_fine_tasks(tmp_path):
    raw = make_task_profile(
        model="m",
        task_names=[
            "layer_0_qkv",
            "layer_0_sdpa",
            "layer_0_o_proj",
            "layer_0_mlp_gate_up",
            "layer_0_mlp_down_proj",
        ],
        split_at=5,
        hidden_size=8,
        num_kv_heads=1,
        head_dim=4,
    )
    raw["tasks"][2]["rank"] = 1
    raw["tasks"][4]["rank"] = 1
    path = tmp_path / "dense_fine.json"
    path.write_text(json.dumps(raw))

    profile = load_profile(path, model_name="m")

    assert profile.task_count == 7
    assert profile.task_by_name["layer_0_qkv"].rank == 0
    assert profile.task_by_name["layer_0_sdpa"].rank == 1
    assert profile.task_by_name["layer_0_o_proj"].rank == 0
    assert profile.task_by_name["layer_0_mlp_gate_up"].rank == 1
    assert profile.task_by_name["layer_0_mlp_down_proj"].rank == 0
    assert owned_layer_ids(profile, 0) == [0]


def test_profile_accepts_fine_attention_with_moe_feed_forward(tmp_path):
    raw = make_task_profile(
        model="m",
        task_names=[
            "layer_0_qkv",
            "layer_0_sdpa",
            "layer_0_o_proj",
            "layer_0_moe_router",
            "layer_0_moe_experts",
            "layer_0_moe_combine",
        ],
        split_at=6,
        hidden_size=8,
        num_kv_heads=1,
        head_dim=4,
    )
    path = tmp_path / "attn_fine_moe.json"
    path.write_text(json.dumps(raw))

    profile = load_profile(path, model_name="m")

    assert profile.task_count == 8
    assert "layer_0_qkv" in profile.task_by_name
    assert "layer_0_moe_experts" in profile.task_by_name


def test_make_ranked_task_profile_infers_rank_count(tmp_path):
    raw = make_ranked_task_profile(
        model="m",
        task_assignments=[
            ("layer_0_attn", 0),
            ("layer_0_mlp", 2),
        ],
        hidden_size=8,
        num_kv_heads=1,
        head_dim=4,
    )
    path = tmp_path / "ranked.json"
    path.write_text(json.dumps(raw))

    profile = load_profile(path, model_name="m")

    assert profile.rank_count == 3
    assert profile.task_by_name["layer_0_mlp"].rank == 2
