from __future__ import annotations

from fluidgpu_torch.prebuilt_profiles import prebuilt_profile


def test_qwen3_prebuilt_profile_has_expected_shape():
    profile = prebuilt_profile("qwen3_235b_a22b_moe_fine_split")

    assert profile["model"] == "Qwen/Qwen3-235B-A22B"
    assert profile["hidden_size"] == 4096
    assert profile["num_kv_heads"] == 4
    assert profile["head_dim"] == 128
    assert profile["task_count"] == 378
    assert profile["tasks"][0]["name"] == "embed"
    assert profile["tasks"][-1]["name"] == "norm_lm_head"
    assert profile["tasks"][1]["name"] == "layer_0_attn"
    assert profile["tasks"][2]["name"] == "layer_0_moe_router"
    assert profile["tasks"][3]["name"] == "layer_0_moe_experts"
    assert profile["tasks"][4]["name"] == "layer_0_moe_combine"
    assert profile["tasks"][1]["rank"] == 0
    assert profile["tasks"][3]["rank"] == 1


def test_qwen3_prebuilt_profile_flips_expert_rank_after_halfway_point():
    profile = prebuilt_profile("qwen3_235b_a22b_moe_fine_split")
    tasks = {item["name"]: item["rank"] for item in profile["tasks"]}

    assert tasks["layer_46_attn"] == 0
    assert tasks["layer_46_moe_experts"] == 1
    assert tasks["layer_47_attn"] == 1
    assert tasks["layer_47_moe_experts"] == 0
