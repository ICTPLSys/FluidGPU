from __future__ import annotations

from fluidgpu_torch.graph_decode import _build_segments, profile_supports_graph_decode
from fluidgpu_torch.profile import make_ranked_task_profile, profile_from_dict


def make_profile(assignments: list[tuple[str, int]]):
    raw = make_ranked_task_profile(
        model="tiny",
        task_assignments=assignments,
        hidden_size=8,
        num_kv_heads=1,
        head_dim=4,
    )
    return profile_from_dict(raw, model_name="tiny")


def test_zigzag_profile_builds_alternating_segments():
    profile = make_profile(
        [
            ("layer_0_attn", 0),
            ("layer_0_mlp", 1),
            ("layer_1_attn", 0),
            ("layer_1_mlp", 1),
        ]
    )
    assert profile_supports_graph_decode(profile)

    rank0 = _build_segments(profile, 0)
    # embed+attn0 | attn1 | norm_lm_head
    assert [seg.tasks for seg in rank0] == [
        ["embed", "layer_0_attn"],
        ["layer_1_attn"],
        ["norm_lm_head"],
    ]
    assert [(seg.recv_src, seg.send_dst) for seg in rank0] == [
        (None, 1),
        (1, 1),
        (1, None),
    ]
    assert rank0[0].starts_with_embed and not rank0[0].ends_with_head
    assert rank0[-1].ends_with_head

    rank1 = _build_segments(profile, 1)
    assert [seg.tasks for seg in rank1] == [["layer_0_mlp"], ["layer_1_mlp"]]
    assert [(seg.recv_src, seg.send_dst) for seg in rank1] == [(0, 0), (0, 0)]


def test_homogeneous_profile_gives_idle_rank_empty_plan():
    profile = make_profile([("layer_0_attn", 0), ("layer_0_mlp", 0)])
    assert _build_segments(profile, 1) == []
    rank0 = _build_segments(profile, 0)
    assert len(rank0) == 1
    assert rank0[0].tasks == ["embed", "layer_0_attn", "layer_0_mlp", "norm_lm_head"]
    assert (rank0[0].recv_src, rank0[0].send_dst) == (None, None)


def test_fine_grained_profile_rejected():
    profile = make_profile(
        [
            ("layer_0_qkv", 0),
            ("layer_0_sdpa", 0),
            ("layer_0_o_proj", 0),
            ("layer_0_mlp", 1),
        ]
    )
    assert not profile_supports_graph_decode(profile)
