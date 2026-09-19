"""Per-phase placement: rank_for(mode) routing, KV binding, and parser guards.

The per-phase MILP path lets stateless FFN groups decode on a different rank
than they prefill on, while attention stays bound to its KV rank. This is the
runtime contract the executor and the decode CUDA-graph segmenter rely on
(rank_for("decode")); it previously had no direct test coverage.
"""
import pytest

from fluidgpu_torch.profile import owned_layer_ids, profile_from_dict


def _raw(tasks):
    return {
        "model": "test/tiny",
        "rank_count": 2,
        "hidden_size": 8,
        "num_kv_heads": 2,
        "head_dim": 4,
        "max_seq_len": 32,
        "dtype": "bfloat16",
        "tasks": tasks,
    }


_PER_PHASE_TASKS = [
    {"name": "embed", "rank": 0},
    {"name": "layer_0_attn", "rank": 0},
    {"name": "layer_0_mlp", "rank": 0, "decode_rank": 1},
    {"name": "layer_1_attn", "rank": 1},
    {"name": "layer_1_mlp", "rank": 1, "decode_rank": 0},
    {"name": "norm_lm_head", "rank": 0},
]


def test_rank_for_moves_ffn_only_in_decode_mode():
    by = profile_from_dict(_raw(_PER_PHASE_TASKS)).task_by_name
    # Stateless FFN groups move rank between phases...
    assert by["layer_0_mlp"].rank_for("prefill") == 0
    assert by["layer_0_mlp"].rank_for("decode") == 1
    assert by["layer_1_mlp"].rank_for("prefill") == 1
    assert by["layer_1_mlp"].rank_for("decode") == 0
    # ...while attention (the KV owner) is rank-bound across both phases.
    assert by["layer_0_attn"].rank_for("prefill") == by["layer_0_attn"].rank_for("decode") == 0
    assert by["layer_1_attn"].rank_for("prefill") == by["layer_1_attn"].rank_for("decode") == 1


def test_kv_ownership_ignores_decode_rank():
    # KV ownership follows the attention (prefill) rank regardless of any FFN
    # per-phase movement, so a request's KV cache never migrates mid-decode.
    prof = profile_from_dict(_raw(_PER_PHASE_TASKS))
    assert owned_layer_ids(prof, 0) == [0]
    assert owned_layer_ids(prof, 1) == [1]


def test_decode_rank_rejected_on_attention():
    with pytest.raises(AssertionError):
        profile_from_dict(_raw([
            {"name": "embed", "rank": 0},
            {"name": "layer_0_attn", "rank": 0, "decode_rank": 1},
            {"name": "layer_0_mlp", "rank": 0},
            {"name": "norm_lm_head", "rank": 0},
        ]))


def test_decode_rank_rejected_on_fine_grained_ffn():
    # Fine-grained FFN sub-groups exchange intra-group aux tensors whose peers
    # are resolved from the static prefill rank (phase-agnostic), so a
    # phase-crossing decode_rank on them would deadlock and must be rejected.
    with pytest.raises(AssertionError):
        profile_from_dict(_raw([
            {"name": "embed", "rank": 0},
            {"name": "layer_0_qkv", "rank": 0},
            {"name": "layer_0_sdpa", "rank": 0},
            {"name": "layer_0_o_proj", "rank": 0},
            {"name": "layer_0_mlp_gate_up", "rank": 0},
            {"name": "layer_0_mlp_down_proj", "rank": 0, "decode_rank": 1},
            {"name": "norm_lm_head", "rank": 0},
        ]))
