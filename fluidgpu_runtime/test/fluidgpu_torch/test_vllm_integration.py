from __future__ import annotations

import pytest
import torch

pytest.importorskip("vllm")

from fluidgpu_torch import vllm_integration as vi  # noqa: E402


def test_fluid_compilation_config_shape() -> None:
    cfg = vi.fluid_compilation_config()
    assert cfg["cudagraph_mode"] == "PIECEWISE", (
        "FULL-containing modes capture the whole decode forward, including "
        "the cross-GPU op"
    )
    ops = cfg["splitting_ops"]
    # our boundaries plus the stock attention/KV ops (the list replaces the
    # default, so the defaults must be carried along)
    assert vi.MOE_FORWARD_OP in ops
    assert vi.REMOTE_MLP_OP in ops
    assert "vllm::unified_attention" in ops
    assert "vllm::unified_kv_cache_update" in ops
    assert len(ops) == len(set(ops)), "duplicate splitting ops"


def test_remote_mlp_op_registered() -> None:
    assert hasattr(torch.ops.vllm, "fluidgpu_remote_mlp")


def test_stable_output_view_vs_fresh() -> None:
    buf = vi._StableOutput(max_tokens=16, width=8, dtype=torch.float32, device="cpu")
    small = torch.randn(4, 8)
    out = buf.stage(small, home="cpu")
    assert out.data_ptr() == buf.buffer.data_ptr(), "capture-size result must view the persistent buffer"
    assert torch.equal(out, small)
    out2 = buf.stage(torch.randn(4, 8), home="cpu")
    assert out2.data_ptr() == out.data_ptr(), "same num_tokens must reuse the same address"

    large = torch.randn(32, 8)
    out3 = buf.stage(large, home="cpu")
    assert out3.data_ptr() != buf.buffer.data_ptr(), "beyond-capture batches must not squeeze into the buffer"
    assert torch.equal(out3, large)


def test_stable_output_disabled_when_no_capture() -> None:
    buf = vi._StableOutput(max_tokens=0, width=8, dtype=torch.float32, device="cpu")
    assert buf.buffer is None
    t = torch.randn(4, 8)
    assert buf.stage(t, home="cpu") is t


def test_free_local_expert_weights() -> None:
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(torch.zeros(2, 2), requires_grad=False)
    layer.w13_bias = torch.zeros(3, 3)  # plain attr, as after marlin repack
    layer.other = torch.nn.Parameter(torch.zeros(5), requires_grad=False)
    freed = vi._free_local_expert_weights(layer)
    assert freed == 3
    assert layer.w13_weight.numel() == 0
    assert layer.w2_weight_scale.numel() == 0
    assert layer.w13_bias.numel() == 0
    assert layer.other.numel() == 5, "non-expert params must be untouched"
