"""Fused top-k MoE execution for dequantized gpt-oss checkpoints.

The HF bf16 (dequantized) inference path computes ALL experts densely and
weight-sums by the routing scores (`GptOssExperts.forward`, CUDA/eval branch):
every decode token reads all 32 experts' weights instead of its routed top-4,
an ~8x memory-traffic penalty on a bandwidth-bound step. GPUs with FP4
hardware (sm >= 8.9) never hit this — the hub kernel
(`MegaBlocksMoeMLP`) replaces the forward with a fused MXFP4 top-k kernel —
so the penalty lands exactly on the ranks that dequantize (e.g. A100) and
silently handicaps them as baselines and as placement targets.

This module swaps each layer's `GptOssMLP.forward` for vLLM's `fused_experts`
triton kernel (routed top-k, bias + swigluoai supported, CUDA-graph
capturable). Measured on A100 (gpt-oss-20b shapes, 24-layer aggregate):
decode 24.3ms -> 5.3ms (4.6x), prefill@1024 287ms -> 49ms (5.9x).

Weights are re-laid-out once at patch time (HF (E, H, 2I) -> vLLM (E, 2I, H),
interleaved gate/up rows preserved) and the original parameters are released
to keep the memory footprint flat. Fine-grained MoE sub-tasks
(moe_router/experts/combine) are incompatible with the patch and fail fast.

Disable with FLUIDGPU_GPTOSS_FUSED_MOE=0.
"""
from __future__ import annotations

import os
from typing import Any

import torch


def maybe_patch_gptoss_fused_moe(model: Any) -> int:
    """Patch dequantized gpt-oss MoE layers in place.

    Returns the number of layers patched (0 when disabled or not applicable:
    non-gpt-oss models, quantized/hub-kernel experts, CPU weights).
    """
    if os.environ.get("FLUIDGPU_GPTOSS_FUSED_MOE", "1") in ("0", "false", ""):
        return 0
    if getattr(getattr(model, "config", None), "model_type", "") != "gpt_oss":
        return 0
    try:
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.config import biased_moe_quant_config
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    except Exception:
        return 0

    patched = 0
    for layer in model.model.layers:
        mlp = getattr(layer, "mlp", None)
        experts = getattr(mlp, "experts", None)
        gate_up = getattr(experts, "gate_up_proj", None)
        if not isinstance(gate_up, torch.nn.Parameter):
            continue  # quantized experts (hub kernel path) keep their forward
        if gate_up.dtype is not torch.bfloat16 or gate_up.dim() != 3:
            continue
        if gate_up.device.type != "cuda":
            continue

        w1 = gate_up.detach().permute(0, 2, 1).contiguous()
        w2 = experts.down_proj.detach().permute(0, 2, 1).contiguous()
        quant = biased_moe_quant_config(
            w1_bias=experts.gate_up_proj_bias.detach(),
            w2_bias=experts.down_proj_bias.detach(),
        )
        # Release the original layouts so the footprint stays ~flat; any
        # residual consumer of the raw parameters must fail loudly instead of
        # computing on empty tensors.
        empty = torch.nn.Parameter(
            torch.empty(0, dtype=torch.bfloat16, device=gate_up.device),
            requires_grad=False,
        )
        experts.gate_up_proj = empty
        experts.down_proj = empty
        experts.forward = _raise_fine_grained_unsupported
        mlp.forward = _make_fused_forward(
            mlp, w1, w2, quant, fused_experts, MoEActivation.SWIGLUOAI
        )
        patched += 1

    if patched:
        torch.cuda.empty_cache()
    return patched


def _raise_fine_grained_unsupported(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError(
        "GptOssExperts.forward is unavailable: the fused-MoE patch re-laid-out "
        "the expert weights. Fine-grained MoE sub-tasks are unsupported here; "
        "set FLUIDGPU_GPTOSS_FUSED_MOE=0 to restore the dense HF path."
    )


def _make_fused_forward(mlp, w1, w2, quant, fused_experts_fn, activation):
    hidden_dim = mlp.router.hidden_dim

    def forward(hidden_states: torch.Tensor):
        batch_size = hidden_states.shape[0]
        flat = hidden_states.reshape(-1, hidden_dim)
        router_scores, router_indices = mlp.router(flat)
        topk_weights = router_scores.gather(1, router_indices)
        out = fused_experts_fn(
            flat,
            w1,
            w2,
            topk_weights=topk_weights,
            topk_ids=router_indices.to(torch.int32),
            activation=activation,
            quant_config=quant,
        )
        return out.view(batch_size, -1, hidden_dim), router_scores

    return forward
