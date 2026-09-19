from __future__ import annotations

from typing import Any


def prebuilt_profile(name: str, *, model: str | None = None) -> dict[str, Any]:
    normalized = name.lower()
    if normalized == "qwen3_235b_a22b_moe_fine_split":
        return qwen3_235b_a22b_moe_fine_split(model=model)
    raise AssertionError(f"unknown prebuilt profile {name!r}")


def qwen3_235b_a22b_moe_fine_split(*, model: str | None = None) -> dict[str, Any]:
    resolved_model = model or "Qwen/Qwen3-235B-A22B"
    num_layers = 94
    split_at = num_layers // 2
    tasks: list[dict[str, int | str]] = [
        {"task_id": 0, "kernel_group_id": 0, "name": "embed", "rank": 0}
    ]
    task_id = 1
    for layer_id in range(num_layers):
        attn_rank = 0 if layer_id < split_at else 1
        expert_rank = 1 - attn_rank
        for name, rank in (
            (f"layer_{layer_id}_attn", attn_rank),
            (f"layer_{layer_id}_moe_router", attn_rank),
            (f"layer_{layer_id}_moe_experts", expert_rank),
            (f"layer_{layer_id}_moe_combine", attn_rank),
        ):
            tasks.append(
                {
                    "task_id": task_id,
                    "kernel_group_id": task_id,
                    "name": name,
                    "rank": rank,
                }
            )
            task_id += 1
    tasks.append(
        {
            "task_id": task_id,
            "kernel_group_id": task_id,
            "name": "norm_lm_head",
            "rank": 0,
        }
    )
    return {
        "model": resolved_model,
        "rank_count": 2,
        "task_count": len(tasks),
        "hidden_size": 4096,
        "num_kv_heads": 4,
        "head_dim": 128,
        "dtype": "bfloat16",
        "max_seq_len": 40960,
        "moe_fine_grained": {
            "runtime_compatible": True,
            "source_granularity": "attn_moe_router_moe_experts_moe_combine",
            "side_tensor_transport": "router_to_experts_to_combine",
            "note": (
                "Prebuilt end-to-end profile for Qwen3-235B-A22B. "
                "Attention/router/combine stay on the preferred rank while experts are offloaded."
            ),
        },
        "tasks": tasks,
    }
