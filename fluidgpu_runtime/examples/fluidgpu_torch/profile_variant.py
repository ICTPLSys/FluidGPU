from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BOUNDARY_TASKS = {"embed", "norm_lm_head"}
ATTN_SUFFIXES = ("_attn", "_qkv", "_sdpa", "_o_proj")
FFN_SUFFIXES = (
    "_mlp",
    "_mlp_gate_up",
    "_mlp_down_proj",
    "_moe_router",
    "_moe_experts",
    "_moe_combine",
)


def main() -> None:
    args = parse_args()
    raw = json.loads(args.input_profile.read_text())
    variant = build_variant(raw, policy=args.policy, model=args.model)
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_text(json.dumps(variant, indent=2) + "\n")


def build_variant(raw: dict[str, Any], *, policy: str, model: str | None) -> dict[str, Any]:
    normalized = normalize_policy(policy)
    out = json.loads(json.dumps(raw))
    if model is not None:
        out["model"] = model
    out["policy_variant"] = normalized

    if normalized == "fluidgpu":
        return out

    for task in out["tasks"]:
        name = str(task["name"])
        # Per-phase decode placement belongs to the fluidgpu policy only; a
        # homo/af variant must place BOTH phases on its target rank, otherwise
        # a decode_rank inherited from a per-phase base plan silently moves the
        # variant's decode back to the other GPU.
        task.pop("decode_rank", None)
        if name in BOUNDARY_TASKS:
            task["rank"] = 0
        elif normalized in ("homogeneous-left", "homogeneous"):
            task["rank"] = 0
        elif normalized == "homogeneous-right":
            task["rank"] = 1
        elif normalized == "af":
            task["rank"] = af_rank(name)
        else:
            raise AssertionError(f"unsupported profile variant policy {policy!r}")
    return out


def normalize_policy(policy: str) -> str:
    table = {
        "fluidgpu": "fluidgpu",
        "kernel": "fluidgpu",
        "kernel-disagg": "fluidgpu",
        "homo-left": "homogeneous-left",
        "homo-l": "homogeneous-left",
        "homogeneous-left": "homogeneous-left",
        "homo-right": "homogeneous-right",
        "homo-r": "homogeneous-right",
        "homogeneous-right": "homogeneous-right",
        "homo": "homogeneous",
        "homogeneous": "homogeneous",
        "af": "af",
        "af-disagg": "af",
    }
    normalized = table.get(policy.lower())
    if normalized is None:
        raise AssertionError(f"unknown policy {policy!r}")
    return normalized


def af_rank(task_name: str) -> int:
    if task_name.endswith(ATTN_SUFFIXES):
        return 0
    if task_name.endswith(FFN_SUFFIXES):
        return 1
    raise AssertionError(f"cannot map task {task_name!r} to AF policy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an fluidgpu_torch E2E profile variant")
    parser.add_argument("--input-profile", type=Path, required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--model")
    return parser.parse_args()


if __name__ == "__main__":
    main()
