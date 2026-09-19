from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch

ATTN_COARSE_SUFFIXES = ("attn",)
ATTN_FINE_SUFFIXES = ("qkv", "sdpa", "o_proj")
DENSE_MLP_COARSE_SUFFIXES = ("mlp",)
DENSE_MLP_FINE_SUFFIXES = ("mlp_gate_up", "mlp_down_proj")
MOE_FF_SUFFIXES = ("moe_router", "moe_experts", "moe_combine")
COARSE_LAYER_SUFFIXES = ATTN_COARSE_SUFFIXES + DENSE_MLP_COARSE_SUFFIXES
MOE_LAYER_SUFFIXES = ATTN_COARSE_SUFFIXES + MOE_FF_SUFFIXES
VALID_LAYER_SUFFIXES = (
    set(ATTN_COARSE_SUFFIXES)
    | set(ATTN_FINE_SUFFIXES)
    | set(DENSE_MLP_COARSE_SUFFIXES)
    | set(DENSE_MLP_FINE_SUFFIXES)
    | set(MOE_FF_SUFFIXES)
)


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    kernel_group_id: int
    name: str
    rank: int
    # Optional per-phase placement: stateless kernel groups (mlp/moe) may run
    # on a different rank during decode. Attention groups own the KV cache and
    # must keep one rank across phases (decode_rank is rejected for them).
    decode_rank: int | None = None

    def rank_for(self, mode: str) -> int:
        if mode == "decode" and self.decode_rank is not None:
            return self.decode_rank
        return self.rank


@dataclass(frozen=True)
class RuntimeProfile:
    model: str
    rank_count: int
    task_count: int
    tasks: tuple[TaskSpec, ...]
    hidden_size: int
    num_kv_heads: int
    head_dim: int
    dtype: torch.dtype
    max_seq_len: int
    decode_weight: float | None = None
    predicted_cost_us: dict[str, float] | None = None

    @property
    def task_by_name(self) -> dict[str, TaskSpec]:
        return {task.name: task for task in self.tasks}

    @property
    def task_index_by_name(self) -> dict[str, int]:
        return {task.name: i for i, task in enumerate(self.tasks)}


def dtype_name(dtype: torch.dtype) -> str:
    if dtype is torch.bfloat16:
        return "bfloat16"
    if dtype is torch.float16:
        return "float16"
    if dtype is torch.float32:
        return "float32"
    raise AssertionError(f"unsupported dtype {dtype}")


def dtype_from_name(name: str) -> torch.dtype:
    normalized = name.replace("torch.", "")
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float16":
        return torch.float16
    if normalized == "float32":
        return torch.float32
    raise AssertionError(f"unsupported dtype {name}")


def load_profile(
    path: str | Path,
    *,
    model_name: str | None = None,
    hf_config: Any | None = None,
) -> RuntimeProfile:
    raw = json.loads(Path(path).read_text())
    return profile_from_dict(raw, model_name=model_name, hf_config=hf_config)


def profile_from_dict(
    raw: dict[str, Any],
    *,
    model_name: str | None = None,
    hf_config: Any | None = None,
) -> RuntimeProfile:
    profile = _parse_profile(raw, hf_config=hf_config)
    validate_profile(profile, model_name=model_name, hf_config=hf_config)
    return profile


def validate_profile(
    profile: RuntimeProfile,
    *,
    model_name: str | None = None,
    hf_config: Any | None = None,
) -> None:
    assert profile.rank_count >= 2, f"rank_count must be >= 2, got {profile.rank_count}"
    assert profile.task_count == len(profile.tasks), (
        f"task_count {profile.task_count} != len(tasks) {len(profile.tasks)}"
    )
    assert profile.dtype is torch.bfloat16, f"profile dtype must be bfloat16, got {profile.dtype}"
    assert profile.max_seq_len > 0, f"max_seq_len must be positive, got {profile.max_seq_len}"
    if model_name is not None:
        assert profile.model == model_name, (
            f"profile model {profile.model} != requested {model_name}"
        )

    names = [task.name for task in profile.tasks]
    assert len(names) == len(set(names)), "task names must be unique"
    assert names.count("embed") == 1, "profile must contain exactly one embed task"
    assert names.count("norm_lm_head") == 1, "profile must contain exactly one norm_lm_head task"
    for i, task in enumerate(profile.tasks):
        assert task.task_id == i, f"tasks must be sorted by task_id; index {i} has {task.task_id}"
        assert task.kernel_group_id == task.task_id, "kernel_group_id must match task_id"
        assert 0 <= task.rank < profile.rank_count, (
            f"invalid rank {task.rank} for {task.name} under rank_count={profile.rank_count}"
        )

    by_name = profile.task_by_name
    assert by_name["embed"].rank == 0, "embed must be assigned to rank 0"
    assert by_name["norm_lm_head"].rank == 0, "norm_lm_head must be assigned to rank 0"

    layer_ids = _layer_ids(names)
    assert layer_ids == list(range(len(layer_ids))), (
        f"layers must be contiguous from 0, got {layer_ids}"
    )
    for layer_id in layer_ids:
        _validate_layer_task_sequence(profile, layer_id)

    if hf_config is not None:
        assert len(layer_ids) == hf_config.num_hidden_layers, (
            f"profile layer count {len(layer_ids)} != model layer count "
            f"{hf_config.num_hidden_layers}"
        )
        assert profile.hidden_size == hf_config.hidden_size, (
            f"profile hidden_size {profile.hidden_size} != model hidden_size "
            f"{hf_config.hidden_size}"
        )
        assert profile.num_kv_heads == hf_config.num_key_value_heads, (
            "profile num_kv_heads "
            f"{profile.num_kv_heads} != model num_key_value_heads {hf_config.num_key_value_heads}"
        )
        expected_head_dim = getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        )
        assert profile.head_dim == expected_head_dim, (
            f"profile head_dim {profile.head_dim} != model head_dim {expected_head_dim}"
        )


def owned_layer_ids(profile: RuntimeProfile, rank: int) -> list[int]:
    owned: list[int] = []
    for task in profile.tasks:
        parsed = _parse_layer_task_name(task.name)
        if parsed is not None and parsed[1] in ("attn", "qkv") and task.rank == rank:
            owned.append(parsed[0])
    return owned


def make_handwritten_profile(
    *,
    model: str,
    n_layers: int,
    split_at: int,
    hidden_size: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int = 2048,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    assert 0 <= split_at <= n_layers, f"split_at must be in [0, {n_layers}], got {split_at}"
    tasks: list[dict[str, int | str]] = [
        {"task_id": 0, "kernel_group_id": 0, "name": "embed", "rank": 0}
    ]
    for i in range(n_layers):
        task_id = i + 1
        tasks.append(
            {
                "task_id": task_id,
                "kernel_group_id": task_id,
                "name": f"layer_{i}",
                "rank": 0 if i < split_at else 1,
            }
        )
    final_task_id = n_layers + 1
    tasks.append(
        {
            "task_id": final_task_id,
            "kernel_group_id": final_task_id,
            "name": "norm_lm_head",
            "rank": 0,
        }
    )
    return {
        "model": model,
        "rank_count": 2,
        "task_count": len(tasks),
        "hidden_size": hidden_size,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": dtype_name(dtype),
        "max_seq_len": max_seq_len,
        "tasks": tasks,
    }


def make_task_profile(
    *,
    model: str,
    task_names: list[str],
    split_at: int,
    hidden_size: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int = 2048,
    dtype: torch.dtype = torch.bfloat16,
) -> dict[str, Any]:
    assert 0 <= split_at <= len(task_names), (
        f"split_at must be in [0, {len(task_names)}], got {split_at}"
    )
    tasks: list[dict[str, int | str]] = [
        {"task_id": 0, "kernel_group_id": 0, "name": "embed", "rank": 0}
    ]
    for index, name in enumerate(task_names):
        task_id = index + 1
        tasks.append(
            {
                "task_id": task_id,
                "kernel_group_id": task_id,
                "name": name,
                "rank": 0 if index < split_at else 1,
            }
        )
    final_task_id = len(task_names) + 1
    tasks.append(
        {
            "task_id": final_task_id,
            "kernel_group_id": final_task_id,
            "name": "norm_lm_head",
            "rank": 0,
        }
    )
    return {
        "model": model,
        "rank_count": 2,
        "task_count": len(tasks),
        "hidden_size": hidden_size,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": dtype_name(dtype),
        "max_seq_len": max_seq_len,
        "tasks": tasks,
    }


def make_ranked_task_profile(
    *,
    model: str,
    task_assignments: list[tuple[str, int]],
    hidden_size: int,
    num_kv_heads: int,
    head_dim: int,
    max_seq_len: int = 2048,
    dtype: torch.dtype = torch.bfloat16,
    rank_count: int | None = None,
) -> dict[str, Any]:
    inferred_rank_count = max(2, max((rank for _, rank in task_assignments), default=0) + 1)
    effective_rank_count = rank_count if rank_count is not None else inferred_rank_count
    assert effective_rank_count >= 2, (
        f"rank_count must be >= 2, got {effective_rank_count}"
    )
    tasks: list[dict[str, int | str]] = [
        {"task_id": 0, "kernel_group_id": 0, "name": "embed", "rank": 0}
    ]
    for index, (name, rank) in enumerate(task_assignments):
        assert 0 <= rank < effective_rank_count, (
            f"invalid rank {rank} for {name} under rank_count={effective_rank_count}"
        )
        task_id = index + 1
        tasks.append(
            {
                "task_id": task_id,
                "kernel_group_id": task_id,
                "name": name,
                "rank": rank,
            }
        )
    final_task_id = len(task_assignments) + 1
    tasks.append(
        {
            "task_id": final_task_id,
            "kernel_group_id": final_task_id,
            "name": "norm_lm_head",
            "rank": 0,
        }
    )
    return {
        "model": model,
        "rank_count": effective_rank_count,
        "task_count": len(tasks),
        "hidden_size": hidden_size,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": dtype_name(dtype),
        "max_seq_len": max_seq_len,
        "tasks": tasks,
    }


def write_profile(path: str | Path, profile: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(profile, indent=2) + "\n")


def _parse_profile(raw: dict[str, Any], *, hf_config: Any | None) -> RuntimeProfile:
    tasks = _parse_tasks(raw["tasks"])
    hidden_size = int(raw.get("hidden_size", getattr(hf_config, "hidden_size", 0)))
    num_kv_heads = int(raw.get("num_kv_heads", getattr(hf_config, "num_key_value_heads", 0)))
    if hf_config is not None:
        default_head_dim = hf_config.hidden_size // hf_config.num_attention_heads
    else:
        default_head_dim = 0
    head_dim = int(raw.get("head_dim", default_head_dim))
    assert hidden_size > 0, "profile hidden_size is required"
    assert num_kv_heads > 0, "profile num_kv_heads is required"
    assert head_dim > 0, "profile head_dim is required"
    return RuntimeProfile(
        model=str(raw["model"]),
        rank_count=int(raw["rank_count"]),
        task_count=len(tasks),
        tasks=tasks,
        hidden_size=hidden_size,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        dtype=dtype_from_name(str(raw.get("dtype", "bfloat16"))),
        max_seq_len=int(raw.get("max_seq_len", 2048)),
        decode_weight=raw.get("decode_weight"),
        predicted_cost_us=raw.get("predicted_cost_us"),
    )


def _layer_ids(names: Iterable[str]) -> list[int]:
    ids: set[int] = set()
    for name in names:
        parsed = _parse_layer_task_name(name)
        if parsed is not None:
            ids.add(parsed[0])
    return sorted(ids)


def _validate_layer_task_sequence(profile: RuntimeProfile, layer_id: int) -> None:
    layer_tasks = [
        task
        for task in profile.tasks
        if (parsed := _parse_layer_task_name(task.name)) is not None and parsed[0] == layer_id
    ]
    suffixes = tuple(_parse_layer_task_name(task.name)[1] for task in layer_tasks)
    valid_sequences = (
        ATTN_COARSE_SUFFIXES + DENSE_MLP_COARSE_SUFFIXES,
        ATTN_FINE_SUFFIXES + DENSE_MLP_COARSE_SUFFIXES,
        ATTN_COARSE_SUFFIXES + DENSE_MLP_FINE_SUFFIXES,
        ATTN_FINE_SUFFIXES + DENSE_MLP_FINE_SUFFIXES,
        ATTN_COARSE_SUFFIXES + MOE_FF_SUFFIXES,
        ATTN_FINE_SUFFIXES + MOE_FF_SUFFIXES,
    )
    if suffixes not in valid_sequences:
        raise AssertionError(
            f"layer_{layer_id} must use a supported attention/feed-forward task sequence, "
            f"got {suffixes}"
        )
    expected = suffixes
    indexes = [profile.task_index_by_name[f"layer_{layer_id}_{suffix}"] for suffix in expected]
    assert indexes == list(range(indexes[0], indexes[0] + len(indexes))), (
        f"layer_{layer_id} tasks must be adjacent and ordered"
    )


def _parse_tasks(raw_tasks: list[dict[str, Any]]) -> tuple[TaskSpec, ...]:
    parsed: list[TaskSpec] = []
    task_id = 0
    for item in raw_tasks:
        name = str(item["name"])
        rank = int(item["rank"])
        decode_rank = item.get("decode_rank")
        if decode_rank is not None:
            decode_rank = int(decode_rank)
            layer_info = _parse_layer_task_name(name)
            suffix = layer_info[1] if layer_info is not None else name
            # Per-phase decode placement is only safe for the coarse whole-FFN
            # group ("mlp"): its hidden-state handoff flows through the
            # mode-aware _prev_rank/_next_rank path. Fine-grained FFN sub-groups
            # (gate_up/down_proj, moe router/experts/combine) also exchange
            # *intra-group* aux tensors whose peers are resolved from the static
            # prefill rank (_dense_mlp_task_ranks / _moe_task_ranks are
            # phase-agnostic). A phase-crossing decode_rank on those would route
            # the main hidden state and the aux tensors to different ranks in the
            # decode phase and silently deadlock. Reject it at load time until the
            # aux routing is made mode-aware.
            assert suffix == "mlp", (
                "decode_rank is only supported on the coarse 'mlp' feed-forward "
                "group; fine-grained FFN sub-groups have phase-agnostic intra-group "
                f"aux routing and cannot cross ranks between phases (got {name})"
            )
        layer = _parse_layer_task_name(name)
        if layer is not None and layer[1] == "layer":
            for suffix in ("attn", "mlp"):
                parsed.append(
                    TaskSpec(
                        task_id=task_id,
                        kernel_group_id=task_id,
                        name=f"layer_{layer[0]}_{suffix}",
                        rank=rank,
                        decode_rank=decode_rank if suffix == "mlp" else None,
                    )
                )
                task_id += 1
            continue
        parsed.append(
            TaskSpec(
                task_id=task_id,
                kernel_group_id=task_id,
                name=name,
                rank=rank,
                decode_rank=decode_rank,
            )
        )
        task_id += 1
    return tuple(parsed)


def _parse_layer_task_name(name: str) -> tuple[int, str] | None:
    if not name.startswith("layer_"):
        return None
    parts = name.split("_")
    assert len(parts) >= 2, f"invalid layer task name {name}"
    assert parts[1].isdigit(), f"invalid layer id in task name {name}"
    layer_id = int(parts[1])
    if len(parts) == 2:
        return layer_id, "layer"
    suffix = "_".join(parts[2:])
    assert suffix in VALID_LAYER_SUFFIXES, f"invalid layer task suffix in {name}"
    return layer_id, suffix
