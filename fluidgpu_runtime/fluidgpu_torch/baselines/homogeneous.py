from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineAssignment:
    name: str
    rank: int


def homogeneous_policy(task_names: list[str] | tuple[str, ...], *, rank: int = 0) -> tuple[BaselineAssignment, ...]:
    assert rank >= 0, f"rank must be non-negative, got {rank}"
    return tuple(BaselineAssignment(name=name, rank=rank) for name in task_names)
