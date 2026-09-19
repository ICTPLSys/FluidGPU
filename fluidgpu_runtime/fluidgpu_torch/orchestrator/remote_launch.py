from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RemoteRank:
    host: str
    rank: int
    command: str


def build_ssh_launch(rank: RemoteRank) -> str:
    return f"ssh {rank.host} 'FLUIDGPU_RANK={rank.rank} {rank.command}'"


def build_torchrun(command: str, *, nproc_per_node: int = 2) -> str:
    return f"torchrun --nproc-per-node={nproc_per_node} {command}"
