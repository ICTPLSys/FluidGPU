from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from pathlib import Path

import torch


def validate_torch_version() -> None:
    version = torch.__version__.split("+", 1)[0]
    assert version.startswith("2.10.0"), f"torch must be 2.10.0, got {torch.__version__}"


@dataclass(frozen=True)
class EngineConfig:
    model_name: str
    profiling_json: str | Path | None = None
    dtype: torch.dtype = torch.bfloat16
    max_seq_len: int = 2048
    master_addr: str = "127.0.0.1"
    master_port: int = 29500
    rank_override: int | None = None
    device_id: int = 0
    device: str | None = None
    world_size: int = 2
    preallocate_recv_buffers: bool = True
    # Capture per-rank decode segments as CUDA graphs (dense attn/mlp profiles
    # only). Collapses host launch overhead so multi-worker pipelining can
    # overlap compute with communication.
    cuda_graph_decode: bool = False
    strict_transformers_version: bool = True
    monitor_output: str | Path | None = None
    comm_transport: Literal["auto", "rdma"] = "auto"
    nccl_socket_ifname: str | None = None
    nccl_ib_hca: str | None = None
    rdma_hca_by_rank: tuple[str, ...] = ("<set-rank0-ib-hca>", "<set-rank1-ib-hca>")
    nccl_ib_gid_index: int | None = None

    def __post_init__(self) -> None:
        assert self.dtype is torch.bfloat16, f"dtype must be torch.bfloat16, got {self.dtype}"
        assert self.max_seq_len > 0, f"max_seq_len must be positive, got {self.max_seq_len}"
        assert self.world_size >= 2, f"world_size must be >= 2, got {self.world_size}"
        assert self.device_id >= 0, f"device_id must be non-negative, got {self.device_id}"
        assert self.comm_transport in ("auto", "rdma"), (
            f"comm_transport must be auto or rdma, got {self.comm_transport}"
        )
        if self.rank_override is not None:
            assert 0 <= self.rank_override < self.world_size, (
                f"rank_override must be in [0, {self.world_size}), got {self.rank_override}"
            )
        if self.comm_transport == "rdma" and self.nccl_ib_hca is None:
            assert len(self.rdma_hca_by_rank) >= self.world_size, (
                "rdma_hca_by_rank must have at least world_size entries when "
                f"comm_transport=rdma, got {self.rdma_hca_by_rank}"
            )

    @property
    def torch_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device(f"cuda:{self.device_id}")

    @property
    def profiling_path(self) -> Path:
        assert self.profiling_json is not None, "profiling_json is required"
        return Path(self.profiling_json)
