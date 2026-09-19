from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

from .config import EngineConfig, validate_torch_version
from .worker_trace import trace_region


@dataclass(frozen=True)
class ProcessGroupComm:
    group: dist.ProcessGroup | None = None

    def get_rank(self) -> int:
        return get_rank()

    def get_world_size(self) -> int:
        return get_world_size()

    def send_tensor(self, t: torch.Tensor, dst: int) -> None:
        assert t.is_contiguous(), "NCCL send requires a contiguous tensor"
        assert t.device.type == "cuda", f"NCCL send requires CUDA tensor, got {t.device}"
        with trace_region(
            kind="communication",
            name="send",
            device=t.device,
            peer=dst,
            bytes=_tensor_nbytes(t),
        ):
            dist.send(t, dst=dst, group=self.group)

    def recv_tensor(self, out: torch.Tensor, src: int) -> None:
        assert out.is_contiguous(), "NCCL recv requires a contiguous tensor"
        assert out.device.type == "cuda", f"NCCL recv requires CUDA tensor, got {out.device}"
        with trace_region(
            kind="communication",
            name="recv",
            device=out.device,
            peer=src,
            bytes=_tensor_nbytes(out),
        ):
            dist.recv(out, src=src, group=self.group)

    def broadcast_tensor(self, t: torch.Tensor, src: int = 0) -> None:
        assert t.is_contiguous(), "broadcast requires a contiguous tensor"
        with trace_region(
            kind="communication",
            name="broadcast",
            device=t.device,
            peer=src,
            bytes=_tensor_nbytes(t),
        ):
            dist.broadcast(t, src=src, group=self.group)

    def destroy(self) -> None:
        if self.group is not None and dist.is_available() and dist.is_initialized():
            dist.destroy_process_group(self.group)


def init_process_group(cfg: EngineConfig) -> None:
    validate_torch_version()
    assert cfg.world_size >= 2, f"world_size must be >= 2, got {cfg.world_size}"
    assert torch.cuda.is_available(), "CUDA is required for fluidgpu_torch NCCL execution"
    if dist.is_available() and dist.is_initialized():
        assert dist.get_world_size() == cfg.world_size, (
            f"existing process group world_size={dist.get_world_size()}"
        )
        return

    rank = _resolve_rank(cfg)
    torch.cuda.set_device(cfg.device_id)
    os.environ.setdefault("MASTER_ADDR", cfg.master_addr)
    os.environ.setdefault("MASTER_PORT", str(cfg.master_port))
    os.environ["WORLD_SIZE"] = str(cfg.world_size)
    os.environ["RANK"] = str(rank)
    _configure_nccl_transport(cfg, rank)

    # Long Splitwise-style runs with multiple GIL-sharing workers can starve a
    # paired p2p op past the 10-minute default; FLUIDGPU_NCCL_TIMEOUT_S widens
    # the watchdog budget without touching cluster-wide NCCL settings.
    timeout_s = int(os.environ.get("FLUIDGPU_NCCL_TIMEOUT_S", "600"))
    dist.init_process_group(
        backend="nccl",
        world_size=cfg.world_size,
        rank=rank,
        device_id=torch.device(f"cuda:{cfg.device_id}"),
        timeout=timedelta(seconds=timeout_s),
    )


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    value = os.environ.get("FLUIDGPU_RANK") or os.environ.get("RANK")
    return int(value) if value is not None else 0


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "2"))


def send_tensor(t: torch.Tensor, dst: int) -> None:
    assert t.is_contiguous(), "NCCL send requires a contiguous tensor"
    assert t.device.type == "cuda", f"NCCL send requires CUDA tensor, got {t.device}"
    with trace_region(
        kind="communication",
        name="send",
        device=t.device,
        peer=dst,
        bytes=_tensor_nbytes(t),
    ):
        dist.send(t, dst=dst)


def recv_tensor(out: torch.Tensor, src: int) -> None:
    assert out.is_contiguous(), "NCCL recv requires a contiguous tensor"
    assert out.device.type == "cuda", f"NCCL recv requires CUDA tensor, got {out.device}"
    with trace_region(
        kind="communication",
        name="recv",
        device=out.device,
        peer=src,
        bytes=_tensor_nbytes(out),
    ):
        dist.recv(out, src=src)


def broadcast_tensor(t: torch.Tensor, src: int = 0) -> None:
    assert t.is_contiguous(), "broadcast requires a contiguous tensor"
    with trace_region(
        kind="communication",
        name="broadcast",
        device=t.device,
        peer=src,
        bytes=_tensor_nbytes(t),
    ):
        dist.broadcast(t, src=src)


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def destroy() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def default_backend() -> ProcessGroupComm:
    return ProcessGroupComm()


def new_worker_backend() -> ProcessGroupComm:
    assert dist.is_available() and dist.is_initialized(), (
        "distributed process group must be initialized before creating worker groups"
    )
    # new_group does NOT inherit the default PG's timeout; apply the same
    # FLUIDGPU_NCCL_TIMEOUT_S budget so long pipelined runs are covered.
    timeout_s = int(os.environ.get("FLUIDGPU_NCCL_TIMEOUT_S", "600"))
    group = dist.new_group(
        ranks=list(range(dist.get_world_size())),
        backend="nccl",
        timeout=timedelta(seconds=timeout_s),
    )
    return ProcessGroupComm(group=group)


def _resolve_rank(cfg: EngineConfig) -> int:
    if cfg.rank_override is not None:
        rank = cfg.rank_override
    else:
        value = (
            os.environ.get("FLUIDGPU_RANK")
            or os.environ.get("LOCAL_RANK")
            or os.environ.get("RANK")
        )
        assert value is not None, "set FLUIDGPU_RANK before launching"
        rank = int(value)
    assert 0 <= rank < cfg.world_size, (
        f"FLUIDGPU_RANK must be in [0, {cfg.world_size}), got {rank}"
    )
    return rank


def _configure_nccl_transport(cfg: EngineConfig, rank: int) -> None:
    if cfg.comm_transport != "rdma":
        return

    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_SHM_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_NET"] = "IB"
    os.environ.setdefault("TORCH_NCCL_SHOW_EAGER_INIT_P2P_SERIALIZATION_WARNING", "false")

    if cfg.nccl_socket_ifname is not None:
        os.environ["NCCL_SOCKET_IFNAME"] = cfg.nccl_socket_ifname
    if cfg.nccl_ib_hca is not None:
        os.environ["NCCL_IB_HCA"] = cfg.nccl_ib_hca
    else:
        if rank >= len(cfg.rdma_hca_by_rank):
            raise AssertionError(
                "rdma_hca_by_rank does not cover the current rank; "
                f"rank={rank}, entries={cfg.rdma_hca_by_rank}"
            )
        chosen_hca = cfg.rdma_hca_by_rank[rank]
        if chosen_hca.startswith("<set-") and chosen_hca.endswith(">"):
            raise AssertionError(
                "rdma_hca_by_rank uses placeholder values; set cfg.nccl_ib_hca or "
                "cfg.rdma_hca_by_rank to machine-specific IB device names."
            )
        os.environ["NCCL_IB_HCA"] = chosen_hca
    if cfg.nccl_ib_gid_index is not None:
        os.environ["NCCL_IB_GID_INDEX"] = str(cfg.nccl_ib_gid_index)


def _tensor_nbytes(t: torch.Tensor) -> int:
    return int(t.numel() * t.element_size())
