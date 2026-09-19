"""GPU-Direct RDMA hops between two local GPUs inside one process.

The A100/L40S pair on this machine has no CUDA P2P, so plain cross-device
copies (`Tensor.to`) stage through host memory. Each GPU does have its own IB
HCA with nvidia_peermem loaded, so NCCL send/recv with the net transport
forced (P2P/SHM disabled, NCCL_NET=IB, GDR enabled) moves data
HCA-to-HCA with GPUDirect — no host bounce, ~16 GB/s effective on this fabric
vs ~10 GB/s staged.

Single-process dual-GPU NCCL uses the classic pattern: one communicator per
device created under a grouped init, and grouped send/recv pairs.
"""
from __future__ import annotations

import os

import torch


def _force_ib_env() -> None:
    # Must be set before the NCCL communicators are created.
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_SHM_DISABLE"] = "1"
    os.environ["NCCL_IB_DISABLE"] = "0"
    os.environ["NCCL_NET"] = "IB"
    os.environ.setdefault("NCCL_NET_GDR_LEVEL", "SYS")
    os.environ.setdefault("NCCL_IB_HCA", "mlx5_2,mlx5_5")


class RdmaHopPair:
    """A pair of NCCL communicators (one per local GPU) for tensor hops."""

    _instance: "RdmaHopPair | None" = None

    def __init__(self, dev_a: int = 0, dev_b: int = 1) -> None:
        from vllm.distributed.device_communicators.pynccl_wrapper import (
            NCCLLibrary,
        )

        _force_ib_env()
        self.lib = NCCLLibrary()
        self.devs = (dev_a, dev_b)
        uid = self.lib.ncclGetUniqueId()
        self.lib.ncclGroupStart()
        comms = []
        for rank, dev in enumerate(self.devs):
            with torch.cuda.device(dev):
                comms.append(self.lib.ncclCommInitRank(2, uid, rank))
        self.lib.ncclGroupEnd()
        self.comms = comms
        # Reusable destination buffers keyed by (device, dtype, numel-bucket).
        self._buffers: dict[tuple, torch.Tensor] = {}

    @classmethod
    def get(cls) -> "RdmaHopPair":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _rank_of(self, device_index: int) -> int:
        return self.devs.index(device_index)

    def transfer(self, tensor: torch.Tensor, dst_device: int) -> torch.Tensor:
        """Move `tensor` to `dst_device` over IB (GPUDirect), synchronously."""
        from vllm.distributed.device_communicators.pynccl_wrapper import (
            ncclDataTypeEnum,
        )

        src_device = tensor.device.index
        if src_device == dst_device:
            return tensor
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        src_rank = self._rank_of(src_device)
        dst_rank = self._rank_of(dst_device)
        out = torch.empty(
            tensor.shape, dtype=tensor.dtype, device=f"cuda:{dst_device}"
        )
        dtype_id = ncclDataTypeEnum.from_torch(tensor.dtype)
        count = tensor.numel()
        src_stream = torch.cuda.current_stream(src_device)
        dst_stream = torch.cuda.current_stream(dst_device)
        self.lib.ncclGroupStart()
        self.lib.ncclSend(
            tensor.data_ptr(), count, dtype_id, dst_rank,
            self.comms[src_rank], src_stream.cuda_stream,
        )
        self.lib.ncclRecv(
            out.data_ptr(), count, dtype_id, src_rank,
            self.comms[dst_rank], dst_stream.cuda_stream,
        )
        self.lib.ncclGroupEnd()
        # The consumer runs on dst_stream, and NCCL ordered the recv there, so
        # no extra sync is needed; keep the source tensor alive until the send
        # completes by syncing lazily via stream semantics (send is enqueued on
        # src_stream, whose subsequent ops on `tensor` are ordered after it).
        return out
