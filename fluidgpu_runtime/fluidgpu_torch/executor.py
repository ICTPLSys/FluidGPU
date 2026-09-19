from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

import torch

from . import comm
from .config import EngineConfig
from .profile import RuntimeProfile
from .worker_trace import trace_region


class LayerExecutor:
    def __init__(
        self,
        profile: RuntimeProfile,
        cfg: EngineConfig,
        comm_backend: Any = comm,
    ) -> None:
        self.profile = profile
        self.cfg = cfg
        self.tasks = list(profile.tasks)
        self.task_index_by_name = profile.task_index_by_name
        self.my_rank = comm_backend.get_rank()
        self._comm = comm_backend
        self.device = cfg.torch_device
        self._recv_buf_prefill: torch.Tensor | None = None
        self._recv_buf_decode: torch.Tensor | None = None
        if cfg.preallocate_recv_buffers:
            self._recv_buf_prefill = torch.empty(
                (1, cfg.max_seq_len, profile.hidden_size),
                dtype=cfg.dtype,
                device=self.device,
            )
            self._recv_buf_decode = torch.empty(
                (1, 1, profile.hidden_size),
                dtype=cfg.dtype,
                device=self.device,
            )

    def run_stage(
        self,
        task_name: str,
        fn: Callable[..., Any],
        value: torch.Tensor | None,
        *,
        mode: str,
        seq_len: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        return self._run(task_name, fn, value, mode=mode, seq_len=seq_len, **kwargs)

    def run_layer(
        self,
        task_name: str,
        layer: Callable[..., Any],
        hidden: torch.Tensor | None,
        *,
        mode: str,
        seq_len: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        def layer_fn(hidden_states: torch.Tensor, **layer_kwargs: Any) -> Any:
            return layer(hidden_states=hidden_states, **layer_kwargs)

        return self._run(task_name, layer_fn, hidden, mode=mode, seq_len=seq_len, **kwargs)

    def run_kernel_group(
        self,
        task_name: str,
        fn: Callable[..., Any],
        hidden: torch.Tensor | None,
        *,
        mode: str,
        seq_len: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        return self._run(task_name, fn, hidden, mode=mode, seq_len=seq_len, **kwargs)

    def send_aux_tensor(self, tensor: torch.Tensor, *, dst: int) -> None:
        self._comm.send_tensor(tensor, dst=dst)

    def recv_aux_tensor(self, tensor: torch.Tensor, *, src: int) -> None:
        self._comm.recv_tensor(tensor, src=src)

    def _run(
        self,
        task_name: str,
        fn: Callable[..., Any],
        value: torch.Tensor | None,
        *,
        mode: str,
        seq_len: int | None,
        **kwargs: Any,
    ) -> torch.Tensor | None:
        assert mode in ("prefill", "decode"), f"invalid mode {mode}"
        idx = self.task_index_by_name[task_name]
        task = self.tasks[idx]
        if task.rank_for(mode) != self.my_rank:
            return None

        prev_rank = self._prev_rank(idx, mode)
        if prev_rank is not None and prev_rank != self.my_rank:
            value = self._recv(mode=mode, seq_len=self._seq_len(value, seq_len), src=prev_rank)
        assert value is not None, f"rank {self.my_rank} entered {task_name} without input"

        with trace_region(
            kind="compute",
            name=task_name,
            device=self.device,
            mode=mode,
        ):
            out = _first_tensor(fn(value, **kwargs))
        if os.environ.get("FLUIDGPU_DEBUG_DUMP"):
            torch.save(out.detach().cpu(), f"debug/rank{self.my_rank}_{task.name}_{mode}.pt")

        next_rank = self._next_rank(idx, mode)
        if next_rank is not None and next_rank != self.my_rank:
            if not out.is_contiguous():
                out = out.contiguous()
            self._comm.send_tensor(out, dst=next_rank)
        return out

    def _recv(self, *, mode: str, seq_len: int, src: int) -> torch.Tensor:
        buf = self._recv_buffer(mode=mode, seq_len=seq_len)
        self._comm.recv_tensor(buf, src=src)
        return buf

    def _recv_buffer(self, *, mode: str, seq_len: int) -> torch.Tensor:
        assert seq_len > 0, f"seq_len must be positive, got {seq_len}"
        if mode == "decode":
            assert seq_len == 1, f"decode seq_len must be 1, got {seq_len}"
            if self._recv_buf_decode is None:
                return torch.empty(
                    (1, 1, self.profile.hidden_size),
                    dtype=self.cfg.dtype,
                    device=self.device,
                )
            return self._recv_buf_decode

        assert seq_len <= self.cfg.max_seq_len, (
            f"prefill seq_len {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}"
        )
        if self._recv_buf_prefill is None:
            return torch.empty(
                (1, seq_len, self.profile.hidden_size),
                dtype=self.cfg.dtype,
                device=self.device,
            )
        view = self._recv_buf_prefill[:, :seq_len, :]
        assert view.is_contiguous(), "recv prefill view must be contiguous"
        return view

    def _seq_len(self, value: torch.Tensor | None, seq_len: int | None) -> int:
        if value is not None:
            assert value.ndim >= 2, f"expected [batch, seq, hidden], got {tuple(value.shape)}"
            return int(value.shape[1])
        assert seq_len is not None, "seq_len is required when receiving into an empty value"
        return seq_len

    def _prev_rank(self, idx: int, mode: str) -> int | None:
        return self.tasks[idx - 1].rank_for(mode) if idx > 0 else None

    def _next_rank(self, idx: int, mode: str) -> int | None:
        return self.tasks[idx + 1].rank_for(mode) if idx < len(self.tasks) - 1 else None


def _first_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, tuple):
        assert value, "layer returned an empty tuple"
        assert isinstance(value[0], torch.Tensor), (
            f"expected first tuple item to be Tensor, got {type(value[0])}"
        )
        return value[0]
    raise TypeError(f"expected Tensor or tuple[Tensor, ...], got {type(value)}")


KernelGroupExecutor = LayerExecutor
