"""CUDA-graph decode plan: capture per-rank contiguous task segments.

The eager decode path launches hundreds of small kernels per step from
Python, so host-side launch latency (and the GIL under multi-worker
pipelining) dominates decode. This module captures each contiguous run of
this rank's kernel groups into one CUDA graph; a decode step then replays a
handful of graphs with eager NCCL send/recv at the segment boundaries.

Constraints (v1): batch 1, dense attn/mlp granularity (embed / layer_i_attn /
layer_i_mlp / norm_lm_head), static max_seq_len shapes. The KV cache must be
in static_mode so updates are index_copy_-based and reads are full-length.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import torch

# Concurrent captures from multiple worker threads on one device are unsafe;
# serialize them (replay afterwards is lock-free).
_CAPTURE_LOCK = Lock()

from .config import EngineConfig
from .profile import RuntimeProfile


_COARSE_SUFFIXES = ("attn", "mlp")


def profile_supports_graph_decode(profile: RuntimeProfile) -> bool:
    for task in profile.tasks:
        if task.name in ("embed", "norm_lm_head"):
            continue
        parts = task.name.split("_")
        if len(parts) != 3 or parts[0] != "layer" or parts[2] not in _COARSE_SUFFIXES:
            return False
    return True


@dataclass
class _Segment:
    tasks: list[str]
    recv_src: int | None
    send_dst: int | None
    graph: torch.cuda.CUDAGraph | None = None
    starts_with_embed: bool = field(init=False)
    ends_with_head: bool = field(init=False)

    def __post_init__(self) -> None:
        self.starts_with_embed = self.tasks[0] == "embed"
        self.ends_with_head = self.tasks[-1] == "norm_lm_head"


class GraphedDecodePlan:
    """Per-runner captured decode plan for one rank."""

    def __init__(
        self,
        runner: Any,
        profile: RuntimeProfile,
        cfg: EngineConfig,
        comm_backend: Any,
        my_rank: int,
    ) -> None:
        assert profile_supports_graph_decode(profile), (
            "CUDA-graph decode supports dense attn/mlp granularity only"
        )
        assert runner.kv_cache.static_mode, "KV cache must be in static_mode"
        self.runner = runner
        self.cfg = cfg
        self.comm = comm_backend
        self.my_rank = my_rank
        self.device = runner.device
        self.segments = _build_segments(profile, my_rank)
        self.has_head = any(segment.ends_with_head for segment in self.segments)

        dtype = cfg.dtype
        hidden_size = runner.hidden_size
        self.token_buf = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        self.hidden_buf = torch.zeros((1, 1, hidden_size), dtype=dtype, device=self.device)
        self.pos_buf = torch.zeros((1,), dtype=torch.long, device=self.device)
        self.logits_buf: torch.Tensor | None = None
        self._kv_arange = torch.arange(cfg.max_seq_len, device=self.device)
        self._mask_zero = torch.zeros((cfg.max_seq_len,), dtype=dtype, device=self.device)
        self._mask_neg = torch.full(
            (cfg.max_seq_len,),
            torch.finfo(dtype).min,
            dtype=dtype,
            device=self.device,
        )
        self._captured = False
        self._pool = None

    # ---- per-step state ----

    def begin_request(self, prompt_len: int, first_token: torch.Tensor) -> None:
        self.pos_buf.fill_(prompt_len)
        self.token_buf.copy_(first_token)

    def advance(self, next_token: torch.Tensor) -> None:
        self.pos_buf += 1
        self.token_buf.copy_(next_token)

    # ---- capture ----

    def capture(self) -> None:
        assert not self._captured, "plan already captured"
        if os.environ.get("FLUIDGPU_GRAPH_DEBUG_EAGER"):
            # Bisection aid: run the same closures eagerly (no capture) so a
            # parity failure separates closure semantics from replay mechanics.
            self._captured = True
            return
        with _CAPTURE_LOCK:
            self._capture_locked()

    def _capture_locked(self) -> None:
        torch.cuda.synchronize(self.device)
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for segment in self.segments:
                self._run_segment_eager(segment)
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)

        self._pool = torch.cuda.graph_pool_handle()
        for segment in self.segments:
            graph = torch.cuda.CUDAGraph()
            # thread_local: the NCCL process-group watchdog polls CUDA events
            # from another thread, which would invalidate a "global" capture.
            with torch.cuda.graph(graph, pool=self._pool, capture_error_mode="thread_local"):
                self._run_segment_eager(segment)
            segment.graph = graph
        torch.cuda.synchronize(self.device)
        self._captured = True

    # ---- execution ----

    def decode_step(self) -> torch.Tensor | None:
        assert self._captured, "capture() must run before decode_step()"
        for segment in self.segments:
            if segment.recv_src is not None:
                self.comm.recv_tensor(self.hidden_buf, src=segment.recv_src)
            if segment.graph is not None:
                segment.graph.replay()
            else:
                self._run_segment_eager(segment)
            if segment.send_dst is not None:
                self.comm.send_tensor(self.hidden_buf, dst=segment.send_dst)
        return self.logits_buf if self.has_head else None

    # ---- internals ----

    def _run_segment_eager(self, segment: _Segment) -> None:
        runner = self.runner
        position_ids = self.pos_buf.view(1, 1)
        cache_position = self.pos_buf
        mask = torch.where(
            self._kv_arange <= self.pos_buf, self._mask_zero, self._mask_neg
        ).view(1, 1, 1, -1)
        position_embeddings = runner._position_embeddings(position_ids, 1)

        hidden = (
            runner.embed_tokens(self.token_buf)
            if segment.starts_with_embed
            else self.hidden_buf
        )
        out: torch.Tensor | None = None
        for name in segment.tasks:
            if name == "embed":
                continue
            if name == "norm_lm_head":
                out = runner._norm_head(hidden)
                continue
            layer_id = int(name.split("_")[1])
            layer = runner.layers[layer_id]
            if name.endswith("_attn"):
                assert layer_id in runner.kv_cache.owned, (
                    f"attn task {name} scheduled on rank {self.my_rank} without its KV slot"
                )
                hidden = runner._run_attn(
                    hidden,
                    layer=layer,
                    attention_mask=mask,
                    position_ids=position_ids,
                    past_key_value=runner.kv_cache,
                    use_cache=True,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                )
            else:
                hidden = runner._run_mlp(hidden, layer=layer)

        if segment.ends_with_head:
            assert out is not None
            if self.logits_buf is None:
                self.logits_buf = torch.zeros_like(out)
            self.logits_buf.copy_(out)
        else:
            self.hidden_buf.copy_(hidden)


def _build_segments(profile: RuntimeProfile, my_rank: int) -> list[_Segment]:
    # Segments follow the decode-phase placement (per-phase FFN moves included).
    tasks = list(profile.tasks)
    segments: list[_Segment] = []
    index = 0
    while index < len(tasks):
        if tasks[index].rank_for("decode") != my_rank:
            index += 1
            continue
        start = index
        while index < len(tasks) and tasks[index].rank_for("decode") == my_rank:
            index += 1
        names = [task.name for task in tasks[start:index]]
        recv_src = tasks[start - 1].rank_for("decode") if start > 0 else None
        send_dst = tasks[index].rank_for("decode") if index < len(tasks) else None
        segments.append(_Segment(tasks=names, recv_src=recv_src, send_dst=send_dst))
    # Homogeneous policies place every task on one rank; the other rank has an
    # empty plan and its decode_step is a no-op.
    return segments
