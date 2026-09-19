"""FluidGPU op-level cross-GPU placement inside vLLM (feature-compatible).

This module makes the kernel-disaggregation placement compose with vLLM's
native machinery instead of bypassing it: continuous batching, paged
attention, prefix caching, torch.compile and piecewise CUDA graphs all keep
working. It is monkey-patch only — no vLLM source edits.

Why the old post-init patch cannot work with CUDA graphs
--------------------------------------------------------
With compilation enabled (vLLM default), Dynamo traces the model during the
memory-profiling forward *inside* ``LLM()`` construction and CUDA graphs are
captured right after — both before ``LLM()`` returns. gpt-oss's FusedMoE is
dispatched through the opaque custom op ``vllm::moe_forward`` whose impl
resolves the layer by name and calls ``layer.runner.forward_impl``; a patched
``experts.forward`` is never reached by the compiled graph (it is dead code),
and a patched plain-Python module forward (llama MLP) would be inline-traced
and hard-fail under ``fullgraph=True`` cross-device.

Mechanism
---------
1. ``FluidGPUWorker`` (a ``worker_cls`` subclass) applies the placement in
   ``load_model()`` — after weights load, before the first traced forward.
   This is the sanctioned hook that works identically for in-process ``LLM``,
   multiprocess engines, and ``vllm serve`` (config travels as a string).
2. gpt-oss / MoE: the eager function behind ``vllm::moe_forward`` is replaced
   per layer (``layer.runner.forward_impl``) with a remote-experts impl that
   hops activations to the expert device, runs marlin natively there, and
   hops back. The traced FX graph is unchanged — it already contains the
   opaque op node — so the same patch serves eager and compiled execution.
3. llama / dense MLP: there is no opaque boundary, so we register our own
   custom op ``vllm::fluidgpu_remote_mlp`` (with a fake impl for tracing) and
   swap each layer's ``mlp`` for a proxy module whose class-level forward
   calls it.
4. CUDA graphs: the ops above are added to ``splitting_ops`` and
   ``cudagraph_mode=PIECEWISE`` is used (see ``fluid_compilation_config``).
   vLLM then splits the FX graph at the cross-GPU boundary: on-GPU segments
   are captured/replayed as CUDA graphs, the cross-GPU op runs eagerly
   between replays — the same treatment attention gets. FULL-containing
   modes are rejected because they capture the whole forward (our op
   included) for decode batches.

Address stability: piecewise pieces bake their input addresses at capture
(vLLM's attention guarantees this by writing into a buffer allocated inside
the preceding captured piece). Our ops are return-style, so for capture-size
batches they return views of one persistent per-process buffer — same
num_tokens, same address on every replay. Within a forward, layer i's output
is consumed by the next piece before layer i+1 writes, so one shared buffer
per op kind is safe. Larger (non-captured) batches return fresh tensors.

Config is passed via environment (a ``worker_cls`` has no ctor args and env
propagates to spawned worker processes):

- ``FLUIDGPU_VLLM_PLACEMENT``: ``none`` (default) | ``af`` | ``af-bf16``
- ``FLUIDGPU_EXPERT_DEVICE``: expert-side device, default ``cuda:1``
- ``FLUIDGPU_HOP``: ``copy`` (default) | ``rdma`` | ``auto``
- ``FLUIDGPU_RDMA_MIN_BYTES``: ``auto`` threshold, default 4 MiB
- ``FLUIDGPU_FREE_LOCAL_EXPERTS``: ``1`` (default) frees the primary-GPU
  copies of rebuilt expert weights so vLLM's memory profiling reclaims them
  for KV cache.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger("fluidgpu.vllm_integration")

_MiB = 1024 * 1024


class _StageTiming:
    """Deferred-sync per-stage GPU timing (gated by FLUIDGPU_STAGE_TIMING=1).

    Records CUDA event pairs during the run WITHOUT synchronizing (so overlap
    is preserved and stage durations are the real isolated GPU-active times),
    then sums elapsed_time once at dump. Each interval's two events live on the
    SAME device (cross-device elapsed_time is invalid). Answers the AF ceiling
    question: FFN compute rate on the expert GPU is the pipeline's bottleneck
    stage, so tokens / sum(ffn_ms) is the best AF pipeline can ever reach.
    """

    _FLUSH_EVERY = 96  # ~2 steps (24 layers x 2 ubatch); sync-flush so a hard
    #                    worker kill still leaves the last totals on disk.

    def __init__(self) -> None:
        self.compute: list[tuple[Any, Any, int]] = []  # (start, stop, ntokens)
        self.hop_in: list[tuple[Any, Any, int]] = []    # (start, stop, nbytes)
        self.c_ms = 0.0
        self.c_tok = 0
        self.h_ms = 0.0
        self.h_bytes = 0
        self.enabled = _env("FLUIDGPU_STAGE_TIMING", "0") == "1"

    _instance: "_StageTiming | None" = None

    @classmethod
    def get(cls) -> "_StageTiming":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def pair(self, stream: torch.cuda.Stream) -> tuple[Any, Any]:
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        e0.record(stream)
        return e0, e1

    def maybe_flush(self) -> None:
        if len(self.compute) < self._FLUSH_EVERY:
            return
        self.dump()

    def dump(self) -> None:
        if not self.enabled or not self.compute:
            return
        torch.cuda.synchronize()
        self.c_ms += sum(a.elapsed_time(b) for a, b, _ in self.compute)
        self.c_tok += sum(n for _, _, n in self.compute)
        if self.hop_in:
            self.h_ms += sum(a.elapsed_time(b) for a, b, _ in self.hop_in)
            self.h_bytes += sum(n for _, _, n in self.hop_in)
        self.compute.clear()
        self.hop_in.clear()
        line = [
            "FLUIDSTAGE",
            f"ffn_ms={self.c_ms:.1f}",
            f"ffn_tokens={self.c_tok}",
            f"ffn_tok_s={1000.0 * self.c_tok / max(self.c_ms, 1e-9):.1f}",
        ]
        if self.h_ms:
            line.append(f"hop_in_ms={self.h_ms:.1f}")
            line.append(f"hop_in_GBs={self.h_bytes / 1e9 / (self.h_ms / 1e3 + 1e-9):.2f}")
        msg = " ".join(line)
        logger.warning(msg)
        path = _env("FLUIDGPU_STAGE_TIMING_FILE", "")
        if path:
            with open(path, "w") as fh:
                fh.write(msg + "\n")


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


# ---------------------------------------------------------------------------
# Cross-GPU hop
# ---------------------------------------------------------------------------


def _hop_mode() -> str:
    mode = _env("FLUIDGPU_HOP", "copy")
    assert mode in ("copy", "rdma", "auto"), f"invalid FLUIDGPU_HOP {mode!r}"
    return mode


def _rdma_pair() -> Any:
    """The examples-dir RdmaHopPair (single-process dual-comm NCCL over IB)."""
    from vllm_rdma_hop import RdmaHopPair  # examples dir must be on sys.path

    return RdmaHopPair.get()


def hop_tensor(tensor: torch.Tensor, dst_device: str) -> torch.Tensor:
    """Move a tensor across GPUs honoring FLUIDGPU_HOP.

    The A100/L40S pair has no CUDA P2P, so ``copy`` stages through host
    memory. ``rdma`` sends HCA-to-HCA with GPUDirect (fast for MB-sized
    prefill hops, ~100us protocol overhead for KB-sized decode hops, hence
    ``auto``).
    """
    dst = torch.device(dst_device)
    if tensor.device == dst:
        return tensor
    if _env("FLUIDGPU_NULL_HOP", "0") == "1":  # de-risk: skip transfer (garbage out)
        return torch.empty(tensor.shape, dtype=tensor.dtype, device=dst)
    mode = _hop_mode()
    if mode == "auto":
        threshold = int(_env("FLUIDGPU_RDMA_MIN_BYTES", str(4 * _MiB)))
        mode = "rdma" if tensor.numel() * tensor.element_size() >= threshold else "copy"
    if mode == "rdma":
        return _rdma_pair().transfer(tensor, dst.index)
    return tensor.to(dst)


def _move_stray_tensors(root: Any, device: str, depth: int = 0, seen: set | None = None) -> int:
    """Move raw-attribute tensors that ``Module.to()`` does not cover.

    Quant-method objects stash tensors as plain attributes (e.g. marlin
    ``workspace``); a kernel on the target device reading them cross-device
    without P2P is an illegal memory access.
    """
    if seen is None:
        seen = set()
    if depth > 4 or id(root) in seen:
        return 0
    seen.add(id(root))
    moved = 0
    if isinstance(root, torch.nn.Module) or hasattr(root, "__dict__"):
        for key, value in list(vars(root).items()):
            if key.startswith("__"):
                continue
            if isinstance(value, torch.Tensor) and value.is_cuda and str(value.device) != device:
                setattr(root, key, value.to(device))
                moved += 1
            elif isinstance(value, (list, tuple, dict)) or hasattr(value, "__dict__"):
                moved += _move_stray_tensors(value, device, depth + 1, seen)
    elif isinstance(root, (list, tuple)):
        for item in root:
            moved += _move_stray_tensors(item, device, depth + 1, seen)
    elif isinstance(root, dict):
        for item in root.values():
            moved += _move_stray_tensors(item, device, depth + 1, seen)
    return moved


# ---------------------------------------------------------------------------
# Graph-stable output staging
# ---------------------------------------------------------------------------


class _StableOutput:
    """Persistent output buffer so piecewise CUDA graphs see fixed addresses.

    The captured piece after our splitting op reads its input from whatever
    address it saw at capture time. For any num_tokens that can be a capture
    size we therefore return ``buffer[:num_tokens]`` — a view of one
    preallocated buffer — and only fall back to fresh tensors for larger
    (never-captured) batches, where the pieces run uncaptured and addresses
    do not matter.
    """

    def __init__(self, max_tokens: int, width: int, dtype: torch.dtype, device: str) -> None:
        self.max_tokens = max_tokens
        self.width = width
        self.dtype = dtype
        self.device = device
        # ONE BUFFER PER MICRO-BATCH. With vLLM DBO's shared compute stream the
        # micro-batches' stage() calls are ordered by stream FIFO, so a single
        # buffer was safe. Under per-ubatch compute streams
        # (FLUIDGPU_UBATCH_COMPUTE_PRIORITY=1) they run CONCURRENTLY and both
        # write the same rows -> torn reads, and since this buffer is exactly the
        # fixed address a piecewise CUDA graph captured, a CUDA illegal memory
        # access. ubatch 0's buffer is preallocated because it is the one capture
        # sees, and allocating during graph capture is illegal.
        self.buffer = (
            torch.zeros((max_tokens, width), dtype=dtype, device=device)
            if max_tokens > 0
            else None
        )
        self._extra: dict[int, torch.Tensor] = {}

    def _buf(self) -> torch.Tensor | None:
        from vllm.v1.worker.ubatching import dbo_current_ubatch_id

        ubid = dbo_current_ubatch_id()  # 0 when DBO is off
        if ubid == 0 or self.buffer is None:
            return self.buffer
        buf = self._extra.get(ubid)
        if buf is None:
            buf = torch.zeros(
                (self.max_tokens, self.width), dtype=self.dtype, device=self.device
            )
            self._extra[ubid] = buf
        return buf

    def stage(self, result: torch.Tensor, home: str) -> torch.Tensor:
        num_tokens = result.shape[0]
        buffer = self._buf()
        if buffer is None or num_tokens > self.max_tokens:
            return hop_tensor(result, home)
        dst = buffer[:num_tokens, : result.shape[1]]
        if _hop_mode() == "copy" or result.device == buffer.device:
            dst.copy_(result)
        else:
            dst.copy_(hop_tensor(result, home))
        return dst


# ---------------------------------------------------------------------------
# Checkpoint access + marlin rebuild (canonical copies; examples re-export)
# ---------------------------------------------------------------------------


class CheckpointReader:
    """Random access to safetensors shards by tensor name."""

    def __init__(self, model_dir: Path) -> None:
        from safetensors import safe_open

        self._open = safe_open
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            weight_map = json.loads(index_path.read_text())["weight_map"]
            self._shard_of = {k: model_dir / v for k, v in weight_map.items()}
        else:
            self._shard_of = {}
            for shard in glob.glob(str(model_dir / "*.safetensors")):
                with safe_open(shard, framework="pt") as f:
                    for key in f.keys():
                        self._shard_of[key] = Path(shard)
        assert self._shard_of, f"no safetensors found under {model_dir}"

    def get(self, name: str) -> torch.Tensor:
        shard = self._shard_of[name]
        with self._open(str(shard), framework="pt") as f:
            return f.get_tensor(name)


class MarlinExpertsOnDevice:
    """One layer's gpt-oss experts, marlin-packed natively for ``device``.

    vLLM's marlin repack is architecture-specific (an sm80-packed weight
    faults on sm89 and vice versa), so a loaded FusedMoE cannot be moved
    across GPUs. Instead the expert weights are rebuilt for the TARGET device
    from the checkpoint, running the exact vLLM pipeline (zero-pad to marlin
    tiles -> ``prepare_moe_fp4_layer_for_marlin`` -> ``fused_marlin_moe``)
    under that device.
    """

    def __init__(
        self,
        layer_idx: int,
        reader: CheckpointReader,
        device: str,
        *,
        num_experts: int = 32,
        hidden_size: int = 2880,
        intermediate_size: int = 2880,
        top_k: int = 4,
        expert_range: tuple[int, int] | None = None,
    ) -> None:
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            get_marlin_input_dtype,
        )
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
            prepare_moe_fp4_layer_for_marlin,
        )
        from vllm.scalar_type import scalar_types

        def round_up(x: int, align: int) -> int:
            return (x + align - 1) // align * align

        self.device = device
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.hidden_pad = round_up(hidden_size, 256)
        inter_pad = round_up(intermediate_size, 128)
        self.quant_type_id = scalar_types.float4_e2m1f.id
        self.input_dtype = get_marlin_input_dtype()

        prefix = f"model.layers.{layer_idx}.mlp.experts"
        gu_blocks = reader.get(f"{prefix}.gate_up_proj_blocks")
        gu_scales = reader.get(f"{prefix}.gate_up_proj_scales")
        d_blocks = reader.get(f"{prefix}.down_proj_blocks")
        d_scales = reader.get(f"{prefix}.down_proj_scales")
        e = num_experts
        assert gu_blocks.shape[0] == e and d_blocks.shape[0] == e, (
            f"unexpected expert count: {gu_blocks.shape} / {d_blocks.shape}"
        )
        gu_blocks = gu_blocks.reshape(e, 2 * intermediate_size, hidden_size // 2)
        gu_scales = gu_scales.reshape(e, 2 * intermediate_size, hidden_size // 32)
        d_blocks = d_blocks.reshape(e, hidden_size, intermediate_size // 2)
        d_scales = d_scales.reshape(e, hidden_size, intermediate_size // 32)

        # Expert subset (bandwidth-balanced decode split): physically load only
        # experts [start, end) so this GPU reads only its share of the weights.
        # global topk stays over all `num_experts`; expert_map routes global ids
        # to local slots (or -1 to skip), so the two GPUs' partial MoE outputs
        # SUM to the exact full result (each of a token's top-k experts is
        # computed on whichever GPU holds it).
        self.global_num_experts = num_experts
        self.expert_map = None
        self._bias_slice = slice(None)
        if expert_range is not None:
            start, end = expert_range
            gu_blocks = gu_blocks[start:end].contiguous()
            gu_scales = gu_scales[start:end].contiguous()
            d_blocks = d_blocks[start:end].contiguous()
            d_scales = d_scales[start:end].contiguous()
            e = end - start
            emap = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
            emap[start:end] = torch.arange(e, dtype=torch.int32, device=device)
            self.expert_map = emap
            self._bias_slice = slice(start, end)

        # Zero-pad to the marlin tile sizes exactly as FusedMoE.create_weights
        # allocates them. Zero fp4 weights x any e8m0 scale contribute zero,
        # and swigluoai (up+1)*gate*sigmoid(alpha*gate) is zero at gate=0, so
        # padded rows/cols are inert.
        fake = torch.nn.Module()
        fake.num_experts = e
        fake.hidden_size = self.hidden_pad
        fake.intermediate_size_per_partition = inter_pad
        fake.params_dtype = torch.bfloat16

        def padded(src: torch.Tensor, rows: int, cols: int) -> torch.nn.Parameter:
            out = torch.zeros(e, rows, cols, dtype=src.dtype, device=device)
            out[:, : src.shape[1], : src.shape[2]] = src.to(device)
            return torch.nn.Parameter(out, requires_grad=False)

        fake.w13_weight = padded(gu_blocks, 2 * inter_pad, self.hidden_pad // 2)
        fake.w13_weight_scale = padded(gu_scales, 2 * inter_pad, self.hidden_pad // 32)
        fake.w2_weight = padded(d_blocks, self.hidden_pad, inter_pad // 2)
        fake.w2_weight_scale = padded(d_scales, self.hidden_pad, inter_pad // 32)

        bias13 = torch.zeros(e, 2 * inter_pad, dtype=torch.bfloat16, device=device)
        bias13[:, : 2 * intermediate_size] = reader.get(
            f"{prefix}.gate_up_proj_bias"
        )[self._bias_slice].to(device, torch.bfloat16)
        bias2 = torch.zeros(e, self.hidden_pad, dtype=torch.bfloat16, device=device)
        bias2[:, :hidden_size] = reader.get(f"{prefix}.down_proj_bias")[
            self._bias_slice
        ].to(device, torch.bfloat16)
        # prepare_... also permutes biases into marlin order — they must be
        # registered on the layer BEFORE the call.
        fake.w13_bias = torch.nn.Parameter(bias13, requires_grad=False)
        fake.w2_bias = torch.nn.Parameter(bias2, requires_grad=False)

        with torch.cuda.device(device):
            prepare_moe_fp4_layer_for_marlin(fake, input_dtype=self.input_dtype)
        self.fake = fake
        # ONE MARLIN WORKSPACE PER MICRO-BATCH (see _workspace). marlin's
        # workspace is `sms * max_blocks_per_sm` ints used as inter-threadblock
        # locks, zeroed on entry and reset on exit
        # (marlin_utils.marlin_make_workspace_new). _overlapped_remote already
        # runs each micro-batch's expert chain on its OWN stream, so two marlin
        # kernels for the same layer can be in flight at once and would trample
        # each other's locks. Today they serialize on occupancy (marlin fills the
        # GPU), but that is luck, not a guarantee -- and it breaks outright under
        # per-ubatch compute streams.
        self._ws_extra: dict[int, torch.Tensor] = {}

    def _workspace(self) -> torch.Tensor:
        """This micro-batch's private marlin lock array."""
        from vllm.v1.worker.ubatching import dbo_current_ubatch_id

        ubid = dbo_current_ubatch_id()  # 0 when DBO is off
        if ubid == 0:
            return self.fake.workspace
        ws = self._ws_extra.get(ubid)
        if ws is None:
            ws = torch.zeros_like(self.fake.workspace)
            self._ws_extra[ubid] = ws
        return ws

    def forward_on_device(
        self, x: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        """Route + run experts on ``self.device``; returns [T, x.shape[-1]]."""
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.fused_marlin_moe import (
            fused_marlin_moe,
        )

        in_width = x.shape[-1]
        top_v, top_i = torch.topk(router_logits, self.top_k, dim=-1)
        # softmax over the selected logits == softmax-all + renormalize.
        top_w = torch.softmax(top_v.to(torch.float32), dim=-1)
        if in_width < self.hidden_pad:
            x = torch.nn.functional.pad(x, (0, self.hidden_pad - in_width))
        with torch.cuda.device(self.device):
            out = fused_marlin_moe(
                hidden_states=x,
                w1=self.fake.w13_weight,
                w2=self.fake.w2_weight,
                bias1=self.fake.w13_bias,
                bias2=self.fake.w2_bias,
                w1_scale=self.fake.w13_weight_scale,
                w2_scale=self.fake.w2_weight_scale,
                topk_weights=top_w,
                topk_ids=top_i.to(torch.int32),
                quant_type_id=self.quant_type_id,
                global_num_experts=self.global_num_experts,
                expert_map=self.expert_map,
                activation=MoEActivation.SWIGLUOAI,
                workspace=self._workspace(),
                input_dtype=self.input_dtype,
            )
        return out[:, :in_width].contiguous() if out.shape[-1] != in_width else out

    # Backward-compatible entry point for the legacy eager driver.
    def forward(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor, home: str
    ) -> torch.Tensor:
        x = hop_tensor(hidden_states, self.device)
        logits = hop_tensor(router_logits, self.device)
        out = self.forward_on_device(x, logits)
        return hop_tensor(out, home)


class DequantExpertsOnDevice:
    """bf16 fallback: dequantized MXFP4 experts + triton fused_experts."""

    def __init__(
        self, layer_idx: int, reader: CheckpointReader, device: str, *, top_k: int = 4
    ) -> None:
        from transformers.integrations.mxfp4 import convert_moe_packed_tensors

        prefix = f"model.layers.{layer_idx}.mlp.experts"
        gate_up = convert_moe_packed_tensors(
            reader.get(f"{prefix}.gate_up_proj_blocks").to(device),
            reader.get(f"{prefix}.gate_up_proj_scales").to(device),
        )  # (E, H, 2I), interleaved gate/up — the HF Parameter layout
        down = convert_moe_packed_tensors(
            reader.get(f"{prefix}.down_proj_blocks").to(device),
            reader.get(f"{prefix}.down_proj_scales").to(device),
        )  # (E, I, H)
        self.w1 = gate_up.permute(0, 2, 1).contiguous()
        del gate_up
        self.w2 = down.permute(0, 2, 1).contiguous()
        del down
        from vllm.model_executor.layers.fused_moe.config import biased_moe_quant_config

        self.quant = biased_moe_quant_config(
            w1_bias=reader.get(f"{prefix}.gate_up_proj_bias").to(device, torch.bfloat16),
            w2_bias=reader.get(f"{prefix}.down_proj_bias").to(device, torch.bfloat16),
        )
        self.device = device
        self.top_k = top_k

    def forward_on_device(
        self, x: torch.Tensor, router_logits: torch.Tensor
    ) -> torch.Tensor:
        from vllm.model_executor.layers.fused_moe.activation import MoEActivation
        from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

        in_width = x.shape[-1]
        hidden = self.w1.shape[-1]
        if in_width > hidden:
            x = x[:, :hidden]
        top_v, top_i = torch.topk(router_logits, self.top_k, dim=-1)
        top_w = torch.softmax(top_v, dim=-1)
        with torch.cuda.device(self.device):
            out = fused_experts(
                x,
                self.w1,
                self.w2,
                topk_weights=top_w,
                topk_ids=top_i.to(torch.int32),
                activation=MoEActivation.SWIGLUOAI,
                quant_config=self.quant,
            )
        if out.shape[-1] < in_width:
            out = torch.nn.functional.pad(out, (0, in_width - out.shape[-1]))
        return out


# ---------------------------------------------------------------------------
# Ping-pong micro-batch pipelining (paper FG: MegaScale-Infer style)
# ---------------------------------------------------------------------------
#
# vLLM 0.18 ships a dual-batch-overlap (DBO) engine: the step's batch is split
# into two micro-batches, each running the full forward on its own host thread
# with a cooperative yield discipline (vllm/v1/worker/ubatching.py) — exactly
# the paper's ping-pong (attention of ubatch B overlaps FFN/experts of ubatch
# A). Stock activation is gated on data_parallel_size > 1 + a DeepEP all2all
# backend; at DP=1 `enable_dbo` is inert. We reuse the whole engine with three
# surgical patches (worker-process only, no vLLM source edits):
#
# 1. flip `parallel_config.enable_dbo` BEFORE Worker.load_model() so the stock
#    init installs the UBatchWrapper and per-ubatch attention-metadata
#    builders (`use_ubatching`/`num_ubatches` are live properties);
# 2. wrap `runner._determine_batch_execution_and_padding` to compute
#    `should_ubatch` at DP=1 (the stock method only asks the DP coordinator);
# 3. wrap `UBatchWrapper.__call__` for the DP=1 case: the stock path builds
#    DPMetadata (whose .make() asserts dp>1); ours goes straight to
#    `_make_ubatch_metadata(dp_metadata=[None,..])` + `_run_ubatches`.
#
# The remote-experts impl then fires its cross-GPU chain asynchronously on
# dedicated streams and calls `dbo_yield()` — the sibling micro-batch's
# attention runs on the primary GPU while the expert GPU computes. Eager-only
# for now (piecewise-cudagraph threads sharing capture state is untested), so
# drive FG rows with --enforce-eager.


def pingpong_enabled() -> bool:
    return _env("FLUIDGPU_PINGPONG", "0") == "1"


def _patch_ubatch_wrapper_class() -> None:
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context
    from vllm.v1.worker.gpu_ubatch_wrapper import UBatchWrapper

    if getattr(UBatchWrapper, "_fluidgpu_dp1_patch", False):
        return

    orig_call = UBatchWrapper.__call__

    def patched_call(self: Any, *args: Any, **kwargs: Any) -> Any:
        fc = get_forward_context()
        if (
            fc.ubatch_slices is None
            or self.vllm_config.parallel_config.data_parallel_size > 1
        ):
            return orig_call(self, *args, **kwargs)
        # DP=1 micro-batch path (eager): stock code would build DPMetadata,
        # whose make() asserts dp_size > 1. Everything else is DP-agnostic.
        ubatch_metadata = self._make_ubatch_metadata(
            ubatch_slices=fc.ubatch_slices,
            attn_metadata=fc.attn_metadata,
            slot_mapping=fc.slot_mapping,
            input_ids=kwargs["input_ids"],
            positions=kwargs["positions"],
            inputs_embeds=kwargs["inputs_embeds"],
            intermediate_tensors=kwargs["intermediate_tensors"],
            compute_stream=torch.cuda.current_stream(),
            dp_metadata=[None] * len(fc.ubatch_slices),
            batch_descriptor=fc.batch_descriptor,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )
        return self._run_ubatches(ubatch_metadata, self.model)

    UBatchWrapper.__call__ = patched_call
    UBatchWrapper._fluidgpu_dp1_patch = True


def _patch_ubatch_priority_streams() -> None:
    """Give each micro-batch its OWN priority-staggered compute stream.

    vLLM's DBO shares ONE compute_stream across all micro-batches (see
    ubatching.make_ubatch_contexts: the same stream object is handed to every
    context), relying on that stream's FIFO order for implicit mutual exclusion
    while the threads alternate on it via CPU-event handoff. Per-micro-batch
    compute streams turn that alternation into TRUE concurrency on the home GPU.

    Doing so previously died with a CUDA illegal memory access under large
    prefill. Root-caused (2026-07-15) to two buffers that the shared stream had
    been silently serializing, both now per-micro-batch:
      * `_StableOutput` staged one output buffer for ALL ubatches — and that
        buffer is the fixed address a piecewise CUDA graph captured.
      * `MarlinExpertsOnDevice` passed one marlin `workspace` (inter-threadblock
        locks, reset on kernel exit) for ALL ubatches.
    `_RemoteStreams`, `_PinnedStage` keys and the attention-metadata builder were
    already per-ubatch. Still OFF by default pending an A/B + parity check.
    """
    if _env("FLUIDGPU_UBATCH_COMPUTE_PRIORITY", "0") != "1":
        return
    import vllm.v1.worker.gpu_ubatch_wrapper as guw

    if getattr(guw, "_fluidgpu_prio_streams", False):
        return
    orig = guw.make_ubatch_contexts
    pool: dict[tuple[str, int], torch.cuda.Stream] = {}

    def patched(num_micro_batches, compute_stream, comm_stream, *a, **kw):
        ctxs = orig(num_micro_batches, compute_stream, comm_stream, *a, **kw)
        dev = compute_stream.device
        for i, ctx in enumerate(ctxs):
            key = (str(dev), i)
            if key not in pool:
                pool[key] = torch.cuda.Stream(device=dev, priority=_ubid_priority(i))
            ctx.compute_stream = pool[key]
            ctx.current_stream = pool[key]
        return ctxs

    guw.make_ubatch_contexts = patched
    guw._fluidgpu_prio_streams = True


def _patch_attn_metadata_cache() -> None:
    """Make FlashAttention rebuild metadata for every micro-batch.

    gpu_model_runner._build_attention_metadata dedupes builds across hybrid
    KV-cache groups via a (kv_cache_spec, builder_type) cache, but the
    micro-batch loop shares that cache across ubatch ids: for ubid > 0 a hit
    returns ubatch 0's built metadata with only block_table/slot_mapping
    swapped in through update_block_table(), pairing ubatch 0's
    query_start_loc/seq_lens with ubatch 1's block table. FlashAttention is
    the only GPU backend with supports_update_block_table=True, so it crashes
    ("block_table must have shape (batch_size, ...)") whenever the two
    micro-batches hold different request counts — and would mis-attend
    silently when they match. Fresh builds keep each micro-batch
    self-consistent; the skipped dedup only saves work on hybrid-group
    models, whose builders never supported the fast path anyway.
    """
    try:
        from vllm.v1.attention.backends.flash_attn import (
            FlashAttentionMetadataBuilder,
        )
    except ImportError:
        return
    FlashAttentionMetadataBuilder.supports_update_block_table = False


# Set True once _patch_phase_split_ubatch() installs (env is read at load_model
# time, after the driver has exported it); the MoE decode branch reads this
# global to decide whether to dbo_yield() per layer.
_PHASE_SPLIT = False


def phase_split_enabled() -> bool:
    return _env("FLUIDGPU_PHASE_SPLIT_UBATCH", "0") == "1"


def _patch_phase_split_ubatch() -> None:
    """Phase-split DBO (lever E): split a MIXED prefill+decode step into a
    DECODE micro-batch (ubatch 0 — FFN local on the home A100) and a PREFILL
    micro-batch (ubatch 1 — FFN remote on the expert L40S). DBO then overlaps
    the decode compute (A100) with the prefill FFN + hop (L40S), so BOTH cards
    are busy in a single engine — the paper's "prefill-of-A concurrent with
    decode-of-B", without PD's two processes. v1 reorders decode requests to the
    FRONT, so the phase boundary is simply num_decode_tokens; forcing the DBO
    token split there (instead of the even halves) yields the two phase groups.
    """
    if not phase_split_enabled():
        return
    global _PHASE_SPLIT
    _PHASE_SPLIT = True
    import sys
    import numpy as np
    import vllm.v1.worker.gpu_model_runner as gmr

    if getattr(gmr, "_fluidgpu_phase_split", False):
        return
    orig = gmr.maybe_create_ubatch_slices

    stats = {"n": 0}
    print("[fluidgpu] phase-split DBO patch installed", file=sys.stderr, flush=True)

    def patched(
        should_ubatch, num_scheduled_tokens, num_tokens_padded,
        num_reqs_padded, num_ubatches, split_point=None,
    ):
        if should_ubatch and split_point is None and num_ubatches == 2:
            arr = np.asarray(num_scheduled_tokens)
            # decode requests carry 1 token and are ordered first; the phase
            # boundary (token index) is their count.
            nd = int((arr == 1).sum())
            if 0 < nd < int(num_tokens_padded):
                split_point = nd
                stats["n"] += 1
                if stats["n"] == 1 or stats["n"] % 50 == 0:
                    print(
                        f"[fluidgpu] phase-split DBO fired x{stats['n']}: "
                        f"decode ubatch={nd} tok, prefill ubatch="
                        f"{int(num_tokens_padded) - nd} tok",
                        file=sys.stderr, flush=True,
                    )
        return orig(
            should_ubatch, num_scheduled_tokens, num_tokens_padded,
            num_reqs_padded, num_ubatches, split_point,
        )

    gmr.maybe_create_ubatch_slices = patched
    gmr._fluidgpu_phase_split = True


def _patch_determine_batch(worker: Any) -> None:
    import numpy as np
    from vllm.v1.worker.ubatch_utils import check_ubatch_thresholds

    runner = worker.model_runner
    parallel = worker.vllm_config.parallel_config
    orig = runner._determine_batch_execution_and_padding

    def _mixed_phase_split(kwargs: Any, args: Any) -> bool:
        # Phase-split lever E: a MIXED prefill+decode step must run EAGER + DBO
        # (decode ubatch computes locally on the A100, prefill ubatch hops to the
        # L40S), so force it OFF the FULL/PIECEWISE graph. Pure-decode steps are
        # left on the FULL graph (lever A); pure-prefill steps take the normal
        # micro-batch path. Only split when the prefill share is big enough to
        # route remote (else both ubatches stay local — no overlap, just cost).
        if not _PHASE_SPLIT or not kwargs.get("allow_microbatching", True):
            return False
        ns = kwargs.get("num_scheduled_tokens_np")
        if ns is None and len(args) >= 3:
            ns = args[2]
        if ns is None:
            return False
        arr = np.asarray(ns)
        nd = int((arr == 1).sum())
        prefill_tok = int(arr.sum()) - nd
        return nd > 0 and check_ubatch_thresholds(parallel, prefill_tok, False)

    def patched(*args: Any, **kwargs: Any) -> Any:
        mixed_split = _mixed_phase_split(kwargs, args)
        if mixed_split and not kwargs.get("force_eager", False):
            kwargs = dict(kwargs)
            kwargs["force_eager"] = True
        ret = orig(*args, **kwargs)
        cudagraph_mode, batch_desc, should_ubatch, ntad, stats = ret
        if should_ubatch:
            return ret
        if mixed_split:
            # eager now (force_eager applied); guarantee the phase-split split
            return (cudagraph_mode, batch_desc, True, ntad, stats)
        if not kwargs.get("allow_microbatching", True):
            return ret
        # eager-only: never microbatch a graph-dispatched batch
        if getattr(cudagraph_mode, "name", "NONE") != "NONE":
            return ret
        num_tokens = kwargs.get("num_tokens", args[0] if args else None)
        max_sched = kwargs.get("max_num_scheduled_tokens")
        if num_tokens is None or max_sched is None:
            return ret
        uniform_decode = kwargs.get("force_uniform_decode")
        if uniform_decode is None:
            uniform_decode = max_sched == 1
        if check_ubatch_thresholds(parallel, num_tokens, uniform_decode):
            should_ubatch = True
        return (cudagraph_mode, batch_desc, should_ubatch, ntad, stats)

    runner._determine_batch_execution_and_padding = patched


def _stream_priority_range() -> tuple[int, int]:
    """(greatest, least) CUDA stream priority. CUDA semantics: lower number =
    higher priority, so `greatest` is the most negative value."""
    try:
        a, b = torch.cuda.Stream.priority_range()  # type: ignore[attr-defined]
        return min(int(a), int(b)), max(int(a), int(b))
    except Exception:
        # Ampere/Ada expose greatest=-3..-5, least=0.
        return -3, 0


def _ubid_priority(ubid: int) -> int:
    """Priority-aware stream scheduling (paper fig10): earlier micro-batches get
    HIGHER priority (more negative) so the GPU hardware scheduler preferentially
    allocates SMs/copy engines to them, staggering the micro-batches'
    communication phases instead of letting them all stall on transfers at once.
    ubid 0 -> greatest (most negative); each later ubatch steps toward `least`
    (0), clamped so deep pipelines don't wrap around.
    """
    if _env("FLUIDGPU_PRIORITY_STREAMS", "1") != "1":
        return 0
    greatest, least = _stream_priority_range()  # e.g. (-3, 0)
    return min(least, greatest + ubid)


class _RemoteStreams:
    """Per-(micro-batch) stream pair + reusable events for the expert chain.

    One pair PER micro-batch id: with a shared pair, ubatch B's inbound hop
    queues behind ubatch A's whole chain (copy engines idle while the expert
    GPU computes) — per-ubatch streams let A's expert kernels overlap B's
    PCIe transfers, which is where the pipeline win lives on a no-P2P pair.
    Streams carry a ubid-staggered priority (paper's priority-aware scheduling).
    Events are owned and re-recorded per call (each ubatch thread runs its
    chain sequentially), avoiding per-layer event allocation churn.
    """

    def __init__(self, home: str, device: str, priority: int = 0) -> None:
        self.hop = torch.cuda.Stream(device=torch.device(home), priority=priority)
        self.remote = torch.cuda.Stream(
            device=torch.device(device), priority=priority
        )
        self.evt_ready = torch.cuda.Event()
        self.evt_hop = torch.cuda.Event()
        self.evt_back = torch.cuda.Event()
        self.evt_done = torch.cuda.Event()

    _by_key: dict[tuple[str, str, int], "_RemoteStreams"] = {}

    @classmethod
    def get(cls, home: str, device: str, ubid: int) -> "_RemoteStreams":
        key = (home, device, ubid)
        if key not in cls._by_key:
            cls._by_key[key] = cls(home, device, _ubid_priority(ubid))
        return cls._by_key[key]


class _PinnedStage:
    """Reusable pinned bounce buffers for stream-scoped cross-GPU hops.

    Without CUDA P2P, `cudaMemcpyPeerAsync` (what tensor.to() issues) stages
    through the driver and synchronizes with BOTH devices' NULL streams —
    every hop then serializes the sibling micro-batch's private streams,
    destroying the ping-pong overlap. An explicit D2H -> pinned -> H2D chain
    stays entirely on our own streams. Buffers are keyed per (micro-batch,
    tensor tag) and guarded by a completion event so a later layer's D2H
    cannot overwrite bytes the previous H2D has not yet consumed.
    """

    def __init__(self) -> None:
        self._bufs: dict[Any, torch.Tensor] = {}
        self._staged_evt: dict[Any, torch.cuda.Event] = {}
        self._read_evt: dict[Any, torch.cuda.Event] = {}
        self._read_valid: set[Any] = set()
        # chunked double-buffer state (FLUIDGPU_HOP_PIPELINE_MB > 0)
        self._pp_bufs: dict[Any, list[torch.Tensor]] = {}
        self._pp_sevt: dict[Any, list[torch.cuda.Event]] = {}
        self._pp_revt: dict[Any, list[torch.cuda.Event]] = {}
        self._pp_read_valid: set[Any] = set()

    _instance: "_PinnedStage | None" = None

    @classmethod
    def get(cls) -> "_PinnedStage":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    _null_hop = _env("FLUIDGPU_NULL_HOP", "0") == "1"
    # >0: stage in chunks of this many MB through TWO alternating pinned
    # buffers per key, so the D2H of slice k+1 overlaps the H2D of slice k.
    # Whole-tensor staging serializes the two PCIe legs (t_D2H + t_H2D);
    # pipelining costs ~max(leg) + one slice. The legs live on different
    # root ports (src device->host, host->dst device), so they truly overlap.
    _pipeline_bytes = int(_env("FLUIDGPU_HOP_PIPELINE_MB", "0")) << 20

    def transfer(
        self,
        t: torch.Tensor,
        dst_device: str,
        key: Any,
        src_stream: torch.cuda.Stream,
        dst_stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        if self._null_hop:
            # De-risk probe: skip the real cross-GPU transfer (output is garbage)
            # to measure the compute-only throughput ceiling — the upper bound of
            # any hop improvement, IBGDA included. Keeps the stream/event shape.
            with torch.cuda.device(dst_device), torch.cuda.stream(dst_stream):
                return torch.empty(t.shape, dtype=t.dtype, device=dst_device)
        flat = t.contiguous().view(-1).view(torch.uint8)
        nbytes = flat.numel()
        if self._pipeline_bytes and nbytes > self._pipeline_bytes:
            return self._transfer_pipelined(
                t, flat, nbytes, dst_device, key, src_stream, dst_stream
            )
        buf = self._bufs.get(key)
        if buf is None or buf.numel() < nbytes:
            buf = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
            self._bufs[key] = buf
            self._read_valid.discard(key)
        pin = buf[:nbytes]
        if key not in self._staged_evt:
            self._staged_evt[key] = torch.cuda.Event()
            self._read_evt[key] = torch.cuda.Event()

        if key in self._read_valid:
            src_stream.wait_event(self._read_evt[key])
        with torch.cuda.stream(src_stream):
            pin.copy_(flat, non_blocking=True)
        self._staged_evt[key].record(src_stream)

        dst_stream.wait_event(self._staged_evt[key])
        with torch.cuda.device(dst_device), torch.cuda.stream(dst_stream):
            out = torch.empty(t.shape, dtype=t.dtype, device=dst_device)
            out.view(-1).view(torch.uint8).copy_(pin, non_blocking=True)
        self._read_evt[key].record(dst_stream)
        self._read_valid.add(key)
        return out

    def _transfer_pipelined(
        self,
        t: torch.Tensor,
        flat: torch.Tensor,
        nbytes: int,
        dst_device: str,
        key: Any,
        src_stream: torch.cuda.Stream,
        dst_stream: torch.cuda.Stream,
    ) -> torch.Tensor:
        """Chunked double-buffered staging (see ``_pipeline_bytes``).

        Slice k+1's D2H (src_stream) overlaps slice k's H2D (dst_stream);
        the slot-reuse guard (``revt``) keeps slice k+2's D2H from
        overwriting a pinned buffer the H2D of slice k has not consumed.
        Byte-exact copies — parity-neutral by construction.
        """
        cs = self._pipeline_bytes
        bufs = self._pp_bufs.get(key)
        if bufs is None or bufs[0].numel() < cs:
            self._pp_bufs[key] = bufs = [
                torch.empty(cs, dtype=torch.uint8, pin_memory=True)
                for _ in range(2)
            ]
            self._pp_sevt[key] = [torch.cuda.Event(), torch.cuda.Event()]
            self._pp_revt[key] = [torch.cuda.Event(), torch.cuda.Event()]
            self._pp_read_valid.discard((key, 0))
            self._pp_read_valid.discard((key, 1))
        sevt, revt = self._pp_sevt[key], self._pp_revt[key]

        with torch.cuda.device(dst_device), torch.cuda.stream(dst_stream):
            out = torch.empty(t.shape, dtype=t.dtype, device=dst_device)
        out_flat = out.view(-1).view(torch.uint8)

        for i in range((nbytes + cs - 1) // cs):
            s = i & 1
            a = i * cs
            b = min(nbytes, a + cs)
            pin = bufs[s][: b - a]
            if (key, s) in self._pp_read_valid:
                src_stream.wait_event(revt[s])
            with torch.cuda.stream(src_stream):
                pin.copy_(flat[a:b], non_blocking=True)
            sevt[s].record(src_stream)
            dst_stream.wait_event(sevt[s])
            with torch.cuda.device(dst_device), torch.cuda.stream(dst_stream):
                out_flat[a:b].copy_(pin, non_blocking=True)
            revt[s].record(dst_stream)
            self._pp_read_valid.add((key, s))
        return out


def _local_ffn_rows(n_total: int, frac: float) -> int:
    """Rows of a prefill micro-batch to compute locally under balanced
    placement. Deterministic in ``n_total`` so each piecewise capture size
    always takes the same split (no capture hazard)."""
    k = int(round(frac * n_total))
    return max(0, min(n_total, k))


def _num_decode_tokens() -> int:
    """Leading decode-token count of the current batch (v1 reorders decode
    requests to the FRONT). Used by per-token PHASE ROUTING: in a mixed step
    the first ``num_decode_tokens`` rows are decode (their FFN wants the home
    A100 — 1.76x faster, memory-bound) and the rest are prefill (FFN wants the
    expert L40S — 1.41x faster, compute-bound). Cached per attn-metadata object
    so the one CPU sync is once/forward, not once/layer. Misclassification only
    shifts which GPU an FFN runs on (both compute it correctly), so parity is
    unaffected — it is a pure affinity hint."""
    from vllm.forward_context import get_forward_context

    try:
        fc = get_forward_context()
        am = getattr(fc, "attn_metadata", None)
        if isinstance(am, (list, tuple)):
            am = am[0] if am else None
        if not am:
            return 0
        md = next(iter(am.values())) if isinstance(am, dict) else am
        qsl = getattr(md, "query_start_loc", None)
        if qsl is None:
            return 0
        # Cache ON the metadata object: under DBO two micro-batches are live
        # at once, so a single-slot module dict thrashes (a .item() sync per
        # layer per ubatch), and an id()-keyed dict can false-hit when a dead
        # metadata's address is reused. The attribute dies with the object.
        ndec = getattr(md, "_fluidgpu_ndec", None)
        if ndec is None:
            qlens = qsl[1:] - qsl[:-1]
            ndec = int((qlens <= 1).sum().item())  # decodes reordered first
            try:
                md._fluidgpu_ndec = ndec
            except Exception:
                pass
        return ndec
    except Exception:
        return 0


def _decode_split_forward(
    local_out_fn: Any,
    remote_compute_fn: Any,
    remote_inputs: list[torch.Tensor],
    device: str,
    home: str,
) -> torch.Tensor:
    """Bandwidth-balanced decode expert split: the home GPU computes its expert
    subset while the expert GPU computes the rest (its hidden-state inputs
    hopped over), concurrently; the two PARTIAL MoE outputs are SUMMED (each of
    a token's top-k experts lives on exactly one GPU, so the sum is the exact
    full output). Splits the memory-bound decode weight reads across both cards'
    bandwidth (2.0 + 0.86 TB/s). Non-DBO single step (no dbo_yield)."""
    streams = _RemoteStreams.get(home, device, 0)
    compute_stream = torch.cuda.current_stream()
    streams.evt_ready.record(compute_stream)
    streams.hop.wait_event(streams.evt_ready)
    stage = _PinnedStage.get()
    moved = []
    for i, t in enumerate(remote_inputs):
        moved.append(
            stage.transfer(
                t, device, key=("ds", i), src_stream=streams.hop,
                dst_stream=streams.remote,
            )
        )
    with torch.cuda.device(device), torch.cuda.stream(streams.remote):
        out_remote = remote_compute_fn(*moved)
    out_r = stage.transfer(
        out_remote, home, key=("dso",), src_stream=streams.remote,
        dst_stream=streams.hop,
    )
    streams.evt_done.record(streams.hop)
    out_local = local_out_fn()  # home GPU expert subset, overlaps the remote one
    compute_stream.wait_event(streams.evt_done)
    out_r.record_stream(compute_stream)
    w = out_local.shape[-1]
    if out_r.shape[-1] < w:
        out_r = torch.nn.functional.pad(out_r, (0, w - out_r.shape[-1]))
    elif out_r.shape[-1] > w:
        out_r = out_r[..., :w]
    return out_local + out_r


def _phase_split_forward(
    local_out_fn: Any,
    remote_compute_fn: Any,
    remote_inputs: list[torch.Tensor],
    device: str,
    home: str,
) -> torch.Tensor:
    """Mixed-step phase routing (no DBO): decode rows compute LOCAL on the home
    GPU while the prefill rows hop to the expert GPU, concurrently, then concat
    [decode | prefill]. The remote chain runs on side streams launched BEFORE
    the local compute is enqueued (so the inbound hop does not wait on it), and
    both overlap. Single-threaded — used for mixed steps below the DBO prefill
    threshold, so there is no dbo_yield."""
    streams = _RemoteStreams.get(home, device, 0)
    compute_stream = torch.cuda.current_stream()
    streams.evt_ready.record(compute_stream)
    streams.hop.wait_event(streams.evt_ready)
    stage = _PinnedStage.get()
    moved = []
    for i, t in enumerate(remote_inputs):
        moved.append(
            stage.transfer(
                t, device, key=("pr", i), src_stream=streams.hop,
                dst_stream=streams.remote,
            )
        )
    with torch.cuda.device(device), torch.cuda.stream(streams.remote):
        out_remote = remote_compute_fn(*moved)
    out_pre = stage.transfer(
        out_remote, home, key=("pro",), src_stream=streams.remote,
        dst_stream=streams.hop,
    )
    streams.evt_done.record(streams.hop)
    # local decode FFN on the compute stream — overlaps the remote prefill chain
    out_dec = local_out_fn()
    compute_stream.wait_event(streams.evt_done)
    out_pre.record_stream(compute_stream)
    w = out_dec.shape[-1]
    if out_pre.shape[-1] < w:
        out_pre = torch.nn.functional.pad(out_pre, (0, w - out_pre.shape[-1]))
    elif out_pre.shape[-1] > w:
        out_pre = out_pre[..., :w]
    return torch.cat([out_dec, out_pre], dim=0)


def _overlapped_remote(
    compute_fn: Any,
    inputs: list[torch.Tensor],
    device: str,
    home: str,
    local_thunk: Any = None,
) -> torch.Tensor:
    """Run hop->compute->hop-back on side streams, yielding to the sibling
    micro-batch while the expert GPU works. Returns the home-device result,
    ordered against the caller's compute stream.

    Balanced placement (``local_thunk`` set): a share of the FFN rows is
    computed on the HOME GPU concurrently with the remote share. ``local_thunk``
    is enqueued on the compute stream AFTER ``evt_ready`` is recorded (so the
    inbound hop does not wait on it) and BEFORE the yield (so it overlaps the
    expert GPU's work) — home-GPU local FFN and expert-GPU remote FFN then run
    at the same time, driving throughput toward sum-of-rates instead of being
    bounded by the expert GPU alone. Its output is concatenated ahead of the
    remote rows (callers split the rows [local | remote] in that order)."""
    from vllm.v1.worker.ubatching import dbo_current_ubatch_id, dbo_yield

    ubid = dbo_current_ubatch_id()
    streams = _RemoteStreams.get(home, device, ubid)
    compute_stream = torch.cuda.current_stream()
    streams.evt_ready.record(compute_stream)
    streams.hop.wait_event(streams.evt_ready)

    if _env("FLUIDGPU_PINGPONG_HOP", "pinned") == "rdma":
        # Single-shot IB GDR transfers (ncclSend/Recv are stream-async and
        # enqueue on each device's CURRENT stream — our nested contexts pin
        # those to the per-ubatch hop/remote pair). Host-side enqueue order
        # across the two micro-batch threads is serialized by the yield
        # discipline, so the shared communicators see paired groups only.
        pair = _rdma_pair()
        dev_index = torch.device(device).index
        home_index = torch.device(home).index
        with torch.cuda.stream(streams.hop):
            with torch.cuda.device(device), torch.cuda.stream(streams.remote):
                moved = []
                for t in inputs:
                    t.record_stream(streams.hop)
                    moved.append(pair.transfer(t.contiguous(), dev_index))
                out_remote = compute_fn(*moved)
                out_home = pair.transfer(out_remote.contiguous(), home_index)
    else:
        stage = _PinnedStage.get()
        _tm = _StageTiming.get()
        moved = []
        h0 = h1 = None
        if _tm.enabled:
            h0, h1 = _tm.pair(streams.hop)
        for i, t in enumerate(inputs):
            r = stage.transfer(
                t, device, key=("in", ubid, i), src_stream=streams.hop,
                dst_stream=streams.remote,
            )
            moved.append(r)
        with torch.cuda.device(device), torch.cuda.stream(streams.remote):
            if _tm.enabled:
                c0, c1 = _tm.pair(streams.remote)
                out_remote = compute_fn(*moved)
                c1.record(streams.remote)
                _tm.compute.append((c0, c1, int(inputs[0].shape[0])))
                h1.record(streams.hop)
                _tm.hop_in.append(
                    (h0, h1, sum(t.numel() * t.element_size() for t in inputs))
                )
            else:
                out_remote = compute_fn(*moved)
        out_home = stage.transfer(
            out_remote, home, key=("out", ubid), src_stream=streams.remote,
            dst_stream=streams.hop,
        )
        if _tm.enabled:
            _tm.maybe_flush()
    streams.evt_done.record(streams.hop)

    # Balanced share: enqueue the home-GPU FFN on the compute stream now — it
    # overlaps the expert GPU's remote FFN (launched just above on side streams)
    # rather than serializing behind it. evt_ready was already recorded, so the
    # inbound hop does not wait on this work.
    local_out = local_thunk() if local_thunk is not None else None

    # Hand the host (and the primary GPU's compute stream) to the sibling
    # micro-batch; its attention overlaps our expert-GPU work.
    dbo_yield()

    torch.cuda.current_stream().wait_event(streams.evt_done)
    out_home.record_stream(torch.cuda.current_stream())
    if local_out is not None:
        w = local_out.shape[-1]
        if out_home.shape[-1] < w:
            out_home = torch.nn.functional.pad(out_home, (0, w - out_home.shape[-1]))
        elif out_home.shape[-1] > w:
            out_home = out_home[..., :w]
        return torch.cat([local_out, out_home], dim=0)
    return out_home


# ---------------------------------------------------------------------------
# Remote executors + the custom op for dense MLP
# ---------------------------------------------------------------------------

# Per-process registries the opaque ops resolve their python state from
# (mirrors vLLM's static_forward_context pattern for attention/MoE layers).
_REMOTE_MLPS: dict[str, "RemoteDenseMLP"] = {}


class RemoteDenseMLP:
    """A dense MLP relocated to the expert device, with boundary hops.

    The compute path is PLAIN functional ops on snapshotted weights — not the
    vLLM module: vLLM's CustomOp layers (SiluAndMul, ...) carry
    ``maybe_compile`` wrappers when ``custom_ops="none"`` (the inductor
    default), and a lazy torch.compile triggered inside a micro-batch thread
    compiles against the wrong device context ("no kernel image is available
    for execution on the device"). Functional ops have no compile machinery
    and no device-context sensitivity.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        device: str,
        home: str,
        stable_out: _StableOutput,
        phase_aware: bool = False,
        decode_threshold: int = 0,
        balanced: bool = False,
        local_ffn_frac: float = 0.0,
        phase_route: bool = False,
    ) -> None:
        gate_up = getattr(module, "gate_up_proj", None)
        down = getattr(module, "down_proj", None)
        assert gate_up is not None and down is not None, (
            f"unsupported dense MLP layout {type(module).__name__}"
        )
        assert getattr(gate_up, "bias", None) is None and getattr(
            down, "bias", None
        ) is None, "biased dense MLP is not supported"
        self.phase_aware = phase_aware
        self.decode_threshold = decode_threshold
        # Balanced / phase_route need the home-GPU weight copy (as phase-aware
        # does); keep it whenever any is on.
        keep_local = phase_aware or balanced or phase_route
        self.balanced = balanced
        self.local_ffn_frac = local_ffn_frac
        self.phase_route = phase_route
        # Keep a home-GPU copy of the weights (computed BEFORE the module is
        # relocated to the expert device): phase-aware uses it for the decode
        # path (per-layer hop is pure loss for memory-bound decode), balanced
        # placement uses it for the local prefill row-share.
        if keep_local:
            self.w_gate_up_local = gate_up.weight.data.detach().clone()
            self.w_down_local = down.weight.data.detach().clone()
        module.to(device)
        _move_stray_tensors(module, device)
        # MergedColumnParallelLinear packs [gate; up] along the output dim.
        self.w_gate_up = gate_up.weight.data
        self.w_down = down.weight.data
        self.device = device
        self.home = home
        self.stable_out = stable_out

    @staticmethod
    def _mlp(x: torch.Tensor, w_gate_up: torch.Tensor, w_down: torch.Tensor):
        gu = torch.nn.functional.linear(x, w_gate_up)
        gate, up = gu.chunk(2, dim=-1)
        return torch.nn.functional.linear(
            torch.nn.functional.silu(gate) * up, w_down
        )

    def _compute_on_device(self, x: torch.Tensor) -> torch.Tensor:
        return self._mlp(x, self.w_gate_up, self.w_down)

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        from vllm.v1.worker.ubatching import dbo_enabled

        if (
            (self.phase_aware or self.phase_route)
            and hidden_states.shape[0] <= self.decode_threshold
            and hasattr(self, "w_gate_up_local")
        ):
            # decode: compute locally on the home GPU (zero hop), stage for
            # piecewise-graph address stability like the remote path.
            out = self._mlp(hidden_states, self.w_gate_up_local, self.w_down_local)
            return self.stable_out.stage(out, self.home)
        if (
            self.phase_route
            and hasattr(self, "w_gate_up_local")
            and not dbo_enabled()
        ):
            # mixed step per-token phase routing: decode rows' MLP local (A100),
            # prefill rows' MLP remote (L40S), overlapped.
            n = hidden_states.shape[0]
            nd = _num_decode_tokens()
            if 0 < nd < n:
                hs_d, hs_p = hidden_states[:nd], hidden_states[nd:]
                out = _phase_split_forward(
                    lambda: self._mlp(hs_d, self.w_gate_up_local, self.w_down_local),
                    self._compute_on_device,
                    [hs_p],
                    self.device,
                    self.home,
                )
                if out.shape[0] <= self.stable_out.max_tokens:
                    return self.stable_out.stage(out, self.home)
                return out
        if dbo_enabled():
            if self.balanced and hasattr(self, "w_gate_up_local"):
                # balanced placement: first `k` rows' MLP on the home GPU,
                # concurrent with the remote rows' expert-GPU MLP.
                n = hidden_states.shape[0]
                k = _local_ffn_rows(n, self.local_ffn_frac)
                if 0 < k < n:
                    hs_l, hs_r = hidden_states[:k], hidden_states[k:]
                    return _overlapped_remote(
                        self._compute_on_device,
                        [hs_r],
                        self.device,
                        self.home,
                        local_thunk=lambda: self._mlp(
                            hs_l, self.w_gate_up_local, self.w_down_local
                        ),
                    )
            return _overlapped_remote(
                self._compute_on_device, [hidden_states], self.device, self.home
            )
        x = hop_tensor(hidden_states, self.device)
        with torch.cuda.device(self.device):
            out = self._compute_on_device(x)
        return self.stable_out.stage(out, self.home)


def _remote_mlp(hidden_states: torch.Tensor, layer_key: str) -> torch.Tensor:
    return _REMOTE_MLPS[layer_key](hidden_states)


def _remote_mlp_fake(hidden_states: torch.Tensor, layer_key: str) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def _register_remote_mlp_op() -> None:
    from vllm.utils.torch_utils import direct_register_custom_op

    if hasattr(torch.ops.vllm, "fluidgpu_remote_mlp"):
        return
    direct_register_custom_op(
        op_name="fluidgpu_remote_mlp",
        op_func=_remote_mlp,
        mutates_args=[],
        fake_impl=_remote_mlp_fake,
        tags=(torch.Tag.needs_fixed_stride_order,),
    )


_register_remote_mlp_op()

REMOTE_MLP_OP = "vllm::fluidgpu_remote_mlp"
MOE_FORWARD_OP = "vllm::moe_forward"


class FluidRemoteMLP(torch.nn.Module):
    """Proxy that stands in for a relocated dense MLP.

    The class-level forward is what Dynamo reliably traces (instance-patched
    forwards are bypassed by compiled graphs); it emits one opaque custom-op
    node carrying the registry key.
    """

    def __init__(self, layer_key: str) -> None:
        super().__init__()
        self.layer_key = layer_key

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.ops.vllm.fluidgpu_remote_mlp(hidden_states, self.layer_key)


def _make_remote_forward_impl(
    remote: Any,
    device: str,
    home: str,
    stable_out: _StableOutput,
    local_impl: Any = None,
    decode_threshold: int = 0,
    balanced: bool = False,
    local_ffn_frac: float = 0.0,
    local_experts: Any = None,
    phase_route: bool = False,
    ubatch_phase_route: bool = False,
    decode_local: Any = None,
    decode_remote: Any = None,
):
    """Replacement for DefaultMoERunner.forward_impl (behind vllm::moe_forward).

    The opaque op calls ``layer.runner.forward_impl(layer, hidden_states,
    router_logits, shared_experts_input)``; assigning a plain function to the
    runner instance shadows the method, and the op passes ``layer``
    explicitly, so no self-binding is needed. Reached identically in eager
    and compiled execution.

    Phase-aware placement (``local_impl`` set): decode-size batches
    (``num_tokens <= decode_threshold``) run the STOCK local expert compute on
    the home GPU — the per-layer hop chain is pure loss for memory-bound decode
    (measured +5-7 ms/step vs homo), and decode never saturates the expert GPU.
    Prefill batches still disaggregate + ping-pong (compute-bound, both GPUs
    busy). The branch is shape-driven, so each piecewise capture size always
    takes the same path (no capture hazard); the local output is staged through
    the same persistent buffer for graph-address stability.
    """

    def forward_impl(
        layer: Any,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_experts_input: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.v1.worker.ubatching import dbo_enabled

        assert shared_experts_input is None, "shared-experts MoE is not supported"
        if (
            hidden_states.shape[0] <= decode_threshold
            and decode_local is not None
            and decode_remote is not None
        ):
            # bandwidth-balanced decode expert split: home GPU experts [0,Na) +
            # expert GPU experts [Na,32), concurrent, summed. (Forfeits the full
            # decode graph — the remote share is a cross-GPU op — so this trades
            # lever A's fast local decode for split weight-read bandwidth.)
            out = _decode_split_forward(
                lambda: decode_local.forward_on_device(hidden_states, router_logits),
                decode_remote.forward_on_device,
                [hidden_states, router_logits],
                device,
                home,
            )
            if out.shape[-1] < hidden_states.shape[-1]:
                out = torch.nn.functional.pad(
                    out, (0, hidden_states.shape[-1] - out.shape[-1])
                )
            return stable_out.stage(out, home)
        # The stock runner (local_impl) hard-asserts `not dbo_enabled()` and
        # cannot run inside a ubatch thread. A DBO micro-batch is prefill work
        # regardless of its token count (e.g. max-num-seqs 1024 makes the
        # threshold 2048 = one ubatch of a 4096-token step), so routing it to
        # the remote path is the correct semantics, not just crash avoidance.
        # phase-split's decode micro-batch uses local_experts, which supports
        # DBO, and keeps its local branch.
        if hidden_states.shape[0] <= decode_threshold and (
            local_experts is not None
            or (local_impl is not None and not dbo_enabled())
        ):
            # decode: compute locally on the home GPU (zero hop), stage for
            # piecewise-graph address stability like the remote path does.
            if local_experts is not None:  # balanced: home-side marlin copy
                out = local_experts.forward_on_device(hidden_states, router_logits)
                if out.shape[-1] < hidden_states.shape[-1]:
                    out = torch.nn.functional.pad(
                        out, (0, hidden_states.shape[-1] - out.shape[-1])
                    )
            else:  # phase-aware: stock runner on the kept local experts
                out = local_impl(
                    layer, hidden_states, router_logits, shared_experts_input
                )
            staged = stable_out.stage(out, home)
            if _PHASE_SPLIT and dbo_enabled():
                # phase-split DBO: this is the DECODE micro-batch. Yield per
                # layer so the sibling PREFILL micro-batch's FFN + hop (on L40S)
                # overlaps this decode compute (on A100) — both cards busy.
                from vllm.v1.worker.ubatching import dbo_yield
                dbo_yield()
            return staged
        if phase_route and local_experts is not None and not dbo_enabled():
            # per-token PHASE ROUTING in a MIXED prefill+decode step: decode
            # tokens' FFN -> home A100 (memory-bound, 1.76x faster there),
            # prefill tokens' FFN -> expert L40S (compute-bound, 1.41x faster),
            # overlapped -> both GPUs busy through the decode phase. v1 puts
            # decode rows first, so the split is contiguous at num_decode_tokens.
            n = hidden_states.shape[0]
            nd = _num_decode_tokens()
            if 0 < nd < n:
                hs_d, hs_p = hidden_states[:nd], hidden_states[nd:]
                rl_d, rl_p = router_logits[:nd], router_logits[nd:]
                out = _phase_split_forward(
                    lambda: local_experts.forward_on_device(hs_d, rl_d),
                    remote.forward_on_device,
                    [hs_p, rl_p],
                    device,
                    home,
                )
                if out.shape[-1] < hidden_states.shape[-1]:
                    out = torch.nn.functional.pad(
                        out, (0, hidden_states.shape[-1] - out.shape[-1])
                    )
                # mixed steps at a capture size replay piecewise graphs (the op
                # runs eager between pieces) -> stage for address stability.
                if out.shape[0] <= stable_out.max_tokens:
                    return stable_out.stage(out, home)
                return out
        if dbo_enabled():
            # ping-pong micro-batch path: async chain + yield (eager mode, so
            # no graph-address stability is needed — return fresh tensors)
            if ubatch_phase_route and local_experts is not None:
                # per-token phase routing INSIDE a DBO micro-batch: with the
                # decode-cap scheduler every step is mixed, and v1's token
                # split leaves all decode rows at the FRONT of ubatch 0. Their
                # FFN runs on the HOME marlin copy (memory-bound, A100-affine,
                # and the stock runner cannot run in a ubatch thread) via
                # local_thunk — enqueued before the yield so it overlaps the
                # prefill rows' expert-GPU work. Exactly one dbo_yield per
                # layer either way, so the two ubatches' yield counts match.
                n = hidden_states.shape[0]
                nd = _num_decode_tokens()
                if 0 < nd < n:
                    hs_d, hs_p = hidden_states[:nd], hidden_states[nd:]
                    rl_d, rl_p = router_logits[:nd], router_logits[nd:]
                    out = _overlapped_remote(
                        remote.forward_on_device,
                        [hs_p, rl_p],
                        device,
                        home,
                        local_thunk=lambda: local_experts.forward_on_device(
                            hs_d, rl_d
                        ),
                    )
                    if out.shape[-1] < hidden_states.shape[-1]:
                        out = torch.nn.functional.pad(
                            out, (0, hidden_states.shape[-1] - out.shape[-1])
                        )
                    return out
            if balanced and local_experts is not None:
                # balanced placement: compute the first `k` rows' experts on the
                # home GPU (concurrent with the remote rows' expert GPU work),
                # so both GPUs stay busy instead of L40S being FFN-bound. The
                # local share uses a HOME-side copy of the same marlin kernel
                # (context-free forward_on_device — the stock runner asserts on a
                # sub-slice because it reads the full-ubatch forward context).
                n = hidden_states.shape[0]
                k = _local_ffn_rows(n, local_ffn_frac)
                if 0 < k < n:
                    hs_l, hs_r = hidden_states[:k], hidden_states[k:]
                    rl_l, rl_r = router_logits[:k], router_logits[k:]
                    out = _overlapped_remote(
                        remote.forward_on_device,
                        [hs_r, rl_r],
                        device,
                        home,
                        local_thunk=lambda: local_experts.forward_on_device(hs_l, rl_l),
                    )
                    if out.shape[-1] < hidden_states.shape[-1]:
                        out = torch.nn.functional.pad(
                            out, (0, hidden_states.shape[-1] - out.shape[-1])
                        )
                    return out
            out = _overlapped_remote(
                remote.forward_on_device,
                [hidden_states, router_logits],
                device,
                home,
            )
            if out.shape[-1] < hidden_states.shape[-1]:
                out = torch.nn.functional.pad(
                    out, (0, hidden_states.shape[-1] - out.shape[-1])
                )
            return out
        x = hop_tensor(hidden_states, device)
        logits = hop_tensor(router_logits, device)
        out = remote.forward_on_device(x, logits)
        if out.shape[-1] < hidden_states.shape[-1]:
            out = torch.nn.functional.pad(
                out, (0, hidden_states.shape[-1] - out.shape[-1])
            )
        return stable_out.stage(out, home)

    return forward_impl


# ---------------------------------------------------------------------------
# Placement application (runs inside the worker, post-load / pre-compile)
# ---------------------------------------------------------------------------


def _model_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    inner = getattr(model, "model", None)
    layers = getattr(inner, "layers", None)
    assert layers is not None, f"cannot find decoder layers on {type(model).__name__}"
    return list(layers)


def _free_local_expert_weights(layer: torch.nn.Module) -> int:
    """Release the primary-GPU marlin expert tensors after the remote rebuild.

    The patched forward_impl never dispatches to the local quant method
    again, so these parameters are dead weight; freeing them BEFORE vLLM's
    memory profiling hands the reclaimed VRAM to the KV cache.
    """
    freed = 0
    params = getattr(layer, "_parameters", {})
    for name in list(params):
        if name.startswith(("w13_", "w2_")):
            param = params[name]
            if param is not None and param.numel() > 0:
                params[name] = torch.nn.Parameter(
                    torch.empty(0, dtype=param.dtype, device=param.device),
                    requires_grad=False,
                )
                freed += 1
    # marlin repack may re-attach processed weights as plain attributes
    for name, value in list(vars(layer).items()):
        if (
            name.startswith(("w13_", "w2_"))
            and isinstance(value, torch.Tensor)
            and value.numel() > 0
        ):
            setattr(layer, name, torch.empty(0, dtype=value.dtype, device=value.device))
            freed += 1
    return freed


def _resolve_model_dir(model_config: Any) -> Path:
    path = Path(str(model_config.model))
    assert path.is_dir(), (
        f"FLUIDGPU placement needs a local checkpoint dir, got {model_config.model!r}"
    )
    return path


def apply_fluid_placement(worker: Any) -> None:
    placement = _env("FLUIDGPU_VLLM_PLACEMENT", "none")
    if placement == "none":
        return
    assert placement in ("af", "af-bf16"), f"unknown placement {placement!r}"

    vllm_config = worker.vllm_config
    parallel = vllm_config.parallel_config
    assert (
        parallel.tensor_parallel_size == 1 and parallel.pipeline_parallel_size == 1
    ), "FLUIDGPU placement supports TP=1/PP=1 (the second GPU is the expert side)"

    device = _env("FLUIDGPU_EXPERT_DEVICE", "cuda:1")
    home = str(worker.device)
    comp = vllm_config.compilation_config
    graphs_enabled = (
        not vllm_config.model_config.enforce_eager
        and getattr(comp.cudagraph_mode, "name", "NONE") != "NONE"
    )
    _full_decode_ok = (
        _env("FLUIDGPU_FULL_DECODE_GRAPH", "0") == "1"
        and _env("FLUIDGPU_PHASE_AWARE", "0") == "1"
    )
    if graphs_enabled and not _full_decode_ok:
        assert not comp.cudagraph_mode.has_full_cudagraphs(), (
            "FLUIDGPU placement requires cudagraph_mode=PIECEWISE: FULL modes "
            "capture the whole forward (cross-GPU op included) for decode "
            "batches. Pass compilation_config=fluid_compilation_config(). "
            "(FULL_AND_PIECEWISE is allowed only with FLUIDGPU_FULL_DECODE_GRAPH=1 "
            "+ FLUIDGPU_PHASE_AWARE=1, where decode is all-local.)"
        )
        assert MOE_FORWARD_OP in comp.splitting_ops or REMOTE_MLP_OP in comp.splitting_ops, (
            "FLUIDGPU ops missing from splitting_ops; pass "
            "compilation_config=fluid_compilation_config() so the cross-GPU "
            "op runs eagerly between piecewise CUDA graphs."
        )
    max_capture = int(getattr(comp, "max_cudagraph_capture_size", 0) or 0)

    if _hop_mode() != "copy" or (
        pingpong_enabled() and _env("FLUIDGPU_PINGPONG_HOP", "pinned") == "rdma"
    ):
        _rdma_pair()  # deterministic init inside the worker process

    model = worker.model_runner.model
    layers = _model_layers(model)
    dtype = vllm_config.model_config.dtype
    hf_config = vllm_config.model_config.hf_config

    reader: CheckpointReader | None = None
    stable_out: _StableOutput | None = None
    # Phase-aware placement: decode runs experts locally on the home GPU (zero
    # hop), prefill disaggregates. Requires keeping the local expert weights, so
    # it forces FREE_LOCAL_EXPERTS off (the KV cache gets less room in trade).
    phase_aware = _env("FLUIDGPU_PHASE_AWARE", "0") == "1"
    # Balanced placement: split each prefill micro-batch's FFN rows between the
    # home GPU (local share) and the expert GPU (remote share) so both stay
    # busy — lifts the ceiling from expert-GPU-FFN-bound to sum-of-rates. Needs
    # the home-GPU weight copy (like phase-aware) and reuses the local compute
    # path for the local share. local_ffn_frac = fraction of rows kept local;
    # balance point ~= attn_share + f*ffn_share == (1-f)*ffn_share, weighted by
    # each GPU's speed (gpt-oss ~0.37, llama ~0.53 analytically; tune empirically).
    balanced = _env("FLUIDGPU_BALANCED", "0") == "1"
    local_ffn_frac = float(_env("FLUIDGPU_LOCAL_FFN_FRAC", "0.37"))
    # phase routing (lever B): mixed prefill+decode steps split by phase so both
    # GPUs stay busy through the decode phase. Needs the home-side marlin copy
    # (like balanced) for the decode rows' local FFN.
    phase_route = _env("FLUIDGPU_PHASE_ROUTE", "0") == "1"
    # µbatch-interior phase routing (lever E revival, composes with the
    # decode-cap scheduler): with every step mixed, v1's DBO token split puts
    # all decode rows at the front of ubatch 0; route THEIR FFN to the home
    # marlin copy concurrently with the prefill rows' remote FFN, so the home
    # GPU's decode work overlaps the expert GPU instead of hopping with it.
    ubatch_phase_route = _env("FLUIDGPU_UBATCH_PHASE_ROUTE", "0") == "1"
    # Decode expert split (bandwidth-balanced): fraction of experts to offload
    # to the expert GPU for DECODE, so both cards' memory bandwidth reads the
    # weights in parallel. Forfeits full-graph decode (adds a cross-GPU op), so
    # it competes against lever A rather than composing with it. 0 = off.
    decode_split = float(_env("FLUIDGPU_DECODE_SPLIT", "0"))
    # Decode num_tokens <= max_num_seqs (one token/seq); pad the cutoff so
    # small speculative/chunk-tail steps still count as decode, staying well
    # below any real prefill chunk.
    max_seqs = int(getattr(vllm_config.scheduler_config, "max_num_seqs", 32) or 32)
    decode_threshold = int(
        _env("FLUIDGPU_DECODE_LOCAL_THRESHOLD", str(max(64, 2 * max_seqs)))
    )
    # phase-aware reuses the STOCK experts on the home GPU for its decode path,
    # so it must keep them. balanced / phase_route instead build a compact
    # home-side marlin copy (local_experts) for BOTH decode and the local FFN
    # share, so the heavy stock experts can be freed (else the home GPU holds
    # stock + copy -> OOM). Only PURE phase-aware (no home copy) forces free off.
    # Phase-split DBO (lever E): the decode micro-batch computes the FULL expert
    # set on the home GPU under DBO. The STOCK gpt-oss runner asserts
    # `not dbo_enabled()` (modular_kernel.py) — it cannot run inside a ubatch
    # thread — so phase-split, like balanced/phase_route, must use the home-side
    # marlin copy (local_experts) for its decode path instead of local_impl.
    phase_split = phase_split_enabled()
    build_local_experts = (
        balanced or phase_route or phase_split or ubatch_phase_route
    )
    keep_stock = phase_aware and not build_local_experts and decode_split <= 0
    free_local = _env("FLUIDGPU_FREE_LOCAL_EXPERTS", "1") == "1" and not keep_stock
    moe_patched = 0
    dense_patched = 0

    for idx, layer in enumerate(layers):
        mlp = getattr(layer, "mlp", None)
        assert mlp is not None, f"decoder layer {idx} has no mlp module"
        experts = getattr(mlp, "experts", None)

        if experts is not None:  # MoE (gpt-oss): patch behind vllm::moe_forward
            if reader is None:
                reader = CheckpointReader(_resolve_model_dir(vllm_config.model_config))
            if stable_out is None:
                width = int(experts.runner.moe_config.hidden_dim)
                stable_out = _StableOutput(max_capture, width, dtype, home)
            builder = (
                MarlinExpertsOnDevice if placement == "af" else DequantExpertsOnDevice
            )
            builder_kwargs = dict(top_k=int(hf_config.num_experts_per_tok))
            if placement == "af":
                builder_kwargs.update(
                    num_experts=int(hf_config.num_local_experts),
                    hidden_size=int(hf_config.hidden_size),
                    intermediate_size=int(hf_config.intermediate_size),
                )
            remote = builder(idx, reader, device, **builder_kwargs)
            # Balanced placement: a HOME-side copy of the same context-free
            # marlin kernel runs the local row-share (the stock runner asserts
            # on a sub-slice — it reads the full-ubatch forward context).
            local_experts = (
                builder(idx, reader, home, **builder_kwargs)
                if build_local_experts
                else None
            )
            # Decode expert split: physically partition the experts — home GPU
            # holds [0, na), expert GPU holds [na, num_experts) — so decode reads
            # each half's weights on the two cards' bandwidth in parallel.
            decode_local = decode_remote = None
            if decode_split > 0 and placement == "af":
                n_exp = int(hf_config.num_local_experts)
                na = max(1, min(n_exp - 1, round((1.0 - decode_split) * n_exp)))
                decode_local = builder(
                    idx, reader, home, expert_range=(0, na), **builder_kwargs
                )
                decode_remote = builder(
                    idx, reader, device, expert_range=(na, n_exp), **builder_kwargs
                )
            # Save the stock local forward_impl BEFORE shadowing it, so the
            # phase-aware decode branch can call it (runs on the home GPU with
            # the stock experts we keep only for PURE phase_aware). balanced /
            # phase_route use local_experts (home marlin copy) for decode.
            local_impl = experts.runner.forward_impl if keep_stock else None
            experts.runner.forward_impl = _make_remote_forward_impl(
                remote, device, home, stable_out, local_impl, decode_threshold,
                balanced, local_ffn_frac, local_experts, phase_route,
                ubatch_phase_route, decode_local, decode_remote,
            )
            if free_local:
                _free_local_expert_weights(experts)
            moe_patched += 1
        else:  # dense MLP (llama): relocate module, swap in the op proxy
            assert hasattr(mlp, "gate_up_proj") or hasattr(mlp, "up_proj"), (
                f"layer {idx} mlp {type(mlp).__name__} is neither FusedMoE nor dense"
            )
            if stable_out is None:
                width = int(hf_config.hidden_size)
                stable_out = _StableOutput(max_capture, width, dtype, home)
            key = f"fluid_mlp_{idx}"
            _REMOTE_MLPS[key] = RemoteDenseMLP(
                mlp, device, home, stable_out, phase_aware, decode_threshold,
                balanced, local_ffn_frac, phase_route,
            )
            layer.mlp = FluidRemoteMLP(key)
            dense_patched += 1
        torch.cuda.empty_cache()

    logger.info(
        "fluidgpu: placement=%s moved %d MoE / %d dense layers -> %s "
        "(hop=%s, graphs=%s, phase_aware=%s, balanced=%s%s, stable_buffer=%s)",
        placement,
        moe_patched,
        dense_patched,
        device,
        _hop_mode(),
        "piecewise" if graphs_enabled else "off",
        phase_aware,
        balanced,
        f"@{local_ffn_frac:.2f}" if balanced else "",
        f"{max_capture}x{stable_out.buffer.shape[1]}"
        if stable_out is not None and stable_out.buffer is not None
        else "none",
    )


# ---------------------------------------------------------------------------
# Worker subclass + driver-side compilation config
# ---------------------------------------------------------------------------


def _make_worker_base() -> type:
    from vllm.v1.worker.gpu_worker import Worker

    return Worker


class FluidGPUWorker(_make_worker_base()):  # type: ignore[misc]
    """vLLM Worker that applies the FluidGPU placement post-load, pre-compile.

    Use as ``LLM(..., worker_cls="fluidgpu_torch.vllm_integration.FluidGPUWorker")``
    or ``vllm serve --worker-cls fluidgpu_torch.vllm_integration.FluidGPUWorker``.
    ``load_model`` is the last hook before the memory-profiling forward that
    triggers Dynamo tracing, so the traced graph sees the patched modules.
    """

    def load_model(self, *args: Any, **kwargs: Any) -> None:
        placement_active = _env("FLUIDGPU_VLLM_PLACEMENT", "none") != "none"
        pingpong = placement_active and pingpong_enabled()
        if pingpong:
            comp = self.vllm_config.compilation_config
            _full_decode_ok = (
                _env("FLUIDGPU_FULL_DECODE_GRAPH", "0") == "1"
                and _env("FLUIDGPU_PHASE_AWARE", "0") == "1"
            )
            assert _full_decode_ok or self.vllm_config.model_config.enforce_eager or not getattr(
                comp.cudagraph_mode, "has_full_cudagraphs", lambda: False
            )(), (
                "FLUIDGPU_PINGPONG composes with PIECEWISE cudagraphs or "
                "eager only: decode batches replay piecewise graphs "
                "(never micro-batched — the dispatcher gate), large prefill "
                "chunks run the eager two-thread ping-pong; FULL modes would "
                "capture across the micro-batch threads. (FULL_AND_PIECEWISE "
                "allowed with FLUIDGPU_FULL_DECODE_GRAPH=1 + phase-aware: decode "
                "is never micro-batched (dbo_decode_threshold=1<<30) and all-local.)"
            )
            # Must happen BEFORE the stock load_model: use_ubatching is a live
            # property and the UBatchWrapper install + per-ubatch metadata
            # builder allocation read it during init. Depth > 2 uses the
            # generic ubatch_size path (num_ubatches == ubatch_size); deeper
            # pipelines cover the chain latency with more attention slots.
            n_ubatches = int(_env("FLUIDGPU_PINGPONG_UBATCHES", "2"))
            if n_ubatches > 2:
                self.vllm_config.parallel_config.ubatch_size = n_ubatches
            else:
                self.vllm_config.parallel_config.enable_dbo = True
            # VllmConfig.__post_init__ re-runs on config re-validation after
            # the flip and asserts a DeepEP all2all backend whenever
            # use_ubatching. At EP=1 no all2all manager is ever constructed,
            # so the string only placates the gate — DeepEP is never imported
            # (our patched UBatchWrapper/forward_impl handle the dispatch).
            self.vllm_config.parallel_config.all2all_backend = (
                "deepep_low_latency"
            )
            # Micro-batching decode steps is a structural loss at small batch:
            # expert weight reads (the dominant decode cost) nearly double
            # when top-k routing of each half-batch still touches most
            # experts, and eager attention is launch-bound. Prefill splits
            # cleanly (compute scales with tokens, requests are independent),
            # and the paper's shapes are ~90% prefill tokens — so ping-pong
            # defaults to prefill-only. Override thresholds via env.
            self.vllm_config.parallel_config.dbo_decode_token_threshold = int(
                _env("FLUIDGPU_PINGPONG_DECODE_THRESHOLD", str(1 << 30))
            )
            # 4096, not vLLM's 512: each micro-batch chain carries fixed
            # costs (events, pinned handshakes, launches) and forfeits the
            # compiled unsplit path, so splitting only pays for big prefill
            # steps (>=~2k tokens per micro-batch). Online serving emits
            # ~2k-token chunked-prefill steps that CARRY decode requests;
            # at 512 those mixed steps all dropped to eager two-thread
            # execution and fig7 FG saturated at half of AF's throughput.
            self.vllm_config.parallel_config.dbo_prefill_token_threshold = int(
                _env("FLUIDGPU_PINGPONG_PREFILL_THRESHOLD", "4096")
            )
            _patch_ubatch_wrapper_class()
            _patch_ubatch_priority_streams()
            _patch_attn_metadata_cache()
            _patch_phase_split_ubatch()
        super().load_model(*args, **kwargs)
        apply_fluid_placement(self)
        if _env("FLUIDGPU_STEP_TRACE", "0") == "1":
            self._install_step_trace()
        if pingpong:
            _patch_determine_batch(self)
            logger.info(
                "fluidgpu: ping-pong micro-batching enabled (DBO engine at "
                "DP=1; thresholds decode>=%d prefill>=%d tokens)",
                self.vllm_config.parallel_config.dbo_decode_token_threshold,
                self.vllm_config.parallel_config.dbo_prefill_token_threshold,
            )


    def _install_step_trace(self) -> None:
        """FLUIDGPU_STEP_TRACE=1: log per-step wall time (synchronized) so
        losses can be attributed to step classes (decode / mixed / prefill).
        Diagnostic only — the synchronize perturbs pipelining."""
        import time as _time

        runner = self.model_runner
        orig = runner.execute_model
        trace_path = _env(
            "FLUIDGPU_STEP_TRACE_FILE", "/tmp/fluidgpu_step_trace.log"
        )
        trace_file = open(trace_path, "a", buffering=1)

        def timed(scheduler_output: Any, *a: Any, **k: Any) -> Any:
            tokens = getattr(scheduler_output, "total_num_scheduled_tokens", -1)
            torch.cuda.synchronize()
            t0 = _time.perf_counter()
            out = orig(scheduler_output, *a, **k)
            torch.cuda.synchronize()
            trace_file.write(
                f"FLUIDSTEP tokens={tokens} "
                f"ms={(_time.perf_counter() - t0) * 1e3:.1f}\n"
            )
            return out

        runner.execute_model = timed


WORKER_CLS = "fluidgpu_torch.vllm_integration.FluidGPUWorker"


def _make_scheduler_base() -> type:
    from vllm.v1.core.sched.scheduler import Scheduler

    return Scheduler


class FluidGPUScheduler(_make_scheduler_base()):  # type: ignore[misc]
    """Two-pool scheduler: cap the DECODE batch while letting extra requests
    PREFILL ahead — the single-engine analog of PD's decoupled prefill/decode
    instances.

    On a uniform offline flood the stock scheduler runs prefill and decode in
    lockstep waves; once all slots decode, the expert GPU idles (no prefill
    work) — the structural reason FG loses to PD. With max_num_seqs = 64 and
    FLUIDGPU_DECODE_CAP = 32, this holds decode-stage requests beyond 32 out of
    a step (they keep their KV, just wait), so the freed capacity admits fresh
    requests that PREFILL on the expert GPU while 32 decode on the home GPU.
    Staggered admission then keeps wave N+1's prefill overlapping wave N's
    decode — both GPUs busy through the decode phase.

    Held requests are removed from ``self.running`` only for the duration of the
    super().schedule() call (not preempted — KV stays), and the admission cap is
    lowered by the held count so total in-flight stays bounded. Correctness is
    unaffected: a held request resumes next step and its greedy output is
    independent of WHICH step it runs (autoregressive + deterministic), so
    parity holds. Gated by FLUIDGPU_DECODE_CAP (0 = stock behavior).
    """

    def schedule(self):  # type: ignore[override]
        cap = int(_env("FLUIDGPU_DECODE_CAP", "0"))
        if cap <= 0:
            return super().schedule()
        decode = [
            r for r in self.running
            if r.num_computed_tokens >= r.num_prompt_tokens
        ]
        if len(decode) <= cap:
            return super().schedule()
        excess = decode[cap:]
        for r in excess:
            self.running.remove(r)
        saved_cap = self.max_num_running_reqs
        self.max_num_running_reqs = saved_cap - len(excess)
        try:
            return super().schedule()
        finally:
            self.max_num_running_reqs = saved_cap
            self.running.extend(excess)


SCHEDULER_CLS = "fluidgpu_torch.vllm_integration.FluidGPUScheduler"


def fluid_compilation_config() -> dict[str, Any]:
    """compilation_config enabling piecewise CUDA graphs around our ops.

    - PIECEWISE by default: a plain FULL mode would capture the whole decode
      forward INCLUDING our cross-GPU op (broken for a hop op in a graph).
    - FULL_AND_PIECEWISE when FLUIDGPU_FULL_DECODE_GRAPH=1 AND phase-aware:
      phase-aware makes the decode moe_forward all-LOCAL (no hop), so uniform
      decode batches can be captured as a FULL graph (exactly the homo decode
      path — removes the eager-MoE launch overhead, ~19.6->~12.4ms/step);
      non-uniform prefill still runs PIECEWISE + ping-pong. Only valid with
      phase-aware; the worker asserts this.
    - splitting_ops must be passed as a complete list (vLLM replaces, not
      appends): all stock attention/KV ops plus our boundaries, so vLLM
      splits the FX graph there and the ops run eagerly between graph
      replays (in the PIECEWISE branches).
    """
    from vllm.config.compilation import CompilationConfig

    full_decode = (
        _env("FLUIDGPU_FULL_DECODE_GRAPH", "0") == "1"
        and _env("FLUIDGPU_PHASE_AWARE", "0") == "1"
    )
    return {
        "cudagraph_mode": "FULL_AND_PIECEWISE" if full_decode else "PIECEWISE",
        "splitting_ops": [
            *CompilationConfig._attention_ops,
            "vllm::unified_kv_cache_update",
            "vllm::unified_mla_kv_cache_update",
            MOE_FORWARD_OP,
            REMOTE_MLP_OP,
        ],
    }
