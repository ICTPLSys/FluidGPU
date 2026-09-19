from __future__ import annotations

import atexit
import os
import time
import warnings
from dataclasses import dataclass
from threading import Lock
from typing import Any

import torch

from . import comm
from .config import EngineConfig, validate_torch_version
from .executor import LayerExecutor
from .monitor import OnlineMonitor, QueueingAwareMonitor
from .profile import load_profile, owned_layer_ids
from .runner import CausalLMRunner


@dataclass
class LLMEngineWorker:
    engine: LLMEngine
    worker_id: int
    runner: CausalLMRunner
    comm_backend: Any
    stream: torch.cuda.Stream | None = None

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool = False,
        seed: int = 0,
    ) -> str | None:
        if self.stream is None:
            return self.engine._generate_with_runner(
                self.runner,
                self.comm_backend,
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                seed=seed,
            )

        torch.cuda.set_device(self.engine.cfg.device_id)
        self.stream.wait_stream(torch.cuda.current_stream(self.engine.device))
        with torch.cuda.stream(self.stream):
            result = self.engine._generate_with_runner(
                self.runner,
                self.comm_backend,
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                seed=seed,
            )
        self.stream.synchronize()
        return result

    def close(self) -> None:
        return None


class LLMEngine:
    def __init__(self, cfg: EngineConfig) -> None:
        validate_torch_version()
        self._validate_transformers(cfg)
        self.cfg = cfg
        comm.init_process_group(cfg)
        self.rank = comm.get_rank()
        self.device = cfg.torch_device

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.hf_model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            **_model_dtype_kwargs(cfg.dtype),
            **_quantization_kwargs(cfg),
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.hf_model.eval()

        from .gptoss_fused_moe import maybe_patch_gptoss_fused_moe

        fused_layers = maybe_patch_gptoss_fused_moe(self.hf_model)
        if fused_layers:
            print(
                f"fluidgpu_torch: fused top-k MoE enabled on {fused_layers} "
                "dequantized gpt-oss layers (FLUIDGPU_GPTOSS_FUSED_MOE=0 disables)"
            )

        self.profile = load_profile(
            cfg.profiling_path,
            model_name=cfg.model_name,
            hf_config=self.hf_model.config,
        )
        assert self.profile.rank_count == cfg.world_size, (
            f"profile rank_count {self.profile.rank_count} does not match "
            f"engine world_size {cfg.world_size}"
        )
        self._seed_lock = Lock()
        # Count of worker-private NCCL groups created via create_worker().
        # With more than one, the sync-free decode loop is unsafe: free-running
        # worker threads enqueue different groups' p2p ops in rank-divergent
        # order, and NCCL guarantees no forward progress across communicators
        # sharing a device (multi-communicator ordering deadlock; runbook §8).
        self._worker_group_count = 0
        owned = owned_layer_ids(self.profile, self.rank)
        self.executor = LayerExecutor(self.profile, cfg)
        self.runner = CausalLMRunner(self.hf_model, self.tokenizer, self.executor, cfg, owned)
        self.cuda_graph_decode = cfg.cuda_graph_decode
        if self.cuda_graph_decode:
            from .graph_decode import profile_supports_graph_decode

            assert profile_supports_graph_decode(self.profile), (
                "cuda_graph_decode requires a dense attn/mlp granularity profile"
            )
            self.runner.kv_cache.static_mode = True
        self.monitor = OnlineMonitor(cfg.monitor_output)
        self._policy_lock = Lock()
        self._policy_profiles = {"latency": self.profile, "throughput": self.profile}
        self.active_policy = "latency"
        self.queueing_monitor: QueueingAwareMonitor | None = None
        atexit.register(self.close)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        do_sample: bool = False,
        seed: int = 0,
    ) -> str | None:
        assert not do_sample, "fluidgpu_torch v1 supports greedy generation only"
        first_logits, tokens = self._generate_with_diagnostics(
            self.runner,
            comm.default_backend(),
            prompt,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        if self.rank != 0:
            return None
        input_ids = self._tokenize(prompt)
        output_ids = torch.cat(
            [input_ids, torch.tensor([tokens], dtype=torch.long, device=self.device)],
            dim=1,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def generate_with_diagnostics(
        self,
        prompt: str,
        max_new_tokens: int,
        seed: int = 0,
    ) -> tuple[torch.Tensor | None, list[int]]:
        return self._generate_with_diagnostics(
            self.runner,
            comm.default_backend(),
            prompt,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )

    def create_worker(
        self,
        worker_id: int,
        *,
        use_process_group: bool = False,
        stream_priority: int = 0,
        profile: Any | None = None,
    ) -> LLMEngineWorker:
        """Create a pipeline worker.

        `profile` overrides the engine profile for this worker only. Giving
        concurrent workers complementary placements (paper zig-zag: request A
        runs attention where request B runs FFN) is what keeps BOTH GPUs busy
        every step — with a single shared plan, all in-flight requests are
        phase-aligned and the off-plan GPU idles through the decode phase.
        """
        if use_process_group:
            comm_backend = comm.new_worker_backend()
            self._worker_group_count += 1
        else:
            comm_backend = comm.default_backend()
        worker_profile = profile if profile is not None else self.profile
        assert worker_profile.rank_count == self.cfg.world_size, (
            f"worker profile rank_count {worker_profile.rank_count} != "
            f"world_size {self.cfg.world_size}"
        )
        executor = LayerExecutor(worker_profile, self.cfg, comm_backend=comm_backend)
        owned = owned_layer_ids(worker_profile, self.rank)
        runner = CausalLMRunner(self.hf_model, self.tokenizer, executor, self.cfg, owned)
        if self.cuda_graph_decode:
            if profile is not None:
                from .graph_decode import profile_supports_graph_decode

                assert profile_supports_graph_decode(worker_profile), (
                    "cuda_graph_decode requires a dense attn/mlp granularity "
                    "profile (worker profile override)"
                )
            runner.kv_cache.static_mode = True
        # CUDA stream priority: lower value = higher priority. Priority-aware
        # pipelining gives earlier workers higher priority so concurrent
        # requests stagger their communication phases (paper §III-C).
        stream = torch.cuda.Stream(device=self.device, priority=stream_priority)
        return LLMEngineWorker(
            engine=self,
            worker_id=worker_id,
            runner=runner,
            comm_backend=comm_backend,
            stream=stream,
        )

    def configure_policy_switching(
        self,
        *,
        latency_profile: Any | None = None,
        throughput_profile: Any | None = None,
        window_ms: float = 300.0,
        threshold_beta: float = 1.5,
    ) -> QueueingAwareMonitor:
        if latency_profile is not None:
            self._policy_profiles["latency"] = latency_profile
        if throughput_profile is not None:
            self._policy_profiles["throughput"] = throughput_profile
        self.queueing_monitor = QueueingAwareMonitor(
            window_ms=window_ms,
            threshold_beta=threshold_beta,
            on_switch=self.switch_policy,
            initial_policy=self.active_policy,
        )
        return self.queueing_monitor

    def switch_policy(self, policy: str) -> None:
        assert policy in self._policy_profiles, f"unknown policy {policy}"
        with self._policy_lock:
            profile = self._policy_profiles[policy]
            self.active_policy = policy
            self.profile = profile
            self.executor.profile = profile
            self.executor.tasks = list(profile.tasks)
            self.executor.task_index_by_name = profile.task_index_by_name

    def _generate_with_runner(
        self,
        runner: CausalLMRunner,
        comm_backend: Any,
        prompt: str,
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        seed: int = 0,
    ) -> str | None:
        assert not do_sample, "fluidgpu_torch v1 supports greedy generation only"
        first_logits, tokens = self._generate_with_diagnostics(
            runner,
            comm_backend,
            prompt,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
        if self.rank != 0:
            return None
        input_ids = self._tokenize(prompt)
        output_ids = torch.cat(
            [input_ids, torch.tensor([tokens], dtype=torch.long, device=self.device)],
            dim=1,
        )
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

    def _generate_with_diagnostics(
        self,
        runner: CausalLMRunner,
        comm_backend: Any,
        prompt: str,
        *,
        max_new_tokens: int,
        seed: int = 0,
    ) -> tuple[torch.Tensor | None, list[int]]:
        assert max_new_tokens >= 0, f"max_new_tokens must be non-negative, got {max_new_tokens}"
        with self._seed_lock:
            torch.manual_seed(seed)
        input_ids = self._tokenize(prompt)
        if max_new_tokens == 0:
            return None, []

        with torch.inference_mode():
            logits = runner.prefill(input_ids)
            next_token = torch.empty((1, 1), dtype=torch.long, device=self.device)
            first_logits: torch.Tensor | None = None
            if self.rank == 0:
                assert logits is not None
                first_logits = logits[:, -1, :].detach().clone()
                next_token.fill_(int(first_logits.argmax(dim=-1).item()))
            comm_backend.broadcast_tensor(next_token, src=0)
            tokens = [int(next_token.item())]

            plan = None
            if self.cuda_graph_decode and not os.environ.get("FLUIDGPU_GRAPH_DEBUG_RUNNER"):
                plan = self._graph_plan_for(runner, comm_backend)
                plan.begin_request(int(input_ids.shape[1]), next_token)
                if not plan._captured:
                    plan.capture()

            # The sync-free loop is only safe with a single NCCL communicator:
            # with per-worker groups, unsynchronized host run-ahead lets the two
            # ranks enqueue different groups' ops in divergent order and NCCL
            # deadlocks (multi-communicator ordering hazard; runbook §8). Force
            # the per-step host-synchronized loop whenever multiple worker
            # groups exist; FLUIDGPU_UNSAFE_SYNC_FREE=1 overrides for experiments.
            force_sync = self._worker_group_count > 1 and not os.environ.get(
                "FLUIDGPU_UNSAFE_SYNC_FREE"
            )
            if plan is not None and (force_sync or os.environ.get("FLUIDGPU_SYNC_DECODE")):
                # Legacy per-step host-synchronized loop (A/B reference).
                for _ in range(max_new_tokens - 1):
                    logits = plan.decode_step()
                    if self.rank == 0:
                        assert logits is not None
                        next_token.fill_(int(logits[:, -1, :].argmax(dim=-1).item()))
                    comm_backend.broadcast_tensor(next_token, src=0)
                    tokens.append(int(next_token.item()))
                    plan.advance(next_token)
            elif plan is not None:
                # Sync-free decode loop: argmax stays on device and tokens are
                # collected into a device buffer, so the host runs ahead of the
                # GPU instead of stalling twice per step on .item().
                token_history = torch.empty(
                    max_new_tokens, dtype=torch.long, device=self.device
                )
                token_history[0].copy_(next_token[0, 0])
                for step in range(1, max_new_tokens):
                    logits = plan.decode_step()
                    if self.rank == 0:
                        assert logits is not None
                        next_token.copy_(logits[:, -1, :].argmax(dim=-1, keepdim=True))
                    comm_backend.broadcast_tensor(next_token, src=0)
                    token_history[step].copy_(next_token[0, 0])
                    plan.advance(next_token)
                    # The sync-free loop no longer blocks on .item(), which
                    # used to release the GIL each step; yield explicitly so
                    # sibling pipeline workers keep interleaving fairly.
                    time.sleep(0)
                tokens = [int(value) for value in token_history.tolist()]
            else:
                for _ in range(max_new_tokens - 1):
                    logits = runner.decode_step(next_token)
                    if self.rank == 0:
                        assert logits is not None
                        next_token.fill_(int(logits[:, -1, :].argmax(dim=-1).item()))
                    comm_backend.broadcast_tensor(next_token, src=0)
                    tokens.append(int(next_token.item()))
        return first_logits, tokens

    def _graph_plan_for(self, runner: CausalLMRunner, comm_backend: Any):
        plan = getattr(runner, "_graph_decode_plan", None)
        if plan is None:
            from .graph_decode import GraphedDecodePlan

            # The plan must follow the runner's OWN profile: with per-worker
            # complementary placements (create_worker(profile=...)), the shared
            # engine profile describes a different task->rank assignment than
            # the runner's executor/KV layout and would capture segments for
            # layers this rank does not own.
            plan = GraphedDecodePlan(
                runner,
                runner.executor.profile,
                self.cfg,
                comm_backend,
                self.rank,
            )
            runner._graph_decode_plan = plan
        return plan

    def close(self) -> None:
        self.monitor.flush()
        comm.destroy()

    def _tokenize(self, prompt: str) -> torch.Tensor:
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        if input_ids.shape[1] > self.cfg.max_seq_len:
            warnings.warn(
                f"prompt length {input_ids.shape[1]} exceeds max_seq_len {self.cfg.max_seq_len}; "
                "using the last max_seq_len tokens",
                RuntimeWarning,
                stacklevel=2,
            )
            input_ids = input_ids[:, -self.cfg.max_seq_len :]
        return input_ids

    @staticmethod
    def _validate_transformers(cfg: EngineConfig) -> None:
        if not cfg.strict_transformers_version:
            return
        import transformers

        major_minor = transformers.__version__.split(".", 2)[:2]
        major, minor = int(major_minor[0]), int(major_minor[1])
        assert major == 4 and minor >= 56, (
            f"transformers must be >=4.56,<5.0 (paired with torch 2.10.0), got {transformers.__version__}"
        )


def _model_dtype_kwargs(dtype: torch.dtype) -> dict[str, torch.dtype]:
    import transformers

    major_minor = transformers.__version__.split(".", 2)[:2]
    major, minor = int(major_minor[0]), int(major_minor[1])
    if major > 4 or (major == 4 and minor >= 56):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def _quantization_kwargs(cfg: EngineConfig) -> dict[str, Any]:
    """Per-rank weight representation for MXFP4 checkpoints (gpt-oss).

    Pre-Hopper/Ada GPUs (sm < 8.9) have no FP4/FP8 hardware, so the triton
    MXFP4 unpack GEMMs run ~7x slower than cuBLAS bf16 at prefill; dequantize
    there when the bf16 weights fit. Ada-class GPUs with less memory keep the
    quantized weights. Dequantization is a deterministic lossless expansion,
    so mixed-representation ranks stay numerically consistent.
    Override with FLUIDGPU_MXFP4_DEQUANTIZE=0|1 (FLUIDGPU_GPTOSS_DEQUANTIZE
    is honored as an alias).
    """
    from transformers import AutoConfig

    hf_config = AutoConfig.from_pretrained(cfg.model_name)
    if getattr(hf_config, "quantization_config", None) is None:
        return {}
    quant_method = getattr(hf_config.quantization_config, "quant_method", None) or (
        hf_config.quantization_config.get("quant_method")
        if isinstance(hf_config.quantization_config, dict)
        else None
    )
    if str(quant_method) != "mxfp4":
        return {}

    override = os.environ.get("FLUIDGPU_MXFP4_DEQUANTIZE")
    if override is None:
        override = os.environ.get("FLUIDGPU_GPTOSS_DEQUANTIZE")
    if override is not None:
        dequantize = override == "1"
    else:
        major, minor = torch.cuda.get_device_capability(cfg.device_id)
        free_bytes, _ = torch.cuda.mem_get_info(cfg.device_id)
        # bf16 gpt-oss-20b needs ~52 GB weights + headroom.
        dequantize = (major, minor) < (8, 9) and free_bytes > 60 * 1024**3
    if not dequantize:
        return {}
    from transformers import Mxfp4Config

    return {"quantization_config": Mxfp4Config(dequantize=True)}
