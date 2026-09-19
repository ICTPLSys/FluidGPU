from __future__ import annotations

import dataclasses
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from .config import EngineConfig
from .engine import LLMEngine


log = logging.getLogger("fluidgpu_torch.vllm_compat")


def normalize_hf_model_name(model: str | os.PathLike[str]) -> str:
    """Return the profile model id for HF cache snapshot paths.

    vLLM 0.18 rewrites model ids to local snapshot paths when
    ``HF_HUB_OFFLINE=1``. Profiles keep the original repo id, so normalize
    paths like ``.../models--Qwen--Qwen2.5-1.5B/snapshots/<sha>`` back to
    ``Qwen/Qwen2.5-1.5B``.
    """

    model_str = str(model)
    m = re.search(r"(?:^|/)models--([^/]+?)--(.+?)/snapshots(?:/|$)", model_str)
    if m:
        return f"{m.group(1)}/{m.group(2)}"

    parts = Path(model_str).parts
    for part in parts:
        if part.startswith("models--") and "--" in part[len("models--") :]:
            _, owner, name = part.split("--", 2)
            return f"{owner}/{name}"
    return model_str


@dataclass(init=False)
class SamplingParams:
    max_tokens: int = 16
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    ignore_eos: bool = False
    detokenize: bool = True
    seed: int | None = None
    stop: list[str] | None = None
    stop_token_ids: list[int] | None = None
    logprobs: int | None = None
    prompt_logprobs: int | None = None
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    _extras: dict[str, Any] = field(default_factory=dict)

    def __init__(self, **kwargs: Any) -> None:
        known = {f.name for f in dataclasses.fields(self) if f.name != "_extras"}
        self._extras = {k: v for k, v in kwargs.items() if k not in known}
        for f in dataclasses.fields(self):
            if f.name == "_extras":
                continue
            if f.name in kwargs:
                value = kwargs[f.name]
            elif f.default is not dataclasses.MISSING:
                value = f.default
            elif f.default_factory is not dataclasses.MISSING:
                value = f.default_factory()
            else:
                raise TypeError(f"missing default for SamplingParams.{f.name}")
            setattr(self, f.name, value)

        if self.n != 1:
            raise NotImplementedError(f"vllm_compat only supports n=1, got n={self.n}")


@dataclass
class CompletionOutput:
    index: int
    text: str
    token_ids: list[int]
    cumulative_logprob: float | None = None
    logprobs: Any = None
    finish_reason: str = "length"
    stop_reason: str | None = None


@dataclass
class RequestOutput:
    request_id: str
    prompt: str
    prompt_token_ids: list[int]
    outputs: list[CompletionOutput]
    finished: bool = True
    prompt_logprobs: Any = None
    metrics: Any = None
    lora_request: Any = None
    encoder_prompt: str | None = None
    encoder_prompt_token_ids: list[int] | None = None
    num_cached_tokens: int = 0
    multi_modal_placeholders: Any = None


def _coerce_prompt(prompt: Any, tokenizer: Any) -> tuple[str, list[int]]:
    if isinstance(prompt, str):
        tokenized = tokenizer(prompt, add_special_tokens=False)
        return prompt, list(tokenized.input_ids)

    if isinstance(prompt, dict):
        if prompt.get("multi_modal_data"):
            raise NotImplementedError(
                "MLLM (multi_modal_data) is not supported: the vLLM "
                "integration is CausalLM-only"
            )
        if "prompt_token_ids" in prompt:
            token_ids = list(prompt["prompt_token_ids"])
            return tokenizer.decode(token_ids, skip_special_tokens=False), token_ids
        if "prompt" in prompt:
            text = prompt["prompt"]
            if not isinstance(text, str):
                raise TypeError(f"vllm_compat expected text prompt str, got {type(text)!r}")
            tokenized = tokenizer(text, add_special_tokens=False)
            return text, list(tokenized.input_ids)

    raise TypeError(f"vllm_compat: unsupported prompt type {type(prompt)!r}")


class FluidgpuLLM:
    """Small vLLM LLM shim for vllm.benchmarks.throughput.

    This intentionally implements only the synchronous greedy path. The
    benchmark reads token counts from RequestOutput objects; continuous
    batching, paged KV, and non-greedy sampling remain outside this shim.
    """

    _SUPPORTED_ENGINE_KWARGS = {
        "model",
        "tokenizer",
        "tokenizer_mode",
        "trust_remote_code",
        "dtype",
        "max_model_len",
        "tensor_parallel_size",
        "seed",
    }

    def __init__(self, **engine_kwargs: Any) -> None:
        unsupported_enabled = [
            name
            for name in ("enable_lora", "use_beam_search")
            if bool(engine_kwargs.get(name, False))
        ]
        if unsupported_enabled:
            raise NotImplementedError(
                "vllm_compat does not support " + ", ".join(sorted(unsupported_enabled))
            )

        ignored = sorted(k for k in engine_kwargs if k not in self._SUPPORTED_ENGINE_KWARGS)
        if ignored:
            log.warning("fluidgpu_torch.vllm_compat: ignoring engine kwargs %s", ignored)

        model = engine_kwargs.get("model")
        if not model:
            raise ValueError("vllm_compat requires engine kwarg 'model'")
        model = normalize_hf_model_name(model)

        dtype_arg = engine_kwargs.get("dtype", "bfloat16")
        if dtype_arg not in ("auto", "bfloat16", torch.bfloat16):
            raise NotImplementedError(f"vllm_compat only supports bfloat16, got {dtype_arg!r}")

        tp_size = int(engine_kwargs.get("tensor_parallel_size") or 1)
        if tp_size <= 0:
            raise NotImplementedError(
                f"tensor_parallel_size must be positive, got {tp_size}"
            )

        profile_json = os.environ.get("FLUIDGPU_PROFILE_JSON")
        if not profile_json:
            raise AssertionError("env FLUIDGPU_PROFILE_JSON must point to an fluidgpu profile json")

        rank_override = _optional_int_env("FLUIDGPU_RANK")
        max_model_len = int(engine_kwargs.get("max_model_len") or 2048)
        cfg = EngineConfig(
            model_name=str(model),
            profiling_json=profile_json,
            dtype=torch.bfloat16,
            max_seq_len=max_model_len,
            master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
            master_port=int(os.environ.get("MASTER_PORT", "29500")),
            rank_override=rank_override,
            device_id=0,
            world_size=_optional_int_env("FLUIDGPU_WORLD_SIZE")
            or _optional_int_env("WORLD_SIZE")
            or 2,
            strict_transformers_version=False,
            comm_transport=os.environ.get("FLUIDGPU_COMM_TRANSPORT", "rdma"),  # type: ignore[arg-type]
            nccl_socket_ifname=os.environ.get("FLUIDGPU_NCCL_SOCKET_IFNAME")
            or os.environ.get("NCCL_SOCKET_IFNAME"),
            nccl_ib_hca=os.environ.get("FLUIDGPU_NCCL_IB_HCA") or os.environ.get("NCCL_IB_HCA"),
            nccl_ib_gid_index=_optional_int_env("FLUIDGPU_NCCL_IB_GID_INDEX")
            or _optional_int_env("NCCL_IB_GID_INDEX"),
        )
        self._engine = LLMEngine(cfg)
        self.tokenizer = self._engine.tokenizer
        self.llm_engine = SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=max_model_len, model=str(model)),
            tokenizer=self.tokenizer,
        )
        self._announced = False

    def generate(
        self,
        prompts: Any,
        sampling_params: Any = None,
        *,
        lora_request: Any = None,
        use_tqdm: bool = True,
        **_: Any,
    ) -> list[RequestOutput]:
        self._announce_once()
        if lora_request:
            raise NotImplementedError("vllm_compat does not support lora_request")

        prompt_list = prompts if isinstance(prompts, list) else [prompts]
        params_list = _normalize_sampling_params(sampling_params, len(prompt_list))
        outputs: list[RequestOutput] = []
        for idx, (prompt, params) in enumerate(zip(prompt_list, params_list)):
            prompt_text, prompt_token_ids = _coerce_prompt(prompt, self.tokenizer)
            max_tokens = int(getattr(params, "max_tokens", 16))
            seed = int(getattr(params, "seed", 0) or 0)
            _, token_ids = self._engine.generate_with_diagnostics(
                prompt_text,
                max_new_tokens=max_tokens,
                seed=seed,
            )
            text = (
                self.tokenizer.decode(token_ids, skip_special_tokens=True)
                if getattr(params, "detokenize", True)
                else ""
            )
            outputs.append(
                RequestOutput(
                    request_id=str(idx),
                    prompt=prompt_text,
                    prompt_token_ids=prompt_token_ids,
                    outputs=[CompletionOutput(index=0, text=text, token_ids=list(token_ids))],
                )
            )
        return outputs

    def start_profile(self) -> None:
        return None

    def stop_profile(self) -> None:
        return None

    def close(self) -> None:
        self._engine.close()

    def _announce_once(self) -> None:
        if self._announced:
            return
        self._announced = True
        print(
            "[fluidgpu_torch.vllm_compat] sampling: forced greedy "
            "(ignoring temperature/top_p/top_k)",
            file=sys.stderr,
        )
        print(
            "[fluidgpu_torch.vllm_compat] batching: per-request serial "
            "(no continuous batching / paged KV)",
            file=sys.stderr,
        )
        print(
            "[fluidgpu_torch.vllm_compat] parallelism: fluidgpu kernel-group split "
            f"with world_size={self._engine.cfg.world_size} (not vLLM TP)",
            file=sys.stderr,
        )


def _normalize_sampling_params(sampling_params: Any, count: int) -> list[Any]:
    if sampling_params is None:
        return [SamplingParams() for _ in range(count)]
    if isinstance(sampling_params, list):
        assert len(sampling_params) == count, (
            f"len(prompts)={count} vs len(sampling_params)={len(sampling_params)}"
        )
        return sampling_params
    return [sampling_params for _ in range(count)]


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name)
    return int(value) if value is not None else None
