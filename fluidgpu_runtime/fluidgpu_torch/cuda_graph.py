from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import nn


class StaticCUDAGraph:
    """Standalone CUDA Graph wrapper for fixed-shape inference experiments."""

    def __init__(
        self,
        fn: Callable[..., Any],
        example_args: tuple[Any, ...] = (),
        example_kwargs: Mapping[str, Any] | None = None,
        *,
        warmup_iters: int = 3,
        clone_outputs: bool = True,
    ) -> None:
        assert warmup_iters >= 0, f"warmup_iters must be non-negative, got {warmup_iters}"
        assert torch.cuda.is_available(), "CUDA Graph capture requires CUDA"
        self.fn = fn
        self.clone_outputs = clone_outputs
        self._static_args = _clone_inputs(example_args)
        self._static_kwargs = _clone_inputs(dict(example_kwargs or {}))
        self.device = _first_cuda_device((self._static_args, self._static_kwargs))
        assert self.device is not None, "at least one CUDA tensor input is required"
        _assert_same_cuda_device((self._static_args, self._static_kwargs), self.device)

        self.graph = torch.cuda.CUDAGraph()
        self._static_outputs: Any = None
        self._capture(warmup_iters)

    def replay(self, *args: Any, **kwargs: Any) -> Any:
        _copy_inputs(self._static_args, args)
        _copy_inputs(self._static_kwargs, kwargs)
        self.graph.replay()
        if self.clone_outputs:
            return _clone_outputs(self._static_outputs)
        return self._static_outputs

    def _capture(self, warmup_iters: int) -> None:
        torch.cuda.synchronize(self.device)
        warmup_stream = torch.cuda.Stream(device=self.device)
        warmup_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(warmup_stream):
            for _ in range(warmup_iters):
                self._static_outputs = self.fn(*self._static_args, **self._static_kwargs)
        torch.cuda.current_stream(self.device).wait_stream(warmup_stream)
        torch.cuda.synchronize(self.device)

        with torch.cuda.graph(self.graph):
            self._static_outputs = self.fn(*self._static_args, **self._static_kwargs)
        torch.cuda.synchronize(self.device)


class CUDAGraphModule(nn.Module):
    """Wrap an nn.Module with optional CUDA Graph replay for fixed-shape inference."""

    def __init__(
        self,
        module: nn.Module,
        *,
        warmup_iters: int = 3,
        clone_outputs: bool = True,
        inference_mode: bool = True,
    ) -> None:
        super().__init__()
        assert warmup_iters >= 0, f"warmup_iters must be non-negative, got {warmup_iters}"
        self.module = module
        self.warmup_iters = warmup_iters
        self.clone_outputs = clone_outputs
        self.inference_mode = inference_mode
        self._graph: StaticCUDAGraph | None = None

    @property
    def is_captured(self) -> bool:
        return self._graph is not None

    def capture(self, *example_args: Any, **example_kwargs: Any) -> CUDAGraphModule:
        self.module.eval()
        self._graph = StaticCUDAGraph(
            self._run_module,
            example_args,
            example_kwargs,
            warmup_iters=self.warmup_iters,
            clone_outputs=self.clone_outputs,
        )
        return self

    def reset_capture(self) -> None:
        self._graph = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self._graph is None:
            return self._run_module(*args, **kwargs)
        return self._graph.replay(*args, **kwargs)

    def _run_module(self, *args: Any, **kwargs: Any) -> Any:
        if self.inference_mode:
            with torch.inference_mode():
                return self.module(*args, **kwargs)
        return self.module(*args, **kwargs)


def _clone_inputs(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        assert value.device.type == "cuda", f"expected CUDA tensor, got {value.device}"
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_inputs(item) for item in value)
    if isinstance(value, list):
        return [_clone_inputs(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_inputs(item) for key, item in value.items()}
    return value


def _clone_outputs(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, tuple):
        return tuple(_clone_outputs(item) for item in value)
    if isinstance(value, list):
        return [_clone_outputs(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_outputs(item) for key, item in value.items()}
    return value


def _copy_inputs(static: Any, incoming: Any) -> None:
    if isinstance(static, torch.Tensor):
        assert isinstance(incoming, torch.Tensor), "incoming value must be a tensor"
        assert static.shape == incoming.shape, (
            f"shape mismatch: expected {tuple(static.shape)}, got {tuple(incoming.shape)}"
        )
        assert static.dtype == incoming.dtype, (
            f"dtype mismatch: expected {static.dtype}, got {incoming.dtype}"
        )
        assert static.device == incoming.device, (
            f"device mismatch: expected {static.device}, got {incoming.device}"
        )
        static.copy_(incoming)
        return

    if isinstance(static, tuple):
        assert isinstance(incoming, tuple), "incoming value must be a tuple"
        assert len(static) == len(incoming), (
            f"tuple length mismatch: expected {len(static)}, got {len(incoming)}"
        )
        for static_item, incoming_item in zip(static, incoming):
            _copy_inputs(static_item, incoming_item)
        return

    if isinstance(static, list):
        assert isinstance(incoming, list), "incoming value must be a list"
        assert len(static) == len(incoming), (
            f"list length mismatch: expected {len(static)}, got {len(incoming)}"
        )
        for static_item, incoming_item in zip(static, incoming):
            _copy_inputs(static_item, incoming_item)
        return

    if isinstance(static, dict):
        assert isinstance(incoming, dict), "incoming value must be a dict"
        assert set(static) == set(incoming), (
            f"dict keys mismatch: expected {sorted(static)}, got {sorted(incoming)}"
        )
        for key, static_item in static.items():
            _copy_inputs(static_item, incoming[key])
        return

    assert static == incoming, f"non-tensor graph argument changed: {static!r} != {incoming!r}"


def _first_cuda_device(value: Any) -> torch.device | None:
    if isinstance(value, torch.Tensor):
        return value.device if value.device.type == "cuda" else None
    if isinstance(value, (tuple, list)):
        for item in value:
            device = _first_cuda_device(item)
            if device is not None:
                return device
        return None
    if isinstance(value, dict):
        for item in value.values():
            device = _first_cuda_device(item)
            if device is not None:
                return device
        return None
    return None


def _assert_same_cuda_device(value: Any, expected: torch.device) -> None:
    if isinstance(value, torch.Tensor):
        assert value.device == expected, f"expected tensor on {expected}, got {value.device}"
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            _assert_same_cuda_device(item, expected)
        return
    if isinstance(value, dict):
        for item in value.values():
            _assert_same_cuda_device(item, expected)
