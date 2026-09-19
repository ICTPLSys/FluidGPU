from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:
    from transformers.cache_utils import Cache as _TransformersCache
except Exception:
    _TransformersCache = object


@dataclass(frozen=True)
class CacheSlot:
    layer_id: int
    key: torch.Tensor
    value: torch.Tensor


class LayerKVCache(_TransformersCache):
    def __init__(
        self,
        num_layers: int,
        owned_layer_ids: list[int] | tuple[int, ...],
        max_seq_len: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device | str,
    ) -> None:
        try:
            super().__init__()
        except (TypeError, ValueError):
            pass
        self.num_layers = num_layers
        self.owned = set(owned_layer_ids)
        self.max_seq_len = max_seq_len
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self._cache_position = 0
        # Static mode makes update() CUDA-graph capturable: writes go through
        # index_copy_ with the device cache_position tensor (no .item() sync,
        # no Python-int slicing) and reads return the full max_seq_len views.
        self.static_mode = False
        self.key_cache: list[torch.Tensor | None] = [None] * num_layers
        self.value_cache: list[torch.Tensor | None] = [None] * num_layers

        for layer_id in owned_layer_ids:
            assert 0 <= layer_id < num_layers, f"invalid layer_id {layer_id}"
            key = torch.zeros(
                (1, num_kv_heads, max_seq_len, head_dim),
                dtype=dtype,
                device=self.device,
            )
            self.key_cache[layer_id] = key
            self.value_cache[layer_id] = torch.zeros_like(key)

    @property
    def cache_position(self) -> int:
        return self._cache_position

    def slot(self, layer_id: int) -> CacheSlot | None:
        if layer_id not in self.owned:
            return None
        key = self.key_cache[layer_id]
        value = self.value_cache[layer_id]
        assert key is not None and value is not None, f"missing cache slot for layer {layer_id}"
        return CacheSlot(layer_id=layer_id, key=key, value=value)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert layer_idx in self.owned, f"layer {layer_idx} not owned by this rank"
        key_cache = self.key_cache[layer_idx]
        value_cache = self.value_cache[layer_idx]
        assert key_cache is not None and value_cache is not None
        assert key_states.shape == value_states.shape, "key/value shapes must match"
        assert key_states.shape[0] == 1, f"batch must be 1, got {key_states.shape[0]}"
        assert key_states.shape[1] == self.num_kv_heads, (
            f"num_kv_heads mismatch: got {key_states.shape[1]}, expected {self.num_kv_heads}"
        )
        assert key_states.shape[-1] == self.head_dim, (
            f"head_dim mismatch: got {key_states.shape[-1]}, expected {self.head_dim}"
        )

        seq_len = key_states.shape[-2]
        if self.static_mode:
            cache_position = None if cache_kwargs is None else cache_kwargs.get("cache_position")
            if cache_position is None:
                cache_position = torch.arange(
                    self._cache_position,
                    self._cache_position + seq_len,
                    device=self.device,
                )
            index = cache_position.flatten().to(device=self.device, dtype=torch.long)
            key_cache.index_copy_(2, index, key_states)
            value_cache.index_copy_(2, index, value_states)
            return key_cache, value_cache

        start = self._start_position(cache_kwargs)
        end = start + seq_len
        assert end <= self.max_seq_len, f"exceed max_seq_len: {end} > {self.max_seq_len}"

        key_cache[:, :, start:end, :] = key_states
        value_cache[:, :, start:end, :] = value_states
        return key_cache[:, :, :end, :], value_cache[:, :, :end, :]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._cache_position

    def get_max_length(self) -> int:
        return self.max_seq_len

    def get_max_cache_shape(self) -> int:
        return self.max_seq_len

    def get_mask_sizes(
        self,
        cache_position: torch.Tensor,
        layer_idx: int | None = None,
    ) -> tuple[int, int]:
        assert cache_position.numel() > 0, "cache_position must not be empty"
        if self.static_mode:
            # Full-length KV views; .item() would also break CUDA-graph capture.
            return self.max_seq_len, 0
        kv_length = int(cache_position.flatten()[-1].item()) + 1
        assert kv_length <= self.max_seq_len, f"exceed max_seq_len: {kv_length} > {self.max_seq_len}"
        return kv_length, 0

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        if self._cache_position + new_seq_length <= self.max_seq_len:
            return self._cache_position
        return self.max_seq_len - new_seq_length

    def advance(self, seq_len: int) -> None:
        assert seq_len > 0, f"seq_len must be positive, got {seq_len}"
        next_pos = self._cache_position + seq_len
        assert next_pos <= self.max_seq_len, f"exceed max_seq_len: {next_pos} > {self.max_seq_len}"
        self._cache_position = next_pos

    def reset(self) -> None:
        self._cache_position = 0
        for key, value in zip(self.key_cache, self.value_cache):
            if key is not None:
                key.zero_()
                assert value is not None
                value.zero_()

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        raise NotImplementedError("beam search is not supported in fluidgpu_torch v1")

    def _start_position(self, cache_kwargs: dict[str, Any] | None) -> int:
        if cache_kwargs is None:
            return self._cache_position
        cache_position = cache_kwargs.get("cache_position")
        if cache_position is None:
            return self._cache_position
        assert cache_position.numel() > 0, "cache_position must not be empty"
        return int(cache_position.flatten()[0].item())
