from __future__ import annotations

import importlib
import inspect
from functools import lru_cache
from typing import Any

import torch

from .config import EngineConfig
from .executor import LayerExecutor, _first_tensor
from .kv_cache import LayerKVCache


def extract_components(hf_model: Any) -> dict[str, Any]:
    model = hf_model.model
    return {
        "embed_tokens": model.embed_tokens,
        "layers": list(model.layers),
        "norm": model.norm,
        "lm_head": hf_model.lm_head,
        "rotary_emb": getattr(model, "rotary_emb", None),
    }


class CausalLMRunner:
    def __init__(
        self,
        hf_model: Any,
        tokenizer: Any,
        executor: LayerExecutor,
        cfg: EngineConfig,
        owned_layer_ids: list[int],
    ) -> None:
        components = extract_components(hf_model)
        self.embed_tokens = components["embed_tokens"]
        self.layers = components["layers"]
        self.norm = components["norm"]
        self.lm_head = components["lm_head"]
        self.rotary_emb = components["rotary_emb"]
        self.tokenizer = tokenizer
        self.executor = executor
        self.cfg = cfg
        self.device = cfg.torch_device
        self.hidden_size = hf_model.config.hidden_size
        self.hf_config = hf_model.config
        self._attn_states: dict[int, dict[str, torch.Tensor]] = {}
        self._dense_mlp_states: dict[int, dict[str, torch.Tensor]] = {}
        self._moe_states: dict[int, dict[str, torch.Tensor]] = {}
        head_dim = getattr(
            hf_model.config,
            "head_dim",
            hf_model.config.hidden_size // hf_model.config.num_attention_heads,
        )
        self.kv_cache = LayerKVCache(
            num_layers=hf_model.config.num_hidden_layers,
            owned_layer_ids=owned_layer_ids,
            max_seq_len=cfg.max_seq_len,
            num_kv_heads=hf_model.config.num_key_value_heads,
            head_dim=head_dim,
            dtype=cfg.dtype,
            device=self.device,
        )

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor | None:
        assert input_ids.ndim == 2 and input_ids.shape[0] == 1, "input_ids must be [1, seq]"
        seq_len = int(input_ids.shape[1])
        assert seq_len <= self.cfg.max_seq_len, (
            f"prompt length {seq_len} exceeds max_seq_len {self.cfg.max_seq_len}"
        )
        self.kv_cache.reset()
        past_seen = 0
        position_ids = self._position_ids(past_seen, seq_len)
        cache_position = torch.arange(past_seen, past_seen + seq_len, device=self.device)
        position_embeddings = self._position_embeddings(position_ids, seq_len)
        attention_mask = self._causal_mask(seq_len=seq_len, past_seen=past_seen)

        hidden = self.executor.run_stage(
            "embed",
            self.embed_tokens,
            input_ids,
            mode="prefill",
            seq_len=seq_len,
        )
        for i, layer in enumerate(self.layers):
            hidden = self._run_attention_tasks(
                layer_id=i,
                layer=layer,
                hidden=hidden,
                mode="prefill",
                seq_len=seq_len,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=self.kv_cache if i in self.kv_cache.owned else None,
                use_cache=True,
                cache_position=cache_position,
                **self._position_embedding_kwargs(position_embeddings),
            )

            hidden = self._run_feed_forward_tasks(
                layer_id=i,
                layer=layer,
                hidden=hidden,
                mode="prefill",
                seq_len=seq_len,
            )

        self.kv_cache.advance(seq_len)
        return self.executor.run_stage(
            "norm_lm_head",
            self._norm_head,
            hidden,
            mode="prefill",
            seq_len=seq_len,
        )

    @torch.inference_mode()
    def decode_step(self, token_id: torch.Tensor) -> torch.Tensor | None:
        assert token_id.shape == (1, 1), f"decode token must be [1, 1], got {tuple(token_id.shape)}"
        past_seen = self.kv_cache.get_seq_length()
        seq_len = 1
        position_ids = self._position_ids(past_seen, seq_len)
        cache_position = torch.tensor([past_seen], device=self.device, dtype=torch.long)
        position_embeddings = self._position_embeddings(position_ids, seq_len)
        attention_mask = self._causal_mask(seq_len=seq_len, past_seen=past_seen)

        hidden = self.executor.run_stage(
            "embed",
            self.embed_tokens,
            token_id,
            mode="decode",
            seq_len=seq_len,
        )
        for i, layer in enumerate(self.layers):
            hidden = self._run_attention_tasks(
                layer_id=i,
                layer=layer,
                hidden=hidden,
                mode="decode",
                seq_len=seq_len,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=self.kv_cache if i in self.kv_cache.owned else None,
                use_cache=True,
                cache_position=cache_position,
                **self._position_embedding_kwargs(position_embeddings),
            )

            hidden = self._run_feed_forward_tasks(
                layer_id=i,
                layer=layer,
                hidden=hidden,
                mode="decode",
                seq_len=seq_len,
            )

        self.kv_cache.advance(1)
        return self.executor.run_stage(
            "norm_lm_head",
            self._norm_head,
            hidden,
            mode="decode",
            seq_len=seq_len,
        )

    def _norm_head(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(hidden[:, -1:, :]))

    def _run_attn(
        self,
        hidden: torch.Tensor,
        *,
        layer: Any,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: LayerKVCache | None,
        use_cache: bool,
        cache_position: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        residual = hidden
        hidden = layer.input_layernorm(hidden)
        kwargs = self._attention_kwargs(
            layer=layer,
            hidden=hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        return residual + _first_tensor(layer.self_attn(hidden, **kwargs))

    def _run_attention_tasks(
        self,
        *,
        layer_id: int,
        layer: Any,
        hidden: torch.Tensor | None,
        mode: str,
        seq_len: int,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: LayerKVCache | None,
        use_cache: bool,
        cache_position: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        if not self._has_task(f"layer_{layer_id}_qkv"):
            out = self.executor.run_kernel_group(
                f"layer_{layer_id}_attn",
                self._run_attn,
                hidden,
                mode=mode,
                seq_len=seq_len,
                layer=layer,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            return out if out is not None else hidden

        for suffix, fn in (
            ("qkv", self._run_attn_qkv),
            ("sdpa", self._run_attn_sdpa),
            ("o_proj", self._run_attn_o_proj),
        ):
            out = self.executor.run_kernel_group(
                f"layer_{layer_id}_{suffix}",
                fn,
                hidden,
                mode=mode,
                seq_len=seq_len,
                layer=layer,
                layer_id=layer_id,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            if out is not None:
                hidden = out
                if suffix == "qkv":
                    self._after_attn_qkv(layer_id)
                elif suffix == "sdpa":
                    self._after_attn_sdpa(layer_id)
        return hidden

    def _run_attn_qkv(
        self,
        hidden: torch.Tensor,
        *,
        layer: Any,
        layer_id: int,
        past_key_value: LayerKVCache | None,
        cache_position: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        **_: Any,
    ) -> torch.Tensor:
        residual = hidden
        hidden = layer.input_layernorm(hidden)
        attn = attention_module(layer)
        input_shape = hidden.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)

        query_states = attn.q_proj(hidden).view(hidden_shape).transpose(1, 2)
        key_states = attn.k_proj(hidden).view(hidden_shape).transpose(1, 2)
        value_states = attn.v_proj(hidden).view(hidden_shape).transpose(1, 2)
        query_states, key_states = _apply_rotary_pos_emb(attn, query_states, key_states, position_embeddings)

        if past_key_value is not None:
            cache_kwargs = {
                "cache_position": cache_position,
                "sin": position_embeddings[1],
                "cos": position_embeddings[0],
            }
            key_states, value_states = past_key_value.update(
                key_states,
                value_states,
                attn.layer_idx,
                cache_kwargs,
            )

        self._attn_states[layer_id] = {
            "residual": residual,
            "query_states": query_states,
            "key_states": key_states,
            "value_states": value_states,
        }
        return hidden

    def _run_attn_sdpa(
        self,
        hidden: torch.Tensor,
        *,
        layer: Any,
        layer_id: int,
        attention_mask: torch.Tensor,
        cache_position: torch.Tensor,
        **_: Any,
    ) -> torch.Tensor:
        self._ensure_attn_qkv_state(
            layer_id,
            layer=layer,
            hidden=hidden,
            cache_position=cache_position,
        )
        state = self._attn_states[layer_id]
        attn = attention_module(layer)
        attention_mask = self._attention_mask_for_layer(
            layer=layer,
            hidden=hidden,
            fallback_mask=attention_mask,
            cache_position=cache_position,
            past_key_value=self.kv_cache,
        )
        attention_interface = _attention_interface(attn)
        attn_output, _ = attention_interface(
            attn,
            state["query_states"],
            state["key_states"],
            state["value_states"],
            attention_mask,
            dropout=0.0 if not attn.training else attn.attention_dropout,
            scaling=attn.scaling,
            **_attention_extra_kwargs(attn),
        )
        return attn_output.reshape(*hidden.shape[:-1], -1).contiguous()

    def _run_attn_o_proj(
        self,
        hidden: torch.Tensor,
        *,
        layer: Any,
        layer_id: int,
        **_: Any,
    ) -> torch.Tensor:
        self._ensure_attn_residual(layer_id, hidden=hidden)
        state = self._attn_states.pop(layer_id)
        return state["residual"] + attention_module(layer).o_proj(hidden)

    def _run_mlp(self, hidden: torch.Tensor, *, layer: Any) -> torch.Tensor:
        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        return residual + run_feed_forward(layer, hidden)

    def _run_feed_forward_tasks(
        self,
        *,
        layer_id: int,
        layer: Any,
        hidden: torch.Tensor | None,
        mode: str,
        seq_len: int,
    ) -> torch.Tensor | None:
        if self._has_task(f"layer_{layer_id}_mlp_gate_up"):
            for suffix, fn in (
                ("mlp_gate_up", self._run_mlp_gate_up),
                ("mlp_down_proj", self._run_mlp_down_proj),
            ):
                out = self.executor.run_kernel_group(
                    f"layer_{layer_id}_{suffix}",
                    fn,
                    hidden,
                    mode=mode,
                    seq_len=seq_len,
                    layer=layer,
                    layer_id=layer_id,
                )
                if out is not None:
                    hidden = out
                    if suffix == "mlp_gate_up":
                        self._after_mlp_gate_up(layer_id)
            return hidden

        if not self._has_task(f"layer_{layer_id}_moe_router"):
            out = self.executor.run_kernel_group(
                f"layer_{layer_id}_mlp",
                self._run_mlp,
                hidden,
                mode=mode,
                seq_len=seq_len,
                layer=layer,
            )
            return out if out is not None else hidden

        for suffix, fn in (
            ("moe_router", self._run_moe_router),
            ("moe_experts", self._run_moe_experts),
            ("moe_combine", self._run_moe_combine),
        ):
            out = self.executor.run_kernel_group(
                f"layer_{layer_id}_{suffix}",
                fn,
                hidden,
                mode=mode,
                seq_len=seq_len,
                layer=layer,
                layer_id=layer_id,
            )
            if out is not None:
                hidden = out
                if suffix == "moe_router":
                    self._after_moe_router(layer_id)
                elif suffix == "moe_experts":
                    self._after_moe_experts(layer_id)
        return hidden

    def _run_mlp_gate_up(self, hidden: torch.Tensor, *, layer: Any, layer_id: int) -> torch.Tensor:
        residual = hidden
        hidden = layer.post_attention_layernorm(hidden)
        module = dense_mlp_module(layer)
        intermediate = module.act_fn(module.gate_proj(hidden)) * module.up_proj(hidden)
        self._dense_mlp_states[layer_id] = {
            "residual": residual,
            "intermediate": intermediate,
        }
        return hidden

    def _run_mlp_down_proj(self, hidden: torch.Tensor, *, layer: Any, layer_id: int) -> torch.Tensor:
        self._ensure_dense_mlp_state(layer_id, layer=layer, hidden=hidden)
        state = self._dense_mlp_states.pop(layer_id)
        return state["residual"] + dense_mlp_module(layer).down_proj(state["intermediate"])

    def _run_moe_router(self, hidden: torch.Tensor, *, layer: Any, layer_id: int) -> torch.Tensor:
        residual = hidden
        normalized = layer.post_attention_layernorm(hidden)
        module = moe_module(layer)
        router_scores, router_indices = module.router(normalized)
        self._moe_states[layer_id] = {
            "residual": residual,
            "router_scores": router_scores,
            "router_indices": router_indices,
        }
        return normalized

    def _run_moe_experts(self, hidden: torch.Tensor, *, layer: Any, layer_id: int) -> torch.Tensor:
        self._ensure_moe_router_state(layer_id, layer=layer, hidden=hidden)
        state = self._moe_states[layer_id]
        module = moe_module(layer)
        return module.experts(
            hidden,
            router_indices=state["router_indices"],
            routing_weights=state["router_scores"],
        )

    def _run_moe_combine(self, hidden: torch.Tensor, *, layer: Any, layer_id: int) -> torch.Tensor:
        self._ensure_moe_residual(layer_id, hidden=hidden)
        state = self._moe_states.pop(layer_id)
        return state["residual"] + hidden

    def _has_task(self, task_name: str) -> bool:
        return task_name in self.executor.task_index_by_name

    def _after_attn_qkv(self, layer_id: int) -> None:
        ranks = self._attn_task_ranks(layer_id)
        if ranks["sdpa"] == ranks["qkv"]:
            return

        state = self._attn_states.pop(layer_id)
        dst = ranks["sdpa"]
        self._send_aux_tensor(state["query_states"], dst=dst)
        self._send_aux_tensor(state["key_states"], dst=dst)
        self._send_aux_tensor(state["value_states"], dst=dst)
        self._send_aux_tensor(state["residual"], dst=dst)

    def _after_attn_sdpa(self, layer_id: int) -> None:
        ranks = self._attn_task_ranks(layer_id)
        state = self._attn_states.get(layer_id)
        if state is None:
            return

        state.pop("query_states", None)
        state.pop("key_states", None)
        state.pop("value_states", None)
        if ranks["o_proj"] == ranks["sdpa"]:
            return

        self._send_aux_tensor(state["residual"], dst=ranks["o_proj"])
        self._attn_states.pop(layer_id, None)

    def _ensure_attn_qkv_state(
        self,
        layer_id: int,
        *,
        layer: Any,
        hidden: torch.Tensor,
        cache_position: torch.Tensor,
    ) -> None:
        state = self._attn_states.setdefault(layer_id, {})
        if {"query_states", "key_states", "value_states"}.issubset(state):
            return

        ranks = self._attn_task_ranks(layer_id)
        attn = attention_module(layer)
        batch_size = int(hidden.shape[0])
        query_len = int(hidden.shape[1])
        kv_len = int(cache_position.flatten()[-1].item()) + 1
        head_dim = int(attn.head_dim)
        state["query_states"] = self._recv_aux_tensor(
            (batch_size, _num_attention_heads(attn, self.hf_config), query_len, head_dim),
            dtype=hidden.dtype,
            src=ranks["qkv"],
        )
        state["key_states"] = self._recv_aux_tensor(
            (batch_size, _num_kv_heads(attn, self.hf_config), kv_len, head_dim),
            dtype=hidden.dtype,
            src=ranks["qkv"],
        )
        state["value_states"] = self._recv_aux_tensor(
            (batch_size, _num_kv_heads(attn, self.hf_config), kv_len, head_dim),
            dtype=hidden.dtype,
            src=ranks["qkv"],
        )
        state["residual"] = self._recv_aux_tensor(
            tuple(hidden.shape),
            dtype=hidden.dtype,
            src=ranks["qkv"],
        )

    def _ensure_attn_residual(self, layer_id: int, *, hidden: torch.Tensor) -> None:
        state = self._attn_states.setdefault(layer_id, {})
        if "residual" in state:
            return

        ranks = self._attn_task_ranks(layer_id)
        state["residual"] = self._recv_aux_tensor(
            tuple(hidden.shape),
            dtype=hidden.dtype,
            src=ranks["sdpa"],
        )

    def _attn_task_ranks(self, layer_id: int) -> dict[str, int]:
        by_name = self.executor.profile.task_by_name
        return {
            "qkv": by_name[f"layer_{layer_id}_qkv"].rank,
            "sdpa": by_name[f"layer_{layer_id}_sdpa"].rank,
            "o_proj": by_name[f"layer_{layer_id}_o_proj"].rank,
        }

    def _after_mlp_gate_up(self, layer_id: int) -> None:
        ranks = self._dense_mlp_task_ranks(layer_id)
        if ranks["down_proj"] == ranks["gate_up"]:
            return

        state = self._dense_mlp_states.pop(layer_id)
        dst = ranks["down_proj"]
        self._send_aux_tensor(state["intermediate"], dst=dst)
        self._send_aux_tensor(state["residual"], dst=dst)

    def _ensure_dense_mlp_state(
        self,
        layer_id: int,
        *,
        layer: Any,
        hidden: torch.Tensor,
    ) -> None:
        state = self._dense_mlp_states.setdefault(layer_id, {})
        if "intermediate" in state and "residual" in state:
            return

        ranks = self._dense_mlp_task_ranks(layer_id)
        intermediate_size = _mlp_intermediate_size(dense_mlp_module(layer))
        state["intermediate"] = self._recv_aux_tensor(
            (*hidden.shape[:-1], intermediate_size),
            dtype=hidden.dtype,
            src=ranks["gate_up"],
        )
        state["residual"] = self._recv_aux_tensor(
            tuple(hidden.shape),
            dtype=hidden.dtype,
            src=ranks["gate_up"],
        )

    def _dense_mlp_task_ranks(self, layer_id: int) -> dict[str, int]:
        by_name = self.executor.profile.task_by_name
        return {
            "gate_up": by_name[f"layer_{layer_id}_mlp_gate_up"].rank,
            "down_proj": by_name[f"layer_{layer_id}_mlp_down_proj"].rank,
        }

    def _after_moe_router(self, layer_id: int) -> None:
        ranks = self._moe_task_ranks(layer_id)
        if ranks["experts"] == ranks["router"]:
            return

        state = self._moe_states.pop(layer_id)
        dst = ranks["experts"]
        self._send_aux_tensor(state["router_scores"], dst=dst)
        self._send_aux_tensor(state["router_indices"], dst=dst)
        self._send_aux_tensor(state["residual"], dst=dst)

    def _after_moe_experts(self, layer_id: int) -> None:
        ranks = self._moe_task_ranks(layer_id)
        state = self._moe_states.get(layer_id)
        if state is None:
            return

        state.pop("router_scores", None)
        state.pop("router_indices", None)
        if ranks["combine"] == ranks["experts"]:
            return

        self._send_aux_tensor(state["residual"], dst=ranks["combine"])
        self._moe_states.pop(layer_id, None)

    def _ensure_moe_router_state(self, layer_id: int, *, layer: Any, hidden: torch.Tensor) -> None:
        state = self._moe_states.setdefault(layer_id, {})
        if "router_scores" in state and "router_indices" in state:
            return

        ranks = self._moe_task_ranks(layer_id)
        module = moe_module(layer)
        num_tokens = int(hidden.shape[0] * hidden.shape[1])
        num_experts = _moe_num_experts(module, self.hf_config)
        top_k = _moe_top_k(module, self.hf_config)
        state["router_scores"] = self._recv_aux_tensor(
            (num_tokens, num_experts),
            dtype=hidden.dtype,
            src=ranks["router"],
        )
        state["router_indices"] = self._recv_aux_tensor(
            (num_tokens, top_k),
            dtype=torch.long,
            src=ranks["router"],
        )
        state["residual"] = self._recv_aux_tensor(
            tuple(hidden.shape),
            dtype=hidden.dtype,
            src=ranks["router"],
        )

    def _ensure_moe_residual(self, layer_id: int, *, hidden: torch.Tensor) -> None:
        state = self._moe_states.setdefault(layer_id, {})
        if "residual" in state:
            return

        ranks = self._moe_task_ranks(layer_id)
        state["residual"] = self._recv_aux_tensor(
            tuple(hidden.shape),
            dtype=hidden.dtype,
            src=ranks["experts"],
        )

    def _moe_task_ranks(self, layer_id: int) -> dict[str, int]:
        by_name = self.executor.profile.task_by_name
        return {
            "router": by_name[f"layer_{layer_id}_moe_router"].rank,
            "experts": by_name[f"layer_{layer_id}_moe_experts"].rank,
            "combine": by_name[f"layer_{layer_id}_moe_combine"].rank,
        }

    def _send_aux_tensor(self, tensor: torch.Tensor, *, dst: int) -> None:
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        self.executor.send_aux_tensor(tensor, dst=dst)

    def _recv_aux_tensor(
        self,
        shape: tuple[int, ...],
        *,
        dtype: torch.dtype,
        src: int,
    ) -> torch.Tensor:
        out = torch.empty(shape, dtype=dtype, device=self.device)
        self.executor.recv_aux_tensor(out, src=src)
        return out

    def _attention_kwargs(
        self,
        *,
        layer: Any,
        hidden: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: LayerKVCache | None,
        use_cache: bool,
        cache_position: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> dict[str, Any]:
        attn = layer.self_attn
        params = _forward_parameter_names(attn)
        attention_mask = self._attention_mask_for_layer(
            layer=layer,
            hidden=hidden,
            fallback_mask=attention_mask,
            cache_position=cache_position,
            past_key_value=past_key_value,
        )
        kwargs: dict[str, Any] = {
            "attention_mask": attention_mask,
            "cache_position": cache_position,
        }
        if "position_ids" in params:
            kwargs["position_ids"] = position_ids
        if position_embeddings is not None:
            kwargs["position_embeddings"] = position_embeddings
        if "past_key_values" in params:
            kwargs["past_key_values"] = past_key_value
        else:
            kwargs["past_key_value"] = past_key_value
        if "use_cache" in params:
            kwargs["use_cache"] = use_cache
        return kwargs

    def _attention_mask_for_layer(
        self,
        *,
        layer: Any,
        hidden: torch.Tensor,
        fallback_mask: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_value: LayerKVCache | None,
    ) -> torch.Tensor | None:
        attention_type = getattr(layer, "attention_type", None)
        if attention_type not in ("full_attention", "sliding_attention"):
            return fallback_mask
        if past_key_value is not None and past_key_value.static_mode:
            # transformers' create_causal_mask may return None (sdpa is_causal
            # fast path) once kv_length == max_seq_len; with full-length KV
            # views that would attend over the unwritten zero tail. The
            # explicit full-width mask is required — and CUDA-graph safe.
            if attention_type == "sliding_attention":
                # ...but the fallback is plain causal, and no attention
                # implementation applies the window itself: gpt-oss's
                # eager_attention_forward and sdpa_attention_forward both
                # swallow the sliding_window kwarg through **kwargs and read
                # the window only from the mask. Without this the layer
                # silently attends over its whole history.
                return self._sliding_window_mask(fallback_mask, layer, cache_position)
            return fallback_mask

        from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

        mask_fn = (
            create_sliding_window_causal_mask
            if attention_type == "sliding_attention"
            else create_causal_mask
        )
        return mask_fn(
            config=self.hf_config,
            input_embeds=hidden,
            attention_mask=None,
            cache_position=cache_position,
            past_key_values=past_key_value,
        )

    def _sliding_window_mask(
        self,
        causal_mask: torch.Tensor,
        layer: Any,
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        """Overlay transformers' sliding-window pattern (keep kv_idx >
        q_idx - window) onto an already-causal additive mask.

        Derives the query positions from `cache_position` rather than a host
        int, so this stays CUDA-graph safe: under capture the positions live
        in the graph's device buffer and the overlay is recomputed on every
        replay.
        """
        window = self._sliding_window_size(layer)
        if not window:
            return causal_mask
        kv_positions = torch.arange(causal_mask.shape[-1], device=causal_mask.device)
        outside = kv_positions.unsqueeze(0) <= (cache_position.unsqueeze(1) - window)
        return causal_mask.masked_fill(outside, torch.finfo(causal_mask.dtype).min)

    def _sliding_window_size(self, layer: Any) -> int | None:
        # The attention module carries the per-layer value (None on the
        # full-attention layers of an interleaved model); the config is the
        # fallback for models that only declare it globally.
        for owner in (attention_module(layer), self.hf_config):
            value = getattr(owner, "sliding_window", None)
            if value:
                return int(value)
        return None

    def _position_ids(self, past_seen: int, seq_len: int) -> torch.Tensor:
        return torch.arange(past_seen, past_seen + seq_len, device=self.device).unsqueeze(0)

    def _position_embeddings(
        self,
        position_ids: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if self.rotary_emb is None:
            return None
        dummy = torch.empty(
            (1, seq_len, self.hidden_size),
            dtype=self.cfg.dtype,
            device=self.device,
        )
        return self.rotary_emb(dummy, position_ids)

    def _position_embedding_kwargs(
        self,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> dict[str, Any]:
        if position_embeddings is None:
            return {}
        return {"position_embeddings": position_embeddings}

    def _causal_mask(self, *, seq_len: int, past_seen: int) -> torch.Tensor:
        # In static (CUDA-graph) mode the KV cache returns full max_seq_len
        # views, so the mask spans max_seq_len; positions beyond the query are
        # masked either way, which also hides the unwritten cache tail.
        total_len = (
            self.kv_cache.max_seq_len if self.kv_cache.static_mode else past_seen + seq_len
        )
        q_positions = torch.arange(past_seen, past_seen + seq_len, device=self.device)
        kv_positions = torch.arange(total_len, device=self.device)
        masked = kv_positions.unsqueeze(0) > q_positions.unsqueeze(1)
        mask = torch.zeros((seq_len, total_len), dtype=self.cfg.dtype, device=self.device)
        mask = mask.masked_fill(masked, torch.finfo(self.cfg.dtype).min)
        return mask[None, None, :, :]


def _forward_parameter_names(module: Any) -> set[str]:
    return _forward_parameter_names_for_type(type(module))


@lru_cache(maxsize=None)
def _forward_parameter_names_for_type(module_type: type) -> set[str]:
    signature = inspect.signature(module_type.forward)
    return set(signature.parameters)


def run_feed_forward(layer: Any, hidden: torch.Tensor) -> torch.Tensor:
    if hasattr(layer, "mlp"):
        return _first_tensor(layer.mlp(hidden))
    if hasattr(layer, "block_sparse_moe"):
        return _first_tensor(layer.block_sparse_moe(hidden))
    raise AssertionError(f"layer {type(layer).__name__} has no supported feed-forward module")


def attention_module(layer: Any) -> Any:
    module = getattr(layer, "self_attn", None)
    assert module is not None, f"layer {type(layer).__name__} has no self_attn module"
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        assert hasattr(module, name), f"attention module {type(module).__name__} missing {name}"
    assert hasattr(module, "head_dim"), f"attention module {type(module).__name__} missing head_dim"
    return module


def dense_mlp_module(layer: Any) -> Any:
    module = getattr(layer, "mlp", None)
    assert module is not None, f"layer {type(layer).__name__} has no mlp module"
    for name in ("gate_proj", "up_proj", "down_proj", "act_fn"):
        assert hasattr(module, name), f"dense MLP module {type(module).__name__} missing {name}"
    return module


def moe_module(layer: Any) -> Any:
    module = getattr(layer, "mlp", None)
    assert module is not None, f"layer {type(layer).__name__} has no mlp module"
    assert hasattr(module, "router") and hasattr(module, "experts"), (
        f"layer {type(layer).__name__} does not expose split MoE router/experts"
    )
    return module


def _moe_num_experts(module: Any, config: Any) -> int:
    for owner in (getattr(module, "router", None), getattr(module, "experts", None), module, config):
        if owner is None:
            continue
        for name in ("num_experts", "num_local_experts", "n_experts"):
            value = getattr(owner, name, None)
            if value is not None:
                return int(value)
    raise AssertionError(f"cannot infer MoE num_experts for {type(module).__name__}")


def _moe_top_k(module: Any, config: Any) -> int:
    for owner in (getattr(module, "router", None), module, config):
        if owner is None:
            continue
        for name in ("top_k", "num_experts_per_tok", "experts_per_token", "router_top_k"):
            value = getattr(owner, name, None)
            if value is not None:
                return int(value)
    raise AssertionError(f"cannot infer MoE top_k for {type(module).__name__}")


def _apply_rotary_pos_emb(
    attn: Any,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert position_embeddings is not None, "fine attention split requires position_embeddings"
    module = importlib.import_module(type(attn).__module__)
    apply_rotary_pos_emb = getattr(module, "apply_rotary_pos_emb")
    cos, sin = position_embeddings
    return apply_rotary_pos_emb(query_states, key_states, cos, sin)


def _attention_interface(attn: Any) -> Any:
    module = importlib.import_module(type(attn).__module__)
    interface = getattr(module, "eager_attention_forward")
    implementation = getattr(getattr(attn, "config", None), "_attn_implementation", "eager")
    if implementation != "eager":
        interface = getattr(module, "ALL_ATTENTION_FUNCTIONS")[implementation]
    return interface


def _attention_extra_kwargs(attn: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if hasattr(attn, "sliding_window"):
        kwargs["sliding_window"] = attn.sliding_window
    if hasattr(attn, "sinks"):
        kwargs["s_aux"] = attn.sinks
    return kwargs


def _num_attention_heads(attn: Any, config: Any) -> int:
    for owner in (attn, config):
        for name in ("num_heads", "num_attention_heads", "n_heads"):
            value = getattr(owner, name, None)
            if value is not None:
                return int(value)
    raise AssertionError(f"cannot infer attention heads for {type(attn).__name__}")


def _num_kv_heads(attn: Any, config: Any) -> int:
    for owner in (attn, config):
        for name in ("num_key_value_heads", "num_kv_heads", "n_kv_heads"):
            value = getattr(owner, name, None)
            if value is not None:
                return int(value)
    raise AssertionError(f"cannot infer KV heads for {type(attn).__name__}")


def _mlp_intermediate_size(module: Any) -> int:
    for projection_name in ("up_proj", "gate_proj"):
        projection = getattr(module, projection_name, None)
        value = getattr(projection, "out_features", None)
        if value is not None:
            return int(value)
    for name in ("intermediate_size", "ffn_dim"):
        value = getattr(module, name, None)
        if value is not None:
            return int(value)
    raise AssertionError(f"cannot infer dense MLP intermediate size for {type(module).__name__}")
