from __future__ import annotations

from types import SimpleNamespace

import torch

from fluidgpu_torch.runner import CausalLMRunner

WINDOW = 4
MAX_SEQ_LEN = 12


def make_runner() -> CausalLMRunner:
    # The mask helpers only touch hf_config; skip the (GPU-bound) constructor.
    runner = object.__new__(CausalLMRunner)
    runner.hf_config = SimpleNamespace(sliding_window=WINDOW)
    return runner


def make_layer(attention_type: str, *, window: int | None) -> SimpleNamespace:
    attn = SimpleNamespace(
        q_proj=None, k_proj=None, v_proj=None, o_proj=None,
        head_dim=4, sliding_window=window,
    )
    return SimpleNamespace(attention_type=attention_type, self_attn=attn)


def causal_mask(q_positions: list[int], kv_len: int) -> torch.Tensor:
    q = torch.tensor(q_positions).unsqueeze(1)
    kv = torch.arange(kv_len).unsqueeze(0)
    mask = torch.zeros((len(q_positions), kv_len), dtype=torch.float32)
    return mask.masked_fill(kv > q, torch.finfo(torch.float32).min)[None, None]


def reference_allowed(q_pos: int, kv_pos: int, window: int) -> bool:
    """Ground truth straight from transformers' own mask function."""
    from transformers.masking_utils import sliding_window_causal_mask_function

    fn = sliding_window_causal_mask_function(window)
    return bool(fn(0, 0, torch.tensor(q_pos), torch.tensor(kv_pos)))


def assert_matches_reference(mask: torch.Tensor, q_positions: list[int], window: int) -> None:
    allowed = mask[0, 0] == 0
    for i, q_pos in enumerate(q_positions):
        for kv_pos in range(mask.shape[-1]):
            assert bool(allowed[i, kv_pos]) is reference_allowed(q_pos, kv_pos, window), (
                f"q={q_pos} kv={kv_pos}: expected "
                f"{reference_allowed(q_pos, kv_pos, window)}"
            )


def test_sliding_window_mask_matches_transformers_semantics():
    # Decode step deep enough that the window has left history behind.
    q_positions = [9]
    runner = make_runner()
    mask = runner._sliding_window_mask(
        causal_mask(q_positions, MAX_SEQ_LEN),
        make_layer("sliding_attention", window=WINDOW),
        torch.tensor(q_positions),
    )
    assert_matches_reference(mask, q_positions, WINDOW)


def test_sliding_window_mask_over_a_prefill_block():
    q_positions = list(range(MAX_SEQ_LEN))
    runner = make_runner()
    mask = runner._sliding_window_mask(
        causal_mask(q_positions, MAX_SEQ_LEN),
        make_layer("sliding_attention", window=WINDOW),
        torch.tensor(q_positions),
    )
    assert_matches_reference(mask, q_positions, WINDOW)


def test_static_mode_applies_the_window_to_sliding_layers():
    # The regression: in static (CUDA-graph) mode every layer used to get the
    # plain causal fallback, so sliding layers attended over the whole history.
    q_positions = [9]
    fallback = causal_mask(q_positions, MAX_SEQ_LEN)
    runner = make_runner()
    static_cache = SimpleNamespace(static_mode=True)

    mask = runner._attention_mask_for_layer(
        layer=make_layer("sliding_attention", window=WINDOW),
        hidden=torch.zeros(1, len(q_positions), 8),
        fallback_mask=fallback,
        cache_position=torch.tensor(q_positions),
        past_key_value=static_cache,
    )

    assert_matches_reference(mask, q_positions, WINDOW)
    # Strictly narrower than the causal fallback it was handed.
    assert (mask[0, 0] == 0).sum() < (fallback[0, 0] == 0).sum()


def test_static_mode_leaves_full_attention_layers_alone():
    q_positions = [9]
    fallback = causal_mask(q_positions, MAX_SEQ_LEN)
    runner = make_runner()

    mask = runner._attention_mask_for_layer(
        layer=make_layer("full_attention", window=None),
        hidden=torch.zeros(1, len(q_positions), 8),
        fallback_mask=fallback,
        cache_position=torch.tensor(q_positions),
        past_key_value=SimpleNamespace(static_mode=True),
    )

    assert mask is fallback


def test_sliding_window_mask_is_a_noop_without_a_window():
    q_positions = [9]
    fallback = causal_mask(q_positions, MAX_SEQ_LEN)
    runner = object.__new__(CausalLMRunner)
    runner.hf_config = SimpleNamespace()  # model declares no window anywhere

    mask = runner._sliding_window_mask(
        fallback, make_layer("sliding_attention", window=None), torch.tensor(q_positions)
    )

    assert mask is fallback
