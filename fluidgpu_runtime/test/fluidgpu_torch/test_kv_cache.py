import torch

from fluidgpu_torch.kv_cache import LayerKVCache


def test_kv_cache_prefill_then_decode_cpu():
    cache = LayerKVCache(
        num_layers=4,
        owned_layer_ids=[0, 1],
        max_seq_len=16,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.bfloat16,
        device="cpu",
    )

    key = torch.randn(1, 2, 4, 8, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    out_key, out_value = cache.update(key, value, layer_idx=0)
    assert out_key.shape == (1, 2, 4, 8)
    assert out_value.shape == (1, 2, 4, 8)
    cache.advance(4)
    assert cache.cache_position == 4

    key_one = torch.randn(1, 2, 1, 8, dtype=torch.bfloat16)
    out_key, _ = cache.update(key_one, torch.randn_like(key_one), layer_idx=0)
    assert out_key.shape == (1, 2, 5, 8)


def test_kv_cache_non_owned_raises():
    cache = LayerKVCache(
        num_layers=4,
        owned_layer_ids=[0],
        max_seq_len=16,
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.bfloat16,
        device="cpu",
    )
    key = torch.randn(1, 2, 1, 8, dtype=torch.bfloat16)
    try:
        cache.update(key, torch.randn_like(key), layer_idx=2)
    except AssertionError as exc:
        assert "not owned" in str(exc)
    else:
        raise AssertionError("expected non-owned layer update to fail")


def test_slot_returns_none_for_non_owned_layer():
    cache = LayerKVCache(
        num_layers=2,
        owned_layer_ids=[1],
        max_seq_len=8,
        num_kv_heads=1,
        head_dim=4,
        dtype=torch.bfloat16,
        device="cpu",
    )
    assert cache.slot(0) is None
    assert cache.slot(1) is not None
