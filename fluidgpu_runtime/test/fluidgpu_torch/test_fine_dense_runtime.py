import json
from queue import Queue
from threading import Thread

import torch
import pytest

from fluidgpu_torch.config import EngineConfig
from fluidgpu_torch.executor import LayerExecutor
from fluidgpu_torch.profile import load_profile, make_ranked_task_profile, owned_layer_ids
from fluidgpu_torch.runner import CausalLMRunner


class QueueComm:
    def __init__(self, rank, channels):
        self._rank = rank
        self._channels = channels

    def get_rank(self):
        return self._rank

    def send_tensor(self, tensor, dst):
        self._channels[(self._rank, dst)].put(tensor.detach().clone())

    def recv_tensor(self, tensor, src):
        value = self._channels[(src, self._rank)].get(timeout=5)
        tensor.copy_(value)


def test_tiny_llama_fine_attention_and_mlp_prefill_matches_hf_forward(tmp_path):
    model, input_ids = tiny_llama()
    profile_path = write_profile(
        tmp_path,
        "single_rank_fine.json",
        [
            ("layer_0_qkv", 0),
            ("layer_0_sdpa", 0),
            ("layer_0_o_proj", 0),
            ("layer_0_mlp_gate_up", 0),
            ("layer_0_mlp_down_proj", 0),
        ],
    )
    profile = load_profile(profile_path, model_name="tiny-llama", hf_config=model.config)
    cfg = EngineConfig(
        model_name="tiny-llama",
        profiling_json=profile_path,
        max_seq_len=8,
        device="cpu",
    )
    runner = CausalLMRunner(
        model,
        tokenizer=None,
        executor=LayerExecutor(profile, cfg, comm_backend=QueueComm(0, {(0, 1): Queue(), (1, 0): Queue()})),
        cfg=cfg,
        owned_layer_ids=owned_layer_ids(profile, 0),
    )

    with torch.inference_mode():
        expected = model(input_ids, use_cache=True).logits[:, -1:, :]
        actual = runner.prefill(input_ids)

    assert actual is not None
    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_tiny_llama_split_attention_and_mlp_prefill_matches_hf_forward(tmp_path):
    model, input_ids = tiny_llama()
    profile_path = write_profile(
        tmp_path,
        "split_rank_fine.json",
        [
            ("layer_0_qkv", 0),
            ("layer_0_sdpa", 1),
            ("layer_0_o_proj", 0),
            ("layer_0_mlp_gate_up", 1),
            ("layer_0_mlp_down_proj", 0),
        ],
    )
    profile = load_profile(profile_path, model_name="tiny-llama", hf_config=model.config)
    cfg = EngineConfig(
        model_name="tiny-llama",
        profiling_json=profile_path,
        max_seq_len=8,
        device="cpu",
    )
    channels = {
        (0, 1): Queue(),
        (1, 0): Queue(),
    }
    runners = [
        CausalLMRunner(
            model,
            tokenizer=None,
            executor=LayerExecutor(profile, cfg, comm_backend=QueueComm(rank, channels)),
            cfg=cfg,
            owned_layer_ids=owned_layer_ids(profile, rank),
        )
        for rank in (0, 1)
    ]
    results = {}
    errors = []

    def run_rank(rank):
        try:
            results[rank] = runners[rank].prefill(input_ids)
        except BaseException as exc:
            errors.append(exc)

    threads = [Thread(target=run_rank, args=(rank,)) for rank in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not errors
    assert all(not thread.is_alive() for thread in threads)
    with torch.inference_mode():
        expected = model(input_ids, use_cache=True).logits[:, -1:, :]

    assert results[1] is None
    assert results[0] is not None
    assert torch.allclose(results[0], expected, atol=2e-2, rtol=2e-2)

    decode_results = {}
    decode_errors = []
    token_id = torch.tensor([[5]], dtype=torch.long)

    def decode_rank(rank):
        try:
            decode_results[rank] = runners[rank].decode_step(token_id)
        except BaseException as exc:
            decode_errors.append(exc)

    threads = [Thread(target=decode_rank, args=(rank,)) for rank in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert not decode_errors
    assert all(not thread.is_alive() for thread in threads)
    with torch.inference_mode():
        expected_decode = model(torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)).logits[:, -1:, :]

    assert decode_results[1] is None
    assert decode_results[0] is not None
    assert torch.allclose(decode_results[0], expected_decode, atol=2e-2, rtol=2e-2)


def tiny_llama():
    try:
        from transformers.models.llama import LlamaConfig, LlamaForCausalLM
    except Exception:
        pytest.skip("transformers Llama model is unavailable")

    torch.manual_seed(0)
    config = LlamaConfig(
        num_hidden_layers=1,
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )
    model = LlamaForCausalLM(config).to(dtype=torch.bfloat16)
    model.eval()
    return model, torch.tensor([[1, 2, 3, 4]], dtype=torch.long)


def write_profile(tmp_path, name, assignments):
    path = tmp_path / name
    raw = make_ranked_task_profile(
        model="tiny-llama",
        task_assignments=assignments,
        hidden_size=64,
        num_kv_heads=2,
        head_dim=16,
        max_seq_len=8,
    )
    path.write_text(json.dumps(raw))
    return path
