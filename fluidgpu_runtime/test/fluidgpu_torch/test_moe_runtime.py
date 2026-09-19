import json
from queue import Queue
from threading import Thread

import torch

from fluidgpu_torch.config import EngineConfig
from fluidgpu_torch.executor import LayerExecutor
from fluidgpu_torch.profile import load_profile, make_task_profile, owned_layer_ids
from fluidgpu_torch.runner import CausalLMRunner, run_feed_forward


class FakeComm:
    def get_rank(self):
        return 0

    def send_tensor(self, tensor, dst):
        raise AssertionError("single-rank test should not send")

    def recv_tensor(self, tensor, src):
        raise AssertionError("single-rank test should not recv")


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


def test_run_feed_forward_accepts_tuple_mlp():
    hidden = torch.ones(1, 2, 4)

    class Mlp:
        def __call__(self, value):
            return value + 1, torch.zeros(())

    class Layer:
        mlp = Mlp()

    out = run_feed_forward(Layer(), hidden)

    assert torch.equal(out, hidden + 1)


def test_run_feed_forward_accepts_block_sparse_moe():
    hidden = torch.ones(1, 2, 4)

    class Moe:
        def __call__(self, value):
            return value + 2, torch.zeros(())

    class Layer:
        block_sparse_moe = Moe()

    out = run_feed_forward(Layer(), hidden)

    assert torch.equal(out, hidden + 2)


def test_tiny_gpt_oss_prefill_matches_hf_forward(tmp_path):
    try:
        from transformers.models.gpt_oss import GptOssConfig, GptOssForCausalLM
    except Exception:
        return

    torch.manual_seed(0)
    config = GptOssConfig(
        num_hidden_layers=1,
        num_local_experts=2,
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        sliding_window=2,
        num_experts_per_tok=1,
        layer_types=["sliding_attention"],
    )
    model = GptOssForCausalLM(config).to(dtype=torch.bfloat16)
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    raw_profile = make_task_profile(
        model="tiny-gpt-oss",
        task_names=["layer_0_attn", "layer_0_mlp"],
        split_at=2,
        hidden_size=64,
        num_kv_heads=2,
        head_dim=16,
        max_seq_len=8,
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(raw_profile))
    profile = load_profile(profile_path, model_name="tiny-gpt-oss", hf_config=model.config)
    cfg = EngineConfig(
        model_name="tiny-gpt-oss",
        profiling_json=profile_path,
        max_seq_len=8,
        device="cpu",
    )
    runner = CausalLMRunner(
        model,
        tokenizer=None,
        executor=LayerExecutor(profile, cfg, comm_backend=FakeComm()),
        cfg=cfg,
        owned_layer_ids=owned_layer_ids(profile, 0),
    )

    with torch.inference_mode():
        expected = model(input_ids, use_cache=True).logits[:, -1:, :]
        actual = runner.prefill(input_ids)

    assert actual is not None
    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_tiny_gpt_oss_fine_moe_prefill_matches_hf_forward(tmp_path):
    try:
        from transformers.models.gpt_oss import GptOssConfig, GptOssForCausalLM
    except Exception:
        return

    torch.manual_seed(0)
    config = GptOssConfig(
        num_hidden_layers=1,
        num_local_experts=2,
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        sliding_window=2,
        num_experts_per_tok=1,
        layer_types=["sliding_attention"],
    )
    model = GptOssForCausalLM(config).to(dtype=torch.bfloat16)
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    raw_profile = make_task_profile(
        model="tiny-gpt-oss",
        task_names=[
            "layer_0_attn",
            "layer_0_moe_router",
            "layer_0_moe_experts",
            "layer_0_moe_combine",
        ],
        split_at=4,
        hidden_size=64,
        num_kv_heads=2,
        head_dim=16,
        max_seq_len=8,
    )
    profile_path = tmp_path / "fine_moe_profile.json"
    profile_path.write_text(json.dumps(raw_profile))
    profile = load_profile(profile_path, model_name="tiny-gpt-oss", hf_config=model.config)
    cfg = EngineConfig(
        model_name="tiny-gpt-oss",
        profiling_json=profile_path,
        max_seq_len=8,
        device="cpu",
    )
    runner = CausalLMRunner(
        model,
        tokenizer=None,
        executor=LayerExecutor(profile, cfg, comm_backend=FakeComm()),
        cfg=cfg,
        owned_layer_ids=owned_layer_ids(profile, 0),
    )

    with torch.inference_mode():
        expected = model(input_ids, use_cache=True).logits[:, -1:, :]
        actual = runner.prefill(input_ids)

    assert actual is not None
    assert torch.allclose(actual, expected, atol=2e-2, rtol=2e-2)


def test_tiny_gpt_oss_split_moe_prefill_matches_hf_forward(tmp_path):
    try:
        from transformers.models.gpt_oss import GptOssConfig, GptOssForCausalLM
    except Exception:
        return

    torch.manual_seed(0)
    config = GptOssConfig(
        num_hidden_layers=1,
        num_local_experts=2,
        vocab_size=128,
        hidden_size=64,
        intermediate_size=64,
        head_dim=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        sliding_window=2,
        num_experts_per_tok=1,
        layer_types=["sliding_attention"],
    )
    model = GptOssForCausalLM(config).to(dtype=torch.bfloat16)
    model.eval()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    raw_profile = make_task_profile(
        model="tiny-gpt-oss",
        task_names=[
            "layer_0_attn",
            "layer_0_moe_router",
            "layer_0_moe_experts",
            "layer_0_moe_combine",
        ],
        split_at=4,
        hidden_size=64,
        num_kv_heads=2,
        head_dim=16,
        max_seq_len=8,
    )
    raw_profile["tasks"][3]["rank"] = 1
    profile_path = tmp_path / "split_moe_profile.json"
    profile_path.write_text(json.dumps(raw_profile))
    profile = load_profile(profile_path, model_name="tiny-gpt-oss", hf_config=model.config)
    cfg = EngineConfig(
        model_name="tiny-gpt-oss",
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
