from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from fluidgpu_torch import vllm_compat as compat


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[1, 2, 3])

    def decode(self, token_ids, skip_special_tokens=False):
        return "decoded:" + ",".join(str(x) for x in token_ids)


def test_sampling_params_absorbs_unknown_kwargs():
    params = compat.SamplingParams(max_tokens=5, temperature=0.7, some_new_flag=True)

    assert params.max_tokens == 5
    assert params.temperature == 0.7
    assert params.top_p == 1.0
    assert params._extras == {"some_new_flag": True}


def test_sampling_params_rejects_n_not_one():
    with pytest.raises(NotImplementedError, match="n=1"):
        compat.SamplingParams(n=2)


def test_coerce_str_prompt():
    prompt, token_ids = compat._coerce_prompt("hello", FakeTokenizer())

    assert prompt == "hello"
    assert token_ids == [1, 2, 3]


def test_coerce_text_prompt_dict():
    prompt, token_ids = compat._coerce_prompt({"prompt": "hi"}, FakeTokenizer())

    assert prompt == "hi"
    assert token_ids == [1, 2, 3]


def test_coerce_tokens_prompt_dict():
    prompt, token_ids = compat._coerce_prompt({"prompt_token_ids": [9, 8, 7]}, FakeTokenizer())

    assert prompt == "decoded:9,8,7"
    assert token_ids == [9, 8, 7]


def test_coerce_rejects_multimodal_prompt():
    with pytest.raises(NotImplementedError, match="multi_modal_data"):
        compat._coerce_prompt({"prompt": "hi", "multi_modal_data": {"image": object()}}, FakeTokenizer())


def test_fluidgpu_llm_init_maps_engine_config(monkeypatch, caplog):
    fake_engine = MagicMock()
    fake_engine.tokenizer = FakeTokenizer()
    monkeypatch.setenv("FLUIDGPU_PROFILE_JSON", "/tmp/profile.json")
    monkeypatch.setenv("FLUIDGPU_COMM_TRANSPORT", "rdma")
    monkeypatch.setenv("FLUIDGPU_RANK", "1")
    monkeypatch.setenv("WORLD_SIZE", "3")
    monkeypatch.setenv("NCCL_IB_HCA", "mlx5_0")
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29540")

    with patch.object(compat, "LLMEngine", return_value=fake_engine) as engine_cls:
        llm = compat.FluidgpuLLM(
            model="Qwen/Qwen2.5-1.5B",
            dtype="auto",
            max_model_len=512,
            tensor_parallel_size=1,
            enable_chunked_prefill=True,
        )

    cfg = engine_cls.call_args.args[0]
    assert cfg.model_name == "Qwen/Qwen2.5-1.5B"
    assert str(cfg.profiling_json) == "/tmp/profile.json"
    assert cfg.max_seq_len == 512
    assert cfg.rank_override == 1
    assert cfg.world_size == 3
    assert cfg.master_port == 29540
    assert cfg.comm_transport == "rdma"
    assert llm.llm_engine.model_config.max_model_len == 512
    assert "ignoring engine kwargs" in caplog.text


def test_fluidgpu_llm_init_accepts_tensor_parallel_size_above_two(monkeypatch):
    fake_engine = MagicMock()
    fake_engine.tokenizer = FakeTokenizer()
    monkeypatch.setenv("FLUIDGPU_PROFILE_JSON", "/tmp/profile.json")

    with patch.object(compat, "LLMEngine", return_value=fake_engine) as engine_cls:
        compat.FluidgpuLLM(
            model="Qwen/Qwen2.5-1.5B",
            dtype="auto",
            max_model_len=512,
            tensor_parallel_size=4,
        )

    cfg = engine_cls.call_args.args[0]
    assert cfg.world_size == 2


def test_fluidgpu_llm_init_maps_hf_snapshot_path_to_profile_model(monkeypatch):
    fake_engine = MagicMock()
    fake_engine.tokenizer = FakeTokenizer()
    monkeypatch.setenv("FLUIDGPU_PROFILE_JSON", "/tmp/profile.json")

    snapshot = (
        "/home/test/.cache/huggingface/hub/"
        "models--Qwen--Qwen2.5-1.5B/snapshots/"
        "8faed761d45a263340a0528343f099c05c9a4323"
    )
    with patch.object(compat, "LLMEngine", return_value=fake_engine) as engine_cls:
        compat.FluidgpuLLM(model=snapshot, dtype="auto", max_model_len=512)

    assert engine_cls.call_args.args[0].model_name == "Qwen/Qwen2.5-1.5B"


def test_normalize_hf_model_name_keeps_repo_id():
    assert compat.normalize_hf_model_name("Qwen/Qwen2.5-1.5B") == "Qwen/Qwen2.5-1.5B"


def test_fluidgpu_llm_generate_builds_request_outputs(capsys):
    llm = compat.FluidgpuLLM.__new__(compat.FluidgpuLLM)
    llm._engine = MagicMock()
    llm._engine.generate_with_diagnostics.side_effect = (
        lambda prompt, max_new_tokens, seed=0: (None, list(range(max_new_tokens)))
    )
    llm.tokenizer = FakeTokenizer()
    llm.llm_engine = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=128, model="fake"),
        tokenizer=llm.tokenizer,
    )
    llm._announced = False

    params = compat.SamplingParams(max_tokens=5, seed=3)
    outputs = llm.generate(["hello", {"prompt_token_ids": [7, 7]}], [params, params])

    assert len(outputs) == 2
    assert outputs[0].prompt_token_ids == [1, 2, 3]
    assert outputs[0].outputs[0].token_ids == [0, 1, 2, 3, 4]
    assert outputs[1].prompt == "decoded:7,7"
    assert outputs[1].outputs[0].text == "decoded:0,1,2,3,4"
    assert llm._engine.generate_with_diagnostics.call_count == 2
    assert "forced greedy" in capsys.readouterr().err


def test_fluidgpu_llm_generate_rejects_lora():
    llm = compat.FluidgpuLLM.__new__(compat.FluidgpuLLM)
    llm._announced = True

    with pytest.raises(NotImplementedError, match="lora_request"):
        llm.generate(["hello"], [compat.SamplingParams(max_tokens=1)], lora_request=[object()])


def test_vllm_throughput_patch_switches_request_output_identity():
    pytest.importorskip("vllm")
    launcher = load_launcher()

    import vllm
    import vllm.benchmarks.throughput as throughput

    launcher._patch_vllm()

    assert vllm.LLM is compat.FluidgpuLLM
    assert vllm.SamplingParams is compat.SamplingParams
    assert throughput.RequestOutput is compat.RequestOutput


def test_vllm_throughput_rank1_disables_output_json(monkeypatch):
    launcher = load_launcher()
    args = SimpleNamespace(output_json="/tmp/out.json")
    monkeypatch.setenv("FLUIDGPU_RANK", "1")

    launcher._disable_rank1_result_files(args)

    assert args.output_json is None


def test_vllm_throughput_random_dataset_uses_phase_gate_lengths():
    launcher = load_launcher()
    args = SimpleNamespace(
        dataset_name="random",
        input_len=128,
        output_len=32,
        prefix_len=0,
        random_input_len=1024,
        random_output_len=128,
        random_prefix_len=0,
    )

    launcher._normalize_random_dataset_lengths(args)

    assert args.random_input_len == 128
    assert args.random_output_len == 32
    assert args.random_prefix_len == 0


def test_vllm_throughput_removes_repo_root_from_sys_path(monkeypatch, tmp_path):
    launcher = load_launcher()
    runtime_root = Path(__file__).parents[2]
    repo_root = tmp_path / "pytorch_source"
    for child in ("torch", "aten", "c10"):
        (repo_root / child).mkdir(parents=True)
    original = list(sys.path)
    monkeypatch.setattr(sys, "path", [str(repo_root), *original])

    launcher._avoid_source_torch_shadowing()

    assert str(repo_root) not in sys.path
    assert str(runtime_root) in sys.path


def load_launcher():
    path = Path(__file__).parents[2] / "examples" / "fluidgpu_torch" / "vllm_throughput.py"
    spec = importlib.util.spec_from_file_location("fluidgpu_vllm_throughput", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
