import importlib.util
import json
import sys
from pathlib import Path

from fluidgpu_torch.model_capability import (
    analyze_model_config,
    demo_config,
    get_registered_model_capability,
    load_model_config,
    model_capability_to_dict,
    registered_model_capabilities,
)


def test_dense_qwen_demo_is_runtime_compatible():
    model_name, raw = demo_config("dense_qwen")

    report = analyze_model_config(raw, model_name=model_name)

    assert report.family == "qwen"
    assert report.is_moe is False
    assert report.uses_mxfp4 is False
    assert report.runtime_compatible is True
    assert report.suggested_task_groups == (
        "embed",
        "layer_{i}_qkv",
        "layer_{i}_sdpa",
        "layer_{i}_o_proj",
        "layer_{i}_mlp_gate_up",
        "layer_{i}_mlp_down_proj",
        "norm_lm_head",
    )


def test_gpt_oss_moe_mxfp4_demo_is_runtime_compatible_with_warnings():
    model_name, raw = demo_config("gpt_oss_moe_mxfp4")

    report = analyze_model_config(raw, model_name=model_name)

    assert report.family == "gpt-oss"
    assert report.is_moe is True
    assert report.num_experts == 32
    assert report.experts_per_token == 4
    assert report.uses_mxfp4 is True
    assert report.runtime_compatible is True
    assert report.blockers == ()
    assert set(report.warnings) == {
        "moe_runtime_supports_fine_split_with_side_tensor_transport",
        "mxfp4_delegated_to_transformers_loader",
        "gpt_oss_runtime_uses_model_specific_attention_masks",
    }
    assert report.suggested_task_groups == (
        "embed",
        "layer_{i}_qkv",
        "layer_{i}_sdpa",
        "layer_{i}_o_proj",
        "layer_{i}_moe_router",
        "layer_{i}_moe_experts",
        "layer_{i}_moe_combine",
        "norm_lm_head",
    )


def test_load_model_config_and_recursive_mxfp4_detection(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "_name_or_path": "demo/moe",
                "architectures": ["DemoMoEForCausalLM"],
                "num_hidden_layers": 2,
                "hidden_size": 16,
                "num_attention_heads": 4,
                "num_key_value_heads": 1,
                "num_experts": 8,
                "quantization_config": {"nested": {"format": "MXFP4"}},
            }
        )
    )

    report = analyze_model_config(load_model_config(path))

    assert report.is_moe is True
    assert report.uses_mxfp4 is True


def test_model_capability_report_dict_shape():
    model_name, raw = demo_config("gpt_oss_moe_mxfp4")

    payload = model_capability_to_dict(analyze_model_config(raw, model_name=model_name))

    assert payload["runtime"]["compatible"] is True
    assert payload["capability_probe"]["runtime_integration"] is False
    assert payload["quantization"]["uses_mxfp4"] is True


def test_ad_registered_models_cover_all_paper_workloads():
    expected = {
        "meta-llama/Llama-3.1-8B-Instruct",
        "openai/gpt-oss-20b",
        "Qwen/Qwen2.5-VL-7B-Instruct",
        "mistralai/Mamba-Codestral-7B-v0.1",
        "stabilityai/stable-diffusion-3.5-medium",
    }

    registered = {item.model_name for item in registered_model_capabilities()}

    assert expected <= registered
    for model_name in expected:
        capability = get_registered_model_capability(model_name)
        assert capability is not None
        assert capability.runtime_status


def test_registered_multimodal_demo_mentions_ae_deferral():
    model_name, raw = demo_config("qwen25_vl_7b")

    payload = model_capability_to_dict(analyze_model_config(raw, model_name=model_name))

    assert payload["family"] == "qwen-vl"
    assert payload["registered_capability"]["has_prefill_decode"] is True
    assert "deferred to AE" in payload["registered_capability"]["runtime_status"]


def test_model_capability_probe_demo_cli(tmp_path):
    module = load_cli()
    output = tmp_path / "report.json"

    run_main(module, "--demo", "gpt_oss_moe_mxfp4", "--output-json", str(output))

    raw = json.loads(output.read_text())
    assert raw["family"] == "gpt-oss"
    assert raw["moe"]["enabled"] is True
    assert raw["quantization"]["uses_mxfp4"] is True
    assert raw["runtime"]["compatible"] is True


def load_cli():
    path = Path(__file__).parents[2] / "examples" / "fluidgpu_torch" / "model_capability_probe.py"
    spec = importlib.util.spec_from_file_location("fluidgpu_model_capability_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_main(module, *args: str) -> None:
    old_argv = sys.argv
    sys.argv = ["model_capability_probe.py", *args]
    try:
        module.main()
    finally:
        sys.argv = old_argv
