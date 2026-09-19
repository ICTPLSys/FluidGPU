from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelCapabilityReport:
    model_name: str
    model_type: str | None
    architectures: tuple[str, ...]
    family: str
    num_hidden_layers: int | None
    hidden_size: int | None
    num_attention_heads: int | None
    num_key_value_heads: int | None
    head_dim: int | None
    is_moe: bool
    num_experts: int | None
    experts_per_token: int | None
    uses_mxfp4: bool
    quantization: dict[str, Any] | None
    runtime_compatible: bool
    blockers: tuple[str, ...]
    warnings: tuple[str, ...]
    suggested_task_groups: tuple[str, ...]


@dataclass(frozen=True)
class RegisteredModelCapability:
    model_name: str
    family: str
    architecture_class: str
    has_prefill_decode: bool
    has_attention_ffn_blocks: bool
    default_context_length: int
    runtime_status: str
    notes: tuple[str, ...] = ()


_REGISTERED_MODELS: tuple[RegisteredModelCapability, ...] = (
    RegisteredModelCapability(
        model_name="meta-llama/Llama-3.1-8B-Instruct",
        family="llama",
        architecture_class="dense decoder-only CausalLM",
        has_prefill_decode=True,
        has_attention_ffn_blocks=True,
        default_context_length=8192,
        runtime_status="AD validation target; cached on the AE machine (gated repo, HF_TOKEN needed elsewhere)",
    ),
    RegisteredModelCapability(
        model_name="openai/gpt-oss-20b",
        family="gpt-oss",
        architecture_class="MoE decoder-only CausalLM",
        has_prefill_decode=True,
        has_attention_ffn_blocks=True,
        default_context_length=131072,
        runtime_status="AD validation target on A100+L40S when weights are available",
        notes=("MXFP4 loading is delegated to transformers.",),
    ),
    RegisteredModelCapability(
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        family="qwen-vl",
        architecture_class="multimodal vision-language CausalLM",
        has_prefill_decode=True,
        has_attention_ffn_blocks=True,
        default_context_length=32768,
        runtime_status="Capability registered in AD; end-to-end MLLM evaluation is deferred to AE",
    ),
    RegisteredModelCapability(
        model_name="mistralai/Mamba-Codestral-7B-v0.1",
        family="mamba",
        architecture_class="selective state-space language model",
        has_prefill_decode=True,
        has_attention_ffn_blocks=False,
        default_context_length=256000,
        runtime_status="Capability registered in AD; SSM end-to-end evaluation is deferred to AE",
    ),
    RegisteredModelCapability(
        model_name="stabilityai/stable-diffusion-3.5-medium",
        family="diffusion",
        architecture_class="diffusion transformer image generation",
        has_prefill_decode=False,
        has_attention_ffn_blocks=True,
        default_context_length=77,
        runtime_status="Capability registered in AD; diffusion pipeline evaluation is deferred to AE",
    ),
)


def load_model_config(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def analyze_model_config(
    raw: dict[str, Any],
    *,
    model_name: str | None = None,
) -> ModelCapabilityReport:
    resolved_name = model_name or str(raw.get("_name_or_path") or raw.get("model_name") or "unknown")
    model_type = _optional_str(raw.get("model_type"))
    architectures = tuple(str(item) for item in _as_list(raw.get("architectures")))
    family = _detect_family(resolved_name, model_type, architectures)
    num_hidden_layers = _optional_int(raw.get("num_hidden_layers") or raw.get("n_layer"))
    hidden_size = _optional_int(raw.get("hidden_size") or raw.get("n_embd"))
    num_attention_heads = _optional_int(
        raw.get("num_attention_heads") or raw.get("n_head") or raw.get("num_heads")
    )
    num_key_value_heads = _optional_int(
        raw.get("num_key_value_heads") or raw.get("num_kv_heads") or raw.get("n_kv_head")
    )
    head_dim = _optional_int(raw.get("head_dim"))
    if head_dim is None and hidden_size is not None and num_attention_heads:
        if hidden_size % num_attention_heads == 0:
            head_dim = hidden_size // num_attention_heads

    num_experts = _first_int(
        raw,
        "num_local_experts",
        "num_experts",
        "n_experts",
        "moe_num_experts",
        "expert_count",
    )
    experts_per_token = _first_int(
        raw,
        "num_experts_per_tok",
        "experts_per_token",
        "num_experts_per_token",
        "moe_top_k",
        "router_top_k",
    )
    is_moe = _detect_moe(raw, architectures, num_experts, experts_per_token)
    quantization = _extract_quantization(raw)
    uses_mxfp4 = _detect_mxfp4(raw)
    suggested_task_groups = _suggested_task_groups(is_moe)
    blockers = _runtime_blockers(
        family=family,
        is_moe=is_moe,
        uses_mxfp4=uses_mxfp4,
        num_hidden_layers=num_hidden_layers,
    )
    warnings = _warnings(
        family=family,
        is_moe=is_moe,
        uses_mxfp4=uses_mxfp4,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
    )
    return ModelCapabilityReport(
        model_name=resolved_name,
        model_type=model_type,
        architectures=architectures,
        family=family,
        num_hidden_layers=num_hidden_layers,
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        is_moe=is_moe,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        uses_mxfp4=uses_mxfp4,
        quantization=quantization,
        runtime_compatible=not blockers,
        blockers=blockers,
        warnings=warnings,
        suggested_task_groups=suggested_task_groups,
    )


def model_capability_to_dict(report: ModelCapabilityReport) -> dict[str, Any]:
    registered = get_registered_model_capability(report.model_name)
    return {
        "model_name": report.model_name,
        "model_type": report.model_type,
        "architectures": list(report.architectures),
        "family": report.family,
        "shape": {
            "num_hidden_layers": report.num_hidden_layers,
            "hidden_size": report.hidden_size,
            "num_attention_heads": report.num_attention_heads,
            "num_key_value_heads": report.num_key_value_heads,
            "head_dim": report.head_dim,
        },
        "moe": {
            "enabled": report.is_moe,
            "num_experts": report.num_experts,
            "experts_per_token": report.experts_per_token,
        },
        "quantization": {
            "uses_mxfp4": report.uses_mxfp4,
            "config": report.quantization,
        },
        "runtime": {
            "compatible": report.runtime_compatible,
            "blockers": list(report.blockers),
            "warnings": list(report.warnings),
        },
        "suggested_task_groups": list(report.suggested_task_groups),
        "registered_capability": (
            registered_model_to_dict(registered) if registered is not None else None
        ),
        "capability_probe": {
            "runtime_integration": False,
            "note": "Config-only capability analysis; no model weights are loaded.",
        },
    }


def write_model_capability_report(path: str | Path, report: ModelCapabilityReport) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(model_capability_to_dict(report), indent=2) + "\n")


def demo_config(name: str) -> tuple[str, dict[str, Any]]:
    if name == "llama31_8b":
        return (
            "meta-llama/Llama-3.1-8B-Instruct",
            {
                "_name_or_path": "meta-llama/Llama-3.1-8B-Instruct",
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "num_hidden_layers": 32,
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
            },
        )
    if name == "dense_qwen":
        return (
            "Qwen/Qwen2.5-1.5B",
            {
                "_name_or_path": "Qwen/Qwen2.5-1.5B",
                "architectures": ["Qwen2ForCausalLM"],
                "model_type": "qwen2",
                "num_hidden_layers": 28,
                "hidden_size": 1536,
                "num_attention_heads": 12,
                "num_key_value_heads": 2,
                "head_dim": 128,
                "torch_dtype": "bfloat16",
            },
        )
    if name == "gpt_oss_moe_mxfp4":
        return (
            "openai/gpt-oss-20b",
            {
                "_name_or_path": "openai/gpt-oss-20b",
                "architectures": ["GptOssForCausalLM"],
                "model_type": "gpt_oss",
                "num_hidden_layers": 24,
                "hidden_size": 2880,
                "num_attention_heads": 64,
                "num_key_value_heads": 8,
                "head_dim": 64,
                "num_local_experts": 32,
                "num_experts_per_tok": 4,
                "quantization_config": {
                    "quant_method": "mxfp4",
                    "weight_format": "MXFP4",
                },
            },
        )
    if name == "qwen25_vl_7b":
        return (
            "Qwen/Qwen2.5-VL-7B-Instruct",
            {
                "_name_or_path": "Qwen/Qwen2.5-VL-7B-Instruct",
                "architectures": ["Qwen2_5_VLForConditionalGeneration"],
                "model_type": "qwen2_5_vl",
                "num_hidden_layers": 28,
                "hidden_size": 3584,
                "num_attention_heads": 28,
                "num_key_value_heads": 4,
                "head_dim": 128,
            },
        )
    if name == "mamba_codestral_7b":
        return (
            "mistralai/Mamba-Codestral-7B-v0.1",
            {
                "_name_or_path": "mistralai/Mamba-Codestral-7B-v0.1",
                "architectures": ["MambaForCausalLM"],
                "model_type": "mamba",
                "num_hidden_layers": 64,
                "hidden_size": 4096,
                "num_attention_heads": 0,
                "num_key_value_heads": 0,
            },
        )
    if name == "sd35_medium":
        return (
            "stabilityai/stable-diffusion-3.5-medium",
            {
                "_name_or_path": "stabilityai/stable-diffusion-3.5-medium",
                "architectures": ["SD3Transformer2DModel"],
                "model_type": "sd3",
                "num_hidden_layers": 24,
                "hidden_size": 2432,
                "num_attention_heads": 38,
                "num_key_value_heads": 38,
            },
        )
    raise AssertionError(f"unknown demo config {name}")


def registered_model_capabilities() -> tuple[RegisteredModelCapability, ...]:
    return _REGISTERED_MODELS


def get_registered_model_capability(model_name: str) -> RegisteredModelCapability | None:
    normalized = _canonical_model_name(model_name)
    for capability in _REGISTERED_MODELS:
        if _canonical_model_name(capability.model_name) == normalized:
            return capability
    return None


def registered_model_to_dict(capability: RegisteredModelCapability) -> dict[str, Any]:
    return {
        "model_name": capability.model_name,
        "family": capability.family,
        "architecture_class": capability.architecture_class,
        "has_prefill_decode": capability.has_prefill_decode,
        "has_attention_ffn_blocks": capability.has_attention_ffn_blocks,
        "default_context_length": capability.default_context_length,
        "runtime_status": capability.runtime_status,
        "notes": list(capability.notes),
    }


def _detect_family(
    model_name: str,
    model_type: str | None,
    architectures: tuple[str, ...],
) -> str:
    haystack = " ".join([model_name, model_type or "", *architectures]).lower()
    normalized = haystack.replace("_", "-")
    if "gpt-oss" in normalized or "gptoss" in normalized:
        return "gpt-oss"
    if "qwen2.5-vl" in normalized or "qwen2-5-vl" in normalized:
        return "qwen-vl"
    if "qwen" in normalized:
        return "qwen"
    if "llama" in normalized:
        return "llama"
    if "mamba" in normalized:
        return "mamba"
    if "stable-diffusion" in normalized or "sd3" in normalized:
        return "diffusion"
    return "unknown"


def _detect_moe(
    raw: dict[str, Any],
    architectures: tuple[str, ...],
    num_experts: int | None,
    experts_per_token: int | None,
) -> bool:
    if num_experts is not None and num_experts > 1:
        return True
    if experts_per_token is not None and experts_per_token > 1:
        return True
    if _optional_bool(raw.get("moe")) is True:
        return True
    text = " ".join(_flatten_text({"architectures": architectures, "model_type": raw.get("model_type")}))
    return "moe" in text.lower() or "mixtureofexperts" in text.lower()


def _detect_mxfp4(raw: dict[str, Any]) -> bool:
    text = " ".join(_flatten_text(raw)).lower().replace("_", "")
    return "mxfp4" in text or ("mx" in text and "fp4" in text)


def _extract_quantization(raw: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("quantization_config", "quant_config", "compression_config"):
        value = raw.get(key)
        if isinstance(value, dict):
            return value
    return None


def _runtime_blockers(
    *,
    family: str,
    is_moe: bool,
    uses_mxfp4: bool,
    num_hidden_layers: int | None,
) -> tuple[str, ...]:
    blockers: list[str] = []
    if num_hidden_layers is None:
        blockers.append("missing_num_hidden_layers")
    return tuple(blockers)


def _warnings(
    *,
    family: str,
    is_moe: bool,
    uses_mxfp4: bool,
    hidden_size: int | None,
    num_attention_heads: int | None,
    num_key_value_heads: int | None,
) -> tuple[str, ...]:
    warnings: list[str] = []
    if is_moe:
        warnings.append("moe_runtime_supports_fine_split_with_side_tensor_transport")
    if uses_mxfp4:
        warnings.append("mxfp4_delegated_to_transformers_loader")
    if family == "gpt-oss":
        warnings.append("gpt_oss_runtime_uses_model_specific_attention_masks")
    if family == "unknown":
        warnings.append("unknown_model_family_assume_llama_like_dense_layout")
    if hidden_size is None:
        warnings.append("missing_hidden_size")
    if num_attention_heads is None:
        warnings.append("missing_num_attention_heads")
    if num_key_value_heads is None:
        warnings.append("missing_num_key_value_heads")
    return tuple(warnings)


def _suggested_task_groups(is_moe: bool) -> tuple[str, ...]:
    if is_moe:
        return (
            "embed",
            "layer_{i}_qkv",
            "layer_{i}_sdpa",
            "layer_{i}_o_proj",
            "layer_{i}_moe_router",
            "layer_{i}_moe_experts",
            "layer_{i}_moe_combine",
            "norm_lm_head",
        )
    return (
        "embed",
        "layer_{i}_qkv",
        "layer_{i}_sdpa",
        "layer_{i}_o_proj",
        "layer_{i}_mlp_gate_up",
        "layer_{i}_mlp_down_proj",
        "norm_lm_head",
    )


def _first_int(raw: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = _optional_int(raw.get(key))
        if value is not None:
            return value
    return None


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in ("1", "true", "yes", "y"):
        return True
    if normalized in ("0", "false", "no", "n"):
        return False
    raise AssertionError(f"invalid bool value {value!r}")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _flatten_text(value: Any) -> list[str]:
    if isinstance(value, dict):
        out: list[str] = []
        for key, item in value.items():
            out.append(str(key))
            out.extend(_flatten_text(item))
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            out.extend(_flatten_text(item))
        return out
    if value is None:
        return []
    return [str(value)]


def _canonical_model_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")
