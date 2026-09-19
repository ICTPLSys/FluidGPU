from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from fluidgpu_torch.config import validate_torch_version
from fluidgpu_torch.executor import _first_tensor
from fluidgpu_torch.runner import _forward_parameter_names, extract_components, run_feed_forward


def main() -> None:
    args = parse_args()
    validate_torch_version()
    assert torch.cuda.is_available(), "CUDA is required for profiling"

    device = torch.device(f"cuda:{args.device_id}")
    family = detect_family(args.model)
    if family == "diffusion":
        plan = build_diffusion_plan(args, device)
    elif family == "mamba":
        plan = build_mamba_plan(args, device)
    else:
        plan = build_causal_lm_plan(args, device, family)

    rows: list[dict[str, object]] = []
    for phase, block, fn in plan:
        rows += profile_slice(phase, block, fn, args.repeat, args.warmup)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["kernel_name", "phase", "block", "count", "total_us", "avg_us"],
        )
        writer.writeheader()
        writer.writerows(rows)


def detect_family(model_id: str) -> str:
    from transformers import AutoConfig

    try:
        config = AutoConfig.from_pretrained(model_id)
    except (OSError, ValueError, KeyError):
        # diffusers pipelines (e.g. SD3.5) have no top-level transformers config
        return "diffusion"
    model_type = getattr(config, "model_type", "")
    if model_type in ("mamba2", "mamba"):
        return "mamba"
    if model_type == "qwen2_5_vl":
        return "qwen2_5_vl"
    if model_type == "gpt_oss":
        return "gpt_oss"
    return "llm"


def build_causal_lm_plan(args, device: torch.device, family: str):
    if family == "qwen2_5_vl":
        from transformers import AutoModelForImageTextToText

        vl_model = AutoModelForImageTextToText.from_pretrained(
            args.model,
            **model_dtype_kwargs(torch.bfloat16),
            low_cpu_mem_usage=True,
        ).to(device)
        vl_model.eval()
        # fig2 profiles the language tower; the visual encoder is exercised by
        # the MLLM end-to-end figures, not the kernel-heterogeneity census.
        text_model = vl_model.model.language_model
        comp = {
            "embed_tokens": text_model.embed_tokens,
            "layers": list(text_model.layers),
            "norm": text_model.norm,
            "lm_head": vl_model.lm_head,
            "rotary_emb": getattr(text_model, "rotary_emb", None),
        }
        vocab_size = vl_model.config.vocab_size
        # Qwen2.5-VL M-RoPE expects position_ids of shape [3, batch, seq]
        position_ids_shape = "mrope"
    else:
        from transformers import AutoModelForCausalLM

        load_kwargs = dict(low_cpu_mem_usage=True)
        if family == "gpt_oss":
            # Match the engine's per-rank weight representation (bf16-dequant
            # on sm<8.9 with enough memory, MXFP4 otherwise). A quantized
            # model must be placed via device_map, not .to().
            from fluidgpu_torch.config import EngineConfig
            from fluidgpu_torch.engine import _quantization_kwargs

            probe_cfg = EngineConfig(model_name=args.model, device_id=device.index or 0)
            load_kwargs["dtype"] = "auto"
            load_kwargs.update(_quantization_kwargs(probe_cfg))
            load_kwargs["device_map"] = {"": device.index or 0}
            model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
            from fluidgpu_torch.gptoss_fused_moe import maybe_patch_gptoss_fused_moe

            fused = maybe_patch_gptoss_fused_moe(model)
            if fused:
                print(f"profile_llm_kernels: fused top-k MoE on {fused} layers")
        else:
            load_kwargs.update(model_dtype_kwargs(torch.bfloat16))
            model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).to(device)
        model.eval()
        comp = extract_components(model)
        vocab_size = model.config.vocab_size
        position_ids_shape = "standard"

    input_ids = torch.randint(vocab_size, (1, args.prompt_len), device=device)
    decode_ids = torch.randint(vocab_size, (1, 1), device=device)
    hidden_prefill = comp["embed_tokens"](input_ids)
    hidden_decode = comp["embed_tokens"](decode_ids)
    prefill_mask = causal_mask(seq_len=args.prompt_len, past_seen=0, device=device)
    decode_mask = causal_mask(seq_len=1, past_seen=0, device=device)
    prefill_pos = torch.arange(args.prompt_len, device=device).unsqueeze(0)
    decode_pos = torch.zeros((1, 1), dtype=torch.long, device=device)
    if position_ids_shape == "mrope":
        prefill_rope_pos = prefill_pos.unsqueeze(0).expand(3, -1, -1)
        decode_rope_pos = decode_pos.unsqueeze(0).expand(3, -1, -1)
    else:
        prefill_rope_pos = prefill_pos
        decode_rope_pos = decode_pos
    prefill_emb = position_embeddings(comp, hidden_prefill, prefill_rope_pos)
    decode_emb = position_embeddings(comp, hidden_decode, decode_rope_pos)

    return [
        ("prefill", "embed", lambda: comp["embed_tokens"](input_ids)),
        ("decode", "embed", lambda: comp["embed_tokens"](decode_ids)),
        (
            "prefill",
            "attn",
            lambda: run_all_attn(comp, hidden_prefill, prefill_mask, prefill_pos, prefill_emb),
        ),
        (
            "decode",
            "attn",
            lambda: run_all_attn(comp, hidden_decode, decode_mask, decode_pos, decode_emb),
        ),
        ("prefill", "ffn", lambda: run_all_mlp(comp, hidden_prefill)),
        ("decode", "ffn", lambda: run_all_mlp(comp, hidden_decode)),
        ("prefill", "head", lambda: comp["lm_head"](comp["norm"](hidden_prefill[:, -1:, :]))),
        ("decode", "head", lambda: comp["lm_head"](comp["norm"](hidden_decode[:, -1:, :]))),
    ]


def build_mamba_plan(args, device: torch.device):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        **model_dtype_kwargs(torch.bfloat16),
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    backbone = model.backbone
    layers = list(backbone.layers)
    vocab_size = model.config.vocab_size

    # transformers' eager Mamba2 torch_forward materializes ~4 GB of chunked
    # scan intermediates per layer at 512 prompt tokens, which OOMs a 44 GB
    # L40S. Cap the prompt so both GPUs of a pair profile identical shapes.
    prompt_len = min(args.prompt_len, 256)
    input_ids = torch.randint(vocab_size, (1, prompt_len), device=device)
    decode_ids = torch.randint(vocab_size, (1, 1), device=device)
    hidden_prefill = backbone.embeddings(input_ids)
    hidden_decode = backbone.embeddings(decode_ids)

    def run_all_blocks(hidden: torch.Tensor) -> torch.Tensor:
        current = hidden
        for layer in layers:
            current = _first_tensor(layer(current))
        return current

    return [
        ("prefill", "embed", lambda: backbone.embeddings(input_ids)),
        ("decode", "embed", lambda: backbone.embeddings(decode_ids)),
        ("prefill", "mixer", lambda: run_all_blocks(hidden_prefill)),
        ("decode", "mixer", lambda: run_all_blocks(hidden_decode)),
        ("prefill", "head", lambda: model.lm_head(backbone.norm_f(hidden_prefill[:, -1:, :]))),
        ("decode", "head", lambda: model.lm_head(backbone.norm_f(hidden_decode[:, -1:, :]))),
    ]


def build_diffusion_plan(args, device: torch.device):
    from diffusers import SD3Transformer2DModel

    transformer = SD3Transformer2DModel.from_pretrained(
        args.model,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)
    transformer.eval()
    cfg = transformer.config
    latent = torch.randn(
        (1, cfg.in_channels, cfg.sample_size, cfg.sample_size),
        dtype=torch.bfloat16,
        device=device,
    )
    prompt_embeds = torch.randn(
        (1, 333, cfg.joint_attention_dim), dtype=torch.bfloat16, device=device
    )
    pooled = torch.randn(
        (1, cfg.pooled_projection_dim), dtype=torch.bfloat16, device=device
    )
    timestep = torch.tensor([500.0], dtype=torch.bfloat16, device=device)

    def denoise_step() -> torch.Tensor:
        with torch.inference_mode():
            return transformer(
                hidden_states=latent,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled,
                return_dict=False,
            )[0]

    return [("denoise", "mmdit", denoise_step)]


def profile_slice(
    phase: str,
    block: str,
    fn,
    repeat: int,
    warmup: int,
) -> list[dict[str, object]]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(repeat):
            fn()
        torch.cuda.synchronize()
    aggregated: dict[str, tuple[int, float]] = {}
    for event in prof.key_averages():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        total_us = float(event.device_time_total)
        if total_us <= 0.0:
            continue
        count, total = aggregated.get(event.key, (0, 0.0))
        aggregated[event.key] = (count + int(event.count), total + total_us)
    rows: list[dict[str, object]] = []
    for kernel_name, (count, total_us) in aggregated.items():
        if count == 0:
            continue
        rows.append(
            {
                "kernel_name": kernel_name,
                "phase": phase,
                "block": block,
                "count": count,
                "total_us": f"{total_us:.3f}",
                "avg_us": f"{total_us / count:.3f}",
            }
        )
    return rows


def run_all_attn(comp, hidden, attention_mask, position_ids, position_emb):
    current = hidden
    kwargs_base = position_embedding_kwargs(position_emb)
    for layer in comp["layers"]:
        current = call_attn(layer, current, attention_mask, position_ids, kwargs_base)
    return current


def run_all_mlp(comp, hidden):
    current = hidden
    for layer in comp["layers"]:
        current = call_mlp(layer, current)
    return current


def call_attn(layer, hidden, attention_mask, position_ids, position_kwargs):
    residual = hidden
    hidden = layer.input_layernorm(hidden)
    kwargs = {
        "attention_mask": attention_mask,
        "cache_position": position_ids.flatten(),
        **position_kwargs,
    }
    params = _forward_parameter_names(layer.self_attn)
    if "position_ids" in params:
        kwargs["position_ids"] = position_ids
    if "use_cache" in params:
        kwargs["use_cache"] = False
    return residual + _first_tensor(layer.self_attn(hidden, **kwargs))


def call_mlp(layer, hidden):
    residual = hidden
    hidden = layer.post_attention_layernorm(hidden)
    return residual + run_feed_forward(layer, hidden)


def position_embeddings(comp, hidden: torch.Tensor, position_ids: torch.Tensor):
    if comp["rotary_emb"] is None:
        return None
    return comp["rotary_emb"](hidden, position_ids)


def position_embedding_kwargs(value):
    if value is None:
        return {}
    return {"position_embeddings": value}


def causal_mask(*, seq_len: int, past_seen: int, device: torch.device) -> torch.Tensor:
    total_len = past_seen + seq_len
    q_positions = torch.arange(past_seen, past_seen + seq_len, device=device)
    kv_positions = torch.arange(total_len, device=device)
    masked = kv_positions.unsqueeze(0) > q_positions.unsqueeze(1)
    mask = torch.zeros((seq_len, total_len), dtype=torch.bfloat16, device=device)
    return mask.masked_fill(masked, torch.finfo(torch.bfloat16).min)[None, None, :, :]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile CausalLM execution at the CUDA kernel granularity."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    return parser.parse_args()


def model_dtype_kwargs(dtype: torch.dtype) -> dict[str, torch.dtype]:
    import transformers

    major_minor = transformers.__version__.split(".", 2)[:2]
    major, minor = int(major_minor[0]), int(major_minor[1])
    if major > 4 or (major == 4 and minor >= 56):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


if __name__ == "__main__":
    main()
