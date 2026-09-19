"""Kernel Analyzer: measure per-kernel-group latency for the policy planner.

Runs the model single-GPU through the real LayerExecutor + CausalLMRunner
(with an all-rank-0 profile and a stub comm backend), so the measured
kernel groups are exactly the units the runtime schedules. Emits the CSV
consumed by schedule_llm_layers.py: name,prefill_us,decode_us.

Run once per GPU model (pin the device with CUDA_VISIBLE_DEVICES), then feed
both CSVs to schedule_llm_layers.py (--solver dp|milp).
"""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import torch

from fluidgpu_torch.config import EngineConfig, validate_torch_version
from fluidgpu_torch.executor import LayerExecutor
from fluidgpu_torch.profile import make_ranked_task_profile, profile_from_dict
from fluidgpu_torch.runner import CausalLMRunner
from fluidgpu_torch.worker_trace import (
    clear_worker_trace_recorder,
    install_worker_trace_recorder,
)


class _SingleRankComm:
    """Stub comm backend: every task lives on rank 0, so no transfer may occur."""

    @staticmethod
    def get_rank() -> int:
        return 0

    @staticmethod
    def send_tensor(tensor: torch.Tensor, *, dst: int) -> None:
        raise AssertionError("single-rank profiling must not send tensors")

    @staticmethod
    def recv_tensor(tensor: torch.Tensor, *, src: int) -> None:
        raise AssertionError("single-rank profiling must not receive tensors")


def main() -> None:
    args = parse_args()
    validate_torch_version()
    assert torch.cuda.is_available(), "CUDA is required for kernel-group profiling"
    assert args.prompt_len + args.decode_steps <= args.max_seq_len, (
        f"prompt_len {args.prompt_len} + decode_steps {args.decode_steps} "
        f"must fit in max_seq_len {args.max_seq_len}"
    )
    torch.manual_seed(args.seed)

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    cfg = EngineConfig(
        model_name=args.model,
        dtype=torch.bfloat16,
        max_seq_len=args.max_seq_len,
        device_id=args.device_id,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_type = getattr(AutoConfig.from_pretrained(args.model), "model_type", "")
    if model_type == "gpt_oss":
        # Match the engine's per-rank weight representation (bf16-dequant on
        # sm<8.9 with enough memory, MXFP4 otherwise) so measured costs are
        # what the runtime actually executes. Quantized models are placed via
        # device_map, not .to().
        from fluidgpu_torch.engine import _quantization_kwargs

        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype="auto",
            **_quantization_kwargs(cfg),
            low_cpu_mem_usage=True,
            device_map={"": args.device_id},
        )
        from fluidgpu_torch.gptoss_fused_moe import maybe_patch_gptoss_fused_moe

        fused = maybe_patch_gptoss_fused_moe(hf_model)
        if fused:
            print(f"profile_llm_tasks: fused top-k MoE on {fused} layers")
    else:
        hf_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            **model_dtype_kwargs(torch.bfloat16),
            low_cpu_mem_usage=True,
        ).to(f"cuda:{args.device_id}")
    hf_model.eval()

    num_layers = int(hf_model.config.num_hidden_layers)
    task_names = [
        f"layer_{i}_{suffix}" for i in range(num_layers) for suffix in ("attn", "mlp")
    ]
    head_dim = getattr(
        hf_model.config,
        "head_dim",
        hf_model.config.hidden_size // hf_model.config.num_attention_heads,
    )
    raw_profile = make_ranked_task_profile(
        model=args.model,
        task_assignments=[(name, 0) for name in task_names],
        hidden_size=hf_model.config.hidden_size,
        num_kv_heads=hf_model.config.num_key_value_heads,
        head_dim=head_dim,
        max_seq_len=args.max_seq_len,
        dtype=torch.bfloat16,
    )
    profile = profile_from_dict(raw_profile, model_name=args.model, hf_config=hf_model.config)

    executor = LayerExecutor(profile, cfg, comm_backend=_SingleRankComm())
    runner = CausalLMRunner(hf_model, tokenizer, executor, cfg, list(range(num_layers)))

    input_ids = torch.randint(
        low=0,
        high=int(hf_model.config.vocab_size),
        size=(1, args.prompt_len),
        device=cfg.torch_device,
        dtype=torch.long,
    )

    for _ in range(args.warmup):
        run_cycle(runner, input_ids, args.decode_steps, cfg.torch_device)

    recorder = install_worker_trace_recorder(rank=0)
    try:
        for _ in range(args.repeat):
            run_cycle(runner, input_ids, args.decode_steps, cfg.torch_device)
        events = list(recorder.events)
    finally:
        clear_worker_trace_recorder()

    durations: dict[tuple[str, str], list[float]] = {}
    for event in events:
        if event["kind"] != "compute":
            continue
        key = (str(event["name"]), str(event["mode"]))
        durations.setdefault(key, []).append(float(event["duration_ms"]) * 1000.0)

    ordered_names = ["embed", *task_names, "norm_lm_head"]
    rows: list[dict[str, object]] = []
    for name in ordered_names:
        prefill_samples = durations.get((name, "prefill"), [])
        decode_samples = durations.get((name, "decode"), [])
        assert len(prefill_samples) == args.repeat, (
            f"{name}: expected {args.repeat} prefill samples, got {len(prefill_samples)}"
        )
        assert len(decode_samples) == args.repeat * args.decode_steps, (
            f"{name}: expected {args.repeat * args.decode_steps} decode samples, "
            f"got {len(decode_samples)}"
        )
        rows.append(
            {
                "name": name,
                "prefill_us": f"{statistics.median(prefill_samples):.3f}",
                "decode_us": f"{statistics.median(decode_samples):.3f}",
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "prefill_us", "decode_us"])
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"profiled model={args.model} device={torch.cuda.get_device_name(args.device_id)} "
        f"tasks={len(rows)} prompt_len={args.prompt_len} decode_steps={args.decode_steps} "
        f"repeat={args.repeat} output={args.output}"
    )


def run_cycle(
    runner: CausalLMRunner,
    input_ids: torch.Tensor,
    decode_steps: int,
    device: torch.device,
) -> None:
    with torch.inference_mode():
        logits = runner.prefill(input_ids)
        assert logits is not None, "rank 0 prefill must return logits"
        next_token = logits[:, -1, :].argmax(dim=-1).reshape(1, 1).to(device=device)
        for _ in range(decode_steps):
            logits = runner.decode_step(next_token)
            assert logits is not None, "rank 0 decode must return logits"
            next_token = logits[:, -1, :].argmax(dim=-1).reshape(1, 1).to(device=device)
    torch.cuda.synchronize(device)


def model_dtype_kwargs(dtype: torch.dtype) -> dict[str, torch.dtype]:
    import transformers

    major_minor = transformers.__version__.split(".", 2)[:2]
    major, minor = int(major_minor[0]), int(major_minor[1])
    if major > 4 or (major == 4 and minor >= 56):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure per-kernel-group prefill/decode latency for the planner"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    main()
