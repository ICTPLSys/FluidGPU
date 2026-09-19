from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any
import os

import torch

import fluidgpu_torch
from fluidgpu_torch.config import validate_torch_version
from fluidgpu_torch.worker_trace import (
    clear_worker_trace_recorder,
    install_worker_trace_recorder,
)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    trace_recorder = None
    try:
        if args.worker_trace_output is not None:
            trace_rank = 0 if args.single_gpu else fluidgpu_torch.get_rank()
            trace_recorder = install_worker_trace_recorder(rank=trace_rank)

        if args.single_gpu:
            results = run_single_gpu_requests(args)
            emit_results(args, results)
            return

        cfg = fluidgpu_torch.EngineConfig(
            model_name=args.model,
            profiling_json=args.profile,
            dtype=torch.bfloat16,
            max_seq_len=args.max_seq_len,
            master_addr=args.master_addr,
            master_port=args.master_port,
            device_id=args.device_id,
            strict_transformers_version=not args.allow_transformers_mismatch,
            comm_transport=args.comm_transport,
            nccl_socket_ifname=args.nccl_socket_ifname,
            nccl_ib_hca=args.nccl_ib_hca,
            nccl_ib_gid_index=args.nccl_ib_gid_index,
            cuda_graph_decode=args.cuda_graph_decode,
            world_size=args.world_size,
        )
        engine = fluidgpu_torch.LLMEngine(cfg)
        workload = None
        if args.dataset_jsonl is not None:
            from fluidgpu_torch.dataset_workload import load_jsonl_workload

            workload = load_jsonl_workload(
                args.dataset_jsonl,
                engine.tokenizer,
                max_seq_len=args.max_seq_len,
                limit=args.requests,
            )
        results: list[dict[str, Any]] = []
        started = time.perf_counter()
        request_count = len(workload) if workload is not None else args.requests
        for request_index in range(request_count):
            pace_request(started, request_index, args.request_rate)
            request_started = time.perf_counter()
            if workload is not None:
                prompt = workload[request_index].prompt
                max_new_tokens = workload[request_index].max_new_tokens
            else:
                prompt = request_prompt(args.prompt, request_index)
                max_new_tokens = args.max_new_tokens
            first_logits, tokens = engine.generate_with_diagnostics(
                prompt,
                max_new_tokens=max_new_tokens,
                seed=args.seed + request_index,
            )
            if fluidgpu_torch.get_rank() == 0:
                input_ids = engine.tokenizer(prompt, return_tensors="pt").input_ids.to(engine.device)
                output_ids = torch.cat(
                    [input_ids, torch.tensor([tokens], dtype=torch.long, device=engine.device)],
                    dim=1,
                )
                text = engine.tokenizer.decode(output_ids[0], skip_special_tokens=True)
                results.append(
                    request_result(
                        args=args,
                        request_index=request_index,
                        prompt=prompt,
                        text=text,
                        tokens=tokens,
                        first_logits=first_logits,
                        elapsed_s=time.perf_counter() - request_started,
                    )
                )
                results[-1]["prompt_tokens"] = int(input_ids.shape[1])
        engine.close()
        if fluidgpu_torch.get_rank() == 0:
            emit_results(args, results, elapsed_s=time.perf_counter() - started)
    finally:
        if trace_recorder is not None:
            trace_recorder.write_jsonl(args.worker_trace_output)
            clear_worker_trace_recorder()


def run_single_gpu_requests(args: argparse.Namespace) -> list[dict[str, Any]]:
    validate_torch_version()
    assert torch.cuda.is_available(), "CUDA is required for LLM generation"
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    from transformers import AutoConfig

    model_type = getattr(AutoConfig.from_pretrained(args.model), "model_type", "")
    if model_type == "gpt_oss":
        # Match the engine's per-rank weight representation and fused-MoE
        # execution so single-GPU baselines measure the same kernels the
        # runtime uses (see engine._quantization_kwargs / gptoss_fused_moe).
        from fluidgpu_torch.config import EngineConfig
        from fluidgpu_torch.engine import _quantization_kwargs
        from fluidgpu_torch.gptoss_fused_moe import maybe_patch_gptoss_fused_moe

        probe_cfg = EngineConfig(model_name=args.model, device_id=args.device_id)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            dtype="auto",
            **_quantization_kwargs(probe_cfg),
            low_cpu_mem_usage=True,
            device_map={"": args.device_id},
        )
        fused = maybe_patch_gptoss_fused_moe(model)
        if fused:
            print(f"llm_generate[single-gpu]: fused top-k MoE on {fused} layers")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            **model_dtype_kwargs(torch.bfloat16),
            low_cpu_mem_usage=True,
        ).to(f"cuda:{args.device_id}")
    model.eval()
    results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for request_index in range(args.requests):
        pace_request(started, request_index, args.request_rate)
        prompt = request_prompt(args.prompt, request_index)
        request_started = time.perf_counter()
        text, tokens, first_logits = generate_single_gpu(
            args=args,
            tokenizer=tokenizer,
            model=model,
            prompt=prompt,
        )
        results.append(
            request_result(
                args=args,
                request_index=request_index,
                prompt=prompt,
                text=text,
                tokens=tokens,
                first_logits=first_logits,
                elapsed_s=time.perf_counter() - request_started,
            )
        )
    return results


def pace_request(started: float, request_index: int, request_rate: float | None) -> None:
    if request_rate is None or request_index == 0:
        return
    target = started + request_index / request_rate
    delay = target - time.perf_counter()
    if delay > 0.0:
        time.sleep(delay)


def generate_single_gpu(
    *,
    args: argparse.Namespace,
    tokenizer,
    model,
    prompt: str,
) -> tuple[str, list[int], torch.Tensor]:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(f"cuda:{args.device_id}")
    if input_ids.shape[1] > args.max_seq_len:
        input_ids = input_ids[:, -args.max_seq_len :]

    tokens: list[int] = []
    with torch.inference_mode():
        out = model(input_ids, use_cache=True)
        logits = out.logits[:, -1, :]
        first_logits = logits.detach().clone()
        next_token = torch.tensor([[int(logits.argmax(dim=-1).item())]], device=input_ids.device)
        tokens.append(int(next_token.item()))
        past = out.past_key_values
        for _ in range(args.max_new_tokens - 1):
            out = model(next_token, past_key_values=past, use_cache=True)
            next_token = torch.tensor(
                [[int(out.logits[:, -1, :].argmax(dim=-1).item())]],
                device=input_ids.device,
            )
            tokens.append(int(next_token.item()))
            past = out.past_key_values
    output_ids = torch.cat([input_ids, torch.tensor([tokens], device=input_ids.device)], dim=1)
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return text, tokens, first_logits


def request_prompt(prompt: str, request_index: int) -> str:
    if request_index == 0:
        return prompt
    return f"{prompt}\nRequest index: {request_index}."


def request_result(
    *,
    args: argparse.Namespace,
    request_index: int,
    prompt: str,
    text: str,
    tokens: list[int],
    first_logits: torch.Tensor | None,
    elapsed_s: float,
) -> dict[str, Any]:
    first_logits_argmax = None
    if first_logits is not None:
        first_logits_argmax = int(first_logits.argmax(dim=-1).item())
    return {
        "request_index": request_index,
        "model": args.model,
        "prompt": prompt,
        "max_new_tokens": args.max_new_tokens,
        "request_rate": args.request_rate,
        "seed": args.seed + request_index,
        "text": text,
        "tokens": tokens,
        "first_logits": first_logits,
        "first_logits_argmax": first_logits_argmax,
        "latency_ms": elapsed_s * 1000.0,
        "output_tokens": len(tokens),
    }


def emit_results(
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    *,
    elapsed_s: float | None = None,
) -> None:
    if elapsed_s is None:
        elapsed_s = sum(float(item["latency_ms"]) for item in results) / 1000.0
    for item in results:
        print("request_index:", item["request_index"])
        print(item["text"])
        print("generated_token_ids:", item["tokens"])
        if item["first_logits_argmax"] is not None:
            print("first_logits_argmax:", item["first_logits_argmax"])
    if args.log_jsonl is not None:
        args.log_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.log_jsonl.open("w") as f:
            for item in results:
                f.write(json.dumps(json_safe_request(item)) + "\n")
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        total_tokens = sum(int(item["output_tokens"]) for item in results)
        prompt_tokens = sum(int(item.get("prompt_tokens", 0)) for item in results)
        summary = {
            "status": "ok",
            "model": args.model,
            "requests": len(results),
            "max_new_tokens": args.max_new_tokens,
            "request_rate": args.request_rate,
            "generated_tokens": total_tokens,
            "elapsed_s": elapsed_s,
            "throughput_req_s": len(results) / elapsed_s if elapsed_s > 0.0 else 0.0,
            "throughput_tok_s": total_tokens / elapsed_s if elapsed_s > 0.0 else 0.0,
            "prompt_tokens": prompt_tokens,
            "throughput_total_tok_s": (
                (total_tokens + prompt_tokens) / elapsed_s if elapsed_s > 0.0 else 0.0
            ),
            "latency_ms_avg": (
                sum(float(item["latency_ms"]) for item in results) / len(results)
                if results
                else 0.0
            ),
        }
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")
    if args.diagnostics_output is not None and results:
        write_diagnostics(args, args.diagnostics_output, results)


def json_safe_request(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in item.items()
        if key != "first_logits"
    }


def write_diagnostics(
    args: argparse.Namespace,
    path: Path,
    results: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    last = results[-1]
    payload = {
        "model": args.model,
        "prompt": last["prompt"],
        "max_new_tokens": args.max_new_tokens,
        "seed": last["seed"],
        "text": last["text"],
        "tokens": last["tokens"],
        "first_logits": (
            last["first_logits"].detach().cpu()
            if last["first_logits"] is not None
            else None
        ),
        "requests": [
            {
                **json_safe_request(item),
                "first_logits": (
                    item["first_logits"].detach().cpu()
                    if item["first_logits"] is not None
                    else None
                ),
            }
            for item in results
        ],
    }
    torch.save(payload, path)
    print("diagnostics_output:", path)


def model_dtype_kwargs(dtype: torch.dtype) -> dict[str, torch.dtype]:
    import transformers

    major_minor = transformers.__version__.split(".", 2)[:2]
    major, minor = int(major_minor[0]), int(major_minor[1])
    if major > 4 or (major == 4 and minor >= 56):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FluidGPU multi-rank greedy LLM generation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--prompt", default="Explain GPU disaggregation in three sentences.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument(
        "--dataset-jsonl",
        type=Path,
        help="Splitwise-style request JSONL; overrides --prompt/--max-new-tokens; --requests limits count.",
    )
    parser.add_argument("--request-rate", type=float)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--world-size", type=int, default=_default_world_size())
    parser.add_argument("--comm-transport", choices=["auto", "rdma"], default="auto")
    parser.add_argument("--nccl-socket-ifname")
    parser.add_argument("--nccl-ib-hca")
    parser.add_argument("--nccl-ib-gid-index", type=int)
    parser.add_argument("--cuda-graph-decode", action="store_true")
    parser.add_argument("--single-gpu", action="store_true")
    parser.add_argument("--allow-transformers-mismatch", action="store_true")
    parser.add_argument("--diagnostics-output", type=Path)
    parser.add_argument("--log-jsonl", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--worker-trace-output", type=Path)
    args = parser.parse_args()
    assert args.requests > 0, f"--requests must be positive, got {args.requests}"
    assert args.world_size >= 2, f"--world-size must be >= 2, got {args.world_size}"
    if args.request_rate is not None:
        assert args.request_rate > 0.0, f"--request-rate must be positive, got {args.request_rate}"
    if not args.single_gpu:
        assert args.profile is not None, "--profile is required for distributed execution"
    return args


def _default_world_size() -> int:
    for name in ("FLUIDGPU_WORLD_SIZE", "WORLD_SIZE"):
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return 2


if __name__ == "__main__":
    main()
