from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

import fluidgpu_torch


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

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
    )
    worker_profiles = None
    if args.worker_profiles:
        from transformers import AutoConfig

        from fluidgpu_torch.profile import load_profile

        hf_config = AutoConfig.from_pretrained(args.model)
        worker_profiles = [
            load_profile(path, model_name=args.model, hf_config=hf_config)
            for path in args.worker_profiles.split(",")
        ]
    with fluidgpu_torch.AsyncLLMEngine.from_config(
        cfg,
        num_workers=args.num_workers,
        priority_scheduling=args.priority_scheduling,
        worker_profiles=worker_profiles,
    ) as engine:
        prompt_tokens = 0
        if args.dataset_jsonl is not None:
            from fluidgpu_torch.dataset_workload import load_jsonl_workload

            workload = load_jsonl_workload(
                args.dataset_jsonl,
                engine.engine.tokenizer,
                max_seq_len=args.max_seq_len,
                limit=args.requests,
            )
            prompt_tokens = sum(item.prompt_tokens for item in workload)
            requests = [
                fluidgpu_torch.GenerationRequest(
                    item.prompt,
                    max_new_tokens=item.max_new_tokens,
                    seed=args.seed + idx,
                )
                for idx, item in enumerate(workload)
            ]
        else:
            requests = [
                fluidgpu_torch.GenerationRequest(
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed + idx,
                )
                for idx, prompt in enumerate(expand_prompts(args))
            ]
        generated_tokens = sum(req.max_new_tokens for req in requests)
        started = time.perf_counter()
        futures = engine.submit_many(requests)
        for idx, future in enumerate(futures):
            text = future.result()
            if fluidgpu_torch.get_rank() == 0 and not args.quiet_requests:
                print(f"request_index: {idx}")
                print(text)
        elapsed_s = time.perf_counter() - started

    if fluidgpu_torch.get_rank() == 0:
        emit_summary(
            args,
            request_count=len(requests),
            elapsed_s=elapsed_s,
            generated_tokens=generated_tokens,
            prompt_tokens=prompt_tokens,
        )


def expand_prompts(args: argparse.Namespace) -> list[str]:
    base = args.prompt
    if args.requests is None:
        return base
    assert args.requests > 0, f"--requests must be positive, got {args.requests}"
    # Cycle the provided prompts with a per-request suffix so each request is
    # distinct (mirrors llm_generate's request_prompt behavior).
    return [
        f"{base[index % len(base)]} (request {index})"
        for index in range(args.requests)
    ]


def emit_summary(
    args: argparse.Namespace,
    *,
    request_count: int,
    elapsed_s: float,
    generated_tokens: int,
    prompt_tokens: int = 0,
) -> None:
    # Greedy decode runs exactly max_new_tokens steps (no EOS early-exit), so
    # the generated-token count is exact.
    total_tokens = generated_tokens
    summary = {
        "status": "ok",
        "model": args.model,
        "requests": request_count,
        "max_new_tokens": args.max_new_tokens,
        "num_workers": args.num_workers,
        "priority_scheduling": args.priority_scheduling,
        "generated_tokens": total_tokens,
        "elapsed_s": elapsed_s,
        "throughput_req_s": request_count / elapsed_s if elapsed_s > 0.0 else 0.0,
        "throughput_tok_s": total_tokens / elapsed_s if elapsed_s > 0.0 else 0.0,
        "prompt_tokens": prompt_tokens,
        "throughput_total_tok_s": (
            (total_tokens + prompt_tokens) / elapsed_s if elapsed_s > 0.0 else 0.0
        ),
    }
    print(
        f"async summary: requests={request_count} workers={args.num_workers} "
        f"priority={args.priority_scheduling} elapsed_s={elapsed_s:.3f} "
        f"req_s={summary['throughput_req_s']:.3f} tok_s={summary['throughput_tok_s']:.3f}"
    )
    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run queued FluidGPU greedy LLM generation")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        help="Prompt to enqueue. Pass more than once to queue multiple requests.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument(
        "--requests",
        type=int,
        default=None,
        help="Total request count; cycles --prompt values with a unique suffix.",
    )
    parser.add_argument("--priority-scheduling", action="store_true")
    parser.add_argument(
        "--dataset-jsonl",
        type=Path,
        help="Splitwise-style request JSONL; overrides --prompt/--max-new-tokens.",
    )
    parser.add_argument("--quiet-requests", action="store_true")
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--worker-profiles",
        help="comma-separated profile JSONs assigned round-robin to workers "
        "(complementary zig-zag placement); overrides --profile per worker "
        "while --profile still defines the engine default",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29500)
    parser.add_argument("--comm-transport", choices=["auto", "rdma"], default="auto")
    parser.add_argument("--nccl-socket-ifname")
    parser.add_argument("--nccl-ib-hca")
    parser.add_argument("--nccl-ib-gid-index", type=int)
    parser.add_argument("--cuda-graph-decode", action="store_true")
    parser.add_argument("--allow-transformers-mismatch", action="store_true")
    args = parser.parse_args()
    if args.prompt is None:
        args.prompt = ["Explain GPU disaggregation."]
    assert args.num_workers > 0, f"num_workers must be positive, got {args.num_workers}"
    return args


if __name__ == "__main__":
    main()
