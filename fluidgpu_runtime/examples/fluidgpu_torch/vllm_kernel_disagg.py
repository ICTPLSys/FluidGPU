"""FluidGPU kernel-group placement inside vLLM's batched engine.

The paper's fig6 baselines are stock-vLLM output tok/s; the AF /
kernel-disaggregation rows therefore must come from a *batched* engine with
cross-GPU kernel placement. Batching is what makes per-layer cross-GPU
hand-offs profitable: a batch-32 hidden hop is ~176 KB (~46 us measured),
so specializing each GPU on the kernels it runs fastest (paper zig-zag /
AF) beats two generalist GPUs.

Mechanism (monkey-patch only, no vLLM source edits) — see
fluidgpu_torch/vllm_integration.py: a ``worker_cls`` subclass applies the
placement inside each worker after ``load_model`` and before the first
traced forward, so it composes with vLLM's native features:

- continuous batching / paged attention / prefix caching: untouched
  (attention, KV cache, router, embed/head stay on the primary GPU);
- torch.compile + piecewise CUDA graphs: the cross-GPU op is a registered
  opaque custom op listed in ``splitting_ops`` — on-GPU segments are
  captured/replayed as CUDA graphs while the hop runs eagerly between
  replays (``--enforce-eager`` disables graphs for A/B);
- multiprocess engine + ``vllm serve``: the worker hook travels as a string
  (no in-process requirement; ``--in-process`` remains for debugging).

gpt-oss experts are rebuilt marlin-native on the target device (packing is
architecture-specific); llama moves each dense MLP wholesale.

Example (gpt-oss AF row, Splitwise, unlimited batch, CUDA graphs on):
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2 \
  python3 vllm_kernel_disagg.py \
    --model <path> --placement af \
    --dataset-jsonl datasets/conversation_benchmark/requests.jsonl \
    --num-prompts 128 --output-json af.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_EXAMPLES_DIR = Path(__file__).resolve().parent
_RUNTIME_ROOT = _EXAMPLES_DIR.parents[1]


def _ensure_importable() -> None:
    """Make fluidgpu_torch importable here AND in spawned worker processes.

    worker_cls is resolved by qualified name inside each worker, so the
    package root must be on PYTHONPATH (env propagates to spawned procs).
    The examples dir is added for vllm_rdma_hop (rdma/auto hop transport).
    """
    for path in (str(_RUNTIME_ROOT), str(_EXAMPLES_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)
    existing = os.environ.get("PYTHONPATH")
    parts = [str(_RUNTIME_ROOT), str(_EXAMPLES_DIR)]
    if existing:
        parts.append(existing)
    os.environ["PYTHONPATH"] = ":".join(parts)


def load_workload(path: Path, num_prompts: int, skip: int = 0) -> list[tuple[str, int]]:
    items: list[tuple[str, int]] = []
    with path.open() as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            rec = json.loads(line)
            items.append((rec["prompt"], int(rec["output_tokens"])))
            if len(items) >= num_prompts:
                break
    assert items, f"no requests in {path} (skip={skip})"
    return items


def main() -> None:
    args = parse_args()
    _ensure_importable()

    if args.in_process:
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # ping-pong micro-batching reuses vLLM's DBO engine; decode batches keep
    # replaying piecewise CUDA graphs (the dispatcher gate excludes them from
    # micro-batching), large prefill chunks run the eager two-thread overlap.
    os.environ["FLUIDGPU_PINGPONG"] = "1" if args.pingpong else "0"
    os.environ["FLUIDGPU_VLLM_PLACEMENT"] = args.placement
    os.environ["FLUIDGPU_EXPERT_DEVICE"] = args.expert_device
    os.environ["FLUIDGPU_HOP"] = args.hop
    os.environ["FLUIDGPU_FREE_LOCAL_EXPERTS"] = "0" if args.keep_local_experts else "1"
    os.environ["FLUIDGPU_PHASE_AWARE"] = "1" if args.phase_aware else "0"

    os.environ["FLUIDGPU_PHASE_ROUTE"] = "1" if args.phase_route else "0"
    os.environ["FLUIDGPU_DECODE_CAP"] = str(args.decode_cap)
    os.environ["FLUIDGPU_PHASE_SPLIT_UBATCH"] = "1" if args.phase_split else "0"

    from fluidgpu_torch.vllm_integration import (
        SCHEDULER_CLS,
        WORKER_CLS,
        fluid_compilation_config,
    )
    from vllm import LLM, SamplingParams

    engine_kwargs: dict = {}
    if args.placement != "none":
        engine_kwargs["worker_cls"] = WORKER_CLS
        if not args.enforce_eager:
            # Piecewise CUDA graphs with the cross-GPU op as a splitting op.
            engine_kwargs["compilation_config"] = fluid_compilation_config()
        if args.decode_cap > 0:
            # Two-pool scheduler: cap decode batch, prefill the rest ahead.
            engine_kwargs["scheduler_cls"] = SCHEDULER_CLS

    if args.max_num_batched_tokens:
        engine_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if args.no_chunked_prefill:
        # Offline-throughput protocol knob: schedule whole prompts per step
        # (no decode mixing). Requires max_num_batched_tokens >= max_model_len.
        engine_kwargs["enable_chunked_prefill"] = False
        engine_kwargs.setdefault("max_num_batched_tokens", args.max_model_len)
    if args.quantization:
        engine_kwargs["quantization"] = args.quantization
    if args.kv_cache_dtype:
        engine_kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    if args.async_scheduling:
        # Overlap step N+1's CPU scheduling/prep with step N's GPU execution.
        # Without it the batch queue serializes on each decode step's sampled
        # tokens (engine main blocks in step_with_batch_queue future.result),
        # leaving the home GPU idle for the CPU segment of every decode step.
        engine_kwargs["async_scheduling"] = True
    if args.kv_cache_memory_gb > 0:
        # Pin the KV pool size, bypassing the memory profiler. Needed for
        # co-located engine instances: the profiler measures FREE memory, so
        # two concurrent instances race — whoever profiles second sees the
        # winner's KV allocation as "used" and gets none.
        engine_kwargs["kv_cache_memory_bytes"] = int(
            args.kv_cache_memory_gb * (1 << 30)
        )
    if args.profile_dir:
        # Profile from INSIDE the worker: the worker process holds BOTH GPUs, so one
        # trace carries home-GPU and expert-GPU kernels on a common clock -- the only
        # way to see whether the ping-pong micro-batches actually overlap the hop
        # (home attention concurrent with expert FFN) or merely alternate.
        # vLLM 0.18 dropped VLLM_TORCH_PROFILER_DIR; ProfilerConfig is the entry point.
        from vllm.config.profiler import ProfilerConfig

        os.makedirs(args.profile_dir, exist_ok=True)
        engine_kwargs["profiler_config"] = ProfilerConfig(
            profiler="torch", torch_profiler_dir=args.profile_dir
        )
    llm = LLM(
        model=args.model,
        enforce_eager=args.enforce_eager,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=not args.no_prefix_cache,
        seed=0,
        **engine_kwargs,
    )
    mode = "eager" if args.enforce_eager else "piecewise-cudagraph"
    if args.pingpong:
        mode = "pingpong-eager"
    print(
        f"vllm_kernel_disagg: placement={args.placement} hop={args.hop} "
        f"exec={mode} multiprocessing="
        f"{os.environ.get('VLLM_ENABLE_V1_MULTIPROCESSING', '1')}"
    )

    workload = load_workload(Path(args.dataset_jsonl), args.num_prompts, args.skip_prompts)
    # Mirror dataset_workload.py: re-tokenize and truncate each prompt so that
    # prompt + declared output fits the context window.
    tokenizer = llm.get_tokenizer()
    prompts = []
    params = []
    for text, out_len in workload:
        ids = tokenizer.encode(text)
        budget = max(args.max_model_len - out_len - 8, 16)
        if len(ids) > budget:
            ids = ids[:budget]
            text = tokenizer.decode(ids)
        prompts.append(text)
        params.append(
            SamplingParams(max_tokens=out_len, temperature=0.0, ignore_eos=True)
        )

    if args.profile_dir:
        llm.generate(prompts, params)  # warm up / compile -- not profiled
        llm.start_profile()
    started = time.perf_counter()
    outputs = llm.generate(prompts, params)
    elapsed = time.perf_counter() - started
    if args.profile_dir:
        llm.stop_profile()

    out_tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
    in_tokens = sum(len(o.prompt_token_ids) for o in outputs)
    result = {
        "placement": args.placement,
        "hop": args.hop,
        "exec_mode": mode,
        "requests": len(outputs),
        "elapsed_s": elapsed,
        "input_tokens": in_tokens,
        "output_tokens": out_tokens,
        "output_tok_s": out_tokens / elapsed,
        "total_tok_s": (in_tokens + out_tokens) / elapsed,
    }
    print(json.dumps(result, indent=1))
    if args.sanity:
        for o in outputs[:2]:
            print("SANITY:", repr(o.outputs[0].text[:120]))
    if args.dump_texts:
        Path(args.dump_texts).parent.mkdir(parents=True, exist_ok=True)
        Path(args.dump_texts).write_text(
            json.dumps([o.outputs[0].text for o in outputs], indent=0)
        )
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(result, indent=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--placement", choices=["none", "af", "af-bf16"], default="none"
    )
    parser.add_argument(
        "--no-prefix-cache",
        action="store_true",
        help="disable automatic prefix caching (identical synthetic prompts "
        "otherwise get near-free prefill and inflate throughput)",
    )
    parser.add_argument(
        "--no-chunked-prefill",
        action="store_true",
        help="disable chunked prefill (whole prompts per step, no decode "
        "mixing; offline-throughput protocol knob)",
    )
    parser.add_argument(
        "--quantization",
        default=None,
        help="vLLM quantization mode (e.g. fp8 for on-the-fly W8A8/weight-only)",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default=None,
        help="KV cache dtype (e.g. fp8) — halves KV reads for decode-bound "
        "long-context shapes",
    )
    parser.add_argument(
        "--hop",
        choices=["copy", "rdma", "auto"],
        default="copy",
        help="cross-GPU transport: host-staged copy, GPUDirect RDMA over the "
        "per-GPU IB HCAs, or auto (RDMA for hops >= 4MB)",
    )
    parser.add_argument(
        "--pingpong",
        action="store_true",
        help="paper-FG ping-pong: split each step into 2 micro-batches so the "
        "attention GPU computes one while the expert GPU computes the other "
        "(reuses vLLM's DBO engine at DP=1; implies --enforce-eager)",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="disable torch.compile + CUDA graphs (legacy A/B reference; "
        "default runs piecewise CUDA graphs around the cross-GPU op)",
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="set VLLM_ENABLE_V1_MULTIPROCESSING=0 (debugging; the worker "
        "hook no longer requires it)",
    )
    parser.add_argument(
        "--keep-local-experts",
        action="store_true",
        help="keep the primary-GPU expert weights after the remote rebuild "
        "(default frees them so profiling reclaims the VRAM for KV cache)",
    )
    parser.add_argument(
        "--phase-aware",
        action="store_true",
        help="paper-FG phase-aware placement: decode-size batches compute "
        "experts locally on the home GPU (zero hop — the hop chain is pure "
        "loss for memory-bound decode), prefill batches disaggregate + "
        "ping-pong; keeps local expert weights (implies --keep-local-experts)",
    )
    parser.add_argument(
        "--phase-route", action="store_true",
        help="per-token phase routing: in mixed prefill+decode steps route "
        "decode tokens' FFN to the home GPU (affinity) and prefill tokens' FFN "
        "to the expert GPU, overlapped (needs a home marlin copy like balanced)",
    )
    parser.add_argument(
        "--decode-cap", type=int, default=0,
        help="two-pool scheduler: cap the decode batch at N while extra "
        "requests prefill ahead (set --max-num-seqs to N + prefill-pool). "
        "0 = stock scheduler.",
    )
    parser.add_argument(
        "--phase-split", action="store_true",
        help="lever E: phase-split DBO. A MIXED prefill+decode step is "
        "micro-batched into a DECODE ubatch (FFN local on the home A100) and a "
        "PREFILL ubatch (FFN remote on the L40S); DBO overlaps them so both "
        "cards run at once. Needs --phase-aware --pingpong.",
    )
    parser.add_argument("--expert-device", default="cuda:1")
    parser.add_argument("--dataset-jsonl", required=True)
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument(
        "--skip-prompts", type=int, default=0,
        help="skip the first N dataset records (disjoint shards for "
        "concurrent mirrored instances)",
    )
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument(
        "--max-num-batched-tokens", type=int, default=0,
        help="scheduler token budget per step (0 = vLLM default); larger "
        "prefill chunks amortize eager-python overhead per token",
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument(
        "--kv-cache-memory-gb", type=float, default=0,
        help="pin the KV cache pool to this many GiB (bypasses the memory "
        "profiler; required when co-locating multiple engine instances on "
        "one GPU, where concurrent profiling races)",
    )
    parser.add_argument(
        "--async-scheduling", action="store_true",
        help="engine async_scheduling=True: schedule step N+1 before step N's "
        "sampled tokens return, keeping the GPU fed through each step's CPU "
        "segment (greedy offline runs only; incompatible with logprobs)",
    )
    parser.add_argument("--sanity", action="store_true")
    parser.add_argument(
        "--dump-texts",
        help="write all generated texts to this json file (parity diffing)",
    )
    parser.add_argument("--output-json")
    parser.add_argument(
        "--profile-dir",
        help="dump a torch trace here (worker-side, so it covers BOTH GPUs). "
        "Runs generate() twice: once to warm up, once profiled. Keep "
        "--num-prompts small; traces get large fast.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
