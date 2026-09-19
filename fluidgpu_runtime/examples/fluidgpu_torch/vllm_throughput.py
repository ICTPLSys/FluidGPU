from __future__ import annotations

import argparse
import builtins
import os
import sys
from pathlib import Path


def _silence_if_not_rank0() -> None:
    if os.environ.get("FLUIDGPU_RANK", "0") == "0":
        return

    def _noop(*args, **kwargs) -> None:
        return None

    builtins.print = _noop


def _avoid_source_torch_shadowing() -> None:
    kept_paths: list[str] = []
    removed_source_root = False
    for entry in sys.path:
        try:
            resolved = Path(entry or os.getcwd()).resolve()
        except OSError:
            kept_paths.append(entry)
            continue
        if _looks_like_pytorch_source_root(resolved):
            removed_source_root = True
            continue
        kept_paths.append(entry)

    if not removed_source_root:
        return

    sys.path[:] = kept_paths
    runtime_root = Path(__file__).resolve().parents[2]
    if str(runtime_root) not in sys.path:
        sys.path.insert(0, str(runtime_root))


def _looks_like_pytorch_source_root(path: Path) -> bool:
    return (path / "torch").is_dir() and (path / "aten").is_dir() and (path / "c10").is_dir()


def _patch_vllm() -> None:
    import vllm
    import vllm.benchmarks.throughput as throughput
    import vllm.outputs as vllm_outputs
    from fluidgpu_torch import vllm_compat as compat

    vllm.LLM = compat.FluidgpuLLM
    vllm.SamplingParams = compat.SamplingParams
    vllm.RequestOutput = compat.RequestOutput
    vllm.CompletionOutput = compat.CompletionOutput
    vllm_outputs.RequestOutput = compat.RequestOutput
    vllm_outputs.CompletionOutput = compat.CompletionOutput
    throughput.RequestOutput = compat.RequestOutput


def _ensure_benchmark_platform_for_argparse() -> None:
    from vllm import platforms

    if not platforms.current_platform.is_unspecified():
        return

    from vllm.platforms.cpu import CpuPlatform

    platforms.current_platform = CpuPlatform()


def _parse_args() -> argparse.Namespace:
    from vllm.benchmarks.throughput import add_cli_args

    parser = argparse.ArgumentParser(
        description="Run vLLM throughput benchmark on fluidgpu_torch 2-rank runtime"
    )
    add_cli_args(parser)
    return parser.parse_args()


def _validate_supported_args(args: argparse.Namespace) -> None:
    if args.backend != "vllm":
        raise NotImplementedError(
            f"vllm_compat only supports backend='vllm', got {args.backend!r}"
        )
    if getattr(args, "async_engine", False):
        raise NotImplementedError("vllm_compat does not support --async-engine")
    if getattr(args, "n", 1) != 1:
        raise NotImplementedError("vllm_compat only supports --n 1")
    if getattr(args, "enable_lora", False) or getattr(args, "lora_path", None):
        raise NotImplementedError("vllm_compat does not support LoRA")


def _normalize_random_dataset_lengths(args: argparse.Namespace) -> None:
    if getattr(args, "dataset_name", None) != "random":
        return
    if getattr(args, "input_len", None) is not None:
        args.random_input_len = args.input_len
    if getattr(args, "output_len", None) is not None:
        args.random_output_len = args.output_len
    if getattr(args, "prefix_len", None) is not None:
        args.random_prefix_len = args.prefix_len


def _disable_rank1_result_files(args: argparse.Namespace) -> None:
    if os.environ.get("FLUIDGPU_RANK", "0") != "0":
        args.output_json = None


def main() -> None:
    _silence_if_not_rank0()
    _avoid_source_torch_shadowing()
    _patch_vllm()
    _ensure_benchmark_platform_for_argparse()
    args = _parse_args()
    _validate_supported_args(args)
    _normalize_random_dataset_lengths(args)
    _disable_rank1_result_files(args)

    from vllm.benchmarks.throughput import main as throughput_main

    throughput_main(args)


if __name__ == "__main__":
    main()
