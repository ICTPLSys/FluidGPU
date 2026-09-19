from __future__ import annotations

import argparse
from pathlib import Path

from fluidgpu_torch.model_capability import (
    analyze_model_config,
    demo_config,
    load_model_config,
    write_model_capability_report,
)


def main() -> None:
    args = parse_args()
    if args.demo is not None:
        model_name, raw = demo_config(args.demo)
    else:
        raw = load_model_config(args.config_json)
        model_name = args.model_name
    report = analyze_model_config(raw, model_name=model_name)
    write_model_capability_report(args.output_json, report)
    print(
        "model_capability "
        f"model={report.model_name} family={report.family} "
        f"moe={int(report.is_moe)} mxfp4={int(report.uses_mxfp4)} "
        f"runtime_compatible={int(report.runtime_compatible)} "
        f"blockers={len(report.blockers)} output={args.output_json}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone model capability probe")
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--model-name")
    parser.add_argument(
        "--demo",
        choices=(
            "llama31_8b",
            "dense_qwen",
            "gpt_oss_moe_mxfp4",
            "qwen25_vl_7b",
            "mamba_codestral_7b",
            "sd35_medium",
        ),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    assert args.demo is not None or args.config_json is not None, (
        "provide --demo or --config-json"
    )
    return args


if __name__ == "__main__":
    main()
