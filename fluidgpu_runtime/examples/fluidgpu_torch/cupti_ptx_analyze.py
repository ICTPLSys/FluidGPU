from __future__ import annotations

import argparse
from pathlib import Path

from fluidgpu_torch.cupti_ptx import (
    KernelRecord,
    analyze_records,
    load_ptx_dir,
    parse_cupti_csv,
    write_analysis_json,
    write_labeled_csv,
)


def main() -> None:
    args = parse_args()
    if args.demo:
        records = demo_records()
        ptx_by_kernel = demo_ptx()
    else:
        assert args.input_csv is not None, "provide --input-csv or --demo"
        records = parse_cupti_csv(
            args.input_csv,
            name_column=args.name_column,
            duration_column=args.duration_column,
            duration_unit=args.duration_unit,
        )
        ptx_by_kernel = load_ptx_dir(args.ptx_dir) if args.ptx_dir is not None else {}

    result = analyze_records(records, ptx_by_kernel=ptx_by_kernel)
    if args.output_json is not None:
        write_analysis_json(args.output_json, result)
    if args.output_csv is not None:
        write_labeled_csv(args.output_csv, result)

    totals = result.group_totals_us
    group_text = " ".join(f"{group}_us={totals[group]:.3f}" for group in sorted(totals))
    print(
        "cupti_ptx_analysis "
        f"kernels={len(result.kernels)} total_us={result.total_us:.3f} {group_text}"
    )
    if any(result.attention_split_weights.values()):
        split = result.attention_split_weights
        print(
            "attention_split "
            f"qkv={split['qkv']:.6f} sdpa={split['sdpa']:.6f} "
            f"o_proj={split['o_proj']:.6f}"
        )


def demo_records() -> tuple[KernelRecord, ...]:
    return (
        KernelRecord(
            name="layer_0_qkv_cublasLtMatmul",
            duration_us=35.0,
            layer_id=0,
            phase="prefill",
            ordinal=0,
        ),
        KernelRecord(
            name="layer_0_flash_attention_fwd",
            duration_us=45.0,
            layer_id=0,
            phase="prefill",
            ordinal=1,
        ),
        KernelRecord(
            name="layer_0_o_proj_cublasLtMatmul",
            duration_us=20.0,
            layer_id=0,
            phase="prefill",
            ordinal=2,
        ),
        KernelRecord(
            name="layer_0_mlp_swiglu_down_proj",
            duration_us=80.0,
            layer_id=0,
            phase="prefill",
            ordinal=3,
        ),
        KernelRecord(
            name="layer_0_residual_add",
            duration_us=5.0,
            layer_id=0,
            phase="prefill",
            ordinal=4,
        ),
    )


def demo_ptx() -> dict[str, str]:
    tensor_core_ptx = """
.visible .entry layer_0_qkv_cublasLtMatmul() {
  .reg .b32 %r<16>;
  ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1,%r2,%r3,%r4}, [%r5];
  mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%r6,%r7}, {%r1}, {%r2}, {%r6,%r7};
  st.global.b32 [%r8], %r6;
}
"""
    flash_ptx = """
.visible .entry layer_0_flash_attention_fwd() {
  .reg .b32 %r<32>;
  ld.global.b32 %r1, [%r2];
  bar.sync 0;
  st.global.b32 [%r3], %r1;
}
"""
    return {
        "layer_0_qkv_cublasLtMatmul": tensor_core_ptx,
        "layer_0_flash_attention_fwd": flash_ptx,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline CUPTI/PTX kernel analyzer probe")
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--ptx-dir", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--name-column")
    parser.add_argument("--duration-column")
    parser.add_argument("--duration-unit", default="auto", choices=("auto", "ns", "us", "ms"))
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    assert args.demo or args.input_csv is not None, "provide --input-csv or --demo"
    assert args.output_json is not None or args.output_csv is not None, (
        "provide --output-json or --output-csv"
    )
    return args


if __name__ == "__main__":
    main()
