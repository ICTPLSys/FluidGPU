import csv
import importlib.util
import json
import sys
from pathlib import Path

from fluidgpu_torch.cupti_ptx import (
    KernelRecord,
    analyze_ptx,
    analyze_records,
    classify_kernel,
    instrument_ptx_memory_ranges,
    parse_cupti_csv,
)


def test_classify_kernel_name_groups():
    assert classify_kernel("layer_0_qkv_cublasLtMatmul").group == "qkv"
    assert classify_kernel("layer_0_flash_attention_fwd").group == "sdpa"
    assert classify_kernel("layer_0_o_proj_cublasLtMatmul").group == "o_proj"
    assert classify_kernel("layer_0_mlp_swiglu_down_proj").group == "mlp"
    assert classify_kernel("aten_residual_add").group == "other"


def test_analyze_ptx_features():
    ptx = """
.visible .entry kernel() {
  .reg .b32 %r<12>;
  ld.global.b32 %r1, [%r2];
  ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r3,%r4,%r5,%r6}, [%r7];
  mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%r8}, {%r3}, {%r4}, {%r8};
  bar.sync 0;
  st.global.b32 [%r9], %r8;
}
"""
    features = analyze_ptx(ptx)

    assert features.entry_names == ("kernel",)
    assert features.register_count == 12
    assert features.tensor_core_instructions == 1
    assert features.memory_instructions == 3
    assert features.barrier_instructions == 1
    assert features.compute_kind == "tensor_core"


def test_instrument_ptx_memory_ranges_injects_listing_chain():
    ptx = """.version 8.0
.target sm_80
.address_size 64

.visible .entry test_kernel(
    .param .u64 input,
    .param .u64 output
)
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.global.b32 %r1, [%rd1+4];
    st.global.b32 [%rd2], %r1;
    ret;
}
"""
    instrumentation = instrument_ptx_memory_ranges(ptx)

    assert ".param .u64 fluidgpu_inst_buf" in instrumentation.instrumented_ptx
    assert "ld.param.u64 %fluidgpu_inst, [fluidgpu_inst_buf];" in instrumentation.instrumented_ptx
    assert "atom.global.min.u64 %fluidgpu_unused, [%fluidgpu_slot], %fluidgpu_addr;" in instrumentation.instrumented_ptx
    assert "atom.global.max.u64 %fluidgpu_unused, [%fluidgpu_slot+8], %fluidgpu_end;" in instrumentation.instrumented_ptx
    assert "add.u64 %fluidgpu_end, %fluidgpu_addr, 3;" in instrumentation.instrumented_ptx
    assert [access.access_kind for access in instrumentation.accesses] == ["read", "write"]
    assert [access.width_bytes for access in instrumentation.accesses] == [4, 4]
    assert instrumentation.accesses[0].address_expr == "%rd1+4"


def test_instrument_ptx_memory_ranges_covers_vector_accesses():
    # Vector operand lists are brace-enclosed; the brace must not be mistaken
    # for the body-opening brace (which would skip instrumentation of exactly
    # the widest accesses).
    ptx = """.version 8.0
.target sm_80
.address_size 64

.visible .entry vec_kernel(
    .param .u64 input,
    .param .u64 output
)
{
    .reg .f32 %f<9>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd1];
    st.global.v2.f32 [%rd2+8], {%f1, %f2};
    ret;
}
"""
    instrumentation = instrument_ptx_memory_ranges(ptx)

    assert [access.mnemonic for access in instrumentation.accesses] == [
        "ld.global.v4.f32",
        "st.global.v2.f32",
    ]
    assert [access.width_bytes for access in instrumentation.accesses] == [16, 8]
    assert instrumentation.accesses[1].address_expr == "%rd2+8"
    assert "add.u64 %fluidgpu_end, %fluidgpu_addr, 15;" in instrumentation.instrumented_ptx


def test_instrument_ptx_memory_ranges_predicates_guarded_accesses():
    # A guarded access must be recorded under the same predicate; otherwise
    # the injected atomics execute when the guard is false and record an
    # address that never happened.
    ptx = """.version 8.0
.target sm_80
.address_size 64

.visible .entry guard_kernel(
    .param .u64 output
)
{
    .reg .pred %p<2>;
    .reg .f32 %f<2>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [output];
    @%p1 st.global.f32 [%rd1+16], %f1;
    @!%p1 ld.global.f32 %f1, [%rd1];
    ret;
}
"""
    instrumentation = instrument_ptx_memory_ranges(ptx)

    assert len(instrumentation.accesses) == 2
    text = instrumentation.instrumented_ptx
    assert "@%p1 add.u64 %fluidgpu_addr, %rd1, +16;" in text
    assert "@%p1 atom.global.min.u64 %fluidgpu_unused, [%fluidgpu_slot], %fluidgpu_addr;" in text
    assert "@!%p1 mov.u64 %fluidgpu_addr, %rd1;" in text
    assert "@!%p1 atom.global.max.u64 %fluidgpu_unused, [%fluidgpu_slot+8], %fluidgpu_end;" in text
    # The unpredicated form of the injected atomics must not appear anywhere.
    for line in text.splitlines():
        if "atom.global" in line:
            assert line.lstrip().startswith(("@%p1", "@!%p1"))


def test_parse_cupti_csv_auto_units(tmp_path):
    input_path = tmp_path / "trace.csv"
    with input_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Name", "Duration (ns)", "Layer ID", "Phase", "Stream"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "Name": "layer_0_qkv_cublasLtMatmul",
                "Duration (ns)": "35000",
                "Layer ID": "0",
                "Phase": "prefill",
                "Stream": "7",
            }
        )

    records = parse_cupti_csv(input_path)

    assert len(records) == 1
    assert records[0].duration_us == 35.0
    assert records[0].layer_id == 0
    assert records[0].phase == "prefill"
    assert records[0].stream_id == "7"


def test_analyze_records_attention_split_weights():
    records = [
        KernelRecord("layer_0_qkv_cublasLtMatmul", 35.0, layer_id=0, phase="prefill"),
        KernelRecord("layer_0_flash_attention_fwd", 45.0, layer_id=0, phase="prefill"),
        KernelRecord("layer_0_o_proj_cublasLtMatmul", 20.0, layer_id=0, phase="prefill"),
        KernelRecord("layer_0_mlp_swiglu_down_proj", 80.0, layer_id=0, phase="prefill"),
    ]

    result = analyze_records(records)

    assert result.group_totals_us["qkv"] == 35.0
    assert result.group_totals_us["sdpa"] == 45.0
    assert result.group_totals_us["o_proj"] == 20.0
    assert result.group_totals_us["mlp"] == 80.0
    assert result.attention_split_weights == {
        "qkv": 0.35,
        "sdpa": 0.45,
        "o_proj": 0.2,
    }


def test_cupti_ptx_analyze_demo_cli(tmp_path):
    module = load_cli()
    output_json = tmp_path / "analysis.json"
    output_csv = tmp_path / "labeled.csv"

    run_main(
        module,
        "--demo",
        "--output-json",
        str(output_json),
        "--output-csv",
        str(output_csv),
    )

    raw = json.loads(output_json.read_text())
    assert raw["kernel_count"] == 5
    assert raw["attention_split_weights"] == {
        "qkv": 0.35,
        "sdpa": 0.45,
        "o_proj": 0.2,
    }
    rows = list(csv.DictReader(output_csv.open()))
    assert [row["group"] for row in rows] == ["qkv", "sdpa", "o_proj", "mlp", "other"]


def test_ptx_memory_probe_builds_access_table_and_raw_ddg(tmp_path):
    module = load_example("ptx_memory_probe.py", "fluidgpu_ptx_memory_probe")
    out_dir = tmp_path / "probe"
    out_dir.mkdir()

    instrumentation = instrument_ptx_memory_ranges(module.SAMPLE_PTX)
    module.write_access_csv(out_dir / "memory_accesses.csv", instrumentation.accesses)
    static_ranges = module.infer_static_ranges(module.SAMPLE_PTX, instrumentation.accesses)
    raw_edges = module.build_raw_edges(static_ranges)

    rows = list(csv.DictReader((out_dir / "memory_accesses.csv").open()))

    assert len(instrumentation.accesses) == 3
    assert raw_edges == [
        {
            "src": "producer_kernel",
            "dst": "consumer_kernel",
            "buffer": "intermediate",
            "write_access_id": 0,
            "read_access_id": 1,
        }
    ]
    assert [row["access_kind"] for row in rows] == ["write", "read", "write"]
    assert "ld.param.u64 %fluidgpu_inst, [fluidgpu_inst_buf];" in instrumentation.instrumented_ptx
    assert "atom.global.min.u64" in instrumentation.instrumented_ptx
    assert "atom.global.max.u64" in instrumentation.instrumented_ptx


def load_cli():
    return load_example("cupti_ptx_analyze.py", "fluidgpu_cupti_ptx_analyze")


def load_example(filename: str, module_name: str):
    path = Path(__file__).parents[2] / "examples" / "fluidgpu_torch" / filename
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_main(module, *args: str, prog: str = "cupti_ptx_analyze.py") -> None:
    old_argv = sys.argv
    sys.argv = [prog, *args]
    try:
        module.main()
    finally:
        sys.argv = old_argv
