from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fluidgpu_torch.cupti_ptx import (
    PTXMemoryAccess,
    analyze_ptx,
    instrument_ptx_memory_ranges,
)


# Opaque-kernel sample structured to mirror paper Listing 1. The instrumented
# output written to <run>/instrumented.ptx demonstrates the
# atom.global.min/max injection pattern described in the paper:
#
#   .visible .entry opaque_kernel(
#       ... original params ...
#       .param .u64 inst_buf)
#   {
#       ld.param.u64 %rd_inst, [inst_buf];
#       mad.wide.u32 %rd_slot, k, 16, %rd_inst;      // slot for access k
#       atom.global.min.u64 %_, [%rd_slot   ], %rd_addr;
#       atom.global.max.u64 %_, [%rd_slot+8 ], %rd_addr;
#       ld.global.f32 %f, [%rd_addr];                // original access
#       ...
#   }
#
# The two kernels below form a RAW chain (producer writes `intermediate`, then
# consumer reads it) so the probe can also exercise data-dependency extraction.
SAMPLE_PTX = """.version 8.0
.target sm_80
.address_size 64

// Opaque kernel #1: performs a short arithmetic body and emits one f32 store.
.visible .entry producer_kernel(
    .param .u64 output
)
{
    .reg .f32 %f<4>;
    .reg .b64 %rd<2>;
    ld.param.u64 %rd1, [output];
    mov.f32 %f1, 0f3f800000;        // 1.0
    mov.f32 %f2, 0f40000000;        // 2.0
    add.f32 %f3, %f1, %f2;          // 3.0
    st.global.f32 [%rd1], %f3;      // single global write
    ret;
}

// Opaque kernel #2: reads the producer's buffer (RAW edge) and writes a
// transformed value. Two global accesses -> two (min,max) atomic pairs after
// instrumentation.
.visible .entry consumer_kernel(
    .param .u64 input,
    .param .u64 output
)
{
    .reg .f32 %f<3>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [input];
    ld.param.u64 %rd2, [output];
    ld.global.f32 %f1, [%rd1];      // RAW read from producer's buffer
    mul.f32 %f2, %f1, 0f40000000;   // * 2.0
    st.global.f32 [%rd2], %f2;      // single global write
    ret;
}
"""


SAMPLE_PARAM_BUFFERS = {
    "producer_kernel": {"output": "intermediate"},
    "consumer_kernel": {"input": "intermediate", "output": "final"},
}
SAMPLE_BUFFER_BASES = {
    "intermediate": 0x1000_0000_0000,
    "final": 0x2000_0000_0000,
}


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ptx = args.input_ptx.read_text() if args.input_ptx is not None else SAMPLE_PTX
    instrumentation = instrument_ptx_memory_ranges(ptx)
    original_path = args.output_dir / "original.ptx"
    instrumented_path = args.output_dir / "instrumented.ptx"
    original_path.write_text(ptx)
    instrumented_path.write_text(instrumentation.instrumented_ptx)
    write_access_csv(args.output_dir / "memory_accesses.csv", instrumentation.accesses)

    static_ranges = infer_static_ranges(ptx, instrumentation.accesses)
    raw_edges = build_raw_edges(static_ranges)
    execution = maybe_execute_probe(
        args,
        instrumented_path=instrumented_path,
        access_count=len(instrumentation.accesses),
    )
    report = {
        "status": "ok",
        "tool": "ptx_memory_probe",
        "input_ptx": str(original_path),
        "instrumented_ptx": str(instrumented_path),
        "access_csv": str(args.output_dir / "memory_accesses.csv"),
        "instrumentation": {
            "inst_buffer_param": instrumentation.inst_buffer_param,
            "record_bytes": instrumentation.record_bytes,
            "access_count": len(instrumentation.accesses),
            "accesses": [asdict(access) for access in instrumentation.accesses],
            "inserted_atomics_per_access": 2,
        },
        "ptx_features": {
            "original": asdict(analyze_ptx(ptx)),
            "instrumented": asdict(analyze_ptx(instrumentation.instrumented_ptx)),
        },
        "static_ranges": static_ranges,
        "raw_ddg": {"edges": raw_edges},
        "execution": execution,
        "runtime_range_check": summarize_runtime_ranges(execution),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def write_access_csv(path: Path, accesses: tuple[PTXMemoryAccess, ...]) -> None:
    with path.open("w", newline="") as f:
        fieldnames = [
            "access_id",
            "entry_name",
            "line_number",
            "mnemonic",
            "access_kind",
            "address_expr",
            "width_bytes",
            "original_line",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for access in accesses:
            writer.writerow(asdict(access))


def infer_static_ranges(
    ptx: str,
    accesses: tuple[PTXMemoryAccess, ...],
) -> list[dict[str, Any]]:
    param_registers = infer_param_registers(ptx)
    ranges: list[dict[str, Any]] = []
    for access in accesses:
        base_reg, offset = split_address_expr(access.address_expr)
        param_name = param_registers.get(access.entry_name, {}).get(base_reg)
        buffer_name = SAMPLE_PARAM_BUFFERS.get(access.entry_name, {}).get(param_name or "")
        base_addr = SAMPLE_BUFFER_BASES.get(buffer_name or "")
        if base_addr is None:
            min_addr = None
            max_addr = None
        else:
            min_addr = base_addr + offset
            max_addr = min_addr + access.width_bytes - 1
        ranges.append(
            {
                "access_id": access.access_id,
                "entry_name": access.entry_name,
                "kind": access.access_kind,
                "buffer": buffer_name or "unknown",
                "param": param_name or "unknown",
                "address_expr": access.address_expr,
                "width_bytes": access.width_bytes,
                "min_addr": min_addr,
                "max_addr": max_addr,
            }
        )
    return ranges


def infer_param_registers(ptx: str) -> dict[str, dict[str, str]]:
    by_entry: dict[str, dict[str, str]] = {}
    current_entry: str | None = None
    for raw_line in ptx.splitlines():
        stripped = raw_line.strip()
        if ".entry" in stripped:
            name = stripped.split(".entry", 1)[1].split("(", 1)[0].strip()
            current_entry = name
            by_entry.setdefault(name, {})
            continue
        if current_entry is None:
            continue
        if stripped.startswith("}"):
            current_entry = None
            continue
        if not stripped.startswith("ld.param.u64"):
            continue
        left, right = stripped.rstrip(";").split(",", 1)
        reg = left.split()[-1]
        param = right.strip().strip("[]")
        by_entry[current_entry][reg] = param
    return by_entry


def split_address_expr(address_expr: str) -> tuple[str, int]:
    compact = address_expr.replace(" ", "")
    for op in ("+", "-"):
        if op in compact[1:]:
            reg, raw_offset = compact.split(op, 1)
            offset = int(raw_offset, 0)
            return reg, offset if op == "+" else -offset
    return compact, 0


def build_raw_edges(ranges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    last_writer: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for item in ranges:
        buffer_name = item["buffer"]
        if buffer_name == "unknown":
            continue
        if item["kind"] == "read" and buffer_name in last_writer:
            writer = last_writer[buffer_name]
            if writer["entry_name"] != item["entry_name"]:
                edges.append(
                    {
                        "src": writer["entry_name"],
                        "dst": item["entry_name"],
                        "buffer": buffer_name,
                        "write_access_id": writer["access_id"],
                        "read_access_id": item["access_id"],
                    }
                )
        if item["kind"] == "write":
            last_writer[buffer_name] = item
    return edges


def maybe_execute_probe(
    args: argparse.Namespace,
    *,
    instrumented_path: Path,
    access_count: int,
) -> dict[str, Any]:
    if args.input_ptx is not None:
        raise RuntimeError(
            "execution runner supports the built-in producer/consumer PTX only; "
            "remove --input-ptx or provide a matching runner"
        )

    missing = [tool for tool in ("ptxas", "nvcc") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError("missing required CUDA tools: " + ", ".join(missing))

    build_dir = args.output_dir / "build"
    build_dir.mkdir(exist_ok=True)
    cubin_path = build_dir / "instrumented.cubin"
    runner_path = build_dir / "ptx_memory_probe_runner"
    commands = [
        ["ptxas", f"-arch={args.arch}", "-o", str(cubin_path), str(instrumented_path)],
        [
            "nvcc",
            "-std=c++17",
            str(args.runner_source),
            "-lcuda",
            "-o",
            str(runner_path),
        ],
    ]
    for command in commands:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            status = {
                "status": "failed",
                "command": command,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
            raise RuntimeError(json.dumps(status, indent=2))

    result = subprocess.run(
        [str(runner_path), str(cubin_path), str(access_count)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        status = {
            "status": "failed",
            "command": [str(runner_path), str(cubin_path), str(access_count)],
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
        raise RuntimeError(json.dumps(status, indent=2))
    return json.loads(result.stdout)


def summarize_runtime_ranges(execution: dict[str, Any]) -> dict[str, Any]:
    if execution.get("status") != "ok":
        return {
            "status": execution.get("status", "unknown"),
            "matched": None,
            "checked_accesses": 0,
        }
    ranges = execution.get("ranges", [])
    checked = [item for item in ranges if "match" in item]
    return {
        "status": "ok",
        "matched": all(bool(item["match"]) for item in checked),
        "checked_accesses": len(checked),
    }


def parse_args() -> argparse.Namespace:
    default_runner = Path(__file__).with_name("ptx_memory_probe_runner.cu")
    parser = argparse.ArgumentParser(description="PTX memory instrumentation and RAW-DDG probe")
    parser.add_argument("--input-ptx", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execute", choices=("required",), default="required")
    parser.add_argument("--arch", default="sm_80")
    parser.add_argument("--runner-source", type=Path, default=default_runner)
    return parser.parse_args()


if __name__ == "__main__":
    main()
