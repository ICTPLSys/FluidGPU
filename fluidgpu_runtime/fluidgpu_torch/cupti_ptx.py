from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


FINE_GROUPS = ("qkv", "sdpa", "o_proj", "mlp", "other")
ATTENTION_FINE_GROUPS = ("qkv", "sdpa", "o_proj")


@dataclass(frozen=True)
class KernelRecord:
    name: str
    duration_us: float
    layer_id: int | None = None
    phase: str | None = None
    ordinal: int | None = None
    stream_id: str | None = None


@dataclass(frozen=True)
class KernelLabel:
    group: str
    confidence: float
    reason: str


@dataclass(frozen=True)
class PTXFeatures:
    entry_names: tuple[str, ...]
    instruction_counts: dict[str, int]
    total_instructions: int
    tensor_core_instructions: int
    memory_instructions: int
    barrier_instructions: int
    shared_memory_refs: int
    register_count: int | None

    @property
    def compute_kind(self) -> str:
        if self.tensor_core_instructions > 0:
            return "tensor_core"
        if self.memory_instructions > self.total_instructions * 0.5:
            return "memory"
        if self.barrier_instructions > 0 or self.shared_memory_refs > 0:
            return "cooperative"
        return "scalar"


@dataclass(frozen=True)
class PTXMemoryAccess:
    access_id: int
    entry_name: str
    line_number: int
    mnemonic: str
    access_kind: str
    address_expr: str
    width_bytes: int
    original_line: str


@dataclass(frozen=True)
class PTXInstrumentation:
    instrumented_ptx: str
    accesses: tuple[PTXMemoryAccess, ...]
    inst_buffer_param: str
    record_bytes: int = 16


@dataclass(frozen=True)
class LabeledKernel:
    record: KernelRecord
    label: KernelLabel
    ptx_features: PTXFeatures | None = None


@dataclass(frozen=True)
class GroupSummary:
    phase: str
    layer_id: int | None
    group: str
    kernel_count: int
    total_us: float


@dataclass(frozen=True)
class AnalysisResult:
    kernels: tuple[LabeledKernel, ...]
    summaries: tuple[GroupSummary, ...]

    @property
    def total_us(self) -> float:
        return sum(item.record.duration_us for item in self.kernels)

    @property
    def group_totals_us(self) -> dict[str, float]:
        totals = {group: 0.0 for group in FINE_GROUPS}
        for item in self.kernels:
            totals[item.label.group] += item.record.duration_us
        return totals

    @property
    def attention_split_weights(self) -> dict[str, float]:
        totals = self.group_totals_us
        denominator = sum(totals[group] for group in ATTENTION_FINE_GROUPS)
        if denominator == 0.0:
            return {group: 0.0 for group in ATTENTION_FINE_GROUPS}
        return {group: totals[group] / denominator for group in ATTENTION_FINE_GROUPS}


def parse_cupti_csv(
    path: str | Path,
    *,
    name_column: str | None = None,
    duration_column: str | None = None,
    duration_unit: str = "auto",
) -> tuple[KernelRecord, ...]:
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames is not None, f"{path} has no CSV header"
        fields = tuple(reader.fieldnames)
        name_col = name_column or first_existing(
            fields,
            ("kernel_name", "Kernel Name", "Name", "name", "Kernel"),
        )
        duration_col = duration_column or first_existing(
            fields,
            (
                "duration_us",
                "Duration (us)",
                "Duration (ns)",
                "Duration (nsec)",
                "Duration (msec)",
                "Duration",
                "gpu_time_us",
            ),
        )
        phase_col = optional_existing(fields, ("phase", "Phase"))
        layer_col = optional_existing(fields, ("layer_id", "Layer ID", "layer", "Layer"))
        stream_col = optional_existing(fields, ("stream_id", "Stream", "Stream ID"))
        records: list[KernelRecord] = []
        for ordinal, row in enumerate(reader):
            records.append(
                KernelRecord(
                    name=row[name_col],
                    duration_us=parse_duration_us(
                        row[duration_col],
                        column_name=duration_col,
                        unit=duration_unit,
                    ),
                    layer_id=parse_optional_int(row.get(layer_col) if layer_col else None),
                    phase=parse_optional_str(row.get(phase_col) if phase_col else None),
                    ordinal=ordinal,
                    stream_id=parse_optional_str(row.get(stream_col) if stream_col else None),
                )
            )
        return tuple(records)


def analyze_records(
    records: Iterable[KernelRecord],
    *,
    ptx_by_kernel: Mapping[str, str] | None = None,
) -> AnalysisResult:
    ptx_by_kernel = ptx_by_kernel or {}
    labeled: list[LabeledKernel] = []
    for record in records:
        ptx = lookup_ptx(record.name, ptx_by_kernel)
        features = analyze_ptx(ptx) if ptx is not None else None
        labeled.append(
            LabeledKernel(
                record=record,
                label=classify_kernel(record.name, ptx_features=features),
                ptx_features=features,
            )
        )
    return AnalysisResult(kernels=tuple(labeled), summaries=summarize_groups(labeled))


def classify_kernel(
    name: str,
    *,
    ptx_features: PTXFeatures | None = None,
) -> KernelLabel:
    lower = name.lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", lower)

    marker_groups = (
        (
            "o_proj",
            (
                "o_proj",
                "out_proj",
                "output_proj",
                "attention_out",
                "attn_out",
                "self_attn_o",
            ),
        ),
        (
            "qkv",
            (
                "qkv",
                "query_key_value",
                "querykeyvalue",
                "fused_qkv",
                "in_proj",
                "q_proj",
                "k_proj",
                "v_proj",
                "wqkv",
            ),
        ),
        (
            "sdpa",
            (
                "sdpa",
                "scaled_dot",
                "flash_attn",
                "flash_attention",
                "fmha",
                "paged_attention",
                "attention_kernel",
                "softmax",
            ),
        ),
        (
            "mlp",
            (
                "mlp",
                "ffn",
                "feed_forward",
                "gate_proj",
                "up_proj",
                "down_proj",
                "swiglu",
                "silu",
                "moe",
                "expert",
            ),
        ),
    )
    for group, markers in marker_groups:
        marker = first_marker(normalized, markers)
        if marker is not None:
            return KernelLabel(group=group, confidence=0.95, reason=f"name:{marker}")

    if ptx_features is not None:
        return KernelLabel(
            group="other",
            confidence=0.25,
            reason=f"ptx:{ptx_features.compute_kind}",
        )
    return KernelLabel(group="other", confidence=0.0, reason="unmatched")


def analyze_ptx(ptx: str) -> PTXFeatures:
    instruction_counts: dict[str, int] = defaultdict(int)
    entry_names: list[str] = []
    register_count: int | None = None
    shared_refs = 0
    for raw_line in ptx.splitlines():
        line = strip_ptx_comment(raw_line).strip()
        if not line:
            continue
        entry_match = re.search(r"\.entry\s+([A-Za-z_.$][\w.$]*)\s*\(", line)
        if entry_match is not None:
            entry_names.append(entry_match.group(1))
            continue
        reg_match = re.search(r"\.reg\s+\.\w+\s+%?\w+<(\d+)>", line)
        if reg_match is not None:
            register_count = max(register_count or 0, int(reg_match.group(1)))
            continue
        if line.startswith("."):
            shared_refs += line.count(".shared")
            continue
        mnemonic = parse_ptx_mnemonic(line)
        if mnemonic is None:
            continue
        instruction_counts[mnemonic] += 1
        shared_refs += line.count(".shared")

    total = sum(instruction_counts.values())
    tensor_core = sum(
        count
        for mnemonic, count in instruction_counts.items()
        if mnemonic.startswith("mma") or mnemonic.startswith("wgmma")
    )
    memory = sum(
        count
        for mnemonic, count in instruction_counts.items()
        if mnemonic.startswith(("ld.", "st.", "cp.async", "ldmatrix"))
    )
    barrier = sum(
        count
        for mnemonic, count in instruction_counts.items()
        if mnemonic.startswith(("bar.", "mbarrier"))
    )
    return PTXFeatures(
        entry_names=tuple(entry_names),
        instruction_counts=dict(sorted(instruction_counts.items())),
        total_instructions=total,
        tensor_core_instructions=tensor_core,
        memory_instructions=memory,
        barrier_instructions=barrier,
        shared_memory_refs=shared_refs,
        register_count=register_count,
    )


def instrument_ptx_memory_ranges(
    ptx: str,
    *,
    inst_buffer_param: str = "fluidgpu_inst_buf",
) -> PTXInstrumentation:
    lines = _add_instrumentation_param(ptx.splitlines(), inst_buffer_param)
    output: list[str] = []
    accesses: list[PTXMemoryAccess] = []
    current_entry = ""
    pending_body_init = False
    body_init_inserted = False

    for line_number, raw_line in enumerate(lines, start=1):
        stripped = strip_ptx_comment(raw_line).strip()
        entry_match = re.search(r"\.entry\s+([A-Za-z_.$][\w.$]*)\s*\(", stripped)
        if entry_match is not None:
            current_entry = entry_match.group(1)
            body_init_inserted = False

        output.append(raw_line)
        # Only a block-opening brace starts the body; instruction lines end in
        # ';' and may carry braces of their own (vector operand lists, e.g.
        # ld.global.v4.f32 {%f1,...},[%rd1];) and must reach the
        # instrumentation stage below.
        if (
            "{" in stripped
            and current_entry
            and not body_init_inserted
            and ";" not in stripped
        ):
            pending_body_init = True
            continue

        if pending_body_init and not body_init_inserted:
            if _is_body_declaration(stripped) or not stripped:
                continue
            output.insert(
                len(output) - 1,
                "    .reg .b64 %fluidgpu_inst, %fluidgpu_addr, %fluidgpu_end, %fluidgpu_slot, %fluidgpu_unused;",
            )
            output.insert(
                len(output) - 1,
                f"    ld.param.u64 %fluidgpu_inst, [{inst_buffer_param}];",
            )
            body_init_inserted = True
            pending_body_init = False

        if stripped == "}":
            current_entry = ""
            pending_body_init = False
            body_init_inserted = False
            continue

        guard_match = re.match(r"@!?%[A-Za-z_.$][\w.$]*", stripped)
        guard = guard_match.group(0) if guard_match is not None else ""
        mnemonic = parse_ptx_mnemonic(stripped)
        if mnemonic is None or not _is_global_memory_mnemonic(mnemonic):
            continue
        address_expr = _memory_address_expr(stripped)
        if address_expr is None:
            continue

        access_id = len(accesses)
        kind = "write" if mnemonic.startswith("st.global") else "read"
        width_bytes = ptx_memory_width_bytes(mnemonic)
        accesses.append(
            PTXMemoryAccess(
                access_id=access_id,
                entry_name=current_entry,
                line_number=line_number,
                mnemonic=mnemonic,
                access_kind=kind,
                address_expr=address_expr,
                width_bytes=width_bytes,
                original_line=raw_line.strip(),
            )
        )
        insert_at = len(output) - 1
        output[insert_at:insert_at] = _instrumentation_lines(
            access_id,
            address_expr,
            width_bytes=width_bytes,
            guard=guard,
        )

    return PTXInstrumentation(
        instrumented_ptx="\n".join(output) + "\n",
        accesses=tuple(accesses),
        inst_buffer_param=inst_buffer_param,
    )


def ptx_memory_width_bytes(mnemonic: str) -> int:
    base_width = 4
    for marker, width in (
        (".b8", 1),
        (".u8", 1),
        (".s8", 1),
        (".b16", 2),
        (".u16", 2),
        (".s16", 2),
        (".f16", 2),
        (".bf16", 2),
        (".b32", 4),
        (".u32", 4),
        (".s32", 4),
        (".f32", 4),
        (".b64", 8),
        (".u64", 8),
        (".s64", 8),
        (".f64", 8),
    ):
        if marker in mnemonic:
            base_width = width
            break
    vector_match = re.search(r"\.v([248])\.", mnemonic)
    if vector_match is not None:
        return base_width * int(vector_match.group(1))
    return base_width


def summarize_groups(kernels: Iterable[LabeledKernel]) -> tuple[GroupSummary, ...]:
    totals: dict[tuple[str, int | None, str], list[float]] = {}
    for item in kernels:
        key = (item.record.phase or "unknown", item.record.layer_id, item.label.group)
        bucket = totals.setdefault(key, [0.0, 0.0])
        bucket[0] += 1.0
        bucket[1] += item.record.duration_us
    out = [
        GroupSummary(
            phase=phase,
            layer_id=layer_id,
            group=group,
            kernel_count=int(values[0]),
            total_us=values[1],
        )
        for (phase, layer_id, group), values in totals.items()
    ]
    return tuple(sorted(out, key=lambda item: (item.phase, item.layer_id or -1, item.group)))


def load_ptx_dir(path: str | Path) -> dict[str, str]:
    root = Path(path)
    assert root.is_dir(), f"PTX path must be a directory: {root}"
    out: dict[str, str] = {}
    for ptx_path in sorted(root.glob("*.ptx")):
        text = ptx_path.read_text()
        out[ptx_path.stem] = text
        features = analyze_ptx(text)
        for entry in features.entry_names:
            out[entry] = text
            out[sanitize_kernel_name(entry)] = text
    return out


def lookup_ptx(name: str, ptx_by_kernel: Mapping[str, str]) -> str | None:
    return (
        ptx_by_kernel.get(name)
        or ptx_by_kernel.get(sanitize_kernel_name(name))
        or ptx_by_kernel.get(Path(name).stem)
    )


def write_analysis_json(path: str | Path, result: AnalysisResult) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(analysis_to_dict(result), indent=2) + "\n")


def write_labeled_csv(path: str | Path, result: AnalysisResult) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "ordinal",
        "phase",
        "layer_id",
        "name",
        "duration_us",
        "group",
        "confidence",
        "reason",
        "ptx_compute_kind",
    ]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in result.kernels:
            writer.writerow(labeled_kernel_to_row(item))


def analysis_to_dict(result: AnalysisResult) -> dict[str, Any]:
    return {
        "kernel_count": len(result.kernels),
        "total_us": result.total_us,
        "group_totals_us": result.group_totals_us,
        "attention_split_weights": result.attention_split_weights,
        "summaries": [
            {
                "phase": summary.phase,
                "layer_id": summary.layer_id,
                "group": summary.group,
                "kernel_count": summary.kernel_count,
                "total_us": summary.total_us,
            }
            for summary in result.summaries
        ],
        "kernels": [labeled_kernel_to_row(item) for item in result.kernels],
    }


def labeled_kernel_to_row(item: LabeledKernel) -> dict[str, Any]:
    return {
        "ordinal": item.record.ordinal,
        "phase": item.record.phase or "",
        "layer_id": item.record.layer_id,
        "name": item.record.name,
        "duration_us": item.record.duration_us,
        "group": item.label.group,
        "confidence": item.label.confidence,
        "reason": item.label.reason,
        "ptx_compute_kind": item.ptx_features.compute_kind if item.ptx_features else "",
    }


def first_existing(fields: Iterable[str], candidates: Iterable[str]) -> str:
    fields_set = set(fields)
    for candidate in candidates:
        if candidate in fields_set:
            return candidate
    raise AssertionError(f"missing any of columns: {', '.join(candidates)}")


def optional_existing(fields: Iterable[str], candidates: Iterable[str]) -> str | None:
    fields_set = set(fields)
    for candidate in candidates:
        if candidate in fields_set:
            return candidate
    return None


def parse_duration_us(value: str, *, column_name: str, unit: str) -> float:
    raw = float(value.replace(",", "").strip())
    normalized_unit = unit.lower()
    assert normalized_unit in ("auto", "ns", "us", "ms"), f"unsupported duration unit {unit}"
    if normalized_unit == "auto":
        lower = column_name.lower()
        if "ns" in lower:
            normalized_unit = "ns"
        elif "ms" in lower:
            normalized_unit = "ms"
        else:
            normalized_unit = "us"
    if normalized_unit == "ns":
        return raw / 1000.0
    if normalized_unit == "ms":
        return raw * 1000.0
    return raw


def parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def parse_optional_str(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def first_marker(value: str, markers: Iterable[str]) -> str | None:
    for marker in markers:
        normalized_marker = re.sub(r"[^a-z0-9]+", "_", marker.lower()).strip("_")
        if normalized_marker in value:
            return marker
    return None


def strip_ptx_comment(line: str) -> str:
    return line.split("//", 1)[0]


def parse_ptx_mnemonic(line: str) -> str | None:
    while line.startswith("@"):
        parts = line.split(None, 1)
        if len(parts) != 2:
            return None
        line = parts[1].strip()
    match = re.match(r"([A-Za-z][A-Za-z0-9_.]*)\b", line)
    if match is None:
        return None
    return match.group(1)


def sanitize_kernel_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def _add_instrumentation_param(lines: list[str], inst_buffer_param: str) -> list[str]:
    out: list[str] = []
    in_params = False
    last_param_index: int | None = None
    for raw_line in lines:
        stripped = strip_ptx_comment(raw_line).strip()
        if ".entry" in stripped and "(" in raw_line and ")" in raw_line:
            before_params, after_open = raw_line.split("(", 1)
            params, after_close = after_open.split(")", 1)
            out.append(before_params + "(")
            for param in [item.strip() for item in params.split(",") if item.strip()]:
                out.append(f"    {param.rstrip(',')},")
            out.append(f"    .param .u64 {inst_buffer_param}")
            out.append(")" + after_close)
            continue
        if ".entry" in stripped and "(" in stripped and ")" not in stripped:
            in_params = True
            last_param_index = None

        if in_params and stripped.startswith(".param"):
            last_param_index = len(out)

        if in_params and stripped.startswith(")"):
            if last_param_index is not None and not out[last_param_index].rstrip().endswith(","):
                out[last_param_index] = out[last_param_index].rstrip() + ","
            out.append(f"    .param .u64 {inst_buffer_param}")
            in_params = False

        out.append(raw_line)
    return out


def _is_body_declaration(stripped: str) -> bool:
    return stripped.startswith((".reg", ".local", ".shared", ".param", ".pragma"))


def _is_global_memory_mnemonic(mnemonic: str) -> bool:
    return (
        mnemonic.startswith(("ld.global", "st.global"))
        and not mnemonic.startswith("atom.global")
    )


def _memory_address_expr(line: str) -> str | None:
    match = re.search(r"\[([^\]]+)\]", line)
    if match is None:
        return None
    return match.group(1).strip()


def _instrumentation_lines(
    access_id: int, address_expr: str, *, width_bytes: int, guard: str = ""
) -> list[str]:
    slot_offset = access_id * 16
    end_line = (
        "    mov.u64 %fluidgpu_end, %fluidgpu_addr;"
        if width_bytes == 1
        else f"    add.u64 %fluidgpu_end, %fluidgpu_addr, {width_bytes - 1};"
    )

    def predicated(line: str) -> str:
        # An access guarded by @%p must be recorded under the same predicate:
        # unpredicated instrumentation would log addresses that never execute
        # (the address register can hold an out-of-range value when the guard
        # is false), inflating the observed ranges.
        if not guard:
            return line
        indent = line[: len(line) - len(line.lstrip())]
        return f"{indent}{guard} {line.lstrip()}"

    return [
        f"    // FluidGPU instrumentation for memory access {access_id}",
        *(predicated(line) for line in _address_to_register_lines(address_expr)),
        predicated(end_line),
        predicated(f"    add.u64 %fluidgpu_slot, %fluidgpu_inst, {slot_offset};"),
        predicated("    atom.global.min.u64 %fluidgpu_unused, [%fluidgpu_slot], %fluidgpu_addr;"),
        predicated("    atom.global.max.u64 %fluidgpu_unused, [%fluidgpu_slot+8], %fluidgpu_end;"),
    ]


def _address_to_register_lines(address_expr: str) -> list[str]:
    compact = address_expr.replace(" ", "")
    direct = re.fullmatch(r"%[A-Za-z_.$][\w.$]*", compact)
    if direct is not None:
        return [f"    mov.u64 %fluidgpu_addr, {compact};"]
    offset = re.fullmatch(r"(%[A-Za-z_.$][\w.$]*)([+-]\d+)", compact)
    if offset is not None:
        return [f"    add.u64 %fluidgpu_addr, {offset.group(1)}, {offset.group(2)};"]
    raise AssertionError(f"unsupported PTX memory address expression: {address_expr}")
