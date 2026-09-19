from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_ATTN_SPLIT = "qkv=0.35,sdpa=0.45,o_proj=0.20"


@dataclass(frozen=True)
class SplitWeight:
    suffix: str
    weight: float


def main() -> None:
    args = parse_args()
    splits = parse_split_weights(args.attn_split)
    if args.input_profile is not None:
        raw = json.loads(args.input_profile.read_text())
        refined = refine_profile(raw, splits)
        args.output_profile.parent.mkdir(parents=True, exist_ok=True)
        args.output_profile.write_text(json.dumps(refined, indent=2) + "\n")
        print(
            "refined_profile "
            f"tasks={refined['task_count']} output={args.output_profile}"
        )
    if args.input_csv is not None:
        rows = read_csv_rows(args.input_csv)
        refined_rows = refine_csv_rows(rows, splits)
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv_rows(args.output_csv, refined_rows)
        print(f"refined_csv rows={len(refined_rows)} output={args.output_csv}")


def parse_split_weights(value: str) -> tuple[SplitWeight, ...]:
    splits: list[SplitWeight] = []
    for item in value.split(","):
        name, raw_weight = item.split("=", 1)
        name = name.strip()
        weight = float(raw_weight)
        assert name, f"empty split suffix in {value}"
        assert weight > 0.0, f"split weight for {name} must be positive"
        splits.append(SplitWeight(name, weight))
    total = sum(split.weight for split in splits)
    assert abs(total - 1.0) < 1e-6, f"attention split weights must sum to 1, got {total}"
    assert len({split.suffix for split in splits}) == len(splits), "split suffixes must be unique"
    return tuple(splits)


def refine_profile(raw: dict[str, Any], splits: tuple[SplitWeight, ...]) -> dict[str, Any]:
    tasks: list[dict[str, int | str | float]] = []
    for task in raw["tasks"]:
        for refined in refine_task(task, splits):
            refined["task_id"] = len(tasks)
            refined["kernel_group_id"] = len(tasks)
            tasks.append(refined)
    out = dict(raw)
    out["task_count"] = len(tasks)
    out["tasks"] = tasks
    out["fine_grained"] = {
        "runtime_compatible": True,
        "source_granularity": "attn_mlp",
        "attention_splits": {split.suffix: split.weight for split in splits},
        "note": "Runtime-compatible attention split; dense MLP remains a single mlp task.",
    }
    return out


def refine_task(
    task: dict[str, Any],
    splits: tuple[SplitWeight, ...],
) -> list[dict[str, int | str | float]]:
    name = str(task["name"])
    parsed = parse_layer_task_name(name)
    if parsed is None:
        return [copy_task(task)]
    layer_id, suffix = parsed
    if suffix == "attn":
        refined: list[dict[str, int | str | float]] = []
        for split in splits:
            item = copy_task(task)
            item["name"] = f"layer_{layer_id}_{split.suffix}"
            item["parent_task"] = name
            item["split_weight"] = split.weight
            refined.append(item)
        return refined
    if suffix == "mlp":
        return [copy_task(task)]
    raise AssertionError(f"unsupported layer task {name}")


def copy_task(task: dict[str, Any]) -> dict[str, int | str | float]:
    item: dict[str, int | str | float] = {
        "task_id": int(task["task_id"]),
        "kernel_group_id": int(task["kernel_group_id"]),
        "name": str(task["name"]),
        "rank": int(task["rank"]),
    }
    return item


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def refine_csv_rows(
    rows: list[dict[str, str]],
    splits: tuple[SplitWeight, ...],
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for row in rows:
        out.extend(refine_csv_row(row, splits))
    for idx, row in enumerate(out):
        if "task_id" in row:
            row["task_id"] = str(idx)
        if "kernel_group_id" in row:
            row["kernel_group_id"] = str(idx)
    return out


def refine_csv_row(
    row: dict[str, str],
    splits: tuple[SplitWeight, ...],
) -> list[dict[str, str]]:
    name = row["name"]
    parsed = parse_layer_task_name(name)
    if parsed is None:
        return [dict(row)]
    layer_id, suffix = parsed
    if suffix != "attn":
        return [dict(row)]
    refined: list[dict[str, str]] = []
    for split in splits:
        item = dict(row)
        item["name"] = f"layer_{layer_id}_{split.suffix}"
        scale_float_columns(item, split.weight, ("prefill_us", "decode_us"))
        if "parent_task" in item:
            item["parent_task"] = name
        if "split_weight" in item:
            item["split_weight"] = f"{split.weight:.9g}"
        refined.append(item)
    return refined


def scale_float_columns(row: dict[str, str], weight: float, columns: tuple[str, ...]) -> None:
    for column in columns:
        if column in row:
            row[column] = f"{float(row[column]) * weight:.9f}"


def write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    assert rows, "no rows to write"
    fieldnames = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_layer_task_name(name: str) -> tuple[int, str] | None:
    if not name.startswith("layer_"):
        return None
    parts = name.split("_")
    assert len(parts) >= 3, f"expected layer_<id>_<suffix>, got {name}"
    assert parts[1].isdigit(), f"invalid layer id in {name}"
    suffix = "_".join(parts[2:])
    if suffix == "o":
        raise AssertionError("use layer_<id>_o_proj instead of layer_<id>_o")
    return int(parts[1]), suffix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Refine LLM attn/mlp kernel groups")
    parser.add_argument("--input-profile", type=Path)
    parser.add_argument("--output-profile", type=Path)
    parser.add_argument("--input-csv", type=Path)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--attn-split", default=DEFAULT_ATTN_SPLIT)
    args = parser.parse_args()
    assert args.input_profile is not None or args.input_csv is not None, (
        "provide --input-profile or --input-csv"
    )
    if args.input_profile is not None:
        assert args.output_profile is not None, "--output-profile is required with --input-profile"
    if args.input_csv is not None:
        assert args.output_csv is not None, "--output-csv is required with --input-csv"
    return args


if __name__ == "__main__":
    main()
