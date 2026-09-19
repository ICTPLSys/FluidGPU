from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
from itertools import product
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExpandedRun:
    figure: str
    run_id: str
    repeat: int
    kind: str
    command: str
    output_dir: Path
    requires_gpus: tuple[str, ...]
    payload: dict[str, Any]


def load_spec(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        raw = yaml.safe_load(f)
    assert isinstance(raw, dict), f"{path} must contain a YAML mapping"
    assert "figure" in raw, f"{path} is missing figure"
    assert "runs" in raw and isinstance(raw["runs"], list), f"{path} is missing runs"
    return raw


def expand_runs(
    spec: dict[str, Any],
    *,
    only: str | None = None,
    repeat_override: int | None = None,
) -> list[ExpandedRun]:
    figure = str(spec["figure"])
    base_output = Path(spec.get("artifact_root", "artifacts/logs")) / figure
    default_repeat = int(spec.get("repeat", 1))
    runs: list[ExpandedRun] = []
    for row in spec["runs"]:
        assert isinstance(row, dict), f"{figure} run rows must be mappings"
        for expanded_row in expand_row_matrix(row):
            row_id = str(expanded_row["id"])
            if only is not None and row_id != only:
                continue
            repeat_count = repeat_override if repeat_override is not None else int(expanded_row.get("repeat", default_repeat))
            for repeat in range(repeat_count):
                payload = {
                    k: v
                    for k, v in expanded_row.items()
                    if k not in {"command", "command_template", "kind"}
                }
                run_id = row_id if repeat_count == 1 else f"{row_id}_r{repeat}"
                command = str(expanded_row.get("command", ""))
                runs.append(
                    ExpandedRun(
                        figure=figure,
                        run_id=run_id,
                        repeat=repeat,
                        kind=str(expanded_row.get("kind", "command")),
                        command=command,
                        output_dir=base_output / run_id,
                        requires_gpus=tuple(str(item) for item in expanded_row.get("requires_gpus", ())),
                        payload=payload,
                    )
                )
    return runs


def expand_row_matrix(row: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = row.get("matrix")
    if matrix is None:
        return [row]
    assert isinstance(matrix, dict), "matrix must be a mapping"
    keys = list(matrix)
    combos = product(*(matrix[key] for key in keys))
    rows: list[dict[str, Any]] = []
    for combo_values in combos:
        combo: dict[str, Any] = {}
        for key, value in zip(keys, combo_values):
            if isinstance(value, dict):
                combo.update(value)
            else:
                combo[key] = value
        merged = {k: v for k, v in row.items() if k != "matrix"}
        merged.update(combo)
        fmt = _FormatDict(merged)
        merged["id"] = str(row.get("id_template", row["id"])).format_map(fmt)
        if "command_template" in row:
            merged["command"] = str(row["command_template"]).format_map(fmt)
        if "requires_gpus" in merged:
            merged["requires_gpus"] = [
                str(item).format_map(fmt) for item in merged["requires_gpus"]
            ]
        rows.append(merged)
    return rows


class _FormatDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def run_expanded(expanded: ExpandedRun) -> bool:
    environment_error = hardware_error_reason(expanded.requires_gpus)
    expanded.output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "figure": expanded.figure,
        "run_id": expanded.run_id,
        "kind": expanded.kind,
        "command": expanded.command,
        "payload": expanded.payload,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment_fingerprint(),
        "environment_error": environment_error,
    }
    (expanded.output_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    if environment_error:
        message = (
            f"{expanded.figure}/{expanded.run_id}: environment check failed: "
            f"{environment_error}"
        )
        (expanded.output_dir / "log.jsonl").write_text(
            json.dumps({"status": "failed", "reason": environment_error}) + "\n"
        )
        sys.stderr.write(message + "\n")
        return False

    command_result = execute_shell_command(
        expanded.command,
        cwd=repo_root(),
        output_dir=expanded.output_dir,
    )
    (expanded.output_dir / "stdout.log").write_text(command_result["stdout"])
    (expanded.output_dir / "stderr.log").write_text(command_result["stderr"])
    succeeded = command_result["returncode"] == 0
    status_path = expanded.output_dir / "run_status.jsonl"
    if not (expanded.output_dir / "log.jsonl").exists():
        status_path = expanded.output_dir / "log.jsonl"
    status_path.write_text(
        json.dumps(
            {
                "status": "ok" if succeeded else "failed",
                "command": expanded.command,
                "returncode": command_result["returncode"],
                "elapsed_s": command_result["elapsed_s"],
            }
        )
        + "\n"
    )
    outcome = "wrote" if succeeded else "FAILED (see stderr.log) at"
    print(f"{expanded.figure}/{expanded.run_id}: {outcome} {expanded.output_dir}")
    return succeeded


def hardware_error_reason(required: tuple[str, ...]) -> str | None:
    if not required:
        return None
    available = available_gpu_names()
    missing = [name for name in required if not any(name.lower() in gpu.lower() for gpu in available)]
    if missing:
        return "missing GPUs: " + ", ".join(missing)
    return None


def available_gpu_names() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def environment_fingerprint() -> dict[str, Any]:
    return {
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "gpus": available_gpu_names(),
        "git_commit": git_commit(),
    }


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root(),
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def execute_shell_command(command: str, *, cwd: Path, output_dir: Path) -> dict[str, Any]:
    env = os.environ.copy()
    runtime_root = str(cwd / "fluidgpu_runtime")
    env["PYTHONPATH"] = runtime_root + os.pathsep + env.get("PYTHONPATH", "")
    env["FLUIDGPU_RUN_OUTPUT_DIR"] = str(output_dir)
    env_overrides, argv = split_env_command(command)
    env.update(env_overrides)
    start = datetime.now(timezone.utc)
    started_ns = time_monotonic_ns()
    result = subprocess.run(
        argv,
        cwd=cwd,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    elapsed_s = (time_monotonic_ns() - started_ns) / 1e9
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "started_at": start.isoformat(),
        "elapsed_s": elapsed_s,
    }


_ENV_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=.*$")


def split_env_command(command: str) -> tuple[dict[str, str], list[str]]:
    tokens = shlex.split(command)
    env: dict[str, str] = {}
    index = 0
    while index < len(tokens) and _ENV_ASSIGN_RE.match(tokens[index]):
        key, value = tokens[index].split("=", 1)
        env[key] = value
        index += 1
    argv = tokens[index:]
    if not argv:
        raise SystemExit(f"invalid command with no executable: {command!r}")
    return env, argv


def time_monotonic_ns() -> int:
    import time

    return time.monotonic_ns()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def main(args: argparse.Namespace | None = None) -> None:
    if args is None:
        parser = argparse.ArgumentParser(description="Run a FluidGPU experiment YAML spec")
        parser.add_argument("spec")
        parser.add_argument("--only")
        parser.add_argument("--repeat", type=int)
        args = parser.parse_args()
    spec = load_spec(args.spec)
    runs = expand_runs(spec, only=args.only, repeat_override=args.repeat)
    # One failing run (missing weights, absent GPU, crashed command) must not
    # abort the remaining grid; record it, keep going, and fail at the end.
    failed: list[str] = []
    for expanded in runs:
        if not run_expanded(expanded):
            failed.append(f"{expanded.figure}/{expanded.run_id}")
    if failed:
        sys.stderr.write(
            f"orchestrator: {len(failed)}/{len(runs)} runs failed: "
            + ", ".join(failed)
            + "\n"
        )
        raise SystemExit(1)
    print(f"orchestrator: all {len(runs)} runs completed")


if __name__ == "__main__":
    main()
