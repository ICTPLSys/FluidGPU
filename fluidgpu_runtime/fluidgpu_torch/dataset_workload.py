"""Load benchmark request workloads (Splitwise-style conversation JSONL).

Each line: {"request_id", "arrival_s", "prompt", "input_tokens",
"output_tokens", ...}. The synthetic prompts are not tokenizer-aware, so the
loader re-tokenizes and truncates each prompt to its declared input_tokens
(clamped so input + output fits max_seq_len), keeping the executed workload
faithful to the declared Splitwise lengths on any tokenizer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WorkloadRequest:
    request_id: str
    prompt: str
    prompt_tokens: int
    max_new_tokens: int
    arrival_s: float


def load_jsonl_workload(
    path: str | Path,
    tokenizer: Any,
    *,
    max_seq_len: int,
    limit: int | None = None,
) -> list[WorkloadRequest]:
    requests: list[WorkloadRequest] = []
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            output_tokens = int(row["output_tokens"])
            assert output_tokens > 0, f"non-positive output_tokens in {row.get('request_id')}"
            budget = max_seq_len - output_tokens
            assert budget > 0, (
                f"request {row.get('request_id')} output {output_tokens} exceeds "
                f"max_seq_len {max_seq_len}"
            )
            target = min(int(row["input_tokens"]), budget)
            ids = tokenizer(row["prompt"], add_special_tokens=False).input_ids[:target]
            prompt = tokenizer.decode(ids, skip_special_tokens=True)
            requests.append(
                WorkloadRequest(
                    request_id=str(row.get("request_id", len(requests))),
                    prompt=prompt,
                    prompt_tokens=len(ids),
                    max_new_tokens=output_tokens,
                    arrival_s=float(row.get("arrival_s", 0.0)),
                )
            )
            if limit is not None and len(requests) >= limit:
                break
    assert requests, f"no requests loaded from {path}"
    return requests
