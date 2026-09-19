from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
import json
from pathlib import Path
from threading import Lock
import time
from typing import Any, Iterator

import torch


@dataclass
class WorkerTraceRecorder:
    rank: int
    events: list[dict[str, Any]] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)

    @contextmanager
    def region(
        self,
        *,
        kind: str,
        name: str,
        device: torch.device | None,
        mode: str | None = None,
        peer: int | None = None,
        bytes: int | None = None,
    ) -> Iterator[None]:
        start_ns = time.perf_counter_ns()
        duration_ms = 0.0
        stream_label = f"rank{self.rank}:stream-host"
        stream = None
        start_event = None
        end_event = None
        if device is not None and device.type == "cuda" and torch.cuda.is_available():
            stream = torch.cuda.current_stream(device)
            stream_label = _stream_label(self.rank, stream)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record(stream)
        try:
            yield
        finally:
            if end_event is not None and stream is not None:
                end_event.record(stream)
                end_event.synchronize()
                try:
                    duration_ms = float(start_event.elapsed_time(end_event))
                except RuntimeError:
                    duration_ms = 0.0
            end_ns = time.perf_counter_ns()
            if duration_ms <= 0.0:
                duration_ms = (end_ns - start_ns) / 1_000_000.0
            event = {
                "rank": self.rank,
                "kind": kind,
                "name": name,
                "stream": stream_label,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ms": duration_ms,
            }
            if mode is not None:
                event["mode"] = mode
            if peer is not None:
                event["peer"] = peer
            if bytes is not None:
                event["bytes"] = bytes
            with self._lock:
                self.events.append(event)

    def write_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for event in sorted(self.events, key=lambda item: int(item["start_ns"])):
                f.write(json.dumps(event) + "\n")


_RECORDER_LOCK = Lock()
_RECORDER: WorkerTraceRecorder | None = None


def install_worker_trace_recorder(*, rank: int) -> WorkerTraceRecorder:
    recorder = WorkerTraceRecorder(rank=rank)
    global _RECORDER
    with _RECORDER_LOCK:
        _RECORDER = recorder
    return recorder


def get_worker_trace_recorder() -> WorkerTraceRecorder | None:
    with _RECORDER_LOCK:
        return _RECORDER


def clear_worker_trace_recorder() -> None:
    global _RECORDER
    with _RECORDER_LOCK:
        _RECORDER = None


def trace_region(
    *,
    kind: str,
    name: str,
    device: torch.device | None,
    mode: str | None = None,
    peer: int | None = None,
    bytes: int | None = None,
):
    recorder = get_worker_trace_recorder()
    if recorder is None:
        return nullcontext()
    return recorder.region(
        kind=kind,
        name=name,
        device=device,
        mode=mode,
        peer=peer,
        bytes=bytes,
    )


def _stream_label(rank: int, stream: torch.cuda.Stream) -> str:
    raw_stream = getattr(stream, "cuda_stream", None)
    if raw_stream is None:
        return f"rank{rank}:stream"
    return f"rank{rank}:stream{int(raw_stream)}"
