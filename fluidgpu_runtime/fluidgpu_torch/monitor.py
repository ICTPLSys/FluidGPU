from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class MonitorEvent:
    name: str
    mode: str
    elapsed_us: float
    bytes: int = 0


@dataclass(frozen=True)
class QueueingWindow:
    window_start_s: float
    window_end_s: float
    request_count: int
    l_req_us: float
    l_exec_us: float
    ratio: float
    policy: str


@dataclass
class OnlineMonitor:
    output: str | Path | None = None
    events: list[MonitorEvent] = field(default_factory=list)

    def record(self, name: str, mode: str, elapsed_us: float, bytes: int = 0) -> None:
        self.events.append(MonitorEvent(name=name, mode=mode, elapsed_us=elapsed_us, bytes=bytes))

    def time_block(self, name: str, mode: str, bytes: int = 0):
        return _Timer(self, name, mode, bytes)

    def flush(self) -> None:
        if self.output is None:
            return
        path = Path(self.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "mode", "elapsed_us", "bytes"])
            writer.writeheader()
            for event in self.events:
                writer.writerow(
                    {
                        "name": event.name,
                        "mode": event.mode,
                        "elapsed_us": f"{event.elapsed_us:.3f}",
                        "bytes": event.bytes,
                    }
                )


class QueueingAwareMonitor:
    """Windowed queueing monitor for latency/throughput policy adaptation."""

    def __init__(
        self,
        *,
        window_ms: float = 300.0,
        threshold_beta: float = 1.5,
        on_switch: Callable[[str], None] | None = None,
        initial_policy: str = "latency",
    ) -> None:
        assert window_ms > 0, f"window_ms must be positive, got {window_ms}"
        assert threshold_beta > 0, (
            f"threshold_beta must be positive, got {threshold_beta}"
        )
        assert initial_policy in ("latency", "throughput"), (
            f"initial_policy must be latency or throughput, got {initial_policy}"
        )
        self.window_ms = float(window_ms)
        self.threshold_beta = float(threshold_beta)
        self.on_switch = on_switch
        self.current_policy = initial_policy
        self.windows: list[QueueingWindow] = []
        self.switch_count = 0
        self._events: list[tuple[float, float, float, float]] = []
        self._window_start_s: float | None = None

    def record_request(
        self,
        arrival_ts: float,
        start_ts: float,
        finish_ts: float,
        exec_us: float,
    ) -> None:
        assert finish_ts >= arrival_ts, "finish_ts must be >= arrival_ts"
        assert finish_ts >= start_ts, "finish_ts must be >= start_ts"
        assert exec_us >= 0.0, f"exec_us must be non-negative, got {exec_us}"
        if self._window_start_s is None:
            self._window_start_s = float(arrival_ts)

        window_s = self.window_ms / 1000.0
        assert self._window_start_s is not None
        while self._events and finish_ts >= self._window_start_s + window_s:
            self._close_window(self._window_start_s + window_s)
            assert self._window_start_s is not None

        self._events.append((float(arrival_ts), float(start_ts), float(finish_ts), float(exec_us)))

    def flush(self) -> None:
        if self._events and self._window_start_s is not None:
            self._close_window(self._window_start_s + self.window_ms / 1000.0)

    def summary(self) -> dict[str, float | int | str]:
        self.flush()
        if not self.windows:
            return {
                "window_count": 0,
                "switch_count": self.switch_count,
                "avg_ratio": 0.0,
                "current_policy": self.current_policy,
            }
        avg_ratio = sum(window.ratio for window in self.windows) / len(self.windows)
        return {
            "window_count": len(self.windows),
            "switch_count": self.switch_count,
            "avg_ratio": avg_ratio,
            "current_policy": self.current_policy,
        }

    def _close_window(self, window_end_s: float) -> None:
        assert self._window_start_s is not None
        events = self._events
        self._events = []
        if not events:
            self._window_start_s = window_end_s
            return

        l_req_us = sum((finish - arrival) * 1e6 for arrival, _, finish, _ in events) / len(events)
        l_exec_us = sum(exec_us for _, _, _, exec_us in events) / len(events)
        ratio = l_req_us / max(l_exec_us, 1e-9)
        next_policy = "throughput" if ratio >= self.threshold_beta else "latency"
        if next_policy != self.current_policy:
            self.current_policy = next_policy
            self.switch_count += 1
            if self.on_switch is not None:
                self.on_switch(next_policy)

        self.windows.append(
            QueueingWindow(
                window_start_s=self._window_start_s,
                window_end_s=window_end_s,
                request_count=len(events),
                l_req_us=l_req_us,
                l_exec_us=l_exec_us,
                ratio=ratio,
                policy=next_policy,
            )
        )
        self._window_start_s = window_end_s


class _Timer:
    def __init__(self, monitor: OnlineMonitor, name: str, mode: str, bytes: int) -> None:
        self.monitor = monitor
        self.name = name
        self.mode = mode
        self.bytes = bytes
        self.start = 0

    def __enter__(self):
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        elapsed_us = (time.perf_counter_ns() - self.start) / 1000.0
        self.monitor.record(self.name, self.mode, elapsed_us, self.bytes)
