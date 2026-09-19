from __future__ import annotations

from fluidgpu_torch.monitor import QueueingAwareMonitor


def test_queueing_monitor_switches_from_latency_to_throughput():
    switches: list[str] = []
    monitor = QueueingAwareMonitor(
        window_ms=10,
        threshold_beta=1.5,
        on_switch=switches.append,
    )

    monitor.record_request(0.000, 0.000, 0.002, 1800.0)
    monitor.record_request(0.003, 0.003, 0.004, 1200.0)
    monitor.record_request(0.011, 0.011, 0.014, 1000.0)
    monitor.record_request(0.012, 0.020, 0.025, 1000.0)
    summary = monitor.summary()

    assert switches == ["throughput"]
    assert summary["current_policy"] == "throughput"
    assert summary["switch_count"] == 1
    assert monitor.windows[-1].ratio >= 1.5


def test_queueing_monitor_threshold_reduces_switches_monotonically():
    low = replay_with_threshold(1.5)
    high = replay_with_threshold(20.0)

    assert low.switch_count >= high.switch_count


def replay_with_threshold(threshold: float) -> QueueingAwareMonitor:
    monitor = QueueingAwareMonitor(window_ms=10, threshold_beta=threshold)
    for arrival, start, finish in (
        (0.000, 0.000, 0.002),
        (0.011, 0.011, 0.013),
        (0.012, 0.018, 0.025),
    ):
        monitor.record_request(arrival, start, finish, 1000.0)
    monitor.flush()
    return monitor
