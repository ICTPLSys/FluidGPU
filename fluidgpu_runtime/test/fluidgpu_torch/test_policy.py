import csv
import importlib.util
import json
import sys
from pathlib import Path

from fluidgpu_torch.policy import (
    PolicyCandidate,
    PolicyConfig,
    choose_policy,
    read_policy_candidates,
)


def test_policy_selects_latency_winner():
    decision = choose_policy(demo_candidates(), PolicyConfig(objective="latency"))

    assert decision.winner.candidate.name == "low_latency_profile"
    assert len(decision.rejected) == 1
    assert decision.rejected[0].reason == "not_runtime_compatible"


def test_policy_selects_throughput_winner():
    decision = choose_policy(demo_candidates(), PolicyConfig(objective="throughput"))

    assert decision.winner.candidate.name == "high_throughput_profile"


def test_policy_selects_balanced_winner():
    decision = choose_policy(demo_candidates(), PolicyConfig(objective="balanced"))

    assert decision.winner.candidate.name == "balanced_profile"
    assert decision.winner.reason == "weighted_latency_throughput"


def test_policy_filters_constraints():
    decision = choose_policy(
        demo_candidates(),
        PolicyConfig(
            objective="throughput",
            max_decode_p95_us=200.0,
            min_tokens_per_s=800.0,
        ),
    )

    assert decision.winner.candidate.name == "balanced_profile"
    assert {item.candidate.name: item.reason for item in decision.rejected} == {
        "low_latency_profile": "tokens_per_s_under_limit",
        "high_throughput_profile": "decode_p95_over_limit",
        "offline_fine_profile": "not_runtime_compatible",
    }


def test_read_policy_candidates_csv(tmp_path):
    path = tmp_path / "candidates.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "profile_path",
                "prefill_us",
                "decode_us",
                "tokens_per_s",
                "decode_p95_us",
                "error_rate",
                "runtime_compatible",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "name": "candidate_a",
                "profile_path": "/tmp/a.json",
                "prefill_us": "10",
                "decode_us": "1",
                "tokens_per_s": "1000",
                "decode_p95_us": "2",
                "error_rate": "0.01",
                "runtime_compatible": "true",
            }
        )

    candidates = read_policy_candidates(path)

    assert candidates == [
        PolicyCandidate(
            name="candidate_a",
            profile_path="/tmp/a.json",
            prefill_us=10.0,
            decode_us=1.0,
            tokens_per_s=1000.0,
            decode_p95_us=2.0,
            error_rate=0.01,
            runtime_compatible=True,
        )
    ]


def test_policy_probe_demo_cli(tmp_path):
    module = load_cli()
    output = tmp_path / "decision.json"

    run_main(module, "--demo", "--output-json", str(output))

    raw = json.loads(output.read_text())
    assert raw["objective"] == "balanced"
    assert raw["winner"]["candidate"]["name"] == "balanced_profile"
    assert raw["policy_probe"]["runtime_compatible"] is False


def demo_candidates():
    return [
        PolicyCandidate("low_latency_profile", 10000.0, 100.0, 700.0, 150.0, 0.0, True),
        PolicyCandidate("high_throughput_profile", 16000.0, 160.0, 1400.0, 240.0, 0.0, True),
        PolicyCandidate("balanced_profile", 13500.0, 110.0, 1200.0, 180.0, 0.0, True),
        PolicyCandidate("offline_fine_profile", 9000.0, 90.0, 1500.0, 130.0, 0.0, False),
    ]


def load_cli():
    path = Path(__file__).parents[2] / "examples" / "fluidgpu_torch" / "policy_probe.py"
    spec = importlib.util.spec_from_file_location("fluidgpu_policy_probe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def run_main(module, *args: str) -> None:
    old_argv = sys.argv
    sys.argv = ["policy_probe.py", *args]
    try:
        module.main()
    finally:
        sys.argv = old_argv
