from __future__ import annotations

PD_BASELINE_SCRIPT = "scripts/run_pd_baseline_vllm.sh"


def pd_policy(*args: object, **kwargs: object) -> dict[str, str]:
    return {
        "baseline": "pd",
        "script_path": PD_BASELINE_SCRIPT,
        "entrypoint": f"bash {PD_BASELINE_SCRIPT}",
    }
