from __future__ import annotations

AF_BASELINE_SCRIPT = "scripts/run_af_baseline_fluidgpu.sh"


def af_policy(*args: object, **kwargs: object) -> dict[str, str]:
    return {
        "baseline": "af",
        "script_path": AF_BASELINE_SCRIPT,
        "entrypoint": f"bash {AF_BASELINE_SCRIPT}",
    }
