from __future__ import annotations

from collections.abc import Callable

from .homogeneous import homogeneous_policy


def select_baseline(name: str) -> Callable[..., object]:
    normalized = name.replace("-", "_").lower()
    if normalized in ("homogeneous", "homo", "homo_left", "homo_right"):
        return homogeneous_policy
    if normalized in ("pd", "pd_disaggregation"):
        from .pd_disaggregation import pd_policy

        return pd_policy
    if normalized in ("af", "af_disaggregation"):
        from .af_disaggregation import af_policy

        return af_policy
    raise KeyError(f"unknown baseline {name!r}")


__all__ = ["homogeneous_policy", "select_baseline"]
