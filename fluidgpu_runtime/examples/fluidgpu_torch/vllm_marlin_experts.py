"""Compatibility shim — the canonical implementation lives in the package.

The marlin-native expert rebuild moved to fluidgpu_torch.vllm_integration so
that vLLM worker processes can import it by qualified name (worker_cls). This
file keeps the old example-local import path working.
"""
from __future__ import annotations

import sys
from pathlib import Path

_RUNTIME_ROOT = Path(__file__).resolve().parents[2]
if str(_RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(_RUNTIME_ROOT))

from fluidgpu_torch.vllm_integration import (  # noqa: E402,F401
    CheckpointReader,
    MarlinExpertsOnDevice,
)

_CheckpointReader = CheckpointReader
