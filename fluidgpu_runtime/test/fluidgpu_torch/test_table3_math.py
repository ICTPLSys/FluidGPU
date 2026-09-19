from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def test_perf_per_dollar_math():
    repo = Path(__file__).resolve().parents[3]
    plot_dir = repo / "scripts" / "plot"
    if str(plot_dir) not in sys.path:
        sys.path.insert(0, str(plot_dir))
    path = plot_dir / "make_table3.py"
    spec = importlib.util.spec_from_file_location("make_table3", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.perf_per_dollar(96.0, 4.8) == 20.0
