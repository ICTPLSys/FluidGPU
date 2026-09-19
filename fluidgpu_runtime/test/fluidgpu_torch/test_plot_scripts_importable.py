from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def test_plot_scripts_import_without_running_figures():
    repo = Path(__file__).resolve().parents[3]
    plot_dir = repo / "scripts" / "plot"
    sys.path.insert(0, str(plot_dir))
    scripts = [
        "plot_fig2.py",
        "plot_fig3.py",
        "plot_fig6.py",
        "plot_fig7.py",
        "plot_fig9.py",
        "plot_fig10.py",
        "plot_fig11.py",
        "plot_fig12a.py",
        "plot_fig12b.py",
        "make_table3.py",
        "plot_all.py",
    ]

    for script in scripts:
        path = plot_dir / script
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
