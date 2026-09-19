from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


PLOT_SCRIPTS = [
    "plot_fig2.py",
    "plot_fig3.py",
    "plot_fig6.py",
    "plot_fig7.py",
    "plot_fig8.py",
    "plot_fig9.py",
    "plot_fig10.py",
    "plot_fig11.py",
    "plot_fig12a.py",
    "plot_fig12b.py",
    "make_table3.py",
]


def import_script(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot all FluidGPU AD figures with available logs")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    sys.path.insert(0, str(root))
    for script in PLOT_SCRIPTS:
        path = root / script
        import_script(path)
        if args.check:
            print(f"check: import ok {script}")
    if args.check:
        return
    for script in PLOT_SCRIPTS:
        if script == "plot_all.py":
            continue
        namespace = {"__name__": "__main__", "__file__": str(root / script)}
        exec((root / script).read_text(), namespace)


if __name__ == "__main__":
    main()
