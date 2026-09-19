"""This script reproduces Figure 12b.
Input: artifacts/logs/fig12b/*/summary.json with solver timing rows.
Output: artifacts/figures/fig12b.pdf and .png.
Trend: solver time increases with kernel count and GPU count.
"""
from __future__ import annotations

from collections import defaultdict

from common import (
    PALETTE,
    PAPER_MARKER_EDGE,
    init_matplotlib,
    load_run_records,
    save_figure,
    style_axis,
    value_or_none,
    write_pending,
)


def main() -> None:
    import matplotlib
    records = load_run_records("fig12b")
    if not records:
        write_pending("fig12b")
    plt = init_matplotlib()

    by_gpu: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for row in records:
        kernels = row.summary.get("kernels", row.payload.get("kernels"))
        gpus = row.summary.get("gpus", row.payload.get("gpus"))
        solver = row.summary.get("solver")
        solver_ms = value_or_none(row.summary.get("solver_ms"))
        if solver is not None and solver != "gurobi":
            raise SystemExit(f"fig12b: expected Gurobi summaries, got solver={solver!r} in {row.run_id}")
        if kernels is None or gpus is None or solver_ms is None:
            continue
        by_gpu[int(gpus)].append((int(kernels), solver_ms))
    if not by_gpu:
        raise SystemExit("fig12b: no usable solver summaries found in artifacts/logs/fig12b")

    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    # Paper gives each GPU count its own marker shape as well as its own
    # colour, measured from the figure's legend glyphs.
    gpu_style = {
        2: {"color": PALETTE["hummingbird"], "marker": "o"},
        3: {"color": PALETTE["orion"], "marker": "s"},
        4: {"color": PALETTE["reef"], "marker": "^"},
    }
    for gpus in sorted(by_gpu):
        points = sorted(by_gpu[gpus], key=lambda x: x[0])
        style = gpu_style.get(gpus, {"color": PALETTE["hummingbird"], "marker": "o"})
        ax.plot(
            [x for x, _ in points],
            [y / 1000.0 for _, y in points],  # paper reports seconds
            linewidth=2.0,
            markersize=8,
            label=f"{gpus} GPUs",
            **style,
            **PAPER_MARKER_EDGE,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Kernels per DDG", fontweight="bold")
    ax.set_ylabel("Solver Time (s)", fontweight="bold")
    style_axis(ax, grid_axis="both")
    ax.set_xticks([0, 200, 400, 600, 800, 1000])
    ax.set_xlim(left=0)
    # Paper prints plain decade labels (0.01 ... 100), not 10^k, and boxes the
    # legend inside the panel.
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda v, _pos: ("%g" % v) if v >= 1 else ("%g" % v).rstrip("0").rstrip(".") or "0"
        )
    )
    ax.legend(
        loc="upper left",
        frameon=True,
        framealpha=0.92,
        edgecolor="#b0b0b0",
        prop={"weight": "bold", "size": 10},
    )
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_figure(fig, "fig12b")


if __name__ == "__main__":
    main()
