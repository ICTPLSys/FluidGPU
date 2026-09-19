"""Reproduce Figure 3 (kernel duration ratio for GPT-oss 20B, A100 vs L40s).

Paper: two scatter panels over the same kernels, (a) grouped by prefill /
decode phase and (b) grouped by attention / FFN block. Kernels above the
dashed line at ratio = 1 run faster on the A100, those below run faster on the
L40s. The point of the figure is that both groupings straddle the line: neither
phase-level nor block-level disaggregation captures the kernel-level split.

Input: artifacts/logs/fig3/gptoss20b_{phase,block}_points/ records written by
`scripts/preprocess/ingest_kernel_census.py`. Each summary points at its
point table under artifacts/validation/kernel_census/, which carries
`x_log10_a100` and `y_l40s_over_a100` per plotted point.
"""
from __future__ import annotations

from pathlib import Path

from common import (
    REPO_ROOT,
    PALETTE,
    init_matplotlib,
    load_run_records,
    save_figure,
    style_axis,
    value_or_none,
    write_pending,
)


# Paper's series colors and markers for the two groupings.
SERIES = {
    "decode": {"label": "Decode", "color": PALETTE["reef"], "marker": "o"},
    "prefill": {"label": "Prefill", "color": PALETTE["lithos"], "marker": "^"},
    "attention": {"label": "Attention", "color": PALETTE["orion"], "marker": "o"},
    "ffn": {"label": "FFN", "color": PALETTE["hummingbird"], "marker": "^"},
}
PANELS = [
    ("gptoss20b_phase_points", "phase", ["decode", "prefill"]),
    ("gptoss20b_block_points", "category", ["attention", "ffn"]),
]
Y_TICKS = [0.25, 0.5, 1, 2, 4, 8, 16]


def read_points(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if header is None:
            header = parts
            continue
        rows.append(dict(zip(header, parts)))
    return rows


def main() -> None:
    records = {r.run_id: r for r in load_run_records("fig3")}
    if not records:
        write_pending("fig3")
    if not any(run_id in records for run_id, _, _ in PANELS):
        raise SystemExit(
            "fig3: no kernel-census records; run "
            "python3 scripts/preprocess/ingest_kernel_census.py first"
        )

    plt = init_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 2.85), dpi=120, sharey=True)

    for ax, (run_id, group_key, groups) in zip(axes, PANELS):
        record = records.get(run_id)
        if record is None:
            continue
        rows = read_points(REPO_ROOT / str(record.summary.get("points_file")))

        for group in groups:
            xs, ys = [], []
            for row in rows:
                if row.get(group_key) != group:
                    continue
                x_log = value_or_none(row.get("x_log10_a100"))
                y = value_or_none(row.get("y_l40s_over_a100"))
                if x_log is None or y is None:
                    continue
                xs.append(10.0**x_log)
                ys.append(y)
            spec = SERIES[group]
            ax.scatter(
                xs,
                ys,
                s=34,
                marker=spec["marker"],
                color=spec["color"],
                label=spec["label"],
                edgecolors="white",
                linewidths=0.4,
                zorder=3,
            )

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(0.22, 20.0)
        ax.set_yticks(Y_TICKS)
        ax.set_yticklabels([f"{t:g}" for t in Y_TICKS])
        ax.set_xlim(0.8, 2.0e4)
        ax.text(
            0.97, 0.90, "A100 faster", transform=ax.transAxes, ha="right",
            fontsize=9, fontweight="bold", color="#555555",
        )
        ax.text(
            0.97, 0.06, "L40s faster", transform=ax.transAxes, ha="right",
            fontsize=9, fontweight="bold", color="#555555",
        )
        style_axis(ax, grid_axis="y")
        ax.legend(
            loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=2,
            frameon=False, fontsize=9, handletextpad=0.3, columnspacing=1.4,
        )

        # The paper calls out one memory-bandwidth-bound kernel the A100 wins
        # and one attention kernel the L40s wins; anchors resolved at ingest.
        for ann in record.summary.get("annotations") or []:
            x = value_or_none(ann.get("x_us"))
            y = value_or_none(ann.get("y"))
            if x is None or y is None:
                continue
            above = y > 1.0
            ax.annotate(
                str(ann.get("label", "")),
                xy=(x, y),
                xytext=(-14, 26) if above else (30, -22),
                textcoords="offset points",
                fontsize=8,
                ha="right" if above else "left",
                va="bottom" if above else "top",
                bbox={"boxstyle": "round,pad=0.25", "fc": "white",
                      "ec": "#999999", "lw": 0.6},
                arrowprops={"arrowstyle": "-", "color": "#666666", "lw": 0.7},
                zorder=5,
            )

    axes[0].set_ylabel("Duration Ratio\n(L40s/A100)", fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.20)
    fig.supxlabel("Kernel Duration on A100 (µs)", fontsize=10, y=0.03)
    save_figure(fig, "fig3")


if __name__ == "__main__":
    main()
