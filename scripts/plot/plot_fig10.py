"""Reproduce Figure 10 (pipeline ablation, GPT-oss 20B).

Paper:
  (a) Throughput under three pipeline configurations (w/o Pipe, Pipe, Pipe+Prio)
      with a red dashed line marking the zero-communication optimal.
  (b) Time breakdown on the bottleneck GPU: computation vs communication %.

Data sources:
- summary.json per run: throughput_tok_s for the bar heights.
- summary.json may contain zero_comm_optimal_tok_s. The bars here are measured
  and this artifact has no measured zero-communication bound, so when no record
  carries one the line is simply omitted rather than synthesised from the
  paper's "96.6% of optimal" claim -- that claim is about the paper's own
  priority result, which does not reproduce here.
- monitor.csv per run (OnlineMonitor.flush): rows (name, mode, elapsed_us, bytes).
  Computation mode == "compute" or "cuda"; communication mode == "comm",
  "send", "recv", or "nccl".
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from common import (
    PAPER_OPTIMAL_LINE_COLOR,
    PAPER_BAR_EDGE,
    PALETTE,
    paper_value_labels,
    init_matplotlib,
    load_run_records,
    save_figure,
    style_axis,
    value_or_none,
    write_pending,
)


PIPELINE_ORDER = ["off", "naive", "priority"]
PIPELINE_LABELS = {"off": "w/o Pipe.", "naive": "Pipe.", "priority": "Pipe.+Prio."}
PIPELINE_COLORS = {
    "off": PALETTE["orion"],
    "naive": PALETTE["lithos"],
    "priority": PALETTE["hummingbird"],
}
COMPUTE_MODES = {"compute", "cuda", "kernel"}
COMM_MODES = {"comm", "communication", "send", "recv", "nccl"}
PAPER_PRIORITY_FRACTION = 0.966


def monitor_breakdown(run_dir: Path) -> tuple[float, float] | None:
    path = run_dir / "monitor.csv"
    if not path.exists():
        return None
    compute_us = 0.0
    comm_us = 0.0
    with path.open() as f:
        for row in csv.DictReader(f):
            mode = str(row.get("mode", "")).strip().lower()
            elapsed = value_or_none(row.get("elapsed_us"))
            if elapsed is None or elapsed < 0.0:
                continue
            if mode in COMPUTE_MODES:
                compute_us += elapsed
            elif mode in COMM_MODES:
                comm_us += elapsed
    if compute_us <= 0.0 and comm_us <= 0.0:
        return None
    return compute_us, comm_us


def main() -> None:
    records = load_run_records("fig10")
    if not records:
        write_pending("fig10")

    throughput: dict[str, float] = {}
    optimal_values: list[float] = []
    breakdowns: dict[str, tuple[float, float]] = {}
    for row in records:
        mode = row.payload.get("pipeline")
        if isinstance(mode, bool):
            # YAML parses a bare `off` as boolean False.
            mode = "off" if mode is False else None
        if mode is None:
            tail = row.run_id.split("_")[-1]
            mode = tail if tail in PIPELINE_ORDER else None
        if mode is None:
            continue
        mode = str(mode)
        tok_s = value_or_none(row.summary.get("throughput_tok_s"))
        if tok_s is None:
            tok_s = value_or_none(row.summary.get("throughput_req_s"))
        if tok_s is None:
            continue
        throughput[mode] = float(tok_s)
        opt = value_or_none(row.summary.get("zero_comm_optimal_tok_s"))
        if opt is not None and opt > 0.0:
            optimal_values.append(opt)
        br = monitor_breakdown(row.run_dir)
        if br is not None:
            breakdowns[mode] = br

    if not throughput:
        raise SystemExit(
            "fig10: missing throughput_{tok,req}_s summaries; run bash scripts/repro.sh --fig 10 first"
        )

    plt = init_matplotlib()
    import numpy as np

    # The paper gives the three-bar panel more width than the two-bar one, and
    # titles neither: "(a)"/"(b)" live in the caption.
    fig, (ax_thr, ax_break) = plt.subplots(
        1, 2, figsize=(7.6, 3.0), gridspec_kw={"width_ratios": [3, 2]}
    )

    modes = [m for m in PIPELINE_ORDER if m in throughput]
    x = np.arange(len(modes))
    values = [throughput[m] for m in modes]
    ax_thr.bar(
        x,
        values,
        width=0.62,
        color=[PIPELINE_COLORS[m] for m in modes],
        **PAPER_BAR_EDGE,
    )
    # The paper prints these bold, above each bar.
    paper_value_labels(ax_thr, x, values, fontsize=9, fontweight="bold")
    ax_thr.set_xticks(x)
    ax_thr.set_xticklabels([PIPELINE_LABELS[m] for m in modes])
    ax_thr.set_ylabel("Throughput (tokens/s)", fontweight="bold")
    style_axis(ax_thr, grid_axis="y")
    ax_thr.set_ylim(bottom=0.0)
    for label in ax_thr.get_yticklabels():
        label.set_fontweight("bold")

    if optimal_values:
        optimal = max(optimal_values)
        # Paper's colour and dash for the zero-communication bound, with its
        # label just above the line at the left (anchored in axes fraction so
        # it cannot fall outside the x range).
        ax_thr.axhline(optimal, color=PAPER_OPTIMAL_LINE_COLOR, linestyle="--", linewidth=1.3)
        ax_thr.annotate(
            f"Optimal: {optimal:.0f}",
            xy=(0.04, optimal),
            xycoords=ax_thr.get_yaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=PAPER_OPTIMAL_LINE_COLOR,
        )
        ax_thr.set_ylim(top=optimal * 1.14)
    else:
        # No measured zero-communication bound exists here; drawing the paper's
        # would put a reference line over measured bars.
        ax_thr.set_ylim(top=max(values) * 1.18)

    if breakdowns:
        # Stacked percentage bars on the bottleneck GPU.
        total = {m: sum(breakdowns[m]) for m in breakdowns if sum(breakdowns[m]) > 0}
        modes_b = [m for m in PIPELINE_ORDER if m in total]
        xb = np.arange(len(modes_b))
        compute_pct = [breakdowns[m][0] / total[m] * 100.0 for m in modes_b]
        comm_pct = [breakdowns[m][1] / total[m] * 100.0 for m in modes_b]
        ax_break.bar(xb, compute_pct, width=0.42, color=PALETTE["lithos"],
                     label="Computation", **PAPER_BAR_EDGE)
        ax_break.bar(xb, comm_pct, width=0.42, bottom=compute_pct, color=PALETTE["reef"],
                     label="Communication", **PAPER_BAR_EDGE)
        ax_break.set_xlim(-0.62, len(modes_b) - 0.38)
        ax_break.set_xticks(xb)
        ax_break.set_xticklabels([PIPELINE_LABELS[m] for m in modes_b])
        ax_break.set_ylabel("Time Breakdown (%)", fontweight="bold")
        # A 100% stack would otherwise sit flush against the frame; the paper
        # leaves a little headroom above its top tick.
        ax_break.set_ylim(0.0, 107.0)
        # Paper's quarter ticks and its boxed in-panel legend.
        ax_break.set_yticks([0, 25, 50, 75, 100])
        ax_break.legend(
            loc="center left",
            fontsize=8,
            frameon=True,
            framealpha=0.92,
            edgecolor="#b0b0b0",
            prop={"weight": "bold", "size": 8},
        )
        style_axis(ax_break, grid_axis="y")
        for label in ax_break.get_yticklabels():
            label.set_fontweight("bold")
    else:
        ax_break.text(
            0.5,
            0.5,
            "monitor.csv not found;\nwrite OnlineMonitor output\nto <run>/monitor.csv",
            ha="center",
            va="center",
            transform=ax_break.transAxes,
            fontsize=10,
            color="#555555",
        )
        ax_break.set_xticks([])
        ax_break.set_yticks([])
        ax_break.set_title("(b) Bottleneck GPU time breakdown (pending)")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(fig, "fig10")


if __name__ == "__main__":
    main()
