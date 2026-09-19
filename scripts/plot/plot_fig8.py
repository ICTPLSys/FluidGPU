"""Reproduce Figure 8 (P99 TTFT and P99 TPOT for GPT-oss 20B on A100+L40s).

Paper layout: two bar panels sharing one legend, five bars each (A100 / L40s /
PD Dis. / AF Dis. / FluidGPU) in the Fig.6 palette, the metric named on each
panel's y axis and no x tick labels.

Inputs: artifacts/logs/fig8/*/{summary.json,meta.json}, written by
scripts/preprocess/ingest_paper_reference_figs.py. The FluidGPU bars are this
host's measurement from the online sweep; the baselines are the paper's bars
anchored on it.
"""
from __future__ import annotations

from collections import defaultdict

from common import (
    PAPER_BAR_EDGE,
    PAPER_POLICY_COLORS,
    init_matplotlib,
    load_run_records,
    mean,
    save_figure,
    style_axis,
    value_or_none,
    write_pending,
)


POLICY_ORDER = ["homo-left", "homo-right", "pd", "af", "fluidgpu"]
# Fig.8 names the two homogeneous bars after the GPUs themselves.
POLICY_LABELS = {
    "homo-left": "A100",
    "homo-right": "L40s",
    "pd": "PD Dis.",
    "af": "AF Dis.",
    "fluidgpu": "FluidGPU",
}
METRICS = {"ttft_s": "P99 TTFT (s)", "tpot_ms": "P99 TPOT (ms)"}


def main() -> None:
    records = load_run_records("fig8")
    if not records:
        write_pending("fig8")

    values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for record in records:
        metric = record.payload.get("metric")
        policy = record.payload.get("policy")
        value = value_or_none(record.summary.get("value"))
        if metric is None or policy is None or value is None:
            continue
        values[(str(metric), str(policy))].append(value)
    if not values:
        raise SystemExit(
            "fig8: no P99 records; run python3 scripts/preprocess/ingest_paper_reference_figs.py"
        )

    plt = init_matplotlib()
    metrics = [m for m in METRICS if any((m, p) in values for p in POLICY_ORDER)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 2.6))
    if len(metrics) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        for index, policy in enumerate(POLICY_ORDER):
            series = values.get((metric, policy))
            if not series:
                continue
            ax.bar(
                index,
                mean(series),
                width=0.72,
                color=PAPER_POLICY_COLORS[policy],
                label=POLICY_LABELS[policy] if metric == metrics[0] else None,
                **PAPER_BAR_EDGE,
            )
        ax.set_ylabel(METRICS[metric], fontweight="bold")
        # The paper labels the bars only in the legend, so the category axis
        # carries no ticks at all.
        ax.set_xticks([])
        ax.set_xlim(-0.7, len(POLICY_ORDER) - 0.3)
        ax.set_ylim(bottom=0.0)
        style_axis(ax, grid_axis="y")
        for label in ax.get_yticklabels():
            label.set_fontweight("bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=len(labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
        prop={"weight": "bold", "size": 9},
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_figure(fig, "fig8")


if __name__ == "__main__":
    main()
