"""This script reproduces Figure 9 (cluster-scale offline throughput).

Paper layout: one panel per cluster, three bars each (PD / AF / FluidGPU), the
cluster named below the x tick labels and its own y scale per panel. The paper
prints each bar's value above it; this artifact does not, because only the
FluidGPU bar of the three-card panel is a measurement and
printed numbers would read as measured throughput for all of them.

Input: artifacts/logs/fig9/*/{summary.json,meta.json}.
Output: artifacts/figures/fig9.pdf and .png.
"""
from __future__ import annotations

from collections import defaultdict

from common import (
    PAPER_BAR_EDGE,
    PALETTE,
    init_matplotlib,
    load_run_records,
    mean,
    paper_panel_labels,
    save_figure,
    style_axis,
    value_or_none,
    write_pending,
)

POLICY_ORDER = ["pd", "af", "fluidgpu"]
POLICY_COLORS = {
    "pd": PALETTE["lithos"],
    "af": PALETTE["orion"],
    "fluidgpu": PALETTE["hummingbird"],
}
POLICY_LABELS = {"pd": "PD Dis.", "af": "AF Dis.", "fluidgpu": "FluidGPU"}
# Paper order: the three-card cluster first.
CLUSTER_ORDER = ["two_a100_one_l40s_gptoss", "eight_b200_eight_h100_qwen3"]


def main() -> None:
    records = load_run_records("fig9")
    if not records:
        write_pending("fig9")

    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    clusters: set[str] = set()
    for row in records:
        cluster = row.payload.get("cluster_label")
        policy = row.payload.get("policy")
        throughput = value_or_none(row.summary.get("throughput_tok_s"))
        if throughput is None:
            throughput = value_or_none(row.summary.get("throughput_req_s"))
        if cluster is None or policy is None or throughput is None:
            continue
        cluster_key = str(cluster)
        policy_key = str(policy)
        grouped[(cluster_key, policy_key)].append(throughput)
        clusters.add(cluster_key)
    if not grouped:
        raise SystemExit(
            "fig9: missing throughput summaries; "
            "run python3 scripts/preprocess/ingest_vllm_exp.py first"
        )

    cluster_order = [c for c in CLUSTER_ORDER if c in clusters] + sorted(clusters - set(CLUSTER_ORDER))
    cluster_labels = {
        "two_a100_one_l40s_gptoss": "(a) GPT-oss 20B\n2\u00d7A100 + 1\u00d7L40s",
        "eight_b200_eight_h100_qwen3": "(b) Qwen3 235B\n8\u00d7B200 + 8\u00d7H100",
    }
    plt = init_matplotlib()
    import numpy as np

    # Each cluster keeps its own y scale, as in the paper: an 8xB200 + 8xH100
    # cluster and a three-card node are three thousand tokens/s apart.
    fig, axes = plt.subplots(1, len(cluster_order), figsize=(3.4 * len(cluster_order), 2.9))
    if len(cluster_order) == 1:
        axes = [axes]
    x = np.arange(len(POLICY_ORDER))

    for ax, cluster in zip(axes, cluster_order):
        present = [i for i, p in enumerate(POLICY_ORDER) if (cluster, p) in grouped]
        missing = [i for i, p in enumerate(POLICY_ORDER) if (cluster, p) not in grouped]
        ax.bar(
            [x[i] for i in present],
            [mean(grouped[(cluster, POLICY_ORDER[i])]) for i in present],
            color=[POLICY_COLORS[POLICY_ORDER[i]] for i in present],
            **PAPER_BAR_EDGE,
        )
        if missing:
            # Same convention as fig6: a policy with no measurable arm on this
            # cluster is marked with a red X at y=0 (paper Fig. 6 caption).
            ax.scatter([x[i] for i in missing], [0.0] * len(missing), marker="x", color="#d62728", s=36, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([POLICY_LABELS[p] for p in POLICY_ORDER])
        style_axis(ax, grid_axis="y")
        ax.set_ylim(bottom=0.0)
        for label in ax.get_xticklabels():
            label.set_fontweight("bold")
    axes[0].set_ylabel("Throughput (tokens/s)", fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    # The paper names each cluster below its x tick labels, not as a title.
    paper_panel_labels(
        fig,
        [(ax, cluster_labels.get(c, c)) for ax, c in zip(axes, cluster_order)],
        fontsize=11,
    )
    save_figure(fig, "fig9")


if __name__ == "__main__":
    main()
