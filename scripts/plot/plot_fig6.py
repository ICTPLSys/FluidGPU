"""Reproduce Figure 6 (offline throughput across 5 workloads x 3 GPU pairs).

Paper layout: one subplot per GPU pair; within each subplot, a bar group per
model, 6 bars per group (H-L / H-R / Request Dist. / PD / AF / FG; the
camera-ready added the request-distribution baseline). Left y-axis reports
tokens/s for LLM/SSM/MLLM workloads; right y-axis reports images/min for the
diffusion model. Inapplicable (policy, model) combinations are marked with red
"X" markers at y=0 (paper Fig. 6 caption).

Inputs: artifacts/logs/fig6/*/{summary.json,meta.json}. summary.json must
contain throughput_tok_s and throughput_req_s (llm_generate.emit_results
writes both).
"""
from __future__ import annotations

from collections import defaultdict

from common import (
    PAPER_BAR_EDGE,
    PAPER_MODEL_TICKS,
    PAPER_POLICY_COLORS,
    PAPER_POLICY_LABELS,
    PALETTE,
    init_matplotlib,
    load_run_records,
    mean,
    paper_panel_labels,
    save_figure,
    strip_repeat_suffix,
    style_axis,
    value_or_none,
    write_pending,
)


POLICY_ORDER = ["homo-left", "homo-right", "request-dist", "pd", "af", "fluidgpu"]
POLICY_LABELS = PAPER_POLICY_LABELS
POLICY_COLORS = PAPER_POLICY_COLORS
# Paper's Fig.6 pair names (its Fig.7 spells the same pair "A100 + L40s").
PAIR_LABELS = {
    "a100_l40s": "A100 + L40S",
    "h100_rtxpro6000": "H100 + RTX Pro 6000",
    "b200_h100": "B200 + H100",
}
PAIR_ORDER = list(PAIR_LABELS)
# Model ordering: LLMs/SSM/MLLM first (tokens/s), diffusion last (images/min).
MODEL_ORDER = [
    "llama31_8b",
    "gptoss20b",
    "qwen25vl7b",
    "mamba_codestral7b",
    "sd35_medium",
]
DIFFUSION_MODELS = {"sd35_medium"}


def infer_policy(record) -> str | None:
    policy = record.payload.get("policy")
    if policy:
        return str(policy)
    base = strip_repeat_suffix(record.run_id)
    if base.endswith("_pd"):
        return "pd"
    if base.endswith("_af"):
        return "af"
    return None


def main() -> None:
    records = load_run_records("fig6")
    if not records:
        write_pending("fig6")

    # (model, pair, policy) -> list of throughput values in the correct unit.
    grouped: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in records:
        policy = infer_policy(row)
        model = row.payload.get("model_label")
        pair = row.payload.get("pair_label")
        if not (policy and model and pair):
            continue
        if pair not in PAIR_LABELS:
            continue
        summary = row.summary
        if str(model) in DIFFUSION_MODELS:
            # Convert requests/s to images/min (1 request = 1 image).
            value = value_or_none(summary.get("throughput_req_s"))
            value = value * 60.0 if value is not None else None
        else:
            # Paper accounting: total (prompt + generated) tokens per second.
            # Fall back to output-only for pre-Splitwise summaries.
            value = value_or_none(summary.get("throughput_total_tok_s"))
            if value is None:
                value = value_or_none(summary.get("throughput_tok_s"))
        if value is None:
            continue
        grouped[(str(model), str(pair), policy)].append(float(value))
    if not grouped:
        raise SystemExit(
            "fig6: missing throughput_{tok,req}_s in summaries; "
            "run python3 scripts/preprocess/ingest_vllm_exp.py first"
        )

    models_present = [m for m in MODEL_ORDER if any((m, p, pol) in grouped for p in PAIR_ORDER for pol in POLICY_ORDER)]
    if not models_present:
        raise SystemExit("fig6: no model data found for the expected model labels")
    pairs_present = [p for p in PAIR_ORDER if any((m, p, pol) in grouped for m in models_present for pol in POLICY_ORDER)]

    has_tokens = any(m not in DIFFUSION_MODELS for m in models_present)
    has_images = any(m in DIFFUSION_MODELS for m in models_present)

    plt = init_matplotlib()
    import numpy as np

    fig, axes = plt.subplots(1, len(pairs_present), figsize=(5.4 * len(pairs_present), 3.9), sharey=False)
    if len(pairs_present) == 1:
        axes = [axes]

    # Six bars per group with a one-bar gap between groups, as the paper draws it.
    width = 1.0 / (len(POLICY_ORDER) + 1)
    x = np.arange(len(models_present))

    twin_axes: list = []
    for ax, pair in zip(axes, pairs_present):
        ax_right = ax.twinx() if has_images and has_tokens else None
        twin_axes.append(ax_right)
        for idx, policy in enumerate(POLICY_ORDER):
            bar_xs_tok: list[float] = []
            bar_ys_tok: list[float] = []
            bar_xs_img: list[float] = []
            bar_ys_img: list[float] = []
            missing_x: list[float] = []
            for mi, model in enumerate(models_present):
                xpos = mi + (idx - (len(POLICY_ORDER) - 1) / 2.0) * width
                values = grouped.get((model, pair, policy), [])
                if values:
                    if model in DIFFUSION_MODELS:
                        bar_xs_img.append(xpos)
                        bar_ys_img.append(mean(values))
                    else:
                        bar_xs_tok.append(xpos)
                        bar_ys_tok.append(mean(values))
                else:
                    missing_x.append(xpos)
            if bar_xs_tok:
                ax.bar(
                    bar_xs_tok,
                    bar_ys_tok,
                    width=width,
                    color=POLICY_COLORS[policy],
                    label=POLICY_LABELS[policy] if pair == pairs_present[0] else None,
                    **PAPER_BAR_EDGE,
                )
            if bar_xs_img:
                target = ax_right if ax_right is not None else ax
                # Diffusion bars sit on the right axis; the paper plots them
                # with the same fills as the token-throughput bars.
                target.bar(
                    bar_xs_img,
                    bar_ys_img,
                    width=width,
                    color=POLICY_COLORS[policy],
                    label=None,
                    **PAPER_BAR_EDGE,
                )
            if missing_x:
                ax.scatter(missing_x, [0.0] * len(missing_x), marker="x", color="#d62728", s=28, zorder=5)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [
                PAPER_MODEL_TICKS.get(m, m)
                for m in models_present
            ]
        )
        style_axis(ax, grid_axis="y")
        # Paper weights: bold model ticks, plain numeric y ticks.
        for label in ax.get_xticklabels():
            label.set_fontweight("bold")
        ax.set_ylim(bottom=0.0)
        if ax_right is not None:
            ax_right.set_ylim(bottom=0.0)

    if has_tokens:
        axes[0].set_ylabel("Throughput (tokens/s)", fontweight="bold")
    if has_images and has_tokens:
        for ax_right in twin_axes:
            if ax_right is not None:
                ax_right.set_ylabel("Throughput (images/min)", fontweight="bold")
    elif has_images:
        axes[0].set_ylabel("Throughput (images/min)", fontweight="bold")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=len(POLICY_ORDER),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        prop={"weight": "bold"},
    )
    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    # The paper names the pair below the model ticks, not as a title.
    paper_panel_labels(fig, [(ax, PAIR_LABELS.get(pair, pair)) for ax, pair in zip(axes, pairs_present)])
    save_figure(fig, "fig6")


if __name__ == "__main__":
    main()
