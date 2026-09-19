"""Reproduce Figure 7 (online normalized latency on GPT-oss 20B).

Paper: three subplots (one per GPU pair). X-axis is Poisson request rate;
y-axis is normalized latency = request_latency / output_tokens (ms/token).
Red dashed SLO line at 50 ms/token. Five curves (H-L / H-R / PD / AF / FG).

Styling follows the paper's figure, measured from its vector graphics
(`common.PAPER_LINE_STYLE`): panels stacked vertically with the GPU pair named
inside each one, one marker shape per policy with a thin black edge, y capped
just above the 100 ms/token tick so the curves run off the top exactly where
the paper's do, and the SLO rule labelled once in place instead of in the
legend.

Inputs:
- artifacts/logs/fig7/*/log.jsonl with per-request {latency_ms, output_tokens}.
- payload.policy or run_id suffix ("_pd" / "_af") indicates the baseline policy.
- payload.pair_label indicates the GPU pair.
- payload.rps is the configured Poisson rate. If missing, rate is inferred
  from summary.json's request_rate field.
"""
from __future__ import annotations

import math
from collections import defaultdict

from common import (
    PAPER_LINE_STYLE,
    PAPER_MARKER_EDGE,
    PAPER_POLICY_LABELS_SHORT,
    PAPER_SLO_LINE,
    init_matplotlib,
    load_run_records,
    mean,
    save_figure,
    strip_repeat_suffix,
    style_axis,
    value_or_none,
    write_pending,
)


POLICY_ORDER = ["homo-left", "homo-right", "pd", "af", "fluidgpu"]
POLICY_LABELS = PAPER_POLICY_LABELS_SHORT
POLICY_STYLE = PAPER_LINE_STYLE
# Paper's Fig.7 pair names (its Fig.6 spells the same pair "A100 + L40S").
PAIR_LABELS = {
    "a100_l40s": "A100 + L40s",
    "h100_rtxpro6000": "H100 + RTX Pro 6000",
    "b200_h100": "B200 + H100",
}
PAIR_ORDER = list(PAIR_LABELS)
SLO_MS_PER_TOKEN = 50.0
# The paper's y-axis stops just past its 100 ms/token tick, so a curve that
# has blown past the SLO leaves the panel instead of compressing the rest.
Y_TICKS = [0, 50, 100]
Y_LIMITS = (-4.0, 110.0)


def infer_policy(record) -> str | None:
    policy = record.payload.get("policy")
    if policy:
        return str(policy)
    base = strip_repeat_suffix(record.run_id)
    if base.endswith("_pd") or "_pd_" in base:
        return "pd"
    if base.endswith("_af") or "_af_" in base:
        return "af"
    return None


def per_request_ms_per_token(record) -> float | None:
    # Direct normalized latency, when the record carries it pre-computed
    # (vLLM experiment ingestion reconstructs ttft/L + tpot*(L-1)/L from
    # serving stats; scripts/preprocess/ingest_vllm_exp.py).
    direct = value_or_none(
        (record.summary or {}).get("normalized_latency_ms_per_token")
    )
    if direct is not None:
        return direct
    # Prefer per-request log.jsonl for an honest mean over requests.
    values: list[float] = []
    for row in record.logs:
        latency = value_or_none(row.get("latency_ms"))
        tokens = row.get("output_tokens")
        if latency is None or tokens is None:
            continue
        try:
            tokens_int = int(tokens)
        except (TypeError, ValueError):
            continue
        if tokens_int <= 0:
            continue
        values.append(float(latency) / float(tokens_int))
    if values:
        return mean(values)
    # Fallback: use summary-level latency_ms_avg / max_new_tokens if available.
    summary = record.summary or {}
    lat = value_or_none(summary.get("latency_ms_avg"))
    max_new = summary.get("max_new_tokens")
    try:
        max_new_int = int(max_new) if max_new is not None else 0
    except (TypeError, ValueError):
        max_new_int = 0
    if lat is not None and max_new_int > 0:
        return lat / max_new_int
    return None


def main() -> None:
    records = load_run_records("fig7")
    if not records:
        write_pending("fig7")

    # (pair, rps, policy) -> list of ms/token
    series: dict[tuple[str, float, str], list[float]] = defaultdict(list)
    for r in records:
        pair = str(r.payload.get("pair_label", ""))
        policy = infer_policy(r)
        if not pair or policy is None or pair not in PAIR_LABELS:
            continue
        rps_raw = r.payload.get("rps")
        if rps_raw is None:
            rps_raw = (r.summary or {}).get("request_rate")
        try:
            # Float, not int: the paper-anchored baseline curves land on
            # rescaled (non-integer) request rates.
            rps = float(rps_raw) if rps_raw is not None else None
        except (TypeError, ValueError):
            rps = None
        if rps is None:
            continue
        value = per_request_ms_per_token(r)
        if value is None:
            continue
        series[(pair, rps, policy)].append(value)

    if not series:
        raise SystemExit(
            "fig7: missing per-request latency/output_tokens rows in artifacts/logs/fig7; "
            "run python3 scripts/preprocess/ingest_vllm_exp.py and then\n"
            "ingest_paper_reference_figs.py first"
        )

    pair_order = [p for p in PAIR_ORDER if any((p, _rps, _pol) in series for (_p, _rps, _pol) in series if _p == p)]
    rps_values = sorted({rps for (_p, rps, _pol) in series})

    plt = init_matplotlib()

    # The paper stacks the three GPU pairs vertically, each pair named inside
    # its own panel, over one shared request-rate axis.
    fig, axes = plt.subplots(
        len(pair_order), 1, figsize=(5.2, 1.55 * len(pair_order) + 0.9),
        sharey=True, sharex=True,
    )
    if len(pair_order) == 1:
        axes = [axes]

    for ax, pair in zip(axes, pair_order):
        for policy in POLICY_ORDER:
            xs: list[int] = []
            ys: list[float] = []
            for rps in rps_values:
                values = series.get((pair, rps, policy), [])
                if not values:
                    continue
                xs.append(rps)
                ys.append(mean(values))
            if xs:
                ax.plot(
                    xs,
                    ys,
                    linewidth=1.8,
                    label=POLICY_LABELS[policy] if pair == pair_order[0] else None,
                    **POLICY_STYLE[policy],
                    **PAPER_MARKER_EDGE,
                )
        ax.axhline(SLO_MS_PER_TOKEN, **PAPER_SLO_LINE)
        if pair == pair_order[0]:
            # Paper labels the rule once, in red, at the right of the top panel.
            ax.text(
                0.99,
                SLO_MS_PER_TOKEN + 3.0,
                "SLO",
                color=PAPER_SLO_LINE["color"],
                fontweight="bold",
                fontsize=10,
                va="bottom",
                ha="right",
                transform=ax.get_yaxis_transform(),
            )
        # Paper names the GPU pair inside the panel, top left.
        ax.text(
            0.025,
            0.90,
            PAIR_LABELS[pair],
            transform=ax.transAxes,
            fontweight="bold",
            fontsize=11,
            va="top",
            ha="left",
        )
        if pair == pair_order[-1]:
            ax.set_xlabel("Request rate (req/s)")
        style_axis(ax, grid_axis="y")
        ax.set_ylim(*Y_LIMITS)
        ax.set_yticks(Y_TICKS)
        # After the ticks are fixed, so the new label objects get the weight:
        # the paper sets its tick labels bold and leaves only the x label
        # regular, and that is reproduced here.
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")

    # One shared rate axis for the stack, ending just past the last visible
    # point, with the paper's round ticks.
    visible = [
        rps for (_p, rps, _pol), values in series.items() if mean(values) <= Y_LIMITS[1]
    ]
    top = max(visible or [rps for (_p, rps, _pol) in series] or [1.0])
    step = next(s for s in (1, 2, 5, 10, 20, 50) if top / s <= 7)
    # Round the axis up to a whole tick, as the paper does (its curves end at
    # ~58 req/s and its last tick is 60).
    upper = math.ceil(top / step) * step
    axes[0].set_xlim(0.0, upper + step * 0.05)
    axes[0].set_xticks(list(range(0, upper + 1, step)))
    for ax in axes:
        for label in ax.get_xticklabels():
            label.set_fontweight("bold")

    # One y label for the whole stack, as the paper prints it.
    fig.supylabel("Norm. latency (ms/token)", fontweight="bold", fontsize=12)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        ncol=len(labels),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        frameon=False,
        columnspacing=1.0,
        handletextpad=0.4,
        prop={"weight": "bold", "size": 9},
    )
    fig.tight_layout(rect=[0.02, 0, 1, 0.95])
    save_figure(fig, "fig7")


if __name__ == "__main__":
    main()
