"""Reproduce Figure 12a (slow-network robustness on A100+L40s, GPT-oss 20B).

Paper: single figure with dual y-axis.
  Left y  : Normalized latency (ms/token) under "online" mode (light load).
  Right y : Throughput (tokens/s) under "offline" mode.
  X-axis  : Interconnect bandwidth (Gbps), swept 200 -> 100 -> 50 -> 25.
  Horizontal reference: Homo. A100 normalized latency, annotated as a dashed
  line on the left axis.

Inputs per run (artifacts/logs/fig12a/*):
- payload.bandwidth_gbps, payload.mode ("offline" | "online").
- summary.json: throughput_tok_s; norm_latency_ms_per_token (fallback to
  per-request log.jsonl or latency_ms_avg / max_new_tokens).
- Homo. A100 reference: looked up from payload.homo_a100_norm_latency_ms_tok
  on any online run if set, or the summary field `homo_a100_norm_latency_ms_tok`.
"""
from __future__ import annotations

import math
from collections import defaultdict

from common import (
    PAPER_MARKER_EDGE,
    PAPER_OPTIMAL_LINE_COLOR,
    PAPER_BAR_EDGE,
    PALETTE,
    paper_value_labels,
    init_matplotlib,
    load_run_records,
    mean,
    save_figure,
    style_axis,
    value_or_none,
    write_pending,
)


BANDWIDTH_ORDER = [200, 100, 50, 25]


def norm_latency(record) -> float | None:
    summary = record.summary or {}
    direct = value_or_none(summary.get("norm_latency_ms_per_token"))
    if direct is not None:
        return direct
    per_req: list[float] = []
    for row in record.logs:
        lat = value_or_none(row.get("latency_ms"))
        tokens = row.get("output_tokens")
        try:
            t = int(tokens) if tokens is not None else 0
        except (TypeError, ValueError):
            t = 0
        if lat is not None and t > 0:
            per_req.append(float(lat) / float(t))
    if per_req:
        return mean(per_req)
    lat_avg = value_or_none(summary.get("latency_ms_avg"))
    max_new = summary.get("max_new_tokens")
    try:
        max_int = int(max_new) if max_new is not None else 0
    except (TypeError, ValueError):
        max_int = 0
    if lat_avg is not None and max_int > 0:
        return lat_avg / max_int
    return None


def main() -> None:
    records = load_run_records("fig12a")
    if not records:
        write_pending("fig12a")

    offline_tok: dict[int, list[float]] = defaultdict(list)
    online_lat: dict[int, list[float]] = defaultdict(list)
    homo_ref_candidates: list[float] = []
    for r in records:
        mode = str(r.payload.get("mode", "")).strip().lower()
        bw_raw = r.payload.get("bandwidth_gbps")
        try:
            bw = int(float(bw_raw)) if bw_raw is not None else None
        except (TypeError, ValueError):
            bw = None
        if bw is None:
            continue
        if mode == "offline":
            tok = value_or_none(r.summary.get("throughput_tok_s"))
            if tok is not None:
                offline_tok[bw].append(tok)
        elif mode == "online":
            nl = norm_latency(r)
            if nl is not None:
                online_lat[bw].append(nl)
        ref = value_or_none(r.payload.get("homo_a100_norm_latency_ms_tok"))
        if ref is None:
            ref = value_or_none(r.summary.get("homo_a100_norm_latency_ms_tok"))
        if ref is not None:
            homo_ref_candidates.append(ref)

    if not online_lat and not offline_tok:
        raise SystemExit(
            "fig12a: no offline throughput or online latency found; "
            "run bash scripts/repro.sh --fig 12a first"
        )

    plt = init_matplotlib()

    fig, ax_left = plt.subplots(figsize=(6.2, 3.8))
    ax_right = ax_left.twinx()

    xs = [bw for bw in BANDWIDTH_ORDER if bw in online_lat or bw in offline_tok]
    lat_series = [mean(online_lat[bw]) if online_lat.get(bw) else None for bw in xs]
    tok_series = [mean(offline_tok[bw]) if offline_tok.get(bw) else None for bw in xs]

    # The paper normalizes latency to the fastest link (the leftmost point)
    # and plots it as bars, with throughput as a line on the right axis.
    lat_ref = next((v for v in lat_series if v), None)
    lat_norm = [v / lat_ref if v and lat_ref else None for v in lat_series]
    positions = list(range(len(xs)))
    bar_pos = [p for p, v in zip(positions, lat_norm) if v is not None]
    bar_val = [v for v in lat_norm if v is not None]
    ax_left.bar(
        bar_pos,
        bar_val,
        width=0.55,
        color=PALETTE["orion"],
        label="Norm. Latency",
        **PAPER_BAR_EDGE,
    )
    paper_value_labels(ax_left, bar_pos, bar_val, fmt="{:.2f}", dy=0.02,
                       fontsize=9, fontweight="bold")
    line_pos = [p for p, v in zip(positions, tok_series) if v is not None]
    line_val = [v for v in tok_series if v is not None]
    ax_right.plot(
        line_pos,
        line_val,
        marker="o",
        markersize=7,
        linewidth=2.0,
        color=PALETTE["tangerine"],
        label="Throughput",
        **PAPER_MARKER_EDGE,
    )
    # The paper prints the throughput values BELOW their markers, so they do
    # not collide with the bar labels above.
    span = (max(line_val) - min(line_val)) or 1.0
    for x, v in zip(line_pos, line_val):
        ax_right.annotate(
            f"{v:.0f}", (x, v - span * 0.22), ha="center", va="top",
            fontsize=9, fontweight="bold", color="#222222",
        )
    ax_left.set_xticks(positions)
    ax_left.set_xticklabels([str(b) for b in xs])

    ref = 0.0
    if homo_ref_candidates and lat_ref:
        ref = mean(homo_ref_candidates) / lat_ref
        # Paper draws this reference in red, labelled in bold above it.
        ax_left.axhline(ref, color=PAPER_OPTIMAL_LINE_COLOR, linestyle="--", linewidth=1.6)
        ax_left.annotate(
            f"Homo. A100: {ref:.2f}",
            xy=(0.5, ref),
            xycoords=ax_left.get_yaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color=PAPER_OPTIMAL_LINE_COLOR,
        )

    ax_left.set_xlabel("Bandwidth (Gbps)", fontweight="bold")
    ax_left.set_ylabel("Norm. Latency", fontweight="bold")
    ax_right.set_ylabel("Tput (tokens/s)", fontweight="bold")
    # Categorical x positions (BANDWIDTH_ORDER is already 200 -> 25, the
    # paper's left-to-right order), so no tick/inversion pass here.
    # Paper's left axis: 0 to a round step above the reference line (0.0-1.5
    # for a 1.18 reference), ticked in halves.
    left_step = 0.5
    left_top = math.ceil(max(bar_val + [ref]) / left_step) * left_step
    ax_left.set_ylim(0.0, left_top)
    ax_left.set_yticks([i * left_step for i in range(int(round(left_top / left_step)) + 1)])
    if line_val:
        # The paper does not start this axis at zero: it frames the throughput
        # series so its degradation is visible (2000-3200 for a 2490-2641
        # series). Pad by three times the series' own span and round outward to
        # a nice step, which reproduces that framing from the data itself.
        span = max(max(line_val) - min(line_val), 1e-9)
        lo, hi = min(line_val) - 3 * span, max(line_val) + 3 * span
        exponent = math.floor(math.log10((hi - lo) / 3))
        step = min(
            (m * 10 ** exponent for m in (1, 2, 2.5, 4, 5, 10)),
            key=lambda cand: abs(cand - (hi - lo) / 3),
        )
        lo = max(0.0, math.floor(lo / step) * step)
        hi = math.ceil(hi / step) * step
        ax_right.set_ylim(lo, hi)
        ax_right.set_yticks([lo + i * step for i in range(int(round((hi - lo) / step)) + 1)])
    style_axis(ax_left, grid_axis="both")

    # Paper keeps this legend inside the panel, boxed, at the lower left.
    handles, labels = ax_left.get_legend_handles_labels()
    handles_r, labels_r = ax_right.get_legend_handles_labels()
    ax_left.legend(
        handles + handles_r,
        labels + labels_r,
        loc="lower left",
        frameon=True,
        framealpha=0.92,
        edgecolor="#b0b0b0",
        prop={"weight": "bold", "size": 9},
    )
    for label in ax_left.get_xticklabels() + ax_left.get_yticklabels() + ax_right.get_yticklabels():
        label.set_fontweight("bold")
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_figure(fig, "fig12a")


if __name__ == "__main__":
    main()
