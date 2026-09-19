"""Reproduce Figure 11 (online monitor sensitivity on GPT-oss 20B).

Paper layout:
  (a) W sweep at β = 1.5.   Left y: norm. latency (ms/token). Right y: policy switch count.
  (b) β sweep at W = 300ms. Same two axes.
  Both sub-panels use a line for latency and a bar chart for policy switches.

Inputs per run (artifacts/logs/fig11/*):
- payload.window_ms, payload.beta (injected by the orchestrator matrix).
- summary.json: normalized latency (ms/token). We prefer a dedicated field
  `norm_latency_ms_per_token`; otherwise we derive from latency_ms_avg /
  max_new_tokens as a fallback.
- summary.json: policy_switch_count (exposed by QueueingAwareMonitor.summary()
  and written by llm_generate.emit_results when the monitor is active).
"""
from __future__ import annotations

from collections import defaultdict

from common import (
    PALETTE,
    init_matplotlib,
    load_run_records,
    mean,
    save_figure,
    style_axis,
    value_or_none,
    write_pending,
)


DEFAULT_W_MS = 300.0
DEFAULT_BETA = 1.5


def norm_latency(record) -> float | None:
    summary = record.summary or {}
    direct = value_or_none(summary.get("norm_latency_ms_per_token"))
    if direct is not None:
        return direct
    # Fallback: prefer per-request log.jsonl computation.
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


def switch_count(record) -> int | None:
    summary = record.summary or {}
    for key in ("policy_switch_count", "switch_count"):
        raw = summary.get(key)
        if raw is None:
            continue
        try:
            return int(raw)
        except (TypeError, ValueError):
            continue
    queueing = summary.get("queueing_summary") if isinstance(summary, dict) else None
    if isinstance(queueing, dict):
        raw = queueing.get("switch_count")
        try:
            return int(raw) if raw is not None else None
        except (TypeError, ValueError):
            return None
    return None


def main() -> None:
    records = load_run_records("fig11")
    if not records:
        write_pending("fig11")

    # Collect (window_ms, beta) -> (norm_latency, switch_count).
    points_lat: dict[tuple[float, float], list[float]] = defaultdict(list)
    points_switch: dict[tuple[float, float], list[int]] = defaultdict(list)
    for r in records:
        w = r.payload.get("window_ms")
        b = r.payload.get("beta")
        if w is None or b is None:
            continue
        try:
            wv = float(w)
            bv = float(b)
        except (TypeError, ValueError):
            continue
        lat = norm_latency(r)
        if lat is not None:
            points_lat[(wv, bv)].append(lat)
        sc = switch_count(r)
        if sc is not None:
            points_switch[(wv, bv)].append(sc)

    if not points_lat:
        raise SystemExit("fig11: no normalized-latency data found; run bash scripts/repro.sh --fig 11 first")

    windows = sorted({w for (w, _) in points_lat})
    betas = sorted({b for (_, b) in points_lat})
    near_beta = min(betas, key=lambda v: abs(v - DEFAULT_BETA)) if betas else DEFAULT_BETA
    near_window = min(windows, key=lambda v: abs(v - DEFAULT_W_MS)) if windows else DEFAULT_W_MS

    def at(points, w, b):
        values = points.get((w, b), [])
        return mean(values) if values else None

    plt = init_matplotlib()

    fig, (ax_w, ax_b) = plt.subplots(1, 2, figsize=(10.2, 3.8))
    ax_w_tw = ax_w.twinx()
    ax_b_tw = ax_b.twinx()

    xs_w = [w for w in windows if (w, near_beta) in points_lat]
    lat_w = [at(points_lat, w, near_beta) for w in xs_w]
    sw_w = [at(points_switch, w, near_beta) for w in xs_w]

    if any(v is not None for v in sw_w):
        # Per-point multiplicative widths: the x-axis is logarithmic, so a
        # constant linear width renders the low-W bars panel-wide.
        ax_w_tw.bar(
            xs_w,
            [v if v is not None else 0.0 for v in sw_w],
            width=[w * 0.32 for w in xs_w],
            color=PALETTE["amber"],
            label="Policy Switches",
        )
    ax_w.plot(
        xs_w,
        lat_w,
        marker="o",
        linewidth=2.0,
        color=PALETTE["brick"],
        label="Norm. latency",
    )
    ax_w.set_xscale("log")
    ax_w.set_xlabel("Window size W (ms)")
    ax_w.set_ylabel("Norm. Latency\n(ms/token)")
    ax_w_tw.set_ylabel("Policy Switches")
    ax_w.set_title(rf"(a) W sweep ($\beta$={near_beta:g})")
    ax_w.set_xticks(xs_w)
    ax_w.set_xticklabels([str(int(w)) for w in xs_w])
    style_axis(ax_w, grid_axis="both")

    xs_b = [b for b in betas if (near_window, b) in points_lat]
    lat_b = [at(points_lat, near_window, b) for b in xs_b]
    sw_b = [at(points_switch, near_window, b) for b in xs_b]

    if any(v is not None for v in sw_b):
        width = (max(xs_b) - min(xs_b)) / 24.0 if len(xs_b) > 1 else 0.15
        ax_b_tw.bar(
            xs_b,
            [v if v is not None else 0.0 for v in sw_b],
            width=width,
            color=PALETTE["amber"],
            label="Policy Switches",
        )
    ax_b.plot(
        xs_b,
        lat_b,
        marker="s",
        linewidth=2.0,
        color=PALETTE["brick"],
        label=None,
    )
    ax_b.set_xlabel(r"Threshold $\beta$")
    ax_b.set_ylabel("Norm. Latency\n(ms/token)")
    ax_b_tw.set_ylabel("Policy Switches")
    ax_b.set_title(rf"(b) $\beta$ sweep (W={near_window:g} ms)")
    ax_b.set_xticks(xs_b)
    style_axis(ax_b, grid_axis="both")

    # Keep the latency lines above the twin-axis switch bars.
    for ax, tw in ((ax_w, ax_w_tw), (ax_b, ax_b_tw)):
        ax.set_zorder(tw.get_zorder() + 1)
        ax.patch.set_visible(False)

    handles, labels = ax_w.get_legend_handles_labels()
    handles_tw, labels_tw = ax_w_tw.get_legend_handles_labels()
    fig.legend(handles + handles_tw, labels + labels_tw, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    save_figure(fig, "fig11")


if __name__ == "__main__":
    main()
