from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]

PALETTE = {
    "orion": "#91CAE8",
    "reef": "#F48892",
    "lithos": "#FBCE6A",
    "hummingbird": "#A9CA70",
    "gray": "#8c8c8c",
    "light_gray": "#b8b8b8",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def find_logs(figure: str) -> list[Path]:
    root = Path("artifacts/logs") / figure
    if not root.exists():
        return []
    return sorted(root.glob("*/log.jsonl"))


@dataclass(frozen=True)
class RunRecord:
    figure: str
    run_id: str
    run_dir: Path
    meta: dict[str, Any]
    payload: dict[str, Any]
    summary: dict[str, Any]
    logs: list[dict[str, Any]]


def figure_root(figure: str) -> Path:
    return Path("artifacts/logs") / figure


def load_run_records(figure: str) -> list[RunRecord]:
    root = figure_root(figure)
    if not root.exists():
        return []
    records: list[RunRecord] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        meta_path = run_dir / "meta.json"
        meta = read_json(meta_path) if meta_path.exists() else {}
        payload = dict(meta.get("payload", {}))
        summary_path = run_dir / "summary.json"
        summary = read_json(summary_path) if summary_path.exists() else {}
        log_path = run_dir / "log.jsonl"
        logs = load_jsonl(log_path) if log_path.exists() else []
        records.append(
            RunRecord(
                figure=figure,
                run_id=run_dir.name,
                run_dir=run_dir,
                meta=meta,
                payload=payload,
                summary=summary,
                logs=logs,
            )
        )
    return records


def strip_repeat_suffix(run_id: str) -> str:
    return re.sub(r"_r\d+$", "", run_id)


def value_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_value(mapping: dict[str, Any], *keys: str) -> float | int | str | None:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def request_latency_ms_per_token(summary: dict[str, Any], logs: list[dict[str, Any]]) -> float | None:
    per_request: list[float] = []
    for row in logs:
        latency_ms = value_or_none(row.get("latency_ms"))
        output_tokens = value_or_none(row.get("output_tokens"))
        if latency_ms is None or output_tokens is None or output_tokens <= 0.0:
            continue
        per_request.append(latency_ms / output_tokens)
    if per_request:
        return mean(per_request)

    latency_ms_avg = value_or_none(
        first_value(summary, "latency_ms_per_token", "normalized_latency_ms_per_token")
    )
    if latency_ms_avg is not None:
        return latency_ms_avg

    latency_ms_avg = value_or_none(summary.get("latency_ms_avg"))
    generated_tokens = value_or_none(summary.get("generated_tokens"))
    requests = value_or_none(summary.get("requests"))
    if (
        latency_ms_avg is None
        or generated_tokens is None
        or requests is None
        or generated_tokens <= 0.0
        or requests <= 0.0
    ):
        return None
    avg_output_tokens = generated_tokens / requests
    if avg_output_tokens <= 0.0:
        return None
    return latency_ms_avg / avg_output_tokens


def throughput_tok_s(summary: dict[str, Any], logs: list[dict[str, Any]]) -> float | None:
    tok_s = value_or_none(summary.get("throughput_tok_s"))
    if tok_s is not None:
        return tok_s

    generated_tokens = value_or_none(summary.get("generated_tokens"))
    elapsed_s = value_or_none(summary.get("elapsed_s"))
    if generated_tokens is not None and elapsed_s is not None and elapsed_s > 0.0:
        return generated_tokens / elapsed_s

    if not logs:
        return None
    total_tokens = 0.0
    total_latency_ms = 0.0
    for row in logs:
        output_tokens = value_or_none(row.get("output_tokens"))
        latency_ms = value_or_none(row.get("latency_ms"))
        if output_tokens is None or latency_ms is None:
            continue
        total_tokens += output_tokens
        total_latency_ms += latency_ms
    if total_tokens <= 0.0 or total_latency_ms <= 0.0:
        return None
    return total_tokens / (total_latency_ms / 1000.0)


def throughput_images_min(summary: dict[str, Any]) -> float | None:
    images_min = value_or_none(
        first_value(summary, "throughput_images_min", "images_per_min")
    )
    if images_min is not None:
        return images_min
    req_s = value_or_none(summary.get("throughput_req_s"))
    if req_s is None:
        return None
    return req_s * 60.0


def switch_count(summary: dict[str, Any], logs: list[dict[str, Any]]) -> int | None:
    raw = first_value(summary, "switch_count", "policy_switch_count")
    if raw is not None:
        return int(raw)
    for row in reversed(logs):
        raw = first_value(row, "switch_count", "policy_switch_count")
        if raw is not None:
            return int(raw)
    return None


def mean(values: Iterable[float]) -> float:
    items = list(values)
    if not items:
        raise AssertionError("mean() expects a non-empty iterable")
    return sum(items) / len(items)


def save_figure(fig, figure: str) -> None:
    out_dir = Path("artifacts/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{figure}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out_dir / f"{figure}.png", dpi=180, bbox_inches="tight", pad_inches=0.02)
    print(f"{figure}: wrote {out_dir / f'{figure}.pdf'}")


def init_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    )
    import matplotlib.pyplot as plt

    return plt


def style_axis(ax, *, grid_axis: str = "both") -> None:
    ax.set_axisbelow(True)
    if grid_axis in ("both", "x"):
        ax.grid(axis="x", linestyle="--", alpha=0.35, linewidth=0.8)
    if grid_axis in ("both", "y"):
        ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.0)
        spine.set_color("black")
    ax.tick_params(axis="both", labelsize=11, width=1.0)


def write_pending(figure: str) -> None:
    raise SystemExit(f"{figure}: missing input logs; run scripts/repro.sh --fig {figure.removeprefix('fig')} first")


def simple_bar_plot(figure: str, title: str) -> bool:
    logs = find_logs(figure)
    if not logs:
        write_pending(figure)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [path.parent.name for path in logs]
    values = [metric_value(path) for path in logs]
    out_dir = Path("artifacts/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(max(4, len(labels) * 0.7), 3))
    ax.bar(labels, values, color="#2f6f6d")
    ax.set_title(title)
    ax.set_ylabel("AD sanity value")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out_dir / f"{figure}.pdf")
    fig.savefig(out_dir / f"{figure}.png", dpi=160)
    print(f"{figure}: wrote {out_dir / f'{figure}.pdf'}")
    return True


def metric_value(log_path: Path) -> float:
    summary_path = log_path.parent / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        for key in ("throughput_req_s", "latency_ms_avg", "solver_ms", "elapsed_s"):
            if key in summary:
                return float(summary[key])
    rows = load_jsonl(log_path)
    for row in rows:
        if row.get("status") == "failed":
            raise SystemExit(f"{log_path}: run failed: {row.get('reason', 'unknown reason')}")
        for key in ("throughput_req_s", "latency_ms_avg", "solver_ms", "elapsed_s"):
            if key in row:
                return float(row[key])
    raise SystemExit(f"{log_path}: no numeric metric found")


# ---------------------------------------------------------------------------
# Paper-matching style
#
# The paper draws every figure from one 5-hue family; the four hues already in
# PALETTE are exactly its colors, IRIS is the fifth. Per-figure assignments and
# legend wording below are transcribed from the paper's own vector graphics and
# text layer, so the artifact's figures read as the same figure set. The
# camera-ready adds a sixth Fig.6 series (Request Dist.) drawn in neutral gray,
# outside that hue family -- it is a routing baseline, not a disaggregation
# policy, and the paper sets it apart the same way.
#
# Deliberate deviations from the dataviz defaults, because the paper is the
# spec here: the paper uses dual y-axes (Fig.6/10/11a) and its adjacent
# purple/salmon pair sits below the normal-vision separation floor (measured
# dE 11.9). Mitigation kept in-style: the paper's own thin black bar edges
# (PAPER_BAR_EDGE) plus direct value labels act as the secondary encoding.
# Fig.7 needs no such mitigation -- there the paper gives every curve its own
# marker shape (PAPER_LINE_STYLE), so colour is not the only channel.
# ---------------------------------------------------------------------------

PALETTE["iris"] = "#C4A6D6"       # Fig.6 homogeneous-left
PALETTE["fern"] = "#2CA02C"       # Fig.2 fifth model series
PALETTE["amber"] = "#FFC208"      # Fig.11 policy-switch bars
PALETTE["brick"] = "#D42E2E"      # Fig.11 latency line
PALETTE["tangerine"] = "#E8913D"  # Fig.12a throughput line
PALETTE["slate"] = "#B3B3B3"      # Fig.6 request distribution (camera-ready)
PALETTE["silver"] = "#BFBFBF"     # Fig.7 homogeneous-left line
PALETTE["mint"] = "#A8D5BA"       # Fig.7 homogeneous-right line

PAPER_POLICY_COLORS = {
    "homo-left": PALETTE["iris"],
    "homo-right": PALETTE["reef"],
    "request-dist": PALETTE["slate"],
    "pd": PALETTE["lithos"],
    "af": PALETTE["orion"],
    "fluidgpu": PALETTE["hummingbird"],
}
# Fig.6 legend wording; Fig.7 abbreviates the two homogeneous entries and has
# no Request Dist. series.
PAPER_POLICY_LABELS = {
    "homo-left": "Homogeneous (left)",
    "homo-right": "Homogeneous (right)",
    "request-dist": "Request Dist.",
    "pd": "PD Dis.",
    "af": "AF Dis.",
    "fluidgpu": "FluidGPU",
}
PAPER_POLICY_LABELS_SHORT = {
    "homo-left": "Homo. (Left)",
    "homo-right": "Homo. (Right)",
    "request-dist": "Request Dist.",
    "pd": "PD Dis.",
    "af": "AF Dis.",
    "fluidgpu": "FluidGPU",
}
# Paper's model axis ticks (Fig.6) and its spelled-out names (Fig.2 legend).
PAPER_MODEL_TICKS = {
    "llama31_8b": "LM",
    "gptoss20b": "GT",
    "qwen25vl7b": "QW",
    "mamba_codestral7b": "MB",
    "sd35_medium": "SD",
}
PAPER_MODEL_NAMES = {
    "llama31_8b": "Llama3 8B",
    "gptoss20b": "GPT-oss 20B",
    "qwen25vl7b": "Qwen2.5VL 7B",
    "mamba_codestral7b": "Mamba 7B",
    "sd35_medium": "SD3.5",
}
PAPER_BAR_EDGE = {"edgecolor": "black", "linewidth": 0.4}

# Fig.7 draws curves, not bars, and the paper gives it its own series set:
# every curve is a filled marker with a thin black edge, the line in the
# marker's fill colour, and the SHAPE carries the identity (the two
# homogeneous curves are gray/mint, which no bar figure uses). Measured from
# the paper's vector graphics; the diamond is drawn 1.41x the others so its
# edge length matches theirs, as in the paper. PD and AF swapped colour and
# shape between paper revisions (PD is now the gold square).
PAPER_LINE_STYLE = {
    "homo-left": {"color": PALETTE["silver"], "marker": "o", "markersize": 6.0},
    "homo-right": {"color": PALETTE["mint"], "marker": "v", "markersize": 6.0},
    "pd": {"color": PALETTE["lithos"], "marker": "s", "markersize": 6.0},
    "af": {"color": PALETTE["orion"], "marker": "^", "markersize": 6.0},
    "fluidgpu": {"color": PALETTE["reef"], "marker": "D", "markersize": 7.5},
}
PAPER_MARKER_EDGE = {"markeredgecolor": "black", "markeredgewidth": 0.5}
# Fig.7's SLO rule: pure red, dashed, thicker than the gridlines, and labelled
# in-place rather than in the legend.
PAPER_SLO_LINE = {"color": "#FF0000", "linestyle": "--", "linewidth": 1.3}
# Fig.10(a) marks the zero-communication bound in a softer red than Fig.7's
# SLO rule; both are measured from the paper's vector graphics.
PAPER_OPTIMAL_LINE_COLOR = "#E74C3C"


def paper_panel_labels(fig, labelled_axes, *, pad_points: float = 4.0, fontsize: int = 13) -> None:
    """Name each panel's hardware pair below its x-axis, as the paper does.

    `labelled_axes` is an iterable of (axes, text). The paper puts no title
    above a panel; the pair sits under the x-axis in bold, one step larger than
    the tick labels.

    Call this AFTER the layout is final (e.g. after `fig.tight_layout()`): the
    label is placed just below the space the x-axis actually occupies, which is
    what keeps the gap identical across panel heights and across panels that do
    or do not carry an x-axis label. Anchoring the same text with
    `annotate(xycoords=ax.xaxis)` instead looks equivalent but is silently
    dropped from the output.
    """
    fig.canvas.draw()  # resolve tick/label extents before measuring them
    renderer = fig.canvas.get_renderer()
    to_figure = fig.transFigure.inverted()
    for ax, text in labelled_axes:
        bbox = ax.xaxis.get_tightbbox(renderer)
        if bbox is None:
            continue
        position = ax.get_position()
        y = to_figure.transform((0.0, bbox.y0 - pad_points * fig.dpi / 72.0))[1]
        fig.text(
            (position.x0 + position.x1) / 2.0,
            y,
            text,
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=fontsize,
        )


def paper_value_labels(ax, xs, values, fmt="{:.0f}", fontsize=7, dy=0.012, fontweight="normal"):
    """Direct value labels above bars, as the paper prints them."""
    if not values:
        return
    span = max(values)
    for x, v in zip(xs, values):
        if v is None:
            continue
        ax.annotate(
            fmt.format(v),
            (x, v + span * dy),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            fontweight=fontweight,
            color="#222222",
        )
