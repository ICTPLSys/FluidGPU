"""Reproduce Figure 2 (kernel heterogeneity between A100 and L40s).

Paper:
- (a) CDF of the kernel execution-time ratio L40s/A100, one curve per model,
      with the value at ratio = 1 marked (the fraction of kernels that run
      faster on the L40s).
- (b) E2E time ratio: of the total kernel time on the A100, the percentage
      contributed by kernels that run faster on the L40s.

Input: artifacts/logs/fig2/<model>_kernel_census/ records written by
`scripts/preprocess/ingest_kernel_census.py`. Each summary points at its ratio
list under artifacts/validation/kernel_census/ and carries the derived
`e2e_time_ratio_pct` and `cdf_at_ratio_1`.
"""
from __future__ import annotations

from pathlib import Path

from common import (
    REPO_ROOT,
    PAPER_MODEL_NAMES,
    PALETTE,
    init_matplotlib,
    load_run_records,
    save_figure,
    value_or_none,
    write_pending,
)


# Paper's Fig.2 legend order and series colors.
MODEL_ORDER = [
    "sd35_medium",
    "llama31_8b",
    "gptoss20b",
    "qwen25vl7b",
    "mamba_codestral7b",
]
MODEL_COLORS = {
    "sd35_medium": PALETTE["orion"],
    "llama31_8b": PALETTE["reef"],
    "gptoss20b": PALETTE["lithos"],
    "qwen25vl7b": PALETTE["hummingbird"],
    "mamba_codestral7b": PALETTE["fern"],
}
# Plotting every sample of a 1.3M-kernel census is pointless for a monotone
# step curve; evenly spaced order statistics render identically.
MAX_CURVE_POINTS = 4000


def read_ratios(path: Path) -> list[float]:
    values: list[float] = []
    for chunk in path.read_text().split():
        v = value_or_none(chunk)
        if v is not None and v > 0.0:
            values.append(v)
    values.sort()
    return values


def ecdf(values: list[float]) -> tuple[list[float], list[float]]:
    n = len(values)
    if n == 0:
        return [], []
    if n <= MAX_CURVE_POINTS:
        idx = range(n)
    else:
        step = (n - 1) / (MAX_CURVE_POINTS - 1)
        idx = sorted({int(round(i * step)) for i in range(MAX_CURVE_POINTS)})
    return [values[i] for i in idx], [(i + 1) / n for i in idx]


def main() -> None:
    records = load_run_records("fig2")
    if not records:
        write_pending("fig2")

    by_model = {}
    for row in records:
        model = str(row.payload.get("model_label", "")).strip()
        ratios_file = row.summary.get("ratios_file")
        if not model or not ratios_file:
            continue
        by_model[model] = (REPO_ROOT / str(ratios_file), row.summary)

    models = [m for m in MODEL_ORDER if m in by_model]
    if not models:
        raise SystemExit(
            "fig2: no kernel-census records; run "
            "python3 scripts/preprocess/ingest_kernel_census.py first"
        )

    plt = init_matplotlib()
    from matplotlib.patches import Patch
    from matplotlib.ticker import ScalarFormatter

    fig, (ax_cdf, ax_bar) = plt.subplots(
        1, 2, figsize=(7.1, 3.0), dpi=120, gridspec_kw={"wspace": 0.3}
    )

    # --- (a) CDF of the duration ratio -------------------------------------
    ax_cdf.set_xscale("log")
    ax_cdf.axvspan(0.2, 1.0, facecolor="#2c3e50", alpha=0.1)
    ax_cdf.axvspan(1.0, 5.0, facecolor="#ecf0f1", alpha=0.5)
    ax_cdf.axvline(1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.8)

    for model in models:
        path, summary = by_model[model]
        xs, ys = ecdf(read_ratios(path))
        color = MODEL_COLORS[model]
        ax_cdf.plot(xs, ys, color=color, linewidth=1.5, zorder=3)
        at_one = value_or_none(summary.get("cdf_at_ratio_1"))
        if at_one is not None:
            ax_cdf.scatter(
                [1.0], [at_one], color=color, s=35, zorder=4, edgecolors="white"
            )

    ax_cdf.set_ylabel("CDF (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax_cdf.set_xlim(0.25, 3.5)
    ax_cdf.set_ylim(-0.02, 1.02)
    ax_cdf.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax_cdf.set_xticks([0.25, 0.5, 1, 2, 3])
    ax_cdf.xaxis.set_major_formatter(ScalarFormatter())
    ax_cdf.text(
        0.45, 0.85, r"$\leftarrow$L40s Faster",
        color="#2c3e50", fontweight="bold", ha="center", fontsize=9,
    )
    ax_cdf.text(
        2.2, 0.15, r"A100 Faster$\rightarrow$",
        color="#7f8c8d", fontweight="bold", ha="center", fontsize=9,
    )
    ax_cdf.grid(True, which="both", linestyle=":", alpha=0.4)

    # --- (b) E2E time ratio ------------------------------------------------
    values = [value_or_none(by_model[m][1].get("e2e_time_ratio_pct")) or 0.0 for m in models]
    colors = [MODEL_COLORS[m] for m in models]
    ax_bar.bar(
        range(len(models)),
        values,
        width=0.72,
        color=colors,
        alpha=0.6,
        edgecolor=colors,
        linewidth=2,
    )
    ax_bar.set_ylabel("E2E Time Ratio (%)", fontsize=11, fontweight="bold", labelpad=8)
    ax_bar.set_ylim(0, 70)
    ax_bar.set_yticks([0, 20, 40, 60])
    ax_bar.set_xticks([])
    ax_bar.margins(x=0.08)
    ax_bar.grid(axis="y", linestyle=":", alpha=0.4)

    fig.legend(
        handles=[
            Patch(facecolor=MODEL_COLORS[m], label=PAPER_MODEL_NAMES.get(m, m))
            for m in models
        ],
        loc="lower left",
        bbox_to_anchor=(0.08, 0.84, 0.86, 0.08),
        ncol=len(models),
        mode="expand",
        prop={"size": 8.4, "weight": "bold"},
        handlelength=0.85,
        handletextpad=0.28,
        columnspacing=0.95,
        frameon=False,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(left=0.13, right=0.965, bottom=0.19, top=0.79, wspace=0.3)
    save_figure(fig, "fig2")


if __name__ == "__main__":
    main()
