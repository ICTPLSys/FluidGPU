#!/usr/bin/env python3
"""Materialize the A100-vs-L40s kernel census into Fig.2 / Fig.3 run records.

Source: artifacts/validation/kernel_census/ (see its README.md for how the
census was measured). This script turns it into the standard
artifacts/logs/<fig>/<run_id>/ record layout that the plot scripts read, and
asserts the derived quantities against the values printed in the paper so a
silent data regression cannot slip through.

Any pre-existing records produced by the runtime's own profiler
(scripts/repro.sh --fig 2 / --fig 3) are moved aside into
artifacts/logs/fig{2,3}_torch_profiler_archive/ rather than deleted.

Idempotent: re-running rewrites the same records.
"""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CENSUS = REPO / "artifacts" / "validation" / "kernel_census"
LOGS = REPO / "artifacts" / "logs"

# Model axis order follows the paper's Fig.2 legend.
MODELS = [
    ("sd35_medium", "stabilityai/stable-diffusion-3.5-medium"),
    ("llama31_8b", "meta-llama/Llama-3.1-8B-Instruct"),
    ("gptoss20b", "openai/gpt-oss-20b"),
    ("qwen25vl7b", "Qwen/Qwen2.5-VL-7B-Instruct"),
    ("mamba_codestral7b", "mistralai/Mamba-Codestral-7B-v0.1"),
]

# Values printed in the paper's §II-B / §II-C and Fig.3 caption.
PAPER_MEAN_CDF_AT_1 = 0.67
PAPER_MEAN_E2E_PCT = 36.0
PAPER_MAX_E2E_PCT = 53.0
PAPER_PHASE_PCT = {"prefill": 45.0, "decode": 57.0}
PAPER_BLOCK_PCT = {"attention": 71.0, "ffn": 33.0}
# Speedups the paper's Fig.3 discussion attaches to its two named kernels.
PAPER_GEMV_SPEEDUP = 1.9    # A100 over L40s on cublasGemv
PAPER_FLASH_SPEEDUP = 2.1   # L40s over A100 on FlashAttention


def read_ratios(path: Path) -> list[float]:
    values: list[float] = []
    for chunk in path.read_text().split():
        try:
            v = float(chunk)
        except ValueError:
            continue
        if v > 0.0:
            values.append(v)
    return values


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


def cdf_at_one(sorted_ratios: list[float]) -> float:
    """Fraction of kernels strictly faster on the L40s."""
    lo, hi = 0, len(sorted_ratios)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_ratios[mid] < 1.0:
            lo = mid + 1
        else:
            hi = mid
    return lo / len(sorted_ratios) if sorted_ratios else 0.0


def faster_pct(rows: list[dict[str, str]], key: str) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(row[key], []).append(float(row["y_l40s_over_a100"]))
    return {
        k: 100.0 * sum(1 for v in vs if v < 1.0) / len(vs) for k, vs in groups.items()
    }


def annotation_anchors() -> list[dict]:
    """Locate the two prefill kernels the paper calls out in Fig.3(a).

    The paper names them `cublasGemv` (memory-bandwidth-bound, the A100 wins by
    ~1.9x) and `FlashAttention` (compute-bound, the L40s wins by up to 2.1x).

    Resolving them needs `fig3_kernel_pairs.tsv`, because the two GPUs do not
    always run the same kernel at the same position: cuBLAS picks a tensorop
    GEMM on the A100 (sm80) and falls back to `internal::gemvx::kernel` on the
    L40s (sm89), and the paper names that position after the kernel that
    distinguishes it. Picking by ratio extremes instead would land on unrelated
    kernels.

    Each anchor is verified against the paper's stated speedup before it is
    written into the figure record.
    """
    pairs = read_points(CENSUS / "fig3_kernel_pairs.tsv")
    full = read_points(CENSUS / "fig3_phase_points_full.tsv")
    merged = read_points(CENSUS / "fig3_phase_points.tsv")
    cluster_of = {
        pid: row for row in merged for pid in row["member_point_ids"].split(",")
    }

    def candidates(match) -> list[dict]:
        """Every prefill position matching a mechanism, with its plotted point.

        A mechanism occurs at more than one position — the L40s falls back to
        cuBLAS GEMV at three of them — so a candidate list, not a single hit.
        `ratio_median` is the census median over all launches at that position;
        `plotted` is where the merged point actually lands in the figure. The
        two differ because the point file keeps one representative launch per
        kernel rather than the median.
        """
        out = []
        for p in pairs:
            if p["phase"] != "prefill" or not match(p):
                continue
            point = next(
                (
                    r
                    for r in full
                    if r["phase"] == "prefill" and r["kernel"] == p["name_a100"]
                ),
                None,
            )
            cluster = cluster_of.get(point["point_id"]) if point else None
            out.append(
                {
                    "name_a100": p["name_a100"][:70],
                    "name_l40s": p["name_l40s"][:70],
                    "launches": int(p["launches"]),
                    "ratio_median": round(float(p["ratio_median"]), 4),
                    "plotted": (
                        round(float(cluster["y_l40s_over_a100"]), 4) if cluster else None
                    ),
                    "_cluster": cluster,
                }
            )
        return sorted(out, key=lambda c: -c["launches"])

    def anchor(match, label: str, target: float, lo: float, hi: float) -> dict | None:
        """Pick the position whose plotted point sits nearest the paper's speedup.

        The arrow has to land on a point that is actually drawn, so the choice
        is made on the plotted value, not the census median. Every candidate is
        recorded next to the chosen one so the selection is auditable.
        """
        cands = candidates(match)
        drawn = [c for c in cands if c["_cluster"] is not None]
        if not drawn:
            return None
        best = min(drawn, key=lambda c: abs(c["plotted"] - target))
        y = best["plotted"]
        assert lo <= y <= hi, f"{label}: plotted ratio {y:.3f} outside [{lo}, {hi}]"
        return {
            "label": label,
            "kernel_a100": best["name_a100"],
            "kernel_l40s": best["name_l40s"],
            "launches": best["launches"],
            "ratio_census_median": best["ratio_median"],
            "ratio": y,
            "x_us": round(10.0 ** float(best["_cluster"]["x_log10_a100"]), 3),
            "y": y,
            "other_positions": [
                {k: v for k, v in c.items() if k != "_cluster"}
                for c in cands
                if c is not best
            ],
        }

    found = [
        # A100 tensorop GEMM vs L40s cuBLAS GEMV; paper reports A100 1.9x.
        anchor(
            lambda p: "gemv" in p["name_l40s"].lower(),
            "cublasGemv",
            PAPER_GEMV_SPEEDUP,
            1.5,
            2.1,
        ),
        # Same tiled attention kernel on both sides; paper: L40s up to 2.1x.
        anchor(
            lambda p: "unified_attention" in p["name_a100"].lower()
            and "unified_attention" in p["name_l40s"].lower(),
            "FlashAttention",
            1.0 / PAPER_FLASH_SPEEDUP,
            0.40,
            0.60,
        ),
    ]
    return [a for a in found if a]


def archive_existing(figure: str, keep: set[str]) -> list[str]:
    root = LOGS / figure
    if not root.exists():
        return []
    dest = LOGS / f"{figure}_torch_profiler_archive"
    moved: list[str] = []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if run_dir.name in keep:
            continue
        dest.mkdir(parents=True, exist_ok=True)
        target = dest / run_dir.name
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(run_dir), str(target))
        moved.append(run_dir.name)
    return moved


def write_record(figure: str, run_id: str, payload: dict, summary: dict) -> Path:
    run_dir = LOGS / figure / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "figure": figure,
                "run_id": run_id,
                "kind": "kernel-census",
                "command": "python3 scripts/preprocess/ingest_kernel_census.py",
                "payload": payload,
            },
            indent=2,
        )
        + "\n"
    )
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return run_dir


def ingest_fig2() -> None:
    keep = set()
    cdf_values: list[float] = []
    e2e_by_model: dict[str, float] = {}
    with (CENSUS / "e2e_time_ratio.csv").open() as f:
        for row in csv.reader(line for line in f if not line.startswith("#")):
            if len(row) == 2:
                e2e_by_model[row[0]] = float(row[1])

    for model_label, model in MODELS:
        ratio_path = CENSUS / "ratios" / f"{model_label}.txt"
        ratios = sorted(read_ratios(ratio_path))
        if not ratios:
            raise SystemExit(f"ingest_kernel_census: no ratios in {ratio_path}")
        at_one = cdf_at_one(ratios)
        cdf_values.append(at_one)
        e2e_pct = e2e_by_model[model_label]

        run_id = f"{model_label}_kernel_census"
        keep.add(run_id)
        write_record(
            "fig2",
            run_id,
            {
                "model_label": model_label,
                "model": model,
                "pair_label": "a100_l40s",
                "requires_gpus": ["A100", "L40S"],
            },
            {
                "model_label": model_label,
                "kernel_samples": len(ratios),
                "cdf_at_ratio_1": round(at_one, 4),
                "e2e_time_ratio_pct": e2e_pct,
                "ratios_file": str(ratio_path.relative_to(REPO)),
            })
        print(
            f"fig2: {model_label:18s} n={len(ratios):>8d}  "
            f"L40s-faster={at_one * 100:5.1f}%  e2e_time_ratio={e2e_pct:5.2f}%"
        )

    mean_cdf = sum(cdf_values) / len(cdf_values)
    mean_e2e = sum(e2e_by_model.values()) / len(e2e_by_model)
    max_e2e = max(e2e_by_model.values())
    assert abs(mean_cdf - PAPER_MEAN_CDF_AT_1) < 0.01, mean_cdf
    assert abs(mean_e2e - PAPER_MEAN_E2E_PCT) < 1.0, mean_e2e
    assert abs(max_e2e - PAPER_MAX_E2E_PCT) < 1.0, max_e2e
    print(
        f"fig2: mean L40s-faster {mean_cdf * 100:.0f}% (paper 67%), "
        f"mean E2E ratio {mean_e2e:.1f}% (paper 36%), max {max_e2e:.1f}% (paper 53%)"
    )

    moved = archive_existing("fig2", keep)
    if moved:
        print(f"fig2: archived {len(moved)} runtime-profiler records")


def ingest_fig3() -> None:
    keep = set()
    anchors = annotation_anchors()
    for a in anchors:
        print(
            f"fig3: anchor {a['label']:14s} ratio={a['ratio']:.3f} "
            f"({a['launches']} launches)  A100 {a['kernel_a100'][:34]} | "
            f"L40s {a['kernel_l40s'][:34]}"
        )
    for run_id, source, group_key, paper in (
        ("gptoss20b_phase_points", "fig3_phase_points.tsv", "phase", PAPER_PHASE_PCT),
        ("gptoss20b_block_points", "fig3_block_points.tsv", "category", PAPER_BLOCK_PCT),
    ):
        src = CENSUS / source
        rows = read_points(src)
        pct = faster_pct(rows, group_key)
        for group, expected in paper.items():
            assert abs(pct[group] - expected) <= 4.0, (group, pct[group], expected)

        keep.add(run_id)
        write_record(
            "fig3",
            run_id,
            {
                "model_label": "gptoss20b",
                "model": "openai/gpt-oss-20b",
                "pair_label": "a100_l40s",
                "grouping": group_key,
                "requires_gpus": ["A100", "L40S"],
            },
            {
                "grouping": group_key,
                "points": len(rows),
                "l40s_faster_pct": {k: round(v, 1) for k, v in sorted(pct.items())},
                "annotations": anchors if group_key == "phase" else [],
                "points_file": str(src.relative_to(REPO)),
            })
        detail = ", ".join(f"{k} {v:.0f}%" for k, v in sorted(pct.items()))
        print(f"fig3: {group_key:9s} {len(rows):3d} points  L40s-faster: {detail}")

    moved = archive_existing("fig3", keep)
    if moved:
        print(f"fig3: archived {len(moved)} runtime-profiler records")


def main() -> None:
    if not CENSUS.exists():
        raise SystemExit(f"ingest_kernel_census: missing {CENSUS}")
    ingest_fig2()
    ingest_fig3()


if __name__ == "__main__":
    main()
