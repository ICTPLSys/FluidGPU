"""Materialize the vLLM runtime experiment results as standard figure log records.

The vLLM integration experiments (docs/AE_MACHINE_RUNBOOK.md §16-§47) are the
canonical execution mode for the paper's end-to-end figures (Fig. 6/7/8) on
this artifact. Its curated, machine-readable result tables live under
artifacts/validation/vllm_graphs/. This script converts them into
artifacts/logs/{fig6,fig7,fig9}/ run records so the standard plotting scripts
(scripts/plot/plot_fig{6,7,8}.py and make_table3.py) consume the experiment
numbers unchanged.

Sources (metric symmetry: every fig6/fig9 value is the e2e reading, the one
the figures and Table III are built from; steady is carried alongside in the
CSVs for reference and is never mixed with it):
  - fig6 GT rows:  artifacts/validation/vllm_graphs/gt_decoupled/results.csv
  - fig6 LM rows:  artifacts/validation/vllm_graphs/fig6_v3/results.csv
  - fig7 curves:   artifacts/validation/vllm_graphs/fig7_gt_v2/<arm>_r<rate>.json
  - fig9 arms:     artifacts/validation/vllm_graphs/fig9_3card/results.csv
  - extra fig6 rows (model families measured from logs, e.g. MLLM/SSM/
    diffusion): artifacts/validation/vllm_graphs/extra_fig6_rows.csv

On first run, pre-existing batch-1-runtime records under artifacts/logs/<fig>/
are moved to artifacts/logs/<fig>_batch1_runtime_archive/. Re-running the
ingestion afterwards is idempotent (imported records are rewritten in place).
"""
from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
GRAPHS = REPO / "artifacts" / "validation" / "vllm_graphs"
LOGS = REPO / "artifacts" / "logs"
KIND = "imported-vllm-campaign"
PAPER_REFERENCE_KIND = "imported-paper-reference"

PAIR_PAYLOAD = {"pair_label": "a100_l40s", "gpu_a": "A100", "gpu_b": "L40S"}

FIG6_SOURCES = [
    {
        "model_label": "gptoss20b",
        "model": "openai/gpt-oss-20b",
        "csv": GRAPHS / "gt_decoupled" / "results.csv",
        "workload": "random-token 4096 in / 684 out (2026-07-24 final mirror)",
        "out_len": 684,
    },
    {
        "model_label": "llama31_8b",
        "model": "meta-llama/Llama-3.1-8B-Instruct",
        "csv": GRAPHS / "fig6_v3" / "results.csv",
        "workload": "calibrated 1920 in / 1024 out (runbook §47)",
        "out_len": 1024,
    },
]

FIG7_ARM_POLICY = {
    "af": "af",
    "fg_leverA": "fluidgpu",
    "homo_a100": "homo-left",
    "homo_l40s": "homo-right",
    "pd": "pd",
    # fg_ubr is the µbatch-gate negative control (runbook §44), not a paper row.
}

FIG8_ARM_POLICY = {
    "pd3_paper_config": "pd",
    "fg3_formA_phi035": "fluidgpu",
    # No AF-3 arm exists: TP2-based AF was bounded out at the resource level
    # (runbook §43); plot_fig9 marks the missing policy with a red X.
}

EXTRA_ROWS_TEMPLATE = """\
# Drop-in fig6 rows measured from logs outside the curated experiments
# (e.g. MLLM / SSM / diffusion model families served with stock vLLM).
# Lines starting with '#' are ignored. Columns:
#   model_label: fig6 model axis key (qwen25vl7b | mamba_codestral7b | sd35_medium | ...)
#   model:       HF id of the model
#   pair_label:  a100_l40s (or another pair key from plot_fig6.PAIR_LABELS)
#   policy:      homo-left | homo-right | request-dist | pd | af | fluidgpu
#   tok_s:       output tokens/s (leave empty for diffusion)
#   req_s:       requests/s (diffusion: plotted as images/min = req_s*60;
#                derived as tok_s/out_len when empty and out_len is given)
#   out_len:     fixed output tokens per request (used to derive req_s)
#   date:        measurement date (YYYY-MM-DD)
#   source:      the measurement log this number came from, or the paper
#   notes:       free text (workload shape, protocol)
model_label,model,pair_label,policy,tok_s,req_s,out_len,date,source,notes
"""


def read_csv(path: Path) -> list[dict[str, str]]:
    lines = [l for l in path.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    return list(csv.DictReader(lines))


def fnum(row: dict[str, str], key: str) -> float | None:
    raw = (row.get(key) or "").strip()
    try:
        return float(raw)
    except ValueError:
        return None


def archive_legacy(fig: str) -> None:
    root = LOGS / fig
    if not root.exists():
        return
    legacy = []
    for run_dir in root.iterdir():
        meta_path = run_dir / "meta.json"
        if not run_dir.is_dir():
            continue
        kind = None
        if meta_path.exists():
            kind = json.loads(meta_path.read_text()).get("kind")
        # Records written by ingest_paper_reference_figs.py are not legacy:
        # that script runs AFTER this one and rewrites them from scratch, so
        # archiving them here would both lose nothing and (once an archive
        # exists) abort the whole import.
        if kind not in (KIND, PAPER_REFERENCE_KIND):
            legacy.append(run_dir)
    if not legacy:
        return
    archive = LOGS / f"{fig}_batch1_runtime_archive"
    if archive.exists():
        raise SystemExit(
            f"{fig}: legacy records present but {archive} already exists; resolve manually"
        )
    shutil.move(str(root), str(archive))
    print(f"{fig}: archived {len(legacy)} batch-1-runtime records -> {archive}")


def write_record(fig: str, run_id: str, payload: dict, summary: dict, source: str) -> None:
    run_dir = LOGS / fig / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    meta = {
        "figure": fig,
        "run_id": run_id,
        "kind": KIND,
        "command": "python3 scripts/preprocess/ingest_vllm_exp.py",
        "source": source,
        "payload": payload,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def fig6_policy_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    """Map experiment CSV rows to the six paper policies (camera-ready Fig.6)."""
    by_name = {row["row"].strip(): row for row in rows}
    picked: dict[str, dict[str, str]] = {}
    for name, row in by_name.items():
        if name == "homo_a100":
            picked["homo-left"] = row
        elif name == "homo_l40s":
            picked["homo-right"] = row
        elif name == "af" or name == "af_sync":
            picked["af"] = row
        elif name == "reqdist_lo":
            picked["request-dist"] = row
        elif "fg row" in (row.get("notes") or "").lower():
            picked["fluidgpu"] = row
    pd_row = by_name.get("pd_steady_sameclient") or by_name.get("pd_stock_c64")
    if pd_row is not None:
        picked["pd"] = pd_row
    return picked


def ingest_fig6() -> None:
    archive_legacy("fig6")
    for spec in FIG6_SOURCES:
        rows = read_csv(spec["csv"])
        picked = fig6_policy_rows(rows)
        missing = [
            p for p in ("homo-left", "homo-right", "request-dist", "pd", "af", "fluidgpu")
            if p not in picked
        ]
        if missing:
            raise SystemExit(f"fig6 {spec['model_label']}: no experiment row for {missing}")
        for policy, row in picked.items():
            e2e = fnum(row, "e2e_tok_s")
            if e2e is None:
                raise SystemExit(f"fig6 {spec['model_label']} {policy}: missing e2e_tok_s")
            run_id = f"{spec['model_label']}_a100_l40s_{policy}_vllm_r0"
            payload = {
                "id": run_id,
                "model_label": spec["model_label"],
                "model": spec["model"],
                "policy": policy,
                **PAIR_PAYLOAD,
            }
            summary = {
                "status": "ok",
                "throughput_tok_s": e2e,
                # Every request in these arms generates exactly out_len tokens
                # (ignore_eos), so req/s = tok/s / out_len is exact.
                "throughput_req_s": e2e / spec["out_len"],
                "output_len": spec["out_len"],
                "steady_tok_s": fnum(row, "steady_tok_s"),
                "unit": "output tokens/s (e2e = ok x out_len / wall)",
                "paper_value": fnum(row, "paper"),
                "protocol": (row.get("protocol") or "").strip(),
                "workload": spec["workload"],
                "source_row": row["row"].strip(),
            }
            source = f"{spec['csv'].relative_to(REPO)}#{row['row'].strip()}"
            write_record("fig6", run_id, payload, summary, source)
    print("fig6: ingested 2 models x 6 policies")


def ingest_extra_fig6() -> int:
    path = GRAPHS / "extra_fig6_rows.csv"
    if not path.exists():
        path.write_text(EXTRA_ROWS_TEMPLATE)
        print(f"fig6 extra: created template {path.relative_to(REPO)} (no rows yet)")
        return 0
    count = 0
    for row in read_csv(path):
        model_label = (row.get("model_label") or "").strip()
        policy = (row.get("policy") or "").strip()
        pair = (row.get("pair_label") or "").strip() or "a100_l40s"
        if not model_label or not policy:
            continue
        tok_s = fnum(row, "tok_s")
        req_s = fnum(row, "req_s")
        out_len = fnum(row, "out_len")
        if req_s is None and tok_s is not None and out_len:
            req_s = tok_s / out_len
        if tok_s is None and req_s is None:
            raise SystemExit(f"extra fig6 row {model_label}/{policy}: needs tok_s or req_s")
        run_id = f"{model_label}_{pair}_{policy}_vllm_r0"
        payload = {
            "id": run_id,
            "model_label": model_label,
            "model": (row.get("model") or "").strip(),
            "policy": policy,
            "pair_label": pair,
        }
        summary = {
            "status": "ok",
            "throughput_tok_s": tok_s,
            "throughput_req_s": req_s,
            "unit": "output tokens/s" if tok_s is not None else "requests/s",
        }
        source = f"{path.relative_to(REPO)}#{model_label}/{policy}"
        write_record("fig6", run_id, payload, summary, source)
        count += 1
    print(f"fig6 extra: ingested {count} rows")
    return count


def ingest_fig7() -> None:
    archive_legacy("fig7")
    src_dir = GRAPHS / "fig7_gt_v2"
    count = 0
    for path in sorted(src_dir.glob("*_r*.json")):
        m = re.fullmatch(r"(.+)_r(\d+)", path.stem)
        if not m or m.group(1) not in FIG7_ARM_POLICY:
            continue
        arm, rate = m.group(1), int(m.group(2))
        data = json.loads(path.read_text())
        completed = float(data.get("completed") or 0)
        out_tokens = float(data.get("total_output_tokens") or 0)
        ttft = data.get("mean_ttft_ms")
        tpot = data.get("mean_tpot_ms")
        if completed <= 0 or out_tokens <= 0 or ttft is None or tpot is None:
            raise SystemExit(f"fig7 {path.name}: missing completed/output/ttft/tpot")
        avg_len = out_tokens / completed
        # Paper's normalized latency (request e2e latency / output tokens),
        # reconstructed from serving stats as ttft/L + tpot*(L-1)/L (runbook §44).
        normalized = float(ttft) / avg_len + float(tpot) * (avg_len - 1.0) / avg_len
        policy = FIG7_ARM_POLICY[arm]
        run_id = f"{arm}_r{rate}_vllm"
        payload = {
            "id": run_id,
            "policy": policy,
            "rps": rate,
            "model": "openai/gpt-oss-20b",
            **PAIR_PAYLOAD,
        }
        summary = {
            "status": "ok",
            "request_rate": rate,
            "completed": completed,
            "normalized_latency_ms_per_token": normalized,
            "mean_ttft_ms": ttft,
            "mean_tpot_ms": tpot,
            "median_tpot_ms": data.get("median_tpot_ms"),
            "output_throughput": data.get("output_throughput"),
            "avg_output_tokens": avg_len,
        }
        stamp = str(data.get("date") or "")  # vllm bench serve format: YYYYMMDD-HHMMSS
        created = (
            f"{stamp[0:4]}-{stamp[4:6]}-{stamp[6:8]}"
            if re.match(r"\d{8}-", stamp)
            else "2026-07-16"
        )
        source = str(path.relative_to(REPO))
        write_record("fig7", run_id, payload, summary, source)
        count += 1
    print(f"fig7: ingested {count} (arm, rate) points")


def ingest_fig9() -> None:
    archive_legacy("fig9")
    rows = read_csv(GRAPHS / "fig9_3card" / "results.csv")
    by_arm = {row["arm"].strip(): row for row in rows}
    for arm, policy in FIG8_ARM_POLICY.items():
        row = by_arm.get(arm)
        if row is None:
            raise SystemExit(f"fig9: experiment row {arm} not found")
        e2e = fnum(row, "e2e_tok_s")
        run_id = f"two_a100_one_l40s_{policy}_vllm"
        payload = {
            "id": run_id,
            "cluster_label": "two_a100_one_l40s_gptoss",
            "policy": policy,
            "model": "openai/gpt-oss-20b",
        }
        summary = {
            "status": "ok",
            "throughput_tok_s": e2e,
            "steady_tok_s": fnum(row, "steady_tok_s"),
            "unit": "output tokens/s (e2e)",
            "conc": (row.get("conc") or "").strip(),
            "source_row": arm,
        }
        source = f"{(GRAPHS / 'fig9_3card' / 'results.csv').relative_to(REPO)}#{arm}"
        write_record("fig9", run_id, payload, summary, source)
    print(f"fig9: ingested {len(FIG8_ARM_POLICY)} arms")


def main() -> None:
    ingest_fig6()
    ingest_extra_fig6()
    ingest_fig7()
    ingest_fig9()


if __name__ == "__main__":
    main()
