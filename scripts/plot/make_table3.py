"""This script reproduces Table III (cost efficiency, Perf/$).
Input: artifacts/logs/fig6/*/{summary.json,meta.json} and configs/gpu_prices.yaml.
Output:
  - artifacts/tables/table3.csv / table3.md: paper Table III layout
    (rows = policies, columns = GPU pairs, values normalized to Homo. (left)).
  - artifacts/tables/table3_raw.csv: per-run throughput/cost for debugging.

Trend: FluidGPU achieves the highest Perf/$ on each heterogeneous pair.
"""
from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

from common import load_run_records, strip_repeat_suffix, value_or_none


PAIR_GPUS = {
    "a100_l40s": ("a100", "l40s"),
    "h100_rtxpro6000": ("h100", "rtx_pro_6000"),
    "b200_h100": ("b200", "h100"),
}
PAIR_HEADERS = {
    "a100_l40s": "A100+L40s",
    "h100_rtxpro6000": "H100+RTX Pro 6000",
    "b200_h100": "B200+H100",
}
POLICY_COST_MODE = {
    "homo-left": "left",
    "homo-right": "right",
    # Request Dist. runs one replica per GPU, so it rents both, like the
    # disaggregation policies.
    "request-dist": "both",
    "pd": "both",
    "af": "both",
    "fluidgpu": "both",
}
POLICY_ROW_LABELS = {
    "homo-left": "Homo. (left)",
    "homo-right": "Homo. (right)",
    "request-dist": "Request Dist.",
    "pd": "PD Dis.",
    "af": "AF Dis.",
    "fluidgpu": "FluidGPU",
}
# Row order of the paper's Table III, which does NOT list Request Dist. (it is a
# Fig.6-only baseline). The policy still has a cost mode so that its per-run
# Perf/$ lands in table3_raw.csv.
POLICY_ORDER = ["homo-left", "homo-right", "pd", "af", "fluidgpu"]


def policy_cost(pair: str, policy: str, prices: dict) -> float:
    mode = POLICY_COST_MODE[policy]
    left, right = PAIR_GPUS[pair]
    if mode == "left":
        return float(prices[left])
    if mode == "right":
        return float(prices[right])
    return float(prices[left]) + float(prices[right])


def perf_per_dollar(throughput: float, cost: float) -> float:
    assert cost > 0.0, f"non-positive GPU-hour cost: {cost}"
    return float(throughput) / float(cost)


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
    out_dir = Path("artifacts/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    prices = yaml.safe_load(Path("configs/gpu_prices.yaml").read_text())

    records = load_run_records("fig6")

    # (model, pair, policy) -> list of throughput across repeats. The unit is
    # per-model (tokens/s for LLM/SSM/MLLM, requests/s for diffusion); it
    # cancels in the per-model normalization below, so both can be mixed.
    throughputs: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    raw_rows: list[dict[str, object]] = []
    for record in records:
        pair = record.payload.get("pair_label")
        model = record.payload.get("model_label")
        policy = infer_policy(record)
        throughput = value_or_none(record.summary.get("throughput_tok_s"))
        if throughput is None:
            throughput = value_or_none(record.summary.get("throughput_req_s"))
        if pair is None or model is None or policy is None or throughput is None:
            continue
        assert pair in PAIR_GPUS, f"unknown gpu_pair label: {pair}"
        assert policy in POLICY_COST_MODE, f"unknown policy: {policy}"
        throughputs[(str(model), str(pair), policy)].append(float(throughput))
        cost = policy_cost(str(pair), policy, prices)
        raw_rows.append(
            {
                "run_id": record.run_id,
                "gpu_pair": pair,
                "model": model,
                "policy": policy,
                "throughput": float(throughput),
                "pair_cost": cost,
                "perf_per_cost": perf_per_dollar(float(throughput), cost),
            }
        )

    if not throughputs:
        raise SystemExit(
            "table3: missing fig6 summaries; "
            "run python3 scripts/preprocess/ingest_vllm_exp.py first"
        )

    # Repeat-average, then Perf/$ per (model, pair, policy).
    per_model_ppc = {
        (model, pair, policy): perf_per_dollar(
            statistics.mean(values), policy_cost(pair, policy, prices)
        )
        for (model, pair, policy), values in throughputs.items()
    }

    # Normalize WITHIN each model against that model's homo-left baseline
    # before averaging across models. Averaging raw throughput first would be
    # wrong: policies cover different model sets (the paper marks PD and AF
    # inapplicable to the diffusion model, AF also to the SSM), and per-model
    # throughputs differ by an order of magnitude, so a policy that happens to
    # skip a low-throughput model would score higher for that reason alone.
    # Normalizing first also cancels the per-model unit.
    by_pair_policy: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (model, pair, policy), value in per_model_ppc.items():
        baseline = per_model_ppc.get((model, pair, "homo-left"))
        if baseline is None or baseline <= 0.0:
            continue
        by_pair_policy[(pair, policy)].append(value / baseline)
    ppc = {key: statistics.mean(values) for key, values in by_pair_policy.items()}

    # Paper column order; drop pairs with no homo-left baseline.
    pairs_in_order = [
        pair
        for pair in PAIR_HEADERS
        if (pair, "homo-left") in ppc and ppc[(pair, "homo-left")] > 0.0
    ]
    if not pairs_in_order:
        raise SystemExit("table3: no homo-left baseline found; cannot normalize columns")

    pivot: dict[tuple[str, str], float] = {}
    for pair in pairs_in_order:
        baseline = ppc[(pair, "homo-left")]
        for policy in POLICY_ORDER:
            if (pair, policy) in ppc:
                pivot[(pair, policy)] = ppc[(pair, policy)] / baseline

    def cell(pair: str, policy: str) -> str:
        val = pivot.get((pair, policy))
        if val is None:
            return "-"
        return f"{val:.2f}"

    csv_path = out_dir / "table3.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["policy"] + [PAIR_HEADERS[p] for p in pairs_in_order])
        for policy in POLICY_ORDER:
            writer.writerow(
                [POLICY_ROW_LABELS[policy]] + [cell(p, policy) for p in pairs_in_order]
            )

    md_lines = [
        "| Policy | " + " | ".join(PAIR_HEADERS[p] for p in pairs_in_order) + " |",
        "|---" + "|---:" * len(pairs_in_order) + "|",
    ]
    for policy in POLICY_ORDER:
        cells = " | ".join(cell(p, policy) for p in pairs_in_order)
        md_lines.append(f"| {POLICY_ROW_LABELS[policy]} | {cells} |")
    (out_dir / "table3.md").write_text("\n".join(md_lines) + "\n")

    raw_csv_path = out_dir / "table3_raw.csv"
    with raw_csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(raw_rows[0]))
        writer.writeheader()
        writer.writerows(raw_rows)

    print(f"table3: wrote {csv_path}")


if __name__ == "__main__":
    main()
