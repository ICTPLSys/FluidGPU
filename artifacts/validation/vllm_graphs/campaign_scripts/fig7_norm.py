#!/usr/bin/env python3
"""fig7 v2 dual-metric summary: median TPOT (§16 continuity) AND the paper's
normalized latency (e2e / output tokens, approximated as median_e2el /
mean_output_len — aggregates only; distribution is tight, med out 130).
Prints 50ms-SLO crossings under both metrics per row."""
import glob
import json
import os
import re
import sys

OUT = sys.argv[1]
SLO = 50.0

rows = {}
for path in glob.glob(os.path.join(OUT, "*_r*.json")):
    m = re.match(r"(.+)_r([\d.]+)$", os.path.basename(path)[:-5])
    if not m:
        continue
    d = json.load(open(path))
    if isinstance(d, list):
        d = d[-1]
    rows.setdefault(m.group(1), {})[float(m.group(2))] = d


def crossing(pts, key):
    prev = None
    for rate, v in pts:
        if v is None:
            continue
        if v > SLO:
            if prev is None:
                return f"<{rate:g}"
            pr, pv = prev
            return f"{pr + (rate - pr) * (SLO - pv) / (v - pv):.1f}"
        prev = (rate, v)
    return f">{pts[-1][0]:g}"


for row in sorted(rows):
    pts = sorted(rows[row].items())
    tpot = []
    norm = []
    print(f"\n== {row} ==")
    print(f"{'rate':>5} {'medTPOT':>8} {'normLat':>8} {'medTTFT':>9} {'out tok/s':>9} {'fail':>4}")
    for rate, d in pts:
        mt = d.get("median_tpot_ms")
        comp = d.get("completed") or 1
        mean_out = (d.get("total_output_tokens") or 0) / comp
        # saved aggregates lack e2el; reconstruct per-request identity
        # e2e = ttft + (L-1)*tpot  =>  normalized = e2e/L  (approx: medians+mean L)
        ttft = d.get("median_ttft_ms")
        nl = None
        if mt is not None and ttft is not None and mean_out > 1:
            nl = (ttft + (mean_out - 1) * mt) / mean_out
        tpot.append((rate, mt))
        norm.append((rate, nl))
        print(f"{rate:>5g} {mt or 0:>8.1f} {nl or 0:>8.1f} {d.get('median_ttft_ms', 0):>9.0f} "
              f"{d.get('output_throughput', 0):>9.1f} "
              f"{(d.get('failed') or d.get('num_failed') or 0):>4}")
    print(f"  50ms crossing: TPOT-metric = {crossing(tpot, 't')} req/s | "
          f"normalized-latency (paper) = {crossing(norm, 'n')} req/s")
