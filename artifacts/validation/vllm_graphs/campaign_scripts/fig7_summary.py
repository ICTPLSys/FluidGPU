#!/usr/bin/env python3
"""Summarize fig7 sweep results: TPOT vs rate per row + 50ms SLO crossing."""
import glob
import json
import os
import re
import sys

OUT = sys.argv[1] if len(sys.argv) > 1 else (
    os.path.expanduser("~/workspace/FluidGPU/artifacts/validation/vllm_graphs/fig7_gt")
)
SLO_MS = 50.0

rows = {}
for path in glob.glob(os.path.join(OUT, "*_r*.json")):
    m = re.match(r"(.+)_r([\d.]+)$", os.path.basename(path)[:-5])
    if not m:
        continue
    row, rate = m.group(1), float(m.group(2))
    d = json.load(open(path))
    if isinstance(d, list):
        d = d[-1]
    rows.setdefault(row, {})[rate] = d

for row in sorted(rows):
    pts = sorted(rows[row].items())
    print(f"\n== {row} ==")
    print(f"{'rate':>6} {'medTPOT':>8} {'p99TPOT':>8} {'medTTFT':>9} "
          f"{'out tok/s':>9} {'req/s got':>9} {'fail':>4}")
    crossing = None
    prev = None
    for rate, d in pts:
        med = d.get("median_tpot_ms")
        print(f"{rate:>6} {med:>8.1f} {d.get('p99_tpot_ms', 0):>8.1f} "
              f"{d.get('median_ttft_ms', 0):>9.0f} "
              f"{d.get('output_throughput', 0):>9.1f} "
              f"{d.get('request_throughput', 0):>9.2f} "
              f"{d.get('failed', d.get('num_failed', 0)) or 0:>4}")
        if crossing is None and med is not None and med > SLO_MS:
            if prev is not None:
                r0, m0 = prev
                # linear interpolation between the straddling points
                crossing = r0 + (rate - r0) * (SLO_MS - m0) / (med - m0)
            else:
                crossing = rate
        if med is not None:
            prev = (rate, med)
    if crossing is not None:
        print(f"  -> median-TPOT {SLO_MS:.0f}ms SLO crossing ~= {crossing:.1f} req/s")
    else:
        print(f"  -> median TPOT stayed under {SLO_MS:.0f}ms across the sweep")
