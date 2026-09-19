"""Summarize thread_cpu_sampler.py output: per-thread CPU% over the run tail.

The generation phase is the last part of the run; report per-thread CPU%
(ticks/HZ per wall second) for the whole run and for the last N seconds.
Usage: python analyze_threads.py FILE.threads.csv [tail_seconds]
"""
import collections
import os
import sys

HZ = os.sysconf("SC_CLK_TCK")


def main():
    path = sys.argv[1]
    tail_s = float(sys.argv[2]) if len(sys.argv) > 2 else 60.0
    rows = []
    with open(path) as f:
        next(f)
        for line in f:
            p = line.rstrip("\n").split(",")
            if len(p) != 6:
                continue
            ts, pid, role, tid, comm, d = p
            rows.append((float(ts), int(pid), role, int(tid), comm, int(d)))
    if not rows:
        print("no samples")
        return
    t0, t1 = rows[0][0], rows[-1][0]
    for label, lo in (("WHOLE RUN", t0), (f"LAST {tail_s:.0f}s", t1 - tail_s)):
        span = t1 - lo
        if span <= 0:
            continue
        agg = collections.defaultdict(int)
        proc = collections.defaultdict(int)
        for ts, pid, role, tid, comm, d in rows:
            if ts < lo:
                continue
            agg[(role, tid, comm)] += d
            proc[role] += d
        print(f"\n== {label} ({span:.0f}s wall, {t0:.0f}..{t1:.0f}) ==")
        for role, ticks in sorted(proc.items(), key=lambda kv: -kv[1]):
            print(f"  {role:8s} process total: {100*ticks/HZ/span:6.1f}% CPU")
        print("  top threads:")
        for (role, tid, comm), ticks in sorted(agg.items(), key=lambda kv: -kv[1])[:14]:
            pct = 100 * ticks / HZ / span
            if pct < 1:
                break
            print(f"    {role:8s} tid={tid:<8d} {comm:24s} {pct:6.1f}% CPU")


if __name__ == "__main__":
    main()
