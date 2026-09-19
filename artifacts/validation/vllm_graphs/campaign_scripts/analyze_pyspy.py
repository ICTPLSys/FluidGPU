"""Aggregate a py-spy speedscope JSON (record --subprocesses) into a readable report.

Per profile (= one process thread): classify each sample as load / gen / other
using stack markers, then print top self-time and total-time frames for the
gen+other samples. Marker-based (not time-based) so late-spawned threads and
per-profile clock offsets don't matter.

Usage: python analyze_pyspy.py FILE.speedscope.json [--top N] [--min-pct 1.0]
"""
import json
import sys
from collections import Counter, defaultdict

LOAD_MARKERS = (
    "load_model", "safetensors", "from_pretrained", "_dynamo", "_inductor",
    "compile_fx", "determine_available_memory", "profile_run", "capture_model",
    "graph_capture", "prepare_moe_fp4", "process_weights_after_loading",
    "wait_for_engine_startup", "initialize", "load_weights",
)
GEN_MARKERS = (
    "execute_model", "run_busy_loop", "core_busy_loop", "busy_loop", "generate",
    "_run_engine", "detokenize", "output_processor", "process_output",
    "ubatch", "dbo", "_overlapped_remote", "step",
)


def frame_label(fr):
    name = fr.get("name", "?")
    f = fr.get("file", "")
    line = fr.get("line", 0)
    for cut in ("/site-packages/", "/fluidgpu_runtime/", "/lib/python3.12/"):
        if cut in f:
            f = f.split(cut, 1)[1]
            break
    return f"{name} ({f}:{line})"


def main():
    path = sys.argv[1]
    top_n = 15
    min_pct = 0.8
    if "--top" in sys.argv:
        top_n = int(sys.argv[sys.argv.index("--top") + 1])
    if "--min-pct" in sys.argv:
        min_pct = float(sys.argv[sys.argv.index("--min-pct") + 1])

    data = json.load(open(path))
    frames = data["shared"]["frames"]
    labels = [frame_label(fr) for fr in frames]
    lowers = [l.lower() for l in labels]

    frame_phase = []
    for l in lowers:
        if any(m in l for m in LOAD_MARKERS):
            frame_phase.append("load")
        elif any(m in l for m in GEN_MARKERS):
            frame_phase.append("gen")
        else:
            frame_phase.append("")

    grand = defaultdict(float)
    print(f"== {path}")
    for prof in data["profiles"]:
        name = prof.get("name", "?")
        samples = prof["samples"]
        weights = prof.get("weights") or [1.0] * len(samples)
        total_t = sum(weights)
        if total_t < 0.5:
            continue
        phase_t = Counter()
        self_t = defaultdict(lambda: defaultdict(float))
        tot_t = defaultdict(lambda: defaultdict(float))
        for s, w in zip(samples, weights):
            ph = ""
            for idx in s:
                p = frame_phase[idx]
                if p == "load":
                    ph = "load"
                    break
                if p == "gen":
                    ph = "gen"
            ph = ph or "other"
            phase_t[ph] += w
            if ph == "load":
                continue
            if s:
                self_t[ph][s[-1]] += w
            for idx in set(s):
                tot_t[ph][idx] += w
        grand[name.split(")")[0] + ")"] += total_t
        print(f"\n---- {name}  total={total_t:.1f}s  "
              + "  ".join(f"{k}={v:.1f}s" for k, v in sorted(phase_t.items())))
        for ph in ("gen", "other"):
            pt = phase_t.get(ph, 0.0)
            if pt < 1.0:
                continue
            print(f"  [{ph}] top self-time:")
            for idx, w in sorted(self_t[ph].items(), key=lambda kv: -kv[1])[:top_n]:
                pct = 100 * w / pt
                if pct < min_pct:
                    break
                print(f"    {w:7.1f}s {pct:5.1f}%  {labels[idx]}")
            print(f"  [{ph}] top total-time (frame anywhere in stack):")
            for idx, w in sorted(tot_t[ph].items(), key=lambda kv: -kv[1])[:top_n]:
                pct = 100 * w / pt
                if pct < min_pct:
                    break
                print(f"    {w:7.1f}s {pct:5.1f}%  {labels[idx]}")

    print("\n== per-process sampled time (s):")
    for k, v in sorted(grand.items(), key=lambda kv: -kv[1]):
        print(f"  {v:9.1f}s  {k}")


if __name__ == "__main__":
    main()
