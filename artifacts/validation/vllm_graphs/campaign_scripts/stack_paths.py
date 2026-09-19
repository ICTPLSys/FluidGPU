"""Print the dominant FULL call paths for a given thread in a py-spy speedscope file.

Groups samples by their full stack (collapsed), prints the top paths by time.
Usage: python stack_paths.py FILE.json THREAD_SUBSTR [--top N] [--leaf SUBSTR]
"""
import json
import sys
from collections import Counter


def label(fr):
    name = fr.get("name", "?")
    f = fr.get("file", "")
    for cut in ("/site-packages/", "/fluidgpu_runtime/", "/lib/python3.12/"):
        if cut in f:
            f = f.split(cut, 1)[1]
            break
    return f"{name}[{f}:{fr.get('line',0)}]"


def main():
    path, tsub = sys.argv[1], sys.argv[2]
    top_n = int(sys.argv[sys.argv.index("--top") + 1]) if "--top" in sys.argv else 8
    leaf = None
    if "--leaf" in sys.argv:
        leaf = sys.argv[sys.argv.index("--leaf") + 1].lower()
    data = json.load(open(path))
    frames = data["shared"]["frames"]
    labs = [label(fr) for fr in frames]
    for prof in data["profiles"]:
        if tsub not in prof.get("name", ""):
            continue
        weights = prof.get("weights") or [1.0] * len(prof["samples"])
        agg = Counter()
        tot = 0.0
        for s, w in zip(prof["samples"], weights):
            if leaf and (not s or leaf not in labs[s[-1]].lower()):
                continue
            agg[tuple(s)] += w
            tot += w
        print(f"\n==== {prof['name']}  (matched {tot:.1f}s) ====")
        for stack, w in agg.most_common(top_n):
            print(f"-- {w:.1f}s ({100*w/max(tot,1e-9):.0f}%):")
            for idx in stack[-14:]:
                print(f"     {labs[idx]}")


if __name__ == "__main__":
    main()
