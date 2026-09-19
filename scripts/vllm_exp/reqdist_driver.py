"""Request distribution (Fig.6 `Request Dist.` baseline): one client, two replicas.

Each GPU of the heterogeneous pair runs an INDEPENDENT full vLLM replica (no KV
transfer, no cross-GPU kernel execution); this client is the load balancer and
routes every complete request to exactly one replica. That is the paper's
Request Dist. baseline, camera-ready Fig.6.

Why a single client instead of two `rdma_driver.py --local-prefill-frac 1.0`
processes (how the LM `dual_homo` control was measured, runbook §47): a static
per-replica request count is an *oracle* split -- it needs the capability ratio
measured up front, and it leaves the fast replica idle at the tail whenever the
guess is off. One client with a real routing policy needs no oracle and emits
the exact same prompt stream as the FG bar (same seed, same distinct random
token ids), so the two bars are the same workload on the same two GPUs, which
is what the paper's Fig.6 comparison claims.

Policies (`--policy`, the paper's "best-performing vLLM load-balancing policy
among those evaluated"):

  least-outstanding  dispatch to the replica with the fewest in-flight requests
                     (ties -> fewest assigned). Self-balancing on a
                     heterogeneous pair: the faster replica drains first and
                     therefore receives more requests. This is the policy the
                     vLLM production router calls `leastreq`.
  round-robin        alternate replicas; the reference policy, and the one that
                     leaves the slow replica queueing at the tail.
  static             Bresenham-spread fixed fraction to replica B
                     (`--static-frac`), i.e. the oracle split.

Reports e2e AND the 25%->75% steady window, same accounting as
`rdma_driver.py`, plus the realized per-replica request counts.
"""
import argparse
import asyncio
import json
import os
import random
import time

import aiohttp


def spread(i: int, frac: float) -> bool:
    """Bresenham-style pick: exactly `frac` of indices, EVENLY SPREAD.

    Same helper as rdma_driver.is_local -- a front-loaded clump would skew both
    the routing balance and the steady-state window.
    """
    return frac > 0.0 and int((i + 1) * frac) > int(i * frac)


def build_prompts(model: str, num_prompts: int, input_len: int) -> list[list[int]]:
    """Bit-identical prompt stream to rdma_driver.py (seed 1234, raw token ids)."""
    vocab_high = 199_000
    try:
        with open(os.path.join(model, "config.json")) as f:
            vs = json.load(f).get("vocab_size")
        if vs:
            vocab_high = min(vocab_high, int(vs) - 256)
    except Exception:
        pass
    rng = random.Random(1234)
    prompts = [[rng.randrange(256, vocab_high) for _ in range(input_len)]
               for _ in range(num_prompts)]
    assert len({tuple(p) for p in prompts}) == num_prompts, "prompts must be distinct"
    return prompts


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls", default=os.environ.get(
        "FG_REQDIST_URLS",
        "http://127.0.0.1:8300/v1/completions,http://127.0.0.1:8400/v1/completions"),
        help="comma-separated replica completion endpoints (A100 first)")
    ap.add_argument("--labels", default="A100,L40S")
    ap.add_argument("--policy", default="least-outstanding",
                    choices=["least-outstanding", "round-robin", "static"])
    ap.add_argument("--static-frac", type=float, default=0.5,
                    help="policy=static: fraction of requests sent to replica B")
    ap.add_argument("--num-prompts", type=int, default=256)
    ap.add_argument("--input-len", type=int, default=4096)
    ap.add_argument("--output-len", type=int, default=384)
    ap.add_argument("--concurrency", type=int, default=64,
                    help="TOTAL in-flight requests across both replicas")
    ap.add_argument("--model", default=os.environ.get("FLUIDGPU_MODEL_GT",
                                                      "openai/gpt-oss-20b"))
    a = ap.parse_args()

    urls = [u for u in a.urls.split(",") if u]
    labels = (a.labels.split(",") + [f"r{i}" for i in range(len(urls))])[:len(urls)]
    # Two replicas = the Request Dist. baseline; one = the single-replica
    # control that the baseline is reported as a fraction of.
    assert 1 <= len(urls) <= 2, "expected one or two replica endpoints"

    prompts = build_prompts(a.model, a.num_prompts, a.input_len)
    n = len(urls)
    inflight = [0] * n
    assigned = [0] * n
    finished = [0] * n
    lat: list[list[float]] = [[] for _ in range(n)]
    done = [0]
    lock = asyncio.Lock()
    t = {"n": a.num_prompts, "t25": None, "t75": None}
    errs: list[str] = []

    def pick(i: int) -> int:
        if a.policy == "round-robin":
            return i % n
        if a.policy == "static":
            return 1 if spread(i, a.static_frac) else 0
        # least-outstanding: fewest in-flight, ties -> fewest assigned so far
        return min(range(n), key=lambda k: (inflight[k], assigned[k], k))

    sem = asyncio.Semaphore(a.concurrency)
    timeout = aiohttp.ClientTimeout(total=3600)
    conn = aiohttp.TCPConnector(limit=a.concurrency * 3)

    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as sess:
        async def one(i: int, salt: str) -> str | None:
            # Routing decision is taken at dispatch time (after the global
            # semaphore admits the request), so least-outstanding sees the
            # queue state it is actually balancing.
            async with lock:
                k = pick(i)
                inflight[k] += 1
                assigned[k] += 1
            body = {"model": a.model, "prompt": prompts[i], "max_tokens": a.output_len,
                    "temperature": 0.0, "stream": False, "ignore_eos": True}
            hdr = {"X-Request-Id": f"req-{i}{salt}"}
            t0 = time.perf_counter()
            try:
                async with sess.post(urls[k], json=body, headers=hdr) as r:
                    if r.status != 200:
                        return f"{labels[k]} HTTP {r.status}: {(await r.text())[:140]}"
                    await r.json()
            finally:
                async with lock:
                    inflight[k] -= 1
            async with lock:
                finished[k] += 1
                lat[k].append(time.perf_counter() - t0)
                done[0] += 1
                c = done[0]
                if t["t25"] is None and c >= max(1, t["n"] // 4):
                    t["t25"] = (time.perf_counter(), c)
                if t["t75"] is None and c >= (3 * t["n"]) // 4:
                    t["t75"] = (time.perf_counter(), c)
                if c % 32 == 0:
                    print(f"  completed={c}/{t['n']}  "
                          f"{'/'.join(f'{labels[j]}:{finished[j]}' for j in range(n))}",
                          flush=True)
            return None

        async def guarded(i: int) -> None:
            async with sem:
                e = None
                for attempt in ("", "-r"):
                    try:
                        e = await one(i, attempt)
                        break
                    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                        e = f"transport {type(exc).__name__}: {exc}"
                        if not attempt:
                            await asyncio.sleep(0.5)
                if e:
                    errs.append(e)

        t0 = time.perf_counter()
        await asyncio.gather(*[guarded(i) for i in range(a.num_prompts)])
        wall = time.perf_counter() - t0

    ok = a.num_prompts - len(errs)
    e2e = ok * a.output_len / wall if wall > 0 else 0.0
    steady = None
    if t["t25"] and t["t75"] and t["t75"][0] > t["t25"][0]:
        (at, ac), (bt, bc) = t["t25"], t["t75"]
        steady = (bc - ac) * a.output_len / (bt - at)

    print(f"\n=== request distribution ({a.policy}): {ok}/{a.num_prompts} ok ===")
    for k in range(n):
        mean_lat = sum(lat[k]) / len(lat[k]) if lat[k] else float("nan")
        print(f"  {labels[k]:>6}: assigned={assigned[k]:4d} ok={finished[k]:4d} "
              f"mean_lat={mean_lat:6.1f}s  share={finished[k] / max(1, ok):.3f}  "
              f"out_tok_s={finished[k] * a.output_len / wall:7.1f}")
    print(f"wall={wall:.1f}s  e2e={e2e:.0f} out_tok/s  "
          f"steady={steady and round(steady) or 'n/a'} out_tok/s")
    if errs:
        print(f"errors ({len(errs)}), first 3:")
        for e in errs[:3]:
            print("  ", e)


if __name__ == "__main__":
    asyncio.run(main())
