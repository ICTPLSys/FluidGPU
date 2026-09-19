"""Decoupled engine over RDMA (MooncakeConnector) WITHOUT the stock HTTP proxy.

NCCL P2P is impossible on this A100(sm80)<->L40S(sm89) SYS/no-P2P pair, so we use
the transport that provably works here: Mooncake RDMA (mlx5_5 <-> mlx5_2). The only
thing we strip vs stock PD is the proxy hop: this driver IS the client and talks to
P and D directly, replicating the proxy's handshake:

  1) GET  {bootstrap_addr}/query            -> {dp_rank: {engine_id: ...}}
  2) POST P  kv_transfer_params={do_remote_decode:True,  do_remote_prefill:False,
             transfer_id}                   max_tokens=1        (prefill on L40S)
  3) POST D  kv_transfer_params={do_remote_decode:False, do_remote_prefill:True,
             remote_bootstrap_addr, remote_engine_id, transfer_id}   (decode on A100)

N concurrent such chains => prefill(L40S) || decode(A100) pipeline.
Reports e2e AND steady (25%->75% window) so it is comparable to PD's numbers
(PD Mooncake: e2e@256=1827, steady@768=1892; L40S-prefill ceiling 2023).
"""
import argparse, asyncio, json, os, random, time
import aiohttp

P_URL = os.environ.get("FG_P_URL", "http://127.0.0.1:8100/v1/completions")
D_URL = os.environ.get("FG_D_URL", "http://127.0.0.1:8200/v1/completions")
BOOTSTRAP = os.environ.get("FG_BOOTSTRAP", "http://127.0.0.1:8998")


async def get_prefiller_engine_id(sess):
    """Replicates proxy get_prefiller_info(): GET bootstrap /query -> engine_id."""
    for _ in range(60):
        try:
            async with sess.get(BOOTSTRAP + "/query") as r:
                if r.status == 200:
                    data = await r.json()
                    # {dp_rank: {"engine_id": ...}}
                    first = data[list(data.keys())[0]]
                    return first["engine_id"]
        except Exception:
            pass
        await asyncio.sleep(1)
    raise RuntimeError("bootstrap /query never answered")


def is_local(i: int, frac: float) -> bool:
    """Bresenham-style pick: exactly `frac` of indices, EVENLY SPREAD.

    NOT `(i % 1000) < frac*1000` -- that clusters every local request into the FIRST
    frac*1000 indices, so a 256-prompt run at frac=0.05 would route indices 0..49 =
    19.5% (and 0.25 routed 4/4 in the smoke test). Spreading also matters for
    steady-state: a front-loaded clump would skew the 25%->75% window.
    """
    return frac > 0.0 and int((i + 1) * frac) > int(i * frac)


async def one_request(sess, i, prompt, out_len, model, eid, done, lock, t, local_frac=0.0,
                      salt="", d_url=None, local_url=None):
    rid = f"req-{i}{salt}"
    xfer = f"xfer-{rid}"
    d_url = d_url or D_URL
    local_url = local_url or d_url

    # PREFILL-OFFLOAD (runbook 28): the phase layout leaves the A100 idle ~22% of the
    # time (per-request L40S prefill 163ms vs A100 decode 144ms), and the A100's real
    # prefill rate is only 1.26x off the L40S (17,716 vs 22,319 tok/s). So routing a
    # small fraction of requests' WHOLE prefill to the A100 balances the two cards.
    # Whole-request routing => NO intra-layer hop and no KV transfer at all: D just
    # prefills+decodes locally (omit kv_transfer_params entirely).
    # Balance point ~5.3% analytically -> ideal 2485 vs the phase layout's 2354.
    if is_local(i, local_frac):
        body = {"model": model, "prompt": prompt, "max_tokens": out_len,
                "temperature": 0.0, "stream": False, "ignore_eos": True}
        async with sess.post(local_url, json=body, headers={"X-Request-Id": rid}) as r:
            if r.status != 200:
                return f"local HTTP {r.status}: {(await r.text())[:140]}"
            await r.json()
        async with lock:
            done[0] += 1
            c = done[0]
            if t["t25"] is None and c >= max(1, t["n"] // 4): t["t25"] = (time.perf_counter(), c)
            if t["t75"] is None and c >= (3 * t["n"]) // 4: t["t75"] = (time.perf_counter(), c)
        return None

    # 1) PREFILL on L40S (1 token); KV is pushed to D over RDMA
    body = {"model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0.0,
            "stream": False,
            "kv_transfer_params": {"do_remote_decode": True, "do_remote_prefill": False,
                                   "transfer_id": xfer}}
    hdr = {"X-Request-Id": rid, "X-data-parallel-rank": "0"}
    async with sess.post(P_URL, json=body, headers=hdr) as r:
        if r.status != 200:
            return f"prefill HTTP {r.status}: {(await r.text())[:140]}"
        await r.json()
    # 2) DECODE on A100: pulls that KV; must NOT re-prefill
    body = {"model": model, "prompt": prompt, "max_tokens": out_len, "temperature": 0.0,
            "stream": False, "ignore_eos": True,
            "kv_transfer_params": {"do_remote_decode": False, "do_remote_prefill": True,
                                   "remote_bootstrap_addr": BOOTSTRAP,
                                   "remote_engine_id": eid, "transfer_id": xfer}}
    hdr = {"X-Request-Id": rid}
    async with sess.post(d_url, json=body, headers=hdr) as r:
        if r.status != 200:
            return f"decode HTTP {r.status}: {(await r.text())[:140]}"
        await r.json()
    async with lock:
        done[0] += 1
        c = done[0]
        if t["t25"] is None and c >= max(1, t["n"] // 4): t["t25"] = (time.perf_counter(), c)
        if t["t75"] is None and c >= (3 * t["n"]) // 4: t["t75"] = (time.perf_counter(), c)
        if c % 32 == 0:
            print(f"  completed={c}/{t['n']}", flush=True)
    return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=256)
    ap.add_argument("--input-len", type=int, default=4096)
    ap.add_argument("--output-len", type=int, default=384)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--model", default=os.environ.get("FLUIDGPU_MODEL_GT", "openai/gpt-oss-20b"))
    ap.add_argument("--d-url2", default="",
                    help="second decode instance URL; requests round-robin "
                    "between D_URL and this (paper-fig9 PD-3: one L40S "
                    "prefiller feeding two independent A100 decoders)")
    ap.add_argument("--local-url", default="",
                    help="target for is_local plain-completion requests "
                    "(fig9 FG-3 form A: an independent full-stack instance "
                    "on the third GPU takes a phi fraction of requests)")
    ap.add_argument("--local-prefill-frac", type=float, default=0.0,
                    help="fraction of requests whose WHOLE prefill runs on the A100 "
                         "(D instance) instead of the L40S -- balances the idle A100. "
                         "0 = pure phase layout (today's 1960).")
    a = ap.parse_args()

    # DISTINCT random-token prompts, one per request -- NOT " ".join(["hello"]*N).
    # A single reused prompt makes all concurrent requests share a hidden state, so the
    # MoE router picks the SAME top-4 of 32 experts for every token and decode FFN never
    # becomes weight-bound (measured, runbook 28: A100 marlin decode 4.132 degenerate vs
    # 5.557 diverse = 1.34x artificially cheap; L40S 2.49x). That flatters FG's decode
    # (A100) and inflates the A100's slack, which is exactly what the phi sweep measures.
    # It also made FG's workload EASIER than the PD baseline's, which uses
    # `--dataset-name random` = distinct random token ids. Same distribution now.
    # Pass raw TOKEN IDS (a list of ints is a valid /v1/completions `prompt`), not text:
    # "tok12345" tokenizes to ~3 tokens, so a 4096-word string is ~12.2k tokens and the
    # input length would not match PD's 4096. Token ids give EXACTLY input_len, which is
    # also how vLLM's own `--dataset-name random` builds its prompts.
    # Id range must stay inside the model's vocab (llama-3.1 = 128256; the old
    # hardcoded 199_000 was gpt-oss-sized and made every llama request 400:
    # "Token id N is out of vocabulary"). Read vocab_size from the model dir;
    # cap at 199_000 so the gpt-oss id stream is bit-identical to prior runs.
    vocab_high = 199_000
    try:
        with open(os.path.join(a.model, "config.json")) as _f:
            _vs = json.load(_f).get("vocab_size")
        if _vs:
            vocab_high = min(vocab_high, int(_vs) - 256)
    except Exception:
        pass
    rng = random.Random(1234)
    prompts = [[rng.randrange(256, vocab_high) for _ in range(a.input_len)]
               for _ in range(a.num_prompts)]
    assert len({tuple(p) for p in prompts}) == a.num_prompts, "prompts must be distinct"
    done = [0]; lock = asyncio.Lock()
    t = {"n": a.num_prompts, "t25": None, "t75": None}
    sem = asyncio.Semaphore(a.concurrency)
    errs = []

    timeout = aiohttp.ClientTimeout(total=3600)
    conn = aiohttp.TCPConnector(limit=a.concurrency * 3)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as sess:
        if a.local_prefill_frac >= 1.0:
            # every request goes as a plain completion to D_URL (e.g. the stock
            # PD proxy) — no P handshake, no bootstrap query needed
            eid = "n/a"
        else:
            eid = await get_prefiller_engine_id(sess)
        print(f"prefiller engine_id={eid}", flush=True)

        async def guarded(i):
            async with sem:
                # Transport-level errors (ServerDisconnectedError on a recycled
                # keep-alive conn, ClientOSError) used to escape and kill the
                # whole gather (runbook §35's harness bug). Retry once with a
                # fresh request/transfer id; count as an error after that.
                e = None
                du = a.d_url2 if (a.d_url2 and i % 2 == 1) else None
                for attempt in ("", "-r"):
                    try:
                        e = await one_request(sess, i, prompts[i], a.output_len,
                                              a.model, eid, done, lock, t,
                                              a.local_prefill_frac, salt=attempt,
                                              d_url=du, local_url=a.local_url or None)
                        break
                    except (aiohttp.ClientError, asyncio.TimeoutError, OSError) as exc:
                        e = f"transport {type(exc).__name__}: {exc}"
                        if not attempt:
                            await asyncio.sleep(0.5)
                if e: errs.append(e)

        t0 = time.perf_counter()
        await asyncio.gather(*[guarded(i) for i in range(a.num_prompts)])
        wall = time.perf_counter() - t0

    ok = a.num_prompts - len(errs)
    e2e = ok * a.output_len / wall if wall > 0 else 0
    steady = None
    if t["t25"] and t["t75"] and t["t75"][0] > t["t25"][0]:
        (at, ac), (bt, bc) = t["t25"], t["t75"]
        steady = (bc - ac) * a.output_len / (bt - at)
    nloc = sum(1 for i in range(a.num_prompts) if is_local(i, a.local_prefill_frac))
    print(f"\n=== decoupled engine (Mooncake RDMA, NO proxy): {ok}/{a.num_prompts} ok ===")
    print(f"local-prefill-frac={a.local_prefill_frac} -> {nloc}/{a.num_prompts} requests "
          f"prefilled on the A100, {a.num_prompts - nloc} on the L40S")
    print(f"wall={wall:.1f}s  e2e={e2e:.0f} out_tok/s  steady={steady and round(steady) or 'n/a'} out_tok/s")
    print(f"vs stock PD (same RDMA, WITH proxy): e2e@256=1827, steady@768=1892 ; ceiling 2023")
    if errs:
        print(f"errors ({len(errs)}), first 3:")
        for e in errs[:3]: print("  ", e)


if __name__ == "__main__":
    asyncio.run(main())
