"""Decoupled-engine driver: replaces the stock PD HTTP proxy.

Per request we mint a request_id that CARRIES both peer addresses:
    <base>___prefill_addr_<ip>:<kv_port_P>___decode_addr_<ip>:<kv_port_D>___
P2pNcclConnector.parse_request_id() reads them (p2p_nccl_connector.py:503):
  - prefill side  -> "___decode_addr_(.*):(\\d+)"    : where to SEND the KV
  - decode  side  -> "___prefill_addr_(.*):(\\d+)___" : where to RECV the KV from
So P streams the KV straight to D over NCCL P2P (GPU->GPU). No proxy involved.

Flow per request: POST P (max_tokens=1, prefill) -> then POST D (max_tokens=OUT).
Running many of these concurrently naturally pipelines prefill(L40S) || decode(A100).

Reports steady-state throughput (25%->75% completion window) to match how the
decoupled prototype and the PD ramp-amortized numbers were measured.
"""
import argparse, asyncio, time, sys
import os
import aiohttp

P_URL = "http://127.0.0.1:8600/v1/completions"
D_URL = "http://127.0.0.1:8700/v1/completions"
IP = "192.168.22.87"
KV_P, KV_D = 21001, 22001


def make_rid(i: int) -> str:
    return f"req{i}___prefill_addr_{IP}:{KV_P}___decode_addr_{IP}:{KV_D}___"


async def one_request(sess, i, prompt, out_len, model, done, lock, t):
    rid = make_rid(i)
    hdr = {"X-Request-Id": rid}
    # 1) PREFILL on L40S: 1 token; its KV is pushed to D over NCCL
    body = {"model": model, "prompt": prompt, "max_tokens": 1,
            "temperature": 0.0, "stream": False}
    async with sess.post(P_URL, json=body, headers=hdr) as r:
        if r.status != 200:
            return f"prefill HTTP {r.status}: {(await r.text())[:120]}"
        await r.json()
    # 2) DECODE on A100: pulls that KV, decodes OUT tokens without re-prefilling
    body = {"model": model, "prompt": prompt, "max_tokens": out_len,
            "temperature": 0.0, "stream": False, "ignore_eos": True}
    async with sess.post(D_URL, json=body, headers=hdr) as r:
        if r.status != 200:
            return f"decode HTTP {r.status}: {(await r.text())[:120]}"
        js = await r.json()
    n = len(js.get("choices", [{}])[0].get("text", ""))
    async with lock:
        done[0] += 1
        c = done[0]
        if t["n"] and c == max(1, t["n"] // 4): t["t25"] = (time.perf_counter(), c)
        if t["n"] and c == (3 * t["n"]) // 4: t["t75"] = (time.perf_counter(), c)
        if c % 32 == 0:
            print(f"  completed={c}/{t['n']}", flush=True)
    return None


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-prompts", type=int, default=256)
    ap.add_argument("--input-len", type=int, default=4096)
    ap.add_argument("--output-len", type=int, default=384)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--model", default=os.path.expanduser("~/.cache/fluidgpu/models/openai--gpt-oss-20b"))
    a = ap.parse_args()

    # synthetic prompts at the target input length (token-ish: 1 word ~ 1 token)
    prompt = " ".join(["hello"] * a.input_len)
    done = [0]; lock = asyncio.Lock()
    t = {"n": a.num_prompts, "t25": None, "t75": None}
    sem = asyncio.Semaphore(a.concurrency)
    errs = []

    async def guarded(i):
        async with sem:
            e = await one_request(sess, i, prompt, a.output_len, a.model, done, lock, t)
            if e: errs.append(e)

    timeout = aiohttp.ClientTimeout(total=3600)
    conn = aiohttp.TCPConnector(limit=a.concurrency * 2)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as sess:
        t0 = time.perf_counter()
        await asyncio.gather(*[guarded(i) for i in range(a.num_prompts)])
        wall = time.perf_counter() - t0

    ok = a.num_prompts - len(errs)
    e2e = ok * a.output_len / wall
    steady = None
    if t["t25"] and t["t75"] and t["t75"][0] > t["t25"][0]:
        (at, ac), (bt, bc) = t["t25"], t["t75"]
        steady = (bc - ac) * a.output_len / (bt - at)
    print(f"\n=== decoupled engine (P2pNccl, no proxy) : {ok}/{a.num_prompts} ok ===")
    print(f"wall={wall:.1f}s  e2e={e2e:.0f} out_tok/s  steady={steady and round(steady) or 'n/a'} out_tok/s")
    print(f"compare: PD(Mooncake) e2e@256=1827, steady(np768)=1892 ; ceiling 2023")
    if errs:
        print(f"errors ({len(errs)}), first 3:")
        for e in errs[:3]: print("  ", e)


if __name__ == "__main__":
    asyncio.run(main())
