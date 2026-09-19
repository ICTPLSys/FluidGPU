# Request Dist. baseline — driver outputs (Fig.6, camera-ready sixth series)

Produced by `bash scripts/vllm_exp/run_fig6_request_distribution.sh {gt,lm}` on 2026-07-22
(gpuserver86, GPU1 = A100 80G PCIe, GPU2 = L40S). Two independent vLLM 0.18
replicas, one per GPU, `--max-num-seqs 32` and `--no-enable-prefix-caching`
each; no KV transfer and no cross-GPU kernel execution. The client
(`scripts/vllm_exp/reqdist_driver.py`) is the load balancer and emits the
same prompt stream as the FluidGPU bar of each model (distinct random token
ids, seed 1234), 256 requests at total concurrency 64.

| file | what it is |
|---|---|
| `single_a100.out`, `single_l40s.out` | one replica alone, 128 requests at conc 32 — the rates the paired run is reported as a fraction of |
| `reqdist_least-outstanding.out` | dispatch to the replica with fewest in-flight requests — the best policy on both models, and the Fig.6 row |
| `reqdist_round-robin.out` | alternate replicas |
| `reqdist_static.out` | fixed 50/50 split (`--static-frac`) |

Shapes at measurement time: GT = gpt-oss-20b at 4096 in / 384 out; LM =
Llama-3.1-8B-Instruct at 1920 in / 1024 out. GT later moved to 4096/684, so
the archived GT measurement is no longer used directly as its Fig.6 bar.

| model | LO | RR | static | singles | LO / sum | measured FG |
|---|---:|---:|---:|---|---:|---:|
| GT | 1645 | 1636 | 1638 | 952 + 837 | 92% | 1852 |
| LM | 2015 | 1511 | 1509 | 1289 + 759 | 98% | 1818 |

All values are e2e output tokens/s, `ok x out_len / wall`. The numbers land in
`{gt_decoupled,fig6_v3}/results.csv` as the `reqdist_lo` and
`serve_single_*` rows; the plotted Fig.6 bars are FG-anchored rescales of the
paper's own Request Dist. values (GT re-anchored to the 4096/684 FG result).

The engine-side `vllm serve` logs are not tracked; they stay under
`artifacts/validation/vllm_graphs/reruns/reqdist_*/` on the evaluation machine.
