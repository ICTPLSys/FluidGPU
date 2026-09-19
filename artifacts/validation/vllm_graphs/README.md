# vLLM-Runtime Experiment Archive (2026-07)

Master index of the fig6/fig7/fig9 measurement experiments run with the
`fluidgpu_torch.vllm_integration` runtime (vLLM 0.18, pure-Python monkey-patch,
greedy token-match parity as hard gate). Full narrative, every negative result,
and all corrections: `docs/AE_MACHINE_RUNBOOK.md` §16–§48. Canonical result
tables and reviewer disclosures: `docs/EXPERIMENTS.md` → "vLLM-Runtime Experiment
Results".

## Environment constants

- Machine: gpuserver86 — GPU0/GPU1 = A100 80G PCIe, GPU2 = L40S; always
  `CUDA_DEVICE_ORDER=PCI_BUS_ID` (FASTEST_FIRST reorders devices and corrupts
  device-indexed probes — runbook §42 correction).
- 2-card configurations: `CUDA_VISIBLE_DEVICES=1,2` (A100 + L40S). 3-card adds GPU0.
- RDMA: Mooncake connector, P side mlx5_5, D side mlx5_2, bootstrap 8998.
  Measured link ~20 GB/s; A100↔L40S staged copy 23.06 GB/s; A100↔A100 P2P
  22.6 GB/s aggregate.
- Models (offline HF cache, `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`):
  - GT = openai/gpt-oss-20b (vocab 201088), random-token 4096 in / 684 out
  - LM = meta-llama/Llama-3.1-8B-Instruct (vocab 128256), calibrated
    1920 in / 1024 out (v3; earlier 1536/1024 = v2 archived)
- Engines: `--no-enable-prefix-caching`, batched-token budget 16384, bf16
  weights; the GT final mirror uses `--max-num-seqs 128` on both engines,
  while the earlier campaigns and LM use 32 unless noted.
- Metrics: `e2e` = ok×out_len/wall; `steady` = completions×out_len over the
  25%→75% completion window. Report both-e2e or both-steady, never mixed
  (§41 addendum). Same-day pairing only (~2.5% day drift).

## Experiment index

| dir | question | headline result | runbook |
|---|---|---|---|
| `fig6_spliced/`, `homo_align/`, … | GT offline ladder (homo/AF/FG evolution) | lever A 1282 = 1.48× AF; homo 1004/1007 | §30–§38 |
| `gt_decoupled/` | GT 4096/684: every policy ms32, same client conc64 | FG hybrid φ=0.18 2213/2365 vs PD 2030/2160 = 1.09×/1.10×; RD 2021 and AF 1235.6 measured | §48 |
| `fig9_3card/` | 2×A100+1×L40S: FG-3 form A vs PD-3 | 2949 vs 1990 steady = 1.48× (paper 1.42× bracketed) | §42–§43 |
| `fig7_gt_v2/` | GT online latency curves, 50 ms TPOT crossings | FG detoxified 2.1→5.4 req/s > AF 4.5; PD >12 | §44 |
| `lm_fig6/` | LM v2 experiment @1536/1024 + shape grid + vocab-bug repair | FG(decoupled) ≡ PD 1648/1633; homo +9.8%/−7.6% | §45 |
| `lm_kernel_layout/` | per-kernel census, transfer/overlap measurement, layout frontier | hybrid = P/D + φ overflow → 2174 steady (+32% over P/D); balanced f=0.35 +4% | §46 |
| `fig6_v3/` | LM re-calibration @1920/1024 (homo A100 pinned at paper) | homo A100 1293 (+0.23%); hybrid φ=0.22 → 1818/2049 = 1.32× PD steady | §47 |
| `campaign_scripts/` | every driving script from all experiments (flat archive) | — | — |

Key single-file assets inside `lm_kernel_layout/`:
- `census_a100.csv`, `census_l40s.csv` — per-kernel-group prefill/decode µs
  (repo profiler `profile_llm_tasks.py`, prompt 1536, 5 repeats)
- `b_leverA_trace.steps`, `b_homoW_trace.steps` — per-step walls
  (`FLUIDGPU_STEP_TRACE=1`; ~7% sync tax, diagnostic only)
- `b_leverA_trace.stage`, `b_af_trace.stage` — deferred-sync stage timing
  (`FLUIDGPU_STAGE_TIMING=1`, ~free): remote-FFN rate + hop GB/s
- `llama_plan_dp.json` — repo DP planner output on the census (alternating
  attn/mlp plan; documented as misleading — serial objective, no per-hop
  latency, no overlap credit)

## The measurement client (`campaign_scripts/rdma_driver.py`)

One async client covers every served-configuration topology (distinct seed-1234
random-token-id prompts, exact input length, ignore_eos, per-request
transport retry, vocab bound derived from the model's `config.json` —
llama 128000, gpt-oss 199000):

| mode | flags | used for |
|---|---|---|
| P/D decoupled | (default; P :8100 → D :8200) | FG decoupled rows |
| pure proxy | `--local-prefill-frac 1.0` + `FG_D_URL=<proxy>` | PD-steady same-client rows |
| hybrid overflow | `--local-prefill-frac φ --local-url <P-engine>` | FG hybrid rows (φ share whole to P, zero KV transfer) |
| dual-homo | two processes, each frac=1.0 against its own engine | routing baseline |
| dual decoders | `--d-url2` | fig9 PD-3 |
| whole-request third card | `--local-url <homo instance>` | fig9 FG-3 form A |

Env overrides: `FG_P_URL`, `FG_D_URL`, `FG_BOOTSTRAP`.

## Hard-won operational rules (details in runbook)

1. Readiness through a proxy must require a complete `"choices"` JSON body,
   never just HTTP 200 (proxy answers 200 + truncated payload while engines
   load) — §45.
2. Raw-token-id clients must derive the prompt id range from the model's
   vocab_size — §45.
3. `CUDA_DEVICE_ORDER=PCI_BUS_ID` before any device-indexed probe — §42.
4. Experiment cleanups kill engines/proxies but not parent bash scripts — kill
   parents by PID; beware `pkill -f` self-matching literal script paths.
5. Co-located instances need `--kv-cache-memory-gb` pinning (memory-profiler
   race) — §38.
6. steady vs e2e must be paired symmetrically; mixing manufactures ~10%
   artifacts (§41 addendum).
