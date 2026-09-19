# vLLM Runtime Experiments

These scripts rerun the canonical Fig. 6/7/8 rows of the vLLM-integration
execution mode (`fluidgpu_torch.vllm_integration`, vLLM 0.18, pure-Python
monkey-patch, greedy token-match parity as a hard gate). They are adapted from
the measurement scripts archived in
`artifacts/validation/vllm_graphs/campaign_scripts/`, with every
machine-specific value lifted into `exp_env.sh`.

Step-by-step instructions, including the expected values and the fairness
rules that make a comparison meaningful, are in `docs/EXPERIMENTS.md` §T3.

## Configuration

All four scripts source `exp_env.sh`, which resolves the repository root
from its own location and reads these overrides (defaults are the authors'
evaluation host):

| Variable | Default | Meaning |
|---|---|---|
| `FLUIDGPU_PYTHON` | `python` on PATH | Interpreter with vLLM 0.18 |
| `FLUIDGPU_VLLM` | `vllm` on PATH | vLLM CLI |
| `FLUIDGPU_MODEL_GT` | `~/.cache/fluidgpu/models/openai--gpt-oss-20b` | GT weights |
| `FLUIDGPU_MODEL_LM` | `~/.cache/fluidgpu/models/meta-llama--Llama-3.1-8B-Instruct` | LM weights |
| `FLUIDGPU_GPU_P` | `2` | Prefill side, the weaker GPU (L40S here) |
| `FLUIDGPU_GPU_D` | `1` | Decode side, the stronger GPU (A100 here) |
| `FLUIDGPU_GPU_X` | `0` | Third GPU, Fig.9 only |
| `FLUIDGPU_HCA_P` / `FLUIDGPU_HCA_D` | `mlx5_5` / `mlx5_2` | IB HCA per side |
| `FLUIDGPU_RERUN_OUT` | `artifacts/validation/vllm_graphs/reruns/` | Output root |

GPU indices are read under `CUDA_DEVICE_ORDER=PCI_BUS_ID`, which the scripts
export. Check the resolution without running anything:

```bash
bash -c 'source scripts/vllm_exp/exp_env.sh; echo "$PY | $VB | $MODEL_GT | GPUs $GPU_P/$GPU_D/$GPU_X | $HCA_P/$HCA_D"'
```

## Experiments

| script | reruns | canonical results it should reproduce |
|---|---|---|
| `run_fig6.sh` | Fig.6 both LLM columns in one run: GT then LM; the T3 entry point | `gt_decoupled/results.csv`, `fig6_v3/results.csv` |
| `run_fig6_gptoss20b.sh` | GT (gpt-oss-20b), 4096/684; every policy ms32, same client conc64; FG row = hybrid φ=0.18 (analytic φ*), 3 reps | `gt_decoupled/results.csv` (FG 2213 vs PD 2030 = 1.09×) |
| `run_fig6_gptoss20b_steady.sh` | GT stock-PD at ms32/conc64 with the identical client and 25%–75% steady window used by FG | `gt_decoupled/results.csv#pd_steady_sameclient` |
| `run_fig6_llama31.sh` | LM (Llama-3.1-8B) full v3 ladder @1920/1024: offline configurations, FG decoupled, hybrid phi sweep, stock PD, PD-steady, dual-homo control | `fig6_v3/results.csv` (hybrid phi=0.22 1818 vs PD 1501 = 1.21×) |
| `run_fig6_request_distribution.sh [gt\|lm]` | Fig.6 `Request Dist.`: one independent vLLM replica per GPU (ms32 each), single-replica controls, then the routing-policy sweep (least-outstanding / round-robin / static) at the model's Fig.6 shape and total conc 64 | `{gt_decoupled,fig6_v3}/results.csv` rows `reqdist_lo` (GT 2021 at 4096/684, LM 2015) and `serve_single_{a100,l40s}` |
| `run_fig7_gptoss20b_online.sh` | GT online Poisson sweep, rates 2-12, all five rows + parity gate | `fig7_gt_v2/crossings.csv` (FG 5.4 / AF 4.5 / PD >12 req/s at 50 ms TPOT) |
| `run_fig9.sh` | 2xA100+1xL40S: FG-2 reference, PD-3 (paper config), FG-3 form A phi sweep | `fig9_3card/results.csv` (FG-3 2606 vs PD-3 1894 = 1.38×) |

| `run_fig10_pipeline.sh` | Fig.10 pipeline ablation inside the vLLM engine: w/o Pipe. / Pipe. / Pipe.+Prio. at the Fig.6 GT workload, so the bars sit on the same footing as Fig.6's rather than on the batch-1 torch runtime's | `fig10_pipeline/results.csv` |

## Operational notes

- The scripts kill every GPU process between configurations (`cleanup`); run them only
  on an otherwise-idle box.
- Mooncake/RDMA rows need the point-to-point IB fabrics up (subnet manager +
  `nvidia_peermem` after a reboot — `docs/AE_MACHINE_RUNBOOK.md` §3).
- Runs use the offline HF cache (`HF_HUB_OFFLINE=1`), set by the scripts.
- `run_fig6.sh` plus `run_fig9.sh` make up the ~2-3 h T3 budget; the fig7 sweep is separate;
  `run_fig6_request_distribution.sh` is the shortest at about 15 minutes per model, then
  `run_fig6_gptoss20b.sh` at about an hour.
- Day-to-day drift on identical configs is ~2.5%, so headline pairings must be
  same-day, same-client and on the same metric.

After a rerun, compare against the canonical CSVs. Only if you intend to
update the shipped figures: edit the CSVs, then run
`python3 scripts/preprocess/ingest_vllm_exp.py` and the plot scripts
(`scripts/plot/plot_fig{6,7,8}.py`, `make_table3.py`).
