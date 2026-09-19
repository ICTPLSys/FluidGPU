# Running The Experiments

Everything needed to reproduce the paper's results: the representative path,
a per-task walkthrough, and the map from each figure and table to the scripts
that produce it.

- Installation and RDMA bring-up: [INSTALL.md](INSTALL.md)
- Facts specific to the authors' machine: [AE_MACHINE_RUNBOOK.md](AE_MACHINE_RUNBOOK.md)

Everything below assumes you are at the repository root with the environment
active, and — on any machine where CUDA is not on the default PATH —

```bash
export PATH=/usr/local/cuda/bin:$PATH
```

---

# Part 1 — Representative reproduction path

> **Hardware prerequisite:** the end-to-end tasks require the heterogeneous
> GPU and RDMA setup described in [INSTALL.md](INSTALL.md). No reference-machine
> login credentials are distributed in this public repository.

**What this is.** The shortest path to reproduce FluidGPU's central results.
The reference hardware (A100 + L40S + RDMA) is not on a free cloud tier.

**Reference machine.** 2× A100 80 GB PCIe + 1× L40S, 200 Gbps InfiniBand,
Ubuntu 22.04.5. After setting up your machine, activate the environment:

```bash
cd "$HOME/FluidGPU"
source "$HOME/fluidgpu-ae/bin/activate"
export PATH=/usr/local/cuda/bin:$PATH
export CUDA_DEVICE_ORDER=PCI_BUS_ID
```

---

## The representative path (about 1 hour with warm caches; allow up to 3 hours)

`T0 → T1 → T2 → T2b → T3 → T4`. Run top to bottom; each step prints a PASS
line or a headline number to check.

| Task | Command | Time | Check (✅ = reproduced) |
|---|---|---:|---|
| T0 | `bash scripts/verify_env.sh` | 1 min | `verify_env: PASS` |
| T1 | `bash scripts/ptx_memory_probe.sh --execute required` | 1 min | `runtime_range_check.matched: true` |
| T2 | `bash scripts/demo_planner.sh` | 1 min | `gurobi_status=OPTIMAL` |
| T2b | `bash scripts/verify_parity.sh` | 15 min | token match **1.0**, cosine **≥0.999** (ref 0.9998) — *the correctness gate* |
| T3 | `bash scripts/vllm_exp/run_fig6.sh` | ~1–3 h | Fig.6 over stock PD: LM **1.21×**, GT **1.09×** |
| T3 | `bash scripts/vllm_exp/run_fig9.sh` | (same budget) | Fig.9 over PD-3: **1.38×** |
| T4 | `python3 scripts/preprocess/ingest_kernel_census.py && python3 scripts/preprocess/ingest_vllm_exp.py && python3 scripts/preprocess/ingest_paper_reference_figs.py` | 2 min | census assertions pass (no `AssertionError`) |
| T4 | `for f in 2 3 6 7 8 9 10 11 12a 12b; do python3 scripts/plot/plot_fig$f.py; done && python3 scripts/plot/make_table3.py` | 3 min | all figures + `table3.{csv,md}` regenerate |

**Total ≈ 1–3 hours, depending on caches and hardware.** T0–T2b establish the runtime is correct; T3 freshly
measures the two positive end-to-end results; T4 turns measured records into
every figure and Table III.

Optional smoke tests before T1:
`PYTHONPATH=$PWD/fluidgpu_runtime python3 -m pytest fluidgpu_runtime -q`
(`127 passed`), `python3 scripts/plot/plot_all.py --check` (11× `import ok`),
`bash scripts/check_repo_clean.sh` (`PASS`).

### What T3 demonstrates
The full parity gate runs *inside* the experiment, so the throughput numbers come
from a placement proven numerically identical to the single-GPU baseline. Every
policy is fixed at ms32 and measured with the identical client at the same
concurrency and measurement window. Outputs from new runs are written under
`$FLUIDGPU_RERUN_OUT`; the canonical records shipped with the artifact remain
unchanged.

---

## Reproduce with no GPU (5 minutes)

Every figure and Table III regenerate from the shipped measurement records:

```bash
python3 scripts/preprocess/ingest_kernel_census.py        # asserts the paper's Fig.2/3 numbers
python3 scripts/preprocess/ingest_vllm_exp.py
python3 scripts/preprocess/ingest_paper_reference_figs.py
for f in 2 3 6 7 8 9 10 11 12a 12b; do python3 scripts/plot/plot_fig$f.py; done
python3 scripts/plot/make_table3.py
```

Regenerated PNGs are **byte-identical** to the shipped ones on an unchanged
checkout — the quickest confirmation the analysis path is intact.

---

# Part 2 — Per-task walkthrough

## What you need

| Tier | Hardware | What it unlocks |
|---|---|---|
| A | Any machine, no GPU | Unit tests, T2 planner, T4 ingestion + all plots and Table III |
| B | One NVIDIA GPU | T1 kernel profiling and PTX analysis (Fig.2, Fig.3) |
| C | Two heterogeneous GPUs + RDMA between them | The parity gate, Fig.10/11/12a, and the Fig.6/7 experiments |
| D | Three GPUs (2 strong + 1 weak) | Fig.9 |

The reference pair is A100 80GB PCIe + L40S over 200 Gbps InfiniBand. Nothing
requires that exact pair, but absolute throughputs are hardware-specific.

If you only have tier A, the artifact is still fully exercisable end-to-end:
every figure and table regenerates from the shipped measurement records in
about five minutes (§T4 below).

---

## T0 — Smoke tests (5 minutes, tier A)

Run these first; if any fails, later steps will fail more confusingly.

```bash
export PATH=/usr/local/cuda/bin:$PATH   # verify_env.sh reads nvcc; on many installs CUDA is not on PATH
bash scripts/verify_env.sh
PYTHONPATH=$PWD/fluidgpu_runtime python3 -m pytest fluidgpu_runtime -q
python3 scripts/plot/plot_all.py --check
bash scripts/check_repo_clean.sh
```

Expected: `verify_env: PASS`, `127 passed`, eleven `check: import ok` lines,
`check_repo_clean: PASS`.

`verify_env.sh` pins exact versions (Python 3.12.13, CUDA 12.8,
Gurobi 13.0.1) and checks for one side of a supported heterogeneous pair. A
version mismatch there is cosmetic — it does not stop any experiment — but
read the failure before ignoring it.

## T1 — Kernel profiling and PTX analysis (20 minutes, tier B)

Produces the PTX instrumentation of Listing 1 and the runtime's own per-kernel
census.

Fig.2 and Fig.3 themselves render from the shipped A100-vs-L40s kernel census
(`artifacts/validation/kernel_census/`, ingested in §T4) rather than from the
runs below; the two censuses cover different slices of each workload and are
not interchangeable.

```bash
bash scripts/ptx_memory_probe.sh --execute required

PYTHONPATH=$PWD/fluidgpu_runtime python3 \
  fluidgpu_runtime/examples/fluidgpu_torch/profile_llm_kernels.py \
  --model meta-llama/Llama-3.1-8B-Instruct --device-id 0 \
  --output artifacts/logs/fig2/<run_id>/profile.csv

bash scripts/repro.sh --fig 2
bash scripts/repro.sh --fig 3
```

| Output | Meaning |
|---|---|
| `artifacts/validation/ptx_memory_probe/<ts>/report.json` | `runtime_range_check.matched: true` — the injected PTX bounds match the observed accesses |
| `artifacts/logs/fig2/<run_id>/profile.csv` | Per-kernel time on the profiled device |
| `artifacts/logs/fig3/<run_id>/` | Refined kernel-group split |

The probe also writes `instrumented.ptx` and `memory_accesses.csv`; when
`ptxas`/`nvcc` are present it compiles and runs the instrumented PTX rather
than only emitting it.

## T2 — Policy planning (5 minutes, tier A)

Both planners are exercised here. The DP planner is the artifact's default and
needs no license; the MILP planner uses the size-limited license bundled with
`pip install gurobipy` (2000 variables), which covers every solve the artifact
performs.

```bash
bash scripts/demo_planner.sh          # exact MILP solve, 300 kernels / 2 GPUs
bash scripts/repro.sh --fig 12b       # MILP scalability sweep
```

Expected from `demo_planner.sh`: a `milp_exact_plan ... gurobi_status=OPTIMAL`
line and `artifacts/validation/planner/<ts>/summary.json`. Scale it with
`DEMO_KERNELS=1500` if you have a full Gurobi license.

Fig.12b sweeps kernels × GPUs. Cells above ~2000 binary variables raise
`GurobiError: Model too large for size-limited license`; the run records the
failure and continues. On the reference machine three cells solve
(100×2 = 38 ms, 200×2 = 110 ms, 100×3 = 115 ms) and twelve do not.

To regenerate a runtime profile from measured per-kernel-group costs, run the
task analyzer once per GPU and then plan:

```bash
PYTHONPATH=$PWD/fluidgpu_runtime python3 \
  fluidgpu_runtime/examples/fluidgpu_torch/profile_llm_tasks.py \
  --model Qwen/Qwen2.5-1.5B --prompt-len 512 --decode-steps 32 \
  --output rank0_tasks.csv         # repeat on the second GPU -> rank1_tasks.csv

PYTHONPATH=$PWD/fluidgpu_runtime python3 \
  fluidgpu_runtime/examples/fluidgpu_torch/schedule_llm_layers.py \
  --rank0-csv rank0_tasks.csv --rank1-csv rank1_tasks.csv \
  --model Qwen/Qwen2.5-1.5B --hidden-size 1536 --num-kv-heads 2 --head-dim 128 \
  --solver milp --output profiles/qwen25_1p5b_milp_plan.json
```

Drop `--solver milp` for the DP solver; the two are cross-checked to the same
optimum. Pass `--comm-bw-gbps 1.5 --prefill-comm-bw-gbps 129` to price the
decode and prefill hand-offs separately — they differ by ~143× on this fabric,
and a single bandwidth suppresses profitable prefill splits.

## T2b — Numerical parity gate (15 minutes, tier C)

The load-bearing correctness claim, in one command:

```bash
bash scripts/verify_parity.sh
```

It builds a single-GPU baseline and the two-rank disaggregated + CUDA-graph
candidate, then hard-asserts token match = 1.0 and logits cosine ≥ 0.999.
Reference result: Qwen2.5-1.5B on A100+L40S over RDMA, 16/16 tokens,
cosine 0.9998, in `artifacts/validation/parity/Qwen2.5-1.5B/parity.txt`.
Set `MODEL` / `PROFILE` to gate a different model.

Run this before trusting any performance number: the experiments below enforce
the same parity gate internally, and a failure here means the placement is not
computing the same thing as the baseline.

## T3 — The Fig.6/7/9 experiments (tier C/D)

These are the canonical throughput and latency results. Each script runs one
full experiment — launching engines, sweeping configurations, writing a `results.csv` —
and each is a rerun of the exact experiment that produced the shipped numbers.

The representative path runs the two of them that T3 comprises:

```bash
bash scripts/vllm_exp/run_fig6.sh   # tier C — both Fig.6 LLM columns, GT then LM
bash scripts/vllm_exp/run_fig9.sh   # tier D — the three-GPU experiment
```

Together these take about one hour on the reference machine with warm caches;
allow up to three hours for cold starts and slower hardware. The remaining experiments are
independent and can be added individually:

```bash
bash scripts/vllm_exp/run_fig6_gptoss20b.sh                   # tier C — the GT half of run_fig6.sh
bash scripts/vllm_exp/run_fig6_llama31.sh                     # tier C — the LM half of run_fig6.sh
bash scripts/vllm_exp/run_fig6_request_distribution.sh gt     # ~15 min, tier C
bash scripts/vllm_exp/run_fig6_request_distribution.sh lm     # ~20 min, tier C
bash scripts/vllm_exp/run_fig7_gptoss20b_online.sh            # ~2-3 h, tier C
```

**Before the first run**, point the scripts at your machine. Every value below
is an override with the reference machine as default; check them with
`nvidia-smi --query-gpu=index,name --format=csv` under
`CUDA_DEVICE_ORDER=PCI_BUS_ID`:

```bash
export FLUIDGPU_PYTHON=$(command -v python)     # interpreter with vLLM 0.18
export FLUIDGPU_VLLM=$(command -v vllm)
export FLUIDGPU_MODEL_GT=/path/to/gpt-oss-20b
export FLUIDGPU_MODEL_LM=/path/to/Llama-3.1-8B-Instruct
export FLUIDGPU_GPU_P=2        # prefill side, the weaker GPU (L40S here)
export FLUIDGPU_GPU_D=1        # decode side, the stronger GPU (A100 here)
export FLUIDGPU_GPU_X=0        # third GPU, Fig.9 only
export FLUIDGPU_HCA_P=mlx5_5   # IB HCA on the prefill side
export FLUIDGPU_HCA_D=mlx5_2   # IB HCA on the decode side
export FLUIDGPU_RERUN_OUT=$PWD/artifacts/validation/vllm_graphs/reruns
```

All of these live in `scripts/vllm_exp/exp_env.sh`, which every
script sources. Resolution is checkable without running anything:

```bash
bash -c 'source scripts/vllm_exp/exp_env.sh; echo "$PY | $VB | $MODEL_GT | GPUs $GPU_P/$GPU_D/$GPU_X | $HCA_P/$HCA_D"'
```

Operational notes:

- The scripts kill every GPU process between configurations. Run them on an idle box.
- Mooncake/RDMA rows need the point-to-point fabric up (subnet manager +
  `nvidia_peermem` after a reboot — see AE_MACHINE_RUNBOOK.md).
- Results land in `$FLUIDGPU_RERUN_OUT/<experiment>/`, never overwriting the
  canonical tables under `artifacts/validation/vllm_graphs/`.

### What each experiment should produce

Compare your `results.csv` against the canonical one named in the last column.

| Experiment | Headline to check | Canonical |
|---|---|---|
| `run_fig6.sh` | both Fig.6 LLM columns in one run: GT then LM | `gt_decoupled/results.csv`, `fig6_v3/results.csv` |
| `run_fig6_gptoss20b.sh` | 4096/684, all ms32, same client conc64: FG hybrid φ=0.18 2213 vs PD 2030 = 1.09× | `gt_decoupled/results.csv` |
| `run_fig6_llama31.sh` | hybrid φ=0.22 at 1818 vs stock PD 1501 = 1.21× | `fig6_v3/results.csv` |
| `run_fig6_request_distribution.sh gt` / `lm` | Request Dist. (best policy = least-outstanding): GT 2021 = 90% of the 1252+997 single-replica sum; LM 2015 = 98% of 1289+759, which exceeds the measured LM FG bar | `{gt_decoupled,fig6_v3}/results.csv` rows `reqdist_lo`, `serve_single_*` |
| `run_fig7_gptoss20b_online.sh` | crossing rates at 50 ms median TPOT: FG 5.4, AF 4.5, homo A100 6.2, homo L40S 7.1, PD >12 req/s | `fig7_gt_v2/crossings.csv` |
| `run_fig9.sh` | FG-3 2606 vs PD-3 1894 = 1.38× | `fig9_3card/results.csv` |

## T3b — Drilldown figures (about 1 h combined, tier C)

```bash
bash scripts/repro.sh --fig 10     # pipeline ablation
bash scripts/repro.sh --fig 11    # monitor sensitivity (W, beta sweeps)
bash scripts/repro.sh --fig 12a   # slow-network sweep
```

Also available as standalone baselines:

```bash
bash scripts/run_pd_baseline_vllm.sh        # vLLM + Mooncake disaggregated serving
bash scripts/run_af_baseline_fluidgpu.sh    # attention/FFN whole-model split
```

Every orchestrated run writes `artifacts/logs/<fig>/<run_id>/`. When a
required device, weight, credential, or dataset is missing, the script exits
with an error naming the missing dependency rather than producing a partial
result.

## T4 — Ingestion, figures, and Table III (5 minutes, tier A)

This is the whole analysis stage, and it works from a fresh checkout with no
GPU: the curated measurement records ship with the artifact.

```bash
python3 scripts/preprocess/ingest_kernel_census.py
python3 scripts/preprocess/ingest_vllm_exp.py
python3 scripts/preprocess/ingest_paper_reference_figs.py

for f in 2 3 6 7 8 9 10 11 12a 12b; do python3 scripts/plot/plot_fig$f.py; done
python3 scripts/plot/make_table3.py
```

`ingest_kernel_census.py` recomputes the Fig.2/Fig.3 quantities from the
shipped A100-vs-L40s kernel census and asserts each against the value printed
in the paper (67% of kernels faster on the L40s, 36% mean / 53% max E2E time
ratio, 45% prefill and 57% decode, 71% attention and 33% FFN). It fails loudly
rather than plotting a figure that no longer matches the paper.

Outputs: `artifacts/figures/fig*.{pdf,png}` and
`artifacts/tables/table3.{csv,md}` plus `table3_raw.csv`.

All three ingestion steps are idempotent and can be re-run at any time. The
first materializes the kernel census into Fig.2/Fig.3 records; the second
materializes the experiment CSVs into records under
`artifacts/logs/{fig6,fig7,fig9}/`; the third adds the paper-derived records
for Fig.11/12a/12b. Each archives whatever records it
displaces rather than deleting them.

### Substituting your own measurements

To put a rerun into the figures, edit the canonical CSV in place and re-run
the ingestion:

- Fig.6/7/9 rows: `artifacts/validation/vllm_graphs/<experiment>/results.csv`
- Extra Fig.6 bars (other model families, or a policy you re-measured):
  one line per bar in
  `artifacts/validation/vllm_graphs/extra_fig6_rows.csv`, setting
  `source=<your log path>`; the file's header documents every column. A `(policy, model)` combination with no row renders
  as the paper's red-X "inapplicable" marker.

## Time budget

| Scope | Wall time |
|---|---|
| Setup | ~90 min |
| T0 + T2 + T4 (no GPU needed) | ~15 min |
| T1 + T2b | ~35 min |
| Representative path (T0 → T1 → T2 → T2b → T3 → T4) | ~1-3 h |
| All T3 experiments + T3b + T4 | ~8-10 h |

The representative path is T0 → T1 → T2 → T2b → T3 → T4, where T3 is
`run_fig6.sh` followed by `run_fig9.sh`: it exercises the PTX probe, the
planner, the full vLLM-integrated runtime with its parity gate, the ingestion
path, and every figure and table, in about one hour on the warm-cache reference
machine; cold starts or slower hardware can take longer.

---

# Part 3 — Figure and table map

| Claim | Paper artifact | Orchestration / generation script(s) | Analysis / plotting script(s) | Outputs |
|---|---|---|---|---|
| C1 | Fig.2 | shipped census: `artifacts/validation/kernel_census/` (see its README); runtime's own profiler: `experiments/fig2_kernel_heterogeneity.yaml`, `bash scripts/repro.sh --fig 2` | `python3 scripts/preprocess/ingest_kernel_census.py`; `python3 scripts/plot/plot_fig2.py` | `artifacts/logs/fig2/<run_id>/`; `artifacts/figures/fig2.{pdf,png}` |
| C1 | Fig.3 | shipped census: `artifacts/validation/kernel_census/fig3_*.tsv`; kernel-group refinement probe: `experiments/fig3_coarse_granularity.yaml`, `bash scripts/repro.sh --fig 3` | `python3 scripts/preprocess/ingest_kernel_census.py`; `python3 scripts/plot/plot_fig3.py` | `artifacts/logs/fig3/<run_id>/`; `artifacts/figures/fig3.{pdf,png}` |
| C2 | Listing 1 (PTX injection) | `bash scripts/ptx_memory_probe.sh`; `python3 fluidgpu_runtime/examples/fluidgpu_torch/ptx_memory_probe.py --output-dir artifacts/validation/ptx_memory_probe/<ts> --execute required --arch sm_80` | N/A | `artifacts/validation/ptx_memory_probe/<ts>/` |
| C3 | Fig.6 | `bash scripts/vllm_exp/run_fig6_gptoss20b.sh`; `bash scripts/vllm_exp/run_fig6_llama31.sh` (curated results: `artifacts/validation/vllm_graphs/{gt_decoupled,fig6_v3}/results.csv` + `extra_fig6_rows.csv`) | `python3 scripts/preprocess/ingest_vllm_exp.py`; `python3 scripts/plot/plot_fig6.py` | `artifacts/logs/fig6/<run_id>/`; `artifacts/figures/fig6.{pdf,png}` |
| C3 | Fig.7 | `bash scripts/vllm_exp/run_fig7_gptoss20b_online.sh` (curated results: `artifacts/validation/vllm_graphs/fig7_gt_v2/`) | `python3 scripts/preprocess/ingest_vllm_exp.py`; `python3 scripts/preprocess/ingest_paper_reference_figs.py`; `python3 scripts/plot/plot_fig7.py` | `artifacts/logs/fig7/<run_id>/`; `artifacts/figures/fig7.{pdf,png}` |
| C3 | Fig.8 | same online sweep as Fig.7 (`run_fig7_gptoss20b_online.sh`; P99 columns of `fig7_gt_v2/*.json`) | `python3 scripts/preprocess/ingest_paper_reference_figs.py`; `python3 scripts/plot/plot_fig8.py` | `artifacts/logs/fig8/<run_id>/`; `artifacts/figures/fig8.{pdf,png}` |
| C3 | Fig.9 | `bash scripts/vllm_exp/run_fig9.sh` (curated results: `artifacts/validation/vllm_graphs/fig9_3card/results.csv`) | `python3 scripts/preprocess/ingest_vllm_exp.py`; `python3 scripts/plot/plot_fig9.py` | `artifacts/logs/fig9/<run_id>/`; `artifacts/figures/fig9.{pdf,png}` |
| C3 | Fig.10 | paper values; the measured ablation (disclosed, not plotted) comes from `bash scripts/vllm_exp/run_fig10_pipeline.sh` → `artifacts/validation/vllm_graphs/fig10_pipeline/results.csv` | `python3 scripts/preprocess/ingest_paper_reference_figs.py`; `python3 scripts/plot/plot_fig10.py` | `artifacts/logs/fig10/<run_id>/`; `artifacts/figures/fig10.{pdf,png}` |
| C3 | Fig.11 | measured mode: `bash scripts/repro.sh --fig 11` (archived); shipped records: `python3 scripts/preprocess/ingest_paper_reference_figs.py` | `python3 scripts/plot/plot_fig11.py` | `artifacts/logs/fig11/<run_id>/`; `artifacts/figures/fig11.{pdf,png}` |
| C3 | Fig.12a | measured mode: `bash scripts/repro.sh --fig 12a` (native-link only, archived); shipped records: `python3 scripts/preprocess/ingest_paper_reference_figs.py` | `python3 scripts/plot/plot_fig12a.py` | `artifacts/logs/fig12a/<run_id>/`; `artifacts/figures/fig12a.{pdf,png}` |
| C3 | Fig.12b | `bash scripts/repro.sh --fig 12b` (license-capped cells fail under the trial license); paper-derived fill: `python3 scripts/preprocess/ingest_paper_reference_figs.py` | `python3 scripts/plot/plot_fig12b.py` | `artifacts/logs/fig12b/<run_id>/`; `artifacts/figures/fig12b.{pdf,png}` |
| C3 | Table III | same fig6 records (see Fig.6 row) | `python3 scripts/preprocess/ingest_vllm_exp.py`; `python3 scripts/plot/make_table3.py` | `artifacts/tables/table3.{csv,md}` |

Full numerical reproduction across all hardware pairs and dataset sweeps (Fig. 6, Fig. 9) requires the paper's multi-node environments. The same execution scripts ship here; reviewers can exercise any subset of figures supported by the available hardware.
