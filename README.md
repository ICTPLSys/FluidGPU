# FluidGPU Artifact

FluidGPU is the artifact for fine-grained kernel disaggregation on
heterogeneous GPUs. It integrates with PyTorch 2.10 and vLLM 0.18 for LLM
serving and exposes the paper's PTX analyzer, policy planner, GPU worker
runtime, and online monitor as inspectable scripts and modules.

## Start Here

| Document | Purpose |
|---|---|
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Reproduction path, per-task walkthrough, result tables, protocol, and disclosures. |
| [docs/INSTALL.md](docs/INSTALL.md) | Installation and deployment on Ubuntu 22.04 with A100/L40S GPUs. |
| [docs/AE_MACHINE_RUNBOOK.md](docs/AE_MACHINE_RUNBOOK.md) | Reference-machine topology and operational notes. |

The end-to-end experiments require a heterogeneous GPU pair with an RDMA
interconnect. The reference setup used 2× A100 80 GB PCIe and 1× L40S with
200 Gbps InfiniBand. Software, data, and reproduction scripts are included;
model weights and large public datasets are fetched separately. The temporary
AE reviewer login and WireGuard credentials are not part of this public
repository. To reproduce on your own machine, follow [docs/INSTALL.md](docs/INSTALL.md)
and then [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Artifact Map

| Component | Path | Notes |
|---|---|---|
| PTX analyzer | [fluidgpu_runtime/fluidgpu_torch/cupti_ptx.py](fluidgpu_runtime/fluidgpu_torch/cupti_ptx.py) | Classifies PTX/CUPTI kernel records and powers the PTX memory probe. |
| Policy planner | [fluidgpu_runtime/fluidgpu_torch/planner.py](fluidgpu_runtime/fluidgpu_torch/planner.py), [fluidgpu_runtime/fluidgpu_torch/milp_planner.py](fluidgpu_runtime/fluidgpu_torch/milp_planner.py) | Dynamic-programming and MILP planners for heterogeneous placement. |
| GPU worker runtime | [fluidgpu_runtime/fluidgpu_torch/engine.py](fluidgpu_runtime/fluidgpu_torch/engine.py), [fluidgpu_runtime/fluidgpu_torch/executor.py](fluidgpu_runtime/fluidgpu_torch/executor.py), [fluidgpu_runtime/fluidgpu_torch/runner.py](fluidgpu_runtime/fluidgpu_torch/runner.py) | PyTorch CausalLM execution with NCCL-backed rank handoff. |
| vLLM integration | [fluidgpu_runtime/fluidgpu_torch/vllm_integration.py](fluidgpu_runtime/fluidgpu_torch/vllm_integration.py) | Pure-Python integration with vLLM 0.18. |
| Online monitor | [fluidgpu_runtime/fluidgpu_torch/monitor.py](fluidgpu_runtime/fluidgpu_torch/monitor.py) | CSV latency logger and queueing-aware policy monitor. |
| C++ runtime | [cpp_runtime](cpp_runtime) | Standalone static library, buildable with CUDA and RDMA dependencies. |
| Experiment entry points | [scripts/vllm_exp](scripts/vllm_exp) | Rerun scripts for the end-to-end figures. |

## Quick Start

```bash
# Environment and smoke tests (see docs/INSTALL.md first)
bash scripts/setup_env.sh
source "$HOME/fluidgpu-ae/bin/activate"
bash scripts/verify_env.sh
PYTHONPATH=$PWD/fluidgpu_runtime python3 -m pytest fluidgpu_runtime -q

# Regenerate figures and Table III from the included records (no GPU needed)
python3 scripts/preprocess/ingest_kernel_census.py
python3 scripts/preprocess/ingest_vllm_exp.py
python3 scripts/preprocess/ingest_paper_reference_figs.py
for f in 2 3 6 7 8 9 10 11 12a 12b; do python3 scripts/plot/plot_fig$f.py; done
python3 scripts/plot/make_table3.py

# Weights, datasets, component probes
bash scripts/fetch_weights.sh
bash scripts/fetch_datasets.sh
bash scripts/ptx_memory_probe.sh --execute required
bash scripts/demo_planner.sh
bash scripts/build_cpp_runtime.sh

# Correctness gate, then a full experiment
bash scripts/verify_parity.sh
bash scripts/vllm_exp/run_fig6_gptoss20b.sh
```

## License

MIT — see [LICENSE](LICENSE). Citation metadata is in
[CITATION.cff](CITATION.cff).
