# Installation

Installation path for the minimum A100 80GB PCIe + L40S configuration.
Once this completes, go to [EXPERIMENTS.md](EXPERIMENTS.md) for what to run.

## 1. Unpack

Download the archive from the artifact's persistent DOI,
<https://doi.org/10.5281/zenodo.19710819>, and unpack it:

```bash
cd /path/to/workspace
tar -xzf FluidGPU-*.tar.gz     # or unzip the Zenodo download
cd FluidGPU
```

PyTorch comes from the `cu129` wheel index configured in `requirements.txt`
(§2). Those wheels carry their own CUDA runtime, so they do not have to match
the 12.8 toolkit the host uses for `nvcc`/`ptxas`.

## 2. Python Environment

`scripts/setup_env.sh` is the entry point: it creates a virtual environment
under `${HOME}/fluidgpu-ae`, installs `requirements.txt`, and installs
`fluidgpu_runtime` editable. It requires Python 3.12.13 — point
`FLUIDGPU_PYTHON` at that interpreter if it is not the default `python3`, and
`FLUIDGPU_VENV` elsewhere if you want the environment somewhere else.

```bash
bash scripts/setup_env.sh
source "$HOME/fluidgpu-ae/bin/activate"
```

The runtime expects PyTorch 2.10.0, vLLM 0.18.0, NCCL 2.27.5 and Gurobi
13.0.1. No Gurobi license file is needed: the size-limited license bundled
with `pip install gurobipy` covers every solve the artifact performs, and the
default DP planner needs none at all.

`fluidgpu_torch` is importable from the editable install; if you run modules
directly from a different interpreter, prefix them with
`PYTHONPATH=$PWD/fluidgpu_runtime`.

Verify the install with the unit tests:

```bash
PYTHONPATH=$PWD/fluidgpu_runtime python3 -m pytest fluidgpu_runtime -q
```

Expected: `127 passed`. They need no GPU and take about 15 seconds.

## 3. C++/CUDA Components

The C++/CUDA runtime sources build as a standalone static library under
`cpp_runtime/`:

```bash
bash scripts/build_cpp_runtime.sh
```

The script checks for CMake, `nvcc`, CUDA headers, `g++`, `libibverbs`,
`librdmacm` and the vendored `cpp_runtime/third_party/nlohmann/json.hpp`
header, and exits before configuring if any is missing, naming the one it
could not find.

## 4. Environment Verification

CUDA must be on `PATH` for this step and for `build_cpp_runtime.sh` /
`ptx_memory_probe.sh` — on many installs it is not by default:

```bash
export PATH=/usr/local/cuda/bin:$PATH
bash scripts/verify_env.sh
```

It checks Python, CUDA 12.8, `ptxas`, Gurobi, InfiniBand and the A100/L40S
GPU names, then prints the repository commit and tree hash. Versions are
pinned exactly; a mismatch is reported as a failure but does not stop any
experiment from running.

For RDMA/NCCL experiments, source the communication setup in each rank shell before launching:

```bash
source scripts/setup_comm_backend.sh 0
source scripts/setup_comm_backend.sh 1
```

`scripts/setup_comm_backend.sh` defaults to the authors' HCAs (`mlx5_2` for rank 0, `mlx5_5` for rank 1) and refuses to continue if the device is absent, listing what the host does have. Override with `NCCL_IB_HCA` (or `PREFILL_MOONCAKE_DEVICES` / `DECODE_MOONCAKE_DEVICES` in the PD scripts) on any other machine.

### RDMA fabric bring-up (subnet manager and `nvidia_peermem`)

`verify_env.sh` requires at least one ACTIVE IB port. Two pieces of state do
not survive a reboot and must be restored with `sudo`:

```bash
sudo modprobe ib_umad                  # required by opensm
sudo opensm -B -g <port-GUID>          # one per fabric; any port on it
sudo modprobe nvidia_peermem           # GPUDirect RDMA (NCCL "GDR 1")
```

List candidate GUIDs with `ibstat` (`Port GUID`), and start one `opensm` per
independent fabric — a point-to-point link between two HCAs has no external
subnet manager, so nothing comes up ACTIVE until one is running. The exact
GUIDs used on the authors' machine are in
[AE_MACHINE_RUNBOOK.md](AE_MACHINE_RUNBOOK.md).

Verify with `rdma link show` (ports ACTIVE) and, during a two-rank run, NCCL
reporting `via NET/IB/.../GDRDMA` with `GDR 1`. Without `nvidia_peermem` NCCL
falls back to a staged host path (`GDR 0`), which is slower but still correct.

If `modprobe nvidia_peermem` fails with `Invalid argument`, a stale
`/usr/src/ofa_kernel` (a MOFED symbol tree built for a different kernel) was
picked up by the DKMS build. Move it aside, rebuild the NVIDIA module for the
running kernel (`dkms remove` + `dkms install nvidia/<driver-version> -k
$(uname -r)`), move it back, then `modprobe` again.

Without a subnet manager at all, the full acceptance set still runs over any
single ACTIVE HCA (`--nccl-ib-hca <device>`), using `NET/IB/Shared` instead of
GDRDMA.

## 5. Data And Weights

```bash
bash scripts/fetch_weights.sh
bash scripts/fetch_datasets.sh
bash scripts/preprocess_datasets.sh --task all
```

Set `HF_TOKEN` for gated Hugging Face models. The default model directory is
`${FLUIDGPU_MODEL_DIR:-~/.cache/fluidgpu/models}`; datasets land under
`${FLUIDGPU_DATASET_DIR:-datasets}`.

### What those three commands fetch

| Resource | Source URL or checkpoint | Download command | Target | AD usage |
|---|---|---|---|---|
| Llama 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` | `bash scripts/fetch_weights.sh` | `${FLUIDGPU_MODEL_DIR}/meta-llama--Llama-3.1-8B-Instruct` | A100+L40S validation workload; gated HF license requires `HF_TOKEN`. |
| GPT-oss 20B | `openai/gpt-oss-20b` | `bash scripts/fetch_weights.sh` | `${FLUIDGPU_MODEL_DIR}/openai--gpt-oss-20b` | MoE profile and AD validation target. |
| Qwen2.5-VL 7B Instruct | `Qwen/Qwen2.5-VL-7B-Instruct` | `bash scripts/fetch_weights.sh` | `${FLUIDGPU_MODEL_DIR}/Qwen--Qwen2.5-VL-7B-Instruct` | MLLM capability registration; AE completes end-to-end run. |
| Mamba-Codestral 7B | `mistralai/Mamba-Codestral-7B-v0.1` | `bash scripts/fetch_weights.sh` | `${FLUIDGPU_MODEL_DIR}/mistralai--Mamba-Codestral-7B-v0.1` | SSM capability registration; AE completes end-to-end run. |
| Stable Diffusion 3.5 Medium | `stabilityai/stable-diffusion-3.5-medium` | `bash scripts/fetch_weights.sh` | `${FLUIDGPU_MODEL_DIR}/stabilityai--stable-diffusion-3.5-medium` | Diffusion capability registration; AE completes end-to-end run. |
| Splitwise-style requests | `https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_code.csv`, `https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/AzureLLMInferenceTrace_conv.csv` | `bash scripts/fetch_datasets.sh` | `${FLUIDGPU_DATASET_DIR}/splitwise` | Conversation request source aligned with Splitwise paper trace release. |
| COCO captioning | `http://images.cocodataset.org/zips/val2017.zip`, `http://images.cocodataset.org/annotations/annotations_trainval2017.zip` | `bash scripts/fetch_datasets.sh` | `${FLUIDGPU_DATASET_DIR}/coco` | Raw image/caption source for MLLM preprocessing. |
| PartiPrompts | `hf://nateraw/parti-prompts` | `bash scripts/fetch_datasets.sh` | `${FLUIDGPU_DATASET_DIR}/parti-prompts` | Raw prompt source for diffusion preprocessing. |
| Azure conversation trace | `https://github.com/Azure/AzurePublicDataset` | `bash scripts/fetch_datasets.sh` | `${FLUIDGPU_DATASET_DIR}/azure` | Trace replay source for online monitor experiments. |

### Preprocessing

```bash
bash scripts/preprocess_datasets.sh --task coco
bash scripts/preprocess_datasets.sh --task parti
bash scripts/preprocess_datasets.sh --task conversations
```

Entry points and outputs:

| Preprocess target | Script | Output |
|---|---|---|
| COCO 512x512 manifest | `scripts/preprocess/coco_512.py` | `${FLUIDGPU_DATASET_DIR}/coco_512/manifest.json` |
| PartiPrompts 1024x1024, 28 denoising steps | `scripts/preprocess/parti_prompts.py` | `${FLUIDGPU_DATASET_DIR}/parti_prompts_1024/prompts.jsonl` |
| Conversation benchmark requests | `scripts/preprocess/conversation_requests.py` | `${FLUIDGPU_DATASET_DIR}/conversation_benchmark/requests.jsonl` |

The preprocessing scripts write manifests and command-level artifacts; full
resized-image / prompt materialization is an AE-scale run.

## 6. Small Validation Workload

```bash
bash scripts/validation.sh --requests 100
```

If model weights are unavailable, the validation script exits before generation and prints the next step (`scripts/fetch_weights.sh` or set `HF_TOKEN`).
