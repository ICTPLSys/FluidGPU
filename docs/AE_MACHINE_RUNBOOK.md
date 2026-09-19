# AE Machine Runbook

Machine-specific facts for the authors' evaluation host (`gpuserver86`:
2×A100 80GB PCIe + 1×L40S).

Generic installation lives in [INSTALL.md](INSTALL.md); what to run for each
figure lives in [EXPERIMENTS.md](EXPERIMENTS.md). This file is only the
"what is special about this box" note.

> The full measurement log — every experiment, every negative result, every
> retraction — is preserved in the repository history:
> `git log --follow -p docs/AE_MACHINE_RUNBOOK.md`.

---

## 1. Environment activation

The validated interpreter is a Python 3.12.13 venv (torch 2.10.0+cu129, vLLM
0.18.0, transformers 4.57.6, gurobipy 13.0.1). In the container it is
`~/fluidgpu-ae`; on the bare host it is a separate venv of the same build.

```bash
source "$HOME/fluidgpu-ae/bin/activate"
export PATH=/usr/local/cuda/bin:$PATH
```

The second line is not optional: CUDA 12.8 lives at `/usr/local/cuda-12.8` and
is *not* on the default `PATH`, while `verify_env.sh`, `build_cpp_runtime.sh`
and `ptx_memory_probe.sh` all need `nvcc`/`ptxas`.

Dependency pin that must not float: `huggingface-hub<1.0` together with
`kernels>=0.10,<0.11`. `pip install kernels` without the pin pulls
`huggingface-hub 1.x`, which breaks `transformers<5` imports.

If you run modules from an interpreter that does not have `fluidgpu_runtime`
installed — the host venv does not — prefix them with
`PYTHONPATH=$PWD/fluidgpu_runtime`. The scripts already do.

## 2. GPU topology and device mapping

Physical (PCI_BUS_ID) order: 0 = A100, 1 = A100, 2 = L40S. The experiment
YAMLs assume `gpu_id 0 = A100` and `gpu_id 1 = L40S`, so every run must remap:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1,2
```

`scripts/vllm_exp/exp_env.sh` encodes the same mapping as
`FLUIDGPU_GPU_D=1` (A100, decode side), `FLUIDGPU_GPU_P=2` (L40S, prefill
side), `FLUIDGPU_GPU_X=0` (second A100, Fig.9 only).

Any device-index probe must set `CUDA_DEVICE_ORDER=PCI_BUS_ID` first;
`FASTEST_FIRST` (the default) reorders the L40S to index 0 and silently
inverts every conclusion.

A100↔A100 P2P is available at 21–23 GB/s (PCIe gen4, no NVLink); anything
involving the L40S is `NS` in `nvidia-smi topo -p2p`.

## 3. RDMA state after a reboot (requires sudo)

The 200 Gbps loopback fabrics (mlx5_2↔mlx5_5 and mlx5_3↔mlx5_4) lose their
subnet manager and the GDR kernel module on reboot. Restore with:

```bash
sudo modprobe ib_umad
sudo opensm -B -g 0xb81d2d0300cadb35   # mlx5_2<->mlx5_5 fabric
sudo opensm -B -g 0xb81d2d0300cadb34   # mlx5_3<->mlx5_4 fabric
sudo modprobe nvidia_peermem
```

Check: `rdma link show` must report mlx5_2..5 ACTIVE, and NCCL logs must show
`via NET/IB/0/GDRDMA` plus `GDR 1`. Two-rank runs pin `NCCL_IB_HCA=mlx5_2`
(rank 0) and `mlx5_5` (rank 1) with `NCCL_NET_GDR_LEVEL=SYS`.

If `modprobe nvidia_peermem` fails with "Invalid argument": the DKMS build
picked up the stale `/usr/src/ofa_kernel` (built for a 6.8 kernel). Move it
aside, `dkms remove` + `dkms install nvidia/570.172.08 -k $(uname -r)`, move
it back, then modprobe again.

Without a subnet manager the whole acceptance set still runs over
`--nccl-ib-hca mlx5_1` (100G, NET/IB/Shared instead of GDRDMA).

## 4. Network

Outbound access from this host is restricted, so nothing is fetched at run
time. Every model weight and benchmark input the evaluation needs is already
staged in the container, and all runs set
`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`.

## 5. Weights and datasets

- Model weights: HF hub cache, symlinked into
  `~/.cache/fluidgpu/models/<org>--<name>` for the runner scripts.
- gpt-oss-20b runs MXFP4-native (`kernels` package); the MXFP4 triton kernel
  bundles for both sm80 (A100) and sm89 (L40S) are pre-fetched into the hub
  cache, so offline runs hit the cache. A bf16 dequant is 43 GB and OOMs the
  L40S.
- The runtime examples check the transformers version they were validated
  against and refuse a mismatch; pass `--allow-transformers-mismatch` to run
  them on 4.57.6.
- Datasets live in `datasets/` (COCO val2017 → `coco_512/`, PartiPrompts →
  `parti_prompts_1024/`, Azure/Splitwise traces → `conversation_benchmark/`,
  1000 requests, median 1020/129 tokens).
- Raw multi-GB run archives from the experiments stay on this machine under
  `artifacts/logs/*archive*/` and are excluded from the shipped tarball; the
  curated per-experiment `results.csv` files are tracked.

---

## Operational rules

- Any device-index probe sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` first.
- Raw-token-id clients derive the prompt id range from the model's
  `config.json` `vocab_size`.
