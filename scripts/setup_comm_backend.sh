#!/usr/bin/env bash
set -euo pipefail

rank="${FLUIDGPU_RANK:-${1:-0}}"
transport="${FLUIDGPU_COMM_TRANSPORT:-rdma}"
if [[ "$rank" != "0" && "$rank" != "1" ]]; then
  echo "setup_comm_backend.sh: rank must be 0 or 1, got $rank" >&2
  exit 1
fi

export FLUIDGPU_COMM_TRANSPORT="$transport"
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export NCCL_IB_DISABLE=0
export NCCL_NET=IB
export NCCL_NET_GDR_LEVEL="${NCCL_NET_GDR_LEVEL:-SYS}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

# Defaults are the two ends of the authors' 200 Gbps point-to-point fabric,
# matching FLUIDGPU_HCA_D / FLUIDGPU_HCA_P in scripts/vllm_exp/exp_env.sh.
# Override per machine with NCCL_IB_HCA, or with the two variables below.
rank0_hca="${FLUIDGPU_IB_HCA_RANK0:-mlx5_2}"
rank1_hca="${FLUIDGPU_IB_HCA_RANK1:-mlx5_5}"
if [[ "$rank" == "0" ]]; then
  export NCCL_IB_HCA="${NCCL_IB_HCA:-$rank0_hca}"
else
  export NCCL_IB_HCA="${NCCL_IB_HCA:-$rank1_hca}"
fi

# Fail early on a device this host does not have, rather than inside NCCL.
if command -v ibv_devinfo >/dev/null 2>&1; then
  if ! ibv_devinfo -l 2>/dev/null | grep -qw "${NCCL_IB_HCA%%:*}"; then
    echo "setup_comm_backend.sh: IB device '${NCCL_IB_HCA}' not present on this host." >&2
    echo "  available: $(ibv_devinfo -l 2>/dev/null | tail -n +2 | tr -d '\t' | tr '\n' ' ')" >&2
    echo "  set NCCL_IB_HCA=<device> and re-source scripts/setup_comm_backend.sh ${rank}" >&2
    exit 1
  fi
fi

echo "FLUIDGPU_COMM_TRANSPORT=${FLUIDGPU_COMM_TRANSPORT}"
echo "NCCL_IB_HCA=${NCCL_IB_HCA}"
echo "NCCL_NET_GDR_LEVEL=${NCCL_NET_GDR_LEVEL}"
echo "This script is intended to be sourced when launching a serving rank:"
echo "  source scripts/setup_comm_backend.sh ${rank}"
