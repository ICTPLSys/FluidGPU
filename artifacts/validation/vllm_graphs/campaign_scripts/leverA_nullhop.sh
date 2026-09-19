#!/bin/bash
# NULL_HOP A/B: bound the hop DATA-PLANE's contribution to lever A's gap.
#
# FLUIDGPU_NULL_HOP=1 keeps the entire coordination structure (streams, events,
# dbo_yield, Python per layer x ubatch) but skips the actual D2H/H2D bytes
# (returns garbage on the dst device). Output text is garbage BY DESIGN —
# only throughput is meaningful.
#   nullhop tput ~= baseline 1452.9  -> transfers are NOT the cost; it's all
#                                       coordination/launch (CPU). Any data-plane
#                                       upgrade (RDMA, IBGDA) is bounded at ~0.
#   nullhop tput >> baseline         -> the data plane matters after all.
# Unprofiled, same workload as §36 leverA_duty (ref ms128 = 1452.9).
set -u
S=/tmp/fluidgpu-scratchpad
cd $HOME/workspace/FluidGPU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
V=$HOME/.python/vllm0.18/bin/python
DRV=fluidgpu_runtime/examples/fluidgpu_torch/vllm_kernel_disagg.py
GTOSS=$HOME/.cache/fluidgpu/models/openai--gpt-oss-20b
UDS=datasets/splitwise_spliced_gptoss_4096_384.jsonl
O=$S/leverA_nullhop; mkdir -p "$O"

cleanup() {
  pkill -9 -f vllm_kernel_disagg 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | xargs -r kill -9 2>/dev/null
  sleep 5; true
}

run() {
  local tag=$1 ms=$2; shift 2
  cleanup
  echo "######## $tag (ms=$ms) : $(date +%H:%M:%S) ########"
  nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits -lms 100 \
    > "$O/$tag.duty.csv" 2>&1 &
  local SMI=$!
  FLUIDGPU_FULL_DECODE_GRAPH=1 "$@" timeout -k 30 1800 $V $DRV \
    --model $GTOSS --dataset-jsonl $UDS \
    --num-prompts 192 --max-num-seqs $ms --max-model-len 5128 --no-prefix-cache \
    --placement af --phase-aware --pingpong \
    --output-json "$O/$tag.json" > "$O/$tag.log" 2>&1
  echo "rc=$?  tput=$(grep -oE '"output_tok_s"[: ]*[0-9.]+' "$O/$tag.json" 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)"
  kill -9 $SMI 2>/dev/null
  sleep 1
}

run nullhop_ms128 128 env FLUIDGPU_NULL_HOP=1
run nullhop_ms32  32  env FLUIDGPU_NULL_HOP=1
cleanup
echo "LEVERA_NULLHOP_DONE"
