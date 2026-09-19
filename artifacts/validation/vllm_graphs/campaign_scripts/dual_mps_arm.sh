#!/bin/bash
# MPS arm of the dual-instance probe: same 2 x ms16 pair, but under a
# user-space CUDA MPS daemon so the two processes' kernels CO-SCHEDULE on
# each GPU instead of time-slicing whole contexts. Isolates "no overlap
# because time-slicing" from "no overlap, period". refs: dual no-MPS sum
# 981.8, single ms32 1134.9, single ms16 947.
set -u
S=/tmp/fluidgpu-scratchpad
cd $HOME/workspace/FluidGPU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
V=$HOME/.python/vllm0.18/bin/python
DRV=fluidgpu_runtime/examples/fluidgpu_torch/vllm_kernel_disagg.py
GTOSS=$HOME/.cache/fluidgpu/models/openai--gpt-oss-20b
UDS=datasets/splitwise_spliced_gptoss_4096_384.jsonl
O=$S/dual_mps; mkdir -p "$O"
export CUDA_MPS_PIPE_DIRECTORY=$O/pipe CUDA_MPS_LOG_DIRECTORY=$O/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

cleanup() {
  pkill -9 -f vllm_kernel_disagg 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | xargs -r kill -9 2>/dev/null
  sleep 5; true
}
stop_mps() { echo quit | nvidia-cuda-mps-control 2>/dev/null; sleep 2; true; }

cleanup
stop_mps
nvidia-cuda-mps-control -d
sleep 2
if ! pgrep -f nvidia-cuda-mps-control >/dev/null; then
  echo "MPS_DAEMON_FAILED"; exit 1
fi
echo "mps daemon up"

one_instance() {
  local tag=$1 ms=$2 np=$3 skip=$4
  FLUIDGPU_FULL_DECODE_GRAPH=1 timeout -k 30 2400 $V $DRV \
    --model $GTOSS --dataset-jsonl $UDS \
    --num-prompts $np --skip-prompts $skip --max-num-seqs $ms \
    --max-model-len 5128 --no-prefix-cache \
    --gpu-memory-utilization 0.46 --kv-cache-memory-gb 12 \
    --placement af --phase-aware --pingpong \
    --output-json "$O/$tag.json" > "$O/$tag.log" 2>&1
  echo "rc_$tag=$?"
}

echo "######## mps_ms16 (2 x ms=16, 96+96, MPS) : $(date +%H:%M:%S) ########"
nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits -lms 100 \
  > "$O/mps_ms16.duty.csv" 2>&1 &
SMI=$!
one_instance mps_ms16_i1 16 96 0 &
P1=$!
one_instance mps_ms16_i2 16 96 96 &
P2=$!
wait $P1 $P2
kill -9 $SMI 2>/dev/null
for i in i1 i2; do
  echo "mps_ms16-$i tput=$(grep -oE '"output_tok_s"[: ]*[0-9.]+' "$O/mps_ms16_$i.json" 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)"
done
cleanup
stop_mps
echo "DUAL_MPS_DONE"
