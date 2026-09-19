#!/bin/bash
# Inter-batch pipelining probe at PROCESS granularity (user: "不做ubatch,
# 按论文做真实 requests batch 间的流水线").
#
# Two SAME-placement lever-A instances share both GPUs, each owning half the
# requests (disjoint via --skip-prompts). When instance 1's FFN activations
# are hopping to / computing on the L40S, instance 2's attention kernels run
# on the A100 (and vice versa) — the paper's fig10 multi-stream mechanism
# approximated by the driver's inter-process time-slicing. Zero new code.
# This bounds what K concurrent step-loops (the deep in-engine fork) could buy.
#
# Arms: 2 x ms16 (32 total decode rows = fig6 bs32 protocol) and 2 x ms32
# (64 total, compare vs single ms64 = 1349.9). Memory: each instance's home
# (A100) side gets util 0.46 (0.40 left -0.02 GiB KV: weights+activations+
# graphs eat exactly 32 GB); the L40S expert rebuild (~12 GB/inst) sits
# outside vLLM accounting (cf. mirror runs' 0.55 note; here 2 instances).
# refs: single lever A ms32 1122-1128 / ms64 1349.9 / ms128 1452.9; PD 1827.
set -u
S=/tmp/fluidgpu-scratchpad
cd $HOME/workspace/FluidGPU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
V=$HOME/.python/vllm0.18/bin/python
DRV=fluidgpu_runtime/examples/fluidgpu_torch/vllm_kernel_disagg.py
GTOSS=$HOME/.cache/fluidgpu/models/openai--gpt-oss-20b
UDS=datasets/splitwise_spliced_gptoss_4096_384.jsonl
O=$S/dual_instance_probe; mkdir -p "$O"

cleanup() {
  pkill -9 -f vllm_kernel_disagg 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | xargs -r kill -9 2>/dev/null
  sleep 5; true
}

one_instance() {
  local tag=$1 ms=$2 np=$3 skip=$4 util=$5
  FLUIDGPU_FULL_DECODE_GRAPH=1 timeout -k 30 2400 $V $DRV \
    --model $GTOSS --dataset-jsonl $UDS \
    --num-prompts $np --skip-prompts $skip --max-num-seqs $ms \
    --max-model-len 5128 --no-prefix-cache \
    --gpu-memory-utilization $util --kv-cache-memory-gb 12 \
    --placement af --phase-aware --pingpong \
    --output-json "$O/$tag.json" > "$O/$tag.log" 2>&1
  echo "rc_$tag=$?"
}

pair() {
  local name=$1 ms=$2
  cleanup
  echo "######## $name (2 x ms=$ms, 96+96 prompts) : $(date +%H:%M:%S) ########"
  nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits -lms 100 \
    > "$O/$name.duty.csv" 2>&1 &
  local SMI=$!
  one_instance "${name}_i1" $ms 96 0  0.46 &
  local P1=$!
  one_instance "${name}_i2" $ms 96 96 0.46 &
  local P2=$!
  wait $P1 $P2
  kill -9 $SMI 2>/dev/null
  for i in i1 i2; do
    echo "$name-$i tput=$(grep -oE '"output_tok_s"[: ]*[0-9.]+' "$O/${name}_$i.json" 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)"
  done
}

pair dual_ms16 16
pair dual_ms32 32
cleanup
echo "DUAL_INSTANCE_DONE"
