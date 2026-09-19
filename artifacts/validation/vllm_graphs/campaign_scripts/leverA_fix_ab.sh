#!/bin/bash
# Fix A/Bs for lever A's two quantified gaps (from the py-spy + NULL_HOP session):
#   GAP1 hop data-plane: NULL_HOP (free hops) = 1772 vs baseline 1452.9 (+22%).
#        -> FLUIDGPU_PINGPONG_HOP=rdma: GDR single-leg ~19.6 GB/s vs pinned
#           two serialized legs (~7-8 GB/s effective round trip).
#   GAP2 decode serialization: engine main blocks in step_with_batch_queue
#        future.result() (48% of gen wall) because step N+1's inputs need
#        step N's sampled tokens -> GPU idles through every decode step's CPU
#        segment. -> --async-scheduling (vLLM native EngineArgs).
# refs: baseline ms128=1452.9 ms32=1122..1128 | NULL_HOP 1772/1247 | PD 1827.
# Parity smoke at the end: async must not change greedy tokens vs sync lever A.
set -u
S=/tmp/fluidgpu-scratchpad
cd $HOME/workspace/FluidGPU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
V=$HOME/.python/vllm0.18/bin/python
DRV=fluidgpu_runtime/examples/fluidgpu_torch/vllm_kernel_disagg.py
GTOSS=$HOME/.cache/fluidgpu/models/openai--gpt-oss-20b
UDS=datasets/splitwise_spliced_gptoss_4096_384.jsonl
O=$S/leverA_fix_ab; mkdir -p "$O"

cleanup() {
  pkill -9 -f vllm_kernel_disagg 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | xargs -r kill -9 2>/dev/null
  sleep 5; true
}

run() {
  local tag=$1 ms=$2 np=$3; shift 3
  cleanup
  echo "######## $tag (ms=$ms np=$np) : $(date +%H:%M:%S) ########"
  nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits -lms 100 \
    > "$O/$tag.duty.csv" 2>&1 &
  local SMI=$!
  FLUIDGPU_FULL_DECODE_GRAPH=1 timeout -k 30 1800 $V $DRV \
    --model $GTOSS --dataset-jsonl $UDS \
    --num-prompts $np --max-num-seqs $ms --max-model-len 5128 --no-prefix-cache \
    --placement af --phase-aware --pingpong \
    --output-json "$O/$tag.json" "$@" > "$O/$tag.log" 2>&1
  echo "rc=$?  tput=$(grep -oE '"output_tok_s"[: ]*[0-9.]+' "$O/$tag.json" 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)"
  kill -9 $SMI 2>/dev/null
  sleep 1
}

export FLUIDGPU_PINGPONG_HOP=rdma
run rdma_ms128 128 192
export FLUIDGPU_PINGPONG_HOP=pinned

run async_ms128 128 192 --async-scheduling

export FLUIDGPU_PINGPONG_HOP=rdma
run combo_ms128 128 192 --async-scheduling
export FLUIDGPU_PINGPONG_HOP=pinned

run async_ms32 32 192 --async-scheduling
export FLUIDGPU_PINGPONG_HOP=rdma
run rdma_ms32 32 192
export FLUIDGPU_PINGPONG_HOP=pinned

# parity smoke: async vs sync lever A, 8 prompts, greedy texts must match
run parity_base8  32 8 --dump-texts "$O/parity_base8.texts.json"
run parity_async8 32 8 --async-scheduling --dump-texts "$O/parity_async8.texts.json"
$V - <<'EOF'
import json
a = json.load(open("/tmp/fluidgpu-scratchpad/leverA_fix_ab/parity_base8.texts.json"))
b = json.load(open("/tmp/fluidgpu-scratchpad/leverA_fix_ab/parity_async8.texts.json"))
match = sum(1 for x, y in zip(a, b) if x == y)
print(f"ASYNC_PARITY {match}/{len(a)}")
EOF
cleanup
echo "LEVERA_FIX_AB_DONE"
