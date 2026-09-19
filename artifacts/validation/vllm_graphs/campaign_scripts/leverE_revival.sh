#!/bin/bash
# Lever E revival: the one unmeasured cell in the E family.
#
# Original lever E (phase-split ubatch) died twice: maximally imbalanced
# ubatches (31 decode vs 4876 prefill) x DBO strict alternation. §38's duty
# decomposition shows the REAL prize it never touched: pure-decode segments
# (ms32 41.1% / ms128 27.8% of gen wall) where the L40S idles because the
# wave-synchronized scheduler has no prefill admitted — a SCHEDULER property.
#
# New cell = decode-cap two-pool scheduler (lever C machinery: keeps prefill
# available through decode phases) x BALANCED stock DBO token split (decode
# rows all land at the front of ubatch 0, no phase-split imbalance) x
# FLUIDGPU_UBATCH_PHASE_ROUTE=1 (NEW: ubatch0's decode rows run on the home
# marlin copy via local_thunk, overlapping the prefill rows' remote FFN).
# This fixes both diagnosed killers of lever C's 682 (decode rows hopped every
# step; steps could fall below the ping-pong threshold into serial AF).
#
# refs (spliced 4096/384, 192 prompts): lever A ms32 1122-1128 / ms64 1345 /
# ms128 1452.9; NULL_HOP 1772; PD 1827. Stop-loss: if cap arms < plain ms64
# baseline, the E family is closed for good.
set -u
S=/tmp/fluidgpu-scratchpad
cd $HOME/workspace/FluidGPU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
V=$HOME/.python/vllm0.18/bin/python
DRV=fluidgpu_runtime/examples/fluidgpu_torch/vllm_kernel_disagg.py
GTOSS=$HOME/.cache/fluidgpu/models/openai--gpt-oss-20b
UDS=datasets/splitwise_spliced_gptoss_4096_384.jsonl
O=$S/leverE_revival; mkdir -p "$O"

cleanup() {
  pkill -9 -f vllm_kernel_disagg 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | xargs -r kill -9 2>/dev/null
  sleep 5; true
}

run() {
  local tag=$1 ms=$2 np=$3 upr=$4 cap=$5; shift 5
  cleanup
  echo "######## $tag (ms=$ms np=$np upr=$upr cap=$cap) : $(date +%H:%M:%S) ########"
  nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits -lms 100 \
    > "$O/$tag.duty.csv" 2>&1 &
  local SMI=$!
  local capargs=()
  if [ "$cap" -gt 0 ]; then capargs=(--decode-cap "$cap"); fi
  FLUIDGPU_FULL_DECODE_GRAPH=1 FLUIDGPU_UBATCH_PHASE_ROUTE=$upr \
    timeout -k 30 1800 $V $DRV \
    --model $GTOSS --dataset-jsonl $UDS \
    --num-prompts $np --max-num-seqs $ms --max-model-len 5128 --no-prefix-cache \
    --placement af --phase-aware --pingpong \
    "${capargs[@]}" \
    --output-json "$O/$tag.json" "$@" > "$O/$tag.log" 2>&1
  echo "rc=$?  tput=$(grep -oE '"output_tok_s"[: ]*[0-9.]+' "$O/$tag.json" 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)"
  kill -9 $SMI 2>/dev/null
  sleep 1
}

# fail-fast parity gate: full stack (UPR + cap) greedy texts vs lever A base
run upr_parity8 64 8 1 32 --dump-texts "$O/upr_parity8.texts.json"
$V - <<'EOF'
import json
a = json.load(open("/tmp/fluidgpu-scratchpad/leverA_fix_ab/parity_base8.texts.json"))
b = json.load(open("/tmp/fluidgpu-scratchpad/leverE_revival/upr_parity8.texts.json"))
m = sum(1 for x, y in zip(a, b) if x == y)
print(f"UPR_PARITY {m}/{len(a)}")
EOF

run base_ms64  64 192 0 0
run upr_ms64   64 192 1 0
run cap32_ms64 64 192 1 32
run cap32_ms96 96 192 1 32
cleanup
echo "LEVERE_REVIVAL_DONE"
