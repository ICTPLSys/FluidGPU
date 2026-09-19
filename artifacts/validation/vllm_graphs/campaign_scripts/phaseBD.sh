#!/bin/bash
# Phase B2 + D of the LM kernel-layout study.
# B2: b_homoW_trace = same worker+trace instrumentation as lever A but with the
#     ping-pong prefill threshold set above any step (=> all-local compute):
#     the apples-to-apples homo prefill us/token.
# D:  192-req untraced A/B of the resource model's top candidate:
#     balanced prefill split f*=0.20 (A100 attn + 20% MLP rows local, 80% MLP
#     remote on L40S) composed with lever A, vs same-day lever A re-ref.
set -u
S=/tmp/fluidgpu-scratchpad
O=$S/lm_kernel_layout
cd $HOME/workspace/FluidGPU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
V=$HOME/.python/vllm0.18/bin/python
LLAMA=$HOME/.cache/fluidgpu/models/meta-llama--Llama-3.1-8B-Instruct
UDS=datasets/splitwise_spliced_llama_4096_1024.jsonl
DRV=fluidgpu_runtime/examples/fluidgpu_torch/vllm_kernel_disagg.py
MML=2568

cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  sleep 5; true
}

run() {  # tag cvd nreq env-pairs... -- extra-args...
  local tag=$1 cvd=$2 nreq=$3; shift 3
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  cleanup
  echo "######## $tag (n=$nreq) : $(date +%H:%M:%S) ########"
  env CUDA_VISIBLE_DEVICES=$cvd "${envs[@]}" timeout -k 30 2400 $V $DRV \
    --model $LLAMA --dataset-jsonl $UDS \
    --num-prompts $nreq --max-num-seqs 32 --max-model-len $MML --no-prefix-cache \
    --output-json "$O/$tag.json" "$@" > "$O/$tag.log" 2>&1
  echo "rc=$?  tput=$(grep -oE '"output_tok_s"[: ]*[0-9.]+' "$O/$tag.json" 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)"
}

# B2: instrumented homo (all compute local through the same worker)
run b_homoW_trace 1,2 64 \
  FLUIDGPU_STEP_TRACE=1 FLUIDGPU_STEP_TRACE_FILE=$O/b_homoW_trace.steps \
  FLUIDGPU_FULL_DECODE_GRAPH=1 \
  FLUIDGPU_PINGPONG_PREFILL_THRESHOLD=999999999 \
  -- --placement af --phase-aware --pingpong

# D: parity gate for balanced (8 prompts, dump texts, compare vs base)
cleanup
echo "######## d_bal020 parity gate : $(date +%H:%M:%S) ########"
env CUDA_VISIBLE_DEVICES=1,2 FLUIDGPU_FULL_DECODE_GRAPH=1 \
  FLUIDGPU_BALANCED=1 FLUIDGPU_LOCAL_FFN_FRAC=0.20 \
  timeout -k 30 1200 $V $DRV \
  --model $LLAMA --dataset-jsonl $UDS \
  --num-prompts 8 --max-num-seqs 32 --max-model-len $MML --no-prefix-cache \
  --placement af --phase-aware --pingpong \
  --dump-texts "$O/d_bal020_parity8.texts.json" \
  --output-json "$O/d_bal020_parity8.json" > "$O/d_bal020_parity.log" 2>&1
env CUDA_VISIBLE_DEVICES=1 timeout -k 30 1200 $V $DRV \
  --model $LLAMA --dataset-jsonl $UDS \
  --num-prompts 8 --max-num-seqs 32 --max-model-len $MML --no-prefix-cache \
  --placement none \
  --dump-texts "$O/d_base_parity8.texts.json" \
  --output-json "$O/d_base_parity8.json" > "$O/d_base_parity.log" 2>&1
$V - <<'EOF'
import json
a = json.load(open("/tmp/fluidgpu-scratchpad/lm_kernel_layout/d_base_parity8.texts.json"))
b = json.load(open("/tmp/fluidgpu-scratchpad/lm_kernel_layout/d_bal020_parity8.texts.json"))
m = sum(1 for x, y in zip(a, b) if x == y)
print(f"BAL020_PARITY {m}/{len(a)}")
EOF

# D: 192-req A/Bs (untraced, campaign protocol)
run d_leverA_ref 1,2 192 \
  FLUIDGPU_FULL_DECODE_GRAPH=1 \
  -- --placement af --phase-aware --pingpong
run d_bal020 1,2 192 \
  FLUIDGPU_FULL_DECODE_GRAPH=1 FLUIDGPU_BALANCED=1 FLUIDGPU_LOCAL_FFN_FRAC=0.20 \
  -- --placement af --phase-aware --pingpong
run d_bal035 1,2 192 \
  FLUIDGPU_FULL_DECODE_GRAPH=1 FLUIDGPU_BALANCED=1 FLUIDGPU_LOCAL_FFN_FRAC=0.35 \
  -- --placement af --phase-aware --pingpong

cleanup
echo "PHASE_BD_DONE"
