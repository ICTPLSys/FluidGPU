#!/bin/bash
# Phase B of the LM kernel-layout study: measure inside the REAL engine
# (shape 1536/1024, ms32, 64 req — diagnostics; traces perturb pipelining):
#  b_homo_trace   : A100 alone, STEP_TRACE          -> homo prefill/decode step walls
#  b_leverA_trace : lever A,   STEP_TRACE+STAGE     -> lever A step walls + L40S FFN rate + hop GB/s
#  b_af_trace     : pure AF,   STAGE                -> AF-path stage rates (decode hops incl.)
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

run() {  # tag cvd env-pairs... -- extra-args...
  local tag=$1 cvd=$2; shift 2
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  cleanup
  echo "######## $tag : $(date +%H:%M:%S) ########"
  rm -f "$O/$tag.steps" "$O/$tag.stage"
  env CUDA_VISIBLE_DEVICES=$cvd "${envs[@]}" timeout -k 30 1500 $V $DRV \
    --model $LLAMA --dataset-jsonl $UDS \
    --num-prompts 64 --max-num-seqs 32 --max-model-len $MML --no-prefix-cache \
    --output-json "$O/$tag.json" "$@" > "$O/$tag.log" 2>&1
  echo "rc=$?  tput=$(grep -oE '"output_tok_s"[: ]*[0-9.]+' "$O/$tag.json" 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)"
  [ -f "$O/$tag.steps" ] && echo "steps=$(wc -l < "$O/$tag.steps")"
  [ -f "$O/$tag.stage" ] && tail -1 "$O/$tag.stage"
}

run b_homo_trace 1 \
  FLUIDGPU_STEP_TRACE=1 FLUIDGPU_STEP_TRACE_FILE=$O/b_homo_trace.steps \
  -- --placement none

run b_leverA_trace 1,2 \
  FLUIDGPU_STEP_TRACE=1 FLUIDGPU_STEP_TRACE_FILE=$O/b_leverA_trace.steps \
  FLUIDGPU_STAGE_TIMING=1 FLUIDGPU_STAGE_TIMING_FILE=$O/b_leverA_trace.stage \
  FLUIDGPU_FULL_DECODE_GRAPH=1 \
  -- --placement af --phase-aware --pingpong

run b_af_trace 1,2 \
  FLUIDGPU_STAGE_TIMING=1 FLUIDGPU_STAGE_TIMING_FILE=$O/b_af_trace.stage \
  FLUIDGPU_FULL_DECODE_GRAPH=1 \
  -- --placement af --pingpong

cleanup
echo "PHASE_B_DONE"
