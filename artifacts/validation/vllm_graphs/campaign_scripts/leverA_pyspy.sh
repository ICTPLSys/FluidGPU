#!/bin/bash
# P0 CPU-side profile of lever A (§37's queued next step).
#
# Question: lever A ms128 runs both GPUs at ~50% duty and looks CPU/launch-bound
# (§36/§37). Where does the CPU/wall time go?  Suspects (all unmeasured):
#   ping-pong _cpu_yield threading.Event alternation | per-layer _PinnedStage
#   D2H->H2D + event syncs | wait_event blocking | _overlapped_remote Python
#   x24 layers x2 ubatches | vLLM core step overhead (sched/sample/detok/zmq)
#
# ptrace_scope=1 => cannot attach; run py-spy as ANCESTOR with --subprocesses
# (EngineCore is a descendant, so it gets profiled too).
# Two arms, identical workload (== §36 leverA_duty ms128, ref tput 1452.9):
#   idle : py-spy --idle  -> wall-clock view incl. blocked threads
#   oncpu: py-spy default -> on-CPU view (where CPU actually burns)
# Each arm also gets: nvidia-smi duty csv + /proc per-thread CPU sampler.
# py-spy's own distortion is quantified by final output_tok_s vs 1452.9.
set -u
S=/tmp/fluidgpu-scratchpad
cd $HOME/workspace/FluidGPU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
V=$HOME/.python/vllm0.18/bin/python
PYSPY=$HOME/.python/vllm0.18/bin/py-spy
DRV=fluidgpu_runtime/examples/fluidgpu_torch/vllm_kernel_disagg.py
GTOSS=$HOME/.cache/fluidgpu/models/openai--gpt-oss-20b
UDS=datasets/splitwise_spliced_gptoss_4096_384.jsonl
O=$S/leverA_pyspy; mkdir -p "$O"

cleanup() {
  pkill -9 -f vllm_kernel_disagg 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | xargs -r kill -9 2>/dev/null
  sleep 5; true
}

run() {
  local tag=$1; shift
  cleanup
  echo "######## $tag : $(date +%H:%M:%S) ########"
  nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits -lms 100 \
    > "$O/$tag.duty.csv" 2>&1 &
  local SMI=$!
  $V "$S/thread_cpu_sampler.py" "$O/$tag.threads.csv" 2>> "$O/$tag.sampler.log" &
  local SAMP=$!
  FLUIDGPU_FULL_DECODE_GRAPH=1 timeout -k 30 2400 \
    "$PYSPY" record --subprocesses --nonblocking --rate 50 --format speedscope \
    -o "$O/$tag.speedscope.json" "$@" -- \
    $V $DRV \
    --model $GTOSS --dataset-jsonl $UDS \
    --num-prompts 192 --max-num-seqs 128 --max-model-len 5128 --no-prefix-cache \
    --placement af --phase-aware --pingpong \
    --output-json "$O/$tag.json" > "$O/$tag.log" 2>&1
  local rc=$?
  echo "rc=$rc  tput=$(grep -oE '"output_tok_s"[: ]*[0-9.]+' "$O/$tag.json" 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)  (ref 1452.9 unprofiled)"
  kill -9 $SMI $SAMP 2>/dev/null
  sleep 1
}

run idle --idle
run oncpu
cleanup
echo "LEVERA_PYSPY_DONE"
