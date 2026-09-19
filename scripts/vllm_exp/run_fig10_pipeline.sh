#!/bin/bash
# Fig.10 (pipeline ablation) measured INSIDE the vLLM engine, so its bars are
# on the same footing as Fig.6's rather than on the batch-1 torch runtime's.
#
#   usage: bash scripts/vllm_exp/run_fig10_pipeline.sh
#
# The paper's three configurations, mapped onto this integration:
#   w/o Pipe.   phase-aware kernel disaggregation, no micro-batching at all
#               (one ubatch per step, so every hop is exposed)
#   Pipe.       + ping-pong micro-batches (DBO), all streams at equal priority
#               (FLUIDGPU_PRIORITY_STREAMS=0)
#   Pipe.+Prio. + ubid-staggered CUDA stream priorities, the paper's
#               priority-aware scheduling (FLUIDGPU_PRIORITY_STREAMS=1)
#
# Workload is the Fig.6 GT one (spliced Splitwise 4096 in / 384 out, 192
# requests, max_num_seqs 32), so the absolute numbers are directly comparable
# to the Fig.6 GT column. --max-model-len 4480 = 4096 + 384 is what pins the
# prompt cap at 4096: a looser cap silently feeds ~640 more input tokens per
# request, which costs ~12% of output tokens/s at identical total token rate.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/exp_env.sh"
MODEL="$MODEL_GT"
exp_preflight "$MODEL"
O=$S/fig10_pipeline; mkdir -p "$O"
DRV=fluidgpu_runtime/examples/fluidgpu_torch/vllm_kernel_disagg.py
UDS=datasets/splitwise_spliced_gptoss_4096_384.jsonl
NP=${FLUIDGPU_FIG10_N:-192}
# Repeats matter here: the headline is that the priority lever measures NOTHING
# outside this host's ~2.5% day-to-day drift, and a single run cannot say that.
REPEATS=${FLUIDGPU_FIG10_REPEATS:-3}

cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  pkill -9 -f vllm_kernel_disagg 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
  sleep 5; true
}

run() {  # tag  prio  extra-driver-args...
  local tag=$1 prio=$2; shift 2
  cleanup
  echo "######## $tag (priority_streams=$prio) : $(date +%H:%M:%S) ########"
  env CUDA_VISIBLE_DEVICES=$GPU_PAIR FLUIDGPU_FULL_DECODE_GRAPH=1 \
      FLUIDGPU_PRIORITY_STREAMS=$prio \
  timeout -k 30 2400 $PY $DRV \
    --model "$MODEL" --dataset-jsonl $UDS \
    --num-prompts $NP --max-num-seqs 32 --max-model-len 4480 --no-prefix-cache \
    --placement af --phase-aware "$@" \
    --output-json "$O/$tag.json" > "$O/$tag.log" 2>&1
  echo "rc=$?  tput=$(grep -oE '\"output_tok_s\"[: ]*[0-9.]+' "$O/$tag.json" 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)"
}

for r in $(seq 1 "$REPEATS"); do
  run "pipe_off_r$r"      0
  run "pipe_naive_r$r"    0 --pingpong
  run "pipe_priority_r$r" 1 --pingpong
done

echo "######## results : $(date +%H:%M:%S) ########"
{
  echo "arm,repeat,output_tok_s,pipeline,priority_streams,date,notes"
  for r in $(seq 1 "$REPEATS"); do
    for arm in pipe_off pipe_naive pipe_priority; do
      tput=$(grep -oE '"output_tok_s"[: ]*[0-9.]+' "$O/${arm}_r$r.json" 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)
      case $arm in
        pipe_off)      pipe=off;    prio=0 ;;
        pipe_naive)    pipe=naive;  prio=0 ;;
        pipe_priority) pipe=naive;  prio=1 ;;
      esac
      echo "$arm,$r,${tput:-},$pipe,$prio,$(date +%F),GT 4096/384 ${NP}req ms32 phase-aware+full-decode-graph"
    done
  done
} | tee "$O/results.csv"
cleanup
echo "FIG10_PIPELINE_DONE"
