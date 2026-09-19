#!/bin/bash
# Fig.6 `Request Dist.` baseline (camera-ready): two INDEPENDENT vLLM replicas,
# one per GPU of the heterogeneous pair, with the client acting as the load
# balancer. No KV transfer, no cross-GPU kernel execution -- each complete
# request runs on exactly one GPU.
#
#   usage: bash scripts/vllm_exp/run_fig6_request_distribution.sh [gt|lm]
#
# Shapes match the FG bar of each model so the two are the same workload on the
# same two GPUs (the comparison the paper's Fig.6 text makes):
#   gt  gpt-oss-20b        4096 in / 684  out, 256 prompts, total conc 64
#   lm  Llama-3.1-8B-Inst. 1920 in / 1024 out, 256 prompts, total conc 64
# Each replica is served with --max-num-seqs 32, i.e. the same batch depth as
# the homogeneous / PD / FG instances.
#
# Policies are swept and the BEST is the Request Dist. bar ("best-performing
# load-balancing policy among those evaluated", paper §V-A).
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/exp_env.sh"

WHICH="${1:-gt}"
case "$WHICH" in
  gt) MODEL="$MODEL_GT"; IN=4096; OUTL=684;  TMO=1200 ;;
  lm) MODEL="$MODEL_LM"; IN=1920; OUTL=1024; TMO=2400 ;;
  *)  echo "usage: $0 [gt|lm]" >&2; exit 2 ;;
esac
NPROMPTS="${FLUIDGPU_REQDIST_N:-256}"
CONC="${FLUIDGPU_REQDIST_CONC:-64}"
exp_preflight "$MODEL"
O=$S/reqdist_$WHICH; mkdir -p "$O"

cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
  sleep 6; true
}
cleanup

echo "######## replicas up (A100=GPU$GPU_D:8300, L40S=GPU$GPU_P:8400) : $(date +%H:%M:%S) ########"
CUDA_VISIBLE_DEVICES=$GPU_D $VB serve "$MODEL" --port 8300 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  > "$O/a100_serve.log" 2>&1 &
CUDA_VISIBLE_DEVICES=$GPU_P $VB serve "$MODEL" --port 8400 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  > "$O/l40s_serve.log" 2>&1 &

ready=0
for i in $(seq 1 130); do
  a=$(curl -s -m 3 -X POST http://127.0.0.1:8300/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"prompt\":[1,2,3],\"max_tokens\":1}" 2>/dev/null)
  b=$(curl -s -m 3 -X POST http://127.0.0.1:8400/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"prompt\":[1,2,3],\"max_tokens\":1}" 2>/dev/null)
  case "$a" in *'"choices"'*) case "$b" in *'"choices"'*) ready=1; echo "replicas ready (${i}x3s)"; break;; esac;; esac
  sleep 3
done
if [ "$ready" != "1" ]; then
  echo "REPLICAS_NOT_READY"
  grep -aiE 'error|Traceback' "$O/a100_serve.log" | tail -3
  grep -aiE 'error|Traceback' "$O/l40s_serve.log" | tail -3
  cleanup; echo "REQDIST_${WHICH}_DONE"; exit 1
fi

echo "######## smoke (4 req out=16) ########"
timeout -k 10 300 $PY -u "$EXP_DIR/reqdist_driver.py" --model "$MODEL" \
  --num-prompts 4 --input-len $IN --output-len 16 --concurrency 2 \
  > "$O/smoke.out" 2>&1
grep -aE 'ok ===|wall=|errors' "$O/smoke.out" | head -3

# Single-replica controls with the IDENTICAL client, so the Request Dist. row
# can be reported as a fraction of the sum of the two single-GPU rates measured
# the same way (the paper's "approximately 85% of the sum" statistic).
for side in a100 l40s; do
  [ "$side" = a100 ] && U=http://127.0.0.1:8300/v1/completions || U=http://127.0.0.1:8400/v1/completions
  echo "######## single $side (n=$((NPROMPTS / 2)) conc $((CONC / 2))) : $(date +%H:%M:%S) ########"
  timeout -k 20 $TMO $PY -u "$EXP_DIR/reqdist_driver.py" --model "$MODEL" \
    --urls "$U" --labels "$side" --policy round-robin \
    --num-prompts $((NPROMPTS / 2)) --input-len $IN --output-len $OUTL \
    --concurrency $((CONC / 2)) > "$O/single_$side.out" 2>&1
  grep -aE 'wall=' "$O/single_$side.out" | head -1
done

for pol in least-outstanding round-robin static; do
  extra=""
  [ "$pol" = static ] && extra="--static-frac ${FLUIDGPU_REQDIST_FRAC:-0.5}"
  echo "######## reqdist $pol (n=$NPROMPTS conc $CONC) : $(date +%H:%M:%S) ########"
  timeout -k 20 $TMO $PY -u "$EXP_DIR/reqdist_driver.py" --model "$MODEL" \
    --policy "$pol" $extra \
    --num-prompts $NPROMPTS --input-len $IN --output-len $OUTL --concurrency $CONC \
    > "$O/reqdist_$pol.out" 2>&1
  grep -aE 'ok ===|A100:|L40S:|wall=|errors' "$O/reqdist_$pol.out" | head -6
done

grep -aiE 'error|Traceback' "$O/a100_serve.log" | tail -2
grep -aiE 'error|Traceback' "$O/l40s_serve.log" | tail -2
cleanup
echo "REQDIST_${WHICH}_DONE"
