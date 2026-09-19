#!/bin/bash
# RE-RUN: phase-level decoupled engine (user request), with a SAME-DAY stock-PD
# paired baseline so the comparison is self-contained.
#
#   FG (phase-level decoupling): P = GPU_P (L40S) / kv_producer / HCA_P on
#     http:8100, D = GPU_D (A100) / kv_consumer / HCA_D on http:8200,
#     MooncakeConnector, NO proxy
#     (rdma_driver.py talks to P and D directly). FAIR workload: 256 DISTINCT
#     random-token-id prompts, exactly 4096 in / 684 out, ignore_eos.
#   PD baseline: identical instances + stock HTTP proxy, vllm bench serve,
#     random 4096/684, 256 prompts, rate=inf.
#
# Every policy, including FluidGPU, uses max_num_seqs=32 and the same
# rdma_driver client at conc64. The selected FluidGPU row is the hybrid at the
# analytic capacity-ratio optimum (phi*=0.184 -> phi=0.18, the steady peak of
# the bs32 phi sweep), 3 repeats. PD steady is measured with the identical
# client, concurrency, and 25%-75% completion window.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/exp_env.sh"
MODEL="$MODEL_GT"
exp_preflight "$MODEL"
IN=4096
OUTL=684
FG_MAX_NUM_SEQS=32
FG_CONCURRENCY=64
FG_LOCAL_PREFILL_FRAC=0.18
BASELINE_MAX_NUM_SEQS=32
BASELINE_CONCURRENCY=64
O=$S/gt684_final; mkdir -p "$O"

cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
  pkill -9 -f mooncake_connector_proxy 2>/dev/null; pkill -9 -f 'bench serve' 2>/dev/null
  sleep 6; true
}
cleanup

echo "######## FG decoupled: launching P+D : $(date +%H:%M:%S) ########"
MC_MS_AUTO_DISC=0 MOONCAKE_DEVICE=$HCA_P VLLM_MOONCAKE_BOOTSTRAP_PORT=8998 \
CUDA_VISIBLE_DEVICES=$GPU_P $VB serve "$MODEL" --port 8100 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs $FG_MAX_NUM_SEQS --no-enable-prefix-caching \
  --kv-transfer-config "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"device_name\":\"$HCA_P\"}}" \
  > "$O/prefill.log" 2>&1 &
MC_MS_AUTO_DISC=0 MOONCAKE_DEVICE=$HCA_D \
CUDA_VISIBLE_DEVICES=$GPU_D $VB serve "$MODEL" --port 8200 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs $FG_MAX_NUM_SEQS --no-enable-prefix-caching \
  --kv-transfer-config "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"device_name\":\"$HCA_D\"}}" \
  > "$O/decode.log" 2>&1 &

ready=0
for i in $(seq 1 110); do
  a=$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8100/health 2>/dev/null)
  b=$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8200/health 2>/dev/null)
  [ "$a" = "200" ] && [ "$b" = "200" ] && { ready=1; echo "both ready (${i}x3s)"; break; }
  sleep 3
done
if [ "$ready" != "1" ]; then
  echo "SERVE_NOT_READY"
  grep -aiE 'error|Traceback' "$O/prefill.log" | tail -3
  grep -aiE 'error|Traceback' "$O/decode.log" | tail -3
  cleanup; echo DECOUPLED_RERUN_DONE; exit 1
fi

echo "######## smoke (4 req out=16) ########"
timeout -k 10 240 $PY -u "$EXP_DIR/rdma_driver.py" --model "$MODEL" \
  --num-prompts 4 --input-len $IN --output-len 16 --concurrency 2 \
  > "$O/smoke.out" 2>&1
grep -aE 'ok ===|wall=|errors' "$O/smoke.out" | head -3

for repeat in r1 r2 r3; do
  echo "######## FG bench 256req ${IN}/${OUTL} conc=$FG_CONCURRENCY ($repeat): $(date +%H:%M:%S) ########"
  timeout -k 20 1800 $PY -u "$EXP_DIR/rdma_driver.py" \
    --model "$MODEL" --num-prompts 256 --input-len $IN --output-len $OUTL \
    --concurrency $FG_CONCURRENCY --local-prefill-frac $FG_LOCAL_PREFILL_FRAC \
    --local-url http://127.0.0.1:8100/v1/completions \
    > "$O/fg_${repeat}_conc${FG_CONCURRENCY}.out" 2>&1
  grep -aE 'ok ===|wall=|errors' "$O/fg_${repeat}_conc${FG_CONCURRENCY}.out" | head -3
done
grep -aiE 'NCCL error|unhandled cuda|Traceback|mooncake.*error|transfer.*fail' "$O/prefill.log" | tail -2
grep -aiE 'NCCL error|unhandled cuda|Traceback|mooncake.*error|transfer.*fail' "$O/decode.log" | tail -2
cleanup

echo "######## stock PD paired baseline (proxy, bench serve, rate=inf): $(date +%H:%M:%S) ########"
FLUIDGPU_RUN_OUTPUT_DIR="$O/pd" \
MODEL="$MODEL" VLLM_BIN="$VB" PYTHON_BIN="$PY" \
PREFILL_GPUS=2 DECODE_GPUS=1 \
PREFILL_MOONCAKE_DEVICES=$HCA_P DECODE_MOONCAKE_DEVICES=$HCA_D \
BENCH_RANDOM_INPUT_LEN=$IN BENCH_RANDOM_OUTPUT_LEN=$OUTL BENCH_IGNORE_EOS=1 \
BENCH_NUM_PROMPTS=256 BENCH_REQUEST_RATE=inf \
DECODE_MAX_SEQS=$BASELINE_MAX_NUM_SEQS PREFILL_MAX_SEQS=$BASELINE_MAX_NUM_SEQS \
TIMEOUT_SECONDS=1200 \
bash scripts/run_pd_baseline_vllm.sh > "$O/pd.log" 2>&1
echo "pd rc=$?  PD_tput=$(grep -oE '"throughput_tok_s"[: ]*[0-9.]+' "$O/pd/summary.json" 2>/dev/null | grep -oE '[0-9.]+' | tail -1)"
grep -aiE 'error|fail|timeout' "$O/pd.log" | grep -aviE 'INFO|reasoning' | tail -3
cleanup

echo "######## stock PD identical-client steady measurement: $(date +%H:%M:%S) ########"
GT_STEADY_OUT="$O/pd_steady" IN=$IN OUTL=$OUTL NUM_PROMPTS=256 \
CONCURRENCY=$BASELINE_CONCURRENCY MAX_NUM_SEQS=$BASELINE_MAX_NUM_SEQS \
bash scripts/vllm_exp/run_fig6_gptoss20b_steady.sh

echo "DECOUPLED_RERUN_DONE"
