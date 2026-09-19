#!/bin/bash
# RE-RUN: phase-level decoupled engine (user request), with a SAME-DAY stock-PD
# paired baseline so the comparison is self-contained.
#
#   FG (phase-level decoupling): P=L40S(GPU2)/kv_producer/mlx5_5 http:8100,
#     D=A100(GPU1)/kv_consumer/mlx5_2 http:8200, MooncakeConnector, NO proxy
#     (rdma_driver.py talks to P and D directly). FAIR workload: 256 DISTINCT
#     random-token-id prompts, exactly 4096 in / 384 out, ignore_eos.
#   PD baseline: identical instances + stock HTTP proxy, vllm bench serve,
#     random 4096/384, 256 prompts, rate=inf (pd_tune.sh's exact config).
#
# refs: fair decoupled phi=0 = 1856 (§32) | stock PD = 1815-1827 (§24/§32)
#       | kernel-bound phase-layout ceiling ~2023-2093.
set -u
S=/tmp/fluidgpu-scratchpad
cd $HOME/workspace/FluidGPU
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,$(hostname -I 2>/dev/null | tr ' ' ',' | sed 's/,$//')"
export no_proxy="$NO_PROXY"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
VB=$HOME/.python/vllm0.18/bin/vllm
PY=$HOME/.python/vllm0.18/bin/python
MODEL=$HOME/.cache/fluidgpu/models/openai--gpt-oss-20b
O=$S/decoupled_rerun; mkdir -p "$O"

cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
  pkill -9 -f mooncake_connector_proxy 2>/dev/null; pkill -9 -f 'bench serve' 2>/dev/null
  sleep 6; true
}
cleanup

echo "######## FG decoupled: launching P+D : $(date +%H:%M:%S) ########"
MC_MS_AUTO_DISC=0 MOONCAKE_DEVICE=mlx5_5 VLLM_MOONCAKE_BOOTSTRAP_PORT=8998 \
CUDA_VISIBLE_DEVICES=2 $VB serve "$MODEL" --port 8100 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"device_name":"mlx5_5"}}' \
  > "$O/prefill.log" 2>&1 &
MC_MS_AUTO_DISC=0 MOONCAKE_DEVICE=mlx5_2 \
CUDA_VISIBLE_DEVICES=1 $VB serve "$MODEL" --port 8200 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"device_name":"mlx5_2"}}' \
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
timeout -k 10 240 $PY -u "$S/rdma_driver.py" --num-prompts 4 --output-len 16 --concurrency 2 \
  > "$O/smoke.out" 2>&1
grep -aE 'ok ===|wall=|errors' "$O/smoke.out" | head -3

for cfg in "64 r1" "64 r2" "64 r3" "256 r4"; do
  set -- $cfg
  echo "######## FG bench 256req 4096/384 conc=$1 ($2): $(date +%H:%M:%S) ########"
  timeout -k 20 1800 $PY -u "$S/rdma_driver.py" \
    --num-prompts 256 --output-len 384 --concurrency $1 \
    > "$O/fg_$2_conc$1.out" 2>&1
  grep -aE 'ok ===|wall=|errors' "$O/fg_$2_conc$1.out" | head -3
done
grep -aiE 'NCCL error|unhandled cuda|Traceback|mooncake.*error|transfer.*fail' "$O/prefill.log" | tail -2
grep -aiE 'NCCL error|unhandled cuda|Traceback|mooncake.*error|transfer.*fail' "$O/decode.log" | tail -2
cleanup

echo "######## stock PD paired baseline (proxy, bench serve, rate=inf): $(date +%H:%M:%S) ########"
FLUIDGPU_RUN_OUTPUT_DIR="$O/pd" \
MODEL="$MODEL" VLLM_BIN="$VB" PYTHON_BIN="$PY" \
PREFILL_GPUS=2 DECODE_GPUS=1 \
PREFILL_MOONCAKE_DEVICES=mlx5_5 DECODE_MOONCAKE_DEVICES=mlx5_2 \
BENCH_RANDOM_INPUT_LEN=4096 BENCH_RANDOM_OUTPUT_LEN=384 BENCH_IGNORE_EOS=1 \
BENCH_NUM_PROMPTS=256 BENCH_REQUEST_RATE=inf \
DECODE_MAX_SEQS=32 PREFILL_MAX_SEQS=32 \
TIMEOUT_SECONDS=1200 \
bash scripts/run_pd_baseline_vllm.sh > "$O/pd.log" 2>&1
echo "pd rc=$?  PD_tput=$(grep -oE '"throughput_tok_s"[: ]*[0-9.]+' "$O/pd/summary.json" 2>/dev/null | grep -oE '[0-9.]+' | tail -1)"
grep -aiE 'error|fail|timeout' "$O/pd.log" | grep -aviE 'INFO|reasoning' | tail -3
cleanup
echo "DECOUPLED_RERUN_DONE"
