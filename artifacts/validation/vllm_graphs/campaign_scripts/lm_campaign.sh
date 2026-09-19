#!/bin/bash
# LM (Llama-3.1-8B-Instruct) full fig6-row campaign — today's GT methodology
# transplanted: same-day homo x2 + lever A re-ref (offline driver, spliced
# 4096/1024, 192 req, ms32) then decoupled FG (NEVER measured for LM) with
# same-day PD pairing + conc64 control + dual metrics, identical client.
# Standing refs (07-12/13 era): homo 857/526, AF 618, lever A 868, PD(swap) 1055.
# LM is DECODE-heavy (out/in = 0.25): PD here is decode-bound (TPOT ~28ms
# earlier) — the FG-vs-PD outcome need not mirror GT's; that is the point.
set -u
S=/tmp/fluidgpu-scratchpad
cd $HOME/workspace/FluidGPU
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,$(hostname -I 2>/dev/null | tr ' ' ',' | sed 's/,$//')"
export no_proxy="$NO_PROXY"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
V=$HOME/.python/vllm0.18/bin/python
VB=$HOME/.python/vllm0.18/bin/vllm
LLAMA=$HOME/.cache/fluidgpu/models/meta-llama--Llama-3.1-8B-Instruct
UDS=datasets/splitwise_spliced_llama_4096_1024.jsonl
DRV=fluidgpu_runtime/examples/fluidgpu_torch/vllm_kernel_disagg.py
O=$S/lm_campaign; mkdir -p "$O"

cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
  pkill -9 -f mooncake_connector_proxy 2>/dev/null; pkill -9 -f 'bench serve' 2>/dev/null
  sleep 6; true
}

offline() {  # tag cvd extra-args...
  local tag=$1 cvd=$2; shift 2
  cleanup
  echo "######## $tag : $(date +%H:%M:%S) ########"
  CUDA_VISIBLE_DEVICES=$cvd FLUIDGPU_FULL_DECODE_GRAPH=1 timeout -k 30 2400 $V $DRV \
    --model $LLAMA --dataset-jsonl $UDS \
    --num-prompts 192 --max-num-seqs 32 --max-model-len 5128 --no-prefix-cache \
    --output-json "$O/$tag.json" "$@" > "$O/$tag.log" 2>&1
  echo "rc=$?  tput=$(grep -oE '"output_tok_s"[: ]*[0-9.]+' "$O/$tag.json" 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)"
}

# ---- offline rows (same-day re-refs) ----
offline lm_leverA 1,2 --placement af --phase-aware --pingpong
offline lm_homo_a100 1 --placement none
offline lm_homo_l40s 2 --placement none

# ---- LM decoupled FG (first measurement) ----
cleanup
echo "######## LM decoupled: launching P(L40S)+D(A100) : $(date +%H:%M:%S) ########"
MC_MS_AUTO_DISC=0 MOONCAKE_DEVICE=mlx5_5 VLLM_MOONCAKE_BOOTSTRAP_PORT=8998 \
CUDA_VISIBLE_DEVICES=2 $VB serve "$LLAMA" --port 8100 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"device_name":"mlx5_5"}}' \
  > "$O/prefill.log" 2>&1 &
MC_MS_AUTO_DISC=0 MOONCAKE_DEVICE=mlx5_2 \
CUDA_VISIBLE_DEVICES=1 $VB serve "$LLAMA" --port 8200 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"device_name":"mlx5_2"}}' \
  > "$O/decode.log" 2>&1 &
ready=0
for i in $(seq 1 110); do
  a=$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8100/health 2>/dev/null)
  b=$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8200/health 2>/dev/null)
  [ "$a" = "200" ] && [ "$b" = "200" ] && { ready=1; echo "LM pair ready (${i}x3s)"; break; }
  sleep 3
done
if [ "$ready" = "1" ]; then
  echo "######## LM fg smoke ########"
  timeout -k 10 300 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
    --num-prompts 4 --input-len 4096 --output-len 16 --concurrency 2 \
    > "$O/fg_smoke.out" 2>&1
  grep -aE 'ok ===|errors' "$O/fg_smoke.out" | head -2
  for cfg in "64 r1" "64 r2" "256 r3"; do
    set -- $cfg
    echo "######## LM fg bench 256req 4096/1024 conc=$1 ($2): $(date +%H:%M:%S) ########"
    timeout -k 20 2400 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
      --num-prompts 256 --input-len 4096 --output-len 1024 --concurrency $1 \
      > "$O/fg_$2_conc$1.out" 2>&1
    grep -aE 'ok ===|wall=|errors' "$O/fg_$2_conc$1.out" | head -3
  done
else
  echo "LM_PAIR_NOT_READY"
  grep -aiE 'error|Traceback' "$O/prefill.log" "$O/decode.log" | tail -4
fi
cleanup

# ---- LM stock PD pairing (rate=inf + conc64) ----
for pdcfg in "inf pd_inf" "64 pd_c64"; do
  set -- $pdcfg
  RATE_ARGS=""
  [ "$1" != "inf" ] && export BENCH_MAX_CONCURRENCY=$1 || unset BENCH_MAX_CONCURRENCY
  echo "######## LM stock PD ($2) : $(date +%H:%M:%S) ########"
  FLUIDGPU_RUN_OUTPUT_DIR="$O/$2" \
  MODEL="$LLAMA" VLLM_BIN="$VB" PYTHON_BIN=$V \
  PREFILL_GPUS=2 DECODE_GPUS=1 \
  PREFILL_MOONCAKE_DEVICES=mlx5_5 DECODE_MOONCAKE_DEVICES=mlx5_2 \
  BENCH_RANDOM_INPUT_LEN=4096 BENCH_RANDOM_OUTPUT_LEN=1024 BENCH_IGNORE_EOS=1 \
  BENCH_NUM_PROMPTS=256 BENCH_REQUEST_RATE=inf \
  DECODE_MAX_SEQS=32 PREFILL_MAX_SEQS=32 \
  TIMEOUT_SECONDS=2400 \
  bash scripts/run_pd_baseline_vllm.sh > "$O/$2.log" 2>&1
  echo "rc=$?  PD_tput=$(grep -oE '"throughput_tok_s"[: ]*[0-9.]+' "$O/$2/summary.json" 2>/dev/null | grep -oE '[0-9.]+' | tail -1)"
  cleanup
done
unset BENCH_MAX_CONCURRENCY

# ---- LM PD-steady, identical client (proxy, frac=1.0) ----
echo "######## LM PD-steady stack (MODE=serve) : $(date +%H:%M:%S) ########"
FLUIDGPU_RUN_OUTPUT_DIR="$O/pd_steady" \
MODEL="$LLAMA" VLLM_BIN="$VB" PYTHON_BIN=$V MODE=serve \
PREFILL_GPUS=2 DECODE_GPUS=1 \
PREFILL_MOONCAKE_DEVICES=mlx5_5 DECODE_MOONCAKE_DEVICES=mlx5_2 \
DECODE_MAX_SEQS=32 PREFILL_MAX_SEQS=32 \
TIMEOUT_SECONDS=3600 \
bash scripts/run_pd_baseline_vllm.sh > "$O/pd_steady_stack.log" 2>&1 &
pd_ready=0
for i in $(seq 1 130); do
  c=$(curl -s -m 3 -o /dev/null -w "%{http_code}" -X POST http://127.0.0.1:8030/v1/completions \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$LLAMA\",\"prompt\":[1,2,3],\"max_tokens\":1}" 2>/dev/null)
  [ "$c" = "200" ] && { pd_ready=1; echo "LM pd proxy ready (${i}x4s)"; break; }
  sleep 4
done
if [ "$pd_ready" = "1" ]; then
  FG_D_URL=http://127.0.0.1:8030/v1/completions \
  timeout -k 20 2400 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
    --num-prompts 256 --input-len 4096 --output-len 1024 --concurrency 64 \
    --local-prefill-frac 1.0 \
    > "$O/pd_steady_conc64.out" 2>&1
  grep -aE 'ok ===|wall=|errors' "$O/pd_steady_conc64.out" | head -3
else
  echo "LM_PD_STEADY_NOT_READY"
fi
cleanup
echo "LM_CAMPAIGN_DONE"
