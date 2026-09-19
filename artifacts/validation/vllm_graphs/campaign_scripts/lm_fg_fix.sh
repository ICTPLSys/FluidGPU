#!/bin/bash
# LM campaign2 repair: re-run the arms killed by the vocab bug (driver ids
# [256,199000) were gpt-oss-sized; llama-3.1 vocab=128256 -> every request 400).
# Driver now derives vocab_high from the model's config.json (llama=128000,
# gpt-oss unchanged 199000). Also hardens the pd_steady readiness gate: the old
# gate trusted the HTTP status code, but the proxy listens instantly and
# returns 200-header + truncated payload while engines are still loading
# (TransferEncodingError x256 in 4s). New gate requires a COMPLETE completion
# JSON ("choices" in body, curl rc=0).
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
IN=1536
OUTL=1024
O=$S/lm_campaign2; mkdir -p "$O"

cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
  pkill -9 -f mooncake_connector_proxy 2>/dev/null; pkill -9 -f 'bench serve' 2>/dev/null
  sleep 6; true
}

# ---- LM decoupled FG at 1536/1024 (vocab-fixed driver) ----
cleanup
echo "######## LM decoupled pair up : $(date +%H:%M:%S) ########"
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
  timeout -k 10 300 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
    --num-prompts 4 --input-len $IN --output-len 16 --concurrency 2 \
    > "$O/fg_smoke.out" 2>&1
  grep -aE 'ok ===|errors' "$O/fg_smoke.out" | head -2
  if grep -aq ' 4/4 ok' "$O/fg_smoke.out"; then
    for cfg in "64 r1" "64 r2" "256 r3"; do
      set -- $cfg
      echo "######## LM fg 256req ${IN}/${OUTL} conc=$1 ($2): $(date +%H:%M:%S) ########"
      timeout -k 20 2400 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
        --num-prompts 256 --input-len $IN --output-len $OUTL --concurrency $1 \
        > "$O/fg_$2_conc$1.out" 2>&1
      grep -aE 'ok ===|wall=|errors' "$O/fg_$2_conc$1.out" | head -3
    done
  else
    echo "LM_FG_SMOKE_FAILED — skipping bench arms"
    tail -12 "$O/fg_smoke.out"
  fi
else
  echo "LM_PAIR_NOT_READY"
  grep -aiE 'error|Traceback' "$O/prefill.log" "$O/decode.log" | tail -4
fi
cleanup

# ---- LM PD-steady, identical client (hardened readiness gate) ----
echo "######## LM PD-steady stack : $(date +%H:%M:%S) ########"
FLUIDGPU_RUN_OUTPUT_DIR="$O/pd_steady" \
MODEL="$LLAMA" VLLM_BIN="$VB" PYTHON_BIN=$V MODE=serve \
PREFILL_GPUS=2 DECODE_GPUS=1 \
PREFILL_MOONCAKE_DEVICES=mlx5_5 DECODE_MOONCAKE_DEVICES=mlx5_2 \
DECODE_MAX_SEQS=32 PREFILL_MAX_SEQS=32 \
TIMEOUT_SECONDS=3600 \
bash scripts/run_pd_baseline_vllm.sh > "$O/pd_steady_stack.log" 2>&1 &
STACK_PID=$!
pd_ready=0
for i in $(seq 1 130); do
  body=$(curl -s -m 8 -X POST http://127.0.0.1:8030/v1/completions \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$LLAMA\",\"prompt\":[1,2,3],\"max_tokens\":1}" 2>/dev/null)
  rc=$?
  if [ "$rc" = "0" ] && printf '%s' "$body" | grep -q '"choices"'; then
    pd_ready=1; echo "LM pd proxy ready+complete (${i}x4s)"; break
  fi
  sleep 4
done
if [ "$pd_ready" = "1" ]; then
  FG_D_URL=http://127.0.0.1:8030/v1/completions \
  timeout -k 20 2400 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
    --num-prompts 256 --input-len $IN --output-len $OUTL --concurrency 64 \
    --local-prefill-frac 1.0 \
    > "$O/pd_steady_conc64.out" 2>&1
  grep -aE 'ok ===|wall=|errors' "$O/pd_steady_conc64.out" | head -3
else
  echo "LM_PD_STEADY_NOT_READY"
  tail -6 "$O/pd_steady_stack.log"
fi
kill -9 $STACK_PID 2>/dev/null
cleanup
echo "LM_FG_FIX_DONE"
