#!/bin/bash
# Phase H of the LM kernel-layout study — the resource model's top cross-engine
# candidates:
#  H1 hybrid: Mooncake P/D pair + phi fraction of requests routed WHOLE to the
#     P engine (plain completion, no KV transfer; fig9 form A machinery).
#     A100 stays a pure decoder (ceiling 1690); the L40S's ~83% idle is filled.
#     Analytic phi* = 0.28, ideal 2349. Sweep phi {0.22, 0.28, 0.34}, conc64.
#  H2 dual-homo saturation control: per-side conc 40/40 (each engine ms32
#     saturated; the phi sweep at split-conc<32 undersaturated the L40S).
set -u
S=/tmp/fluidgpu-scratchpad
O=$S/lm_kernel_layout
cd $HOME/workspace/FluidGPU
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
export no_proxy="$NO_PROXY"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
V=$HOME/.python/vllm0.18/bin/python
VB=$HOME/.python/vllm0.18/bin/vllm
LLAMA=$HOME/.cache/fluidgpu/models/meta-llama--Llama-3.1-8B-Instruct

cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
  sleep 6; true
}

# ---- H1: Mooncake pair + whole-request overflow to P ----
cleanup
echo "######## H1 pair up (P=L40S:8100 producer, D=A100:8200 consumer) : $(date +%H:%M:%S) ########"
MC_MS_AUTO_DISC=0 MOONCAKE_DEVICE=mlx5_5 VLLM_MOONCAKE_BOOTSTRAP_PORT=8998 \
CUDA_VISIBLE_DEVICES=2 $VB serve "$LLAMA" --port 8100 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"device_name":"mlx5_5"}}' \
  > "$O/h_prefill.log" 2>&1 &
MC_MS_AUTO_DISC=0 MOONCAKE_DEVICE=mlx5_2 \
CUDA_VISIBLE_DEVICES=1 $VB serve "$LLAMA" --port 8200 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"device_name":"mlx5_2"}}' \
  > "$O/h_decode.log" 2>&1 &
ready=0
for i in $(seq 1 110); do
  a=$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8100/health 2>/dev/null)
  b=$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8200/health 2>/dev/null)
  [ "$a" = "200" ] && [ "$b" = "200" ] && { ready=1; echo "H1 pair ready (${i}x3s)"; break; }
  sleep 3
done
if [ "$ready" = "1" ]; then
  echo "-- smoke: plain completion on the P engine (producer serving homo) --"
  pc=$(curl -s -m 60 -X POST http://127.0.0.1:8100/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$LLAMA\",\"prompt\":[1,2,3,4],\"max_tokens\":8,\"ignore_eos\":true}" 2>/dev/null)
  case "$pc" in *'"choices"'*) echo "P_PLAIN_OK";; *) echo "P_PLAIN_FAIL: ${pc:0:160}";; esac
  timeout -k 10 300 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
    --num-prompts 8 --input-len 1536 --output-len 16 --concurrency 4 \
    --local-prefill-frac 0.25 --local-url http://127.0.0.1:8100/v1/completions \
    > "$O/h_smoke.out" 2>&1
  grep -aE 'ok ===|errors' "$O/h_smoke.out" | head -2
  if grep -aq ' 8/8 ok' "$O/h_smoke.out"; then
    for PHI in 0.22 0.28 0.34; do
      echo "######## H1 hybrid phi=$PHI (256req 1536/1024 conc64) : $(date +%H:%M:%S) ########"
      timeout -k 20 2400 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
        --num-prompts 256 --input-len 1536 --output-len 1024 --concurrency 64 \
        --local-prefill-frac $PHI --local-url http://127.0.0.1:8100/v1/completions \
        > "$O/h_phi${PHI}.out" 2>&1
      grep -aE 'ok ===|frac=|wall=|errors' "$O/h_phi${PHI}.out" | head -4
    done
  else
    echo "H1_SMOKE_FAILED"; tail -8 "$O/h_smoke.out"
  fi
else
  echo "H1_PAIR_NOT_READY"
fi
cleanup

# ---- H2: dual-homo, both sides saturated (conc 40/40) ----
echo "######## H2 dual-homo pair up : $(date +%H:%M:%S) ########"
CUDA_VISIBLE_DEVICES=1 $VB serve "$LLAMA" --port 8300 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  > "$O/h2_a100_serve.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 $VB serve "$LLAMA" --port 8400 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  > "$O/h2_l40s_serve.log" 2>&1 &
ready=0
for i in $(seq 1 110); do
  a=$(curl -s -m 3 -X POST http://127.0.0.1:8300/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$LLAMA\",\"prompt\":[1,2,3],\"max_tokens\":1}" 2>/dev/null)
  b=$(curl -s -m 3 -X POST http://127.0.0.1:8400/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$LLAMA\",\"prompt\":[1,2,3],\"max_tokens\":1}" 2>/dev/null)
  case "$a" in *'"choices"'*) case "$b" in *'"choices"'*) ready=1; echo "H2 ready (${i}x3s)"; break;; esac;; esac
  sleep 3
done
if [ "$ready" = "1" ]; then
  echo "######## H2 dual-homo phi=0.367 conc40/40 (A100 n=162 | L40S n=94) : $(date +%H:%M:%S) ########"
  T0=$(date +%s.%N)
  FG_D_URL=http://127.0.0.1:8300/v1/completions \
  timeout -k 20 1200 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
    --num-prompts 162 --input-len 1536 --output-len 1024 --concurrency 40 \
    --local-prefill-frac 1.0 > "$O/h2_a100.out" 2>&1 &
  PA=$!
  FG_D_URL=http://127.0.0.1:8400/v1/completions \
  timeout -k 20 1200 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
    --num-prompts 94 --input-len 1536 --output-len 1024 --concurrency 40 \
    --local-prefill-frac 1.0 > "$O/h2_l40s.out" 2>&1 &
  PL=$!
  wait $PA; wait $PL
  T1=$(date +%s.%N)
  WALL=$(echo "$T1 $T0" | awk '{printf "%.1f", $1-$2}')
  echo "combined_wall=${WALL}s  combined_e2e=$(echo "$WALL" | awk '{printf "%.0f", 262144/$1}')"
  grep -aE 'ok ===|wall=' "$O/h2_a100.out" | head -2
  grep -aE 'ok ===|wall=' "$O/h2_l40s.out" | head -2
else
  echo "H2_NOT_READY"
fi
cleanup
echo "PHASE_H_DONE"
