#!/bin/bash
# H3: dual-homo strict-parity control — total client conc 64 (32/32), matching
# the decoupled/PD arms' conc64 exactly. (H2's 40/40=80 total could be argued
# a client-side edge; engine seats are 2x32 in every arm.)
set -u
S=/tmp/fluidgpu-scratchpad
O=$S/lm_kernel_layout
cd $HOME/workspace/FluidGPU
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"; export no_proxy="$NO_PROXY"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
V=$HOME/.python/vllm0.18/bin/python
VB=$HOME/.python/vllm0.18/bin/vllm
LLAMA=$HOME/.cache/fluidgpu/models/meta-llama--Llama-3.1-8B-Instruct
cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
  sleep 6; true
}
cleanup
echo "######## H3 dual-homo pair up : $(date +%H:%M:%S) ########"
CUDA_VISIBLE_DEVICES=1 $VB serve "$LLAMA" --port 8300 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  > "$O/h3_a100_serve.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 $VB serve "$LLAMA" --port 8400 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  > "$O/h3_l40s_serve.log" 2>&1 &
ready=0
for i in $(seq 1 110); do
  a=$(curl -s -m 3 -X POST http://127.0.0.1:8300/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$LLAMA\",\"prompt\":[1,2,3],\"max_tokens\":1}" 2>/dev/null)
  b=$(curl -s -m 3 -X POST http://127.0.0.1:8400/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$LLAMA\",\"prompt\":[1,2,3],\"max_tokens\":1}" 2>/dev/null)
  case "$a" in *'"choices"'*) case "$b" in *'"choices"'*) ready=1; echo "H3 ready (${i}x3s)"; break;; esac;; esac
  sleep 3
done
if [ "$ready" = "1" ]; then
  echo "######## H3 dual-homo phi=0.367 conc32/32 (A100 n=162 | L40S n=94) : $(date +%H:%M:%S) ########"
  T0=$(date +%s.%N)
  FG_D_URL=http://127.0.0.1:8300/v1/completions \
  timeout -k 20 1200 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
    --num-prompts 162 --input-len 1536 --output-len 1024 --concurrency 32 \
    --local-prefill-frac 1.0 > "$O/h3_a100.out" 2>&1 &
  PA=$!
  FG_D_URL=http://127.0.0.1:8400/v1/completions \
  timeout -k 20 1200 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
    --num-prompts 94 --input-len 1536 --output-len 1024 --concurrency 32 \
    --local-prefill-frac 1.0 > "$O/h3_l40s.out" 2>&1 &
  PL=$!
  wait $PA; wait $PL
  T1=$(date +%s.%N)
  WALL=$(echo "$T1 $T0" | awk '{printf "%.1f", $1-$2}')
  echo "combined_wall=${WALL}s  combined_e2e=$(echo "$WALL" | awk '{printf "%.0f", 262144/$1}')"
  grep -aE 'ok ===|wall=' "$O/h3_a100.out" | head -2
  grep -aE 'ok ===|wall=' "$O/h3_l40s.out" | head -2
else
  echo "H3_NOT_READY"
fi
cleanup
echo "PHASE_H3_DONE"
