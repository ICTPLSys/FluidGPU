#!/bin/bash
# Phase E+F of the LM kernel-layout study.
# E: native-prefill probes (out=8 dataset) on stock homo A100 and L40S ->
#    the stock engines' true prefill us/token (anchors the frontier).
# F: DUAL-HOMO layout (the resource model's cross-engine winner): two stock
#    single-card instances (A100 :8300, L40S :8400, NO KV transfer), one
#    rdma_driver process per side in plain-completion mode (frac=1.0), phi
#    fraction of the 256 requests to the L40S. phi* = 822/(1417+822) = 0.367.
#    Sweep phi in {0.30, 0.367, 0.44}, conc split 64*phi.
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
DRV=fluidgpu_runtime/examples/fluidgpu_torch/vllm_kernel_disagg.py
MML=2568

cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
  sleep 6; true
}

# ---- E: native prefill probes ----
for probe in "e_pf_a100 1" "e_pf_l40s 2"; do
  set -- $probe
  cleanup
  echo "######## $1 : $(date +%H:%M:%S) ########"
  CUDA_VISIBLE_DEVICES=$2 timeout -k 30 1200 $V $DRV \
    --model $LLAMA --dataset-jsonl "$O/lm_out8.jsonl" \
    --num-prompts 64 --max-num-seqs 32 --max-model-len $MML --no-prefix-cache \
    --placement none \
    --output-json "$O/$1.json" > "$O/$1.log" 2>&1
  echo "rc=$?  json=$(cat "$O/$1.json" 2>/dev/null | tr -d '\n' | head -c 300)"
done

# ---- F: dual-homo pair ----
cleanup
echo "######## dual-homo pair up : $(date +%H:%M:%S) ########"
CUDA_VISIBLE_DEVICES=1 $VB serve "$LLAMA" --port 8300 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  > "$O/f_a100_serve.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 $VB serve "$LLAMA" --port 8400 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  > "$O/f_l40s_serve.log" 2>&1 &
ready=0
for i in $(seq 1 110); do
  a=$(curl -s -m 3 -X POST http://127.0.0.1:8300/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$LLAMA\",\"prompt\":[1,2,3],\"max_tokens\":1}" 2>/dev/null)
  b=$(curl -s -m 3 -X POST http://127.0.0.1:8400/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$LLAMA\",\"prompt\":[1,2,3],\"max_tokens\":1}" 2>/dev/null)
  case "$a" in *'"choices"'*) case "$b" in *'"choices"'*) ready=1; echo "dual-homo ready+complete (${i}x3s)"; break;; esac;; esac
  sleep 3
done
if [ "$ready" = "1" ]; then
  for cfg in "0.367 94 34 23 41" "0.30 77 32 19 45" "0.44 113 36 28 36"; do
    set -- $cfg
    PHI=$1; NL=$2; TAGL=l40s$2; NA=$((256 - $2)); CL=$4; CA=$5
    echo "######## dual-homo phi=$PHI (A100 n=$NA conc=$CA | L40S n=$NL conc=$CL) : $(date +%H:%M:%S) ########"
    T0=$(date +%s.%N)
    FG_D_URL=http://127.0.0.1:8300/v1/completions \
    timeout -k 20 1200 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
      --num-prompts $NA --input-len 1536 --output-len 1024 --concurrency $CA \
      --local-prefill-frac 1.0 \
      > "$O/f_phi${PHI}_a100.out" 2>&1 &
    PA=$!
    FG_D_URL=http://127.0.0.1:8400/v1/completions \
    timeout -k 20 1200 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
      --num-prompts $NL --input-len 1536 --output-len 1024 --concurrency $CL \
      --local-prefill-frac 1.0 \
      > "$O/f_phi${PHI}_l40s.out" 2>&1 &
    PL=$!
    wait $PA; wait $PL
    T1=$(date +%s.%N)
    WALL=$(echo "$T1 $T0" | awk '{printf "%.1f", $1-$2}')
    echo "combined_wall=${WALL}s  combined_e2e=$(echo "$WALL" | awk '{printf "%.0f", 262144/$1}')"
    grep -aE 'ok ===|wall=' "$O/f_phi${PHI}_a100.out" | head -2
    grep -aE 'ok ===|wall=' "$O/f_phi${PHI}_l40s.out" | head -2
  done
else
  echo "DUAL_HOMO_NOT_READY"
fi
cleanup

# ---- G: balanced-fraction peak search (0.20=1439.7 < 0.35=1472.1, peak right) ----
UDS=datasets/splitwise_spliced_llama_4096_1024.jsonl
for g in "d_bal050 0.50" "d_bal065 0.65"; do
  set -- $g
  cleanup
  echo "######## $1 (n=192) : $(date +%H:%M:%S) ########"
  CUDA_VISIBLE_DEVICES=1,2 FLUIDGPU_FULL_DECODE_GRAPH=1 \
  FLUIDGPU_BALANCED=1 FLUIDGPU_LOCAL_FFN_FRAC=$2 \
  timeout -k 30 2400 $V $DRV \
    --model $LLAMA --dataset-jsonl $UDS \
    --num-prompts 192 --max-num-seqs 32 --max-model-len $MML --no-prefix-cache \
    --placement af --phase-aware --pingpong \
    --output-json "$O/$1.json" > "$O/$1.log" 2>&1
  echo "rc=$?  tput=$(grep -oE '"output_tok_s"[: ]*[0-9.]+' "$O/$1.json" 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)"
done
cleanup
echo "PHASE_EF_DONE"
