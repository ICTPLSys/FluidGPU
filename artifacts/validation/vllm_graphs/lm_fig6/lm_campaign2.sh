#!/bin/bash
# LM full campaign v2 at the CALIBRATED shape in1536/out1024 (frontier winner:
# homo A100 1416.9 (+9.8% vs paper 1290) / L40S 822.4 (-7.6% vs 890), mean 8.7%;
# the paper's 1.45 card ratio is unreachable in bf16 — ours is pinned at ~1.7,
# consistent with §17's fp8 hypothesis; disclosed).
# homo rows = the calibration probes (identical protocol) — not rerun here.
# Arms: lever A + AF (offline driver) -> decoupled FG (first LM measurement)
# -> stock PD (inf + conc64) -> PD-steady same-client.
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
MML=2568
IN=1536
OUTL=1024
O=$S/lm_campaign2; mkdir -p "$O"

cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
  pkill -9 -f mooncake_connector_proxy 2>/dev/null; pkill -9 -f 'bench serve' 2>/dev/null
  sleep 6; true
}

offline() {  # tag extra-args...
  local tag=$1; shift
  cleanup
  echo "######## $tag : $(date +%H:%M:%S) ########"
  CUDA_VISIBLE_DEVICES=1,2 FLUIDGPU_FULL_DECODE_GRAPH=1 timeout -k 30 2400 $V $DRV \
    --model $LLAMA --dataset-jsonl $UDS \
    --num-prompts 192 --max-num-seqs 32 --max-model-len $MML --no-prefix-cache \
    --output-json "$O/$tag.json" "$@" > "$O/$tag.log" 2>&1
  echo "rc=$?  tput=$(grep -oE '"output_tok_s"[: ]*[0-9.]+' "$O/$tag.json" 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)"
}

offline lm_leverA --placement af --phase-aware --pingpong
offline lm_af --placement af --pingpong

# ---- LM decoupled FG at 1536/1024 ----
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
  for cfg in "64 r1" "64 r2" "256 r3"; do
    set -- $cfg
    echo "######## LM fg 256req ${IN}/${OUTL} conc=$1 ($2): $(date +%H:%M:%S) ########"
    timeout -k 20 2400 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
      --num-prompts 256 --input-len $IN --output-len $OUTL --concurrency $1 \
      > "$O/fg_$2_conc$1.out" 2>&1
    grep -aE 'ok ===|wall=|errors' "$O/fg_$2_conc$1.out" | head -3
  done
else
  echo "LM_PAIR_NOT_READY"
  grep -aiE 'error|Traceback' "$O/prefill.log" "$O/decode.log" | tail -4
fi
cleanup

# ---- LM stock PD (inf + conc64) ----
for pdcfg in "pd_inf" "pd_c64"; do
  [ "$pdcfg" = "pd_c64" ] && export BENCH_MAX_CONCURRENCY=64 || unset BENCH_MAX_CONCURRENCY
  echo "######## LM stock PD ($pdcfg) : $(date +%H:%M:%S) ########"
  FLUIDGPU_RUN_OUTPUT_DIR="$O/$pdcfg" \
  MODEL="$LLAMA" VLLM_BIN="$VB" PYTHON_BIN=$V \
  PREFILL_GPUS=2 DECODE_GPUS=1 \
  PREFILL_MOONCAKE_DEVICES=mlx5_5 DECODE_MOONCAKE_DEVICES=mlx5_2 \
  BENCH_RANDOM_INPUT_LEN=$IN BENCH_RANDOM_OUTPUT_LEN=$OUTL BENCH_IGNORE_EOS=1 \
  BENCH_NUM_PROMPTS=256 BENCH_REQUEST_RATE=inf \
  DECODE_MAX_SEQS=32 PREFILL_MAX_SEQS=32 \
  TIMEOUT_SECONDS=2400 \
  bash scripts/run_pd_baseline_vllm.sh > "$O/$pdcfg.log" 2>&1
  echo "rc=$?  PD_tput=$(grep -oE '"throughput_tok_s"[: ]*[0-9.]+' "$O/$pdcfg/summary.json" 2>/dev/null | grep -oE '[0-9.]+' | tail -1)"
  cleanup
done
unset BENCH_MAX_CONCURRENCY

# ---- LM PD-steady, identical client ----
echo "######## LM PD-steady stack : $(date +%H:%M:%S) ########"
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
    --num-prompts 256 --input-len $IN --output-len $OUTL --concurrency 64 \
    --local-prefill-frac 1.0 \
    > "$O/pd_steady_conc64.out" 2>&1
  grep -aE 'ok ===|wall=|errors' "$O/pd_steady_conc64.out" | head -3
else
  echo "LM_PD_STEADY_NOT_READY"
fi
cleanup
echo "LM_CAMPAIGN2_DONE"
