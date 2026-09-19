#!/bin/bash
# LM full campaign v3 at the re-calibrated shape (user: homo A100 must be at
# or below the paper's 1290; v2's 1536/1024 put it at +9.8%). Shape comes in
# via env: IN (input len), MML (=IN+1024+8). Homo rows = the cal4 probes.
# Arms: offline (leverA, AF, bal035) -> Mooncake pair (FG r1/r2 conc64 +
# hybrid phi sweep PHI1/2/3) -> stock PD (inf, c64) -> PD-steady same-client
# -> dual-homo strict control (phi_req=PHID, n_l40s=NLD, conc 32/32).
set -u
: "${IN:?set IN}"; : "${MML:?set MML}"
: "${PHI1:?}"; : "${PHI2:?}"; : "${PHI3:?}"; : "${PHID:?}"; : "${NLD:?}"
OUTL=1024
S=/tmp/fluidgpu-scratchpad
O=$S/lm_campaign3; mkdir -p "$O"
cd $HOME/workspace/FluidGPU
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"; export no_proxy="$NO_PROXY"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
V=$HOME/.python/vllm0.18/bin/python
VB=$HOME/.python/vllm0.18/bin/vllm
LLAMA=$HOME/.cache/fluidgpu/models/meta-llama--Llama-3.1-8B-Instruct
UDS=datasets/splitwise_spliced_llama_4096_1024.jsonl
DRV=fluidgpu_runtime/examples/fluidgpu_torch/vllm_kernel_disagg.py
NA=$((256 - NLD))

cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
  pkill -9 -f mooncake_connector_proxy 2>/dev/null; pkill -9 -f 'bench serve' 2>/dev/null
  sleep 6; true
}

offline() {  # tag env-pairs... -- extra-args...
  local tag=$1; shift
  local envs=()
  while [ "$1" != "--" ]; do envs+=("$1"); shift; done
  shift
  cleanup
  echo "######## $tag : $(date +%H:%M:%S) ########"
  env CUDA_VISIBLE_DEVICES=1,2 FLUIDGPU_FULL_DECODE_GRAPH=1 "${envs[@]}" \
  timeout -k 30 2400 $V $DRV \
    --model $LLAMA --dataset-jsonl $UDS \
    --num-prompts 192 --max-num-seqs 32 --max-model-len $MML --no-prefix-cache \
    --output-json "$O/$tag.json" "$@" > "$O/$tag.log" 2>&1
  echo "rc=$?  tput=$(grep -oE '"output_tok_s"[: ]*[0-9.]+' "$O/$tag.json" 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)"
}

offline lm_leverA NOOP=1 -- --placement af --phase-aware --pingpong
offline lm_af NOOP=1 -- --placement af --pingpong
offline lm_bal035 FLUIDGPU_BALANCED=1 FLUIDGPU_LOCAL_FFN_FRAC=0.35 -- --placement af --phase-aware --pingpong

# ---- Mooncake pair: FG decoupled + hybrid sweep ----
cleanup
echo "######## pair up (P=L40S:8100, D=A100:8200) : $(date +%H:%M:%S) ########"
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
  [ "$a" = "200" ] && [ "$b" = "200" ] && { ready=1; echo "pair ready (${i}x3s)"; break; }
  sleep 3
done
if [ "$ready" = "1" ]; then
  timeout -k 10 300 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
    --num-prompts 4 --input-len $IN --output-len 16 --concurrency 2 \
    > "$O/fg_smoke.out" 2>&1
  grep -aE 'ok ===|errors' "$O/fg_smoke.out" | head -2
  if grep -aq ' 4/4 ok' "$O/fg_smoke.out"; then
    for r in r1 r2; do
      echo "######## FG decoupled ${IN}/${OUTL} conc64 ($r) : $(date +%H:%M:%S) ########"
      timeout -k 20 2400 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
        --num-prompts 256 --input-len $IN --output-len $OUTL --concurrency 64 \
        > "$O/fg_${r}_conc64.out" 2>&1
      grep -aE 'ok ===|wall=' "$O/fg_${r}_conc64.out" | head -2
    done
    for PHI in $PHI1 $PHI2 $PHI3; do
      echo "######## hybrid phi=$PHI (256req conc64) : $(date +%H:%M:%S) ########"
      timeout -k 20 2400 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
        --num-prompts 256 --input-len $IN --output-len $OUTL --concurrency 64 \
        --local-prefill-frac $PHI --local-url http://127.0.0.1:8100/v1/completions \
        > "$O/h_phi${PHI}.out" 2>&1
      grep -aE 'ok ===|wall=' "$O/h_phi${PHI}.out" | head -2
    done
  else
    echo "FG_SMOKE_FAILED"; tail -8 "$O/fg_smoke.out"
  fi
else
  echo "PAIR_NOT_READY"
fi
cleanup

# ---- stock PD (inf + conc64) ----
for pdcfg in "pd_inf" "pd_c64"; do
  [ "$pdcfg" = "pd_c64" ] && export BENCH_MAX_CONCURRENCY=64 || unset BENCH_MAX_CONCURRENCY
  echo "######## stock PD ($pdcfg) : $(date +%H:%M:%S) ########"
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

# ---- PD-steady, identical client (hardened gate) ----
echo "######## PD-steady stack : $(date +%H:%M:%S) ########"
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
    pd_ready=1; echo "pd proxy ready+complete (${i}x4s)"; break
  fi
  sleep 4
done
if [ "$pd_ready" = "1" ]; then
  FG_D_URL=http://127.0.0.1:8030/v1/completions \
  timeout -k 20 2400 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
    --num-prompts 256 --input-len $IN --output-len $OUTL --concurrency 64 \
    --local-prefill-frac 1.0 \
    > "$O/pd_steady_conc64.out" 2>&1
  grep -aE 'ok ===|wall=' "$O/pd_steady_conc64.out" | head -2
else
  echo "PD_STEADY_NOT_READY"
fi
kill -9 $STACK_PID 2>/dev/null
cleanup

# ---- dual-homo strict control (conc 32/32) ----
echo "######## dual-homo pair up : $(date +%H:%M:%S) ########"
CUDA_VISIBLE_DEVICES=1 $VB serve "$LLAMA" --port 8300 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  > "$O/dh_a100_serve.log" 2>&1 &
CUDA_VISIBLE_DEVICES=2 $VB serve "$LLAMA" --port 8400 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  > "$O/dh_l40s_serve.log" 2>&1 &
ready=0
for i in $(seq 1 110); do
  a=$(curl -s -m 3 -X POST http://127.0.0.1:8300/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$LLAMA\",\"prompt\":[1,2,3],\"max_tokens\":1}" 2>/dev/null)
  b=$(curl -s -m 3 -X POST http://127.0.0.1:8400/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$LLAMA\",\"prompt\":[1,2,3],\"max_tokens\":1}" 2>/dev/null)
  case "$a" in *'"choices"'*) case "$b" in *'"choices"'*) ready=1; echo "dual-homo ready (${i}x3s)"; break;; esac;; esac
  sleep 3
done
if [ "$ready" = "1" ]; then
  echo "######## dual-homo phi=$PHID conc32/32 (A100 n=$NA | L40S n=$NLD) : $(date +%H:%M:%S) ########"
  T0=$(date +%s.%N)
  FG_D_URL=http://127.0.0.1:8300/v1/completions \
  timeout -k 20 1500 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
    --num-prompts $NA --input-len $IN --output-len $OUTL --concurrency 32 \
    --local-prefill-frac 1.0 > "$O/dh_a100.out" 2>&1 &
  PA=$!
  FG_D_URL=http://127.0.0.1:8400/v1/completions \
  timeout -k 20 1500 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
    --num-prompts $NLD --input-len $IN --output-len $OUTL --concurrency 32 \
    --local-prefill-frac 1.0 > "$O/dh_l40s.out" 2>&1 &
  PL=$!
  wait $PA; wait $PL
  T1=$(date +%s.%N)
  WALL=$(echo "$T1 $T0" | awk '{printf "%.1f", $1-$2}')
  echo "combined_wall=${WALL}s  combined_e2e=$(echo "$WALL" | awk '{printf "%.0f", 262144/$1}')"
  grep -aE 'ok ===|wall=' "$O/dh_a100.out" | head -2
  grep -aE 'ok ===|wall=' "$O/dh_l40s.out" | head -2
else
  echo "DUAL_HOMO_NOT_READY"
fi
cleanup
echo "LM_CAMPAIGN3_DONE"
