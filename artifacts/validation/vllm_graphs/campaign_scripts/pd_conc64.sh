#!/bin/bash
# Final control: stock PD with max-concurrency 64 — the SAME client admission
# FG's best arm used. Settles whether FG@conc64's +4.2% (1852 vs PD@inf 1778)
# is architecture or client-side admission control.
set -u
S=/tmp/fluidgpu-scratchpad
cd $HOME/workspace/FluidGPU
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,$(hostname -I 2>/dev/null | tr ' ' ',' | sed 's/,$//')"
export no_proxy="$NO_PROXY"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
O=$S/decoupled_rerun; mkdir -p "$O"

cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
  pkill -9 -f mooncake_connector_proxy 2>/dev/null; pkill -9 -f 'bench serve' 2>/dev/null
  sleep 6; true
}
cleanup
echo "######## stock PD conc=64 control : $(date +%H:%M:%S) ########"
FLUIDGPU_RUN_OUTPUT_DIR="$O/pd_conc64" \
MODEL=$HOME/.cache/fluidgpu/models/openai--gpt-oss-20b \
VLLM_BIN=$HOME/.python/vllm0.18/bin/vllm \
PYTHON_BIN=$HOME/.python/vllm0.18/bin/python \
PREFILL_GPUS=2 DECODE_GPUS=1 \
PREFILL_MOONCAKE_DEVICES=mlx5_5 DECODE_MOONCAKE_DEVICES=mlx5_2 \
BENCH_RANDOM_INPUT_LEN=4096 BENCH_RANDOM_OUTPUT_LEN=384 BENCH_IGNORE_EOS=1 \
BENCH_NUM_PROMPTS=256 BENCH_REQUEST_RATE=inf BENCH_MAX_CONCURRENCY=64 \
DECODE_MAX_SEQS=32 PREFILL_MAX_SEQS=32 \
TIMEOUT_SECONDS=1200 \
bash scripts/run_pd_baseline_vllm.sh > "$O/pd_conc64.log" 2>&1
echo "pd_conc64 rc=$?  PD_tput=$(grep -oE '"throughput_tok_s"[: ]*[0-9.]+' "$O/pd_conc64/summary.json" 2>/dev/null | grep -oE '[0-9.]+' | tail -1)"
grep -aiE 'error|fail|timeout' "$O/pd_conc64.log" | grep -aviE 'INFO|reasoning' | tail -3
cleanup
echo "PD_CONC64_DONE"
