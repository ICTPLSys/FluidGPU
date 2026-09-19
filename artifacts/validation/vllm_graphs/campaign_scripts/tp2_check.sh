#!/bin/bash
# P2P follow-up 1: stock TP=2 across the A100 pair (now that P2P@22GB/s is
# confirmed). This pins the one free parameter in every form-B/PD-3'/AF-3
# capacity argument: how well TP2 scales over PCIe P2P on this pair.
#   refs: single A100 (bs32, 4096/384) = 1004 | fig9 form A = 2949 steady.
# Same client/seed/metric as all of today's rows (plain completions via
# --local-prefill-frac 1.0).
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
O=$S/tp2_check; mkdir -p "$O"

cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null; sleep 6; true
}
cleanup

echo "######## TP2 homo (A100 pair, P2P) launching : $(date +%H:%M:%S) ########"
CUDA_VISIBLE_DEVICES=0,1 $VB serve "$MODEL" --port 8500 --tensor-parallel-size 2 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  > "$O/tp2.log" 2>&1 &
ready=0
for i in $(seq 1 140); do
  [ "$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8500/health 2>/dev/null)" = "200" ] && { ready=1; echo "tp2 ready (${i}x3s)"; break; }
  sleep 3
done
[ "$ready" = "1" ] || { echo "TP2_NOT_READY"; grep -aiE 'error|Traceback' "$O/tp2.log" | tail -4; cleanup; echo TP2_DONE; exit 1; }

nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits -lms 100 \
  > "$O/tp2.duty.csv" 2>&1 &
SMI=$!
for conc in 64 32; do
  echo "######## tp2_conc$conc (384 req 4096/384) : $(date +%H:%M:%S) ########"
  FG_D_URL=http://127.0.0.1:8500/v1/completions \
  timeout -k 20 1800 $PY -u "$S/rdma_driver.py" \
    --num-prompts 384 --output-len 384 --concurrency $conc --local-prefill-frac 1.0 \
    > "$O/tp2_conc$conc.out" 2>&1
  grep -aE 'ok ===|wall=|errors' "$O/tp2_conc$conc.out" | head -3
done
kill -9 $SMI 2>/dev/null
grep -aiE 'P2P|NVLink|shm' "$O/tp2.log" | grep -ai nccl | head -3
cleanup
echo "TP2_DONE"
