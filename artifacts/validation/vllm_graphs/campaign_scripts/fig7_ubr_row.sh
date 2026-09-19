#!/bin/bash
# fig7 v2 extra row: fg_ubr = lever-A online + pingpong threshold 1024 +
# UBATCH_PHASE_ROUTE. Mechanism: online mixed steps carry ~2075-token prefills
# (< the 4096 gate) and run the SERIAL per-layer AF chain — the residual gap
# vs homo (5.4 vs 6.2-7.1 crossing). Lowering the gate lets those steps
# micro-batch (two ~1k halves ping-pong => chain overlapped); §39's
# UBATCH_PHASE_ROUTE keeps the decode rows LOCAL inside ubatch0, removing the
# §16-era poison (decode rows hopping + doubled expert reads).
set -u
S=/tmp/fluidgpu-scratchpad
cd $HOME/workspace/FluidGPU
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,$(hostname -I 2>/dev/null | tr ' ' ',' | sed 's/,$//')"
export no_proxy="$NO_PROXY"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
V=$HOME/.python/vllm0.18/bin/python
VLLM_BIN=$HOME/.python/vllm0.18/bin/vllm
GPTOSS=$HOME/.cache/fluidgpu/models/openai--gpt-oss-20b
DS=datasets/conversation_benchmark/requests.jsonl
UDS=datasets/splitwise_spliced_gptoss_4096_384.jsonl
OUT=artifacts/validation/vllm_graphs/fig7_gt_v2
LOG=$OUT/logs
RUNTIME_ROOT=$PWD/fluidgpu_runtime
EXAMPLES_DIR=$PWD/fluidgpu_runtime/examples/fluidgpu_torch
DRV=fluidgpu_runtime/examples/fluidgpu_torch/vllm_kernel_disagg.py
PORT=8977
RATES="2 4 6 7 8 10 12"

CC_FULL=$(FLUIDGPU_FULL_DECODE_GRAPH=1 FLUIDGPU_PHASE_AWARE=1 $V -c "import json,sys; sys.path.insert(0,'$RUNTIME_ROOT'); from fluidgpu_torch.vllm_integration import fluid_compilation_config; print(json.dumps(fluid_compilation_config()))")

cleanup_gpu() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 $p 2>/dev/null; done
  pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
  sleep 6
}
cleanup_gpu

echo "######## fg_ubr parity gate (8 prompts, threshold 1024) ########"
FLUIDGPU_FULL_DECODE_GRAPH=1 FLUIDGPU_UBATCH_PHASE_ROUTE=1 \
FLUIDGPU_PINGPONG_PREFILL_THRESHOLD=1024 \
timeout -k 30 1200 $V $DRV \
  --model $GPTOSS --dataset-jsonl $UDS \
  --num-prompts 8 --max-num-seqs 32 --max-model-len 5128 --no-prefix-cache \
  --placement af --phase-aware --phase-route --pingpong \
  --dump-texts "$OUT/fg_ubr_parity8.texts.json" \
  --output-json "$OUT/fg_ubr_parity8.json" > "$LOG/ubr_parity.log" 2>&1
$V - <<'EOF'
import json
a = json.load(open("/tmp/fluidgpu-scratchpad/leverA_fix_ab/parity_base8.texts.json"))
b = json.load(open("$HOME/workspace/FluidGPU/artifacts/validation/vllm_graphs/fig7_gt_v2/fg_ubr_parity8.texts.json"))
m = sum(1 for x, y in zip(a, b) if x == y)
print(f"FG_UBR_PARITY {m}/{len(a)}")
EOF
if ! grep -q "FG_UBR_PARITY 8/8" <($V - <<'EOF'
import json
a = json.load(open("/tmp/fluidgpu-scratchpad/leverA_fix_ab/parity_base8.texts.json"))
b = json.load(open("$HOME/workspace/FluidGPU/artifacts/validation/vllm_graphs/fig7_gt_v2/fg_ubr_parity8.texts.json"))
m = sum(1 for x, y in zip(a, b) if x == y)
print(f"FG_UBR_PARITY {m}/{len(a)}")
EOF
); then
  echo "FG_UBR_PARITY_FAILED — aborting row"
  cleanup_gpu; echo FIG7_UBR_DONE; exit 1
fi
cleanup_gpu

echo "######## ROW fg_ubr : $(date +%H:%M:%S) ########"
VLLM_LOGGING_LEVEL=WARNING CUDA_VISIBLE_DEVICES=1,2 \
PYTHONPATH="$RUNTIME_ROOT:$EXAMPLES_DIR" \
FLUIDGPU_VLLM_PLACEMENT=af FLUIDGPU_EXPERT_DEVICE=cuda:1 FLUIDGPU_HOP=copy \
FLUIDGPU_PINGPONG=1 FLUIDGPU_PHASE_AWARE=1 FLUIDGPU_PHASE_ROUTE=1 \
FLUIDGPU_FULL_DECODE_GRAPH=1 FLUIDGPU_UBATCH_PHASE_ROUTE=1 \
FLUIDGPU_PINGPONG_PREFILL_THRESHOLD=1024 \
$VLLM_BIN serve "$GPTOSS" --port $PORT \
  --max-model-len 4096 --gpu-memory-utilization 0.85 --no-enable-prefix-caching \
  --worker-cls fluidgpu_torch.vllm_integration.FluidGPUWorker \
  --compilation-config "$CC_FULL" > "$LOG/fg_ubr_serve.log" 2>&1 &
ready=0
for i in $(seq 1 130); do
  [ "$(curl -s -m 2 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health 2>/dev/null)" = "200" ] && { ready=1; break; }
  sleep 4
done
[ "$ready" = "1" ] || { echo "FG_UBR_SERVE_FAILED"; grep -aiE 'error|Traceback' "$LOG/fg_ubr_serve.log" | tail -3; cleanup_gpu; echo FIG7_UBR_DONE; exit 1; }
for R in $RATES; do
  N=192; [ "${R%%.*}" -gt 6 ] && N=320
  echo "--- fg_ubr rate=$R n=$N: $(date +%H:%M:%S) ---"
  timeout -k 30 900 $VLLM_BIN bench serve --host 127.0.0.1 --port $PORT \
    --backend vllm --model "$GPTOSS" --seed 0 \
    --dataset-name custom --dataset-path "$DS" \
    --custom-output-len -1 --skip-chat-template --ignore-eos \
    --num-prompts $N --request-rate $R \
    --save-result --result-dir "$OUT" \
    --result-filename "fg_ubr_r${R}.json" \
    > "$LOG/fg_ubr_r${R}.log" 2>&1
  grep -E 'Median TPOT' "$LOG/fg_ubr_r${R}.log" | head -1
done
cleanup_gpu
echo "FIG7_UBR_DONE"
