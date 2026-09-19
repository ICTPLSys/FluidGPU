#!/bin/bash
# LM shape calibration (user: adjust input/output length to align with the
# paper's LM fig6 absolutes: homoA100 1290 / homoL40S 890, ratio 1.45).
# Anchors: 2048/64 -> 1448/1159 (overshoot, ratio 1.25); 4096/1024 -> 857/526
# (undershoot, ratio 1.63). Probe intermediate shapes; A100 and L40S run
# CONCURRENTLY (separate cards). Driver truncates prompts to
# (max-model-len - out - 8), so input length is set via --max-model-len.
set -u
S=/tmp/fluidgpu-scratchpad
cd $HOME/workspace/FluidGPU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
V=$HOME/.python/vllm0.18/bin/python
LLAMA=$HOME/.cache/fluidgpu/models/meta-llama--Llama-3.1-8B-Instruct
DRV=fluidgpu_runtime/examples/fluidgpu_torch/vllm_kernel_disagg.py
O=$S/lm_shape_cal; mkdir -p "$O"

cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  sleep 5; true
}

one() {  # tag cvd dataset mml
  CUDA_VISIBLE_DEVICES=$2 timeout -k 30 1800 $V $DRV \
    --model $LLAMA --dataset-jsonl "$3" \
    --num-prompts 192 --max-num-seqs 32 --max-model-len "$4" --no-prefix-cache \
    --placement none \
    --output-json "$O/$1.json" > "$O/$1.log" 2>&1
  echo "rc_$1=$?"
}

shape() {  # name dataset mml
  cleanup
  echo "######## shape $1 (mml=$3) : $(date +%H:%M:%S) ########"
  one "$1_a100" 1 "$2" "$3" &
  P1=$!
  one "$1_l40s" 2 "$2" "$3" &
  P2=$!
  wait $P1 $P2
  for c in a100 l40s; do
    echo "$1-$c tput=$(grep -oE '"output_tok_s"[: ]*[0-9.]+' "$O/$1_$c.json" 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)"
  done
}

shape in2048_out128 "$S/lm_shape_out128.jsonl" 2184
shape in2048_out256 "$S/lm_shape_out256.jsonl" 2312
shape in3072_out256 "$S/lm_shape_out256.jsonl" 3336
shape in1024_out128 "$S/lm_shape_out128.jsonl" 1160
cleanup
echo "LM_SHAPE_CAL_DONE"
