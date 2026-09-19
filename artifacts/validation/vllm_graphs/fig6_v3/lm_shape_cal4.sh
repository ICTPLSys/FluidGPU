#!/bin/bash
# LM shape calibration ROUND 4 (user: homo A100 must come DOWN to <= paper's
# 1290; at 1536/1024 it is 1416.9 = +9.8%, the only row ABOVE the paper).
# Grid slope at out=1024: A100(in) ~ 1417 - 0.3125*(in-1536) => 1290 at
# in ~1930-1950. Probe in=1920 (mml 2952) and in=1984 (mml 3016) to bracket.
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

one() {  # tag cvd mml
  CUDA_VISIBLE_DEVICES=$2 timeout -k 30 2400 $V $DRV \
    --model $LLAMA --dataset-jsonl datasets/splitwise_spliced_llama_4096_1024.jsonl \
    --num-prompts 192 --max-num-seqs 32 --max-model-len "$3" --no-prefix-cache \
    --placement none \
    --output-json "$O/$1.json" > "$O/$1.log" 2>&1
  echo "rc_$1=$?"
}

shape() {  # name mml
  cleanup
  echo "######## shape $1 (mml=$2) : $(date +%H:%M:%S) ########"
  one "$1_a100" 1 "$2" &
  P1=$!
  one "$1_l40s" 2 "$2" &
  P2=$!
  wait $P1 $P2
  for c in a100 l40s; do
    echo "$1-$c tput=$(grep -oE '"output_tok_s"[: ]*[0-9.]+' "$O/$1_$c.json" 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)"
  done
}

shape in1920_out1024 2952
shape in1984_out1024 3016
cleanup
echo "LM_SHAPE_CAL4_DONE"
