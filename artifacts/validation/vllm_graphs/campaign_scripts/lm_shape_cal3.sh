#!/bin/bash
set -u
S=/tmp/fluidgpu-scratchpad
cd $HOME/workspace/FluidGPU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
V=$HOME/.python/vllm0.18/bin/python
LLAMA=$HOME/.cache/fluidgpu/models/meta-llama--Llama-3.1-8B-Instruct
DRV=fluidgpu_runtime/examples/fluidgpu_torch/vllm_kernel_disagg.py
O=$S/lm_shape_cal; mkdir -p "$O"
cleanup() { for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done; sleep 5; true; }
one() { CUDA_VISIBLE_DEVICES=$2 timeout -k 30 2400 $V $DRV \
  --model $LLAMA --dataset-jsonl datasets/splitwise_spliced_llama_4096_1024.jsonl \
  --num-prompts 192 --max-num-seqs 32 --max-model-len "$3" --no-prefix-cache \
  --placement none --output-json "$O/$1.json" > "$O/$1.log" 2>&1; }
cleanup
echo "######## shape in1536_out1024 (mml=2568) : $(date +%H:%M:%S) ########"
one in1536_out1024_a100 1 2568 & P1=$!
one in1536_out1024_l40s 2 2568 & P2=$!
wait $P1 $P2
for c in a100 l40s; do
  echo "in1536_out1024-$c tput=$(grep -oE '"output_tok_s"[: ]*[0-9.]+' "$O/in1536_out1024_$c.json" 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)"
done
cleanup
echo "LM_SHAPE_CAL3_DONE"
