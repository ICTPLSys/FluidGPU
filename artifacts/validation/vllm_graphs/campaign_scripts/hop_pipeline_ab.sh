#!/bin/bash
# Pipelined-hop validation + A/B.
#  1. byte-exactness smoke of _PinnedStage._transfer_pipelined (multi-size,
#     odd tails, slot-reuse across calls) on the real GPU pair
#  2. e2e lever A with FLUIDGPU_HOP_PIPELINE_MB=4 (and 2), ms128 + ms32
#  refs: pinned baseline 1452.9 / 1122-1128 | NULL_HOP ceiling 1772 / 1247.
set -u
S=/tmp/fluidgpu-scratchpad
cd $HOME/workspace/FluidGPU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1,2
V=$HOME/.python/vllm0.18/bin/python
DRV=fluidgpu_runtime/examples/fluidgpu_torch/vllm_kernel_disagg.py
GTOSS=$HOME/.cache/fluidgpu/models/openai--gpt-oss-20b
UDS=datasets/splitwise_spliced_gptoss_4096_384.jsonl
O=$S/hop_pipeline_ab; mkdir -p "$O"

cleanup() {
  pkill -9 -f vllm_kernel_disagg 2>/dev/null
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u | xargs -r kill -9 2>/dev/null
  sleep 5; true
}
cleanup

echo "######## smoke: byte exactness ########"
FLUIDGPU_HOP_PIPELINE_MB=4 PYTHONPATH=fluidgpu_runtime timeout 300 $V - <<'EOF'
import os, torch
from fluidgpu_torch.vllm_integration import _PinnedStage

assert _PinnedStage._pipeline_bytes == 4 << 20, _PinnedStage._pipeline_bytes
stage = _PinnedStage.get()
s01 = torch.cuda.Stream(device="cuda:0")
s10 = torch.cuda.Stream(device="cuda:1")
ok = 0
torch.manual_seed(0)
# sizes: below threshold, exact multiple, odd tail, big; bf16 + fp32; reuse keys
for rep in range(3):
    for shape, dtype in [((1024, 512), torch.bfloat16),      # 1 MiB: whole-tensor path
                         ((4096, 2880), torch.bfloat16),     # 23.6 MB, odd tail
                         ((8192, 1024), torch.float32),      # 32 MiB, exact multiple
                         ((4096, 2881), torch.bfloat16)]:    # odd row width
        t = torch.randn(shape, dtype=dtype, device="cuda:0")
        got = stage.transfer(t, "cuda:1", key=("smoke", shape, dtype), src_stream=s01, dst_stream=s10)
        s10.synchronize(); s01.synchronize()
        ref = t.cpu()
        assert got.device.type == "cuda" and got.device.index == 1
        assert torch.equal(got.cpu(), ref), f"MISMATCH {shape} {dtype} rep{rep}"
        ok += 1
print(f"SMOKE_OK {ok} transfers byte-exact")
EOF
rc=$?; echo "smoke rc=$rc"
if [ $rc -ne 0 ]; then echo "HOP_PIPELINE_SMOKE_FAILED"; exit 1; fi

run() {
  local tag=$1 ms=$2 np=$3 mb=$4; shift 4
  cleanup
  echo "######## $tag (ms=$ms np=$np pipeline=${mb}MB) : $(date +%H:%M:%S) ########"
  FLUIDGPU_FULL_DECODE_GRAPH=1 FLUIDGPU_HOP_PIPELINE_MB=$mb timeout -k 30 1800 $V $DRV \
    --model $GTOSS --dataset-jsonl $UDS \
    --num-prompts $np --max-num-seqs $ms --max-model-len 5128 --no-prefix-cache \
    --placement af --phase-aware --pingpong \
    --output-json "$O/$tag.json" "$@" > "$O/$tag.log" 2>&1
  echo "rc=$?  tput=$(grep -oE '"output_tok_s"[: ]*[0-9.]+' "$O/$tag.json" 2>/dev/null | grep -oE '[0-9.]+$' | tail -1)"
}

run pp4_ms128 128 192 4
run pp2_ms128 128 192 2
run pp4_ms32  32  192 4
# parity record (byte-exact copies, but keep the receipt)
run pp4_parity8 32 8 4 --dump-texts "$O/pp4_parity8.texts.json"
$V - <<'EOF'
import json
a = json.load(open("/tmp/fluidgpu-scratchpad/leverA_fix_ab/parity_base8.texts.json"))
b = json.load(open("/tmp/fluidgpu-scratchpad/hop_pipeline_ab/pp4_parity8.texts.json"))
print(f"PP_PARITY {sum(1 for x,y in zip(a,b) if x==y)}/{len(a)}")
EOF
cleanup
echo "HOP_PIPELINE_AB_DONE"
