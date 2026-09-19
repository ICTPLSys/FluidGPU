#!/bin/bash
# Hybrid peak probe at in1920: phi=0.17 (left neighbor of the current peak
# 0.22=2049; right side declines 0.28=2001, 0.33=1703; phi=0 is 1550).
set -u
S=/tmp/fluidgpu-scratchpad
O=$S/lm_campaign3
cd $HOME/workspace/FluidGPU
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"; export no_proxy="$NO_PROXY"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
V=$HOME/.python/vllm0.18/bin/python
VB=$HOME/.python/vllm0.18/bin/vllm
LLAMA=$HOME/.cache/fluidgpu/models/meta-llama--Llama-3.1-8B-Instruct
cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null
  sleep 6; true
}
cleanup
echo "######## pair up : $(date +%H:%M:%S) ########"
MC_MS_AUTO_DISC=0 MOONCAKE_DEVICE=mlx5_5 VLLM_MOONCAKE_BOOTSTRAP_PORT=8998 \
CUDA_VISIBLE_DEVICES=2 $VB serve "$LLAMA" --port 8100 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer","kv_connector_extra_config":{"device_name":"mlx5_5"}}' \
  > "$O/p17_prefill.log" 2>&1 &
MC_MS_AUTO_DISC=0 MOONCAKE_DEVICE=mlx5_2 \
CUDA_VISIBLE_DEVICES=1 $VB serve "$LLAMA" --port 8200 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer","kv_connector_extra_config":{"device_name":"mlx5_2"}}' \
  > "$O/p17_decode.log" 2>&1 &
ready=0
for i in $(seq 1 110); do
  a=$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8100/health 2>/dev/null)
  b=$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8200/health 2>/dev/null)
  [ "$a" = "200" ] && [ "$b" = "200" ] && { ready=1; echo "pair ready (${i}x3s)"; break; }
  sleep 3
done
if [ "$ready" = "1" ]; then
  echo "######## hybrid phi=0.17 (256req conc64) : $(date +%H:%M:%S) ########"
  timeout -k 20 2400 $V -u "$S/rdma_driver.py" --model "$LLAMA" \
    --num-prompts 256 --input-len 1920 --output-len 1024 --concurrency 64 \
    --local-prefill-frac 0.17 --local-url http://127.0.0.1:8100/v1/completions \
    > "$O/h_phi0.17.out" 2>&1
  grep -aE 'ok ===|wall=' "$O/h_phi0.17.out" | head -2
else
  echo "P17_NOT_READY"
fi
cleanup
echo "H_PEAK17_DONE"
