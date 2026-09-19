#!/bin/bash
# P2P follow-up 2 (form E): PD pair WITHIN the A100s over NCCL P2P.
#   P = A100#0 (GPU0) kv_producer :8600, D = A100#1 (GPU1) kv_consumer :8700,
#   P2pNcclConnector (the §25 asset that failed on the SYS/hetero pair) with
#   NCCL P2P *enabled* (22 GB/s measured). Predicted ~1600-1700 (A100
#   prefill-bound: 17716 tok/s x 384/4096) — dominated by form A; the value is
#   (a) proving the connector works over real P2P, (b) the KV-handoff path
#   without any RDMA NIC.
set -u
S=/tmp/fluidgpu-scratchpad
cd $HOME/workspace/FluidGPU
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy ALL_PROXY all_proxy
export NO_PROXY="localhost,127.0.0.1,0.0.0.0,$(hostname -I 2>/dev/null | tr ' ' ',' | sed 's/,$//')"
export no_proxy="$NO_PROXY"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID
export NCCL_CUMEM_ENABLE=0 NCCL_DEBUG=WARN
VB=$HOME/.python/vllm0.18/bin/vllm
PY=$HOME/.python/vllm0.18/bin/python
MODEL=$HOME/.cache/fluidgpu/models/openai--gpt-oss-20b
O=$S/formE; mkdir -p "$O"

cleanup() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null; sleep 6; true
}
cleanup

CUDA_VISIBLE_DEVICES=0 $VB serve "$MODEL" --port 8600 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_port":"21001","kv_connector_extra_config":{"http_port":8600}}' \
  > "$O/prefill.log" 2>&1 &
CUDA_VISIBLE_DEVICES=1 $VB serve "$MODEL" --port 8700 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_port":"22001","kv_connector_extra_config":{"http_port":8700}}' \
  > "$O/decode.log" 2>&1 &

ready=0
for i in $(seq 1 110); do
  a=$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8600/health 2>/dev/null)
  b=$(curl -s -m 2 -o /dev/null -w "%{http_code}" http://127.0.0.1:8700/health 2>/dev/null)
  [ "$a" = "200" ] && [ "$b" = "200" ] && { ready=1; echo "formE ready (${i}x3s)"; break; }
  sleep 3
done
[ "$ready" = "1" ] || { echo "FORME_NOT_READY"; grep -aiE 'error|Traceback' "$O/prefill.log" | tail -3; cleanup; echo FORME_DONE; exit 1; }

IP=$(grep -aoE 'zmq_address:[0-9.]+:21001' "$O/prefill.log" | head -1 | sed 's/zmq_address://; s/:21001//')
echo "engine zmq IP=$IP"
sed -i "s/^IP = \".*\"/IP = \"$IP\"/" "$S/decoupled_driver.py"

echo "######## formE smoke (4 req out=16) ########"
timeout -k 10 240 $PY -u "$S/decoupled_driver.py" --num-prompts 4 --output-len 16 --concurrency 2 \
  > "$O/smoke.out" 2>&1
grep -aE 'ok|wall=|errors|HTTP' "$O/smoke.out" | tail -3
grep -aiE 'NCCL error|unhandled cuda|Traceback' "$O/prefill.log" "$O/decode.log" | tail -2

echo "######## formE bench (256 req 4096/384 conc64) : $(date +%H:%M:%S) ########"
timeout -k 20 1800 $PY -u "$S/decoupled_driver.py" --num-prompts 256 --output-len 384 --concurrency 64 \
  > "$O/bench.out" 2>&1
grep -aE 'ok|wall=|e2e|steady|errors' "$O/bench.out" | tail -4
cleanup
echo "FORME_DONE"
