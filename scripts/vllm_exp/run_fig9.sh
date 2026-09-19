#!/bin/bash
# fig9 (2xA100 + 1xL40S), paper values: PD 2987 / AF 3144 / Kernel-Dis 4253.
# Arms (same client, same seed-1234 random-token 4096/384 workload, 384 req):
#   fg2   : 2-card reference   P(L40S)+D1(A100#1), conc64
#   pd3   : paper PD-3 config  P(L40S)+D1+D2(A100#0, shares $HCA_D), conc128,
#           driver round-robins decodes (--d-url2)
#   fg3a  : FG-3 form A = 3-GPU phase plan: (P->D1 pair) || HOMO(A100#0 full
#           stack), driver routes phi of requests whole to HOMO, conc96,
#           phi in {0.30,0.35,0.40} (analytic optimum ~1004/2856=0.35)
# Topology facts: A100<->A100 has NO P2P and no 200G fabric ($HCA_D/3 are both
# A100-side) => 2-prefiller form B is closed; form A needs zero new transport.
set -u
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/exp_env.sh"
MODEL="$MODEL_GT"
exp_preflight "$MODEL"
O=$S/fig9_3card; mkdir -p "$O"
D2_PID=

cleanup_all() {
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | sort -u); do kill -9 "$p" 2>/dev/null; done
  pkill -9 -f 'vllm serve' 2>/dev/null; pkill -9 -f EngineCore 2>/dev/null; sleep 6; true
}
kill_gpu0() {
  local used
  [[ -n "$D2_PID" ]] || { echo "kill_gpu0: D2 PID is unknown" >&2; return 1; }

  # nvidia-smi exposes host PIDs inside this container, which cannot be passed
  # to kill(1) in the container PID namespace. D2 is therefore launched in its
  # own session; signal that whole process group so the API server and its
  # EngineCore child both exit.
  kill -TERM -- "-$D2_PID" 2>/dev/null || true
  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null |
      awk -F', ' '$1=="0"{print $2}')
    if [ "${used:-999999}" -lt 1024 ]; then
      wait "$D2_PID" 2>/dev/null || true
      echo "GPU0 released (${used} MiB used)"
      return 0
    fi
    sleep 1
  done
  kill -KILL -- "-$D2_PID" 2>/dev/null || true

  for i in $(seq 1 30); do
    used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null |
      awk -F', ' '$1=="0"{print $2}')
    if [ "${used:-999999}" -lt 1024 ]; then
      wait "$D2_PID" 2>/dev/null || true
      echo "GPU0 released (${used} MiB used)"
      return 0
    fi
    sleep 1
  done
  echo "kill_gpu0: GPU0 did not release (used=${used:-unknown} MiB)" >&2
  return 1
}
wait_health() {  # port timeout_iters
  for i in $(seq 1 "$2"); do
    [ "$(curl -s -m 2 -o /dev/null -w "%{http_code}" "http://127.0.0.1:$1/health" 2>/dev/null)" = "200" ] && return 0
    sleep 3
  done
  return 1
}
bench() {  # tag conc extra-args...
  local tag=$1 conc=$2; shift 2
  echo "######## $tag (conc=$conc) : $(date +%H:%M:%S) ########"
  timeout -k 20 1800 $PY -u "$EXP_DIR/rdma_driver.py" --model "$MODEL" \
    --num-prompts 384 --output-len 384 --concurrency "$conc" "$@" \
    > "$O/$tag.out" 2>&1
  grep -aE 'ok ===|wall=|errors' "$O/$tag.out" | head -3
}

cleanup_all
echo "######## launching P(L40S) + D1(A100#1) : $(date +%H:%M:%S) ########"
MC_MS_AUTO_DISC=0 MOONCAKE_DEVICE=$HCA_P VLLM_MOONCAKE_BOOTSTRAP_PORT=8998 \
CUDA_VISIBLE_DEVICES=$GPU_P $VB serve "$MODEL" --port 8100 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  --kv-transfer-config "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_producer\",\"kv_connector_extra_config\":{\"device_name\":\"$HCA_P\"}}" \
  > "$O/prefill.log" 2>&1 &
MC_MS_AUTO_DISC=0 MOONCAKE_DEVICE=$HCA_D \
CUDA_VISIBLE_DEVICES=$GPU_D $VB serve "$MODEL" --port 8200 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  --kv-transfer-config "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"device_name\":\"$HCA_D\"}}" \
  > "$O/decode1.log" 2>&1 &
wait_health 8100 110 && wait_health 8200 60 || { echo "PD1_STACK_NOT_READY"; cleanup_all; echo FIG9_DONE; exit 1; }
echo "P+D1 ready"

bench fg2_384 64

echo "######## launching D2 (A100#0, $HCA_D, :8300) ########"
MC_MS_AUTO_DISC=0 MOONCAKE_DEVICE=$HCA_D \
CUDA_VISIBLE_DEVICES=$GPU_X setsid "$VB" serve "$MODEL" --port 8300 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  --kv-transfer-config "{\"kv_connector\":\"MooncakeConnector\",\"kv_role\":\"kv_consumer\",\"kv_connector_extra_config\":{\"device_name\":\"$HCA_D\"}}" \
  > "$O/decode2.log" 2>&1 &
D2_PID=$!
wait_health 8300 110 || { echo "D2_NOT_READY"; cleanup_all; echo FIG9_DONE; exit 1; }
echo "D2 ready"
echo "######## pd3 smoke (8 req) ########"
timeout -k 10 300 $PY -u "$EXP_DIR/rdma_driver.py" --model "$MODEL" --num-prompts 8 --output-len 16 --concurrency 4 \
  --d-url2 http://127.0.0.1:8300/v1/completions > "$O/pd3_smoke.out" 2>&1
grep -aE 'ok ===|errors' "$O/pd3_smoke.out" | head -2

bench pd3_384 128 --d-url2 http://127.0.0.1:8300/v1/completions

echo "######## swapping GPU0: D2 -> HOMO stock (:8400) ########"
kill_gpu0 || { echo "GPU0_NOT_RELEASED"; cleanup_all; echo FIG9_DONE; exit 1; }
CUDA_VISIBLE_DEVICES=$GPU_X $VB serve "$MODEL" --port 8400 --tensor-parallel-size 1 \
  --max_num_batched_tokens 16384 --max-num-seqs 32 --no-enable-prefix-caching \
  > "$O/homo.log" 2>&1 &
wait_health 8400 110 || { echo "HOMO_NOT_READY"; cleanup_all; echo FIG9_DONE; exit 1; }
echo "HOMO ready"
echo "######## fg3a smoke (8 req, phi=0.5) ########"
timeout -k 10 300 $PY -u "$EXP_DIR/rdma_driver.py" --model "$MODEL" --num-prompts 8 --output-len 16 --concurrency 4 \
  --local-prefill-frac 0.5 --local-url http://127.0.0.1:8400/v1/completions > "$O/fg3a_smoke.out" 2>&1
grep -aE 'ok ===|errors' "$O/fg3a_smoke.out" | head -2

for phi in 0.30 0.35 0.40; do
  bench "fg3a_phi${phi}" 96 --local-prefill-frac "$phi" \
    --local-url http://127.0.0.1:8400/v1/completions
done

cleanup_all
echo "FIG9_DONE"
