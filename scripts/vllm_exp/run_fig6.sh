#!/bin/bash
# Fig.6 end-to-end entry point: both LLM columns, GT first (shorter) then LM.
#
#   GT (gpt-oss-20b, 4096/684): FluidGPU phase-level decoupled + hybrid overflow
#     against a same-day stock-PD pair, every policy at max_num_seqs=32 and the
#     same client at conc64 -> gt_decoupled/results.csv
#   LM (Llama-3.1-8B, 1920/1024): the full v3 ladder (offline configurations,
#     FG decoupled, hybrid phi sweep, stock PD, PD-steady, dual-homo control)
#     -> fig6_v3/results.csv
#
# The `Request Dist.` bars of the same figure come from the separate
# run_fig6_request_distribution.sh (one independent replica per GPU); they are
# not part of the FluidGPU-vs-baseline comparison this script measures.
#
# Machine bindings (interpreter, model paths, GPU indices, RDMA HCAs) are
# centralized in exp_env.sh. Raw per-run outputs land under $FLUIDGPU_RERUN_OUT.
set -u
EXP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

rc_gt=0
rc_lm=0

echo "######## Fig.6 GT (gpt-oss-20b): $(date +%H:%M:%S) ########"
bash "$EXP_DIR/run_fig6_gptoss20b.sh" || rc_gt=$?

echo "######## Fig.6 LM (Llama-3.1-8B): $(date +%H:%M:%S) ########"
bash "$EXP_DIR/run_fig6_llama31.sh" || rc_lm=$?

echo "######## Fig.6 done: GT rc=$rc_gt LM rc=$rc_lm : $(date +%H:%M:%S) ########"
[ "$rc_gt" -eq 0 ] && [ "$rc_lm" -eq 0 ]
