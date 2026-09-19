# Experiment Scripts — Raw Measurement Records

These 37 scripts are the drivers **as they were run** on the authors'
evaluation host to produce the tables in `docs/EXPERIMENTS.md`. They are kept
as a record of what was executed, so every shipped number can be traced to the
command that produced it. The only edit applied to them is that the authors'
home directory has been written as `$HOME` (`~` in captured output) rather
than its absolute path.

**They are not the reproduction entry points and are not portable.** They
hard-code this machine's layout — paths under `$HOME`, GPU indices, IB HCA
names — and several drive experiments that were run once to close a question
and then abandoned, so the retractions and negative results are in here too.

To rerun an experiment, use the portable entry points instead:

| To rerun | Use |
|---|---|
| Fig.6, both LLM columns | `scripts/vllm_exp/run_fig6.sh` |
| Fig.6 GT only | `scripts/vllm_exp/run_fig6_gptoss20b.sh` |
| Fig.6 LM only | `scripts/vllm_exp/run_fig6_llama31.sh` |
| Fig.7 | `scripts/vllm_exp/run_fig7_gptoss20b_online.sh` |
| Fig.9 | `scripts/vllm_exp/run_fig9.sh` |

Those are adapted from the scripts here, with every machine-specific value
moved into `scripts/vllm_exp/exp_env.sh` as an overridable variable. See
`docs/EXPERIMENTS.md` §T3.

The one script here worth reading on its own is `rdma_driver.py`, the
measurement client used for the same-client pairings — but the maintained copy
lives at `scripts/vllm_exp/rdma_driver.py`.
