from __future__ import annotations

from pathlib import Path

from fluidgpu_torch.orchestrator.run_spec import expand_runs, load_spec, split_env_command


def test_all_ad_experiment_specs_load_and_expand():
    repo = Path(__file__).resolve().parents[3]
    specs = sorted((repo / "experiments").glob("*.yaml"))

    assert {path.name for path in specs} == {
        "fig2_kernel_heterogeneity.yaml",
        "fig3_coarse_granularity.yaml",
        "fig10_pipeline.yaml",
        "fig11_monitor_sensitivity.yaml",
        "fig12a_slow_network.yaml",
        "fig12b_milp_scalability.yaml",
    }
    for path in specs:
        spec = load_spec(path)
        runs = expand_runs(spec)
        assert runs
        assert all(run.command for run in runs)


def test_run_spec_command_parser_accepts_env_prefixes():
    env, argv = split_env_command(
        "MODEL=openai/gpt-oss-20b MODE=bench bash scripts/run_pd_baseline_vllm.sh"
    )

    assert env == {"MODEL": "openai/gpt-oss-20b", "MODE": "bench"}
    assert argv == ["bash", "scripts/run_pd_baseline_vllm.sh"]



def test_fig10_fig11_fig12a_specs_pin_gpt_oss_20b():
    repo = Path(__file__).resolve().parents[3]

    for name in (
        "fig10_pipeline.yaml",
        "fig11_monitor_sensitivity.yaml",
        "fig12a_slow_network.yaml",
    ):
        spec = load_spec(repo / "experiments" / name)
        runs = expand_runs(spec)
        assert runs
        for run in runs:
            assert run.payload.get("model") == "openai/gpt-oss-20b"
            assert "--model openai/gpt-oss-20b" in run.command



