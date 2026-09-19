from __future__ import annotations

from fluidgpu_torch.baselines import select_baseline


def test_homogeneous_policy_assigns_all_tasks_to_one_rank():
    policy = select_baseline("homogeneous")
    assignments = policy(["layer_0_attn", "layer_0_mlp"], rank=1)

    assert [item.rank for item in assignments] == [1, 1]
    assert [item.name for item in assignments] == ["layer_0_attn", "layer_0_mlp"]


def test_pd_and_af_baselines_expose_script_entrypoints():
    expected = {
        "pd": "scripts/run_pd_baseline_vllm.sh",
        "af": "scripts/run_af_baseline_fluidgpu.sh",
    }
    for name, script_path in expected.items():
        policy = select_baseline(name)
        result = policy([])
        assert result["baseline"] == name
        assert result["script_path"] == script_path
        assert result["entrypoint"] == f"bash {script_path}"
