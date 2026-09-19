import torch

import fluidgpu_torch


def test_validation_import():
    cfg = fluidgpu_torch.EngineConfig(model_name="dummy", profiling_json="dummy.json", device="cpu")
    assert fluidgpu_torch.__version__ == "0.1.0"
    assert cfg.dtype is torch.bfloat16


def test_engine_config_accepts_world_size_greater_than_two():
    cfg = fluidgpu_torch.EngineConfig(
        model_name="dummy",
        profiling_json="dummy.json",
        device="cpu",
        world_size=3,
        rank_override=2,
        comm_transport="auto",
    )

    assert cfg.world_size == 3
    assert cfg.rank_override == 2
