import json

import torch

from fluidgpu_torch.config import EngineConfig
from fluidgpu_torch.executor import LayerExecutor
from fluidgpu_torch.profile import load_profile, make_handwritten_profile, make_task_profile


class FakeComm:
    def __init__(self, rank):
        self.rank = rank
        self.send_log = []
        self.recv_log = []

    def get_rank(self):
        return self.rank

    def send_tensor(self, tensor, dst):
        self.send_log.append((tuple(tensor.shape), dst))

    def recv_tensor(self, tensor, src):
        self.recv_log.append((tuple(tensor.shape), src))
        tensor.fill_(src + 1)


def test_rank0_segment_boundary(tmp_path):
    executor, fake = make_executor(tmp_path, rank=0)
    calls = run_fake_forward(executor)

    assert calls == ["layer_0_attn", "layer_0_mlp", "layer_1_attn", "layer_1_mlp"]
    assert fake.send_log == [((1, 3, 4), 1)]
    assert fake.recv_log == [((1, 3, 4), 1)]


def test_rank1_segment_boundary(tmp_path):
    executor, fake = make_executor(tmp_path, rank=1)
    calls = run_fake_forward(executor)

    assert calls == ["layer_2_attn", "layer_2_mlp", "layer_3_attn", "layer_3_mlp"]
    assert fake.send_log == [((1, 3, 4), 0)]
    assert fake.recv_log == [((1, 3, 4), 0)]


def test_kernel_group_attn_mlp_split_boundary(tmp_path):
    executor0, fake0 = make_kernel_group_executor(tmp_path, rank=0)
    calls0 = run_fake_kernel_groups(executor0, layer_count=2)
    assert calls0 == ["layer_0_attn", "layer_1_attn", "layer_1_mlp"]
    assert fake0.send_log == [((1, 3, 4), 1)]
    assert fake0.recv_log == [((1, 3, 4), 1)]

    executor1, fake1 = make_kernel_group_executor(tmp_path, rank=1)
    calls1 = run_fake_kernel_groups(executor1, layer_count=2)
    assert calls1 == ["layer_0_mlp"]
    assert fake1.send_log == [((1, 3, 4), 0)]
    assert fake1.recv_log == [((1, 3, 4), 0)]


def make_executor(tmp_path, rank):
    raw = make_handwritten_profile(
        model="tiny",
        n_layers=4,
        split_at=2,
        hidden_size=4,
        num_kv_heads=1,
        head_dim=4,
        max_seq_len=8,
    )
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(raw))
    profile = load_profile(path, model_name="tiny")
    cfg = EngineConfig(
        model_name="tiny",
        profiling_json=path,
        max_seq_len=8,
        device="cpu",
    )
    fake = FakeComm(rank)
    return LayerExecutor(profile, cfg, comm_backend=fake), fake


def make_kernel_group_executor(tmp_path, rank):
    raw = make_task_profile(
        model="tiny",
        task_names=["layer_0_attn", "layer_0_mlp", "layer_1_attn", "layer_1_mlp"],
        split_at=4,
        hidden_size=4,
        num_kv_heads=1,
        head_dim=4,
        max_seq_len=8,
    )
    raw["tasks"][2]["rank"] = 1
    path = tmp_path / "kernel_group_profile.json"
    path.write_text(json.dumps(raw))
    profile = load_profile(path, model_name="tiny")
    cfg = EngineConfig(
        model_name="tiny",
        profiling_json=path,
        max_seq_len=8,
        device="cpu",
    )
    fake = FakeComm(rank)
    return LayerExecutor(profile, cfg, comm_backend=fake), fake


def run_fake_forward(executor):
    return run_fake_kernel_groups(executor, layer_count=4)


def run_fake_kernel_groups(executor, *, layer_count):
    calls = []
    token_ids = torch.ones(1, 3, dtype=torch.long)
    hidden = executor.run_stage(
        "embed",
        lambda ids: torch.ones(1, 3, 4, dtype=torch.bfloat16),
        token_ids,
        mode="prefill",
        seq_len=3,
    )
    for i in range(layer_count):
        for suffix in ("attn", "mlp"):
            task_name = f"layer_{i}_{suffix}"

            def kernel_group(hidden_states, task_name=task_name, **kwargs):
                calls.append(task_name)
                return hidden_states + 1

            out = executor.run_kernel_group(
                task_name,
                kernel_group,
                hidden,
                mode="prefill",
                seq_len=3,
            )
            if out is not None:
                hidden = out
    out = executor.run_stage(
        "norm_lm_head",
        lambda h: h,
        hidden,
        mode="prefill",
        seq_len=3,
    )
    if out is not None:
        hidden = out
    return calls
