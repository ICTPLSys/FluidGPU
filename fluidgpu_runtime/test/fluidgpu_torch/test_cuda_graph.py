import torch

from fluidgpu_torch.cuda_graph import CUDAGraphModule, StaticCUDAGraph


def test_static_cuda_graph_rejects_cpu_tensor():
    try:
        StaticCUDAGraph(lambda x: x + 1, (torch.ones(2),))
    except AssertionError as exc:
        assert "CUDA" in str(exc)
    else:
        raise AssertionError("expected CPU tensor capture to fail")


def test_static_cuda_graph_replay_cuda():
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda:0")
    example = torch.ones(4, device=device)
    graph = StaticCUDAGraph(lambda x: x * 2 + 1, (example,), warmup_iters=1)

    out = graph.replay(torch.full((4,), 3.0, device=device))

    torch.testing.assert_close(out, torch.full((4,), 7.0, device=device))


def test_static_cuda_graph_validates_replay_shape():
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda:0")
    graph = StaticCUDAGraph(lambda x: x + 1, (torch.ones(2, device=device),), warmup_iters=1)

    try:
        graph.replay(torch.ones(3, device=device))
    except AssertionError as exc:
        assert "shape mismatch" in str(exc)
    else:
        raise AssertionError("expected shape mismatch")


def test_cuda_graph_module_eager_before_capture_cpu():
    module = CUDAGraphModule(torch.nn.Linear(2, 3))
    out = module(torch.ones(1, 2))
    assert out.shape == (1, 3)
    assert not module.is_captured


def test_cuda_graph_module_capture_rejects_cpu_tensor():
    module = CUDAGraphModule(torch.nn.Linear(2, 3))
    try:
        module.capture(torch.ones(1, 2))
    except AssertionError as exc:
        assert "CUDA" in str(exc)
    else:
        raise AssertionError("expected CPU tensor capture to fail")


def test_cuda_graph_module_capture_and_forward_cuda():
    if not torch.cuda.is_available():
        return

    device = torch.device("cuda:0")
    wrapped = CUDAGraphModule(torch.nn.Linear(2, 3, bias=False).to(device), warmup_iters=1)
    example = torch.ones(1, 2, device=device)
    wrapped.capture(example)
    assert wrapped.is_captured

    x = torch.full((1, 2), 2.0, device=device)
    eager = wrapped.module(x)
    replay = wrapped(x)

    torch.testing.assert_close(replay, eager)
