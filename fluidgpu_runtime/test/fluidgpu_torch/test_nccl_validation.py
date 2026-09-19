import os

import pytest
import torch
import torch.multiprocessing as mp

from fluidgpu_torch import comm
from fluidgpu_torch.config import EngineConfig


@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two CUDA devices",
)
def test_nccl_validation_two_rank():
    master_port = 29531
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(rank, master_port)) for rank in range(2)]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join(timeout=120)
    assert all(proc.exitcode == 0 for proc in procs)


def _worker(rank, master_port):
    os.environ["FLUIDGPU_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    cfg = EngineConfig(
        model_name="dummy",
        profiling_json="dummy.json",
        master_port=master_port,
        rank_override=rank,
        device_id=rank,
    )
    comm.init_process_group(cfg)
    tensor = torch.full((1, 2048, 1536), rank + 1, dtype=torch.bfloat16, device="cuda")
    peer = torch.empty_like(tensor)
    if rank == 0:
        comm.send_tensor(tensor, dst=1)
        comm.recv_tensor(peer, src=1)
        assert torch.all(peer == 2)
    else:
        comm.recv_tensor(peer, src=0)
        comm.send_tensor(tensor, dst=0)
        assert torch.all(peer == 1)
    comm.destroy()
