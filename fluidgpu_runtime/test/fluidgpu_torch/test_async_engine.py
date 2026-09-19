import pytest

from fluidgpu_torch.async_engine import AsyncLLMEngine, GenerationRequest


class FakeEngine:
    def __init__(self):
        self.calls = []
        self.closed = False

    def generate(self, prompt, max_new_tokens, do_sample=False, seed=0):
        self.calls.append((prompt, max_new_tokens, do_sample, seed))
        if prompt == "boom":
            raise RuntimeError("boom")
        return f"{prompt}:{max_new_tokens}:{do_sample}:{seed}"

    def close(self):
        self.closed = True


def test_async_engine_runs_requests_in_order():
    fake = FakeEngine()
    engine = AsyncLLMEngine(fake)

    future0 = engine.submit("a", max_new_tokens=2, seed=7)
    future1 = engine.submit("b", max_new_tokens=3, do_sample=False, seed=8)

    assert future0.result(timeout=5) == "a:2:False:7"
    assert future1.result(timeout=5) == "b:3:False:8"
    assert fake.calls == [
        ("a", 2, False, 7),
        ("b", 3, False, 8),
    ]
    engine.close()
    assert fake.closed


def test_async_engine_submit_many():
    fake = FakeEngine()
    engine = AsyncLLMEngine(fake)
    futures = engine.submit_many(
        [
            GenerationRequest("a", max_new_tokens=1),
            GenerationRequest("b", max_new_tokens=2, seed=9),
        ]
    )

    assert [future.result(timeout=5) for future in futures] == [
        "a:1:False:0",
        "b:2:False:9",
    ]
    engine.close()


def test_async_engine_propagates_exceptions():
    fake = FakeEngine()
    engine = AsyncLLMEngine(fake)
    future = engine.submit("boom", max_new_tokens=1)

    with pytest.raises(RuntimeError, match="boom"):
        future.result(timeout=5)
    engine.close()


def test_async_engine_rejects_submit_after_close():
    engine = AsyncLLMEngine(FakeEngine())
    engine.close()

    with pytest.raises(RuntimeError, match="closed"):
        engine.submit("late", max_new_tokens=1)


def test_async_engine_dispatches_requests_to_fixed_workers():
    fake = FakeEngineWithWorkers()
    engine = AsyncLLMEngine(fake, num_workers=2)

    futures = [
        engine.submit("a", max_new_tokens=1),
        engine.submit("b", max_new_tokens=2),
        engine.submit("c", max_new_tokens=3),
        engine.submit("d", max_new_tokens=4),
    ]

    assert [future.result(timeout=5) for future in futures] == [
        "worker0:a:1:0",
        "worker1:b:2:0",
        "worker0:c:3:0",
        "worker1:d:4:0",
    ]
    engine.close()
    assert fake.create_calls == [(0, True, 0), (1, True, 0)]
    assert fake.closed
    assert [worker.closed for worker in fake.workers] == [True, True]


def test_async_engine_priority_scheduling_staggers_stream_priorities():
    fake = FakeEngineWithWorkers()
    engine = AsyncLLMEngine(fake, num_workers=3, priority_scheduling=True)
    engine.close()
    # Earlier workers get higher CUDA stream priority (more negative).
    assert fake.create_calls == [(0, True, -2), (1, True, -1), (2, True, 0)]


def test_async_engine_requires_worker_factory_for_multiple_workers():
    with pytest.raises(AssertionError, match="create_worker"):
        AsyncLLMEngine(FakeEngine(), num_workers=2)


class FakeEngineWithWorkers:
    def __init__(self):
        self.create_calls = []
        self.workers = []
        self.closed = False

    def create_worker(
        self, worker_id, *, use_process_group=False, stream_priority=0, profile=None
    ):
        self.create_calls.append((worker_id, use_process_group, stream_priority))
        self.worker_profile_calls = getattr(self, "worker_profile_calls", [])
        self.worker_profile_calls.append(profile)
        worker = FakeWorker(worker_id)
        self.workers.append(worker)
        return worker

    def close(self):
        self.closed = True


class FakeWorker:
    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.closed = False

    def generate(self, prompt, max_new_tokens, do_sample=False, seed=0):
        return f"worker{self.worker_id}:{prompt}:{max_new_tokens}:{seed}"

    def close(self):
        self.closed = True


def test_async_engine_assigns_worker_profiles_round_robin():
    engine = FakeEngineWithWorkers()
    profiles = ["plan-a", "plan-b"]
    async_engine = AsyncLLMEngine(engine, num_workers=4, worker_profiles=profiles)
    try:
        assert engine.worker_profile_calls == ["plan-a", "plan-b", "plan-a", "plan-b"]
    finally:
        async_engine.close()


def test_async_engine_default_profile_when_no_worker_profiles():
    engine = FakeEngineWithWorkers()
    async_engine = AsyncLLMEngine(engine, num_workers=2)
    try:
        assert engine.worker_profile_calls == [None, None]
    finally:
        async_engine.close()
