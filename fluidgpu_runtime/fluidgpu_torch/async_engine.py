from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from queue import Queue
from threading import Lock, Thread
from typing import Any

from .config import EngineConfig
from .engine import LLMEngine


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    max_new_tokens: int
    do_sample: bool = False
    seed: int = 0


@dataclass(frozen=True)
class _WorkItem:
    request_id: int
    request: GenerationRequest
    future: Future[str | None]


class AsyncLLMEngine:
    def __init__(
        self,
        engine: LLMEngine,
        *,
        num_workers: int = 1,
        priority_scheduling: bool = False,
        worker_profiles: list[Any] | None = None,
    ) -> None:
        assert num_workers > 0, f"num_workers must be positive, got {num_workers}"
        self.engine = engine
        self.num_workers = num_workers
        self.priority_scheduling = priority_scheduling
        # Complementary placement (paper zig-zag): worker i uses
        # worker_profiles[i % len], so concurrent requests occupy different
        # GPUs at the same pipeline stage instead of phase-aligning on one.
        self.worker_profiles = worker_profiles
        self._queues: list[Queue[_WorkItem | None]] = [Queue() for _ in range(num_workers)]
        self._closed = False
        self._close_lock = Lock()
        self._next_request_id = 0
        self._workers = self._make_workers(engine, num_workers)
        self._precapture_graphs(engine)
        self._threads = [
            Thread(
                target=self._run,
                args=(worker_id,),
                name=f"fluidgpu-async-engine-{worker_id}",
                daemon=True,
            )
            for worker_id in range(num_workers)
        ]
        for thread in self._threads:
            thread.start()

    @classmethod
    def from_config(
        cls,
        cfg: EngineConfig,
        *,
        num_workers: int = 1,
        priority_scheduling: bool = False,
        worker_profiles: list[Any] | None = None,
    ) -> AsyncLLMEngine:
        return cls(
            LLMEngine(cfg),
            num_workers=num_workers,
            priority_scheduling=priority_scheduling,
            worker_profiles=worker_profiles,
        )

    def __enter__(self) -> AsyncLLMEngine:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def submit(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        seed: int = 0,
    ) -> Future[str | None]:
        with self._close_lock:
            if self._closed:
                raise RuntimeError("AsyncLLMEngine is closed")
            future: Future[str | None] = Future()
            request_id = self._next_request_id
            self._next_request_id += 1
            worker_id = request_id % self.num_workers
            self._queues[worker_id].put(
                _WorkItem(
                    request_id=request_id,
                    request=GenerationRequest(
                        prompt=prompt,
                        max_new_tokens=max_new_tokens,
                        do_sample=do_sample,
                        seed=seed,
                    ),
                    future=future,
                )
            )
            return future

    def submit_many(self, requests: list[GenerationRequest]) -> list[Future[str | None]]:
        return [
            self.submit(
                request.prompt,
                max_new_tokens=request.max_new_tokens,
                do_sample=request.do_sample,
                seed=request.seed,
            )
            for request in requests
        ]

    def close(self, *, wait: bool = True) -> None:
        with self._close_lock:
            if not self._closed:
                self._closed = True
                for queue in self._queues:
                    queue.put(None)
        if wait:
            for thread in self._threads:
                thread.join()
            self.engine.close()

    def _run(self, worker_id: int) -> None:
        worker = self._workers[worker_id]
        queue = self._queues[worker_id]
        while True:
            item = queue.get()
            if item is None:
                try:
                    close = getattr(worker, "close", None)
                    if close is not None:
                        close()
                finally:
                    queue.task_done()
                return
            request = item.request
            future = item.future
            try:
                if future.set_running_or_notify_cancel():
                    result = worker.generate(
                        request.prompt,
                        max_new_tokens=request.max_new_tokens,
                        do_sample=request.do_sample,
                        seed=request.seed,
                    )
                    future.set_result(result)
            except BaseException as exc:
                future.set_exception(exc)
            finally:
                queue.task_done()

    def _precapture_graphs(self, engine: LLMEngine) -> None:
        """Capture every worker's decode graphs before serving.

        Capture is serialized by a global lock; if it overlapped with serving,
        the two ranks could acquire it in different worker orders and stall a
        paired worker's NCCL recv past the watchdog timeout. Capturing all
        workers in worker-id order here keeps both ranks in lockstep and takes
        the cost off the first request.
        """
        if not getattr(engine, "cuda_graph_decode", False):
            return
        graph_plan_for = getattr(engine, "_graph_plan_for", None)
        if graph_plan_for is None:
            return
        import torch

        zero = torch.zeros((1, 1), dtype=torch.long, device=engine.device)
        for worker in self._workers:
            runner = getattr(worker, "runner", None)
            comm_backend = getattr(worker, "comm_backend", None)
            if runner is None or comm_backend is None:
                continue
            plan = graph_plan_for(runner, comm_backend)
            if not plan._captured:
                # pos=0 is a valid capture state: begin_request resets pos and
                # the prefill rewrites the KV cache before any replay is read.
                plan.begin_request(0, zero)
                plan.capture()

    def _make_workers(self, engine: LLMEngine, num_workers: int) -> list[Any]:
        create_worker = getattr(engine, "create_worker", None)
        if create_worker is None:
            assert num_workers == 1, "num_workers > 1 requires engine.create_worker()"
            return [_LegacyEngineWorker(engine) for _ in range(num_workers)]
        profiles = self.worker_profiles
        return [
            create_worker(
                worker_id,
                use_process_group=num_workers > 1,
                # Priority-aware pipelining (paper §III-C): earlier workers get
                # higher CUDA stream priority (more negative), so concurrent
                # requests stagger instead of stalling on transfers together.
                # torch clamps out-of-range priorities to the device's range.
                stream_priority=(
                    worker_id - (num_workers - 1) if self.priority_scheduling else 0
                ),
                profile=(profiles[worker_id % len(profiles)] if profiles else None),
            )
            for worker_id in range(num_workers)
        ]


class _LegacyEngineWorker:
    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        do_sample: bool = False,
        seed: int = 0,
    ) -> str | None:
        return self.engine.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            seed=seed,
        )

    def close(self) -> None:
        return None
