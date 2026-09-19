__version__ = "0.1.0"

__all__ = [
    "EngineConfig",
    "AsyncLLMEngine",
    "GenerationRequest",
    "LLMEngine",
    "get_rank",
    "get_world_size",
]


def __getattr__(name: str):
    if name == "EngineConfig":
        from .config import EngineConfig

        return EngineConfig
    if name == "LLMEngine":
        from .engine import LLMEngine

        return LLMEngine
    if name in ("AsyncLLMEngine", "GenerationRequest"):
        from .async_engine import AsyncLLMEngine, GenerationRequest

        return {"AsyncLLMEngine": AsyncLLMEngine, "GenerationRequest": GenerationRequest}[name]
    if name in ("get_rank", "get_world_size"):
        from .comm import get_rank, get_world_size

        return {"get_rank": get_rank, "get_world_size": get_world_size}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
