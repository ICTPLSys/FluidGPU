from __future__ import annotations

import argparse

import torch
from torch import nn

from fluidgpu_torch.cuda_graph import CUDAGraphModule, StaticCUDAGraph
from fluidgpu_torch.config import validate_torch_version


class TinyDecodeBlock(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.norm(self.embed(token_ids)))


class TinyKVDecodeBlock(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_heads: int,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.max_seq_len = max_seq_len
        self.scale = self.head_dim**-0.5
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.register_buffer(
            "key_cache",
            torch.zeros(1, num_heads, max_seq_len, self.head_dim),
            persistent=False,
        )
        self.register_buffer(
            "value_cache",
            torch.zeros(1, num_heads, max_seq_len, self.head_dim),
            persistent=False,
        )

    def reset_cache(self) -> None:
        self.key_cache.zero_()
        self.value_cache.zero_()

    def forward(
        self,
        token_ids: torch.Tensor,
        cache_position: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.norm(self.embed(token_ids))
        query = self._heads(self.q_proj(hidden))
        key = self._heads(self.k_proj(hidden))
        value = self._heads(self.v_proj(hidden))
        cache_index = cache_position.view(1, 1, 1, 1).expand(
            1,
            self.num_heads,
            1,
            self.head_dim,
        )
        self.key_cache.scatter_(2, cache_index, key)
        self.value_cache.scatter_(2, cache_index, value)

        scores = torch.matmul(query, self.key_cache.transpose(-1, -2)) * self.scale
        scores = scores + attention_mask
        probs = torch.softmax(scores.float(), dim=-1).to(hidden.dtype)
        context = torch.matmul(probs, self.value_cache)
        context = context.transpose(1, 2).reshape(1, 1, self.hidden_size)
        return self.lm_head(self.o_proj(context))[:, -1, :]

    def _heads(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)


def main() -> None:
    args = parse_args()
    validate_torch_version()
    assert torch.cuda.is_available(), "CUDA Graph probe requires CUDA"
    device = torch.device(f"cuda:{args.device_id}")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)

    run_tensor_probe(device, args.warmup_iters)
    run_tiny_module_probe(device, args.warmup_iters)
    run_tiny_kv_decode_probe(device, args.warmup_iters, args.max_seq_len, args.kv_steps)
    if args.hf_model is not None:
        run_hf_probes(args, device)


def run_tensor_probe(device: torch.device, warmup_iters: int) -> None:
    def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x * 2 + y

    example_x = torch.ones((4,), dtype=torch.float32, device=device)
    example_y = torch.full((4,), 3.0, dtype=torch.float32, device=device)
    graph = StaticCUDAGraph(fn, (example_x, example_y), warmup_iters=warmup_iters)

    x = torch.full((4,), 5.0, dtype=torch.float32, device=device)
    y = torch.full((4,), 7.0, dtype=torch.float32, device=device)
    eager = fn(x, y)
    replay = graph.replay(x, y)
    torch.testing.assert_close(replay, eager)
    print("tensor_probe: ok max_abs=0.000000")


def run_tiny_module_probe(device: torch.device, warmup_iters: int) -> None:
    block = TinyDecodeBlock(vocab_size=128, hidden_size=64).to(device).eval()
    module = CUDAGraphModule(block, warmup_iters=warmup_iters)
    token = torch.tensor([[7]], dtype=torch.long, device=device)
    module.capture(token)
    next_token = torch.tensor([[13]], dtype=torch.long, device=device)
    eager = block(next_token)
    replay = module(next_token)
    max_abs = float((replay - eager).abs().max().item())
    torch.testing.assert_close(replay, eager)
    print(f"tiny_module_probe: ok max_abs={max_abs:.6f}")


def run_tiny_kv_decode_probe(
    device: torch.device,
    warmup_iters: int,
    max_seq_len: int,
    steps: int,
) -> None:
    assert steps > 0, f"kv_steps must be positive, got {steps}"
    assert steps <= max_seq_len, f"kv_steps {steps} exceeds max_seq_len {max_seq_len}"
    eager_block = TinyKVDecodeBlock(
        vocab_size=128,
        hidden_size=64,
        num_heads=4,
        max_seq_len=max_seq_len,
    ).to(device).eval()
    graph_block = TinyKVDecodeBlock(
        vocab_size=128,
        hidden_size=64,
        num_heads=4,
        max_seq_len=max_seq_len,
    ).to(device).eval()
    graph_block.load_state_dict(eager_block.state_dict())

    example_token = torch.tensor([[1]], dtype=torch.long, device=device)
    example_position = torch.tensor([0], dtype=torch.long, device=device)
    example_mask = decode_attention_mask(0, max_seq_len, device)

    @torch.inference_mode()
    def decode_logits(
        token_ids: torch.Tensor,
        cache_position: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return graph_block(token_ids, cache_position, attention_mask)

    graph = StaticCUDAGraph(
        decode_logits,
        (example_token, example_position, example_mask),
        warmup_iters=warmup_iters,
    )
    eager_block.reset_cache()
    graph_block.reset_cache()

    max_abs = 0.0
    with torch.inference_mode():
        for step in range(steps):
            token = torch.tensor([[step + 2]], dtype=torch.long, device=device)
            position = torch.tensor([step], dtype=torch.long, device=device)
            mask = decode_attention_mask(step, max_seq_len, device)
            eager = eager_block(token, position, mask)
            replay = graph.replay(token, position, mask)
            max_abs = max(max_abs, float((replay - eager).abs().max().item()))
            torch.testing.assert_close(replay, eager, rtol=1e-4, atol=1e-4)
    print(f"tiny_kv_decode_probe: ok steps={steps} max_abs={max_abs:.6f}")


def decode_attention_mask(step: int, max_seq_len: int, device: torch.device) -> torch.Tensor:
    kv_positions = torch.arange(max_seq_len, device=device)
    allowed = kv_positions <= step
    mask = torch.zeros((max_seq_len,), dtype=torch.float32, device=device)
    mask = mask.masked_fill(~allowed, torch.finfo(torch.float32).min)
    return mask.view(1, 1, 1, max_seq_len)


def run_hf_probes(args: argparse.Namespace, device: torch.device) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = dtype_from_arg(args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        **model_dtype_kwargs(dtype),
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    run_hf_token_probe(args, device, model, tokenizer)
    if args.hf_kv_decode:
        run_hf_static_cache_decode_probe(args, device, model, tokenizer)


def run_hf_token_probe(
    args: argparse.Namespace,
    device: torch.device,
    model: nn.Module,
    tokenizer: object,
) -> None:
    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids[:, -1:].to(device)
    vocab_size = int(model.config.vocab_size)
    probe_token = (input_ids + 1) % vocab_size

    @torch.inference_mode()
    def token_logits(token_ids: torch.Tensor) -> torch.Tensor:
        out = model(token_ids, use_cache=False)
        return out.logits[:, -1, :]

    graph = StaticCUDAGraph(token_logits, (input_ids,), warmup_iters=args.warmup_iters)
    eager = token_logits(probe_token)
    replay = graph.replay(probe_token)
    max_abs = float((replay - eager).abs().max().item())
    cosine = torch.nn.functional.cosine_similarity(
        replay.float().flatten(),
        eager.float().flatten(),
        dim=0,
    ).item()
    torch.testing.assert_close(replay, eager, rtol=args.rtol, atol=args.atol)
    print(
        "hf_token_probe: ok "
        f"token={int(probe_token.item())} cosine={cosine:.9f} max_abs={max_abs:.9f}"
    )


def run_hf_static_cache_decode_probe(
    args: argparse.Namespace,
    device: torch.device,
    model: nn.Module,
    tokenizer: object,
) -> None:
    from transformers.cache_utils import StaticCache

    input_ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    prompt_len = int(input_ids.shape[1])
    max_cache_len = prompt_len + args.hf_kv_steps + 1
    eager_cache = StaticCache(config=model.config, max_cache_len=max_cache_len)
    graph_cache = StaticCache(config=model.config, max_cache_len=max_cache_len)

    eager_prefill_logits = prefill_static_cache(model, input_ids, eager_cache)
    prefill_static_cache(model, input_ids, graph_cache)
    token = torch.tensor(
        [[int(eager_prefill_logits[:, -1, :].argmax(dim=-1).item())]],
        dtype=torch.long,
        device=device,
    )
    example_position = torch.tensor([prompt_len], dtype=torch.long, device=device)
    example_position_ids = example_position.unsqueeze(0)

    @torch.inference_mode()
    def graph_decode(
        token_ids: torch.Tensor,
        cache_position: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        out = model(
            token_ids,
            past_key_values=graph_cache,
            use_cache=True,
            cache_position=cache_position,
            position_ids=position_ids,
        )
        return out.logits[:, -1, :]

    graph = StaticCUDAGraph(
        graph_decode,
        (token, example_position, example_position_ids),
        warmup_iters=args.warmup_iters,
    )
    eager_prefill_logits = prefill_static_cache(model, input_ids, eager_cache)
    prefill_static_cache(model, input_ids, graph_cache)
    token.fill_(int(eager_prefill_logits[:, -1, :].argmax(dim=-1).item()))

    max_abs = 0.0
    min_cosine = 1.0
    with torch.inference_mode():
        for step in range(args.hf_kv_steps):
            position = torch.tensor([prompt_len + step], dtype=torch.long, device=device)
            position_ids = position.unsqueeze(0)
            eager = eager_decode_logits(model, token, eager_cache, position, position_ids)
            replay = graph.replay(token, position, position_ids)
            step_max_abs = float((replay - eager).abs().max().item())
            step_cosine = torch.nn.functional.cosine_similarity(
                replay.float().flatten(),
                eager.float().flatten(),
                dim=0,
            ).item()
            max_abs = max(max_abs, step_max_abs)
            min_cosine = min(min_cosine, step_cosine)
            torch.testing.assert_close(replay, eager, rtol=args.rtol, atol=args.atol)
            token.fill_(int(eager.argmax(dim=-1).item()))
    print(
        "hf_kv_decode_probe: ok "
        f"steps={args.hf_kv_steps} min_cosine={min_cosine:.9f} max_abs={max_abs:.9f}"
    )


@torch.inference_mode()
def prefill_static_cache(model: nn.Module, input_ids: torch.Tensor, cache: object) -> torch.Tensor:
    cache.reset()
    prompt_len = int(input_ids.shape[1])
    cache_position = torch.arange(prompt_len, dtype=torch.long, device=input_ids.device)
    position_ids = cache_position.unsqueeze(0)
    out = model(
        input_ids,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
        position_ids=position_ids,
    )
    return out.logits


@torch.inference_mode()
def eager_decode_logits(
    model: nn.Module,
    token_ids: torch.Tensor,
    cache: object,
    cache_position: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor:
    out = model(
        token_ids,
        past_key_values=cache,
        use_cache=True,
        cache_position=cache_position,
        position_ids=position_ids,
    )
    return out.logits[:, -1, :]


def dtype_from_arg(value: str) -> torch.dtype:
    if value == "bf16":
        return torch.bfloat16
    if value == "fp16":
        return torch.float16
    if value == "fp32":
        return torch.float32
    raise AssertionError(f"unsupported dtype {value}")


def model_dtype_kwargs(dtype: torch.dtype) -> dict[str, torch.dtype]:
    import transformers

    major_minor = transformers.__version__.split(".", 2)[:2]
    major, minor = int(major_minor[0]), int(major_minor[1])
    if major > 4 or (major == 4 and minor >= 56):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe standalone CUDA Graph replay support")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--max-seq-len", type=int, default=16)
    parser.add_argument("--kv-steps", type=int, default=4)
    parser.add_argument("--hf-model", help="Optional HF CausalLM graph probes.")
    parser.add_argument("--hf-kv-decode", action="store_true", help="Also run HF StaticCache decode.")
    parser.add_argument("--hf-kv-steps", type=int, default=2)
    parser.add_argument("--prompt", default="Explain GPU disaggregation.")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    return parser.parse_args()


if __name__ == "__main__":
    main()
