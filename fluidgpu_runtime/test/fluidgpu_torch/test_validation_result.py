import importlib.util
import json
import sys
from pathlib import Path

import torch


def test_validation_result_writes_manifest(tmp_path):
    tool = load_tool()
    rank0_log = tmp_path / "rank0.log"
    rank1_log = tmp_path / "rank1.log"
    diagnostics = tmp_path / "diag.pt"
    output = tmp_path / "manifest.json"
    rank0_log.write_text(
        "\n".join(
            [
                "Channel 00/0 via NET/IB/0/GDRDMA",
                "generated_token_ids: [1, 2, 3]",
                "first_logits_argmax: 1",
                "request_index: 0",
            ]
        )
    )
    rank1_log.write_text("GDR 1\n")
    torch.save(
        {
            "model": "tiny",
            "prompt": "hello",
            "max_new_tokens": 3,
            "seed": 0,
            "tokens": [1, 2, 3],
            "first_logits": torch.tensor([[0.0, 1.0, 0.5]]),
        },
        diagnostics,
    )

    run_main(
        tool,
        "--name",
        "demo",
        "--rank0-log",
        str(rank0_log),
        "--rank1-log",
        str(rank1_log),
        "--diagnostics",
        str(diagnostics),
        "--output-json",
        str(output),
        "--require-generated-tokens",
        "3",
        "--require-request-count",
        "1",
        "--require-gdr",
    )

    manifest = json.loads(output.read_text())
    assert manifest["name"] == "demo"
    assert manifest["generated_token_ids"] == [1, 2, 3]
    assert manifest["diagnostics_summary"]["first_logits_argmax"] == 1
    assert manifest["gdrdma_seen"] is True
    assert manifest["errors"] == []


def run_main(module, *args: str) -> None:
    old_argv = sys.argv
    sys.argv = ["validation_result.py", *args]
    try:
        module.main()
    finally:
        sys.argv = old_argv


def load_tool():
    path = Path(__file__).parents[2] / "examples" / "fluidgpu_torch" / "validation_result.py"
    spec = importlib.util.spec_from_file_location("fluidgpu_validation_result", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
