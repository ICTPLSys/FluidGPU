import csv
import importlib.util
import sys
from pathlib import Path


def test_refine_profile_splits_attention_tasks():
    refiner = load_refiner()
    raw = {
        "model": "tiny",
        "rank_count": 2,
        "task_count": 4,
        "hidden_size": 8,
        "num_kv_heads": 1,
        "head_dim": 4,
        "dtype": "bfloat16",
        "max_seq_len": 16,
        "tasks": [
            {"task_id": 0, "kernel_group_id": 0, "name": "embed", "rank": 0},
            {"task_id": 1, "kernel_group_id": 1, "name": "layer_0_attn", "rank": 1},
            {"task_id": 2, "kernel_group_id": 2, "name": "layer_0_mlp", "rank": 0},
            {"task_id": 3, "kernel_group_id": 3, "name": "norm_lm_head", "rank": 0},
        ],
    }
    splits = refiner.parse_split_weights("qkv=0.25,sdpa=0.50,o_proj=0.25")

    refined = refiner.refine_profile(raw, splits)

    assert refined["task_count"] == 6
    assert [task["name"] for task in refined["tasks"]] == [
        "embed",
        "layer_0_qkv",
        "layer_0_sdpa",
        "layer_0_o_proj",
        "layer_0_mlp",
        "norm_lm_head",
    ]
    assert [task["task_id"] for task in refined["tasks"]] == list(range(6))
    assert refined["tasks"][1]["rank"] == 1
    assert refined["tasks"][4]["rank"] == 0
    assert refined["fine_grained"]["runtime_compatible"] is True


def test_refine_csv_rows_splits_attention_latency():
    refiner = load_refiner()
    rows = [
        {
            "task_id": "0",
            "kernel_group_id": "0",
            "name": "embed",
            "prefill_us": "1.0",
            "decode_us": "2.0",
            "in_bytes": "8",
            "out_bytes": "8",
        },
        {
            "task_id": "1",
            "kernel_group_id": "1",
            "name": "layer_0_attn",
            "prefill_us": "100.0",
            "decode_us": "10.0",
            "in_bytes": "8",
            "out_bytes": "8",
        },
        {
            "task_id": "2",
            "kernel_group_id": "2",
            "name": "layer_0_mlp",
            "prefill_us": "5.0",
            "decode_us": "6.0",
            "in_bytes": "8",
            "out_bytes": "8",
        },
    ]
    splits = refiner.parse_split_weights("qkv=0.25,sdpa=0.50,o_proj=0.25")

    refined = refiner.refine_csv_rows(rows, splits)

    assert [row["name"] for row in refined] == [
        "embed",
        "layer_0_qkv",
        "layer_0_sdpa",
        "layer_0_o_proj",
        "layer_0_mlp",
    ]
    assert [row["task_id"] for row in refined] == ["0", "1", "2", "3", "4"]
    assert [float(row["prefill_us"]) for row in refined[1:4]] == [25.0, 50.0, 25.0]
    assert [float(row["decode_us"]) for row in refined[1:4]] == [2.5, 5.0, 2.5]


def test_refine_csv_cli(tmp_path):
    refiner = load_refiner()
    input_path = tmp_path / "profile.csv"
    output_path = tmp_path / "fine.csv"
    with input_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_id",
                "kernel_group_id",
                "name",
                "prefill_us",
                "decode_us",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_id": "0",
                "kernel_group_id": "0",
                "name": "layer_0_attn",
                "prefill_us": "12.0",
                "decode_us": "6.0",
            }
        )

    rows = refiner.read_csv_rows(input_path)
    refined = refiner.refine_csv_rows(rows, refiner.parse_split_weights("qkv=0.5,sdpa=0.5"))
    refiner.write_csv_rows(output_path, refined)

    out = list(csv.DictReader(output_path.open()))
    assert [row["name"] for row in out] == ["layer_0_qkv", "layer_0_sdpa"]
    assert [float(row["prefill_us"]) for row in out] == [6.0, 6.0]


def load_refiner():
    path = Path(__file__).parents[2] / "examples" / "fluidgpu_torch" / "refine_llm_kernel_groups.py"
    spec = importlib.util.spec_from_file_location("fluidgpu_refine_llm_kernel_groups", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module
