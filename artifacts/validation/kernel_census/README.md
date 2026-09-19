# Kernel Census (A100 vs L40s) — source data for Fig.2 and Fig.3

Per-kernel execution-time measurements on an A100 and an L40s for the paper's
five workloads. This is the census the paper's motivation section is built
from, and it is what `scripts/plot/plot_fig2.py` and `plot_fig3.py` render.

## How it was measured

Each model was run under Nsight Systems on both GPUs, the CUDA kernel traces
were exported (`nsys stats --report cuda_gpu_trace`), and kernels were matched
across the two traces by name and launch position. Two derived quantities:

- **duration ratio** = L40s kernel time / A100 kernel time. Below 1.0 means the
  kernel runs faster on the L40s.
- **E2E time ratio** = of the total kernel time on the A100, the fraction
  contributed by kernels whose ratio is below 1.0.

The trace-alignment and classification pipeline lives outside this artifact,
in the authors' motivation repository
(`GPU_disaggregation/motivation/data_process/`); the multi-GB raw `.nsys-rep`
traces are not shipped. What is shipped here is its full output — every
matched kernel, not a summary — so both figures are reproducible from this
directory alone.

## Files

| File | Contents |
|---|---|
| `ratios/<model>.txt` | Whitespace-separated L40s/A100 duration ratios, one value per matched kernel instance. Feeds Fig.2(a). |
| `e2e_time_ratio.csv` | Per-model E2E time ratio (%). Feeds Fig.2(b). |
| `fig3_phase_points.tsv` | GPT-oss 20B kernels labelled prefill/decode, with A100 duration and ratio. Feeds Fig.3(a). |
| `fig3_block_points.tsv` | The same kernels labelled attention/FFN. Feeds Fig.3(b). |
| `fig3_phase_points_full.tsv` | The unmerged per-kernel table behind the two above, with CUDA kernel names. |
| `fig3_kernel_pairs.tsv` | Per-position pairing of the A100-side and L40s-side kernel, aggregated over all 303 245 launches. Resolves Fig.3's two callouts. |

`scripts/preprocess/ingest_kernel_census.py` reads these in place and writes
only small summary records under `artifacts/logs/fig{2,3}/`; the data is not
duplicated.

Sample counts in `ratios/`: SD3.5 64, Llama-3.1-8B 354, GPT-oss-20B 302 909,
Qwen2.5-VL-7B 1 267 713, Mamba-Codestral-7B 272 507.

The two Fig.3 files carry one row per plotted point. Nearby kernels are merged
into a single point (the merge rules are in each file's header comments), with
`merged_count` and `member_*` columns recording exactly which raw kernels went
into each point, so the merging is auditable and reversible.

## The two GPUs do not always run the same kernel

At 33% of the matched positions the A100 and the L40s execute *different*
kernels, because cuBLAS and cutlass select different implementations per
architecture. `fig3_kernel_pairs.tsv` records both names per position, which is
what makes the paper's Fig.3 callouts resolvable — the paper names each of them
after the kernel that distinguishes the pair:

- **cublasGemv.** At three prefill positions the A100 runs a tensorop GEMM
  while the L40s falls back to `internal::gemvx::kernel`. The plotted callout
  sits on the one at 22.5 µs, ratio 1.75 (census median 1.66) — the A100 is
  1.75x faster, against the 1.9x the paper reports. The other two GEMV
  positions plot at 2.20 and 2.56, and a third (`dot+reduce` on the A100) runs
  1.23x faster on the L40s instead; all are listed in the figure record's
  `annotations[].other_positions`, so the choice of representative is visible
  rather than implicit.
- **FlashAttention.** Both GPUs run `kernel_unified_attention_3d`, vLLM's tiled
  attention kernel. The callout sits at 18.4 µs, ratio 0.493 — the L40s is
  2.03x faster, against the paper's "up to 2.1x".

Note that a position's plotted ratio and its census median differ (1.75 vs 1.66
above): the point files keep one representative launch per kernel, while the
median is taken over every launch. Both are recorded.

## Cross-checks against the paper

Recomputed from these files by `scripts/preprocess/ingest_kernel_census.py`,
which asserts each one:

| Quantity | From this data | Paper |
|---|---|---|
| Mean fraction of kernels faster on L40s (Fig.2a at ratio 1.0) | 67% | 67% |
| Mean E2E time ratio (Fig.2b) | 36.1% | 36% |
| Max E2E time ratio (diffusion) | 53.2% | 53% |
| Prefill kernels faster on L40s (Fig.3a) | 45% | 45% |
| Decode kernels faster on L40s (Fig.3a) | 57% | 57% |
| Attention kernels faster on L40s (Fig.3b) | 72% | 71% |
| FFN kernels faster on L40s (Fig.3b) | 30% | 33% |

The attention/FFN split lands within a couple of points of the paper's; the
other five reproduce exactly.

## Relationship to `scripts/repro.sh --fig 2`

`fluidgpu_runtime/examples/fluidgpu_torch/profile_llm_kernels.py` also produces
a per-kernel census, and `repro.sh --fig 2` runs it on both GPUs. That path
uses the FluidGPU runtime's own profiler on short synthetic prompts and yields
a much smaller census (38–106 unique kernels per model) over a narrower slice
of the workload, so its ratio distribution is not comparable to the traces
above and does not reproduce the paper's percentages. Those runs are archived
under `artifacts/logs/fig2_torch_profiler_archive/`.
