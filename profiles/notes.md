# Nsight Compute profiling notes

## Week 0 — NCU + WSL2 workflow validation (2026-07-30)

Verified that RTX 3060 Laptop / WSL2 / NCU 2024.1.1.0 can reliably produce
differentiated, exportable profiling evidence — not just a single metric
read that happens to come back non-zero.

**Method.** A throwaway CUDA kernel, `shared_stride_kernel<PAD>`, that reads
shared memory with a fixed-column, varying-row access pattern:
`tile[32][32]` (`PAD=0`) hits the textbook 32-way bank conflict (all 32
threads in a row-varying, column-fixed read hash to the same bank);
`tile[32][33]` (`PAD=1`) removes it by shifting the per-row stride off the
32-bank boundary. This is deliberately different from the earlier
hello-world extension (`add_kernel`), which only touched global memory and
so could never have produced a non-zero bank-conflict count either way —
it wasn't actually capable of testing whether the counter works.

**Results** (two independent runs, `ncu --metrics ... --print-summary
per-kernel`):

| metric | unpadded (`PAD=0`) | padded (`PAD=1`) |
|---|---|---|
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` (run 1) | 63,488,467 | 272 |
| same metric (run 2) | 63,488,473 | 247 |
| `dram__bytes.sum` | 3.20 KB | 570.37 KB |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 31.76% | 32.25% |
| `launch__occupancy_limit_shared_mem` | 20 blocks | 19 blocks |

~233,000x reduction in bank conflicts, stable run-to-run (<0.01% variance
on the dominant unpadded count). `dram__throughput`, and all four
`launch__occupancy_limit_*` metrics populated with sane, non-zero values.
No `ERR_NVGPUCTRPERM` or other permission errors on either run.

**`--set full` also verified.** 38 replay passes, no errors, produced a
real 907 KB `.ncu-rep` file. Re-imported cleanly via `ncu --import <file>
--page raw --csv`. One correction to the original plan: the stall-reason
metrics live under `smsp__pcsamp_warps_issue_stalled_<reason>` (the
PC-sampling family), not `smsp__warp_issue_stalled_*_per_warp_active` as
originally assumed — use the `pcsamp` family name when querying directly
via `--metrics`.

**Conclusion:** Week 0 Check 1 passed. The WSL2 + NCU profiling workflow
is stable and trustworthy for the Week 3–5 optimization loop. No dual-boot
fallback needed.

## Triton v1 naive — first real profile (2026-07-30)

Profiled `_paged_attn_decode_v1_kernel` at the primary target shape (GQA
ratio 6, head_dim 128, page_size 16, seq_len 2048), `--kernel-name
regex:paged_attn` to isolate it from PyTorch's own init/RNG kernels in the
same process.

**batch=1** (the realistic single-request decode scenario):

| metric | value |
|---|---|
| grid | `(1, 2, 1)` — 2 thread blocks, total, on a ~28-SM GPU |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 8.33% |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | 1.27% |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | 5.43% |
| `launch__occupancy_limit_warps` (register/shared-mem based max) | 12 blocks |

Occupancy is capped at 2 blocks not by registers or shared memory
(`launch__occupancy_limit_warps` says up to 12 blocks would fit) but by the
grid itself — `(batch, num_kv_heads) = (1, 2)` only ever creates 2 units of
parallelism, regardless of how much room is left on the SM. This is the
original plan's "~7%" estimate for the naive `(batch, kv_head)` grid,
confirmed by direct measurement (8.33%) rather than assumed.

Stall-reason sampling (`smsp__pcsamp_warps_issue_stalled_*`, average count
across 3 launches): `long_scoreboard` 132.33 (waiting on global memory) is
the dominant stall, ahead of `wait` 107.00, `short_scoreboard` 79.67, and
`barrier` 47.00 — consistent with a memory-bound kernel that doesn't have
enough concurrent warps to hide those stalls, because there are only 2
blocks running in the first place.

**batch=64**, same shape otherwise, for contrast:

| metric | batch=1 | batch=64 |
|---|---|---|
| grid | `(1, 2, 1)` = 2 blocks | `(64, 2, 1)` = 128 blocks |
| `sm__warps_active` (occupancy) | 8.33% | 35.37% |
| `sm__throughput` | 1.27% | 32.20% |
| `dram__throughput` | 5.43% | **95.14%** |

At batch=64 the same kernel — no code changes — reaches 95% of peak DRAM
bandwidth. This confirms the batch=1 numbers are a parallelism-starvation
problem specific to low-batch decode (exactly the real serving scenario:
one request, 2 KV heads under GQA), not a general inefficiency in the
kernel itself. This is the direct, measured motivation for v4's split-K:
the fix isn't "make the kernel faster," it's "give batch=1 more than 2
units of parallelism to schedule."

**Latency** (`bench/bench_decode.py`, fp32 reference-oracle loop vs. fp16
kernel, same shape, cuda-event best-of-30): kernel beats the naive
per-batch-Python-loop reference at every batch size, from 1.9x at batch=1
up to 55.2x at batch=64 — the gap grows with batch because the reference
pays one Python-loop-plus-kernel-launch per sequence while the kernel
processes the whole batch in a single launch. Full sweep in
`bench/results/decode_latency_v1.json`. This is a "custom kernel beats a
naive per-item Python loop" result, not yet a "beats a good baseline"
result — that comparison is what Week 6's FlashInfer benchmark is for.
