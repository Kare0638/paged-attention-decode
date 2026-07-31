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

**Memory-coalescing efficiency check, batch=1** — before assuming v2
("coalesced KV access") has an obvious win waiting, measured whether v1's
access pattern is actually poorly coalesced, rather than guessing from the
low batch=1 `dram__throughput` (5.43%) alone — that number conflates
*coalescing efficiency* with *not enough concurrent requests to saturate
the bus*, and only one of those is fixable by reorganizing the memory
access:

| metric | value |
|---|---|
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` (measured) | 67,688 sectors |
| theoretical minimum (`2 × seq_len × head_dim × 2B × 2(K,V) / 32B`) | 65,536 sectors |
| overhead | 3.28% (96.8% sector efficiency) |
| `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` | 2.17 MB vs. 2.10 MB theoretical minimum |

v1's K/V gather is already within ~3% of the theoretical minimum number of
32-byte sectors, despite going through the `block_table` indirection. The
reason: `k_cache`/`v_cache` are laid out `[num_pages, page_size,
num_kv_heads, head_dim]`, so for a fixed kv-head, each individual
(page, slot) read is a 128-element (256-byte) contiguous run of
`head_dim` that exactly fills 8 sectors with nothing left over — the two
KV heads are interleaved *between* slots, not *within* one, so the
interleaving never fragments a single read below the sector boundary.

**Implication for v2**: the batch=1 DRAM-throughput number (5.43%) is not
telling us the access pattern is uncoalesced — sector efficiency is
already ~97%. It's telling us the same thing the occupancy numbers already
did: 2 blocks can't issue enough concurrent memory requests to keep the
bus busy, no matter how clean each individual request is. Reorganizing the
memory layout for "better coalescing" would be chasing a problem that
mostly isn't there; v2 needs a different, still-to-be-identified angle
(or the honest finding may be that this particular optimization step
doesn't have much room on this workload, which is itself a legitimate
result to report rather than force a win to match the original roadmap
description).

**Latency** (`bench/bench_decode.py`, fp32 reference-oracle loop vs. fp16
kernel, same shape, cuda-event best-of-30): kernel beats the naive
per-batch-Python-loop reference at every batch size, from 1.9x at batch=1
up to 55.2x at batch=64 — the gap grows with batch because the reference
pays one Python-loop-plus-kernel-launch per sequence while the kernel
processes the whole batch in a single launch. This is a "custom kernel
beats a naive per-item Python loop" result, not yet a "beats a good
baseline" result — that comparison is what Week 6's FlashInfer benchmark
is for.

## Triton v2 — wider tiles, and a real latency-vs-occupancy tradeoff (2026-07-30)

v2 (`src/kernel_v2_coalesced.py`) reuses v1's exact kernel body — the only
change is decoupling `BLOCK_N` from `page_size` and defaulting it to 128
(the largest tile that fits this GPU's shared memory at the primary target
shape; 256 fails with `OutOfResources: out of resource: shared memory,
Required: 138240, Hardware limit: 101376`). This came from measuring where
v1's actual headroom was, not from assuming the roadmap's original
"coalesced KV access" framing: v1's total-sector efficiency was already
~97% of the theoretical minimum, and a `num_warps` sweep found Triton's
default (4) was already optimal at both batch=1 and batch=64. What *did*
move the needle was `long_scoreboard` being the dominant stall — at
seq_len 2048 / page_size 16, v1 runs 128 loop iterations per program, each
starting with a `block_table` load that the K/V load depends on. Wider
tiles mean fewer iterations, amortizing that dependent-load chain over
more data per lookup.

**Latency** (`bench/bench_decode.py`, v2 vs. v1, same shape, cuda-event
best-of-30):

| batch | v1 | v2 | v2 vs v1 |
|---|---|---|---|
| 1 | 0.3348 ms | 0.2775 ms | **1.21x** |
| 2 | 0.2508 ms | 0.1718 ms | **1.46x** |
| 4 | 0.2488 ms | 0.1649 ms | **1.51x** |
| 8 | 0.2416 ms | 0.1772 ms | **1.36x** |
| 16 | 0.2499 ms | 0.2456 ms | 1.02x |
| 32 | 0.3195 ms | 0.3553 ms | **0.90x** |
| 64 | 0.5120 ms | 0.5324 ms | **0.96x** |

This is not a clean win — it's a crossover, and the honest thing is to
report both sides rather than only the batches where v2 looks good. v2 is
faster at low batch (up to 1.51x at batch=4) and *slower* than v1 at
batch=32/64. NCU explains why:

| metric | v1, batch=1 | v2, batch=1 | v1, batch=64 | v2, batch=64 |
|---|---|---|---|---|
| `launch__occupancy_limit_shared_mem` | 8 blocks | **1 block** | 8 blocks | **1 block** |
| `sm__warps_active` (occupancy) | 8.33% | 8.33% | 35.37% | **8.33%** |
| `dram__throughput` | 5.43% | 12.22% | 95.14% | **86.95%** |

v2's wider tile uses enough shared memory that only **1 block can be
resident per SM at a time** (v1 allows 8). At batch=1 this doesn't cost
anything — there are only 2 blocks total either way, nowhere near enough
to fill ~28 SMs regardless of the per-SM limit — so v2's shorter, less
latency-bound loop wins outright (12.22% vs 5.43% DRAM throughput, more
than 2x). At batch=64, v1's 128 blocks can pack 8-deep per SM and reach
35.37% occupancy / 95.14% DRAM throughput; v2's 128 blocks are capped at
1-deep per SM, so occupancy stays pinned at the same 8.33% as batch=1 no
matter how many blocks are queued, and DRAM throughput actually drops
versus v1 (86.95% vs 95.14%). This is a textbook latency-vs-occupancy
tradeoff, not a bug: fewer, larger memory transactions per block reduce
per-block latency, but the larger shared-memory footprint that makes that
possible reduces how many blocks can run concurrently — which one wins
depends on whether the workload already has enough blocks to fill the GPU
without help.

**Conclusion**: v2 is the better choice specifically for the
low-batch/single-request decode scenario this project's split-K story
(v4) is about — exactly where it matters most for real serving. It is not
a strict improvement over v1 and shouldn't be presented as one; the
roadmap's original "v2 = coalesced KV access, unconditionally faster"
framing doesn't survive contact with the data, and the batch=32/64
regression is the more interesting finding of the two.
