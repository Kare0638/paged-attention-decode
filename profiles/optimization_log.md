# Optimization log — before/after summary

A scannable summary of every optimization step in this project, each backed
by real Nsight Compute (NCU) data and `bench_decode.py` latency numbers.
This is an extraction, not a new investigation — every number below traces
back to a measurement already recorded in [`profiles/notes.md`](notes.md),
which remains the source of record (full sweeps, false starts, and
methodology fixes included). Two of the six steps below are **null
results** and one is a genuine crossover reported in both directions —
this project's standing rule is to report losses and non-wins as plainly
as wins, not just the numbers that flatter the final kernel.

## Summary

| step | change | verdict | magnitude |
|---|---|---|---|
| [Triton v1→v2](#triton-v1v2--wider-tiles) | wider tiles (`BLOCK_N` 16→128) | crossover | 1.21x-1.58x low batch, 0.88x-0.96x high batch |
| [Triton v2→v3](#triton-v2v3--num_stages-tuning) | `num_stages` 3→4 | null result | within noise, 6 of 7 batch sizes |
| [Triton v3→v4](#triton-v3v4--split-k) | split-K along sequence dim | crossover | 1.83x-2.00x low batch, 0.75x-0.85x high batch |
| [CUDA v1→v2](#cuda-v1v2--batched-shared-memory-reduction) | batched shared-mem reduction | null result | 0.98x-1.04x, noise-level |
| [CUDA v2→v3](#cuda-v2v3--warp-shuffle-reduction) | warp-shuffle reduction | real win | 1.37x-2.00x, growing with batch |
| [CUDA v3→v4](#cuda-v3v4--split-k) | split-K along sequence dim | crossover | 20.64x at batch=1, parity at batch=64 |

## Starting point

Both Triton v1 and CUDA v1's first NCU profiles found the same root cause:
the naive grid, `(batch, num_kv_heads)`, produces only **2 thread blocks
total** at batch=1 (2 KV heads under this project's GQA config) on a
~28-SM GPU — `sm__warps_active` reads 8.33% for both kernels at batch=1,
and `launch__occupancy_limit_registers`/`_shared_mem` show 8-12 blocks
would fit per SM if there were enough grid parallelism to use them. This
was checked against the alternative explanation (poor memory coalescing)
before being accepted: a sector-efficiency measurement on Triton v1's K/V
gather found it already within ~3% of the theoretical minimum number of
32-byte sectors. The bottleneck is grid size, not access pattern — the
shared diagnosis that motivates both split-K versions below (Triton v4,
CUDA v4), the single largest lever found in each language's optimization
line.

---

## Triton v1→v2 — wider tiles

**Change.** `src/kernel_v2_coalesced.py` reuses v1's exact kernel body;
the only change is decoupling `BLOCK_N` from `page_size` and defaulting it
to 128 (up from v1's 16), the largest tile that fits shared memory at the
primary target shape.

**Hypothesis.** At seq_len 2048 / page_size 16, v1 runs 128 loop
iterations per program, each starting with a `block_table` load that the
K/V load depends on (`long_scoreboard` was v1's dominant stall reason).
Wider tiles mean fewer iterations, amortizing that dependent-load chain
over more data per lookup.

**Latency** (post assert-sync fix, `bench_decode.py`):

| batch | v1 | v2 | v2 vs v1 |
|---|---|---|---|
| 1 | 0.123 ms | 0.078 ms | **1.58x** |
| 16 | 0.143 ms | 0.158 ms | 0.91x |
| 64 | 0.419 ms | 0.443 ms | 0.94x |

**NCU:**

| metric | v1, batch=1 | v2, batch=1 | v1, batch=64 | v2, batch=64 |
|---|---|---|---|---|
| `launch__occupancy_limit_shared_mem` | 8 blocks | **1 block** | 8 blocks | **1 block** |
| `sm__warps_active` | 8.33% | 8.33% | 35.37% | **8.33%** |
| `dram__throughput` | 5.43% | 12.22% | 95.14% | **86.95%** |

**Verdict.** A real crossover, not a strict win. v2's wider tile costs
enough shared memory that only 1 block can be resident per SM (v1 allows
8). At batch=1 there are only 2 blocks total either way, so the
per-block win from fewer dependent loads dominates. At batch=64, v1's 128
blocks pack 8-deep per SM and reach 95% DRAM throughput; v2's are capped
at 1-deep, pinning occupancy at the same 8.33% as batch=1 regardless of
how many blocks are queued — a textbook latency-vs-occupancy tradeoff.
See [`profiles/notes.md`](notes.md#triton-v2--wider-tiles-and-a-real-latency-vs-occupancy-tradeoff-2026-07-30).

## Triton v2→v3 — num_stages tuning

**Change.** `src/kernel_v3_online_softmax.py` reuses v1/v2's kernel body
and `BLOCK_N=128`, adding an explicit `num_stages=4` (Triton's own default
is 3, confirmed from the installed 3.6.0 source, not assumed).

**Hypothesis.** An isolated `num_stages` sweep on the raw kernel body
(bypassing the wrapper) had already shown a real ~11% win at v1's
`BLOCK_N=16` (num_stages 5 vs. Triton's default 3) — the question was
whether that translates into a real wrapper-level win at v2's
`BLOCK_N=128`.

**Latency** (`bench_decode.py`, v3 vs v2, full wrapper):

| batch | v2 | v3 | v3 vs v2 |
|---|---|---|---|
| 1 | 0.0584 ms | 0.0584 ms | 1.00x |
| 16 | 0.1546 ms | 0.1751 ms | 0.88x |
| 64 | 0.4390 ms | 0.4393 ms | 1.00x |

**NCU.** Not re-profiled separately — the isolated sweep that motivated
this step already showed the mechanism: at `BLOCK_N=128`, num_stages=3
(v2's default) already lands close to num_stages=4 (0.052ms vs. 0.047ms
at batch=1), so there was little headroom left to capture at the wrapper
level.

**Verdict.** Null result. Within noise at 6 of 7 batch sizes; the one
batch=8 win (1.35x) doesn't repeat at neighboring batch sizes. Triton's
own default (3) was already close to the swept optimum (4) for this
kernel — reported as "checked, marginal at best," not inflated into a
claimed win. See [`profiles/notes.md`](notes.md#triton-v3--num_stages-formalized-and-it-barely-beats-tritons-own-default-2026-07-30).

## Triton v3→v4 — split-K

**Change.** `src/kernel_v4_split_k.py` adds a third grid dimension,
`num_splits`, over chunks of the sequence: phase 1 (grid `(batch,
num_kv_heads, num_splits)`) computes unnormalized per-chunk `(O, m, l)`
online-softmax partials; phase 2 (grid `(batch, num_kv_heads)`) merges
them. Derivation in
[`analysis/split_k_derivation.md`](../analysis/split_k_derivation.md).
`num_splits=16` set from a fresh sweep (median of interleaved trials,
after an earlier best-of-N sweep proved non-reproducible run to run).

**Hypothesis.** Directly targets the starting-point diagnosis: batch=1
gives phase 1 up to `num_splits`x more grid parallelism than the 2-block
naive grid, at the fixed cost of a second kernel launch and phase 2's
merge.

**Latency** (`bench_decode.py`, v4 vs v1, median of 15 interleaved
trials):

| batch | v4 vs. v1 |
|---|---|
| 1 | **1.83x** |
| 16 | 0.94x-1.00x (crossover) |
| 64 | 0.75x-0.82x |

**NCU** (phase 1 vs. v1, the throughput metrics that actually move):

| metric | v1, batch=1 | phase 1, batch=1 | v1, batch=64 | phase 1, batch=64 |
|---|---|---|---|---|
| grid size | 2 | **32** | 128 | **2048** |
| `dram__throughput` | 5.43% | **37.00%** | 95.14% | **78.43%** |

**Verdict.** Crossover, same shape as v1→v2: a real win concentrated at
low batch (the scenario this kernel exists for — one request, few KV
heads), a real loss at high batch. At batch=1, phase 1's 32 blocks reach
most of the GPU's SMs at once and DRAM throughput jumps 6.8x. At batch=64,
splitting fragments what would be efficient `BLOCK_N=128`-sized transfers
into smaller ones — phase 1's DRAM throughput (78.43%) is actually lower
than v1's (95.14%) at that batch, direct NCU evidence for the regression.
See [`profiles/notes.md`](notes.md#triton-v4--split-k-2026-07-31).

## CUDA v1→v2 — batched shared-memory reduction

**Change.** `cuda/kernel_v2_shared_tile.cu` batches the score reduction
across all `gqa_ratio` query rows into one shared-memory tile per token,
instead of v1's independent block-wide tree reduction per `(row, token)`
pair — an exact 6x reduction in `__syncthreads()` call count at the
primary shape (`gqa_ratio * seq_len * 9` → `seq_len * 9`).

**Hypothesis.** NCU's v1 profile showed `barrier` as a top-3 stall
category; cutting `__syncthreads()` call count 6x should cut latency
accordingly.

**Latency** (`bench_decode.py`, cuda_v2 vs cuda_v1, all batch sizes):

0.98x-1.04x — noise-level at every batch size tested, no systematic
direction.

**NCU** (batch=1):

| metric | CUDA v1 | CUDA v2 (padded) |
|---|---|---|
| `gpu__time_duration.sum` | 8.79 ms | 8.85 ms |
| `smsp__inst_executed.sum` | 11,635,480 | 9,181,968 (-21%) |
| `smsp__inst_executed_op_shared_ld.sum` | 1,474,560 | 573,440 (-61%) |
| `...issue_stalled_barrier.ratio` | 2,363,041 | **4,782,348 (higher)** |

**Verdict.** Null result — and the hypothesis was disproved, not just
unconfirmed. v2 genuinely executes 21% fewer instructions and 61% less
shared-memory traffic, yet wall-clock time doesn't move, and the
`barrier` stall ratio goes *up*, the opposite of the predicted direction:
fewer, longer barrier waits (each now serializes 6x more work inside the
`for row` loop) instead of more, shorter ones, netting out to the same
total stall time. This kernel is latency-bound by dependency-chain
*length* (9 sequentially-dependent stages, unchanged), not instruction
*count* — the real, evidence-backed case for v3's warp-shuffle reduction,
which changes the chain structure itself. See
[`profiles/notes.md`](notes.md#cuda-v2--batched-shared-memory-reduction-2026-08-04).

## CUDA v2→v3 — warp-shuffle reduction

**Change.** `cuda/kernel_v3_warp_shuffle.cu` replaces the
tree-reduction-plus-`__syncthreads()` algorithm with warp-shuffle
(`__shfl_down_sync`/`__shfl_sync`) — a different reduction *algorithm*,
not another grouping of the same one. Forces `blockDim(32)` (one warp per
block) instead of v1/v2's `blockDim(head_dim)`.

**Hypothesis.** Two, stated going in: (1) collapsing the reduction from 9
`__syncthreads()`-bound stages to 6 dependent shuffle instructions
shortens the critical path directly; (2) 4x fewer threads/block should
relieve register pressure and raise occupancy.

**Latency** (`bench_decode.py`, cuda_v3 vs cuda_v1):

| batch | cuda_v1 | cuda_v3 | cuda_v3 vs. cuda_v1 |
|---|---|---|---|
| 1 | 5.877 ms | 4.287 ms | 1.37x |
| 16 | 6.865 ms | 4.746 ms | 1.45x |
| 64 | 11.558 ms | 5.791 ms | **2.00x** |

**NCU:**

| metric | v1, batch=1 | v3, batch=1 | v1, batch=64 | v3, batch=64 |
|---|---|---|---|---|
| `smsp__inst_executed.sum` | 11,635,480 | **2,746,770 (-76%)** | 744,670,720 | **175,793,280 (-76%)** |
| `launch__occupancy_limit_registers` | 12 blocks | **48 blocks** | 12 blocks | **48 blocks** |
| `sm__warps_active` (achieved) | 8.33% | **2.08% (lower)** | 34.80% | **8.81% (lower)** |

**Verdict.** A real, mechanistically-understood win — but not for the
reason hypothesis 2 predicted. Hypothesis 1 confirmed cleanly:
instruction count drops ~4.2x at both batch sizes, explaining the win
even at batch=1 (grid size unchanged from v1, ruling out occupancy
there). Hypothesis 2 was directionally right but backwards on the
detail that matters: the occupancy *ceiling* does rise 12→48 blocks/SM
(register pressure relieved), but *achieved* occupancy is actually
*lower* than v1 at every batch tested, because each resident block now
carries only 1 warp instead of 4. v3 is faster with strictly lower
measured occupancy — the win traces to the shorter dependency chain, not
to higher occupancy. See
[`profiles/notes.md`](notes.md#cuda-v3--warp-shuffle-reduction-2026-08-04).

## CUDA v3→v4 — split-K

**Change.** `cuda/kernel_v4_split_k.cu` adds the same third grid
dimension as Triton v4, reusing
[`analysis/split_k_derivation.md`](../analysis/split_k_derivation.md)'s
merge math directly — but built on **CUDA v3's warp-shuffle reduction**,
not v1's tree reduction (the best per-block building block available,
not the first one). `num_splits=64` set from a fresh sweep against CUDA
v3 (not reused from Triton v4's `num_splits=16` — v3's per-block cost
profile differs enough that the sweet spot could plausibly land
elsewhere, and it did).

**Hypothesis.** Same starting-point diagnosis as Triton v4, applied to a
much lower per-block baseline cost (v3 already cut instructions ~4.2x vs.
v1) — an open question stated in the plan before measuring: does
split-K's fixed overhead (second launch, intermediate buffers) still pay
off with less per-block cost to hide behind?

**Latency** (`bench_decode.py`, cuda_v4 vs cuda_v3):

| batch | cuda_v3 | cuda_v4 | cuda_v4 vs. cuda_v3 |
|---|---|---|---|
| 1 | 4.373 ms | 0.212 ms | **20.64x** |
| 16 | 4.789 ms | 1.538 ms | 3.11x |
| 64 | 6.658 ms | 6.575 ms | 1.01x |

**NCU** (phase 1 vs. phase 2, batch=1):

| metric | phase 1, batch=1 | phase 2, batch=1 |
|---|---|---|
| grid | (1, 2, 64) = 128 | (1, 2, 1) = 2 |
| `gpu__time_duration.sum` | 133.6 us | **233.5 us (more than phase 1)** |
| `dram__throughput` | 11.11% | 2.39% |

**Verdict.** Crossover, same shape as the Triton split-K step, but a far
larger low-batch win (20.64x vs. Triton v4's 1.83x) because CUDA v3's
batch=1 baseline had far more idle parallelism to reclaim in absolute
terms. An honest surprise on top of the expected shape: at batch=1,
phase 2 — designed to be the "cheap" `O(num_splits)` half — actually
costs *more* than phase 1, because phase 2's grid never grows with
`num_splits` (still the same 2-block starvation problem split-K exists to
fix), so at `num_splits=64` each of its 2 blocks runs `64 * gqa_ratio(6) =
384` sequential merge iterations. This is also the mechanistic reason the
`num_splits` sweep peaked at 64 and declined by 128, unlike Triton v4's
flat plateau. Even with this win, CUDA v4 still trails Triton v4 by
roughly an order of magnitude in absolute terms — real, substantial
progress across the CUDA line (CUDA v1 was ~44x slower than Triton v1;
CUDA v4 is ~2.8x slower than Triton v4), not a claim of parity. See
[`profiles/notes.md`](notes.md#cuda-v4--split-k-2026-08-05).
