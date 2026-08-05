# Paged Attention Decode Kernel

Decode-phase attention kernel for LLM inference with paged KV cache and GQA support — being implemented from scratch in **Triton** and **CUDA C++**, optimized against Nsight Compute profiling data on a low-SM consumer GPU.

## Status

Triton v1–v4 (naive, wider tiles, num_stages tuning, split-K) and CUDA C++ v1–v4 (naive baseline, batched shared-memory reduction, warp-shuffle reduction, split-K, all as `torch.utils.cpp_extension` extensions) are implemented, tested, and benchmarked — see Results below. This closes out the CUDA roadmap section. CUDA v4's split-K delivers the single largest win in the CUDA line (up to 20.64x vs. CUDA v3 at batch=1) but CUDA still trails Triton by roughly an order of magnitude in absolute terms — real, substantial progress across four versions (CUDA v1 was ~44x slower than Triton v1; CUDA v4 is ~2.8x slower than Triton v4), not a claim of parity, reported honestly in both directions. What remains: consolidating the Nsight optimization log, the FlashInfer/A100 comparison, and a roofline plot. This README gets updated with real benchmark numbers as each version ships — no numbers are reported until they're measured.

## Why this project

Decode-phase attention is memory-bandwidth-bound, not compute-bound: for a Qwen2.5-1.5B-style config (GQA ratio 6, head_dim 128, fp16 KV cache), the arithmetic intensity works out to ~6 FLOP/byte, far below this RTX 3060 Laptop's **measured** ridge point of ~84 FLOP/byte (26.4 TFLOPS fp16 matmul / 314 GB/s D2D — both measured on this machine, not nominal spec-sheet numbers; see [bench/results/](bench/results/)) — so almost every optimization here is about memory access patterns and occupancy, not raw FLOPs.

On low-SM-count GPUs, naive decode kernels that grid over `(batch, kv_head)` leave most SMs idle when batch size and KV head count are both small under GQA (e.g. 2 KV heads on a ~28-SM GPU). This project implements split-K (FlashDecoding-style) parallelism along the sequence dimension to address that, with every optimization step backed by Nsight Compute data rather than guesswork.

## Design

**Target model config** (Qwen2.5-1.5B, verified against `config.json`):

| | |
|---|---|
| Q heads / KV heads | 12 / 2 (GQA ratio 6) |
| head_dim | 128 |
| dtype | fp16 compute, fp32 accumulate |
| page_size | 16 (primary), 32 (comparison) |

**Paged KV cache layout**

```
kv_cache:    [num_pages, page_size, num_kv_heads, head_dim]
block_table: [batch, max_pages_per_seq]   # logical page -> physical page
seq_lens:    [batch]
```

**Split-K reduction** — the sequence dimension is split into chunks; phase 1 computes per-chunk partial `(O, m, l)` online-softmax statistics, phase 2 reduces across chunks. Derivation lives in [analysis/split_k_derivation.md](analysis/split_k_derivation.md).

## Roadmap

- [x] Correctness: fp32 PyTorch reference implementation
- [x] Triton v1: naive, paged KV indexing
- [x] Triton v2: wider tiles (decoupled from page_size) — see Results below; not a strict win over v1
- [x] Triton v3: single-pass online softmax — already true of v1/v2's design (the running-softmax loop is structurally required at real seq_len, not a separate feature). `src/kernel_v3_online_softmax.py` formalizes the one thing left to try, an explicit `num_stages=4` pin (Triton's own default is 3, confirmed from the installed 3.6.0 source), but a real A/B through the wrapper shows it within noise of v2's default at 6 of 7 batch sizes — reported as "checked, marginal at best," not inflated into a claimed win. Investigating it surfaced a bigger, unrelated finding: both v1/v2's wrappers had an `assert` on a GPU tensor's value that forced a device-to-host sync, costing 2-3x the raw kernel latency at batch=1 — fixed, and every latency number below reflects the fix.
- [x] Triton v4: split-K along the sequence dimension — real win at low batch (1.83x vs. v1 at batch=1), real loss at high batch (0.75x-0.85x at 32/64), same tradeoff shape as v2, reported in both directions — see Results below
- [x] CUDA C++: explicit shared-memory tiling, bank-conflict elimination — `cuda/kernel_v2_shared_tile.cu` batches the score reduction across all GQA rows into one shared-memory tile per token instead of one reduction per row; correctness verified (bit-exact vs. CUDA v1), instruction count and shared-memory traffic both measurably lower, but **no latency win** (0.98x-1.04x vs. CUDA v1, noise-level) — the kernel is latency-bound by dependency-chain length, not instruction count, so cutting sync-call count without shortening the chain doesn't help. Bank-conflict elimination checked via a genuine padded-vs-unpadded A/B, not assumed: 0 conflicts in both — see Results below
- [x] CUDA C++: warp-shuffle reduction, occupancy analysis — `cuda/kernel_v3_warp_shuffle.cu` replaces the tree-reduction-plus-`__syncthreads()` algorithm with warp-shuffle (one warp per block instead of one block spanning head_dim), a real, mechanistically-understood win this time: **1.37x-2.00x vs. CUDA v1**, growing with batch, from a ~4.2x drop in total instructions (no shared memory at all). Occupancy story is more nuanced than "more warp-shuffle = more occupancy": the register-bound ceiling rises 12→48 blocks/SM, but *achieved* occupancy (`sm__warps_active`) is actually lower than v1 at every batch tested (2.08% vs. 8.33% at batch=1) since each block now carries 1 warp instead of 4 — the speedup traces to the shorter dependency chain, not higher occupancy — see Results below
- [x] CUDA C++: split-K, packaged as a PyTorch extension — `cuda/kernel_v4_split_k.cu` adds a third grid dimension over sequence chunks, built on CUDA v3's warp-shuffle reduction (not v1's), reusing `analysis/split_k_derivation.md`'s merge math directly. `num_splits=64` (swept fresh against CUDA v3, not reused from Triton v4's `num_splits=16`) delivers a real **20.64x win vs. CUDA v3 at batch=1**, the largest single improvement in this project's CUDA line, shrinking to parity by batch=64 (same tradeoff shape as Triton v2/v4, much larger magnitude) — see Results below for the honest reality check against Triton v4
- [x] Correctness test suite for the reference implementation (page boundaries, extreme/ragged/zero-length sequences, non-contiguous block tables with unreferenced "holes", GQA ratios 1/4/6/8, full input-validation coverage, 2000-case randomized fuzz vs. an independently-written SDPA oracle) plus kernel-vs-reference suites for Triton v1, v2, v3, v4, and CUDA v1, v2, v3, v4 (302 cases at default settings, `tests/`), including both split-K versions' split-invariance checks, a bit-exact `num_splits=1`-vs-v1 equivalence test (Triton), a bit-exact CUDA v1-vs-v2 equivalence test, a close (~6e-8) CUDA v1-vs-v3 equivalence test, and a bit-exact `num_splits=1`-vs-CUDA-v3 equivalence test.
- [ ] Nsight-driven optimization log with before/after profiles
- [ ] Benchmark against FlashInfer (A100)
- [ ] Roofline plot using measured (not nominal) peak bandwidth

## Results

**v1 (Triton, naive)** — GQA ratio 6, head_dim 128, page_size 16, seq_len 2048:

- Latency vs. the naive per-batch reference loop: 5.6x at batch=1, up to
  72.9x at batch=64 — this beats a naive Python loop, it is not yet a
  claim of beating a strong baseline; that comparison (vs. FlashInfer) is
  Week 6's job.
- NCU at batch=1 (the realistic single-request decode case): grid is
  `(1, 2, 1)` — only 2 thread blocks on a ~28-SM GPU, 8.33% occupancy,
  `long_scoreboard` (memory-wait) is the dominant stall reason. At
  batch=64 the *same* kernel hits 95.14% of peak DRAM bandwidth, showing
  batch=1's low numbers are a parallelism problem specific to low-batch
  decode, not a general kernel inefficiency — the direct, measured
  motivation for v4's split-K.

**v2 (Triton, wider tiles)** — before writing this, measured whether v1
actually had a memory-coalescing problem rather than assuming the roadmap's
original framing: total-sector efficiency was already ~97% of theoretical
minimum, and sweeping `num_warps` found Triton's own default was already
optimal. What did move the needle: v1 ties its tile size to `page_size`
(16), so at seq_len 2048 it runs 128 loop iterations, each starting with a
`block_table` load the K/V load depends on — matching `long_scoreboard`
being the dominant stall. v2 decouples the tile size from `page_size`
(default 128, the largest that fits this GPU's shared memory here), same
kernel body, fewer/larger iterations.

**Not a strict win — a latency-vs-occupancy tradeoff, reported honestly in
both directions:**

| batch | v1 | v2 | v2 vs v1 |
|---|---|---|---|
| 1 | 0.123 ms | 0.078 ms | **1.58x** |
| 4 | 0.146 ms | 0.065 ms | **2.27x** |
| 16 | 0.143 ms | 0.158 ms | 0.91x |
| 64 | 0.419 ms | 0.443 ms | **0.94x** |

v2's wider tile needs enough shared memory that only 1 block can be
resident per SM at once (v1 allows 8 — `launch__occupancy_limit_shared_mem`).
At batch=1 that costs nothing (only 2 blocks exist either way), so v2's
shorter, less latency-bound loop wins outright. At batch=64, v1 packs 8
blocks deep per SM and hits 95.14% DRAM throughput; v2 is capped at 1 deep,
stays pinned at the same 8.33% occupancy as batch=1, and DRAM throughput
actually drops to 86.95%. v2 wins specifically in the low-batch/
single-request regime this project's split-K story is about — not
universally, and the roadmap's original "v2 = coalesced access, strictly
better" framing doesn't survive contact with the data. Full numbers and
mechanism in [profiles/notes.md](profiles/notes.md);
[bench/results/decode_latency.json](bench/results/decode_latency.json)
has the full batch sweep.

**Measurement fix that changed every number above**: while investigating
v3's `num_stages` lever, isolating each step of the wrapper call found
`assert torch.all(seq_lens >= 1)` — added deliberately to catch a real
silently-wrong-answer risk — was forcing a device-to-host sync (Python's
`assert` needs the CUDA tensor's `__bool__()`, which blocks on the GPU).
That cost 2-3x the raw kernel launch time at batch=1, confirmed
reproducible across independent runs. Removed from both wrappers'
hot path; the precondition is still documented and still exercised by the
test suite, just not paid for on every call. Every number in this README
and in `profiles/notes.md` reflects the fix — the lesson generalizes past
this one assert: any check on a GPU tensor's *value* (not just its
shape/dtype/device) is a sync point, and at sub-millisecond kernel
latencies that sync can dominate the number being measured.

**v3 (Triton, num_stages pinned)** — same kernel body and tile size as
v2, `num_stages=4` explicit instead of Triton's default (3, confirmed
from the installed 3.6.0 source, not assumed). A real A/B through the
wrapper (v3 vs. v2, same shape) landed within noise at 6 of 7 batch
sizes (0.98x-1.00x, one outlier of 1.35x at batch=8 that doesn't repeat
at neighboring batches) — consistent with an isolated sweep showing 3
and 4 close together (0.052 vs. 0.047 ms at batch=1), the same
conclusion the `num_warps` sweep reached before v2 existed. Reported
honestly as a lever that was checked and found marginal, not as a win.
Of the
levers tried so far, only v1->v2's tile-size change had a real,
consistent, mechanistically-understood effect; the next one with an
a priori large effect is v4's split-K — batch=1 is still capped at 2
thread blocks on a ~28-SM GPU no matter how any of v1/v2/v3 are tuned.

**v4 (Triton, split-K)** — grid over `(batch, num_kv_heads)` alone caps
batch=1 at 2 thread blocks on a ~28-SM GPU no matter how v1/v2/v3 are
tuned. v4 adds a third grid dimension over chunks of the sequence: phase
1 computes unnormalized partial `(O, m, l)` per `(batch, kv_head,
split)`, phase 2 reduces across splits (derivation in
[analysis/split_k_derivation.md](analysis/split_k_derivation.md)).
Double-checked beyond the usual reference comparison: `num_splits=1` at
`block_n=16` (matching v1's tile exactly) reproduces v1's output
**bit-for-bit**, confirming split-K is a pure reassociation of v1's math.

`num_splits` has no default until `bench/bench_v4_num_splits.py`'s sweep
at batch=1 set one — same discipline as v2's `block_n=128` and v3's
`num_stages=4`. Unlike those two, the result is a **broad, flat plateau**
(num_splits 2 through 64 all within ~10% of each other), not one sharp
peak; `num_splits=16` sits in the middle of it. (Getting a stable sweep
at all required fixing the benchmark's own methodology first — at these
sub-0.1ms latencies, single best-of-N readings were dominated by
run-to-run GPU clock noise on this laptop GPU, not real differences
between configs; fixed by switching to median-of-interleaved-trials,
reproduced across independent reruns. Detail in
[profiles/notes.md](profiles/notes.md).)

**Same tradeoff shape as v2 — a real win at low batch, a real loss at
high batch, both reported:**

| batch | v4 vs. v1 |
|---|---|
| 1 | **1.83x** |
| 4 | 1.49x–1.73x |
| 16 | 0.94x–1.00x (crossover) |
| 64 | 0.75x–0.82x |

NCU explains why, with a methodological catch worth flagging: phase 1's
`sm__warps_active` reads 8.33% at batch=1 — identical to v1's number —
which looks like occupancy didn't improve. It's normalized to the
kernel's *own* per-SM ceiling (shared memory still caps 1 block/SM, same
`block_n=128` tile as v2), not to whole-GPU utilization, so it stays
flat whenever that per-block ceiling doesn't change. The metrics that
actually show more of the GPU getting used are throughput-based: at
batch=1, phase 1's 32 blocks (vs. v1's 2) push `dram__throughput` from
5.43% to **37.00%** and `sm__throughput` from 1.27% to **7.21%**. At
batch=64, phase 1's `dram__throughput` (78.43%) is *lower* than v1's at
the same batch (95.14%) — splitting fragments what would otherwise be
efficient transfers into more, smaller ones, and that cost isn't worth
paying once batch alone already supplies enough parallelism. Full
mechanism, phase 1 vs. phase 2 breakdown, and the num_splits sweep table
in [profiles/notes.md](profiles/notes.md);
[bench/results/v4_num_splits_sweep.json](bench/results/v4_num_splits_sweep.json)
and [bench/results/decode_latency.json](bench/results/decode_latency.json)
have the full data.

**CUDA v1 (naive baseline)** — the first CUDA C++ kernel in this project,
and the first kernel here built via `torch.utils.cpp_extension` (JIT)
rather than Triton's own JIT; the pipeline itself was smoke-tested with a
throwaway add-kernel before any real logic depended on it. A direct port
of `_paged_attn_decode_v1_kernel`'s algorithm (same grid, same
online-softmax recurrence) but deliberately more naive: no page-tile
batching (token-by-token dot products via a block-wide shared-memory tree
reduction, instead of Triton's tiled `tl.dot`). K/V is already loaded once
per token into a register and reused across all `gqa_ratio` rows — the
real per-row-repeated cost is the reduction itself, run independently
`gqa_ratio` times per token instead of batched across rows. That gap is
intentional — it's the baseline CUDA v2's shared-memory tiling roadmap
item gets measured against, the same role Triton v1 played for Triton v2.

Correctness: same shape matrix and tolerance as Triton v1's test suite
(`tests/test_kernel_cuda_v1.py`, 31 cases, all passing). Unlike Triton,
CUDA has no `tl.arange` power-of-2 constraint and no `tl.dot` K>=16 floor,
so this kernel's input validation is a genuine subset of Triton v1's.

| batch | Triton v1 | CUDA v1 | CUDA v1 vs. Triton v1 |
|---|---|---|---|
| 1 | 0.131 ms | 6.008 ms | 0.02x (~46x slower) |
| 16 | 0.175 ms | 6.960 ms | 0.03x (~40x slower) |
| 64 | 0.423 ms | 13.348 ms | 0.03x (~32x slower) |

NCU shows a genuinely different bottleneck than Triton v1, not the same
one measured worse: Triton v1 is memory-latency-bound
(`long_scoreboard` dominant). CUDA v1's dominant stall is `wait`
(fixed-latency math-pipe, the per-token `__expf`/division calls), followed
by `short_scoreboard` (shared-memory round trips) and `barrier`
(`__syncthreads()`) — `long_scoreboard` is CUDA v1's *smallest* nonzero
stall category. At batch=1, `dram__throughput` reads 0.92% (this kernel
isn't memory-bound at all) while it issues 1.47M shared-memory load
instructions from just 2 thread blocks — one full block-wide
reduction-and-barrier per token per row, `gqa_ratio * seq_len` times per
block. That instruction/sync volume, not bandwidth, is the cost v2's
batched shared-memory reduction is expected to amortize away. Bank
conflicts (`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`)
measure **0** at batch=1 — the reduction's own stride-1 addressing is
conflict-free by construction, unlike Week 0's deliberately-strided test
kernel — so bank-conflict elimination may not be a real lever here at
all; v2 checks it explicitly with a padded-vs-unpadded A/B rather than
skipping it on that prediction. Full mechanism and the batch=64
numbers in [profiles/notes.md](profiles/notes.md).

**CUDA v2 (batched shared-memory reduction)** — batches the score
reduction across all `gqa_ratio` query rows into one shared-memory tile
per token, instead of CUDA v1's one independent block-wide tree reduction
per `(row, token)` pair. Hypothesis: cutting `__syncthreads()` call count
~6x (from `gqa_ratio * seq_len * 9` to `seq_len * 9`) should cut latency
accordingly, since `barrier` was a top-3 stall category in CUDA v1's
profile.

Correctness: 32 cases pass (`tests/test_kernel_cuda_v2.py`), including a
direct bit-exact comparison against CUDA v1 (max diff 0.0) confirming v2
is a pure reassociation of v1's math.

**The hypothesis was wrong — reported as measured, not adjusted after the
fact.** `cuda_v2` vs. `cuda_v1` lands at 0.98x-1.04x across every batch
size — noise-level, no real win, despite `smsp__inst_executed.sum`
dropping 21% and shared-memory load/store instruction counts dropping
56-61%. NCU explains why: `smsp__average_warp_latency_issue_stalled_
barrier.ratio` is *higher* in v2 (4.78M) than v1 (2.36M), the opposite of
the predicted direction — fewer `__syncthreads()` calls, but each one now
waits on 6x more serialized work before it can release, so total barrier
stall time doesn't drop. Both `sm__throughput` and `dram__throughput`
stay near-zero in both versions at batch=1: this kernel is bound by the
*length* of its serialized dependency chain (still 9 sequentially-
dependent stages per token in both versions — only the width of each
stage changed), not by instruction count or bandwidth. Checked from three
independent angles (wall-clock, NCU-measured kernel duration,
instruction/stall-reason counts) — all agree it's a genuine null result,
not a methodology artifact.

Bank-conflict A/B (padded `row_stride=head_dim+1` vs. unpadded
`row_stride=head_dim`, same templated-kernel technique as Week 0): **0
conflicts in both**, no measurable latency difference at batch=64 (13.74
ms vs. 13.56 ms) — confirms the prediction from CUDA v1's own finding
(head_dim is always the per-thread lane; row/token axes are always looped
serially within a thread, never spread across a warp) rather than
skipping the check. Occupancy: the wider tile drops
`launch__occupancy_limit_shared_mem` from 21 to 15 blocks/SM, but
registers stay the binding ceiling at 12 in both — no occupancy cost,
unlike Triton v2's real regression.

Bottom line: shared-memory tiling implemented and correctness-verified,
bank-conflict elimination checked and confirmed unnecessary for this
access pattern — but not a latency win. The real bottleneck is
dependency-chain length, which is what v3's warp-shuffle reduction (a
different reduction algorithm, not just a different grouping of the same
tree reduction) targets next. Full data in
[profiles/notes.md](profiles/notes.md).

**CUDA v3 (warp-shuffle reduction)** — replaces v1/v2's tree-reduction-
plus-`__syncthreads()` algorithm with warp-shuffle (`__shfl_down_sync`/
`__shfl_sync`): no shared memory, no explicit block-wide barrier at all,
reduction happens in registers via warp-synchronous shuffle instructions.
Forces one warp per block (`blockDim=32`) instead of v1/v2's
`blockDim=head_dim`, so each thread now owns `head_dim/32` lanes — a new,
honestly-documented precondition (`head_dim % 32 == 0`) specific to this
design, satisfied by both head_dim values (32, 128) this project's CUDA
test suite exercises.

Correctness: 32 cases pass (`tests/test_kernel_cuda_v3.py`). v1-vs-v3
comparison (different floating-point reduction order, not bit-exact like
v2 was): measured max diff ~6e-8 — `rtol=1e-4/atol=1e-5` set from that
measurement, not guessed.

**This time the hypothesis holds — a real, mechanistically-understood
win, not another null result:**

| batch | CUDA v1 | CUDA v3 | CUDA v3 vs. CUDA v1 |
|---|---|---|---|
| 1 | 5.877 ms | 4.287 ms | 1.37x |
| 16 | 6.865 ms | 4.746 ms | 1.45x |
| 64 | 11.558 ms | 5.791 ms | 2.00x |

NCU shows why, with a result more nuanced than either half of the
hypothesis predicted alone. Instruction count drops ~4.2x at *both*
batch=1 and batch=64 (`smsp__inst_executed.sum`) — a consistent ratio,
confirming the shorter-critical-path hypothesis independent of grid size
(batch=1's grid is still just 2 blocks, unchanged from v1/v2, ruling out
an occupancy-driven explanation there). The occupancy story is real but
not the one guessed: `launch__occupancy_limit_registers` jumps 12→48
blocks/SM (register pressure relieved by 4x fewer threads/block), so the
hardware's fixed 16-blocks/SM ceiling becomes binding instead — a real
~33% ceiling increase — but **achieved occupancy
(`sm__warps_active`) is *lower* for v3 than v1 at every batch tested**
(2.08% vs. 8.33% at batch=1; 8.81% vs. 34.80% at batch=64), since each
resident block now carries only 1 warp instead of 4. v3 is faster with
strictly lower measured occupancy at every batch size — the same
"achieved occupancy isn't the whole story" lesson already documented for
Triton v4's phase 1, confirmed again from the opposite direction. The
speedup traces mainly to the shorter dependency chain, not to higher
occupancy; the growing margin with batch (1.37x→2.00x) is plausibly more
of v3's smaller blocks fitting concurrently as grid size grows, stacking
on the batch-independent instruction-count win — a reasonable reading of
the data, not independently isolated beyond what's shown here. Bank
conflicts: 0 at both batches — there's no shared memory left in this
kernel for Week 0's padding technique to apply to. Full data in
[profiles/notes.md](profiles/notes.md).

**CUDA v4 (split-K)** — the last CUDA roadmap item. Adds a third grid
dimension over sequence chunks, the same fix already applied once for
Triton (`analysis/split_k_derivation.md`, reused directly — the merge
math is language-agnostic, no new derivation needed). Built on **CUDA
v3's warp-shuffle reduction**, not v1's tree reduction — the best
per-block building block available now, not the first one — so the
primary comparison here is against CUDA v3.

Correctness: 34 cases pass (`tests/test_kernel_cuda_v4.py`). `num_splits=1`
vs. CUDA v3 measured bit-exact (max diff 0.0) before picking a tolerance
— phase 1 at `num_splits=1` is byte-for-byte v3's loop, phase 2
degenerates to a single term.

`num_splits` swept fresh against CUDA v3 (`bench/bench_cuda_v4_num_splits.py`,
batch=1), not reused from Triton v4's `num_splits=16`: a **sharp peak at
64** (20.75x vs. CUDA v3), not Triton's broad flat plateau — dropping off
by 128. Set as the wrapper default.

**The magnitude here dwarfs every prior version in this project, Triton
or CUDA — because CUDA v3's batch=1 baseline had far more idle
parallelism left to reclaim:**

| batch | CUDA v3 | CUDA v4 | CUDA v4 vs. CUDA v3 |
|---|---|---|---|
| 1 | 4.373 ms | 0.212 ms | **20.64x** |
| 16 | 4.789 ms | 1.538 ms | 3.11x |
| 64 | 6.658 ms | 6.575 ms | 1.01x |

Same tradeoff shape as Triton v2/v4 and CUDA v2's comparisons (real win
at low batch, shrinking to parity at high batch), a much larger low-batch
win than any prior version.

**The honest reality check, committed to before measuring**: CUDA v4
still trails Triton v4 substantially in absolute terms —

| batch | Triton v4 | CUDA v4 | CUDA v4 vs. Triton v4 |
|---|---|---|---|
| 1 | 0.077 ms | 0.212 ms | 0.36x (~2.8x slower) |
| 64 | 0.546 ms | 6.575 ms | 0.08x (~12.5x slower) |

Not competitive against Triton in absolute terms — but the *relative*
gap has closed enormously across the CUDA line: CUDA v1 was ~44x slower
than Triton v1 at batch=1; CUDA v4 is ~2.8x slower than Triton v4 at the
same batch, and only ~1.6x slower than **Triton v1** — the naive
baseline this whole CUDA line has been chasing. Real, substantial
progress across four versions, not a claim of parity.

NCU explains both ends of the tradeoff. At batch=1, an ironic finding:
phase 2 (grid `(batch, num_kv_heads)` only — the exact same 2-block
grid-starvation problem split-K exists to fix, since the extra grid
dimension only applies to phase 1) actually costs *more* than phase 1
(233.5 us vs. 133.6 us) — at `num_splits=64`, phase 2 does `64 * 6 = 384`
sequential merge iterations per thread with only 2 blocks to hide that
behind. This is also why the sweep peaked at 64 rather than climbing
further: past that point, phase 2's linearly-growing cost overtakes
phase 1's shrinking per-split cost. At batch=64, phase 1 dominates
(6.41 ms of ~6.7 ms total) and is genuinely DRAM-bandwidth-bound (85.96%
— the highest this kernel has shown at any batch): 8192 blocks at a
`num_splits` tuned for batch=1 and never shrinking fragments what would
be efficient transfers, the same mechanism already documented for
Triton v4's high-batch regression, more pronounced here since the split
count doesn't adapt to batch (a known follow-on, not silently ignored).
Full data in [profiles/notes.md](profiles/notes.md).

## Repo layout

```
src/            Triton kernels + PyTorch reference
cuda/           CUDA C++ kernels + PyTorch extension binding
tests/          Correctness tests (reference-matching, edge cases, fuzz)
bench/          Benchmark scripts and raw results
analysis/       Roofline / sweep plots
profiles/       Nsight Compute captures + notes
```

## Hardware

RTX 3060 Laptop (6GB) on WSL2 for development; cross-hardware validation planned on A100.

## Reproduce

```bash
uv venv --python 3.12
uv pip install -r requirements.txt

# correctness (reference oracle + Triton v1/v2/v3/v4 + CUDA v1/v2/v3/v4 kernels, needs a CUDA GPU for the kernel half)
uv run pytest tests/ -q
# CUDA kernels alone (JIT-compiles cuda/*.cu on first run, cached after that)
uv run pytest tests/test_kernel_cuda_v1.py tests/test_kernel_cuda_v2.py tests/test_kernel_cuda_v3.py tests/test_kernel_cuda_v4.py -q
# full 2000-case reference fuzz / 100-case kernel fuzz (defaults are fast subsets):
PAGED_ATTN_FUZZ_ITERS=2000 uv run pytest tests/test_reference_fuzz.py -q
PAGED_ATTN_KERNEL_FUZZ_ITERS=100 uv run pytest tests/test_kernel_v1.py::test_fuzz_curated_shape_matrix -q

# latency: reference vs. Triton v1 vs. v2 vs. v3 vs. v4 vs. CUDA v1 vs. v2 vs. v3 vs. v4, batch sweep at the primary target shape
uv run python bench/bench_decode.py
# num_splits sweeps at batch=1 (set each wrapper's default)
uv run python bench/bench_v4_num_splits.py
uv run python bench/bench_cuda_v4_num_splits.py
```

## License

Apache License 2.0 — see [LICENSE](LICENSE). This project draws on public techniques from FlashDecoding, FlashInfer, and vLLM (split-K parallelism, the online-softmax recurrence); specific attributions are noted wherever a structure is adapted rather than original.
