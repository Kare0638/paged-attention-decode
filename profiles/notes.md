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
best-of-30). Superseded once below — see "wrapper dispatch overhead" —
but the crossover shape (v2 wins low batch, loses high batch) is the same
story either way, since NCU's occupancy numbers below are unaffected by
Python-side overhead:

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
faster at low batch and *slower* than v1 at batch=32/64. NCU explains why:

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

## v3 investigation: num_stages sweep, and a bigger find in the wrapper (2026-07-30)

The roadmap's v3 item is "single-pass online softmax, don't materialize
the intermediate attention matrix, tune num_stages." The first two are
already true of v1/v2's design (the loop's running `m_i`/`l_i`/`acc`
update is structurally required at real seq_len, not something added
later — see `kernel_v1_naive.py`'s docstring), so the only untried lever
was `num_stages`, which both wrappers pinned to Triton's default so far.

**`num_stages` sweep** (same v1 kernel body, launched directly, bypassing
both wrappers):

| config | batch | num_stages=1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| BLOCK_N=16 (v1) | 1 | 0.168 | 0.124 | 0.124 | 0.121 | **0.111** | 0.113 |
| BLOCK_N=16 (v1) | 64 | 0.422 | 0.423 | 0.421 | 0.420 | 0.421 | 0.421 |
| BLOCK_N=128 (v2) | 1 | 0.059 | **0.047** | 0.052 | 0.047 | OOM | OOM |
| BLOCK_N=128 (v2) | 64 | 0.456 | 0.440 | 0.438 | **0.438** | OOM | OOM |

Real but modest: ~11% at v1/batch=1 (num_stages=5 vs. Triton's default of
3), a few percent for v2, nothing at batch=64 (already occupancy-bound,
nothing left to hide). v2 hits the same shared-memory ceiling at
num_stages>=5 that BLOCK_N=256 hit earlier (`OutOfResources`, wider tiles
and deeper pipelining both spend the same limited budget). Not dramatic
enough to justify a dedicated v3 kernel file — this is a launch-config
tweak on the existing body, same as v2's BLOCK_N change, and small enough
that it isn't folded into either wrapper's defaults for now.

**The bigger find**: cross-checking a raw kernel launch (bypassing both
wrappers) against `paged_attention_decode_v2()`'s full call at batch=1
showed a gap far too large to be noise — 0.055ms raw vs. 0.148ms through
the wrapper. Isolated by adding wrapper steps back one at a time:

| step | latency |
|---|---|
| raw kernel launch | 0.057 ms |
| + `block_table`/`seq_lens` `.to(torch.int32)` | 0.057-0.058 ms (no real cost) |
| + `assert torch.all(seq_lens >= 1)` | **0.146-0.170 ms** |

The `assert` — added deliberately in v1 to catch a real silently-wrong-answer
risk (seq_len=0 divides by zero) — forces a device-to-host sync: `torch.all(...)`
returns a CUDA tensor, and Python's `assert` needs its `__bool__()`, which
blocks until the GPU finishes and a scalar comes back. That sync cost
2-3x the raw kernel time at batch=1, on every call, confirmed reproducible
across 3 independent runs. Fixed by removing the runtime check from both
`kernel_v1_naive.py` and `kernel_v2_coalesced.py` — the precondition is
still documented in both docstrings, and it's exercised by the test suite
during development instead of paid for on every production call. Every
prior latency number in this file and both READMEs used the un-fixed
wrappers; the NCU occupancy numbers above are unaffected (they measure
GPU execution, not Python dispatch), but the latency tables have been
re-measured and corrected.

**Corrected latency** (`bench/bench_decode.py`, post-fix):

| batch | reference | v1 | v2 | v2 vs v1 | v2 vs reference |
|---|---|---|---|---|---|
| 1 | 0.693 ms | 0.123 ms | 0.078 ms | **1.58x** | **8.9x** |
| 2 | 1.248 ms | 0.186 ms | 0.060 ms | **3.08x** | **20.7x** |
| 4 | 1.952 ms | 0.146 ms | 0.065 ms | **2.27x** | **30.3x** |
| 8 | 3.926 ms | 0.180 ms | 0.071 ms | **2.55x** | **55.6x** |
| 16 | 7.549 ms | 0.143 ms | 0.158 ms | 0.91x | 47.9x |
| 32 | 15.250 ms | 0.220 ms | 0.249 ms | 0.88x | 61.3x |
| 64 | 30.542 ms | 0.419 ms | 0.443 ms | 0.94x | 68.9x |

Same crossover shape as before (v2 wins low batch, loses batch>=16) — the
mechanism (shared-memory-limited occupancy) doesn't change, only the
absolute numbers do, and by a lot: v1/v2 both got substantially faster
once the sync left the hot path, and v2's peak advantage over the naive
reference loop went from 55.6x to 68.9x. The lesson generalizes past this
one assert: any `assert`/`if` on a GPU tensor's *value* (not just its
shape/dtype/device) is a synchronization point, and at sub-millisecond
kernel latencies that sync can dominate the number being measured.

## Triton v3 — num_stages formalized, and it barely beats Triton's own default (2026-07-30)

`src/kernel_v3_online_softmax.py` reuses v1's kernel body (same as v2),
block_n=128 (same as v2's default), plus an explicit `num_stages=4`
instead of leaving it at Triton's default. That default is **3**, not an
unknown auto-selected value — confirmed by reading
`triton/backends/nvidia/compiler.py`'s `CUDAOptions` dataclass directly
(`num_stages: int = 3`) in the installed Triton 3.6.0 source. Written up
as a real file — matching the repo layout's documented naming — even
though the roadmap's v3 description (single-pass online softmax, no
intermediate materialization) was already satisfied by v1/v2's design
before this file existed; the only new thing here is the num_stages pin,
i.e. this is a 3->4 change, not "default -> 4."

**Real A/B through the full wrapper** (`bench/bench_decode.py`, v3 vs v2,
both fp16, same shape):

| batch | v2 | v3 | v3 vs v2 |
|---|---|---|---|
| 1 | 0.0584 ms | 0.0584 ms | 1.00x |
| 2 | 0.1331 ms | 0.1352 ms | 0.98x |
| 4 | 0.0891 ms | 0.0891 ms | 1.00x |
| 8 | 0.0942 ms | 0.0696 ms | **1.35x** |
| 16 | 0.1546 ms | 0.1751 ms | 0.88x |
| 32 | 0.2437 ms | 0.2477 ms | 0.98x |
| 64 | 0.4390 ms | 0.4393 ms | 1.00x |

Within noise for 6 of 7 batch sizes, one real win at batch=8 that doesn't
repeat at neighboring batch sizes. This does *not* contradict the earlier
num_stages sweep (which showed a clear 1->4 improvement on the raw kernel
body) — that sweep's num_stages=1 was a forced low baseline for
comparison, not what v2 was actually running. v2 never sets num_stages;
it uses Triton's default of 3, and the isolated sweep already showed 3
landing close to 4 (0.052 vs. 0.047 ms at batch=1). Same conclusion the
num_warps sweep reached before v2 was written: Triton's own defaults for
this kernel are already close to what a manual sweep finds, and the
honest report is "checked, marginal at best" rather than inflating a
noise-level result into a claimed win.

**Where the real wins came from, in order**: v1->v2's BLOCK_N change
(up to 2.15x on the raw kernel, 1.21x-2.55x through the wrapper depending
on batch) is the only Triton-level lever in this project so far with a
consistent, mechanistically-understood effect (fewer dependent
`block_table` loads, traded against occupancy at high batch). Both
`num_warps` and `num_stages` tuning were checked and found to add little
on top of Triton's defaults. The next lever with an *a priori* large,
well-understood effect is v4's split-K — batch=1 is still capped at 2

## Triton v4 — split-K (2026-07-31)

`src/kernel_v4_split_k.py`: phase 1 (grid `(batch, num_kv_heads,
num_splits)`) computes unnormalized partial `(O, m, l)` per chunk of the
sequence; phase 2 (grid `(batch, num_kv_heads)`) reduces across splits
with the standard online-softmax merge, explicitly guarded against
`exp(-inf - -inf)` rather than relying on split 0 always being non-empty.
Full derivation in `analysis/split_k_derivation.md`. Correctness
double-checked beyond the usual reference comparison: `num_splits=1` at
`block_n=16` (matching v1's hardcoded tile exactly) reproduces v1's
output **bit-for-bit** (0.0 max diff) — confirming split-K is a pure
reassociation of v1's math, not a different algorithm that happens to
pass a loose tolerance. Split-invariance (`num_splits` ∈ {1,2,4,8,16} all
agreeing with the fp32 reference independently) holds throughout a
200-iteration fuzz sweep.

One correctness-suite side effect, found while chasing this: a specific
ragged-batch case had one output element near zero (`expected ≈
-0.0044`) fail at the shared `atol=1e-3` tolerance (`tests/
kernel_test_utils.py`). First response was to bump `atol` to `2e-3` —
but the real bug was upstream of the tolerance, in `compare_to_reference`
itself: it computed the fp32 reference from the *original* fp32 inputs,
while the kernel only ever saw fp16-rounded versions of them. That
mixes two different error sources into one measured diff — the kernel's
own compute error, and fp16 input-quantization error the kernel had no
part in — and near-zero output elements are exactly where fp16
quantization error is proportionally largest (~20% relative, confirmed
by checking v1 alone, no split-K involved, against the identical
seed/shape: same near-zero element, same-sized discrepancy). Fixed
`compare_to_reference` to round `q`/`k_cache`/`v_cache` through fp16
*before* computing the reference, so both the reference and the kernel
see identical values — `atol` went back down to `1e-3`, verified across
a 300-iteration fuzz sweep on all four kernel versions (652 cases), not
just the one case that originally failed. Not a reduction bug in v4 —
a test methodology gap that affected every kernel version equally, v4
was just the one that happened to surface it first for a given seed.

### `num_splits` sweep — and a measurement methodology fix

`num_splits` ships with no default until a sweep sets one (same
discipline as v2's `block_n=128`, v3's `num_stages=4`). First attempts
at this sweep (`bench/bench_v4_num_splits.py`, best-of-N single
readings) were not reproducible run to run — the "best" `num_splits`
value changed on every rerun of an unmodified script. Root cause: at
these sub-0.1ms latencies, run-to-run GPU clock/power-state noise on
this laptop GPU dominates the actual differences between configs.
Verified directly: 25 interleaved v1/v4 samples at batch=1 gave v4 a
stdev of 0.0254ms against a mean of 0.068ms (~37% CV) with several
anomalously fast outlier readings, vs. v1's much tighter 0.0148ms stdev
on 0.124ms mean (~12% CV) — v4's two-kernel pipeline is inherently
noisier to measure, not just unluckily sampled.

Fixed by switching both `bench_v4_num_splits.py` and `bench_decode.py`
to **median of several independent, interleaved trials** rather than a
single best-of-N: round-robin across all configs each round (not all of
config A's iterations, then all of config B's) so thermal drift over the
sweep doesn't systematically favor whichever config happens to run
first, and median is far less sensitive than min to one lucky low-noise
reading. This reproduced consistently across repeated full reruns.

**Result — a broad, flat plateau, not a sharp peak** (unlike `block_n`
or `num_stages`, which each had one clear best value):

| num_splits | speedup vs. v1 |
|---|---|
| 1 | 1.26x–1.37x |
| 2 | 1.73x–1.83x |
| 4 | 1.68x–1.88x |
| 8 | 1.61x–1.83x |
| 16 | 1.64x–1.85x |
| 32 | 1.66x–1.76x |
| 64 | 1.57x–1.71x |
| 128 | 1.22x–1.25x |

(ranges across two independent 9-trial-median sweeps). `num_splits=16`
chosen as the default: middle of the plateau, and `batch(1) *
num_kv_heads(2) * num_splits(16) = 32` lands close to the ~28-SM count
this project's occupancy story is built around — both an a priori
reasonable target and empirically inside the measured sweet spot, not
picked for only one of those reasons. `num_splits=1` underperforms (no
parallelism gain, still pays phase 2's fixed overhead); `num_splits=128`
underperforms more (chunks become much shorter than the `block_n=128`
tile, wasting bandwidth on masked lanes within each split, on top of
phase 2 reducing over more terms) — full sweep data (all 9 trials per
config, not just the median) in `bench/results/v4_num_splits_sweep.json`.

### Latency vs. v1 across the batch sweep

`bench_decode.py` updated to the same median-of-15-interleaved-trials
methodology for the same noise reason. Reproduced across two full runs:

| batch | v4 vs. v1 |
|---|---|
| 1 | **1.83x** (both runs) |
| 2 | 1.84x–2.00x |
| 4 | 1.49x–1.73x |
| 8 | 1.17x–1.35x |
| 16 | 0.94x–1.00x (crossover) |
| 32 | 0.80x–0.85x |
| 64 | 0.75x–0.82x |

Same shape as v2's tradeoff: a real win concentrated at low batch (the
scenario this whole kernel exists for), a real loss at high batch,
reported in both directions rather than only the favorable one.

### NCU, batch=1 vs. batch=64, phase 1 and phase 2 separately

A metric-interpretation lesson worth recording: `sm__warps_active.avg.
pct_of_peak_sustained_active` reads **8.33% for phase 1 at both batch=1
and batch=64** — identical to v1's batch=1 number — which looks at first
glance like occupancy didn't improve at all. NVIDIA's definition:
achieved occupancy, the ratio of active warps per active cycle to the
hardware maximum warps per SM (the "achieved_occupancy" successor metric,
per NVIDIA's own Nsight Compute docs) — it does *not* measure how many of
the ~28 SMs received any work at all, only how full the SMs that *did*
run something were while they were running it. Phase 1's number reads
identically to v1's for two different reasons that happen to coincide,
not one: v1 at batch=1 has room for up to 8 resident blocks/SM
(register/shared-mem limit) but only 2 blocks exist in the whole grid, so
whichever 1-2 SMs get work only ever run 1 block each — occupancy is low
because the *grid* is too small to fill even one SM's generous capacity.
Phase 1 (same `block_n=128` tile as v2) has a real, grid-size-independent
ceiling of exactly 1 resident block/SM (`launch__occupancy_limit_shared_mem`),
so any SM that runs it is capped at that same 1-block achieved occupancy
no matter how many total blocks are in flight elsewhere on the GPU. Both
land on the same ratio (1 block's worth of warps against the hardware
max) for unrelated reasons. The metrics that actually capture "did
split-K spread work across more of the GPU" are throughput-based, and
those move a lot:

| metric | v1, batch=1 | phase 1, batch=1 | v1, batch=64 | phase 1, batch=64 |
|---|---|---|---|---|
| grid size | 2 | **32** | 128 | **2048** |
| `dram__throughput` | 5.43% | **37.00%** | 95.14% | **78.43%** |
| `sm__throughput` | 1.27% | **7.21%** | (n/a) | 15.14% |
| `sm__warps_active` | 8.33% | 8.33% (coincidence, see above) | 35.37% | 8.32% (real per-SM ceiling) |

At batch=1, phase 1's 32 blocks reach most of the ~28 SMs at once
(instead of 2), and DRAM throughput jumps 6.8x (5.43%→37.00%) even
though the achieved-occupancy metric reads identically to v1. At
batch=64, phase 1's DRAM throughput (78.43%) is *lower* than v1's at the
same batch (95.14%) — this is the NCU-level evidence for the latency
regression: splitting fragments what would otherwise be efficient
`block_n=128`-sized transfers into more, smaller ones, and that
fragmentation cost isn't worth paying once batch alone already supplies
enough parallelism.

Phase 2 is itself still occupancy-starved at batch=1 by the same
mechanism v1 was built to fix — grid `(batch, num_kv_heads)` = 2 blocks,
`sm__warps_active` 6.79% — but cheap in absolute terms (`sm__throughput`
0.56%, `dram__throughput` 4.75%: O(num_splits) work per program, not
O(seq_len)). At batch=64, phase 2's grid grows to 128 blocks and its own
`sm__warps_active` reaches 25.40% (no shared-memory ceiling to hit at
this buffer size, unlike phase 1) — phase 2 was never the bottleneck at
either batch size; phase 1's occupancy/fragmentation tradeoff is the
whole story.

## CUDA v1 — naive baseline (2026-08-04)

First CUDA C++ kernel in this project, and the first time a kernel here is
built via `torch.utils.cpp_extension.load` (JIT) rather than Triton's own
JIT — verified that path works with a throwaway 5-line add-kernel before
writing any real logic on top of it (same "prove the tool before relying
on it" discipline as Week 0's raw-`nvcc` check).

`cuda/kernel_v1_naive.cu` is a direct port of `_paged_attn_decode_v1_kernel`'s
algorithm (same grid, `(batch, num_kv_heads)`; same online-softmax
recurrence), but deliberately **more naive** than Triton v1: no page-tile
batching (Triton's `tl.dot` computes a whole `[GQA_RATIO_PADDED, BLOCK_N]`
score tile per iteration via one matmul instruction; this kernel loops
token-by-token, one block-wide reduction per `(row, token)` pair). K/V
*is* already loaded once per token into a register and reused across all
`gqa_ratio` rows — no redundant global re-read to fix there. The real
per-row-repeated cost is the reduction itself: each row runs its own
independent 9-`__syncthreads()` tree reduction, `gqa_ratio` times per
token, instead of one reduction batched across all rows at once. That's
intentional — v2's "shared-memory tiling" roadmap item is specifically
what batches it away, and needs a naive baseline to be measured against,
the same role Triton v1 played for Triton v2's tile-size change.

**Correctness**: `tests/test_kernel_cuda_v1.py` mirrors
`tests/test_kernel_v1.py`'s shape matrix exactly (same tolerance,
`rtol=1e-2/atol=1e-3`) — all 31 cases pass, no separate tolerance handling
needed. Unlike Triton, CUDA has no `tl.arange` power-of-2 constraint and no
`tl.dot` K>=16 floor, so the wrapper's validation is a genuine subset of
Triton v1's — see `src/kernel_cuda_v1.py`'s docstring for which checks
don't carry over and why.

**Latency vs. Triton v1** (`bench_decode.py`, same median-of-15-trials
methodology, primary target shape):

| batch | Triton v1 | CUDA v1 | CUDA v1 vs. Triton v1 |
|---|---|---|---|
| 1 | 0.131 ms | 6.008 ms | 0.02x (~46x slower) |
| 16 | 0.175 ms | 6.960 ms | 0.03x (~40x slower) |
| 64 | 0.423 ms | 13.348 ms | 0.03x (~32x slower) |

Reported as what it is: a much slower, much more naive kernel, not a
regression from a working baseline — there was no CUDA baseline before
this. The gap is the direct, expected cost of processing one token at a
time with a full block-synchronizing reduction per token per row, instead
of Triton's tiled matmul.

**NCU mechanism — a different bottleneck than Triton v1's, not the same
one measured worse.** Triton v1 (`profiles/notes.md`, above) is
memory-latency-bound: `long_scoreboard` (waiting on global memory) is its
dominant stall reason. CUDA v1's dominant stalls, at batch=1
(`smsp__average_warp_latency_issue_stalled_*.ratio`, relative units):

| stall reason | ratio |
|---|---|
| `wait` (fixed-latency math pipe — the per-token `__expf`/division calls) | 3,967,707 |
| `short_scoreboard` (shared-memory round trip — the tree reduction) | 2,730,034 |
| `barrier` (explicit `__syncthreads()`) | 2,363,041 |
| `long_scoreboard` (global memory) | 1,234,248 |
| `no_instruction` | 172,318 |
| `not_selected` | 0 |

`long_scoreboard` — Triton v1's dominant reason — is CUDA v1's *smallest*
nonzero category. This kernel isn't memory-bound at all:
`dram__throughput` reads 0.92% at batch=1 (Triton v1: 5.43%, itself
already low), and `sm__throughput` is 1.64%. The bottleneck is raw
instruction/synchronization volume — at batch=1 (only 2 thread blocks
total), the kernel still issues 1,474,560 shared-memory load instructions
and 786,432 shared-memory stores (`smsp__inst_executed_op_shared_{ld,st}.sum`),
one full block-wide tree-reduction-and-barrier per token per row,
`gqa_ratio(6) * seq_len(2048)` times per block. This is exactly what v2's
shared-memory K/V tiling is expected to fix — by amortizing that
reduction/barrier cost over far fewer, larger iterations — not primarily a
memory-bandwidth win the way Triton v1→v2 was framed, a different
mechanism for a similarly-named optimization.

Occupancy: grid `(1, 2, 1)` at batch=1, `sm__warps_active` 8.33% —
identical to Triton v1's number, and the same root cause (2 blocks total
on a ~28-SM GPU; `launch__occupancy_limit_registers` says up to 12 blocks
would fit per SM, so it's a grid-size problem, not a resource ceiling). At
batch=64, grid grows to `(64, 2, 1)` = 128 blocks: `sm__warps_active`
34.79%, `sm__throughput` 63.54%, `dram__throughput` 11.00% — real
utilization gains from more grid parallelism, though this kernel never
approaches Triton v1's 95.14% `dram__throughput` at the same batch, since
it was never bandwidth-bound to begin with.

**Bank conflicts — measured, not the story this kernel's bottleneck turned
out to be.** `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` is
**0** at batch=1 (2 blocks) — the tree reduction's own addressing
(`sdata[tid]`, `sdata[tid+s]`, contiguous stride-1 by `threadIdx.x`) is
conflict-free by construction, structurally different from Week 0's
`shared_stride_kernel<PAD=0>` (`tile[32][32]`, deliberately strided to hit
the textbook 32-way conflict). At batch=64 (128 blocks), the same metric
reads 418,325 (loads) + 100,655 (stores) — nonzero, but against
94,371,840 load and 50,331,648 store shared-memory instructions issued at
that batch size (both scale exactly 64x from batch=1's counts, confirming
per-block work is unchanged), that's under 0.5% of total shared-memory
traffic. Flagging this rather than explaining it: the conflict count
appearing only once many blocks are concurrently resident, despite each
block's own access pattern being unchanged and provably conflict-free in
isolation, isn't a mechanism this investigation pinned down — worth
re-examining in v2, whose `[gqa_ratio][head_dim]` shared-memory tile is
the next structure to check with Week 0's padding trick, though the same
structural reason (threadIdx.x is always the head_dim lane; row/token
indices are always looped serially within a thread, never spread across
a warp) predicts it won't matter there either — v2 measures a padded and
unpadded variant directly rather than assuming either outcome. For v1, bank conflicts
are demonstrably not the dominant cost (the stall-reason and instruction-
count data above account for the latency without invoking them).

## CUDA v2 — batched shared-memory reduction (2026-08-04)

`cuda/kernel_v2_shared_tile.cu` batches the score reduction across all
`gqa_ratio` query rows into one `[gqa_ratio][row_stride]` shared-memory
tile per token, instead of v1's one independent block-wide tree reduction
per `(row, token)` pair. Hypothesis going in: since v1 runs a full
9-`__syncthreads()` reduction sequence `gqa_ratio` times per token, and
NCU's v1 profile showed `barrier` as a top-3 stall category, batching all
rows into every reduction stage should cut `__syncthreads()` call count
from `gqa_ratio * seq_len * 9` to `seq_len * 9` — an exact 6x reduction
at the primary shape — and reduce latency accordingly.

**Correctness**: `tests/test_kernel_cuda_v2.py` mirrors v1's full shape
matrix (32 cases, all passing) plus a direct v1-vs-v2 comparison at
`rtol=1e-4/atol=1e-5` — **max diff 0.0** (bit-exact), confirming v2 is a
pure reassociation of v1's math, not a different algorithm.

**The hypothesis was wrong.** `bench_decode.py`: `cuda_v2` vs. `cuda_v1`
is 0.98x–1.04x across every batch size (1 through 64) — noise-level, no
systematic win in either direction, despite the reduction being real and
measured:

| metric (batch=1) | CUDA v1 | CUDA v2 (padded) |
|---|---|---|
| `gpu__time_duration.sum` | 8.79 ms | 8.85 ms |
| `smsp__inst_executed.sum` (total instructions) | 11,635,480 | 9,181,968 (-21%) |
| `smsp__inst_executed_op_shared_ld.sum` | 1,474,560 | 573,440 (-61%) |
| `smsp__inst_executed_op_shared_st.sum` | 786,432 | 344,064 (-56%) |

v2 genuinely executes far fewer instructions and far less shared-memory
traffic — and it makes **no measurable difference to wall-clock time**.
The stall-reason ratios explain why:
`smsp__average_warp_latency_issue_stalled_barrier.ratio` is **higher** in
v2 (4,782,348) than v1 (2,363,041) — the opposite of the predicted
direction — while `short_scoreboard` drops (891,689 vs. 2,730,034) and
`wait` drops modestly (2,771,659 vs. 3,967,707). Reducing the *count* of
`__syncthreads()` calls doesn't reduce the *total time* spent
synchronizing when each remaining barrier now waits on 6x more serialized
work (the `for row` loop moved inside each reduction stage) before it can
release — fewer, longer barrier waits instead of more, shorter ones,
netting out to roughly the same total stall time. Both `sm__throughput`
(0.78%) and `dram__throughput` (0.85%) stay near-zero at batch=1 in v2,
same as v1 — this kernel is latency-bound by the length of its serialized
dependency chain (compute → shared write → sync → shared read →
compute...), not by instruction throughput or bandwidth, so cutting
instruction *count* without shortening that chain's *length* (still 9
sequentially-dependent stages per token, same as v1 — only the width of
each stage changed) doesn't move the needle. This is the real,
evidence-backed case for v3's warp-shuffle reduction: it's not just
"another way to reduce," it changes the dependency-chain structure
itself, which this version's data shows is the actual lever, not sync
count or instruction volume.

**Bank-conflict A/B**: `forward_v2` (padded, `row_stride = head_dim + 1`)
vs. `forward_v2_unpadded` (`row_stride = head_dim`), both templated off
the same kernel (`PAD` bool, reusing Week 0's `shared_stride_kernel<PAD>`
technique). At batch=64:

| metric | padded | unpadded |
|---|---|---|
| `gpu__time_duration.sum` | 13.74 ms | 13.56 ms |
| `l1tex__data_bank_conflicts_..._ld.sum` | 0 | 0 |
| `l1tex__data_bank_conflicts_..._st.sum` | 0 | 0 |
| `dram__throughput` | 36.68% | 37.38% |
| `sm__warps_active` | 35.04% | 35.05% |

The prediction from v1's analysis holds: **0 conflicts in both variants**,
no measurable latency difference — `threadIdx.x` is still always the
head_dim lane in this tile, and `row`/`token` are still always looped
serially within a thread, never spread across a warp's lanes on one
instruction, so the padding has nothing to fix. Unlike v1 (which measured
a small, unexplained nonzero conflict count at batch=64 specifically),
v2's `[gqa_ratio][head_dim]` tile shows exactly 0 at batch=64 in *both*
variants — that anomaly does not reproduce under this differently-shaped
tile, consistent with it being specific to v1's flat-array structure
rather than a general "many resident blocks" effect.

**Occupancy**: `launch__occupancy_limit_shared_mem` drops from v1's 21
blocks/SM to v2's 15 (the wider tile costs more shared memory, as
expected) but `launch__occupancy_limit_registers` stays at 12 in both —
registers, not shared memory, remain the binding ceiling, so this change
costs nothing in occupancy, unlike Triton v2's real 8-to-1 blocks/SM
regression.

**Bottom line, reported as measured**: the "shared-memory tiling"
roadmap item is implemented and verified correct, and bank-conflict
elimination was checked directly (via a real padded/unpadded A/B, not
assumed) and confirmed unnecessary for this access pattern. It is *not* a
latency win — a genuine null result, not a hidden regression or a
methodology artifact (checked from three angles: raw wall-clock time,
NCU-measured kernel duration, and instruction/stall-reason counts, all
agreeing). The actual bottleneck this kernel needs fixed is
dependency-chain length, which is v3's job.

## CUDA v3 — warp-shuffle reduction (2026-08-04)

`cuda/kernel_v3_warp_shuffle.cu` replaces v1/v2's tree-reduction-plus-
`__syncthreads()` algorithm with warp-shuffle (`__shfl_down_sync`/
`__shfl_sync`) — a genuinely different reduction algorithm, not another
grouping of the same one (that was v2's already-tested lever). No shared
memory, no explicit block-wide barrier at all: reduction happens in
registers via warp-synchronous shuffle instructions. Forces `blockDim(32)`
(one warp per block) instead of v1/v2's `blockDim(head_dim)`, so each
thread now owns `head_dim/32` lanes — a genuine new precondition
(`head_dim % 32 == 0`) specific to this design, satisfied by both
head_dim values (32, 128) this project's CUDA test suite exercises.

**Correctness**: `tests/test_kernel_cuda_v3.py`, 32 cases, all passing.
v1-vs-v3 comparison (different floating-point reduction order, not
expected to be bit-exact like v2 was): measured max diff **~6e-8** at the
exact test shape (both kernels accumulate in fp32 throughout, rounding to
fp16 only once at the final store, so the reordering barely matters in
practice) — `rtol=1e-4/atol=1e-5` set from that measurement, not guessed.

**This time the hypothesis holds — a real, measured win**, not another
null result. `bench_decode.py`: `cuda_v3` vs. `cuda_v1` is 1.34x-2.00x
across every batch size, growing with batch:

| batch | cuda_v1 | cuda_v3 | cuda_v3 vs. cuda_v1 |
|---|---|---|---|
| 1 | 5.877 ms | 4.287 ms | 1.37x |
| 16 | 6.865 ms | 4.746 ms | 1.45x |
| 64 | 11.558 ms | 5.791 ms | 2.00x |

NCU confirms the mechanism directly, with a result more nuanced than
either of the plan's two hypotheses predicted in isolation:

| metric | v1, batch=1 | v3, batch=1 | v1, batch=64 | v3, batch=64 |
|---|---|---|---|---|
| `gpu__time_duration.sum` | 8.79 ms | 6.54 ms | 14.82 ms | 7.68 ms |
| `smsp__inst_executed.sum` | 11,635,480 | **2,746,770** | 744,670,720 | **175,793,280** |
| `launch__occupancy_limit_registers` | 12 blocks | **48 blocks** | 12 blocks | **48 blocks** |
| `launch__occupancy_limit_blocks` (hardware max) | 16 | 16 | 16 | 16 |
| `sm__warps_active` | 8.33% | **2.08%** | 34.80% | **8.81%** |
| bank conflicts (ld/st) | 0 | 0 | (not re-measured) | 0 |

**Hypothesis 1 (shorter critical path -> lower latency at every batch)
confirmed cleanly**: instruction count drops ~4.2x at *both* batch sizes
(a consistent ratio, not batch-dependent) — no shared-memory traffic at
all, and the reduction itself collapses from 9 sequentially-dependent
`__syncthreads()`-bound stages to 6 dependent shuffle instructions. This
alone explains the latency win at batch=1, where grid size (2 blocks,
unchanged from v1/v2) rules out an occupancy-driven explanation.

**Hypothesis 2 (occupancy) was directionally right but not in the way
predicted.** `launch__occupancy_limit_registers` jumps from 12 to 48
blocks/SM — a clean 4x, matching the 4x reduction in threads/block
(128->32) at roughly the same per-thread register footprint — so v3 is
no longer register-bound; the hardware's fixed 16-blocks/SM ceiling
becomes the binding constraint instead (up from register-bound 12, a
real ~33% increase in the *ceiling*). But **achieved occupancy
(`sm__warps_active`) is *lower* for v3 than v1 at both batch sizes**, not
higher — 2.08% vs. 8.33% at batch=1, 8.81% vs. 34.80% at batch=64 —
because each resident block now contributes only 1 warp instead of 4, so
even with more blocks fitting per SM, total resident warps/SM drops.
**v3 is faster with strictly lower measured occupancy at every batch
tested** — the same "achieved occupancy isn't the whole performance
story" lesson already documented for Triton v4's phase 1, now confirmed
again from the opposite direction (lower occupancy, still faster,
instead of flat occupancy, still faster). The growing speedup with batch
(1.37x -> 2.00x) is plausibly explained by more of v3's smaller blocks
fitting concurrently as grid size grows (16-block ceiling vs. v1's
12-block ceiling) stacking on top of the batch-independent instruction-
count win — a reasonable reading of the data, not independently isolated
beyond what's shown here.

**Bank conflicts**: 0 at both batch=1 and batch=64 — consistent with
having no shared memory at all in this kernel, so there is nothing for
Week 0's padding technique to apply to.

**Bottom line**: unlike v2, this is a genuine, mechanistically-understood
win — changing the reduction *algorithm* (not just its grouping)
shortened the actual dependency chain, exactly the fix CUDA v2's
investigation called for. The occupancy story is real but not the one
originally guessed: the *ceiling* rose (register pressure relieved), but
*achieved* occupancy fell (fewer warps per block) — both true at once,
and the latency win traces mainly to the shorter critical path, not to
higher occupancy.

## CUDA v4 — split-K (2026-08-05)

Last CUDA roadmap item. `cuda/kernel_v4_split_k.cu` adds a third grid
dimension over sequence chunks, the same fix already applied once for
Triton (`analysis/split_k_derivation.md`, reused directly — no new
derivation needed, the merge math is language-agnostic). Built on **CUDA
v3's warp-shuffle reduction**, not v1's tree reduction — the best
per-block building block available now, not the first one — so this
section's primary comparison is against CUDA v3, with Triton v4 kept in
view for the honest cross-language reality check.

**Correctness**: `tests/test_kernel_cuda_v4.py`, 34 cases, all passing.
`num_splits=1` vs. CUDA v3 measured **bit-exact (max diff 0.0)** before
picking a tolerance — phase 1 at `num_splits=1` covers the whole sequence
as one chunk, byte-for-byte v3's loop, and phase 2 degenerates to a
single term. Other `num_splits` values show small (~1.22e-4 max diff)
floating-point reordering noise from the online-softmax merge, well
within the standard fp16 tolerance.

**num_splits swept fresh against CUDA v3, not reused from Triton v4.**
`bench/bench_cuda_v4_num_splits.py`, batch=1, median of 9 interleaved
trials:

| num_splits | speedup vs. CUDA v3 |
|---|---|
| 1 | 0.96x |
| 2 | 1.91x |
| 4 | 3.79x |
| 8 | 7.34x |
| 16 | 13.31x |
| 32 | 19.70x |
| **64** | **20.75x** |
| 128 | 14.30x |

A sharp peak at `num_splits=64`, not Triton v4's broad flat plateau —
dropping off by 128. `num_splits=64` set as the wrapper default.

**The magnitude here is much larger than Triton v4's ever was — because
CUDA v3's batch=1 baseline had far more idle parallelism to reclaim.**
`bench_decode.py`, `cuda_v4` vs. `cuda_v3`:

| batch | cuda_v3 | cuda_v4 | cuda_v4 vs. cuda_v3 |
|---|---|---|---|
| 1 | 4.373 ms | 0.212 ms | **20.64x** |
| 4 | 4.735 ms | 0.495 ms | 9.57x |
| 16 | 4.789 ms | 1.538 ms | 3.11x |
| 32 | 5.181 ms | 3.197 ms | 1.62x |
| 64 | 6.658 ms | 6.575 ms | 1.01x |

Same tradeoff *shape* as Triton v2/v4 and CUDA v2's predecessor
comparisons (real win at low batch, shrinking to parity at high batch)
— but a dramatically larger low-batch win than any prior version in this
project, Triton or CUDA.

**The honest reality check this section commits to, stated in the plan
before measuring**: CUDA v4 still trails Triton v4 substantially.

| batch | Triton v4 | cuda_v4 | cuda_v4 vs. Triton v4 |
|---|---|---|---|
| 1 | 0.077 ms | 0.212 ms | 0.36x (~2.8x slower) |
| 16 | 0.138 ms | 1.538 ms | 0.09x (~11x slower) |
| 64 | 0.546 ms | 6.575 ms | 0.08x (~12.5x slower) |

Not a competitive result against Triton in absolute terms — but the
*relative* gap to Triton has closed enormously across the CUDA
optimization line: CUDA v1 was ~44x slower than Triton v1 at batch=1;
CUDA v4 is ~2.8x slower than Triton v4 at the same batch, and only
~1.6x slower than **Triton v1** (0.212ms vs. 0.136ms) — the naive Triton
baseline this whole CUDA line has been chasing. Each CUDA version closed
real ground; split-K alone (v3→v4) was the single largest step, a 20x
cut at batch=1. Reported as what it is: real, substantial progress that
still falls short of a tuned Triton kernel, not a claim of parity.

**NCU: phase 1 vs. phase 2, and why the win shrinks with batch.**

| metric | phase 1, batch=1 | phase 2, batch=1 | phase 1, batch=64 | phase 2, batch=64 |
|---|---|---|---|---|
| grid | (1, 2, 64) = 128 | (1, 2, 1) = 2 | (64, 2, 64) = 8192 | (64, 2, 1) = 128 |
| `gpu__time_duration.sum` | 133.6 us | 233.5 us | 6.41 ms | 304.2 us |
| `dram__throughput` | 11.11% | 2.39% | **85.96%** | 28.19% |
| `sm__warps_active` | 8.82% | 2.08% | 32.98% | 8.36% |

(NCU's per-kernel profiling overhead inflates each isolated measurement
somewhat — these two don't sum exactly to `bench_decode.py`'s unprofiled
total — but the *relative* comparison between phase 1 and phase 2 is
what matters here and is unaffected.)

**An ironic finding**: at batch=1, phase 2 — the stage designed to be
"cheap" (`O(num_splits)`, not `O(seq_len)`, per the Triton v4 precedent)
— actually costs *more* than phase 1. Phase 2's grid is `(batch,
num_kv_heads)` only, the exact same 2-block grid-starvation problem this
entire project's split-K story exists to fix — split-K's extra grid
dimension only ever applies to phase 1. At `num_splits=64`, phase 2 does
`64 * gqa_ratio(6) = 384` sequential merge iterations per thread with
only 2 blocks total to hide that latency behind — cheap in FLOPs, not
cheap in wall-clock time at this specific `num_splits`. This is also the
mechanistic explanation for why the num_splits sweep peaked at 64 instead
of climbing further: past that point, phase 2's linearly-growing
sequential cost overtakes phase 1's shrinking per-split chunk cost.

At batch=64, phase 1 dominates (6.41ms of the ~6.7ms total) and is now
genuinely DRAM-bandwidth-bound (85.96%, the highest `dram__throughput`
this kernel has shown at any batch) — 8192 blocks at `num_splits=64`
fragments what would be efficient transfers into many more, smaller ones,
the same "over-splitting wastes bandwidth once batch alone supplies
enough parallelism" mechanism already documented for Triton v4's
high-batch regression, more pronounced here because `num_splits=64` was
tuned for batch=1 and never shrinks as batch grows (a fixed default, not
a batch-adaptive one — a known, undone follow-on, not silently ignored).

**Bottom line**: split-K delivers a real, large win at low batch (up to
20.64x vs. CUDA v3), the biggest single improvement in this project's
CUDA line, with the tradeoff shape fully expected from precedent (shrinks
to parity at high batch) and a mechanism NCU explains cleanly on both
ends — including an honest surprise (phase 2 becoming the bottleneck at
batch=1, not phase 1). CUDA v4 still trails Triton v4 by roughly an order
of magnitude in absolute terms, closing this gap further is future work
this project doesn't claim to have finished.

## Roofline — measured peak bandwidth and compute (2026-08-05)

`bench/roofline.py`: no new GPU measurement — reads the three JSON files
already on disk (`peak_bw.json`, `peak_compute.json`,
`decode_latency.json`) and computes FLOPs/bytes-moved analytically from
the shape recorded in `decode_latency.json`, rather than re-measuring
anything. This also finally gives the README's "~6 FLOP/byte" claim (in
the "Why this project" section, present since early in the project) a
checked-in, re-runnable derivation instead of leaving it as an uncited
number: `intensity = FLOPs / bytes = (num_q_heads * seq_len * head_dim *
4) / (num_kv_heads * seq_len * head_dim * 4)` — `seq_len * head_dim`
cancels, leaving intensity as a pure function of GQA ratio. Computed at
the primary shape: **6.00 FLOP/byte**, matching the existing claim.
Measured ridge point: **84.19 FLOP/byte** (26.43 TFLOPS / 313.94 GB/s,
both from Week 0's measurements) — confirms the README's "~84 FLOP/byte"
figure was correct as stated.

**One consequence worth stating plainly, not glossed over**: because
intensity is fixed by GQA ratio alone, it is identical — 6.00 FLOP/byte
— for all 9 implementations at every batch size. This isn't the classic
multi-kernel roofline plot where different kernels sit at different
horizontal positions; every point here shares one x-coordinate, and the
entire story is how far below the bandwidth roof each version's *height*
sits. The rendered plot (`profiles/roofline.png`) jitters each series
into its own narrow x-lane purely for visual separation — disclosed in
the plot's own title and axis label, not hidden — with batch size
(1→64) encoded by marker size along each line instead of by x-position.

**% of the measured roofline ceiling reached, batch=1 vs. batch=64**
(full 9x7 table in `bench/results/roofline_data.json`):

| impl | batch=1 | batch=64 |
|---|---|---|
| reference | 1.09% | 1.34% |
| Triton v1 | 4.90% | **101.09%** |
| Triton v2 | 11.44% | 92.98% |
| Triton v3 | 10.69% | 91.96% |
| Triton v4 | 8.73% | 78.33% |
| CUDA v1 | 0.11% | 3.17% |
| CUDA v2 | 0.12% | 3.28% |
| CUDA v3 | 0.15% | 6.42% |
| CUDA v4 | 3.15% | 6.50% |

Consistent with everything measured so far: Triton's batch=64 numbers
sit close to the bandwidth roof (78-101% of ceiling — this is the same
DRAM-saturated regime `dram__throughput` already showed at ~95% for
Triton v1 at batch=64), while every CUDA version stays in the single
digits even at batch=64, matching the ~order-of-magnitude gap to Triton
documented throughout this project. Split-K's batch=1 win is visible
here too: Triton v4 (8.73%) and CUDA v4 (3.15%) both clear their
non-split-K predecessors at batch=1, the direct roofline-view of the
grid-starvation fix.

**An honest anomaly, flagged rather than hidden**: Triton v1 at batch=64
computes to **101.09% of the roofline ceiling** — slightly *over* the
theoretical bandwidth-bound maximum. Not clipped or explained away here;
the most likely cause is a methodology mismatch, not a violation of
physics — `peak_bw.json`'s ceiling comes from a D2D `copy_` benchmark
that moves `2 * size_bytes` (one read, one write, at parity), while the
decode kernel is almost entirely reads (the KV cache) with a tiny output
write, so a read-dominated access pattern plausibly sustains a few
percent higher throughput than a balanced read+write copy on this GPU's
memory controller. Filed as a real, measured discrepancy worth knowing
about if this roofline ceiling is reused elsewhere, not swept under the
rug because it's a slightly awkward number.

## Benchmark against FlashInfer — on RTX 3060 Laptop, not A100 (2026-08-05)

Last roadmap item, originally scoped as an A100 comparison. Run instead
on the same RTX 3060 Laptop (Ampere sm_86, same family as A100's sm_80)
everything else in this project was measured on — a real A100 rental was
planned separately, but this doesn't need to wait on that.

**A real environment hazard, caught before it corrupted anything.**
`uv pip install flashinfer-python` initially pulled in a full CUDA 13
toolchain as a side effect (`nvidia-cutlass-dsl`, `cuda-python`, etc.),
upgrading this project's pinned `torch==2.11.0+cu128` /
`triton==3.6.0` to `torch==2.13.0+cu130` / `triton==3.7.1` — global,
silent, and exactly the kind of change that would have invalidated every
historical number in this file (Triton recompiles its own kernels; a
different Triton *compiler* version is a real, uncontrolled confound,
independent of anything this project did). Caught immediately, reverted
(`torch==2.11.0+cu128` pinned back explicitly by local-version tag, not
just by release number — a plain `torch==2.11.0` re-resolves to whatever
CUDA tag happens to be newest, which was *not* `+cu128` anymore), and the
full 302-test suite re-run to confirm the revert was real, not just
version-number-deep, before touching anything else again.

Built an isolated venv (`.venv-flashinfer`) to let FlashInfer's natural
dependency resolution happen without touching the main environment at
all, and re-verified the full 302-test suite passed there too (under
`torch==2.13.0+cu130`/`triton==3.7.1`) before trusting anything measured
in it. Then, checking whether that isolation was actually load-bearing:
the leftover `flashinfer-python`/`cuda-python`/`nvidia-cutlass-dsl`
packages from the first (reverted) install attempt were still sitting in
the *main* venv, now paired with the reverted, pinned
`torch==2.11.0+cu128`/`triton==3.6.0` — and `tests/test_flashinfer_adapter.py`
passed 8/8 against that combination. **The forced upgrade was `uv`'s
dependency resolver picking the newest mutually-compatible set from a
blank slate, not an actual hard runtime requirement** — FlashInfer
runs correctly against this project's already-pinned environment. The
isolated venv (5GB, since deleted) turned out unnecessary; every number
below comes from a single process, in this project's one pinned
environment, alongside `v4` and `cuda_v4` — a true same-run comparison,
not two runs stitched together on paper.

**Adapter correctness** (`src/flashinfer_adapter.py`,
`tests/test_flashinfer_adapter.py`, 8 cases): FlashInfer's paged KV cache
tensors already use the exact layout this project's own `k_cache`/
`v_cache` use throughout (`[num_pages, page_size, num_kv_heads,
head_dim]`, NHD) — no data-layout conversion needed. The one real piece
of adapter logic is converting this project's dense `block_table
[batch, max_pages_per_seq]` + `seq_lens [batch]` into FlashInfer's
CSR-style `(indptr, indices, last_page_len)`, checked against the
reference oracle at the primary shape, both page-boundary parities
(`seq_len` evenly divisible by `page_size` and not — `last_page_len`'s
two branches), GQA ratios 1/4/6/8, and a ragged batch with independent
per-item `seq_lens`. All 8 pass at this project's standard fp16
tolerance.

**Methodology**: `use_tensor_cores=True` — FlashInfer's own documented
recommendation for GQA decode (this project's GQA ratio 6); the
CUDA-core path is documented to leave real performance on the table for
GQA, so the default would have been comparing against a weaker
configuration than FlashInfer is capable of, the same "give the
comparison a fair shot" discipline already applied when CUDA v4 got its
own fresh `num_splits` sweep instead of reusing Triton v4's. FlashInfer's
`plan()` (building reusable auxiliary structures from `indptr`/`indices`/
`last_page_len`) is called once per batch size, outside the timed trial
loop — only `run()` is timed, matching both FlashInfer's own documented
plan-once/run-many usage and this project's standing "assert-forced-a-sync"
lesson about not measuring one-time setup as a per-call cost.

**Results** (`bench/bench_flashinfer.py`, median of 15 interleaved
trials, full table in `bench/results/flashinfer_comparison.json`):

| batch | v4 (Triton) | cuda_v4 | FlashInfer | FlashInfer vs. v4 | FlashInfer vs. cuda_v4 |
|---|---|---|---|---|---|
| 1 | 0.0799 ms | 0.2386 ms | 0.0387 ms | **2.07x** | **6.17x** |
| 4 | 0.0942 ms | 0.4925 ms | 0.0502 ms | 1.88x | 9.82x |
| 16 | 0.1782 ms | 1.5084 ms | 0.1188 ms | 1.50x | 12.70x |
| 64 | 0.5151 ms | 5.7743 ms | 0.4434 ms | 1.16x | 13.02x |

FlashInfer wins at every batch size, as expected from a heavily-engineered
production kernel library against two from-scratch learning
implementations — not a surprising result, and not the point of this
comparison. What's worth noting: the gap to Triton v4 (1.16x-2.07x) is
far smaller than the gap to CUDA v4 (6.17x-13.02x), consistent with
everything else measured in this project — Triton's tile-based, compiler-
optimized codegen was always closer to competitive than the from-scratch
CUDA line, all the way back to CUDA v1's ~44x gap to Triton v1. CUDA v4's
own cuda_v4 numbers here (e.g. 0.2386ms at batch=1) sit within this
project's already-documented run-to-run noise band for these
sub-millisecond kernels (~12-37% CV, see Triton v4's num_splits sweep
entry above) compared to `bench_decode.py`'s separately-measured 0.212ms
— not a new discrepancy, the same measurement noise already characterized
elsewhere in this file.

**Bottom line**: this closes the last roadmap item. Not run on A100 (a
real A100 rental remains a possible separate future addition, not
promised here) — run instead on this project's own GPU, in this
project's own pinned environment, after catching and fixing a real
environment hazard along the way rather than letting it silently
corrupt the comparison.

**Follow-up, caught during review**: the ad-hoc `uv pip install
flashinfer-python` reproduce step above worked in this session because it
ran against an already-populated venv, but doesn't guarantee the same
outcome for someone reproducing from scratch — a second, unpinned
resolve could in principle land on a different `flashinfer-python`
release than the one these results were measured against. Replaced with
`requirements-flashinfer-comparison.txt` (`-r requirements.txt` +
`flashinfer-python==0.6.16.post1` pinned exactly), so the comparison
environment resolves deterministically in one pass instead of a
second, separate install.

Verifying that file from a genuinely clean venv (not this project's
already-built-up `.venv`) surfaced an unrelated, **pre-existing** bug:
plain `uv pip install -r requirements.txt` — this project's very first
Reproduce step, unrelated to FlashInfer — fails on a clean venv with
"no version of numpy==2.5.1" once `requirements.txt`'s
`--extra-index-url` (needed to pin the exact `+cu128` torch build) is
present, because `uv`'s default index-strategy only considers a
package's *first* index that lists it at all, and PyTorch's index
carries a `numpy` build that isn't `2.5.1`. This had gone undetected
because the working `.venv` here was built up incrementally over many
sessions, never actually re-resolved from scratch — exactly the kind
of gap "it works on my machine" reproduce instructions can hide. Fixed
via `[tool.uv] index-strategy = "unsafe-best-match"` in `pyproject.toml`
(verified: both `requirements.txt` alone and
`requirements-flashinfer-comparison.txt` now resolve cleanly from a
fresh venv, landing on the exact same pins either way) — a project-level
config, not a flag every reproduce command has to remember to pass.

## A100 cross-hardware validation (2026-08-07)

Optional follow-on, not part of the original roadmap: re-ran this
project's own test suite and benchmarks on a rented A100 80GB PCIe
(RunPod), in this project's own pinned environment
(`torch==2.11.0+cu128`, `triton==3.6.0`, nvcc 12.8), to check whether
the findings measured on the RTX 3060 Laptop generalize to a real
datacenter GPU or were specific to this project's own small, low-SM
development card.

**Connectivity note, not a project finding but worth recording for next
time**: RunPod's `ssh.runpod.io` proxy only supports interactive PTY
sessions — non-PTY exec (`ssh host "command"`) is rejected outright
("Your SSH client doesn't support PTY"), and even with a PTY forced,
transferring a large base64-encoded payload through it corrupted the
data (silent line-wrapping inside the interactive channel broke
`base64 -d`). Switched to the pod's direct TCP port (`RUNPOD_PUBLIC_IP`
/ `RUNPOD_TCP_PORT_22`, both exposed in the pod's own environment),
which behaves like a normal `sshd` — plain `scp` worked, and every
transferred file's md5sum was verified to match the pod's own before
being trusted.

**Two real bugs surfaced by testing on a genuinely fresh machine,
neither caught by any amount of local testing on an already-built-up
`.venv`:**

1. `tests/test_flashinfer_adapter.py`'s `pytest.importorskip(...)` call
   aborted the *entire* 302-test collection with a hard error instead of
   skipping cleanly, because pytest 9.1 quietly changed
   `importorskip`'s default `exc_type` from `ImportError` to the
   narrower `ModuleNotFoundError`, and this project's adapter
   deliberately re-raises a more helpful plain `ImportError`. Never
   caught locally because this session's own venv always had
   `flashinfer-python` installed (a leftover from an earlier mistake),
   so the "flashinfer absent" skip path had literally never been
   exercised until a genuinely clean environment hit it. Fixed with the
   documented `exc_type=ImportError` override; verified both directions
   (flashinfer present → 310 passed; flashinfer hidden → 302 passed, 1
   skipped).
2. `bench/roofline.py`'s plot title had "RTX 3060 Laptop" hardcoded as a
   literal string — harmless on the machine it was written on, silently
   wrong the moment it ran anywhere else. Fixed to read `gpu_name` from
   `peak_bw.json` (already recorded there), so the title is correct on
   whatever machine actually produced the data.

**Peak bandwidth/compute** (`bench/measure_peak_bw.py`,
`measure_peak_compute.py`; full records in
`bench/results/peak_bw_a100.json` / `peak_compute_a100.json`):

| | RTX 3060 Laptop | A100 80GB PCIe | ratio |
|---|---|---|---|
| peak bandwidth (measured) | 313.94 GB/s | 1699.39 GB/s | 5.41x |
| peak compute (measured, fp16) | 26.43 TFLOPS | 245.85 TFLOPS | 9.30x |
| ridge point | 84.19 FLOP/byte | 144.67 FLOP/byte | — |

A100's ridge point sits substantially higher — this project's decode
workload (~6 FLOP/byte, fixed by GQA ratio regardless of GPU) is
proportionally *further* left of the ridge point on A100 than on the
laptop, i.e. even more memory-bandwidth-bound in relative terms, despite
the A100 having far more raw bandwidth in absolute terms.

**Correctness**: full 302-test suite — 302 passed, 1 skipped
(`test_flashinfer_adapter.py`, `flashinfer` not installed on this pod)
in 485.72s, first run on this machine (includes JIT-compiling all four
CUDA extensions for sm_80 plus Triton's own kernel compilation — nothing
was cached going in). Confirms every kernel, both languages, compiles
and produces correct output on a different GPU architecture (sm_80 vs.
the laptop's sm_86) without any code changes.

**Latency** (`bench/bench_decode.py`, same methodology, same shape;
full sweep in `bench/results/decode_latency_a100.json`), Triton v4 and
CUDA v4 (this project's best per language) at batch 1/16/64:

| batch | Triton v3060 Laptop | A100 | | CUDA v3060 Laptop | A100 |
|---|---|---|---|---|---|
| 1 | 0.0765 ms | 0.1185 ms | | 0.2119 ms | 0.2952 ms |
| 16 | 0.1382 ms | 0.1198 ms | | 1.5380 ms | 0.5360 ms |
| 64 | 0.5458 ms | 0.1764 ms | | 6.5751 ms | 2.2368 ms |

(Column header shorthand: "Triton"/"CUDA" = `v4`/`cuda_v4`.) In
absolute terms A100 is faster everywhere except Triton v4 at batch=1 —
expected: at batch=1 only 2 KV heads worth of grid parallelism exist
without split-K, and A100's much larger SM count (~108 vs. the laptop's
~28) has proportionally more idle capacity relative to its own peak, so
a small-grid, launch-overhead-dominated kernel doesn't automatically
benefit from a bigger card the way a bandwidth-saturating one does.

**Two cross-hardware findings, the actual point of this exercise:**

1. **`cuda_v4` vs. `cuda_v3` (split-K's win) reaches the *same* headline
   number at batch=1 on both GPUs — 20.64x — but persists much further
   into the batch sweep on A100** (3.04x at batch=64 vs. the laptop's
   1.01x/parity):

   | batch | cuda_v4 vs. cuda_v3, laptop | cuda_v4 vs. cuda_v3, A100 |
   |---|---|---|
   | 1 | 20.64x | 20.64x |
   | 16 | 3.11x | 12.54x |
   | 64 | 1.01x | 3.04x |

   A plausible mechanism, not independently isolated beyond what's shown
   here: `num_splits=64` (this project's default, tuned on the laptop's
   28-SM occupancy profile) fragments transfers enough to erase the win
   by batch=64 on a 28-SM card, but the same fixed split count has much
   more room to still help on a ~108-SM card before over-fragmentation
   catches up — consistent with, not proof of, the occupancy-fragmentation
   story already documented for CUDA v4's batch=64 regression on the
   laptop. **Caveat stated plainly**: `num_splits=64` was never re-swept
   for A100's own occupancy profile; a fresh sweep (mirroring
   `bench_cuda_v4_num_splits.py`'s methodology) could plausibly push the
   A100 numbers higher still. Not done here — an honest scope limit, not
   a claim that 64 is already optimal on this hardware.

2. **`cuda_v4` vs. `Triton v4` (the honest CUDA-vs-Triton reality check)
   has nearly the same *relative* shape on both GPUs**, despite an order
   of magnitude difference in raw hardware capability:

   | batch | cuda_v4 vs. Triton v4, laptop | cuda_v4 vs. Triton v4, A100 |
   |---|---|---|
   | 1 | 0.36x (~2.8x slower) | 0.40x (~2.5x slower) |
   | 16 | 0.09x (~11x slower) | 0.22x (~4.5x slower) |
   | 64 | 0.08x (~12.5x slower) | 0.08x (~12.5x slower) |

   The batch=1 and batch=64 columns are close enough to call the same
   story on both cards; batch=16 diverges more (11x vs. 4.5x slower),
   plausibly downstream of finding 1 above (CUDA v4's relative
   competitiveness at mid-batch depends on how well `num_splits=64`
   suits the specific GPU's SM count, not just batch size alone). Taken
   together: the CUDA-vs-Triton gap documented throughout this project
   looks like it reflects real implementation-level differences (Triton's
   compiler-optimized codegen vs. this project's own hand-written CUDA),
   not an artifact specific to one small consumer GPU.

**Roofline**: `profiles/roofline_a100.png` (regenerated with the
`gpu_name` fix above), same single-workload-intensity story as the
laptop's — full computed table in `bench/results/roofline_data_a100.json`.

**Scope, stated plainly**: this validates that every kernel compiles
and runs correctly on different hardware and that this project's
qualitative findings (split-K's win shape, the CUDA-vs-Triton gap)
aren't laptop-specific artifacts. It does **not** include a
re-tuned `num_splits` sweep, NCU profiling, or a FlashInfer comparison
on A100 — each would be a reasonable next step, not done here by
explicit choice, not an oversight.

**NCU on this A100 pod: tried, confirmed structurally blocked, not a
configuration problem.** `ncu` is installed (`/usr/local/cuda-12.8/bin/ncu`,
just not on `PATH`), but profiling any kernel — verified with a
throwaway `torch.matmul`, not even one of this project's own kernels —
fails immediately:

```
==PROF== Connected to process 17664 (/usr/bin/python3.12)
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to
access NVIDIA GPU Performance Counters on the target device 0.
==PROF== Disconnected from process 17664
```

This is the standard NVIDIA driver restriction
(`NVreg_RestrictProfilingToAdminUsers=1`, the default since driver
~418.43) that normally needs either a system administrator's fix on the
host or the container launched with the `CAP_SYS_ADMIN` capability.
Confirmed via independent reports (not just this one pod) that RunPod
specifically does not support either path: their containers aren't
privileged, exposing `--cap-add=SYS_ADMIN` to tenant containers would be
a real security risk on shared GPU hardware, and `runpodctl`/the pod
creation UI has no option to request privileged mode. This is a
platform-level limitation, not something fixable from inside the pod —
confirmed structurally blocked, not abandoned after a shallow attempt.
Would need a different provider offering privileged containers or
bare-metal instances to get NCU data on A100; not pursued further here
by explicit choice.

## FlashInfer comparison on A100 (2026-08-07)

Re-ran the RTX 3060 Laptop's FlashInfer comparison on the same A100 pod,
same pinned environment, using the already-pinned
`requirements-flashinfer-comparison.txt` (`flashinfer-python==0.6.16.post1`)
— confirmed `torch==2.11.0+cu128`/`triton==3.6.0` stayed exactly pinned
after installing it (no repeat of the earlier dependency-drift incident),
and `tests/test_flashinfer_adapter.py` passed 8/8 before trusting any
latency number. Full sweep in
`bench/results/flashinfer_comparison_a100.json`.

| batch | v4 (Triton) | cuda_v4 | FlashInfer | FlashInfer vs. v4 | FlashInfer vs. cuda_v4 |
|---|---|---|---|---|---|
| 1 | 0.1240 ms | 0.2944 ms | 0.0562 ms | 2.21x | 5.24x |
| 16 | 0.1215 ms | 0.5297 ms | 0.0657 ms | 1.85x | 8.06x |
| 64 | 0.1785 ms | 2.1841 ms | 0.1268 ms | 1.41x | 17.23x |

Compared against the laptop's own numbers (`v4`/`FlashInfer vs. v4`,
`FlashInfer vs. cuda_v4`): 2.07x/6.17x at batch=1, 1.50x/12.70x at
batch=16, 1.16x/13.02x at batch=64.

**FlashInfer's relative edge over this project's own `v4` is *larger* on
A100 than on the laptop at every batch size** (2.21x vs. 2.07x at
batch=1, growing to 1.85x vs. 1.50x at batch=16, 1.41x vs. 1.16x at
batch=64) — the opposite direction from what "everything just runs
faster on a bigger GPU" would predict for a fixed, unretuned `v4`.
Plausible reading, not independently isolated: FlashInfer is a
production library with kernel paths specifically tuned for Ampere
datacenter cards (tensor-core paths for GQA decode, per its own
documentation), while `v4` still uses `num_splits=16` — a value swept on
the laptop's 28-SM profile, never retuned for A100, consistent with the
same caveat already flagged for `cuda_v4`'s `num_splits=64` above.

**The gap to `cuda_v4` doesn't plateau on A100 the way it did on the
laptop — it keeps widening** (5.24x → 8.06x → 17.23x, vs. the laptop's
6.17x → 12.70x → 13.02x, which grew then flattened). This tracks
directly with `cuda_v4`'s own latency at batch=64 (2.1841ms on A100,
worse in absolute terms than the laptop's `num_splits=64` default can
apparently handle at A100 scale) while FlashInfer stays fast
(0.1268ms) — the same over-fragmentation mechanism already documented
for `cuda_v4`'s high-batch regression, compounding against a baseline
(FlashInfer) that isn't standing still.

**Bottom line**: FlashInfer wins at every batch size on A100 too, as
expected — this was never going to flip. The value of running it here
wasn't "does FlashInfer win" (already known) but whether the *shape* of
the gap reproduces across hardware, and it mostly does with one honest
wrinkle: this project's own kernels, tuned once on a laptop and never
retuned, lose a bit more ground to a production library's A100-specific
optimizations than the laptop-vs-laptop comparison alone would suggest
— itself a real, useful finding about the cost of not retuning
`num_splits` per GPU, not a discrepancy to explain away.

## num_splits re-swept on A100 (2026-08-07)

Reran `bench/bench_v4_num_splits.py` and `bench/bench_cuda_v4_num_splits.py`
unmodified on the A100 pod — same methodology (median of 9 interleaved
trials, `NUM_SPLITS_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]`, batch=1),
checking whether the laptop-tuned defaults (`num_splits=16` for Triton
v4, `num_splits=64` for CUDA v4) are actually still good choices on a
~108-SM GPU, rather than assuming they transfer. Full sweeps in
`bench/results/v4_num_splits_sweep_a100.json` /
`cuda_v4_num_splits_sweep_a100.json`.

**The two kernels tell almost opposite stories.**

**CUDA v4: `num_splits=64` is still the best value on A100 — no
retuning actually needed.**

| num_splits | speedup vs. cuda_v3, laptop | speedup vs. cuda_v3, A100 |
|---|---|---|
| 1 | 0.96x | 0.96x |
| 8 | 7.34x | 7.12x |
| 16 | 13.31x | 12.77x |
| 32 | 19.70x | 19.23x |
| **64** | **20.75x** | **20.63x** |
| 128 | 14.30x | 15.39x |

Both curves peak at exactly the same value with nearly identical
magnitude — this project's existing default was already correct for
A100, not a lucky guess validated after the fact by a separate,
independent sweep. Directly explains why the earlier A100 `bench_decode.py`
run found `cuda_v4` vs. `cuda_v3` at batch=1 landing on the *same*
20.64x on both GPUs (documented above): the config actually is
optimal on both, not coincidentally close.

**Triton v4: the whole `num_splits` axis nearly flattens out on A100 —
a real, qualitatively different regime, not just a shifted optimum.**

| num_splits | speedup vs. v1, laptop | speedup vs. v1, A100 |
|---|---|---|
| 1 | 1.26x-1.37x | 1.53x |
| 2 | 1.73x-1.83x | 1.54x |
| 8 | 1.61x-1.83x | 1.55x |
| 16 (default) | 1.64x-1.85x | 1.54x |
| 64 | 1.57x-1.71x | 1.54x |
| 128 | 1.22x-1.25x | 1.51x |

On the laptop this was a real, broad plateau — a genuine ~40-50%
spread between the worst (`num_splits=1` or `128`) and best
(`num_splits=2-16`) configs. On A100, every value from 1 to 64 lands
within **1.51x-1.55x of v1** — a ~3% spread, indistinguishable from
noise given this project's own documented ~12-37% CV at these
sub-millisecond latencies. `num_splits=8` is nominally "best" (1.55x)
but not meaningfully different from the existing default of 16
(1.54x) or even `num_splits=1` (1.53x, no split-K at all).

**Why**: split-K's whole premise is fixing an *occupancy* problem — too
few blocks (`batch * num_kv_heads` = 2 at batch=1) to fill the GPU's
SMs. On the laptop's ~28 SMs, going from 2 blocks to `num_splits`x more
genuinely changes how much of the GPU gets used. On A100's ~108 SMs,
even `num_splits=64`'s ~128 blocks (`batch(1) * num_kv_heads(2) *
num_splits(64)`) is still a small fraction of the chip, and — more to
the point — the kernel is so fast in absolute terms on A100 (Triton v4
at batch=1: 0.1185-0.124ms measured across separate runs) that launch
overhead and other fixed costs plausibly dominate over whatever
occupancy split-K still buys. Not independently isolated via NCU here
(A100 NCU access is structurally blocked, see above) — a plausible
mechanistic reading consistent with the measured flatness, not proven
down to the metric level.

**Bottom line, and no code change made**: both existing defaults
(`num_splits=16` for Triton v4, `64` for CUDA v4) remain reasonable on
A100 — CUDA v4's happens to be exactly optimal, Triton v4's sits
squarely inside a flat band where nothing meaningfully beats it. Not
changing the shipped defaults based on a single GPU's sweep; the
laptop remains this project's primary development target, and this
sweep's value was checking generalization, not chasing a new number.
