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
token-by-token, one block-wide reduction per `(row, token)` pair) and no
K/V sharing across the `gqa_ratio` rows (each row redundantly re-reads the
same K/V values from global memory). That gap is intentional — v2's
"shared-memory tiling" roadmap item is specifically what removes it, and
needs a naive baseline to be measured against, the same role Triton v1
played for Triton v2's tile-size change.

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
re-examining if it grows in v2, where a real 2D K/V tile (indexed in a way
that could plausibly hit a stride-32 pattern) is exactly the shared-memory
structure Week 0's padding trick was rehearsed for. For v1, bank conflicts
are demonstrably not the dominant cost (the stall-reason and instruction-
count data above account for the latency without invoking them).
