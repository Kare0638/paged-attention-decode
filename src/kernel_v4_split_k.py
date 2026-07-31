from __future__ import annotations

import torch
import triton
import triton.language as tl

"""v4: split-K (FlashDecoding-style) parallelism along the sequence dimension.

v1/v2/v3 all grid over (batch, num_kv_heads). At batch=1 with
num_kv_heads=2 (the realistic single-request decode scenario under GQA)
that's 2 thread blocks on a ~28-SM GPU — confirmed via NCU (8.33%
occupancy, long_scoreboard-dominant stall, profiles/notes.md's v1
section). No amount of v1->v2->v3 tuning could fix this: it's a
grid-size problem, not a per-block efficiency problem. Split-K adds a
third grid dimension over chunks of the sequence, trading a second
kernel launch and intermediate memory traffic for more parallelism at
low batch. Full derivation: analysis/split_k_derivation.md.

Two Triton/NVIDIA-backend facts already load-bearing in this codebase,
reused here: tl.arange(0, N) requires N to be a compile-time power of 2
(why GQA_RATIO, e.g. 6, needs the pad-to-8-and-mask trick below);
tl.dot's K (reduction) dimension has a hard floor of 16 (M/N get padded
internally, K does not — verified empirically in kernel_v1_naive.py).
One new fact, confirmed by reading Triton 3.6.0's compiler source
directly (triton/compiler/code_generator.py's visit_For) rather than
assumed: tl.static_range(N) is a plain compile-time Python loop unroll,
never touches tl.arange, and works for any positive N — no power-of-2
constraint, unlike tl.arange. Phase 2's reduction over NUM_SPLITS uses
tl.static_range for exactly this reason: NUM_SPLITS is usually not a
power of 2 either, and unlike GQA_RATIO it doesn't need the pad+mask
treatment at all.
"""


@triton.jit
def _paged_attn_decode_v4_phase1_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, block_table_ptr, seq_lens_ptr,
    o_partial_ptr, m_partial_ptr, l_partial_ptr,
    scale,
    stride_qb, stride_qh, stride_qd,
    stride_kp, stride_ks, stride_kh, stride_kd,
    stride_vp, stride_vs, stride_vh, stride_vd,
    stride_btb, stride_btp,
    stride_opb, stride_opkv, stride_ops, stride_opg, stride_opd,
    stride_mpb, stride_mpkv, stride_mps, stride_mpg,
    stride_lpb, stride_lpkv, stride_lps, stride_lpg,
    GQA_RATIO: tl.constexpr,
    GQA_RATIO_PADDED: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)
    split_idx = tl.program_id(2)

    seq_len = tl.load(seq_lens_ptr + b)
    chunk_size = tl.cdiv(seq_len, NUM_SPLITS)
    split_start = split_idx * chunk_size
    split_end = tl.minimum(split_start + chunk_size, seq_len)

    gqa_local = tl.arange(0, GQA_RATIO_PADDED)
    gqa_mask = gqa_local < GQA_RATIO
    q_head_offsets = kvh * GQA_RATIO + gqa_local
    d_offsets = tl.arange(0, HEAD_DIM)

    q_ptrs = (
        q_ptr
        + b * stride_qb
        + q_head_offsets[:, None] * stride_qh
        + d_offsets[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=gqa_mask[:, None], other=0.0)  # [GQA_RATIO_PADDED, HEAD_DIM]

    m_i = tl.full([GQA_RATIO_PADDED], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([GQA_RATIO_PADDED], dtype=tl.float32)
    acc = tl.zeros([GQA_RATIO_PADDED, HEAD_DIM], dtype=tl.float32)

    # Empty chunk (split_start >= seq_len, e.g. a short sequence with more
    # splits requested than it has tokens): this loop does zero iterations,
    # leaving m_i=-inf, l_i=0, acc=0 — exactly the "empty split" contract
    # phase 2's merge is built to handle, no special-case branch needed.
    for start_n in range(split_start, split_end, BLOCK_N):
        n_offsets = start_n + tl.arange(0, BLOCK_N)
        # Mask against split_end, NOT seq_len — a tile that overshoots this
        # split's own chunk boundary must not silently pull in tokens that
        # belong to the next split's chunk.
        seq_mask = n_offsets < split_end

        page_ids = n_offsets // PAGE_SIZE
        slot_ids = n_offsets % PAGE_SIZE

        phys_pages = tl.load(
            block_table_ptr + b * stride_btb + page_ids * stride_btp,
            mask=seq_mask,
            other=0,
        )

        k_ptrs = (
            k_cache_ptr
            + phys_pages[:, None] * stride_kp
            + slot_ids[:, None] * stride_ks
            + kvh * stride_kh
            + d_offsets[None, :] * stride_kd
        )
        k = tl.load(k_ptrs, mask=seq_mask[:, None], other=0.0)  # [BLOCK_N, HEAD_DIM]

        v_ptrs = (
            v_cache_ptr
            + phys_pages[:, None] * stride_vp
            + slot_ids[:, None] * stride_vs
            + kvh * stride_vh
            + d_offsets[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=seq_mask[:, None], other=0.0)  # [BLOCK_N, HEAD_DIM]

        scores = tl.dot(q, tl.trans(k)) * scale  # [GQA_RATIO_PADDED, BLOCK_N], fp32 accumulate
        scores = tl.where(seq_mask[None, :], scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    # Store UNNORMALIZED partials — no acc/l_i division here, that's phase 2's job.
    o_partial_ptrs = (
        o_partial_ptr
        + b * stride_opb
        + kvh * stride_opkv
        + split_idx * stride_ops
        + gqa_local[:, None] * stride_opg
        + d_offsets[None, :] * stride_opd
    )
    tl.store(o_partial_ptrs, acc, mask=gqa_mask[:, None])

    m_partial_ptrs = m_partial_ptr + b * stride_mpb + kvh * stride_mpkv + split_idx * stride_mps + gqa_local * stride_mpg
    tl.store(m_partial_ptrs, m_i, mask=gqa_mask)

    l_partial_ptrs = l_partial_ptr + b * stride_lpb + kvh * stride_lpkv + split_idx * stride_lps + gqa_local * stride_lpg
    tl.store(l_partial_ptrs, l_i, mask=gqa_mask)


@triton.jit
def _paged_attn_decode_v4_phase2_kernel(
    o_partial_ptr, m_partial_ptr, l_partial_ptr, o_ptr,
    stride_opb, stride_opkv, stride_ops, stride_opg, stride_opd,
    stride_mpb, stride_mpkv, stride_mps, stride_mpg,
    stride_lpb, stride_lpkv, stride_lps, stride_lpg,
    stride_ob, stride_oh, stride_od,
    GQA_RATIO: tl.constexpr,
    GQA_RATIO_PADDED: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)

    gqa_local = tl.arange(0, GQA_RATIO_PADDED)
    gqa_mask = gqa_local < GQA_RATIO
    d_offsets = tl.arange(0, HEAD_DIM)

    m_run = tl.full([GQA_RATIO_PADDED], float("-inf"), dtype=tl.float32)
    l_run = tl.zeros([GQA_RATIO_PADDED], dtype=tl.float32)
    acc_run = tl.zeros([GQA_RATIO_PADDED, HEAD_DIM], dtype=tl.float32)

    # Compile-time unroll (any NUM_SPLITS, no power-of-2 constraint) —
    # see module docstring for why this is safe, confirmed from Triton's
    # own compiler source rather than assumed.
    for s in tl.static_range(NUM_SPLITS):
        m_s_ptrs = m_partial_ptr + b * stride_mpb + kvh * stride_mpkv + s * stride_mps + gqa_local * stride_mpg
        m_s = tl.load(m_s_ptrs, mask=gqa_mask, other=float("-inf"))

        l_s_ptrs = l_partial_ptr + b * stride_lpb + kvh * stride_lpkv + s * stride_lps + gqa_local * stride_lpg
        l_s = tl.load(l_s_ptrs, mask=gqa_mask, other=0.0)

        o_s_ptrs = (
            o_partial_ptr
            + b * stride_opb
            + kvh * stride_opkv
            + s * stride_ops
            + gqa_local[:, None] * stride_opg
            + d_offsets[None, :] * stride_opd
        )
        o_s = tl.load(o_s_ptrs, mask=gqa_mask[:, None], other=0.0)

        m_new = tl.maximum(m_run, m_s)
        # Guarded explicitly rather than relying on split 0 always being
        # non-empty (true given seq_len >= 1, but correctness shouldn't
        # silently depend on iteration order) — an empty split (m == -inf)
        # contributes exactly 0 regardless of where it falls in the fold.
        alpha = tl.where(m_run == float("-inf"), 0.0, tl.exp(m_run - m_new))
        beta = tl.where(m_s == float("-inf"), 0.0, tl.exp(m_s - m_new))

        l_run = l_run * alpha + l_s * beta
        acc_run = acc_run * alpha[:, None] + o_s * beta[:, None]
        m_run = m_new

    out = acc_run / l_run[:, None]

    o_ptrs = (
        o_ptr
        + b * stride_ob
        + (kvh * GQA_RATIO + gqa_local)[:, None] * stride_oh
        + d_offsets[None, :] * stride_od
    )
    tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty), mask=gqa_mask[:, None])


def paged_attention_decode_v4(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_splits: int = 16,
    scale: float | None = None,
    block_n: int = 128,
) -> torch.Tensor:
    """Same contract and non-exhaustive-validation rationale as
    paged_attention_decode_v1 (see its docstring).

    `num_splits=16` comes from bench/bench_v4_num_splits.py's sweep at
    batch=1 (the scenario this kernel exists for), not a guess — matching
    this project's own established discipline (v2's block_n=128, v3's
    num_stages=4 both came from sweeps too). Unlike those two, the sweep
    here found a broad, flat plateau (num_splits 2 through 64 all within
    ~10% of each other, reproduced across independent runs after
    switching the benchmark to median-of-interleaved-trials — raw
    best-of-N was dominated by GPU clock/power-state noise at these
    sub-0.1ms latencies, not by real differences between configs) rather
    than one sharp peak. 16 sits in the middle of that plateau and lands
    close to `batch(1) * num_kv_heads(2) * num_splits(16) = 32`, near the
    ~28-SM count this project's occupancy story is built around — both a
    reasonable a priori target and empirically inside the measured
    sweet spot, not chosen for only one of those reasons. num_splits=1
    (no real parallelism gain, still pays phase 2's overhead) and 128
    (over-splitting: wasted bandwidth on masked lanes within each split's
    undersized chunk, plus phase 2 reducing over more terms) are both
    measurably worse — see bench/results/v4_num_splits_sweep.json.
    """
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "kernel_v4_split_k requires CUDA tensors"
    assert q.dtype == torch.float16 and k_cache.dtype == torch.float16 and v_cache.dtype == torch.float16, (
        "kernel_v4_split_k expects fp16 q/k_cache/v_cache"
    )
    assert num_splits >= 1, "num_splits must be >= 1"

    batch, num_q_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads (GQA)"
    assert head_dim & (head_dim - 1) == 0, "head_dim must be a power of 2 (tl.arange requirement)"
    assert page_size & (page_size - 1) == 0, "page_size must be a power of 2 (page/slot index arithmetic)"
    assert block_n & (block_n - 1) == 0 and block_n >= 16, (
        "block_n must be a power of 2 and >= 16 — tl.dot's K-dimension floor for the "
        "P@V matmul is a hard >=16 on this Triton/NVIDIA backend (verified empirically, "
        "see kernel_v1_naive.py's docstring)"
    )
    gqa_ratio = num_q_heads // num_kv_heads

    block_table = block_table.to(torch.int32)
    seq_lens = seq_lens.to(torch.int32)
    scale = scale if scale is not None else 1.0 / (head_dim ** 0.5)
    gqa_ratio_padded = triton.next_power_of_2(gqa_ratio)

    # Real GQA_RATIO, not padded — GQA_RATIO_PADDED only ever shapes
    # in-kernel registers/masks, never Python-side tensor allocation
    # (same as `out = torch.empty_like(q)` in v1/v2/v3). torch.empty is
    # safe (uninitialized) because every element is written by exactly
    # one phase-1 program, including empty-chunk programs, which still
    # execute their store with the zero/-inf defaults.
    o_partial = torch.empty(batch, num_kv_heads, num_splits, gqa_ratio, head_dim, dtype=torch.float32, device=q.device)
    m_partial = torch.empty(batch, num_kv_heads, num_splits, gqa_ratio, dtype=torch.float32, device=q.device)
    l_partial = torch.empty(batch, num_kv_heads, num_splits, gqa_ratio, dtype=torch.float32, device=q.device)
    out = torch.empty_like(q)

    grid1 = (batch, num_kv_heads, num_splits)
    _paged_attn_decode_v4_phase1_kernel[grid1](
        q, k_cache, v_cache, block_table, seq_lens,
        o_partial, m_partial, l_partial,
        scale,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_table.stride(0), block_table.stride(1),
        o_partial.stride(0), o_partial.stride(1), o_partial.stride(2), o_partial.stride(3), o_partial.stride(4),
        m_partial.stride(0), m_partial.stride(1), m_partial.stride(2), m_partial.stride(3),
        l_partial.stride(0), l_partial.stride(1), l_partial.stride(2), l_partial.stride(3),
        GQA_RATIO=gqa_ratio,
        GQA_RATIO_PADDED=gqa_ratio_padded,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        BLOCK_N=block_n,
        NUM_SPLITS=num_splits,
    )

    # No explicit sync needed here: both kernels dispatch onto
    # torch.cuda.current_stream(), and CUDA guarantees in-order execution
    # of ops enqueued to the same stream, so phase 2 cannot start reading
    # o_partial/m_partial/l_partial before phase 1 finishes writing them.
    # Adding a torch.cuda.synchronize() here would be a pure, measurable
    # regression — the same class of bug as the assert-induced sync
    # already found and removed from v1/v2's wrappers.
    grid2 = (batch, num_kv_heads)
    _paged_attn_decode_v4_phase2_kernel[grid2](
        o_partial, m_partial, l_partial, out,
        o_partial.stride(0), o_partial.stride(1), o_partial.stride(2), o_partial.stride(3), o_partial.stride(4),
        m_partial.stride(0), m_partial.stride(1), m_partial.stride(2), m_partial.stride(3),
        l_partial.stride(0), l_partial.stride(1), l_partial.stride(2), l_partial.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        GQA_RATIO=gqa_ratio,
        GQA_RATIO_PADDED=gqa_ratio_padded,
        HEAD_DIM=head_dim,
        NUM_SPLITS=num_splits,
    )
    return out
