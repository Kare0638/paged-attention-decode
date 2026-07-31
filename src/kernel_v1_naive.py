from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _paged_attn_decode_v1_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, block_table_ptr, seq_lens_ptr, o_ptr,
    scale,
    stride_qb, stride_qh, stride_qd,
    stride_kp, stride_ks, stride_kh, stride_kd,
    stride_vp, stride_vs, stride_vh, stride_vd,
    stride_btb, stride_btp,
    stride_ob, stride_oh, stride_od,
    GQA_RATIO: tl.constexpr,
    GQA_RATIO_PADDED: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    b = tl.program_id(0)
    kvh = tl.program_id(1)

    seq_len = tl.load(seq_lens_ptr + b)

    # tl.arange requires a power-of-2 bound, but GQA_RATIO (e.g. 6 for
    # Qwen2.5-1.5B) usually isn't one. Pad to the next power of 2 and mask
    # the extra lanes on every load/store that touches the head dimension;
    # softmax/matmul rows are independent of each other here (no cross-row
    # reduction in decode attention), so garbage in the padding lanes'
    # intermediate math can't corrupt the real GQA_RATIO rows.
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

    # Structurally required at realistic seq_len (can't materialize a
    # [head_dim, seq_len] tile in one shot), not a "v3 feature" — v3's job
    # is pipelining/num_stages tuning and reduced materialization on top of
    # this same loop, not the loop itself.
    for start_n in range(0, seq_len, BLOCK_N):
        n_offsets = start_n + tl.arange(0, BLOCK_N)
        seq_mask = n_offsets < seq_len

        page_ids = n_offsets // PAGE_SIZE
        slot_ids = n_offsets % PAGE_SIZE

        # Masked-out lanes get physical page 0 (always valid) via `other=0`;
        # the K/V loads below are masked with the same seq_mask, so those
        # lanes' values are never actually used. Address computation happens
        # unconditionally for all lanes, but Triton lowers masked loads to
        # predicated instructions — no memory transaction issues for a
        # masked-off lane, so an address that merely *looks* out of range
        # for those lanes is never dereferenced.
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

    out = acc / l_i[:, None]

    o_ptrs = (
        o_ptr
        + b * stride_ob
        + q_head_offsets[:, None] * stride_oh
        + d_offsets[None, :] * stride_od
    )
    tl.store(o_ptrs, out.to(o_ptr.dtype.element_ty), mask=gqa_mask[:, None])


def paged_attention_decode_v1(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Triton v1: naive paged-decode attention. Correct, not yet optimized for
    memory coalescing (v2), pipelining (v3), or seq-dim parallelism (v4).

    Unlike reference.py's oracle, this does not exhaustively validate every
    input — it's a benchmark-facing kernel, not a public safety API. It
    asserts only the properties that would otherwise produce silently-wrong
    numbers rather than an obvious crash: dtype, device, GQA divisibility,
    and page_size (tl.dot's K-dimension floor is a hard >=16, unlike M/N
    which Triton pads internally — verified empirically, not assumed).

    seq_len >= 1 is a precondition (decode always attends to at least the
    current token; seq_len == 0 is a reference-oracle-only synthetic case),
    documented but deliberately NOT asserted here: `torch.all(seq_lens >=
    1)` forces a device-to-host sync to evaluate as a Python bool, measured
    at ~0.09-0.11ms — 2-3x the raw kernel launch time at batch=1 (0.057ms).
    That's a real cost on every call for a check the test suite already
    exercises during development; caught there instead of paying for it on
    every production call.
    """
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "kernel_v1_naive requires CUDA tensors"
    assert q.dtype == torch.float16 and k_cache.dtype == torch.float16 and v_cache.dtype == torch.float16, (
        "kernel_v1_naive expects fp16 q/k_cache/v_cache"
    )

    batch, num_q_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads (GQA)"
    assert head_dim & (head_dim - 1) == 0, "head_dim must be a power of 2 (tl.arange requirement)"
    assert page_size & (page_size - 1) == 0 and page_size >= 16, (
        "page_size must be a power of 2 and >= 16 — tl.dot's K-dimension floor for the "
        "P@V matmul (BLOCK_N, tied to page_size in this kernel) is a hard >=16 on this "
        "Triton/NVIDIA backend, unlike M/N which get padded internally"
    )
    gqa_ratio = num_q_heads // num_kv_heads

    block_table = block_table.to(torch.int32)
    seq_lens = seq_lens.to(torch.int32)
    scale = scale if scale is not None else 1.0 / (head_dim ** 0.5)

    out = torch.empty_like(q)

    BLOCK_N = page_size  # tile boundaries align with page boundaries by design
    gqa_ratio_padded = triton.next_power_of_2(gqa_ratio)

    grid = (batch, num_kv_heads)
    _paged_attn_decode_v1_kernel[grid](
        q, k_cache, v_cache, block_table, seq_lens, out,
        scale,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_table.stride(0), block_table.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        GQA_RATIO=gqa_ratio,
        GQA_RATIO_PADDED=gqa_ratio_padded,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        BLOCK_N=BLOCK_N,
    )
    return out
