from __future__ import annotations

import torch
import triton

from src.kernel_v1_naive import _paged_attn_decode_v1_kernel

"""v2: wider tiles, decoupled from page_size.

Before writing this, I measured (not assumed) where v1's actual headroom
was: total-sector efficiency was already ~97% of theoretical minimum, and
sweeping num_warps found Triton's own default (4) was already the best
choice at both batch=1 and batch=64 — the "obvious" memory-coalescing
story from the original plan doesn't hold up against v1's numbers.

What did move the needle: v1 ties BLOCK_N to page_size (16), so at seq_len
2048 it runs 128 loop iterations, each starting with a `block_table` load
that the K/V load depends on — a dependent-load chain repeated 128 times,
which matches `long_scoreboard` (memory-wait) being the dominant stall
reason in the batch=1 profile. Decoupling BLOCK_N from page_size and
processing several pages per iteration amortizes that dependent lookup
over more data and cuts the loop trip count. Swept BLOCK_N on the
unmodified v1 kernel body first (same JIT function, just a different
launch config) before writing this file: 16->32->64->128 gave
0.123->0.100->0.069->0.057 ms at batch=1 (a 2.15x speedup at BLOCK_N=128,
correctness-preserving — max diff 6e-5 against BLOCK_N=16, well inside
fp16 tolerance). BLOCK_N=256 hits this GPU's shared-memory limit
(138,240 bytes required vs. 101,376 available) and fails to compile, so
128 is the largest tile size that actually works here.

This kernel reuses v1's exact `_paged_attn_decode_v1_kernel` — the compute
never changed, only the launch configuration did. v1's page/slot indexing
math (`n_offset // PAGE_SIZE`, `n_offset % PAGE_SIZE`) was already written
to handle BLOCK_N spanning multiple pages, so no kernel-body change was
needed, only the wrapper's tiling choice.

Side effect: since BLOCK_N is no longer forced equal to page_size, v1's
`page_size >= 16` constraint (really a BLOCK_N constraint, misattributed
to page_size only because the two were equal at the time) no longer
applies here — page_size itself was never subject to tl.arange/tl.dot's
constraints, only whatever BLOCK_N was tied to it. v2 accepts any
power-of-2 page_size, including the smaller values v1 rejected.
"""


def paged_attention_decode_v2(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float | None = None,
    block_n: int = 128,
) -> torch.Tensor:
    """Same contract as paged_attention_decode_v1 (see its docstring for the
    non-exhaustive-validation rationale). block_n defaults to 128, the
    largest tile size that fit this GPU's shared memory at the primary
    target shape (head_dim=128) — pass a smaller power of 2 (>=16) for
    configs where 128 doesn't fit.
    """
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "kernel_v2_coalesced requires CUDA tensors"
    assert q.dtype == torch.float16 and k_cache.dtype == torch.float16 and v_cache.dtype == torch.float16, (
        "kernel_v2_coalesced expects fp16 q/k_cache/v_cache"
    )
    assert torch.all(seq_lens >= 1), (
        "kernel_v2_coalesced requires seq_len >= 1 for every sequence "
        "(seq_len == 0 is only meaningful for the reference oracle's test coverage)"
    )

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

    out = torch.empty_like(q)
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
        BLOCK_N=block_n,
    )
    return out
