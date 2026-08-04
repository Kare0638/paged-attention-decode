from __future__ import annotations

import torch

from src._cuda_ext import get_ext


def paged_attention_decode_cuda_v1(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """CUDA v1: naive, unoptimized port of kernel_v1_naive.py's algorithm to
    raw CUDA C++ (see cuda/kernel_v1_naive.cu for why it's deliberately more
    naive than Triton v1: token-by-token dot products instead of a tiled
    matmul, one independent block-wide reduction per (row, token) pair
    instead of one reduction batched across all gqa_ratio rows).

    Same tensor-shape contract as paged_attention_decode_v1: q
    [batch, num_q_heads, head_dim]; k_cache/v_cache
    [num_pages, page_size, num_kv_heads, head_dim]; block_table
    [batch, max_pages_per_seq]; seq_lens [batch]. seq_len >= 1 is a
    documented precondition, not asserted, for the same reason as Triton
    v1 (see that file's docstring) — a tensor-value check would force a
    device-to-host sync.

    This wrapper's validation is a genuine strict subset of Triton v1's:
    CUDA has no `tl.arange` power-of-2 constraint (no head_dim-is-pow2
    check needed) and no `tl.dot` K>=16 floor (no page_size>=16 check
    needed) — those were Triton/NVIDIA-backend-specific, not general CUDA
    constraints, so they don't carry over.
    """
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "kernel_cuda_v1 requires CUDA tensors"
    assert q.dtype == torch.float16 and k_cache.dtype == torch.float16 and v_cache.dtype == torch.float16, (
        "kernel_cuda_v1 expects fp16 q/k_cache/v_cache"
    )

    batch, num_q_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads (GQA)"
    gqa_ratio = num_q_heads // num_kv_heads

    block_table = block_table.to(torch.int32)
    seq_lens = seq_lens.to(torch.int32)
    scale = scale if scale is not None else 1.0 / (head_dim ** 0.5)

    ext = get_ext()
    return ext.forward_v1(q, k_cache, v_cache, block_table, seq_lens, scale, gqa_ratio, head_dim, page_size)
