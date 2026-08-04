from __future__ import annotations

import torch

from src._cuda_ext import get_ext


def paged_attention_decode_cuda_v3(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """CUDA v3: replaces CUDA v1/v2's tree-reduction-plus-`__syncthreads()`
    algorithm with warp-shuffle (see cuda/kernel_v3_warp_shuffle.cu) — a
    genuinely different reduction algorithm, not another grouping of the
    same one (that was v2's lever, and it measured no latency win). One
    warp per block (blockDim=32); each thread owns head_dim/32 lanes.

    Same tensor-shape contract as paged_attention_decode_cuda_v1/v2, plus
    one new precondition specific to this one-warp-per-block design:
    head_dim must be a multiple of 32. That's not a general CUDA
    constraint (v1/v2 have no head_dim requirement at all) — it's this
    kernel's own, since warp-shuffle only reduces within a 32-thread
    warp and each thread needs an integer number of head_dim lanes.
    """
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "kernel_cuda_v3 requires CUDA tensors"
    assert q.dtype == torch.float16 and k_cache.dtype == torch.float16 and v_cache.dtype == torch.float16, (
        "kernel_cuda_v3 expects fp16 q/k_cache/v_cache"
    )

    batch, num_q_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads (GQA)"
    assert head_dim % 32 == 0, (
        "kernel_cuda_v3 requires head_dim to be a multiple of 32 — one warp per block, "
        "each thread owns head_dim/32 lanes (see cuda/kernel_v3_warp_shuffle.cu)"
    )
    gqa_ratio = num_q_heads // num_kv_heads

    block_table = block_table.to(torch.int32)
    seq_lens = seq_lens.to(torch.int32)
    scale = scale if scale is not None else 1.0 / (head_dim ** 0.5)

    ext = get_ext()
    return ext.forward_v3(q, k_cache, v_cache, block_table, seq_lens, scale, gqa_ratio, head_dim, page_size)
