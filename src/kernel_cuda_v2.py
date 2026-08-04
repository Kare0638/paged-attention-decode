from __future__ import annotations

import torch

from src._cuda_ext import get_ext


def _validate(q, k_cache, v_cache):
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "kernel_cuda_v2 requires CUDA tensors"
    assert q.dtype == torch.float16 and k_cache.dtype == torch.float16 and v_cache.dtype == torch.float16, (
        "kernel_cuda_v2 expects fp16 q/k_cache/v_cache"
    )


def paged_attention_decode_cuda_v2(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """CUDA v2: batches the score reduction across all gqa_ratio query rows
    into one shared-memory tile per token, instead of CUDA v1's one
    independent block-wide tree reduction per (row, token) pair (see
    cuda/kernel_v2_shared_tile.cu for the mechanism and why bank-conflict
    padding is measured via an A/B rather than assumed necessary).

    Same tensor-shape contract and (reduced, CUDA-specific) validation as
    paged_attention_decode_cuda_v1 — see that function's docstring for why
    Triton v1's power-of-2/K>=16 checks don't carry over to CUDA.
    """
    _validate(q, k_cache, v_cache)

    batch, num_q_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads (GQA)"
    gqa_ratio = num_q_heads // num_kv_heads

    block_table = block_table.to(torch.int32)
    seq_lens = seq_lens.to(torch.int32)
    scale = scale if scale is not None else 1.0 / (head_dim ** 0.5)

    ext = get_ext()
    return ext.forward_v2(q, k_cache, v_cache, block_table, seq_lens, scale, gqa_ratio, head_dim, page_size)


def _paged_attention_decode_cuda_v2_unpadded(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Unpadded shared-memory-tile variant, for the bank-conflict NCU A/B
    only — not part of the public kernel lineup, see
    cuda/kernel_v2_shared_tile.cu's module docstring.
    """
    _validate(q, k_cache, v_cache)

    batch, num_q_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads (GQA)"
    gqa_ratio = num_q_heads // num_kv_heads

    block_table = block_table.to(torch.int32)
    seq_lens = seq_lens.to(torch.int32)
    scale = scale if scale is not None else 1.0 / (head_dim ** 0.5)

    ext = get_ext()
    return ext.forward_v2_unpadded(
        q, k_cache, v_cache, block_table, seq_lens, scale, gqa_ratio, head_dim, page_size
    )
