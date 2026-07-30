from __future__ import annotations

import math

import torch


def paged_attention_decode_reference(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """Slow fp32 reference for decode attention over a paged KV cache.

    This implementation is intentionally direct and loop-based. It is meant to
    be a correctness oracle for Triton/CUDA kernels, not a fast baseline.
    """
    _validate_inputs(q, k_cache, v_cache, block_table, seq_lens)

    batch, num_q_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    gqa_ratio = num_q_heads // num_kv_heads
    attn_scale = (1.0 / math.sqrt(head_dim)) if scale is None else scale

    out = torch.empty_like(q)

    for b in range(batch):
        seq_len = int(seq_lens[b].item())
        if seq_len == 0:
            out[b].zero_()
            continue

        num_seq_pages = (seq_len + page_size - 1) // page_size
        phys_pages = block_table[b, :num_seq_pages].to(
            device=k_cache.device, dtype=torch.long
        )
        if torch.any((phys_pages < 0) | (phys_pages >= k_cache.shape[0])):
            raise ValueError("block_table contains an invalid physical page id")

        k_full = k_cache[phys_pages].reshape(-1, num_kv_heads, head_dim)[:seq_len]
        v_full = v_cache[phys_pages].reshape(-1, num_kv_heads, head_dim)[:seq_len]

        k_full = k_full.repeat_interleave(gqa_ratio, dim=1)
        v_full = v_full.repeat_interleave(gqa_ratio, dim=1)

        scores = torch.einsum("hd,shd->hs", q[b], k_full) * attn_scale
        scores = scores - scores.max(dim=-1, keepdim=True).values
        probs = torch.softmax(scores, dim=-1)
        out[b] = torch.einsum("hs,shd->hd", probs, v_full)

    return out


def _validate_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
) -> None:
    if q.ndim != 3:
        raise ValueError(f"q must have shape [batch, num_q_heads, head_dim], got {q.shape}")
    if k_cache.ndim != 4:
        raise ValueError(
            "k_cache must have shape [num_pages, page_size, num_kv_heads, head_dim], "
            f"got {k_cache.shape}"
        )
    if v_cache.shape != k_cache.shape:
        raise ValueError(f"v_cache shape {v_cache.shape} must match k_cache {k_cache.shape}")
    if block_table.ndim != 2:
        raise ValueError(
            f"block_table must have shape [batch, max_pages_per_seq], got {block_table.shape}"
        )
    if seq_lens.ndim != 1:
        raise ValueError(f"seq_lens must have shape [batch], got {seq_lens.shape}")

    batch, num_q_heads, head_dim = q.shape
    _, page_size, num_kv_heads, cache_head_dim = k_cache.shape

    if (
        q.dtype != torch.float32
        or k_cache.dtype != torch.float32
        or v_cache.dtype != torch.float32
    ):
        raise TypeError("q, k_cache, and v_cache must all be torch.float32")
    if block_table.dtype not in (torch.int32, torch.int64):
        raise TypeError("block_table must be torch.int32 or torch.int64")
    if seq_lens.dtype not in (torch.int32, torch.int64):
        raise TypeError("seq_lens must be torch.int32 or torch.int64")
    if q.device != k_cache.device or q.device != v_cache.device:
        raise ValueError("q, k_cache, and v_cache must be on the same device")
    if block_table.shape[0] != batch:
        raise ValueError("block_table batch dimension must match q")
    if seq_lens.shape[0] != batch:
        raise ValueError("seq_lens batch dimension must match q")
    if head_dim != cache_head_dim:
        raise ValueError("q head_dim must match KV cache head_dim")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError("num_q_heads must be divisible by num_kv_heads for GQA")
    if torch.any(seq_lens < 0):
        raise ValueError("seq_lens must be non-negative")

    max_seq_len = block_table.shape[1] * page_size
    if torch.any(seq_lens > max_seq_len):
        raise ValueError("seq_lens contains values larger than block_table can address")
