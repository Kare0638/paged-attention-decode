from __future__ import annotations

import torch

from src._cuda_ext import get_ext

# From bench/bench_cuda_v4_num_splits.py's sweep at batch=1 (median of 9
# interleaved trials): num_splits=64 peaked at 20.75x vs. CUDA v3 — a much
# sharper, larger win than Triton v4's ~1.83x plateau at num_splits=16,
# because CUDA v3's batch=1 baseline (2 thread blocks total) had far more
# idle parallelism to reclaim. Not reused from Triton v4's num_splits=16:
# this kernel's per-block cost (warp-shuffle, no shared memory) is
# structurally different, and the sweep confirms the sweet spot really is
# different (a sharp peak at 64, dropping off by 128, not a flat
# plateau).
_DEFAULT_NUM_SPLITS = 64


def paged_attention_decode_cuda_v4(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    num_splits: int | None = None,
    scale: float | None = None,
) -> torch.Tensor:
    """CUDA v4: split-K along the sequence dimension, built on CUDA v3's
    warp-shuffle reduction (not v1's tree reduction — the best per-block
    building block available now, not the first one; see
    cuda/kernel_v4_split_k.cu). Same merge math as Triton v4
    (analysis/split_k_derivation.md), reused directly, no new derivation.

    Same tensor-shape contract and head_dim % 32 == 0 precondition as
    CUDA v3 (inherited via the reused reduction).
    """
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "kernel_cuda_v4 requires CUDA tensors"
    assert q.dtype == torch.float16 and k_cache.dtype == torch.float16 and v_cache.dtype == torch.float16, (
        "kernel_cuda_v4 expects fp16 q/k_cache/v_cache"
    )

    batch, num_q_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads (GQA)"
    assert head_dim % 32 == 0, (
        "kernel_cuda_v4 requires head_dim to be a multiple of 32 — one warp per block, "
        "each thread owns head_dim/32 lanes (inherited from CUDA v3's warp-shuffle reduction)"
    )
    gqa_ratio = num_q_heads // num_kv_heads

    if num_splits is None:
        assert _DEFAULT_NUM_SPLITS is not None, (
            "num_splits has no default yet — pass it explicitly, or run "
            "bench/bench_cuda_v4_num_splits.py and set _DEFAULT_NUM_SPLITS"
        )
        num_splits = _DEFAULT_NUM_SPLITS

    block_table = block_table.to(torch.int32)
    seq_lens = seq_lens.to(torch.int32)
    scale = scale if scale is not None else 1.0 / (head_dim ** 0.5)

    ext = get_ext()
    return ext.forward_v4(
        q, k_cache, v_cache, block_table, seq_lens, scale, gqa_ratio, head_dim, page_size, num_splits
    )
