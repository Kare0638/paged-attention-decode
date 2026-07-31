"""Shared correctness-comparison helper for Triton kernel test files.

Each kernel version (v1, v2, ...) gets its own test file, but they all do
the same thing: cast a fp32 reference case to fp16/CUDA, run the kernel
under test, and compare against the reference oracle with the tolerance
this project documents for fp16-compute/fp32-accumulate kernels.
"""

from __future__ import annotations

from typing import Callable

import torch

from src.reference import paged_attention_decode_reference

RTOL, ATOL = 1e-2, 1e-3


def compare_to_reference(
    kernel_fn: Callable[..., torch.Tensor],
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    **kernel_kwargs,
) -> None:
    expected = paged_attention_decode_reference(q, k_cache, v_cache, block_table, seq_lens)

    actual = kernel_fn(
        q.half().cuda(),
        k_cache.half().cuda(),
        v_cache.half().cuda(),
        block_table.cuda(),
        seq_lens.cuda(),
        **kernel_kwargs,
    )

    torch.testing.assert_close(actual.float().cpu(), expected, rtol=RTOL, atol=ATOL)


def make_cache(num_pages, page_size, num_kv_heads, head_dim, seed):
    torch.manual_seed(seed)
    k_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim, dtype=torch.float32)
    v_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim, dtype=torch.float32)
    return k_cache, v_cache
