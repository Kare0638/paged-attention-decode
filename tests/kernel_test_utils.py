"""Shared correctness-comparison helper for Triton kernel test files.

Each kernel version (v1, v2, ...) gets its own test file, but they all do
the same thing: round a fp32 reference case's inputs through fp16 (the
kernel's actual dtype), run both the reference oracle and the kernel under
test on those *same* rounded values, and compare with the tolerance this
project documents for fp16-compute/fp32-accumulate kernels — isolating the
kernel's own compute error from fp16 input-quantization error, which are
not the same thing.
"""

from __future__ import annotations

from typing import Callable

import torch

from src.reference import paged_attention_decode_reference

RTOL, ATOL = 1e-2, 1e-3
# A v4 ragged-batch case briefly needed atol=2e-3 to pass (near-zero output
# element, ~20% relative fp16 error by nature). Root cause turned out to be
# upstream of the tolerance itself: compare_to_reference was computing the
# reference from the original fp32 inputs while the kernel only ever saw
# fp16-rounded ones, so the measured diff mixed the kernel's own compute
# error with fp16 input-quantization error. Rounding the reference's inputs
# through fp16 too (see compare_to_reference below) removed that extra
# error term, and atol=1e-3 was enough again — verified across 300-iteration
# fuzz on all four kernel versions (652 cases), not just the one case that
# originally failed.


def compare_to_reference(
    kernel_fn: Callable[..., torch.Tensor],
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    **kernel_kwargs,
) -> None:
    # Round inputs through fp16 *before* computing the reference, not after.
    # The kernel only ever sees fp16-rounded inputs; if the reference ran on
    # the original fp32 values, the measured diff would mix the kernel's own
    # compute error together with fp16 input-quantization error, overstating
    # how imprecise the kernel itself is.
    q_fp16 = q.half()
    k_fp16 = k_cache.half()
    v_fp16 = v_cache.half()

    expected = paged_attention_decode_reference(q_fp16.float(), k_fp16.float(), v_fp16.float(), block_table, seq_lens)

    actual = kernel_fn(
        q_fp16.cuda(),
        k_fp16.cuda(),
        v_fp16.cuda(),
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
