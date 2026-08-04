"""Correctness only — no latency/perf assertions here (see bench/bench_decode.py).

Mirrors tests/test_kernel_v1.py's shape matrix exactly, against
src/kernel_cuda_v1.py (the CUDA C++ naive baseline) instead of the Triton
kernel — same fp16-vs-fp32-reference tolerance (rtol=1e-2/atol=1e-3, see
tests/kernel_test_utils.py), same reasoning for excluding seq_len == 0
(reference-oracle-only synthetic case). Unlike test_kernel_v1.py, there's
no JIT-recompile-per-shape cost here (CUDA v1 is a single compiled kernel
that takes gqa_ratio/head_dim/page_size as runtime arguments, not
compile-time constants), so the shape matrix doesn't need to be narrowed
for that reason — kept identical to test_kernel_v1.py anyway, for a direct
apples-to-apples correctness comparison between the two kernels.
"""

from __future__ import annotations

import os

import pytest
import torch

from src.kernel_cuda_v1 import paged_attention_decode_cuda_v1
from tests.case_generators import random_case
from tests.kernel_test_utils import compare_to_reference, make_cache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel tests require CUDA")

FUZZ_ITERS = int(os.environ.get("PAGED_ATTN_KERNEL_FUZZ_ITERS", "20"))


def _compare(q, k_cache, v_cache, block_table, seq_lens):
    compare_to_reference(paged_attention_decode_cuda_v1, q, k_cache, v_cache, block_table, seq_lens)


def test_primary_target_shape():
    # Qwen2.5-1.5B-like: GQA ratio 6, head_dim 128, page_size 16.
    batch, num_kv_heads, gqa_ratio, head_dim, page_size = 4, 2, 6, 128, 16
    num_q_heads = num_kv_heads * gqa_ratio
    seq_lens_list = [1, 17, 256, 4000]  # short, non-page-aligned, page-aligned-ish, long+ragged

    pages_needed = [-(-s // page_size) for s in seq_lens_list]
    max_pages_per_seq = max(pages_needed)
    num_physical_pages = sum(pages_needed) + 3

    k_cache, v_cache = make_cache(num_physical_pages, page_size, num_kv_heads, head_dim, seed=0)
    q = torch.randn(batch, num_q_heads, head_dim, dtype=torch.float32)

    block_table = torch.zeros(batch, max_pages_per_seq, dtype=torch.int32)
    cursor = 0
    for b, n in enumerate(pages_needed):
        for j in range(n):
            block_table[b, j] = cursor
            cursor += 1
        for j in range(n, max_pages_per_seq):
            block_table[b, j] = cursor % num_physical_pages

    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)
    _compare(q, k_cache, v_cache, block_table, seq_lens)


def test_page_boundary_not_evenly_divisible():
    seq_len, page_size = 100, 16
    num_kv_heads, gqa_ratio, head_dim = 2, 6, 128
    num_q_heads = num_kv_heads * gqa_ratio
    num_pages = -(-seq_len // page_size)

    k_cache, v_cache = make_cache(num_pages, page_size, num_kv_heads, head_dim, seed=1)
    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.float32)
    block_table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)

    _compare(q, k_cache, v_cache, block_table, seq_lens)


def test_seq_len_of_one():
    page_size = 16
    num_kv_heads, gqa_ratio, head_dim = 2, 6, 128
    num_q_heads = num_kv_heads * gqa_ratio

    k_cache, v_cache = make_cache(1, page_size, num_kv_heads, head_dim, seed=2)
    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.float32)
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    seq_lens = torch.tensor([1], dtype=torch.int32)

    _compare(q, k_cache, v_cache, block_table, seq_lens)


def test_shuffled_block_table_with_unreferenced_holes():
    seq_len, page_size = 48, 16  # exactly 3 full pages
    num_kv_heads, gqa_ratio, head_dim = 2, 6, 128
    num_q_heads = num_kv_heads * gqa_ratio
    num_physical_pages = 10

    k_cache, v_cache = make_cache(num_physical_pages, page_size, num_kv_heads, head_dim, seed=3)
    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.float32)
    block_table = torch.tensor([[7, 2, 9, 5, 5, 5]], dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)

    _compare(q, k_cache, v_cache, block_table, seq_lens)


@pytest.mark.parametrize("gqa_ratio", [1, 4, 6, 8])
def test_gqa_ratios(gqa_ratio):
    page_size, seq_len, head_dim, num_kv_heads = 16, 40, 128, 2
    num_q_heads = num_kv_heads * gqa_ratio
    num_pages = -(-seq_len // page_size)

    k_cache, v_cache = make_cache(num_pages, page_size, num_kv_heads, head_dim, seed=100 + gqa_ratio)
    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.float32)
    block_table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)

    _compare(q, k_cache, v_cache, block_table, seq_lens)


@pytest.mark.parametrize("page_size", [16, 32])
def test_page_sizes(page_size):
    seq_len, head_dim, num_kv_heads, gqa_ratio = 100, 128, 2, 6
    num_q_heads = num_kv_heads * gqa_ratio
    num_pages = -(-seq_len // page_size)

    k_cache, v_cache = make_cache(num_pages, page_size, num_kv_heads, head_dim, seed=200 + page_size)
    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.float32)
    block_table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)

    _compare(q, k_cache, v_cache, block_table, seq_lens)


def test_ragged_batch_independent_seq_lens():
    page_size, num_kv_heads, gqa_ratio, head_dim = 16, 2, 6, 128
    num_q_heads = num_kv_heads * gqa_ratio
    seq_lens_list = [3, 17, 100, 512]

    pages_needed = [-(-s // page_size) for s in seq_lens_list]
    max_pages_per_seq = max(pages_needed)
    num_physical_pages = sum(pages_needed) + 3

    k_cache, v_cache = make_cache(num_physical_pages, page_size, num_kv_heads, head_dim, seed=42)
    q = torch.randn(len(seq_lens_list), num_q_heads, head_dim, dtype=torch.float32)

    block_table = torch.zeros(len(seq_lens_list), max_pages_per_seq, dtype=torch.int32)
    cursor = 0
    for b, n in enumerate(pages_needed):
        for j in range(n):
            block_table[b, j] = cursor
            cursor += 1

    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)
    _compare(q, k_cache, v_cache, block_table, seq_lens)


@pytest.mark.parametrize("seed", range(FUZZ_ITERS))
def test_fuzz_curated_shape_matrix(seed):
    q, k_cache, v_cache, block_table, seq_lens, _page_size = random_case(
        seed,
        gqa_ratio_choices=(1, 4, 6, 8),
        head_dim_choices=(32, 128),
        page_size_choices=(16, 32),
        min_seq_len=1,
    )
    _compare(q, k_cache, v_cache, block_table, seq_lens)
