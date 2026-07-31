"""Correctness only — no latency/perf assertions here (see bench/bench_decode.py).

v3 reuses v1's exact kernel body (see src/kernel_v3_online_softmax.py's
module docstring) with block_n=128 (v2's default) plus an explicit,
tuned num_stages=4. Coverage mirrors test_kernel_v2.py at the primary
target shape; what's new here is sweeping num_stages itself and a
combined block_n/num_stages case.
"""

from __future__ import annotations

import pytest
import torch

from src.kernel_v3_online_softmax import paged_attention_decode_v3
from tests.kernel_test_utils import compare_to_reference, make_cache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel tests require CUDA")


def _compare(q, k_cache, v_cache, block_table, seq_lens, block_n=128, num_stages=4):
    compare_to_reference(
        paged_attention_decode_v3,
        q, k_cache, v_cache, block_table, seq_lens,
        block_n=block_n, num_stages=num_stages,
    )


def test_primary_target_shape():
    # Qwen2.5-1.5B-like: GQA ratio 6, head_dim 128, page_size 16.
    batch, num_kv_heads, gqa_ratio, head_dim, page_size = 4, 2, 6, 128, 16
    num_q_heads = num_kv_heads * gqa_ratio
    seq_lens_list = [1, 17, 256, 4000]

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


@pytest.mark.parametrize("num_stages", [1, 2, 3, 4])
def test_num_stages_values_agree(num_stages):
    seq_len, page_size, head_dim, num_kv_heads, gqa_ratio = 1000, 16, 128, 2, 6
    num_q_heads = num_kv_heads * gqa_ratio
    num_pages = -(-seq_len // page_size)

    k_cache, v_cache = make_cache(num_pages, page_size, num_kv_heads, head_dim, seed=500 + num_stages)
    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.float32)
    block_table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)

    _compare(q, k_cache, v_cache, block_table, seq_lens, num_stages=num_stages)


def test_combined_small_block_n_high_num_stages():
    # v1-style tile (block_n == page_size) with the deeper pipeline that
    # config could actually afford (num_stages=5, which OOMs at block_n=128).
    seq_len, page_size, head_dim, num_kv_heads, gqa_ratio = 500, 16, 128, 2, 6
    num_q_heads = num_kv_heads * gqa_ratio
    num_pages = -(-seq_len // page_size)

    k_cache, v_cache = make_cache(num_pages, page_size, num_kv_heads, head_dim, seed=501)
    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.float32)
    block_table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)

    _compare(q, k_cache, v_cache, block_table, seq_lens, block_n=16, num_stages=5)


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


@pytest.mark.parametrize("gqa_ratio", [1, 4, 6, 8])
def test_gqa_ratios(gqa_ratio):
    page_size, seq_len, head_dim, num_kv_heads = 16, 200, 128, 2
    num_q_heads = num_kv_heads * gqa_ratio
    num_pages = -(-seq_len // page_size)

    k_cache, v_cache = make_cache(num_pages, page_size, num_kv_heads, head_dim, seed=100 + gqa_ratio)
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
