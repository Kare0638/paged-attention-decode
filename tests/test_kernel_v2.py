"""Correctness only — no latency/perf assertions here (see bench/bench_decode.py).

v2 reuses v1's exact kernel body (see src/kernel_v2_coalesced.py's module
docstring) with a wider, page_size-independent tile (block_n, default 128).
Test coverage mostly mirrors test_kernel_v1.py at the primary target shape;
what's new here is (a) sweeping block_n itself, and (b) page_size values
v1 rejects (< 16) that v2 now accepts, since decoupling block_n from
page_size also lifted the page_size >= 16 constraint — that was always a
block_n constraint, only misattributed to page_size while the two were
forced equal in v1.
"""

from __future__ import annotations

import pytest
import torch

from src.kernel_v2_coalesced import paged_attention_decode_v2
from tests.kernel_test_utils import compare_to_reference, make_cache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel tests require CUDA")


def _compare(q, k_cache, v_cache, block_table, seq_lens, block_n=128):
    compare_to_reference(
        paged_attention_decode_v2, q, k_cache, v_cache, block_table, seq_lens, block_n=block_n
    )


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


@pytest.mark.parametrize("block_n", [16, 32, 64, 128])
def test_block_n_values_agree(block_n):
    # A tile size spanning multiple pages (block_n > page_size) must give
    # the same answer as one that doesn't — this is the actual thing v2
    # changes, so it's the thing most worth stress-testing directly.
    seq_len, page_size, head_dim, num_kv_heads, gqa_ratio = 1000, 16, 128, 2, 6
    num_q_heads = num_kv_heads * gqa_ratio
    num_pages = -(-seq_len // page_size)

    k_cache, v_cache = make_cache(num_pages, page_size, num_kv_heads, head_dim, seed=300 + block_n)
    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.float32)
    block_table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)

    _compare(q, k_cache, v_cache, block_table, seq_lens, block_n=block_n)


def test_seq_len_shorter_than_block_n():
    # block_n=128 spans 8 pages at page_size=16, but the sequence only
    # occupies 1 page — the masked-out tail of the tile must not corrupt
    # the result (this is the same masking v1 already relies on, but v2's
    # much wider default tile makes the masked region much bigger).
    page_size = 16
    num_kv_heads, gqa_ratio, head_dim = 2, 6, 128

    k_cache, v_cache = make_cache(1, page_size, num_kv_heads, head_dim, seed=301)
    q = torch.randn(1, num_kv_heads * gqa_ratio, head_dim, dtype=torch.float32)
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    seq_lens = torch.tensor([1], dtype=torch.int32)

    _compare(q, k_cache, v_cache, block_table, seq_lens, block_n=128)


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


@pytest.mark.parametrize("page_size", [8, 16, 32])
def test_page_sizes_below_v1_floor(page_size):
    # v1 required page_size >= 16 (a block_n constraint, since block_n was
    # forced equal to page_size there). v2 decouples block_n from
    # page_size entirely, so page_size=8 — which v1 rejects outright —
    # should just work here with block_n staying at its own default.
    seq_len, head_dim, num_kv_heads, gqa_ratio = 100, 128, 2, 6
    num_q_heads = num_kv_heads * gqa_ratio
    num_pages = -(-seq_len // page_size)

    k_cache, v_cache = make_cache(num_pages, page_size, num_kv_heads, head_dim, seed=400 + page_size)
    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.float32)
    block_table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)

    _compare(q, k_cache, v_cache, block_table, seq_lens)
