"""Correctness only — no latency/perf assertions here (see bench/bench_decode.py
and bench/bench_v4_num_splits.py).

v4-specific coverage beyond what v1/v2/v3's suites already established:
split-invariance (same input, different num_splits, must agree), a tight
num_splits=1-vs-v1 equivalence check (see analysis/split_k_derivation.md
for why this should hold near-exactly), and a named oversplit edge case
(more splits requested than the sequence has tokens for).
"""

from __future__ import annotations

import os

import pytest
import torch

from src.kernel_v1_naive import paged_attention_decode_v1
from src.kernel_v4_split_k import paged_attention_decode_v4
from tests.case_generators import random_case
from tests.kernel_test_utils import compare_to_reference, make_cache

FUZZ_ITERS = int(os.environ.get("PAGED_ATTN_KERNEL_FUZZ_ITERS", "20"))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel tests require CUDA")


def _compare(q, k_cache, v_cache, block_table, seq_lens, num_splits=8, block_n=128):
    compare_to_reference(
        paged_attention_decode_v4,
        q, k_cache, v_cache, block_table, seq_lens,
        num_splits=num_splits, block_n=block_n,
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


@pytest.mark.parametrize("num_splits", [1, 2, 4, 8, 16])
def test_split_invariance(num_splits):
    # The most important test for v4 specifically: a reduction bug would
    # silently produce wrong numbers only at particular split counts, so
    # every value here must independently agree with the fp32 reference.
    seq_len, page_size, head_dim, num_kv_heads, gqa_ratio = 2000, 16, 128, 2, 6
    num_q_heads = num_kv_heads * gqa_ratio
    num_pages = -(-seq_len // page_size)

    k_cache, v_cache = make_cache(num_pages, page_size, num_kv_heads, head_dim, seed=600 + num_splits)
    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.float32)
    block_table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)

    _compare(q, k_cache, v_cache, block_table, seq_lens, num_splits=num_splits)


def test_num_splits_one_matches_v1_exactly():
    # analysis/split_k_derivation.md: at num_splits=1, phase 1's loop is
    # byte-for-byte v1's loop and phase 2's merge degenerates to v1's
    # final acc/l_i. With block_n=16 matching v1's hardcoded
    # BLOCK_N=page_size exactly (so both run the identical sequence of
    # tl.dot calls), this should match v1 bit-for-bit, not just within
    # the fp16-vs-fp32-reference tolerance.
    batch, num_kv_heads, gqa_ratio, head_dim, page_size = 3, 2, 6, 128, 16
    num_q_heads = num_kv_heads * gqa_ratio
    seq_lens_list = [1, 17, 4000]

    pages_needed = [-(-s // page_size) for s in seq_lens_list]
    max_pages_per_seq = max(pages_needed)
    num_physical_pages = sum(pages_needed) + 3

    k_cache, v_cache = make_cache(num_physical_pages, page_size, num_kv_heads, head_dim, seed=42)
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

    q16, k16, v16 = q.half().cuda(), k_cache.half().cuda(), v_cache.half().cuda()
    bt_cuda, sl_cuda = block_table.cuda(), seq_lens.cuda()

    v1_out = paged_attention_decode_v1(q16, k16, v16, bt_cuda, sl_cuda)
    v4_out = paged_attention_decode_v4(q16, k16, v16, bt_cuda, sl_cuda, num_splits=1, block_n=16)

    torch.testing.assert_close(v4_out, v1_out, rtol=1e-4, atol=1e-5)


def test_oversplit_more_splits_than_tokens():
    # seq_len=5, num_splits=8: chunk_size=ceil(5/8)=1, so splits 5-7 have
    # split_start >= seq_len (empty chunks) — exercises both the
    # zero-trip-loop path in phase 1 and the -inf reduction guard in
    # phase 2 directly, not just incidentally via fuzz.
    page_size, head_dim, num_kv_heads, gqa_ratio = 16, 128, 2, 6
    num_q_heads = num_kv_heads * gqa_ratio

    k_cache, v_cache = make_cache(1, page_size, num_kv_heads, head_dim, seed=700)
    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.float32)
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    seq_lens = torch.tensor([5], dtype=torch.int32)

    _compare(q, k_cache, v_cache, block_table, seq_lens, num_splits=8)


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

    k_cache, v_cache = make_cache(num_physical_pages, page_size, num_kv_heads, head_dim, seed=43)
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
    _compare(q, k_cache, v_cache, block_table, seq_lens, num_splits=4)
