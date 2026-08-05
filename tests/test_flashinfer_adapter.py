"""Correctness only, for the adapter's block_table/seq_lens -> FlashInfer
CSR (indptr/indices/last_page_len) conversion -- not a from-scratch
attention kernel like every other tests/test_kernel_*.py file, so this
isn't the full 30+ case shape matrix; it targets the cases that actually
exercise the conversion logic (page boundaries, ragged batches, GQA
ratios) via the same tests/kernel_test_utils.compare_to_reference used
everywhere else in this project.

Requires flashinfer, which only exists in the isolated .venv-flashinfer
venv (see src/flashinfer_adapter.py's module docstring for why it's not
in this project's main venv) -- skipped entirely, not failed, when it's
not importable, so the main venv's test run stays clean.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("src.flashinfer_adapter")

from src.flashinfer_adapter import paged_attention_decode_flashinfer  # noqa: E402
from tests.kernel_test_utils import compare_to_reference, make_cache  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="kernel tests require CUDA")


def _compare(q, k_cache, v_cache, block_table, seq_lens):
    compare_to_reference(paged_attention_decode_flashinfer, q, k_cache, v_cache, block_table, seq_lens)


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


def test_page_boundary_not_evenly_divisible():
    # Exercises last_page_len's remainder branch directly (100 % 16 != 0).
    seq_len, page_size = 100, 16
    num_kv_heads, gqa_ratio, head_dim = 2, 6, 128
    num_q_heads = num_kv_heads * gqa_ratio
    num_pages = -(-seq_len // page_size)

    k_cache, v_cache = make_cache(num_pages, page_size, num_kv_heads, head_dim, seed=1)
    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.float32)
    block_table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)

    _compare(q, k_cache, v_cache, block_table, seq_lens)


def test_page_boundary_evenly_divisible():
    # last_page_len's other branch: seq_len % page_size == 0 must give
    # last_page_len == page_size, not 0 (checked by hand in the plan;
    # this exercises it against the real conversion code, not just algebra).
    seq_len, page_size = 96, 16
    num_kv_heads, gqa_ratio, head_dim = 2, 6, 128
    num_q_heads = num_kv_heads * gqa_ratio
    num_pages = seq_len // page_size

    k_cache, v_cache = make_cache(num_pages, page_size, num_kv_heads, head_dim, seed=2)
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
    # Exercises indptr/indices construction across non-uniform page counts
    # per batch item -- the part of the CSR conversion a uniform-batch
    # test would never touch.
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
