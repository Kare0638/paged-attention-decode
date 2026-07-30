"""Randomized cross-check of src/reference.py against the independent SDPA oracle.

Default iteration count is kept low so this runs fast as part of routine
`pytest`. Set PAGED_ATTN_FUZZ_ITERS=2000 (as called for in the project plan)
before a milestone / in CI to run the full sweep; each iteration is its own
parametrized test case, so a failure names the exact seed to reproduce.
"""

from __future__ import annotations

import math
import os
import random

import pytest
import torch

from src.reference import paged_attention_decode_reference
from tests.oracle_sdpa import sdpa_oracle

FUZZ_ITERS = int(os.environ.get("PAGED_ATTN_FUZZ_ITERS", "50"))


def _random_case(seed: int):
    rng = random.Random(seed)
    torch.manual_seed(seed)

    num_kv_heads = rng.choice([1, 2, 4])
    gqa_ratio = rng.choice([1, 2, 4, 6, 8])
    num_q_heads = num_kv_heads * gqa_ratio
    head_dim = rng.choice([16, 32, 64, 128])
    page_size = rng.choice([1, 4, 8, 16, 32])
    batch = rng.randint(1, 4)

    seq_lens_list = [rng.randint(0, 200) for _ in range(batch)]
    pages_needed = [math.ceil(s / page_size) if s > 0 else 0 for s in seq_lens_list]
    max_pages_per_seq = max(max(pages_needed), 1) + rng.randint(0, 2)  # a few unused hole columns

    total_logical_slots = sum(pages_needed)
    num_physical_pages = total_logical_slots + rng.randint(2, 5)  # spare pages that are never referenced

    physical_ids = list(range(num_physical_pages))
    rng.shuffle(physical_ids)  # physical layout is intentionally NOT logical order

    block_table = torch.zeros(batch, max_pages_per_seq, dtype=torch.int32)
    cursor = 0
    for b in range(batch):
        n = pages_needed[b]
        for j in range(n):
            block_table[b, j] = physical_ids[cursor]
            cursor += 1
        for j in range(n, max_pages_per_seq):
            block_table[b, j] = physical_ids[cursor % num_physical_pages]  # unused slot, never read

    k_cache = torch.randn(num_physical_pages, page_size, num_kv_heads, head_dim, dtype=torch.float32)
    v_cache = torch.randn(num_physical_pages, page_size, num_kv_heads, head_dim, dtype=torch.float32)
    q = torch.randn(batch, num_q_heads, head_dim, dtype=torch.float32)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)

    return q, k_cache, v_cache, block_table, seq_lens, page_size


@pytest.mark.parametrize("seed", range(FUZZ_ITERS))
def test_fuzz_random_paged_gqa_ragged(seed):
    q, k_cache, v_cache, block_table, seq_lens, page_size = _random_case(seed)

    actual = paged_attention_decode_reference(q, k_cache, v_cache, block_table, seq_lens)
    expected = sdpa_oracle(q, k_cache, v_cache, block_table, seq_lens, page_size)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
