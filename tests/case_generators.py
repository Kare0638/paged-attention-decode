"""Shared randomized test-case generation for paged-decode correctness tests.

Used by both the reference fuzz suite (full shape matrix) and the Triton
kernel fuzz suite (a curated, narrower matrix — every distinct
GQA_RATIO/HEAD_DIM/PAGE_SIZE combination triggers a separate Triton JIT
specialization, so the kernel test deliberately doesn't sweep the same
2000-case cross product the reference does).
"""

from __future__ import annotations

import math
import random
from typing import Sequence

import torch


def random_case(
    seed: int,
    *,
    kv_heads_choices: Sequence[int] = (1, 2, 4),
    gqa_ratio_choices: Sequence[int] = (1, 2, 4, 6, 8),
    head_dim_choices: Sequence[int] = (16, 32, 64, 128),
    page_size_choices: Sequence[int] = (1, 4, 8, 16, 32),
    max_batch: int = 4,
    min_seq_len: int = 0,
    max_seq_len: int = 200,
):
    """Generate one randomized (q, k_cache, v_cache, block_table, seq_lens, page_size)
    case: ragged batch, shuffled physical page layout, unreferenced "hole"
    pages/slots, and a random GQA/head_dim/page_size combination drawn from
    the given choices.
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    num_kv_heads = rng.choice(kv_heads_choices)
    gqa_ratio = rng.choice(gqa_ratio_choices)
    num_q_heads = num_kv_heads * gqa_ratio
    head_dim = rng.choice(head_dim_choices)
    page_size = rng.choice(page_size_choices)
    batch = rng.randint(1, max_batch)

    seq_lens_list = [rng.randint(min_seq_len, max_seq_len) for _ in range(batch)]
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
