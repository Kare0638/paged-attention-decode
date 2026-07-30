"""Randomized cross-check of src/reference.py against the independent SDPA oracle.

Default iteration count is kept low so this runs fast as part of routine
`pytest`. Set PAGED_ATTN_FUZZ_ITERS=2000 (as called for in the project plan)
before a milestone / in CI to run the full sweep; each iteration is its own
parametrized test case, so a failure names the exact seed to reproduce.
"""

from __future__ import annotations

import os

import pytest
import torch

from src.reference import paged_attention_decode_reference
from tests.case_generators import random_case
from tests.oracle_sdpa import sdpa_oracle

FUZZ_ITERS = int(os.environ.get("PAGED_ATTN_FUZZ_ITERS", "50"))


@pytest.mark.parametrize("seed", range(FUZZ_ITERS))
def test_fuzz_random_paged_gqa_ragged(seed):
    q, k_cache, v_cache, block_table, seq_lens, page_size = random_case(seed)

    actual = paged_attention_decode_reference(q, k_cache, v_cache, block_table, seq_lens)
    expected = sdpa_oracle(q, k_cache, v_cache, block_table, seq_lens, page_size)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
