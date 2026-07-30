import torch

from src.reference import paged_attention_decode_reference
from tests.oracle_sdpa import sdpa_oracle


def _run_and_compare(q, k_cache, v_cache, block_table, seq_lens, page_size):
    actual = paged_attention_decode_reference(q, k_cache, v_cache, block_table, seq_lens)
    expected = sdpa_oracle(q, k_cache, v_cache, block_table, seq_lens, page_size)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    return actual


def _make_cache(num_pages, page_size, num_kv_heads, head_dim, seed):
    torch.manual_seed(seed)
    k_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim, dtype=torch.float32)
    v_cache = torch.randn(num_pages, page_size, num_kv_heads, head_dim, dtype=torch.float32)
    return k_cache, v_cache


def test_page_boundary_not_evenly_divisible():
    # 100 tokens over page_size=16 -> last page is a partial page (4 slots used of 16).
    seq_len, page_size = 100, 16
    num_kv_heads, num_q_heads, head_dim = 2, 12, 32
    num_pages = -(-seq_len // page_size)  # ceil

    k_cache, v_cache = _make_cache(num_pages, page_size, num_kv_heads, head_dim, seed=0)
    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.float32)
    block_table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)

    _run_and_compare(q, k_cache, v_cache, block_table, seq_lens, page_size)


def test_seq_len_of_one():
    page_size = 16
    num_kv_heads, num_q_heads, head_dim = 2, 8, 32

    k_cache, v_cache = _make_cache(1, page_size, num_kv_heads, head_dim, seed=1)
    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.float32)
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    seq_lens = torch.tensor([1], dtype=torch.int32)

    _run_and_compare(q, k_cache, v_cache, block_table, seq_lens, page_size)


def test_seq_len_exactly_one_full_page():
    # seq_len == page_size: no partial last page at all.
    page_size = 16
    num_kv_heads, num_q_heads, head_dim = 2, 12, 32

    k_cache, v_cache = _make_cache(1, page_size, num_kv_heads, head_dim, seed=2)
    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.float32)
    block_table = torch.zeros(1, 1, dtype=torch.int32)
    seq_lens = torch.tensor([page_size], dtype=torch.int32)

    _run_and_compare(q, k_cache, v_cache, block_table, seq_lens, page_size)


def test_shuffled_block_table_with_unreferenced_holes():
    # Physical pages are NOT laid out in logical order, and there are extra
    # physical pages plus extra block_table columns that are never touched.
    # If the implementation reads anything beyond num_seq_pages, or assumes
    # physical order matches logical order, this test catches it.
    seq_len, page_size = 48, 16  # exactly 3 full pages needed
    num_kv_heads, num_q_heads, head_dim = 2, 12, 32
    num_physical_pages = 10  # far more than the 3 actually used
    max_pages_per_seq = 6  # more columns than needed -> unused trailing slots

    k_cache, v_cache = _make_cache(num_physical_pages, page_size, num_kv_heads, head_dim, seed=3)
    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.float32)

    # Logical page 0 -> physical 7, logical 1 -> physical 2, logical 2 -> physical 9.
    # Trailing (unused) slots are filled with a physical id that holds
    # different random data, so reading it by mistake changes the result.
    block_table = torch.tensor([[7, 2, 9, 5, 5, 5]], dtype=torch.int32)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)

    _run_and_compare(q, k_cache, v_cache, block_table, seq_lens, page_size)


def test_gqa_ratios():
    page_size = 16
    seq_len = 40
    head_dim = 32
    num_kv_heads = 2

    for ratio in (1, 4, 6, 8):
        num_q_heads = num_kv_heads * ratio
        num_pages = -(-seq_len // page_size)
        k_cache, v_cache = _make_cache(num_pages, page_size, num_kv_heads, head_dim, seed=100 + ratio)
        q = torch.randn(1, num_q_heads, head_dim, dtype=torch.float32)
        block_table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)
        seq_lens = torch.tensor([seq_len], dtype=torch.int32)

        _run_and_compare(q, k_cache, v_cache, block_table, seq_lens, page_size)


def test_ragged_batch_independent_seq_lens():
    page_size = 8
    num_kv_heads, num_q_heads, head_dim = 2, 12, 32
    seq_lens_list = [3, 8, 17, 64]
    max_pages_per_seq = max(-(-s // page_size) for s in seq_lens_list)
    num_physical_pages = sum(-(-s // page_size) for s in seq_lens_list)

    k_cache, v_cache = _make_cache(num_physical_pages, page_size, num_kv_heads, head_dim, seed=42)
    q = torch.randn(len(seq_lens_list), num_q_heads, head_dim, dtype=torch.float32)

    block_table = torch.zeros(len(seq_lens_list), max_pages_per_seq, dtype=torch.int32)
    cursor = 0
    for b, s in enumerate(seq_lens_list):
        n_pages = -(-s // page_size)
        for j in range(n_pages):
            block_table[b, j] = cursor
            cursor += 1

    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)

    batched = _run_and_compare(q, k_cache, v_cache, block_table, seq_lens, page_size)

    # Batching must not leak state across rows: each row run in isolation
    # (batch=1) must match the corresponding row of the batched call.
    for b in range(len(seq_lens_list)):
        solo = paged_attention_decode_reference(
            q[b : b + 1], k_cache, v_cache, block_table[b : b + 1], seq_lens[b : b + 1]
        )
        torch.testing.assert_close(batched[b : b + 1], solo, rtol=1e-5, atol=1e-5)


def test_zero_length_sequence_in_batch():
    page_size = 16
    num_kv_heads, num_q_heads, head_dim = 2, 8, 32

    k_cache, v_cache = _make_cache(4, page_size, num_kv_heads, head_dim, seed=7)
    q = torch.randn(2, num_q_heads, head_dim, dtype=torch.float32)
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    seq_lens = torch.tensor([0, 20], dtype=torch.int32)

    actual = paged_attention_decode_reference(q, k_cache, v_cache, block_table, seq_lens)
    assert torch.all(actual[0] == 0.0)

    expected_row1 = sdpa_oracle(
        q[1:2], k_cache, v_cache, block_table[1:2], seq_lens[1:2], page_size
    )
    torch.testing.assert_close(actual[1:2], expected_row1, rtol=1e-4, atol=1e-4)


def test_long_sequence():
    # Stresses the gather/slice path at a size close to what a real decode
    # workload sees, not just a handful of tokens.
    seq_len, page_size = 32768, 16
    num_kv_heads, num_q_heads, head_dim = 2, 12, 128
    num_pages = seq_len // page_size

    k_cache, v_cache = _make_cache(num_pages, page_size, num_kv_heads, head_dim, seed=9)
    q = torch.randn(1, num_q_heads, head_dim, dtype=torch.float32)
    block_table = torch.arange(num_pages, dtype=torch.int32).unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32)

    _run_and_compare(q, k_cache, v_cache, block_table, seq_lens, page_size)
