import torch
import torch.nn.functional as F

from src.reference import paged_attention_decode_reference


def test_reference_matches_sdpa_without_paging_or_ragged_batch():
    torch.manual_seed(0)

    batch = 2
    seq_len = 8
    page_size = 4
    num_q_heads = 4
    num_kv_heads = 2
    head_dim = 16

    q = torch.randn(batch, num_q_heads, head_dim, dtype=torch.float32)
    k_tokens = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=torch.float32)
    v_tokens = torch.randn(batch, seq_len, num_kv_heads, head_dim, dtype=torch.float32)

    k_cache = k_tokens.reshape(batch * (seq_len // page_size), page_size, num_kv_heads, head_dim)
    v_cache = v_tokens.reshape(batch * (seq_len // page_size), page_size, num_kv_heads, head_dim)
    block_table = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32)

    actual = paged_attention_decode_reference(q, k_cache, v_cache, block_table, seq_lens)

    k_expanded = k_tokens.repeat_interleave(num_q_heads // num_kv_heads, dim=2)
    v_expanded = v_tokens.repeat_interleave(num_q_heads // num_kv_heads, dim=2)
    expected = F.scaled_dot_product_attention(
        q.unsqueeze(2),
        k_expanded.transpose(1, 2),
        v_expanded.transpose(1, 2),
    ).squeeze(2)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

