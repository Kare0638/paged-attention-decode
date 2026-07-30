"""Independent ground truth for paged-decode correctness tests.

Deliberately does NOT reuse any code from src/reference.py — it walks the
block table with a plain Python loop + torch.cat instead of the fancy-index
gather reference.py uses, and defers the actual attention math to
torch.nn.functional.scaled_dot_product_attention instead of a hand-rolled
softmax. Two independently-written implementations agreeing is real
evidence; comparing reference.py against itself would not be.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sdpa_oracle(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    batch, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    gqa_ratio = num_q_heads // num_kv_heads

    out = torch.empty_like(q)
    for b in range(batch):
        seq_len = int(seq_lens[b])
        if seq_len == 0:
            out[b] = 0.0
            continue

        k_chunks = []
        v_chunks = []
        remaining = seq_len
        page_idx = 0
        while remaining > 0:
            phys_page = int(block_table[b, page_idx])
            take = min(page_size, remaining)
            k_chunks.append(k_cache[phys_page, :take])
            v_chunks.append(v_cache[phys_page, :take])
            remaining -= take
            page_idx += 1

        k_full = torch.cat(k_chunks, dim=0)  # [seq_len, num_kv_heads, head_dim]
        v_full = torch.cat(v_chunks, dim=0)

        k_full = k_full.repeat_interleave(gqa_ratio, dim=1).transpose(0, 1)  # [num_q_heads, seq_len, head_dim]
        v_full = v_full.repeat_interleave(gqa_ratio, dim=1).transpose(0, 1)

        q_b = q[b].unsqueeze(1)  # [num_q_heads, 1, head_dim]
        o = F.scaled_dot_product_attention(q_b, k_full, v_full)  # [num_q_heads, 1, head_dim]
        out[b] = o.squeeze(1)

    return out
