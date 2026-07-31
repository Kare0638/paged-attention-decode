from __future__ import annotations

import torch
import triton

from src.kernel_v1_naive import _paged_attn_decode_v1_kernel

"""v3: explicit num_stages tuning on top of v2's tiling.

The roadmap's v3 item was "single-pass online softmax, don't materialize
the intermediate attention matrix, tune num_stages." The first two were
already true of v1/v2's design before this file existed — the running
`m_i`/`l_i`/`acc` update is structurally required at real seq_len (can't
materialize a [head_dim, seq_len] tile in one shot), not something added
here. The only genuinely untried part of the roadmap item was
`num_stages`, which both v1 and v2 left at Triton's default — 3, per
`triton/backends/nvidia/compiler.py`'s `CUDAOptions` dataclass
(`num_stages: int = 3`), read directly from the installed 3.6.0 source,
not assumed.

Swept it directly on the unmodified kernel body before writing this file
(same discipline as v2's BLOCK_N sweep — measure before building):
BLOCK_N=128 (v2's config), batch=1, num_stages 1->2->3->4 gave
0.059->0.047->0.052->0.047 ms; num_stages>=5 hits the same shared-memory
ceiling BLOCK_N=256 hit in v2 (OutOfResources: wider tiles and deeper
pipelining spend the same limited shared-memory budget). num_stages=3
(Triton's actual default) already lands close to num_stages=4's result
in this isolated sweep (0.052 vs. 0.047 ms) — v3's real change is 3->4,
not "unknown auto-selected value -> 4".

A real A/B through the full wrapper (bench_decode.py, v3 vs v2, both
fp16) shows v3's pinned num_stages=4 landing within noise of v2's
default (num_stages=3) across most batch sizes (0.98x-1.00x, one outlier
at batch=8 of 1.35x) — consistent with the isolated sweep above showing
3 and 4 close together, and the same conclusion the num_warps sweep
reached for v2. Reported as what it is: a lever that was worth checking,
already close to Triton's own default, not a real win to claim.

Reuses v1's exact `_paged_attn_decode_v1_kernel`, same as v2. The only
change from v2 is exposing `num_stages` as an explicit launch parameter
instead of leaving it at Triton's default of 3.
"""


def paged_attention_decode_v3(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float | None = None,
    block_n: int = 128,
    num_stages: int = 4,
) -> torch.Tensor:
    """Same contract as paged_attention_decode_v1/v2 (see v1's docstring for
    the non-exhaustive-validation rationale and why seq_len >= 1 is a
    documented precondition rather than a runtime-asserted one). block_n
    and num_stages default to the best combination found sweeping
    BLOCK_N=128 (v2's tile size) directly — num_stages=4 tied for fastest
    at both batch=1 and batch=64 and stayed clear of the shared-memory
    ceiling that num_stages>=5 hits at this tile size.
    """
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "kernel_v3_online_softmax requires CUDA tensors"
    assert q.dtype == torch.float16 and k_cache.dtype == torch.float16 and v_cache.dtype == torch.float16, (
        "kernel_v3_online_softmax expects fp16 q/k_cache/v_cache"
    )

    batch, num_q_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads (GQA)"
    assert head_dim & (head_dim - 1) == 0, "head_dim must be a power of 2 (tl.arange requirement)"
    assert page_size & (page_size - 1) == 0, "page_size must be a power of 2 (page/slot index arithmetic)"
    assert block_n & (block_n - 1) == 0 and block_n >= 16, (
        "block_n must be a power of 2 and >= 16 — tl.dot's K-dimension floor for the "
        "P@V matmul is a hard >=16 on this Triton/NVIDIA backend (verified empirically, "
        "see kernel_v1_naive.py's docstring)"
    )
    assert num_stages >= 1, "num_stages must be >= 1"
    gqa_ratio = num_q_heads // num_kv_heads

    block_table = block_table.to(torch.int32)
    seq_lens = seq_lens.to(torch.int32)
    scale = scale if scale is not None else 1.0 / (head_dim ** 0.5)

    out = torch.empty_like(q)
    gqa_ratio_padded = triton.next_power_of_2(gqa_ratio)

    grid = (batch, num_kv_heads)
    _paged_attn_decode_v1_kernel[grid](
        q, k_cache, v_cache, block_table, seq_lens, out,
        scale,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_table.stride(0), block_table.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        GQA_RATIO=gqa_ratio,
        GQA_RATIO_PADDED=gqa_ratio_padded,
        HEAD_DIM=head_dim,
        PAGE_SIZE=page_size,
        BLOCK_N=block_n,
        num_stages=num_stages,
    )
    return out
