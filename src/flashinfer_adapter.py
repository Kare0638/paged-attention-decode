"""Adapter around FlashInfer's BatchDecodeWithPagedKVCacheWrapper, matching
this project's own kernel-function signature (q, k_cache, v_cache,
block_table, seq_lens) -> Tensor.

FlashInfer's paged KV cache tensors are already laid out
[max_num_pages, page_size, num_kv_heads, head_dim] (NHD) -- exactly this
project's own k_cache/v_cache layout throughout src/, so no data-layout
conversion is needed. What differs is the *indexing scheme*: FlashInfer
takes a CSR-style (indptr, indices, last_page_len) triple instead of this
project's dense block_table [batch, max_pages_per_seq] + seq_lens [batch].
_paged_kv_indices() below converts one to the other -- the only real
adapter logic here; verified against the reference oracle in
tests/test_flashinfer_adapter.py, including page-boundary and ragged-batch
cases where the conversion logic actually gets exercised.

use_tensor_cores=True is FlashInfer's own documented recommendation for
GQA decode (this project's config: GQA ratio 6) -- the CUDA-core path
leaves real performance on the table for GQA, so using the CUDA-core
default here would be comparing against a weaker configuration than
FlashInfer is capable of.
"""

from __future__ import annotations

import torch

try:
    import flashinfer
except ImportError as exc:  # pragma: no cover - only hit without optional dependencies
    raise ImportError(
        "flashinfer is not installed in this environment. This project's main venv "
        "deliberately does not depend on it (it pulls in a different torch/CUDA "
        "toolchain) -- install requirements-flashinfer-comparison.txt to run the "
        "comparison. See profiles/notes.md's FlashInfer comparison entry for why."
    ) from exc

_WORKSPACE_BYTES = 128 * 1024 * 1024  # FlashInfer's own recommended size


def _paged_kv_indices(
    block_table: torch.Tensor, seq_lens: torch.Tensor, page_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dense [batch, max_pages_per_seq] block_table + seq_lens -> FlashInfer's
    CSR-style (indptr, indices, last_page_len).

    Per batch item i: num_pages_i = ceil(seq_lens[i] / page_size) pages are
    actually used; block_table[i, :num_pages_i] is that item's slice of
    indices (this project's own kernels already only ever read that same
    prefix -- anything beyond it is documented "holes"). last_page_len_i =
    seq_lens[i] - (num_pages_i - 1) * page_size lands in [1, page_size] for
    both the exact-multiple and remainder cases.
    """
    device = block_table.device
    batch = block_table.shape[0]
    num_pages_per_seq = (seq_lens.to(torch.int64) + page_size - 1) // page_size  # ceil div

    indices_parts = [block_table[i, : num_pages_per_seq[i]] for i in range(batch)]
    indices = torch.cat(indices_parts).to(torch.int32)

    indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    indptr[1:] = torch.cumsum(num_pages_per_seq, dim=0).to(torch.int32)

    last_page_len = (seq_lens.to(torch.int64) - (num_pages_per_seq - 1) * page_size).to(torch.int32)

    return indptr, indices, last_page_len


def paged_attention_decode_flashinfer(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    scale: float | None = None,
) -> torch.Tensor:
    """One-shot convenience wrapper: builds a workspace buffer + wrapper +
    plan() + run() on every call. Matches this project's kernel-function
    signature so it plugs directly into
    tests/kernel_test_utils.compare_to_reference. Not the fast path --
    rebuilding/replanning every call is real, avoidable overhead that a
    real caller (and this project's own bench_flashinfer.py) would not
    pay; see bench_flashinfer.py for the plan-once-run-many benchmark path.
    """
    assert q.is_cuda and k_cache.is_cuda and v_cache.is_cuda, "flashinfer adapter requires CUDA tensors"
    assert q.dtype == torch.float16 and k_cache.dtype == torch.float16 and v_cache.dtype == torch.float16, (
        "flashinfer adapter expects fp16 q/k_cache/v_cache"
    )

    batch, num_q_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape
    assert num_q_heads % num_kv_heads == 0, "num_q_heads must be divisible by num_kv_heads (GQA)"

    block_table = block_table.to(torch.int32)
    seq_lens = seq_lens.to(torch.int32)
    scale = scale if scale is not None else 1.0 / (head_dim**0.5)

    indptr, indices, last_page_len = _paged_kv_indices(block_table, seq_lens, page_size)

    workspace_buffer = torch.zeros(_WORKSPACE_BYTES, dtype=torch.uint8, device=q.device)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD", use_tensor_cores=True)
    wrapper.plan(
        indptr,
        indices,
        last_page_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        sm_scale=scale,
    )
    return wrapper.run(q, (k_cache, v_cache))
