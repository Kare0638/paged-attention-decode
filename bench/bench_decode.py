"""Latency: fp32 reference (per-batch Python loop) vs. Triton v1 vs. v2.

All three run on GPU so the comparison isolates "one fused kernel over the
whole batch" vs. "a Python loop issuing many small per-sequence GPU
launches" — not a CPU-vs-GPU comparison. Each implementation runs in its
own natural dtype (reference.py is fp32-only by design; the kernels are
fp16), matching how each would actually be used, not an artificially
matched dtype.

Fixed at the primary target shape (Qwen2.5-1.5B-like: GQA ratio 6, head_dim
128, page_size 16) and a representative mid-length context (seq_len 2048),
sweeping batch size — this is the axis that matters for the project's
occupancy story (small batch * few KV heads = most SMs idle on this GPU).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `src.*` imports

import torch

from measure_peak_bw import _gpu_power_state  # same dir as this script; reuse the metadata helper
from src.kernel_v1_naive import paged_attention_decode_v1
from src.kernel_v2_coalesced import paged_attention_decode_v2
from src.reference import paged_attention_decode_reference

NUM_KV_HEADS = 2
GQA_RATIO = 6
NUM_Q_HEADS = NUM_KV_HEADS * GQA_RATIO
HEAD_DIM = 128
PAGE_SIZE = 16
SEQ_LEN = 2048


def _make_uniform_batch(batch: int, seq_len: int, seed: int = 0):
    torch.manual_seed(seed)
    num_pages_per_seq = -(-seq_len // PAGE_SIZE)
    num_physical_pages = batch * num_pages_per_seq

    k_cache = torch.randn(num_physical_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32)
    v_cache = torch.randn(num_physical_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float32)
    q = torch.randn(batch, NUM_Q_HEADS, HEAD_DIM, dtype=torch.float32)

    block_table = torch.arange(num_physical_pages, dtype=torch.int32).reshape(batch, num_pages_per_seq)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32)

    return q, k_cache, v_cache, block_table, seq_lens


def _time_cuda(fn, iters: int = 30, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    best_ms = float("inf")
    for _ in range(iters):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        best_ms = min(best_ms, start.elapsed_time(end))
    return best_ms


def main() -> None:
    assert torch.cuda.is_available(), "CUDA device required"

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    results = []

    for batch in batch_sizes:
        q, k_cache, v_cache, block_table, seq_lens = _make_uniform_batch(batch, SEQ_LEN)

        q_fp32 = q.cuda()
        k_fp32 = k_cache.cuda()
        v_fp32 = v_cache.cuda()
        bt_cuda = block_table.cuda()
        sl_cuda = seq_lens.cuda()

        q_fp16 = q.half().cuda()
        k_fp16 = k_cache.half().cuda()
        v_fp16 = v_cache.half().cuda()

        ref_ms = _time_cuda(
            lambda: paged_attention_decode_reference(q_fp32, k_fp32, v_fp32, bt_cuda, sl_cuda)
        )
        v1_ms = _time_cuda(
            lambda: paged_attention_decode_v1(q_fp16, k_fp16, v_fp16, bt_cuda, sl_cuda)
        )
        v2_ms = _time_cuda(
            lambda: paged_attention_decode_v2(q_fp16, k_fp16, v_fp16, bt_cuda, sl_cuda)
        )

        results.append(
            {
                "batch": batch,
                "seq_len": SEQ_LEN,
                "reference_ms": round(ref_ms, 4),
                "kernel_v1_ms": round(v1_ms, 4),
                "kernel_v2_ms": round(v2_ms, 4),
                "v1_speedup_vs_reference": round(ref_ms / v1_ms, 2),
                "v2_speedup_vs_reference": round(ref_ms / v2_ms, 2),
                "v2_speedup_vs_v1": round(v1_ms / v2_ms, 2),
            }
        )
        print(
            f"batch={batch:3d}: reference {ref_ms:9.4f} ms | v1 {v1_ms:8.4f} ms | "
            f"v2 {v2_ms:8.4f} ms | v2 vs v1 {v1_ms / v2_ms:5.2f}x | v2 vs reference {ref_ms / v2_ms:6.1f}x"
        )

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "shape": {
            "num_kv_heads": NUM_KV_HEADS,
            "gqa_ratio": GQA_RATIO,
            "head_dim": HEAD_DIM,
            "page_size": PAGE_SIZE,
            "seq_len": SEQ_LEN,
        },
        "sweep": results,
        **_gpu_power_state(),
        "method": "cuda.Event timing, best-of-30 after 10 warmup iters. reference.py "
        "runs fp32 on GPU (per-batch Python loop); kernels run fp16 (single fused "
        "kernel over the whole batch). Not a dtype-matched comparison — each "
        "implementation in its natural/enforced dtype. v2 uses its default "
        "block_n=128 (v1 has no block_n knob; its tile is fixed to page_size).",
    }

    out_path = Path(__file__).parent / "results" / "decode_latency.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
