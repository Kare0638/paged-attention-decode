"""Latency: fp32 reference (per-batch Python loop) vs. Triton v1/v2/v3/v4
vs. CUDA C++ v1/v2/v3.

All eight run on GPU so the comparison isolates "one fused kernel over the
whole batch" vs. "a Python loop issuing many small per-sequence GPU
launches" — not a CPU-vs-GPU comparison. Each implementation runs in its
own natural dtype (reference.py is fp32-only by design; the kernels are
fp16), matching how each would actually be used, not an artificially
matched dtype. cuda_v1 is deliberately more naive than Triton v1 (see
cuda/kernel_v1_naive.cu) — token-by-token dot products, each row running
its own independent block-wide shared-memory tree reduction instead of
one reduction batched across all GQA rows — so it is expected to lose to
every Triton version here; that's the honest baseline cuda_v2 (batched
shared-memory reduction across GQA rows, see cuda/kernel_v2_shared_tile.cu)
is measured against — a null result on latency (see profiles/notes.md).
cuda_v3 (warp-shuffle reduction, one warp per block, see
cuda/kernel_v3_warp_shuffle.cu) replaces the reduction algorithm itself
rather than reorganizing the same one, the direct follow-up to that
finding.

Fixed at the primary target shape (Qwen2.5-1.5B-like: GQA ratio 6, head_dim
128, page_size 16) and a representative mid-length context (seq_len 2048),
sweeping batch size — this is the axis that matters for the project's
occupancy story (small batch * few KV heads = most SMs idle on this GPU).

Median-of-interleaved-trials, not a single best-of-N: at these
sub-millisecond kernel latencies, a single best-of-N reading turned out
to be dominated by run-to-run GPU clock/power-state noise on this laptop
GPU rather than real differences between implementations (discovered
while sweeping v4's num_splits — repeated identical runs picked different
"best" configs each time). Interleaving trials round-robin across all
five implementations per round, rather than finishing one implementation's
iterations before starting the next, spreads any thermal drift evenly
instead of systematically favoring whichever runs first or last.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `src.*` imports

import torch

from measure_peak_bw import _gpu_power_state  # same dir as this script; reuse the metadata helper
from src.kernel_cuda_v1 import paged_attention_decode_cuda_v1
from src.kernel_cuda_v2 import paged_attention_decode_cuda_v2
from src.kernel_cuda_v3 import paged_attention_decode_cuda_v3
from src.kernel_v1_naive import paged_attention_decode_v1
from src.kernel_v2_coalesced import paged_attention_decode_v2
from src.kernel_v3_online_softmax import paged_attention_decode_v3
from src.kernel_v4_split_k import paged_attention_decode_v4
from src.reference import paged_attention_decode_reference

NUM_KV_HEADS = 2
GQA_RATIO = 6
NUM_Q_HEADS = NUM_KV_HEADS * GQA_RATIO
HEAD_DIM = 128
PAGE_SIZE = 16
SEQ_LEN = 2048

TRIALS = 15  # independent, interleaved best-of-N measurements per implementation


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


def _time_cuda_once(fn, iters: int = 30, warmup: int = 10) -> float:
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


def _median_of_trials(fns: dict) -> dict:
    """fns: {name: callable}. Returns {name: median_ms}, interleaving
    trials round-robin across all names rather than finishing one name's
    trials before starting the next."""
    samples = {name: [] for name in fns}
    for _ in range(TRIALS):
        for name, fn in fns.items():
            samples[name].append(_time_cuda_once(fn))
    return {name: statistics.median(vals) for name, vals in samples.items()}


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

        fns = {
            "reference": lambda: paged_attention_decode_reference(q_fp32, k_fp32, v_fp32, bt_cuda, sl_cuda),
            "v1": lambda: paged_attention_decode_v1(q_fp16, k_fp16, v_fp16, bt_cuda, sl_cuda),
            "v2": lambda: paged_attention_decode_v2(q_fp16, k_fp16, v_fp16, bt_cuda, sl_cuda),
            "v3": lambda: paged_attention_decode_v3(q_fp16, k_fp16, v_fp16, bt_cuda, sl_cuda),
            "v4": lambda: paged_attention_decode_v4(q_fp16, k_fp16, v_fp16, bt_cuda, sl_cuda),
            "cuda_v1": lambda: paged_attention_decode_cuda_v1(q_fp16, k_fp16, v_fp16, bt_cuda, sl_cuda),
            "cuda_v2": lambda: paged_attention_decode_cuda_v2(q_fp16, k_fp16, v_fp16, bt_cuda, sl_cuda),
            "cuda_v3": lambda: paged_attention_decode_cuda_v3(q_fp16, k_fp16, v_fp16, bt_cuda, sl_cuda),
        }
        m = _median_of_trials(fns)
        ref_ms, v1_ms, v2_ms, v3_ms, v4_ms, cuda_v1_ms, cuda_v2_ms, cuda_v3_ms = (
            m["reference"], m["v1"], m["v2"], m["v3"], m["v4"], m["cuda_v1"], m["cuda_v2"], m["cuda_v3"],
        )

        results.append(
            {
                "batch": batch,
                "seq_len": SEQ_LEN,
                "reference_ms": round(ref_ms, 4),
                "kernel_v1_ms": round(v1_ms, 4),
                "kernel_v2_ms": round(v2_ms, 4),
                "kernel_v3_ms": round(v3_ms, 4),
                "kernel_v4_ms": round(v4_ms, 4),
                "kernel_cuda_v1_ms": round(cuda_v1_ms, 4),
                "kernel_cuda_v2_ms": round(cuda_v2_ms, 4),
                "kernel_cuda_v3_ms": round(cuda_v3_ms, 4),
                "v1_speedup_vs_reference": round(ref_ms / v1_ms, 2),
                "v2_speedup_vs_reference": round(ref_ms / v2_ms, 2),
                "v3_speedup_vs_reference": round(ref_ms / v3_ms, 2),
                "v4_speedup_vs_reference": round(ref_ms / v4_ms, 2),
                "cuda_v1_speedup_vs_reference": round(ref_ms / cuda_v1_ms, 2),
                "cuda_v2_speedup_vs_reference": round(ref_ms / cuda_v2_ms, 2),
                "cuda_v3_speedup_vs_reference": round(ref_ms / cuda_v3_ms, 2),
                "v2_speedup_vs_v1": round(v1_ms / v2_ms, 2),
                "v3_speedup_vs_v2": round(v2_ms / v3_ms, 2),
                "v4_speedup_vs_v1": round(v1_ms / v4_ms, 2),
                "cuda_v1_speedup_vs_triton_v1": round(v1_ms / cuda_v1_ms, 2),
                "cuda_v2_speedup_vs_cuda_v1": round(cuda_v1_ms / cuda_v2_ms, 2),
                "cuda_v2_speedup_vs_triton_v1": round(v1_ms / cuda_v2_ms, 2),
                "cuda_v3_speedup_vs_cuda_v1": round(cuda_v1_ms / cuda_v3_ms, 2),
                "cuda_v3_speedup_vs_cuda_v2": round(cuda_v2_ms / cuda_v3_ms, 2),
                "cuda_v3_speedup_vs_triton_v1": round(v1_ms / cuda_v3_ms, 2),
            }
        )
        print(
            f"batch={batch:3d}: reference {ref_ms:9.4f} ms | v1 {v1_ms:8.4f} ms | "
            f"v2 {v2_ms:8.4f} ms | v3 {v3_ms:8.4f} ms | v4 {v4_ms:8.4f} ms | "
            f"cuda_v1 {cuda_v1_ms:9.4f} ms | cuda_v2 {cuda_v2_ms:9.4f} ms | "
            f"cuda_v3 {cuda_v3_ms:9.4f} ms | v4 vs v1 {v1_ms / v4_ms:5.2f}x | "
            f"cuda_v3 vs cuda_v1 {cuda_v1_ms / cuda_v3_ms:5.2f}x | "
            f"cuda_v3 vs v1 {v1_ms / cuda_v3_ms:5.2f}x"
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
        "trials_per_config": TRIALS,
        **_gpu_power_state(),
        "method": f"cuda.Event timing, best-of-30 after 10 warmup iters PER TRIAL, "
        f"{TRIALS} independent trials per implementation in round-robin order "
        "(not blocked per implementation) to spread thermal/clock drift evenly, "
        "median across trials reported. reference.py runs fp32 on GPU (per-batch "
        "Python loop); kernels run fp16 (single fused kernel over the whole "
        "batch, v4 = phase1+phase2). Not a dtype-matched comparison — each "
        "implementation in its natural/enforced dtype. v2/v3/v4 use block_n=128 "
        "(v1 has no block_n knob; its tile is fixed to page_size); v3 additionally "
        "pins num_stages=4; v4 uses num_splits=16 (from bench_v4_num_splits.py's "
        "sweep at batch=1). cuda_v1 is a raw CUDA C++ extension (torch.utils."
        "cpp_extension JIT), grid=(batch, num_kv_heads) same as Triton v1 but "
        "token-by-token dot products, each GQA row running its own independent "
        "block-wide shared-memory tree reduction (see cuda/kernel_v1_naive.cu) — "
        "a deliberately more naive baseline than Triton v1, not a tuned competitor. "
        "cuda_v2 batches that reduction across all GQA rows into one shared-memory "
        "tile per token instead of one reduction per row (see "
        "cuda/kernel_v2_shared_tile.cu), the padded (production) variant. cuda_v3 "
        "replaces the reduction algorithm itself with warp-shuffle, one warp per "
        "block (blockDim=32) instead of v1/v2's blockDim=head_dim (see "
        "cuda/kernel_v3_warp_shuffle.cu).",
    }

    out_path = Path(__file__).parent / "results" / "decode_latency.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
