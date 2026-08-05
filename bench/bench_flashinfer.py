"""Latency: this project's best Triton kernel (v4) and best CUDA C++
kernel (cuda_v4), plus the fp32 reference floor, against FlashInfer's
BatchDecodeWithPagedKVCacheWrapper -- a production-grade external
baseline, not another from-scratch implementation of this project's own.

Originally scoped as an A100 comparison; run instead on the same RTX 3060
Laptop (Ampere sm_86, same family as A100's sm_80) everything else in
this project was measured on, in the same environment as every other
number in this project (torch 2.11.0+cu128, triton 3.6.0) -- an earlier
attempt to install flashinfer pulled in a newer torch/CUDA/triton stack
by default, which would have been a real confound (Triton recompiles
kernels itself; a different Triton version could change codegen and
performance independent of anything this project did) if FlashInfer's
numbers had come from a different environment than v4/cuda_v4's. Checked
directly: flashinfer-python and its GPU-toolkit dependencies (cuda-python,
nvidia-cutlass-dsl) run correctly against this project's pinned
torch/triton after all (verified via tests/test_flashinfer_adapter.py,
8/8 passing) -- the earlier forced upgrade was uv's dependency resolver
picking the newest mutually-compatible set from a blank slate, not an
actual hard requirement. So this benchmark, like every other one in this
project, runs all four implementations in one process, one environment,
interleaved trials -- a true apples-to-apples comparison, not two
separate runs stitched together on paper.

use_tensor_cores=True for FlashInfer's wrapper -- its own documented
recommendation for GQA decode (this project's config: GQA ratio 6); the
CUDA-core path leaves real performance on the table for GQA, so using the
default would be comparing against a weaker configuration than FlashInfer
is capable of.

FlashInfer's plan() (indptr/indices/last_page_len -> reusable auxiliary
structures) is called once per batch size, outside the trial loop --
only run() is timed. This matches FlashInfer's own documented usage
(plan once, run many times for a fixed batch/shape) and this project's
existing "assert-forced-a-sync" lesson about not measuring one-time setup
as if it were a per-call cost.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `src.*` imports

import torch

import flashinfer
from measure_peak_bw import _gpu_power_state  # same dir as this script; reuse the metadata helper
from src.flashinfer_adapter import _paged_kv_indices
from src.kernel_cuda_v4 import paged_attention_decode_cuda_v4
from src.kernel_v4_split_k import paged_attention_decode_v4
from src.reference import paged_attention_decode_reference

NUM_KV_HEADS = 2
GQA_RATIO = 6
NUM_Q_HEADS = NUM_KV_HEADS * GQA_RATIO
HEAD_DIM = 128
PAGE_SIZE = 16
SEQ_LEN = 2048

TRIALS = 15  # matches bench_decode.py's methodology


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
    samples = {name: [] for name in fns}
    for _ in range(TRIALS):
        for name, fn in fns.items():
            samples[name].append(_time_cuda_once(fn))
    return {name: statistics.median(vals) for name, vals in samples.items()}


def main() -> None:
    assert torch.cuda.is_available(), "CUDA device required"

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    results = []

    workspace_buffer = torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    scale = 1.0 / (HEAD_DIM**0.5)

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

        # FlashInfer: plan() once per batch size, outside the timed region.
        indptr, indices, last_page_len = _paged_kv_indices(bt_cuda, sl_cuda, PAGE_SIZE)
        fi_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD", use_tensor_cores=True)
        fi_wrapper.plan(
            indptr,
            indices,
            last_page_len,
            NUM_Q_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            q_data_type=torch.float16,
            sm_scale=scale,
        )

        fns = {
            "reference": lambda: paged_attention_decode_reference(q_fp32, k_fp32, v_fp32, bt_cuda, sl_cuda),
            "v4": lambda: paged_attention_decode_v4(q_fp16, k_fp16, v_fp16, bt_cuda, sl_cuda),
            "cuda_v4": lambda: paged_attention_decode_cuda_v4(q_fp16, k_fp16, v_fp16, bt_cuda, sl_cuda),
            "flashinfer": lambda: fi_wrapper.run(q_fp16, (k_fp16, v_fp16)),
        }
        m = _median_of_trials(fns)
        ref_ms, v4_ms, cuda_v4_ms, fi_ms = m["reference"], m["v4"], m["cuda_v4"], m["flashinfer"]

        results.append(
            {
                "batch": batch,
                "seq_len": SEQ_LEN,
                "reference_ms": round(ref_ms, 4),
                "kernel_v4_ms": round(v4_ms, 4),
                "kernel_cuda_v4_ms": round(cuda_v4_ms, 4),
                "flashinfer_ms": round(fi_ms, 4),
                "v4_speedup_vs_reference": round(ref_ms / v4_ms, 2),
                "cuda_v4_speedup_vs_reference": round(ref_ms / cuda_v4_ms, 2),
                "flashinfer_speedup_vs_reference": round(ref_ms / fi_ms, 2),
                "flashinfer_speedup_vs_v4": round(v4_ms / fi_ms, 2),
                "flashinfer_speedup_vs_cuda_v4": round(cuda_v4_ms / fi_ms, 2),
            }
        )
        print(
            f"batch={batch:3d}: reference {ref_ms:9.4f} ms | v4 {v4_ms:8.4f} ms | "
            f"cuda_v4 {cuda_v4_ms:9.4f} ms | flashinfer {fi_ms:8.4f} ms | "
            f"flashinfer vs v4 {v4_ms / fi_ms:6.2f}x | flashinfer vs cuda_v4 {cuda_v4_ms / fi_ms:6.2f}x"
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
        "flashinfer_version": flashinfer.__version__,
        "method": f"cuda.Event timing, best-of-30 after 10 warmup iters PER TRIAL, "
        f"{TRIALS} independent trials per implementation in round-robin order, median "
        "across trials reported -- same methodology as bench_decode.py. All four "
        "implementations run in this project's single pinned environment (torch "
        "2.11.0+cu128, triton 3.6.0) so the comparison is a true single-process, "
        "single-environment measurement, not two runs stitched together (see this "
        "file's module docstring for why that mattered here). FlashInfer's plan() is "
        "called once per batch size, outside the timed trial loop -- only run() is "
        "timed, matching FlashInfer's own documented plan-once/run-many usage. "
        "use_tensor_cores=True (FlashInfer's own recommendation for GQA decode).",
    }

    out_path = Path(__file__).parent / "results" / "flashinfer_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
