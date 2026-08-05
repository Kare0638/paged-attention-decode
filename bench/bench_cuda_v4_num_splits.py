"""Sweep num_splits at batch=1 for CUDA v4 — the axis that sets its
wrapper default.

Same discipline as bench_v4_num_splits.py's Triton sweep (v2's block_n,
v3's num_stages: measured, not guessed) — but compared against CUDA v3,
not CUDA v1 or Triton v1, since v3 is what CUDA v4 needs to beat, and v3's
per-block cost (warp-shuffle, no shared memory) is structurally different
from what Triton v1's num_splits=16 sweet spot was found against, so
that default isn't assumed to transfer here.

Same median-of-interleaved-trials methodology as every other bench script
in this project — single best-of-N readings are dominated by run-to-run
GPU clock/power-state noise at these latencies, not real config
differences (first discovered sweeping Triton v4's num_splits).
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
from src.kernel_cuda_v3 import paged_attention_decode_cuda_v3
from src.kernel_cuda_v4 import paged_attention_decode_cuda_v4

NUM_KV_HEADS = 2
GQA_RATIO = 6
NUM_Q_HEADS = NUM_KV_HEADS * GQA_RATIO
HEAD_DIM = 128
PAGE_SIZE = 16
SEQ_LEN = 2048
BATCH = 1

NUM_SPLITS_VALUES = [1, 2, 4, 8, 16, 32, 64, 128]
TRIALS = 9  # independent, interleaved best-of-N measurements per config


def _make_batch(batch: int, seq_len: int, seed: int = 0):
    torch.manual_seed(seed)
    num_pages_per_seq = -(-seq_len // PAGE_SIZE)
    num_physical_pages = batch * num_pages_per_seq

    k_cache = torch.randn(num_physical_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
    v_cache = torch.randn(num_physical_pages, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")
    q = torch.randn(batch, NUM_Q_HEADS, HEAD_DIM, dtype=torch.float16, device="cuda")

    block_table = torch.arange(num_physical_pages, dtype=torch.int32, device="cuda").reshape(batch, num_pages_per_seq)
    seq_lens = torch.full((batch,), seq_len, dtype=torch.int32, device="cuda")

    return q, k_cache, v_cache, block_table, seq_lens


def _time_cuda_once(fn, iters: int = 50, warmup: int = 15) -> float:
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

    q, k_cache, v_cache, block_table, seq_lens = _make_batch(BATCH, SEQ_LEN)

    fns = {"cuda_v3": lambda: paged_attention_decode_cuda_v3(q, k_cache, v_cache, block_table, seq_lens)}
    for ns in NUM_SPLITS_VALUES:
        fns[f"cuda_v4_ns{ns}"] = (
            lambda ns=ns: paged_attention_decode_cuda_v4(q, k_cache, v_cache, block_table, seq_lens, num_splits=ns)
        )

    samples = {name: [] for name in fns}
    config_names = list(fns.keys())
    for trial in range(TRIALS):
        for name in config_names:  # same order every trial; round-robin across trials via the outer loop
            samples[name].append(_time_cuda_once(fns[name]))
        print(f"trial {trial + 1}/{TRIALS} done")

    medians = {name: statistics.median(vals) for name, vals in samples.items()}
    v3_ms = medians["cuda_v3"]
    print(f"\ncuda_v3 (no split), median of {TRIALS} trials: {v3_ms:.4f} ms")

    results = []
    for ns in NUM_SPLITS_VALUES:
        v4_ms = medians[f"cuda_v4_ns{ns}"]
        speedup_vs_v3 = v3_ms / v4_ms
        results.append(
            {
                "num_splits": ns,
                "cuda_v4_ms_median": round(v4_ms, 4),
                "cuda_v4_ms_all_trials": [round(x, 4) for x in samples[f"cuda_v4_ns{ns}"]],
                "speedup_vs_cuda_v3": round(speedup_vs_v3, 2),
            }
        )
        print(f"num_splits={ns:4d}: median {v4_ms:9.4f} ms | {speedup_vs_v3:6.2f}x vs cuda_v3")

    best = max(results, key=lambda r: r["speedup_vs_cuda_v3"])
    print(f"\nbest: num_splits={best['num_splits']} ({best['speedup_vs_cuda_v3']}x vs cuda_v3, median of {TRIALS} trials)")

    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "shape": {
            "batch": BATCH,
            "num_kv_heads": NUM_KV_HEADS,
            "gqa_ratio": GQA_RATIO,
            "head_dim": HEAD_DIM,
            "page_size": PAGE_SIZE,
            "seq_len": SEQ_LEN,
        },
        "cuda_v3_ms_median": round(v3_ms, 4),
        "cuda_v3_ms_all_trials": [round(x, 4) for x in samples["cuda_v3"]],
        "sweep": results,
        "best_num_splits": best["num_splits"],
        "trials_per_config": TRIALS,
        **_gpu_power_state(),
        "method": f"cuda.Event timing, best-of-50 after 15 warmup iters PER TRIAL, "
        f"{TRIALS} independent trials per config in round-robin order (not blocked "
        "per config) to spread thermal/clock drift evenly, median across trials "
        "reported. Swept at batch=1 specifically, the scenario CUDA v1/v2/v3 are "
        "stuck at grid=(1, num_kv_heads)=2 thread blocks regardless of tuning. "
        "Compared against CUDA v3 (not v1 or Triton v1) since v3 is what v4 needs "
        "to beat.",
    }

    out_path = Path(__file__).parent / "results" / "cuda_v4_num_splits_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
