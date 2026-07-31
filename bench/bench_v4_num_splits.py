"""Sweep num_splits at batch=1 — the axis that sets v4's wrapper default.

Matching this project's own established discipline (v2's block_n=128,
v3's num_stages=4 both came from sweeps, not guesses): num_splits has no
default in kernel_v4_split_k.py until this sweep produces one.

Fixed at batch=1 specifically — the realistic single-request decode
scenario, and the case v1/v2/v3 all get stuck at grid=(1, num_kv_heads)=2
thread blocks regardless of tuning. Also compares against v1 (no split)
as the baseline this whole exercise is trying to beat.

At these sub-0.1ms kernel latencies, "best of N within one measurement"
(this project's usual `_time_cuda` pattern) turned out to be dominated by
run-to-run GPU clock/power-state noise on this laptop GPU rather than by
real differences between num_splits values — repeated runs picked
different "best" num_splits each time. Fixed by taking the median across
several independent, interleaved trials per config instead of a single
best-of-N: interleaving (round-robin across configs each round, not all
of config A's iterations then all of config B's) spreads any thermal
drift over the sweep evenly across configs rather than systematically
favoring whichever config happens to run first or last, and the median
of several trials is far less sensitive to one lucky low-noise sample
than a single best-of-N reading is.
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
from src.kernel_v1_naive import paged_attention_decode_v1
from src.kernel_v4_split_k import paged_attention_decode_v4

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

    fns = {"v1": lambda: paged_attention_decode_v1(q, k_cache, v_cache, block_table, seq_lens)}
    for ns in NUM_SPLITS_VALUES:
        fns[f"v4_ns{ns}"] = (
            lambda ns=ns: paged_attention_decode_v4(q, k_cache, v_cache, block_table, seq_lens, num_splits=ns)
        )

    samples = {name: [] for name in fns}
    config_names = list(fns.keys())
    for trial in range(TRIALS):
        for name in config_names:  # same order every trial; round-robin across trials via the outer loop
            samples[name].append(_time_cuda_once(fns[name]))
        print(f"trial {trial + 1}/{TRIALS} done")

    medians = {name: statistics.median(vals) for name, vals in samples.items()}
    v1_ms = medians["v1"]
    print(f"\nv1 (no split), median of {TRIALS} trials: {v1_ms:.4f} ms")

    results = []
    for ns in NUM_SPLITS_VALUES:
        v4_ms = medians[f"v4_ns{ns}"]
        speedup_vs_v1 = v1_ms / v4_ms
        results.append(
            {
                "num_splits": ns,
                "v4_ms_median": round(v4_ms, 4),
                "v4_ms_all_trials": [round(x, 4) for x in samples[f"v4_ns{ns}"]],
                "speedup_vs_v1": round(speedup_vs_v1, 2),
            }
        )
        print(f"num_splits={ns:4d}: median {v4_ms:9.4f} ms | {speedup_vs_v1:6.2f}x vs v1")

    best = max(results, key=lambda r: r["speedup_vs_v1"])
    print(f"\nbest: num_splits={best['num_splits']} ({best['speedup_vs_v1']}x vs v1, median of {TRIALS} trials)")

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
        "v1_ms_median": round(v1_ms, 4),
        "v1_ms_all_trials": [round(x, 4) for x in samples["v1"]],
        "sweep": results,
        "best_num_splits": best["num_splits"],
        "trials_per_config": TRIALS,
        **_gpu_power_state(),
        "method": f"cuda.Event timing, best-of-50 after 15 warmup iters PER TRIAL, "
        f"{TRIALS} independent trials per config in round-robin order (not blocked "
        "per config) to spread thermal/clock drift evenly, median across trials "
        "reported. v4 uses its default block_n=128. Swept at batch=1 specifically, "
        "the scenario v1/v2/v3 are stuck at grid=(1, num_kv_heads)=2 thread blocks "
        "regardless of tuning.",
    }

    out_path = Path(__file__).parent / "results" / "v4_num_splits_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
