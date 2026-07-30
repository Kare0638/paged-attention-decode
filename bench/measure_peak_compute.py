"""Measure achieved FP16 matmul throughput — the other half of the roofline.

Same reasoning as measure_peak_bw.py: nominal TFLOPS figures assume desktop
SKUs and don't account for this being a laptop GPU with a variable power
limit, so measure it on this machine instead of trusting a spec sheet. This
is a plain torch.matmul (cuBLAS/tensor-core) benchmark, not a rigorous
speed-of-light microbenchmark — good enough to place the ridge point, not
a claim of the GPU's true theoretical peak.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from measure_peak_bw import _gpu_power_state  # reuse the same metadata helper


def measure_fp16_matmul_tflops(m: int, n: int, k: int, iters: int = 30, warmup: int = 10) -> float:
    a = torch.randn(m, k, dtype=torch.float16, device="cuda")
    b = torch.randn(k, n, dtype=torch.float16, device="cuda")

    for _ in range(warmup):
        torch.matmul(a, b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    best_ms = float("inf")
    for _ in range(iters):
        start.record()
        torch.matmul(a, b)
        end.record()
        torch.cuda.synchronize()
        best_ms = min(best_ms, start.elapsed_time(end))

    flops = 2 * m * n * k  # multiply-add = 2 FLOPs
    seconds = best_ms / 1000.0
    return flops / seconds / 1e12


def main() -> None:
    assert torch.cuda.is_available(), "CUDA device required"

    sizes = [1024, 2048, 4096, 8192]
    results = []
    for s in sizes:
        tflops = measure_fp16_matmul_tflops(s, s, s)
        results.append({"m": s, "n": s, "k": s, "tflops": round(tflops, 2)})
        print(f"{s:5d}^3 fp16 matmul: {tflops:8.2f} TFLOPS")

    peak = max(results, key=lambda r: r["tflops"])
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "measured_peak_tflops_fp16": peak["tflops"],
        "peak_at_size": peak["m"],
        "sweep": results,
        **_gpu_power_state(),
        "method": "torch.matmul fp16 square GEMM, best-of-30 via cuda.Event timing, "
        "FLOPs = 2*M*N*K. Not a speed-of-light microbenchmark.",
    }

    out_path = Path(__file__).parent / "results" / "peak_compute.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nmeasured peak: {peak['tflops']} TFLOPS @ {peak['m']}^3")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
