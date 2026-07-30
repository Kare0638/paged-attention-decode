"""Measure achieved D2D memory bandwidth — the roofline denominator.

Nominal bandwidth numbers are unreliable across RTX 3060 SKUs (desktop
12GB/192-bit, desktop 8GB/128-bit, and multiple laptop variants all exist),
and a laptop GPU's achievable bandwidth additionally depends on the active
power limit (AC vs battery, OEM power mode). So: measure it directly on
this machine, in its current power state, instead of trusting a spec sheet.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import torch


def _parse_float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None  # e.g. "[N/A]" — some fields aren't queryable under WSL2


def _gpu_power_state() -> dict:
    fields = "power.draw,power.limit,clocks.sm,name,driver_version"
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    draw, limit, clock_sm, name, driver = [x.strip() for x in out.split(",")]
    return {
        "gpu_name": name,
        "driver_version": driver,
        "power_draw_w": _parse_float_or_none(draw),
        "power_limit_w": _parse_float_or_none(limit),
        "sm_clock_mhz": _parse_float_or_none(clock_sm),
    }


def measure_d2d_bandwidth(size_bytes: int, iters: int = 50, warmup: int = 10) -> float:
    n_floats = size_bytes // 4
    src = torch.empty(n_floats, dtype=torch.float32, device="cuda")
    dst = torch.empty(n_floats, dtype=torch.float32, device="cuda")
    src.uniform_()

    for _ in range(warmup):
        dst.copy_(src)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    best_ms = float("inf")
    for _ in range(iters):
        start.record()
        dst.copy_(src)
        end.record()
        torch.cuda.synchronize()
        best_ms = min(best_ms, start.elapsed_time(end))

    # D2D copy moves size_bytes read + size_bytes write.
    bytes_moved = 2 * size_bytes
    seconds = best_ms / 1000.0
    return bytes_moved / seconds


def main() -> None:
    assert torch.cuda.is_available(), "CUDA device required"

    sizes_mb = [64, 128, 256, 512, 1024]
    results = []
    for mb in sizes_mb:
        size_bytes = mb * 1024 * 1024
        bw_bytes_per_s = measure_d2d_bandwidth(size_bytes)
        bw_gb_s = bw_bytes_per_s / 1e9
        results.append({"buffer_mb": mb, "bandwidth_gb_s": round(bw_gb_s, 2)})
        print(f"{mb:5d} MB buffer: {bw_gb_s:8.2f} GB/s")

    peak = max(results, key=lambda r: r["bandwidth_gb_s"])
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "measured_peak_bandwidth_gb_s": peak["bandwidth_gb_s"],
        "peak_at_buffer_mb": peak["buffer_mb"],
        "sweep": results,
        **_gpu_power_state(),
        "method": "D2D torch.Tensor.copy_, best-of-50 via cuda.Event timing, "
        "bandwidth = 2*bytes/time (read+write)",
    }

    out_path = Path(__file__).parent / "results" / "peak_bw.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nmeasured peak: {peak['bandwidth_gb_s']} GB/s @ {peak['buffer_mb']} MB")
    print(f"power state at measurement: {record['power_draw_w']}W / {record['power_limit_w']}W limit")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
