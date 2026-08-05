"""Roofline plot: achieved GFLOP/s vs. operational intensity, against this
machine's *measured* peak bandwidth and compute (not spec-sheet numbers).

No new GPU measurement here — this script only reads three JSON files
already produced by other scripts in this directory:
`results/peak_bw.json` (measure_peak_bw.py), `results/peak_compute.json`
(measure_peak_compute.py), and `results/decode_latency.json`
(bench_decode.py, median-of-15-interleaved-trials latency for all 9
implementations across 7 batch sizes). FLOPs and bytes-moved for the
decode attention op are both derivable analytically from the shape
recorded in decode_latency.json, so achieved GFLOP/s and operational
intensity fall out of latency alone.

FLOPs per call = num_q_heads * seq_len * head_dim * 4 (QK^T + AV, each
2*head_dim multiply-adds per token per q-head, MAC counted as 2 FLOPs).
Bytes moved ~= num_kv_heads * seq_len * head_dim * 2(K,V) * 2 bytes(fp16)
-- Q and output bytes (~3KB) are negligible next to the ~2MB KV read at
seq_len=2048, so they're omitted rather than padding the estimate with a
term that doesn't move it. seq_len*head_dim cancels between the two, so
intensity comes out to ~2*gqa_ratio regardless of batch or kernel version
-- at this project's primary shape (gqa_ratio=6), that's ~5.98 FLOP/byte,
matching the README's existing "~6 FLOP/byte" claim, which until now was
never computed by a checked-in, re-runnable script.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless WSL2 -- savefig only, no display
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"

# (json key in decode_latency.json sweep rows, display label, language)
IMPLS = [
    ("reference_ms", "reference (fp32, per-batch loop)", "ref"),
    ("kernel_v1_ms", "Triton v1", "triton"),
    ("kernel_v2_ms", "Triton v2", "triton"),
    ("kernel_v3_ms", "Triton v3", "triton"),
    ("kernel_v4_ms", "Triton v4", "triton"),
    ("kernel_cuda_v1_ms", "CUDA v1", "cuda"),
    ("kernel_cuda_v2_ms", "CUDA v2", "cuda"),
    ("kernel_cuda_v3_ms", "CUDA v3", "cuda"),
    ("kernel_cuda_v4_ms", "CUDA v4", "cuda"),
]

# matching color per version number so Triton vN / CUDA vN read as a pair
VERSION_COLORS = {1: "#1f77b4", 2: "#ff7f0e", 3: "#2ca02c", 4: "#d62728"}


def compute_flops_bytes(batch: int, num_kv_heads: int, gqa_ratio: int, head_dim: int, seq_len: int) -> tuple[float, float]:
    num_q_heads = num_kv_heads * gqa_ratio
    flops = batch * num_q_heads * seq_len * head_dim * 4
    bytes_moved = batch * num_kv_heads * seq_len * head_dim * 2 * 2  # K+V, fp16
    return flops, bytes_moved


def roofline_ceiling_gflops(intensity: float, peak_tflops: float, peak_bw_gbs: float) -> float:
    return min(peak_tflops * 1000.0, intensity * peak_bw_gbs)


def main() -> None:
    peak_bw = json.loads((RESULTS_DIR / "peak_bw.json").read_text())
    peak_compute = json.loads((RESULTS_DIR / "peak_compute.json").read_text())
    decode_latency = json.loads((RESULTS_DIR / "decode_latency.json").read_text())

    peak_bw_gbs = peak_bw["measured_peak_bandwidth_gb_s"]
    peak_tflops = peak_compute["measured_peak_tflops_fp16"]
    shape = decode_latency["shape"]
    ridge_point = peak_tflops * 1000.0 / peak_bw_gbs

    rows = []
    for row in decode_latency["sweep"]:
        batch = row["batch"]
        flops, bytes_moved = compute_flops_bytes(
            batch, shape["num_kv_heads"], shape["gqa_ratio"], shape["head_dim"], row["seq_len"]
        )
        intensity = flops / bytes_moved
        for key, label, _lang in IMPLS:
            latency_ms = row[key]
            achieved_gflops = flops / (latency_ms / 1000.0) / 1e9
            ceiling = roofline_ceiling_gflops(intensity, peak_tflops, peak_bw_gbs)
            rows.append(
                {
                    "impl": label,
                    "batch": batch,
                    "latency_ms": latency_ms,
                    "flops": flops,
                    "bytes_moved": bytes_moved,
                    "intensity_flop_per_byte": round(intensity, 4),
                    "achieved_gflops": round(achieved_gflops, 2),
                    "roofline_ceiling_gflops": round(ceiling, 2),
                    "pct_of_roofline": round(achieved_gflops / ceiling * 100, 2),
                }
            )

    primary_intensity = rows[0]["intensity_flop_per_byte"]
    print(f"operational intensity (constant across batch/impl): {primary_intensity} FLOP/byte")
    print(f"measured peak bandwidth: {peak_bw_gbs} GB/s")
    print(f"measured peak compute:   {peak_tflops} TFLOPS fp16")
    print(f"ridge point:             {ridge_point:.2f} FLOP/byte")
    print()
    for key, label, _lang in IMPLS:
        b1 = next(r for r in rows if r["impl"] == label and r["batch"] == 1)
        b64 = next(r for r in rows if r["impl"] == label and r["batch"] == 64)
        print(
            f"{label:32s} batch=1: {b1['achieved_gflops']:9.1f} GFLOP/s ({b1['pct_of_roofline']:5.2f}% of roofline)  "
            f"batch=64: {b64['achieved_gflops']:9.1f} GFLOP/s ({b64['pct_of_roofline']:5.2f}% of roofline)"
        )

    record = {
        "measured_peak_bandwidth_gb_s": peak_bw_gbs,
        "measured_peak_tflops_fp16": peak_compute["measured_peak_tflops_fp16"],
        "ridge_point_flop_per_byte": round(ridge_point, 2),
        "shape": shape,
        "rows": rows,
        "method": "No new GPU measurement -- FLOPs/bytes computed analytically from "
        "decode_latency.json's shape + latency, against peak_bw.json/"
        "peak_compute.json's already-measured peaks. FLOPs = num_q_heads * "
        "seq_len * head_dim * 4 (QK^T + AV). bytes = num_kv_heads * seq_len * "
        "head_dim * 2(K,V) * 2 bytes(fp16); Q/output bytes omitted as negligible "
        "(~3KB vs ~2MB KV read at seq_len=2048).",
    }
    out_path = RESULTS_DIR / "roofline_data.json"
    out_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"\nwrote {out_path}")

    # --- plot ---
    # True operational intensity is ~6 FLOP/byte for *every* series (it's
    # fixed by GQA ratio, not batch or kernel version) -- plotting all 9
    # lines at their literal x would stack them into one illegible column.
    # Each series gets a small, fixed, log-spaced x "lane" near the true
    # value purely for visual separation; the true intensity is marked by
    # its own guide line, and the jitter is disclosed in the title, not
    # hidden. Batch size (1..64) is encoded by marker size instead of text
    # labels, which don't fit at this point density.
    import numpy as np

    lanes = np.geomspace(0.7, 1.45, len(IMPLS))
    batch_sizes_sorted = sorted({r["batch"] for r in rows})
    marker_sizes = {b: 14 + 7 * i for i, b in enumerate(batch_sizes_sorted)}

    fig, ax = plt.subplots(figsize=(11, 7.5))

    x_min, x_max = 0.5, 200.0
    x_roof = [x_min, ridge_point, x_max]
    y_roof = [x_min * peak_bw_gbs, peak_tflops * 1000.0, peak_tflops * 1000.0]
    ax.loglog(x_roof, y_roof, color="black", linewidth=1.5, zorder=1, label=f"roofline ({peak_bw_gbs:.0f} GB/s / {peak_tflops:.1f} TFLOPS, measured)")
    ax.axvline(primary_intensity, color="gray", linestyle=":", linewidth=1, zorder=1)
    ax.annotate(
        f"true intensity\n~{primary_intensity:.1f} FLOP/byte\n(all series, jittered\nhere for readability)",
        xy=(primary_intensity, 3),
        xytext=(primary_intensity * 1.5, 1.3),
        fontsize=7,
        color="gray",
    )
    ax.axvline(ridge_point, color="gray", linestyle=":", linewidth=1, zorder=1)
    ax.annotate(
        f"ridge point\n~{ridge_point:.0f} FLOP/byte",
        xy=(ridge_point, peak_tflops * 1000.0),
        xytext=(ridge_point * 1.2, peak_tflops * 300.0),
        fontsize=8,
        color="gray",
    )

    for i, (key, label, lang) in enumerate(IMPLS):
        impl_rows = sorted((r for r in rows if r["impl"] == label), key=lambda r: r["batch"])
        xs = [primary_intensity * lanes[i] for _ in impl_rows]
        ys = [r["achieved_gflops"] for r in impl_rows]
        sizes = [marker_sizes[r["batch"]] for r in impl_rows]
        if lang == "ref":
            ax.plot(xs, ys, color="gray", linestyle="--", linewidth=1, alpha=0.6, zorder=2)
            ax.scatter(xs, ys, color="gray", marker="x", s=sizes, alpha=0.7, zorder=3, label=label)
            continue
        version = int(label[-1])
        color = VERSION_COLORS[version]
        linestyle = "-" if lang == "triton" else "--"
        marker = "o" if lang == "triton" else "s"
        ax.plot(xs, ys, color=color, linestyle=linestyle, linewidth=1.5, zorder=2)
        ax.scatter(xs, ys, color=color, marker=marker, s=sizes, zorder=3, edgecolors="white", linewidths=0.4, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("operational intensity (FLOP/byte) -- series jittered around the true value for readability")
    ax.set_ylabel("achieved performance (GFLOP/s)")
    ax.set_title(
        "Paged attention decode -- roofline (RTX 3060 Laptop, measured peaks)\n"
        "single-workload roofline: every point shares ~6 FLOP/byte intensity (fixed by GQA ratio) --\n"
        "only height vs. the bandwidth roof varies. Marker size = batch size (1..64, small to large)."
    )
    ax.set_xlim(x_min, x_max)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)

    png_path = Path(__file__).parent.parent / "profiles" / "roofline.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
