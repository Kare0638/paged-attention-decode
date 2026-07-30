# Nsight Compute profiling notes

## Week 0 — NCU + WSL2 workflow validation (2026-07-30)

Verified that RTX 3060 Laptop / WSL2 / NCU 2024.1.1.0 can reliably produce
differentiated, exportable profiling evidence — not just a single metric
read that happens to come back non-zero.

**Method.** A throwaway CUDA kernel, `shared_stride_kernel<PAD>`, that reads
shared memory with a fixed-column, varying-row access pattern:
`tile[32][32]` (`PAD=0`) hits the textbook 32-way bank conflict (all 32
threads in a row-varying, column-fixed read hash to the same bank);
`tile[32][33]` (`PAD=1`) removes it by shifting the per-row stride off the
32-bank boundary. This is deliberately different from the earlier
hello-world extension (`add_kernel`), which only touched global memory and
so could never have produced a non-zero bank-conflict count either way —
it wasn't actually capable of testing whether the counter works.

**Results** (two independent runs, `ncu --metrics ... --print-summary
per-kernel`):

| metric | unpadded (`PAD=0`) | padded (`PAD=1`) |
|---|---|---|
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` (run 1) | 63,488,467 | 272 |
| same metric (run 2) | 63,488,473 | 247 |
| `dram__bytes.sum` | 3.20 KB | 570.37 KB |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | 31.76% | 32.25% |
| `launch__occupancy_limit_shared_mem` | 20 blocks | 19 blocks |

~233,000x reduction in bank conflicts, stable run-to-run (<0.01% variance
on the dominant unpadded count). `dram__throughput`, and all four
`launch__occupancy_limit_*` metrics populated with sane, non-zero values.
No `ERR_NVGPUCTRPERM` or other permission errors on either run.

**`--set full` also verified.** 38 replay passes, no errors, produced a
real 907 KB `.ncu-rep` file. Re-imported cleanly via `ncu --import <file>
--page raw --csv`. One correction to the original plan: the stall-reason
metrics live under `smsp__pcsamp_warps_issue_stalled_<reason>` (the
PC-sampling family), not `smsp__warp_issue_stalled_*_per_warp_active` as
originally assumed — use the `pcsamp` family name when querying directly
via `--metrics`.

**Conclusion:** Week 0 Check 1 passed. The WSL2 + NCU profiling workflow
is stable and trustworthy for the Week 3–5 optimization loop. No dual-boot
fallback needed.
