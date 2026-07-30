# Paged Attention Decode Kernel

Decode-phase attention kernel for LLM inference with paged KV cache and GQA support — being implemented from scratch in **Triton** and **CUDA C++**, optimized against Nsight Compute profiling data on a low-SM consumer GPU.

## Status

🚧 Early development. Environment and tooling verification in progress; kernel implementations have not landed yet. This README gets updated with real benchmark numbers as each version ships — no numbers are reported until they're measured.

## Why this project

Decode-phase attention is memory-bandwidth-bound, not compute-bound: for a Qwen2.5-1.5B-style config (GQA ratio 6, head_dim 128, fp16 KV cache), the arithmetic intensity works out to ~6 FLOP/byte, far below this RTX 3060 Laptop's **measured** ridge point of ~84 FLOP/byte (26.4 TFLOPS fp16 matmul / 314 GB/s D2D — both measured on this machine, not nominal spec-sheet numbers; see [bench/results/](bench/results/)) — so almost every optimization here is about memory access patterns and occupancy, not raw FLOPs.

On low-SM-count GPUs, naive decode kernels that grid over `(batch, kv_head)` leave most SMs idle when batch size and KV head count are both small under GQA (e.g. 2 KV heads on a ~28-SM GPU). This project implements split-K (FlashDecoding-style) parallelism along the sequence dimension to address that, with every optimization step backed by Nsight Compute data rather than guesswork.

## Design

**Target model config** (Qwen2.5-1.5B, verified against `config.json`):

| | |
|---|---|
| Q heads / KV heads | 12 / 2 (GQA ratio 6) |
| head_dim | 128 |
| dtype | fp16 compute, fp32 accumulate |
| page_size | 16 (primary), 32 (comparison) |

**Paged KV cache layout**

```
kv_cache:    [num_pages, page_size, num_kv_heads, head_dim]
block_table: [batch, max_pages_per_seq]   # logical page -> physical page
seq_lens:    [batch]
```

**Split-K reduction** — the sequence dimension is split into chunks; phase 1 computes per-chunk partial `(O, m, l)` online-softmax statistics, phase 2 reduces across chunks. Derivation lives in [analysis/](analysis/).

## Roadmap

- [x] Correctness: fp32 PyTorch reference implementation
- [ ] Triton v1: naive, paged KV indexing
- [ ] Triton v2: coalesced KV access
- [ ] Triton v3: single-pass online softmax
- [ ] Triton v4: split-K along the sequence dimension
- [ ] CUDA C++: explicit shared-memory tiling, bank-conflict elimination
- [ ] CUDA C++: warp-shuffle reduction, occupancy analysis
- [ ] CUDA C++: split-K, packaged as a PyTorch extension
- [x] Correctness test suite for the reference implementation (page boundaries, extreme/ragged/zero-length sequences, non-contiguous block tables with unreferenced "holes", GQA ratios 1/4/6/8, full input-validation coverage, 2000-case randomized fuzz vs. an independently-written SDPA oracle — 81 cases at default settings, `tests/`). Kernel-vs-reference and cross-implementation tests land once Triton/CUDA versions exist.
- [ ] Nsight-driven optimization log with before/after profiles
- [ ] Benchmark against FlashInfer (A100)
- [ ] Roofline plot using measured (not nominal) peak bandwidth

## Repo layout

```
src/            Triton kernels + PyTorch reference
cuda/           CUDA C++ kernels + PyTorch extension binding
tests/          Correctness tests (reference-matching, edge cases, fuzz)
bench/          Benchmark scripts and raw results
analysis/       Roofline / sweep plots
profiles/       Nsight Compute captures + notes
```

## Hardware

RTX 3060 Laptop (6GB) on WSL2 for development; cross-hardware validation planned on A100.

## Reproduce

Kernel benchmarks are not yet available — kernels aren't implemented. The fp32 PyTorch reference correctness suite can be reproduced with:

```bash
uv venv --python 3.12
uv pip install -r requirements.txt
uv run pytest tests/ -q
# full 2000-case fuzz sweep (default is a fast 50-case subset):
PAGED_ATTN_FUZZ_ITERS=2000 uv run pytest tests/test_reference_fuzz.py -q
```

Once `src/kernel_v1_naive.py` lands, this section will include the one-line kernel benchmark command.

## License

Apache License 2.0 — see [LICENSE](LICENSE). This project draws on public techniques from FlashDecoding, FlashInfer, and vLLM (split-K parallelism, the online-softmax recurrence); specific attributions are noted wherever a structure is adapted rather than original.
