#include "kernels.h"

#include <cuda_fp16.h>
#include <math.h>

/* v3: replaces v1/v2's tree-reduction-plus-`__syncthreads()` algorithm
 * with warp-shuffle (`__shfl_down_sync`/`__shfl_sync`) — a genuinely
 * different reduction algorithm, not another grouping of the same one.
 *
 * CUDA v2's investigation (profiles/notes.md, "CUDA v2 — batched
 * shared-memory reduction") found this kernel is bound by the *length*
 * of its serialized dependency chain (9 sequentially-dependent
 * `__syncthreads()`-bound stages per token: 1 store-sync + 7
 * tree-reduction stages + 1 broadcast-read-sync), not by instruction or
 * sync *count* — v2 cut sync count 6x by batching all `gqa_ratio` rows
 * into each stage and measured no latency change. Warp-shuffle shortens
 * the chain itself: 5 reduction steps + 1 broadcast = 6 dependent
 * instructions per token per row, entirely in registers, no shared
 * memory, no explicit block-wide barrier at all (a warp executes
 * `__shfl_*_sync` in lockstep by construction, given a uniform
 * participation mask and no divergent branch in this reduction).
 *
 * This forces a real design change: warp-shuffle only reduces within one
 * 32-thread warp, so this kernel uses `blockDim(32)` (one warp per
 * block) instead of v1/v2's `blockDim(head_dim)`. Each thread now owns
 * `elems_per_thread = head_dim / 32` head_dim lanes instead of exactly
 * one — `head_dim % 32 == 0` is consequently a genuine new precondition
 * of this specific one-warp-per-block design (not a general CUDA
 * constraint), checked in the wrapper. Both head_dim values this
 * project's CUDA test suite exercises (32, 128 — see
 * tests/case_generators.py's overrides in test_kernel_cuda_v1.py /
 * test_kernel_cuda_v2.py) already satisfy it.
 *
 * Splits into two separately-measured hypotheses rather than one
 * conflated claim (see profiles/notes.md for the results):
 *  1. Shorter critical path -> lower latency at every batch size,
 *     including batch=1 (a per-block effect, independent of grid size).
 *  2. Smaller blocks (32 threads vs. 128) -> more blocks resident per SM
 *     -> occupancy change — but only where grid size isn't already the
 *     limiter. At batch=1 the grid is still just (1, 2) = 2 blocks total
 *     (shrinking block size doesn't add grid blocks), so v3 has *fewer*
 *     total resident warps than v1/v2 there (2 vs. 8) despite being a
 *     latency win — occupancy-ceiling effects are expected at batch=64
 *     instead, not batch=1.
 *
 * Row loop stays sequential (unchanged from v1/v2) — the one lever this
 * version tests is the reduction algorithm itself, kept isolated from
 * v2's already-tested row-batching lever.
 */

namespace {

constexpr int kMaxGqaRatio = 16;       // same bound as v1/v2
constexpr int kMaxElemsPerThread = 4;  // covers head_dim up to 128 at warp size 32
constexpr unsigned kFullMask = 0xffffffffu;

__global__ void paged_attn_decode_cuda_v3_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    half* __restrict__ out,
    float scale,
    int64_t stride_qb, int64_t stride_qh, int64_t stride_qd,
    int64_t stride_kp, int64_t stride_ks, int64_t stride_kh, int64_t stride_kd,
    int64_t stride_vp, int64_t stride_vs, int64_t stride_vh, int64_t stride_vd,
    int64_t stride_btb, int64_t stride_btp,
    int64_t stride_ob, int64_t stride_oh, int64_t stride_od,
    int gqa_ratio, int head_dim, int page_size) {
  const int b = blockIdx.x;
  const int kvh = blockIdx.y;
  const int tid = threadIdx.x;  // lane within the single warp, 0..31

  const int seq_len = seq_lens[b];
  const int elems_per_thread = head_dim / 32;

  float q_reg[kMaxGqaRatio][kMaxElemsPerThread];
  for (int row = 0; row < gqa_ratio; row++) {
    const int q_head = kvh * gqa_ratio + row;
    for (int e = 0; e < elems_per_thread; e++) {
      const int d = tid + e * 32;
      q_reg[row][e] = __half2float(
          q[b * stride_qb + (int64_t)q_head * stride_qh + (int64_t)d * stride_qd]);
    }
  }

  float m_i[kMaxGqaRatio];
  float l_i[kMaxGqaRatio];
  float acc[kMaxGqaRatio][kMaxElemsPerThread];
  for (int row = 0; row < gqa_ratio; row++) {
    m_i[row] = -INFINITY;
    l_i[row] = 0.0f;
    for (int e = 0; e < elems_per_thread; e++) acc[row][e] = 0.0f;
  }

  float k_val[kMaxElemsPerThread];
  float v_val[kMaxElemsPerThread];

  for (int n = 0; n < seq_len; n++) {
    const int page_id = n / page_size;
    const int slot = n % page_size;
    const int phys_page = block_table[b * stride_btb + (int64_t)page_id * stride_btp];

    for (int e = 0; e < elems_per_thread; e++) {
      const int d = tid + e * 32;
      k_val[e] = __half2float(k_cache[
          (int64_t)phys_page * stride_kp + (int64_t)slot * stride_ks +
          (int64_t)kvh * stride_kh + (int64_t)d * stride_kd]);
      v_val[e] = __half2float(v_cache[
          (int64_t)phys_page * stride_vp + (int64_t)slot * stride_vs +
          (int64_t)kvh * stride_vh + (int64_t)d * stride_vd]);
    }

    for (int row = 0; row < gqa_ratio; row++) {
      float partial = 0.0f;
      for (int e = 0; e < elems_per_thread; e++)
        partial += q_reg[row][e] * k_val[e];

#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1)
        partial += __shfl_down_sync(kFullMask, partial, offset);
      const float score = __shfl_sync(kFullMask, partial, 0) * scale;

      const float m_new = fmaxf(m_i[row], score);
      const float alpha = __expf(m_i[row] - m_new);
      const float p = __expf(score - m_new);

      for (int e = 0; e < elems_per_thread; e++)
        acc[row][e] = acc[row][e] * alpha + p * v_val[e];
      l_i[row] = l_i[row] * alpha + p;
      m_i[row] = m_new;
    }
  }

  for (int row = 0; row < gqa_ratio; row++) {
    const int q_head = kvh * gqa_ratio + row;
    for (int e = 0; e < elems_per_thread; e++) {
      const int d = tid + e * 32;
      const float o = acc[row][e] / l_i[row];
      out[b * stride_ob + (int64_t)q_head * stride_oh + (int64_t)d * stride_od] =
          __float2half(o);
    }
  }
}

}  // namespace

torch::Tensor paged_attn_decode_cuda_v3_launch(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    double scale,
    int64_t gqa_ratio,
    int64_t head_dim,
    int64_t page_size) {
  TORCH_CHECK(gqa_ratio <= kMaxGqaRatio,
              "gqa_ratio (", gqa_ratio, ") exceeds kMaxGqaRatio=", kMaxGqaRatio,
              " — fixed-size register array bound in kernel_v3_warp_shuffle.cu");
  TORCH_CHECK(head_dim % 32 == 0 && head_dim / 32 <= kMaxElemsPerThread,
              "head_dim (", head_dim, ") must be a multiple of 32 and <= ",
              32 * kMaxElemsPerThread, " — this kernel is one warp per block, "
              "each thread owns head_dim/32 lanes");

  const int batch = q.size(0);
  const int num_kv_heads = k_cache.size(2);

  auto out = torch::empty_like(q);

  const dim3 grid(batch, num_kv_heads);
  const dim3 block(32);

  paged_attn_decode_cuda_v3_kernel<<<grid, block>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v_cache.data_ptr<at::Half>()),
      block_table.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      static_cast<float>(scale),
      q.stride(0), q.stride(1), q.stride(2),
      k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
      v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
      block_table.stride(0), block_table.stride(1),
      out.stride(0), out.stride(1), out.stride(2),
      static_cast<int>(gqa_ratio), static_cast<int>(head_dim), static_cast<int>(page_size));

  return out;
}
