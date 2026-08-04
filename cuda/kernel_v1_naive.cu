#include "kernels.h"

#include <cuda_fp16.h>
#include <math.h>

/* v1: naive, unoptimized port of src/kernel_v1_naive.py's algorithm to raw
 * CUDA C++ — the CUDA-side equivalent of that file's role for Triton: a
 * correctness baseline, not yet tuned. It is deliberately MORE naive than
 * Triton v1, not a like-for-like port:
 *
 *  - Triton v1 computes a whole [GQA_RATIO_PADDED, BLOCK_N] score tile per
 *    loop iteration via `tl.dot`. This kernel loops token-by-token — no
 *    tiled matmul, one dot product (a full block-wide reduction) per
 *    (row, token) pair.
 *  - Every one of the `gqa_ratio` query rows independently re-reads the
 *    same K/V values from global memory (no sharing across rows). v2's
 *    roadmap item — cooperative shared-memory K/V tiling — is specifically
 *    what removes this redundancy; it does not exist yet here on purpose.
 *
 * Grid is `(batch, num_kv_heads)`, identical to Triton v1's, so the
 * occupancy story (2 thread blocks at batch=1 on this ~28-SM GPU) is
 * directly comparable to the existing profiles/notes.md v1 numbers rather
 * than confounded by a different parallelization strategy.
 *
 * Unlike Triton, CUDA has no `tl.arange` power-of-2 constraint and no
 * `tl.dot` K>=16 floor, so none of Triton v1's masking/padding machinery
 * (GQA_RATIO_PADDED + gqa_mask, BLOCK_N tiling with seq_mask) is needed —
 * this loops the real runtime `gqa_ratio` and `seq_len` directly. That is
 * a genuine CUDA-vs-Triton contrast, not an oversight: see
 * src/kernel_cuda_v1.py's docstring for which of Triton v1's wrapper
 * assertions consequently do not apply here.
 */

namespace {

// Fixed register-array bound for the per-row accumulators. Runtime
// `gqa_ratio` (1/4/6/8 across this project's test matrix) must not exceed
// this — checked host-side in the launcher below, a plain int comparison
// with no device-to-host sync cost (unlike the seq_len>=1 check documented
// in kernel_v1_naive.py, which was expensive specifically because it read
// a *tensor value*).
constexpr int kMaxGqaRatio = 16;

__global__ void paged_attn_decode_cuda_v1_kernel(
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
  const int tid = threadIdx.x;  // one thread per head_dim lane

  const int seq_len = seq_lens[b];

  // Reduction scratch: one float per head_dim lane, reused across every
  // (row, token) pair — a naive shared-memory tree reduction, not yet the
  // warp-shuffle reduction that's a separate, later roadmap item.
  extern __shared__ float sdata[];

  float q_reg[kMaxGqaRatio];
  for (int row = 0; row < gqa_ratio; row++) {
    const int q_head = kvh * gqa_ratio + row;
    q_reg[row] = __half2float(
        q[b * stride_qb + (int64_t)q_head * stride_qh + (int64_t)tid * stride_qd]);
  }

  float m_i[kMaxGqaRatio];
  float l_i[kMaxGqaRatio];
  float acc[kMaxGqaRatio];
  for (int row = 0; row < gqa_ratio; row++) {
    m_i[row] = -INFINITY;
    l_i[row] = 0.0f;
    acc[row] = 0.0f;
  }

  for (int n = 0; n < seq_len; n++) {
    const int page_id = n / page_size;
    const int slot = n % page_size;
    const int phys_page = block_table[b * stride_btb + (int64_t)page_id * stride_btp];

    const float k_val = __half2float(k_cache[
        (int64_t)phys_page * stride_kp + (int64_t)slot * stride_ks +
        (int64_t)kvh * stride_kh + (int64_t)tid * stride_kd]);
    const float v_val = __half2float(v_cache[
        (int64_t)phys_page * stride_vp + (int64_t)slot * stride_vs +
        (int64_t)kvh * stride_vh + (int64_t)tid * stride_vd]);

    for (int row = 0; row < gqa_ratio; row++) {
      sdata[tid] = q_reg[row] * k_val;
      __syncthreads();
      for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
      }
      const float score = sdata[0] * scale;
      __syncthreads();  // all lanes must read sdata[0] before it's reused next row

      const float m_new = fmaxf(m_i[row], score);
      const float alpha = __expf(m_i[row] - m_new);
      const float p = __expf(score - m_new);

      acc[row] = acc[row] * alpha + p * v_val;
      l_i[row] = l_i[row] * alpha + p;
      m_i[row] = m_new;
    }
  }

  for (int row = 0; row < gqa_ratio; row++) {
    const int q_head = kvh * gqa_ratio + row;
    const float o = acc[row] / l_i[row];
    out[b * stride_ob + (int64_t)q_head * stride_oh + (int64_t)tid * stride_od] =
        __float2half(o);
  }
}

}  // namespace

torch::Tensor paged_attn_decode_cuda_v1_launch(
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
              " — fixed-size register array bound in kernel_v1_naive.cu");

  const int batch = q.size(0);
  const int num_kv_heads = k_cache.size(2);

  auto out = torch::empty_like(q);

  const dim3 grid(batch, num_kv_heads);
  const dim3 block(head_dim);
  const size_t shmem_bytes = head_dim * sizeof(float);

  paged_attn_decode_cuda_v1_kernel<<<grid, block, shmem_bytes>>>(
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
