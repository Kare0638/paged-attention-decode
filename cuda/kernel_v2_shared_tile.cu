#include "kernels.h"

#include <cuda_fp16.h>
#include <math.h>

/* v2: batches the score reduction across all `gqa_ratio` query rows into
 * one shared-memory tile per token, instead of v1's one independent
 * block-wide tree reduction per (row, token) pair.
 *
 * CUDA v1's NCU profile (profiles/notes.md, "CUDA v1 — naive baseline")
 * found the bottleneck is instruction/synchronization volume, not memory
 * bandwidth: `wait`, `short_scoreboard`, and `barrier` dominate the stall
 * breakdown, from issuing a full 9-`__syncthreads()` tree reduction
 * `gqa_ratio * seq_len` times per block (once per row per token). K/V
 * itself is already loaded once per token into a register and reused
 * across rows (see kernel_v1_naive.cu) — there was never a memory-traffic
 * redundancy to fix here, only a synchronization one.
 *
 * This kernel keeps v1's tree-reduction algorithm unchanged (warp-shuffle
 * is a separate, later roadmap item — kept out of scope here, one lever
 * per version, same discipline as the Triton line) but widens the shared
 * scratch buffer from `sdata[head_dim]` to `sdata[gqa_ratio][row_stride]`
 * and folds all `gqa_ratio` rows' partial sums into every reduction
 * stage before that stage's `__syncthreads()`. That cuts total
 * `__syncthreads()` calls from `gqa_ratio * seq_len * 9` to
 * `seq_len * 9` — an exact `gqa_ratio`-fold reduction in the exact
 * overhead NCU identified, without touching the reduction algorithm
 * itself.
 *
 * `PAD` (template bool) toggles `row_stride` between `head_dim` and
 * `head_dim + 1`, reusing Week 0's exact `shared_stride_kernel<PAD>`
 * technique to test, rather than assume, whether this tile's layout
 * needs bank-conflict padding. Prediction, stated before measuring: CUDA
 * v1 already measured 0 conflicts because `threadIdx.x` is always the
 * head_dim lane (the fastest-varying per-thread index) and every other
 * axis (row, token) is looped serially within a thread, never spread
 * across a warp's lanes on one instruction — this tile keeps that same
 * structure, row-major and head_dim-contiguous, so padding is not
 * expected to change anything. Both variants are compiled and bound
 * (`forward_v2` = padded, the production path; `forward_v2_unpadded` for
 * the NCU A/B only) so that prediction is actually checked, not skipped.
 */

namespace {

constexpr int kMaxGqaRatio = 16;  // same bound as kernel_v1_naive.cu

template <bool PAD>
__global__ void paged_attn_decode_cuda_v2_kernel(
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
  const int row_stride = PAD ? head_dim + 1 : head_dim;

  // [gqa_ratio][row_stride] tile: all rows' partial products for the
  // current token, reduced together each stage instead of one row at a
  // time.
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

    for (int row = 0; row < gqa_ratio; row++)
      sdata[row * row_stride + tid] = q_reg[row] * k_val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s) {
        for (int row = 0; row < gqa_ratio; row++)
          sdata[row * row_stride + tid] += sdata[row * row_stride + tid + s];
      }
      __syncthreads();
    }

    float score[kMaxGqaRatio];
    for (int row = 0; row < gqa_ratio; row++)
      score[row] = sdata[row * row_stride] * scale;
    __syncthreads();  // all lanes must read sdata before it's reused next token

    for (int row = 0; row < gqa_ratio; row++) {
      const float m_new = fmaxf(m_i[row], score[row]);
      const float alpha = __expf(m_i[row] - m_new);
      const float p = __expf(score[row] - m_new);

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

torch::Tensor launch(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    double scale,
    int64_t gqa_ratio,
    int64_t head_dim,
    int64_t page_size,
    bool pad) {
  TORCH_CHECK(gqa_ratio <= kMaxGqaRatio,
              "gqa_ratio (", gqa_ratio, ") exceeds kMaxGqaRatio=", kMaxGqaRatio,
              " — fixed-size register array bound in kernel_v2_shared_tile.cu");

  const int batch = q.size(0);
  const int num_kv_heads = k_cache.size(2);

  auto out = torch::empty_like(q);

  const dim3 grid(batch, num_kv_heads);
  const dim3 block(head_dim);
  const int row_stride = pad ? head_dim + 1 : head_dim;
  const size_t shmem_bytes = (size_t)gqa_ratio * row_stride * sizeof(float);

  const half* q_ptr = reinterpret_cast<const half*>(q.data_ptr<at::Half>());
  const half* k_ptr = reinterpret_cast<const half*>(k_cache.data_ptr<at::Half>());
  const half* v_ptr = reinterpret_cast<const half*>(v_cache.data_ptr<at::Half>());
  const int32_t* bt_ptr = block_table.data_ptr<int32_t>();
  const int32_t* sl_ptr = seq_lens.data_ptr<int32_t>();
  half* out_ptr = reinterpret_cast<half*>(out.data_ptr<at::Half>());
  const float scale_f = static_cast<float>(scale);
  const int gqa_ratio_i = static_cast<int>(gqa_ratio);
  const int head_dim_i = static_cast<int>(head_dim);
  const int page_size_i = static_cast<int>(page_size);

  if (pad) {
    paged_attn_decode_cuda_v2_kernel<true><<<grid, block, shmem_bytes>>>(
        q_ptr, k_ptr, v_ptr, bt_ptr, sl_ptr, out_ptr, scale_f,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_table.stride(0), block_table.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        gqa_ratio_i, head_dim_i, page_size_i);
  } else {
    paged_attn_decode_cuda_v2_kernel<false><<<grid, block, shmem_bytes>>>(
        q_ptr, k_ptr, v_ptr, bt_ptr, sl_ptr, out_ptr, scale_f,
        q.stride(0), q.stride(1), q.stride(2),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
        block_table.stride(0), block_table.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        gqa_ratio_i, head_dim_i, page_size_i);
  }

  return out;
}

}  // namespace

torch::Tensor paged_attn_decode_cuda_v2_launch(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    double scale,
    int64_t gqa_ratio,
    int64_t head_dim,
    int64_t page_size) {
  return launch(q, k_cache, v_cache, block_table, seq_lens, scale, gqa_ratio, head_dim,
                page_size, /*pad=*/true);
}

torch::Tensor paged_attn_decode_cuda_v2_unpadded_launch(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    double scale,
    int64_t gqa_ratio,
    int64_t head_dim,
    int64_t page_size) {
  return launch(q, k_cache, v_cache, block_table, seq_lens, scale, gqa_ratio, head_dim,
                page_size, /*pad=*/false);
}
