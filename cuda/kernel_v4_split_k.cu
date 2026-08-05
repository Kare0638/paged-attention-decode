#include "kernels.h"

#include <cuda_fp16.h>
#include <math.h>

/* v4: split-K along the sequence dimension — the last CUDA roadmap item.
 * CUDA v1-v3 all grid over (batch, num_kv_heads); at batch=1 that's still
 * just 2 thread blocks on this ~28-SM GPU, unchanged by any of v1->v3's
 * per-block tuning. v4 adds a third grid dimension over chunks of the
 * sequence, the same fix already applied once in the Triton line
 * (src/kernel_v4_split_k.py) — see analysis/split_k_derivation.md for the
 * merge math, reused here directly (language-agnostic, no new derivation
 * needed).
 *
 * Unlike Triton v4 (which reused Triton v1's kernel body unchanged inside
 * phase 1 — split-K there was a pure grid-level change), this reuses
 * CUDA v3's warp-shuffle reduction, not v1's tree reduction: v3 is the
 * best per-block building block available now, not the first one. One
 * consequence: v3 already cut per-block instruction count ~4.2x vs. v1,
 * so split-K's fixed costs (a second kernel launch, intermediate-buffer
 * traffic) have less headroom to hide behind than they did against v1 in
 * the Triton line — whether this still wins at batch=1, and by how much,
 * is measured in profiles/notes.md, not assumed here.
 *
 * Phase 1: same warp-shuffle per-token loop as v3, bounded to
 * [split_start, split_end) instead of [0, seq_len), storing the
 * *unnormalized* (O, m, l) instead of the final acc/l_i division. Empty
 * chunks (split_start >= seq_len) do zero loop iterations, leaving the
 * -inf/0/0 defaults — no special-case branch needed.
 *
 * Phase 2: plain sequential fold over num_splits partials (cheap,
 * O(num_splits) per program, not O(seq_len) — the same "phase 2 is cheap
 * in absolute terms" finding already documented for Triton v4). Only
 * beta_s (the incoming split's rescale) needs the explicit -inf guard;
 * alpha (the running accumulator's rescale) doesn't, because split 0 is
 * always non-empty given this project's seq_len >= 1 precondition, so
 * m_run is only ever -inf on the very first fold, where m_s is
 * guaranteed finite and exp(-inf - finite) = 0, not NaN — ported exactly
 * from analysis/split_k_derivation.md's reasoning, not re-derived.
 */

namespace {

constexpr int kMaxGqaRatio = 16;       // same bound as v1/v2/v3
constexpr int kMaxElemsPerThread = 4;  // covers head_dim up to 128 at warp size 32
constexpr unsigned kFullMask = 0xffffffffu;

__global__ void paged_attn_decode_cuda_v4_phase1_kernel(
    const half* __restrict__ q,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    float* __restrict__ o_partial,
    float* __restrict__ m_partial,
    float* __restrict__ l_partial,
    float scale,
    int64_t stride_qb, int64_t stride_qh, int64_t stride_qd,
    int64_t stride_kp, int64_t stride_ks, int64_t stride_kh, int64_t stride_kd,
    int64_t stride_vp, int64_t stride_vs, int64_t stride_vh, int64_t stride_vd,
    int64_t stride_btb, int64_t stride_btp,
    int64_t stride_opb, int64_t stride_opk, int64_t stride_ops, int64_t stride_opr, int64_t stride_opd,
    int64_t stride_mpb, int64_t stride_mpk, int64_t stride_mps, int64_t stride_mpr,
    int gqa_ratio, int head_dim, int page_size, int num_splits) {
  const int b = blockIdx.x;
  const int kvh = blockIdx.y;
  const int split_idx = blockIdx.z;
  const int tid = threadIdx.x;

  const int seq_len = seq_lens[b];
  const int elems_per_thread = head_dim / 32;

  const int chunk_size = (seq_len + num_splits - 1) / num_splits;
  const int split_start = split_idx * chunk_size;
  const int split_end = min(split_start + chunk_size, seq_len);

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

  for (int n = split_start; n < split_end; n++) {
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
    const int64_t mp_off = b * stride_mpb + (int64_t)kvh * stride_mpk +
                            (int64_t)split_idx * stride_mps + (int64_t)row * stride_mpr;
    if (tid == 0) {
      m_partial[mp_off] = m_i[row];
      l_partial[mp_off] = l_i[row];
    }
    for (int e = 0; e < elems_per_thread; e++) {
      const int d = tid + e * 32;
      const int64_t op_off = b * stride_opb + (int64_t)kvh * stride_opk +
                              (int64_t)split_idx * stride_ops + (int64_t)row * stride_opr +
                              (int64_t)d * stride_opd;
      o_partial[op_off] = acc[row][e];
    }
  }
}

__global__ void paged_attn_decode_cuda_v4_phase2_kernel(
    const float* __restrict__ o_partial,
    const float* __restrict__ m_partial,
    const float* __restrict__ l_partial,
    half* __restrict__ out,
    int64_t stride_opb, int64_t stride_opk, int64_t stride_ops, int64_t stride_opr, int64_t stride_opd,
    int64_t stride_mpb, int64_t stride_mpk, int64_t stride_mps, int64_t stride_mpr,
    int64_t stride_ob, int64_t stride_oh, int64_t stride_od,
    int gqa_ratio, int head_dim, int num_splits) {
  const int b = blockIdx.x;
  const int kvh = blockIdx.y;
  const int tid = threadIdx.x;

  const int elems_per_thread = head_dim / 32;

  for (int row = 0; row < gqa_ratio; row++) {
    float m_run = -INFINITY;
    float l_run = 0.0f;
    float acc_run[kMaxElemsPerThread];
    for (int e = 0; e < elems_per_thread; e++) acc_run[e] = 0.0f;

    const int64_t mp_row_base = b * stride_mpb + (int64_t)kvh * stride_mpk + (int64_t)row * stride_mpr;
    const int64_t op_row_base = b * stride_opb + (int64_t)kvh * stride_opk + (int64_t)row * stride_opr;

    for (int s = 0; s < num_splits; s++) {
      const float m_s = m_partial[mp_row_base + (int64_t)s * stride_mps];
      const float l_s = l_partial[mp_row_base + (int64_t)s * stride_mps];

      const float m_new = fmaxf(m_run, m_s);
      const float alpha = __expf(m_run - m_new);
      const float beta = (m_s == -INFINITY) ? 0.0f : __expf(m_s - m_new);

      l_run = l_run * alpha + l_s * beta;
      for (int e = 0; e < elems_per_thread; e++) {
        const int d = tid + e * 32;
        const float o_s = o_partial[op_row_base + (int64_t)s * stride_ops + (int64_t)d * stride_opd];
        acc_run[e] = acc_run[e] * alpha + o_s * beta;
      }
      m_run = m_new;
    }

    const int q_head = kvh * gqa_ratio + row;
    for (int e = 0; e < elems_per_thread; e++) {
      const int d = tid + e * 32;
      const float o = acc_run[e] / l_run;
      out[b * stride_ob + (int64_t)q_head * stride_oh + (int64_t)d * stride_od] =
          __float2half(o);
    }
  }
}

}  // namespace

torch::Tensor paged_attn_decode_cuda_v4_launch(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    double scale,
    int64_t gqa_ratio,
    int64_t head_dim,
    int64_t page_size,
    int64_t num_splits) {
  TORCH_CHECK(gqa_ratio <= kMaxGqaRatio,
              "gqa_ratio (", gqa_ratio, ") exceeds kMaxGqaRatio=", kMaxGqaRatio,
              " — fixed-size register array bound in kernel_v4_split_k.cu");
  TORCH_CHECK(head_dim % 32 == 0 && head_dim / 32 <= kMaxElemsPerThread,
              "head_dim (", head_dim, ") must be a multiple of 32 and <= ",
              32 * kMaxElemsPerThread, " — one warp per block, each thread owns "
              "head_dim/32 lanes (inherited from CUDA v3's warp-shuffle reduction)");
  TORCH_CHECK(num_splits >= 1, "num_splits must be >= 1");

  const int batch = q.size(0);
  const int num_kv_heads = k_cache.size(2);

  auto float_opts = q.options().dtype(torch::kFloat32);
  auto o_partial = torch::empty({batch, num_kv_heads, num_splits, gqa_ratio, head_dim}, float_opts);
  auto m_partial = torch::empty({batch, num_kv_heads, num_splits, gqa_ratio}, float_opts);
  auto l_partial = torch::empty({batch, num_kv_heads, num_splits, gqa_ratio}, float_opts);
  auto out = torch::empty_like(q);

  const dim3 grid1(batch, num_kv_heads, num_splits);
  const dim3 grid2(batch, num_kv_heads);
  const dim3 block(32);

  paged_attn_decode_cuda_v4_phase1_kernel<<<grid1, block>>>(
      reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(k_cache.data_ptr<at::Half>()),
      reinterpret_cast<const half*>(v_cache.data_ptr<at::Half>()),
      block_table.data_ptr<int32_t>(),
      seq_lens.data_ptr<int32_t>(),
      o_partial.data_ptr<float>(),
      m_partial.data_ptr<float>(),
      l_partial.data_ptr<float>(),
      static_cast<float>(scale),
      q.stride(0), q.stride(1), q.stride(2),
      k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), k_cache.stride(3),
      v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), v_cache.stride(3),
      block_table.stride(0), block_table.stride(1),
      o_partial.stride(0), o_partial.stride(1), o_partial.stride(2), o_partial.stride(3), o_partial.stride(4),
      m_partial.stride(0), m_partial.stride(1), m_partial.stride(2), m_partial.stride(3),
      static_cast<int>(gqa_ratio), static_cast<int>(head_dim), static_cast<int>(page_size),
      static_cast<int>(num_splits));

  paged_attn_decode_cuda_v4_phase2_kernel<<<grid2, block>>>(
      o_partial.data_ptr<float>(),
      m_partial.data_ptr<float>(),
      l_partial.data_ptr<float>(),
      reinterpret_cast<half*>(out.data_ptr<at::Half>()),
      o_partial.stride(0), o_partial.stride(1), o_partial.stride(2), o_partial.stride(3), o_partial.stride(4),
      m_partial.stride(0), m_partial.stride(1), m_partial.stride(2), m_partial.stride(3),
      out.stride(0), out.stride(1), out.stride(2),
      static_cast<int>(gqa_ratio), static_cast<int>(head_dim), static_cast<int>(num_splits));

  return out;
}
