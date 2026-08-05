#pragma once

#include <torch/extension.h>

// Declared here, defined in the matching kernel_v{N}_*.cu file, so
// binding.cpp can bind every version without depending on any one
// kernel file's internals.
torch::Tensor paged_attn_decode_cuda_v1_launch(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    double scale,
    int64_t gqa_ratio,
    int64_t head_dim,
    int64_t page_size);

// v2: padded (production) and unpadded (bank-conflict A/B only) variants
// of the batched-shared-memory-reduction kernel. Defined in
// kernel_v2_shared_tile.cu.
torch::Tensor paged_attn_decode_cuda_v2_launch(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    double scale,
    int64_t gqa_ratio,
    int64_t head_dim,
    int64_t page_size);

torch::Tensor paged_attn_decode_cuda_v2_unpadded_launch(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    double scale,
    int64_t gqa_ratio,
    int64_t head_dim,
    int64_t page_size);

// v3: warp-shuffle reduction, one warp per block. Defined in
// kernel_v3_warp_shuffle.cu.
torch::Tensor paged_attn_decode_cuda_v3_launch(
    torch::Tensor q,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor block_table,
    torch::Tensor seq_lens,
    double scale,
    int64_t gqa_ratio,
    int64_t head_dim,
    int64_t page_size);

// v4: split-K along the sequence dimension, built on v3's warp-shuffle
// reduction. Single launcher runs phase 1 then phase 2 and returns the
// final tensor. Defined in kernel_v4_split_k.cu.
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
    int64_t num_splits);
