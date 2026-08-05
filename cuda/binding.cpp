#include "kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_v1", &paged_attn_decode_cuda_v1_launch,
        "Paged attention decode, CUDA v1 (naive baseline)");
  m.def("forward_v2", &paged_attn_decode_cuda_v2_launch,
        "Paged attention decode, CUDA v2 (batched shared-memory reduction, padded)");
  m.def("forward_v2_unpadded", &paged_attn_decode_cuda_v2_unpadded_launch,
        "Paged attention decode, CUDA v2 (batched shared-memory reduction, unpadded — "
        "bank-conflict A/B only, not the production path)");
  m.def("forward_v3", &paged_attn_decode_cuda_v3_launch,
        "Paged attention decode, CUDA v3 (warp-shuffle reduction, one warp per block)");
  m.def("forward_v4", &paged_attn_decode_cuda_v4_launch,
        "Paged attention decode, CUDA v4 (split-K along the sequence dimension)");
}
