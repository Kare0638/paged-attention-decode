#include "kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_v1", &paged_attn_decode_cuda_v1_launch,
        "Paged attention decode, CUDA v1 (naive baseline)");
  m.def("forward_v2", &paged_attn_decode_cuda_v2_launch,
        "Paged attention decode, CUDA v2 (batched shared-memory reduction, padded)");
  m.def("forward_v2_unpadded", &paged_attn_decode_cuda_v2_unpadded_launch,
        "Paged attention decode, CUDA v2 (batched shared-memory reduction, unpadded — "
        "bank-conflict A/B only, not the production path)");
}
