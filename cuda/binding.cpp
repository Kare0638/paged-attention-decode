#include "kernels.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_v1", &paged_attn_decode_cuda_v1_launch,
        "Paged attention decode, CUDA v1 (naive baseline)");
}
