from __future__ import annotations

import os

from torch.utils.cpp_extension import load

_CUDA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cuda")

# All CUDA kernel .cu files share one binding.cpp / one PYBIND11_MODULE (see
# cuda/kernels.h), so they must all be compiled into the same JIT extension —
# loading a subset under the same extension `name` would either fail to link
# (binding.cpp references every kernel's launcher) or silently reuse a stale
# cached build missing whichever kernel wasn't included. Every wrapper module
# (kernel_cuda_v1.py, kernel_cuda_v2.py, ...) must go through this single
# loader, not build its own sources list.
_SOURCES = [
    "kernel_v1_naive.cu",
    "kernel_v2_shared_tile.cu",
    "binding.cpp",
]

_ext = None


def get_ext():
    global _ext
    if _ext is None:
        _ext = load(
            name="paged_attn_decode_cuda",
            sources=[os.path.join(_CUDA_DIR, src) for src in _SOURCES],
            extra_cuda_cflags=["-O3"],
        )
    return _ext
