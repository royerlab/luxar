/**
 * NVRTC (runtime compilation) compatibility shim.
 *
 * Included ONLY when the kernels are compiled at runtime by NVRTC
 * (`__CUDACC_RTC__` is defined), in place of the host toolkit headers the
 * nvcc build pulls in (<cuda_runtime.h>, <cuda_fp16.h>, <cmath>, <cstdint>,
 * <cassert>). NVRTC ships no C++ standard library and no <cuda_fp16.h>, and
 * a plain `pip install` has no CUDA toolkit to borrow them from, so the
 * runtime-compiled path supplies the handful of definitions the kernels use:
 *
 * - fixed-width integer typedefs (int64_t, ...);
 * - a storage-only `__half` / `__half2` with the exact FP16 -> FP32
 *   conversions and the read-only `__ldg` load the kernels need. FP16 is a
 *   storage format here (all arithmetic happens in FP32), and FP16 -> FP32
 *   conversion is exact, so this is bit-identical to <cuda_fp16.h>.
 *
 * math builtins (sqrtf, fmaxf, __expf, ...), `assert`, and the CUDA built-in
 * variables are provided by NVRTC itself.
 *
 * The nvcc build never sees this file. See nvrtc_backend.py.
 */

#ifndef CUDA_SPLATTING_RTC_COMPAT_CUH
#define CUDA_SPLATTING_RTC_COMPAT_CUH

#ifndef __CUDACC_RTC__
#error "rtc_compat.cuh is only for NVRTC runtime compilation"
#endif

typedef signed char int8_t;
typedef short int16_t;
typedef int int32_t;
typedef long long int64_t;
typedef unsigned char uint8_t;
typedef unsigned short uint16_t;
typedef unsigned int uint32_t;
typedef unsigned long long uint64_t;

// Storage-only IEEE binary16, layout-compatible with torch.float16.
struct __align__(2) __half {
    unsigned short __x;
};

struct __align__(4) __half2 {
    unsigned short __lo;
    unsigned short __hi;
};

__device__ __forceinline__ float __luxar_half_bits_to_float(unsigned short bits) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(bits));
    return f;
}

__device__ __forceinline__ float __half2float(const __half h) {
    return __luxar_half_bits_to_float(h.__x);
}

__device__ __forceinline__ float __low2float(const __half2 h) {
    return __luxar_half_bits_to_float(h.__lo);
}

__device__ __forceinline__ float __high2float(const __half2 h) {
    return __luxar_half_bits_to_float(h.__hi);
}

__device__ __forceinline__ __half __ldg(const __half* ptr) {
    __half h;
    h.__x = __ldg(reinterpret_cast<const unsigned short*>(ptr));
    return h;
}

#endif // CUDA_SPLATTING_RTC_COMPAT_CUH
