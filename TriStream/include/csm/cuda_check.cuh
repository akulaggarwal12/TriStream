#pragma once

#include <cuda_runtime.h>

#include "csm/common.hpp"

#define CUDA_CHECK(call)                                                                              \
  do {                                                                                                \
    cudaError_t err_ = (call);                                                                        \
    if (err_ != cudaSuccess) ::csm::fatal("CUDA error %s at %s:%d: %s", cudaGetErrorName(err_), __FILE__, \
                                          __LINE__, cudaGetErrorString(err_));                        \
  } while (0)

// Checks for launch errors
#ifdef CSM_DEBUG_SYNC
#define CUDA_CHECK_LAUNCH()                     \
  do {                                          \
    CUDA_CHECK(cudaGetLastError());             \
    CUDA_CHECK(cudaDeviceSynchronize());        \
  } while (0)
#else
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())
#endif

namespace csm {

// Measures GPU time between two points on a stream.
class GpuTimer {
 public:
  GpuTimer() {
    CUDA_CHECK(cudaEventCreate(&a_));
    CUDA_CHECK(cudaEventCreate(&b_));
  }
  ~GpuTimer() {
    cudaEventDestroy(a_);
    cudaEventDestroy(b_);
  }
  void start(cudaStream_t s = 0) { CUDA_CHECK(cudaEventRecord(a_, s)); }
  float stop_ms(cudaStream_t s = 0) {
    CUDA_CHECK(cudaEventRecord(b_, s));
    CUDA_CHECK(cudaEventSynchronize(b_));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, a_, b_));
    return ms;
  }

 private:
  cudaEvent_t a_, b_;
};

}  // namespace csm
