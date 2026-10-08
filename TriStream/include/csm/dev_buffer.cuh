// Minimal owning device array where we grow based on only capacity, optional for copying 
#pragma once

#include <cstddef>
#include <utility>

#include "csm/cuda_check.cuh"

namespace csm {

template <class T>
class DevBuf {
 public:
  DevBuf() = default;
  ~DevBuf() { release(); }
  DevBuf(const DevBuf&) = delete;
  DevBuf& operator=(const DevBuf&) = delete;
  DevBuf(DevBuf&& o) noexcept { swap(o); }
  DevBuf& operator=(DevBuf&& o) noexcept {
    swap(o);
    return *this;
  }

  // Ensures capacity >= n. With keep=true the first size() elements survive a reallocation.
  void reserve(size_t n, bool keep = false) {
    if (n <= cap_) return;
    size_t c = cap_ + cap_ / 2;
    if (c < n) c = n;
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, c * sizeof(T)));
    if (keep && size_ > 0) CUDA_CHECK(cudaMemcpy(p, p_, size_ * sizeof(T), cudaMemcpyDeviceToDevice));
    if (p_) CUDA_CHECK(cudaFree(p_));
    p_ = p;
    cap_ = c;
  }
  void resize(size_t n, bool keep = false) {
    reserve(n, keep);
    size_ = n;
  }
  void upload(const T* host, size_t n) {
    resize(n);
    if (n) CUDA_CHECK(cudaMemcpy(p_, host, n * sizeof(T), cudaMemcpyHostToDevice));
  }
  void fill_bytes(int byte) {
    if (size_) CUDA_CHECK(cudaMemset(p_, byte, size_ * sizeof(T)));
  }
  void release() {
    if (p_) cudaFree(p_);
    p_ = nullptr;
    size_ = cap_ = 0;
  }
  void swap(DevBuf& o) noexcept {
    std::swap(p_, o.p_);
    std::swap(size_, o.size_);
    std::swap(cap_, o.cap_);
  }

  T* data() const { return p_; }
  size_t size() const { return size_; }

 private:
  T* p_ = nullptr;
  size_t size_ = 0;
  size_t cap_ = 0;
};

}  // namespace csm
