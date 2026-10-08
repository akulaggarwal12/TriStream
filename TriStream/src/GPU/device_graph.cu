#include "csm/device_graph.cuh"

#include <vector>

#include "csm/update_stream.hpp"

namespace csm {

void DeviceStream::upload(const UpdateStream& s) {
  release();
  const size_t n = s.updates.size();
  std::vector<vid_t> u(n), v(n);
  std::vector<label_t> el(n);
  std::vector<uint8_t> op(n);
  for (size_t i = 0; i < n; ++i) {
    u[i] = s.updates[i].u;
    v[i] = s.updates[i].v;
    el[i] = s.updates[i].el;
    op[i] = static_cast<uint8_t>(s.updates[i].op);
  }
  vid_t *du = nullptr, *dv = nullptr;
  label_t* del = nullptr;
  uint8_t* dop = nullptr;
  if (n > 0) {
    CUDA_CHECK(cudaMalloc(&du, sizeof(vid_t) * n));
    CUDA_CHECK(cudaMalloc(&dv, sizeof(vid_t) * n));
    CUDA_CHECK(cudaMalloc(&del, sizeof(label_t) * n));
    CUDA_CHECK(cudaMalloc(&dop, sizeof(uint8_t) * n));
    CUDA_CHECK(cudaMemcpy(du, u.data(), sizeof(vid_t) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dv, v.data(), sizeof(vid_t) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(del, el.data(), sizeof(label_t) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dop, op.data(), sizeof(uint8_t) * n, cudaMemcpyHostToDevice));
  }
  view_ = DeviceStreamView{n, du, dv, del, dop};
  bytes_ = n * (2 * sizeof(vid_t) + sizeof(label_t) + sizeof(uint8_t));
}

void DeviceStream::release() {
  if (view_.u) cudaFree(const_cast<vid_t*>(view_.u));
  if (view_.v) cudaFree(const_cast<vid_t*>(view_.v));
  if (view_.el) cudaFree(const_cast<label_t*>(view_.el));
  if (view_.op) cudaFree(const_cast<uint8_t*>(view_.op));
  view_ = DeviceStreamView{};
  bytes_ = 0;
}

}  // namespace csm
