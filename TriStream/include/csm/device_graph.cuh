// GPU-resident copies of the three CSM inputs.
#pragma once

#include <cstddef>

#include "csm/common.hpp"
#include "csm/cuda_check.cuh"

namespace csm {

struct UpdateStream;


// First index in keys[b, e) with keys[i] >= x plain binary search
__device__ __forceinline__ eid_t lower_bound_key(const adj_t* keys, eid_t b, eid_t e, adj_t x) {
  while (b < e) {
    const eid_t mid = b + ((e - b) >> 1);
    if (keys[mid] < x) b = mid + 1;
    else e = mid;
  }
  return b;
}

struct DeviceStreamView {
  size_t size;
  const vid_t* u;
  const vid_t* v;
  const label_t* el;
  const uint8_t* op;  // 0 = insert, 1 = delete (UpdateOp)
};

class DeviceStream {
 public:
  DeviceStream() = default;
  ~DeviceStream() { release(); }
  DeviceStream(const DeviceStream&) = delete;
  DeviceStream& operator=(const DeviceStream&) = delete;

  void upload(const UpdateStream& s);
  void release();
  DeviceStreamView view() const { return view_; }
  size_t bytes() const { return bytes_; }

 private:
  DeviceStreamView view_{};
  size_t bytes_ = 0;
};

}  // namespace csm
