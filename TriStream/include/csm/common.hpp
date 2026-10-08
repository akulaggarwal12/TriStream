// Core types and constants shared by host and device code.
#pragma once

#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#ifdef __CUDACC__
#define CSM_HD __host__ __device__ __forceinline__
#else
#define CSM_HD inline
#endif

namespace csm {

using vid_t   = uint32_t;  // dense vertex id, 0..n-1
using label_t = uint32_t;  // dense label id (vertex labels and edge labels use separate maps)
using eid_t   = uint64_t;  // offset into an adjacency array
using adj_t   = uint64_t;  // packed adjacency entry: (label(neighbor) << 32) | neighbor
using ts_t    = uint32_t;  // update timestamp = 1-based position in the stream

constexpr vid_t   kInvalidVid    = 0xFFFFFFFFu;
constexpr label_t kInvalidLabel  = 0xFFFFFFFFu;
constexpr label_t kNoEdgeLabel   = 0;   // dense edge-label id reserved for "edge has no label"
constexpr int     kMaxQueryVertices = 32;  // query adjacency = one 32-bit mask per vertex
constexpr int     kMaxQueryEdges    = 64;

// Adjacency lists are sorted by the packed key, so each list is partitioned by neighbor
CSM_HD adj_t   adj_key(label_t l, vid_t v) { return (static_cast<adj_t>(l) << 32) | v; }
CSM_HD vid_t   adj_vertex(adj_t k)         { return static_cast<vid_t>(k & 0xFFFFFFFFu); }
CSM_HD label_t adj_label(adj_t k)          { return static_cast<label_t>(k >> 32); }

enum class UpdateOp : uint8_t { InsertEdge = 0, DeleteEdge = 1 };

[[noreturn]] inline void fatal(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::fprintf(stderr, "[csm] fatal: ");
  std::vfprintf(stderr, fmt, ap);
  std::fprintf(stderr, "\n");
  va_end(ap);
  std::exit(EXIT_FAILURE);
}

class WallTimer {
 public:
  WallTimer() : start_(std::chrono::steady_clock::now()) {}
  void reset() { start_ = std::chrono::steady_clock::now(); }
  double ms() const {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_).count();
  }

 private:
  std::chrono::steady_clock::time_point start_;
};

}  // namespace csm
