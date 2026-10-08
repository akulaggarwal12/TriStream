// The update stream
#pragma once

#include <cstdint>
#include <vector>

#include "csm/common.hpp"

namespace csm {

struct Update {
  vid_t u, v;
  label_t el;  // dense edge label, kNoEdgeLabel if absent
  UpdateOp op;
};

// Self loops, unnecessary deletions and pre existing insertions are ignored and act as no-op
struct StreamStats {
  uint64_t insertions = 0;
  uint64_t deletions = 0;
  uint64_t new_vertices = 0;          // first seen in the stream (not in the initial graph)
  uint64_t vertex_lines = 0;          // 'v id label' lines in the stream
  uint64_t vertex_deletions_ignored = 0;  // '-v' lines (isolated vertex deletion cannot change matches)
  uint64_t self_loops_dropped = 0;
  uint64_t malformed_lines = 0;
};

struct UpdateStream {
  std::vector<Update> updates;
  StreamStats stats;
  size_t size() const { return updates.size(); }
};

}  // namespace csm
