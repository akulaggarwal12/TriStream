// Edge identities are shared by the GPU graph and the CPU execution.
#pragma once

#include <cstdint>

#include "csm/common.hpp"
#include "csm/host_graph.hpp"  // mix_entry

namespace csm {

CSM_HD uint64_t pack_sk(vid_t src, vid_t dst) { return (static_cast<uint64_t>(src) << 32) | dst; }

CSM_HD uint64_t canonical_edge(vid_t u, vid_t v, bool directed) {
  return (directed || u < v) ? pack_sk(u, v) : pack_sk(v, u);
}

CSM_HD uint64_t edge_print(uint64_t canonical) { return mix_entry(canonical, 0x5EEDull); }

}  // namespace csm
