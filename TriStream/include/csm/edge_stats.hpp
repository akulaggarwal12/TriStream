// Edge-type statistics are the priors of the S1 cost model
#pragma once

#include <cstdint>
#include <unordered_map>

#include "csm/common.hpp"
#include "csm/host_graph.hpp"

namespace csm {

struct EdgeStats {
  std::unordered_map<label_t, uint64_t> n;
  std::unordered_map<uint64_t, uint64_t> m;

  static uint64_t key(label_t lp, uint32_t s, label_t lq) {
    return (static_cast<uint64_t>(lp) << 33) | (static_cast<uint64_t>(s) << 32) | lq;
  }
  double fanout(label_t lp, uint32_t s, label_t lq) const;
  double closure(label_t lp, uint32_t s, label_t lq) const;
};

EdgeStats compute_edge_stats(const HostGraph& g);

}  // namespace csm
