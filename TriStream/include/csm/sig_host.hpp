// S2 signatures on the host graph
#pragma once

#include <cstdint>
#include <vector>

#include "csm/common.hpp"
#include "csm/query_graph.hpp"
#include "csm/signature.hpp"

namespace csm {

struct IncEdge {
  vid_t w;
  uint32_t d;  // 0 = out, 1 = in 
  label_t el;
};

// Distinct incident directed edges of every vertex of one snapshot.
struct HostAdj {
  bool directed = false;
  std::vector<label_t> label;
  std::vector<std::vector<IncEdge>> inc;
};

struct SigTable {
  size_t n = 0;
  size_t words = 0;
  std::vector<uint32_t> rows; // Compressed rows sorted.
  std::vector<uint32_t> degout, degin;
  const uint32_t* row(size_t v) const { return rows.data() + v * words; }
};

struct KeyLayout {
  int n1 = 0, ns = 0, nm = 0, nf = 0, np = 0, nel = 1;
  bool use_el = false, folded = false;
  std::vector<uint8_t> lq, le;
  std::vector<int8_t> s1, st, mid, far;
  std::vector<int16_t> pair;
  std::vector<uint32_t> pm;
  std::vector<uint16_t> moff;
  std::vector<uint8_t> mfar, mslot;
  sig::Layout view() const;
};

KeyLayout build_key_layout(const HostAdj& q, bool use_el, label_t num_vlabels);

// marks hub during s2 computation
void compute_signatures(const HostAdj& g, bool use_el, const sig::Layout& L, SigTable& out);

HostAdj query_adjacency(const QueryGraph& q);

}  // namespace csm
