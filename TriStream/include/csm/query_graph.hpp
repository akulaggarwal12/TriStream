// Query graph: 
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "csm/common.hpp"
#include "csm/id_map.hpp"

namespace csm {

struct QueryEdge {
  uint8_t src, dst;  // for undirected queries src < dst
  label_t el;        // dense edge label, kNoEdgeLabel if absent
};

struct QueryGraph {
  int k = 0;  // number of vertices
  bool directed = false;
  bool has_edge_labels = false;  // edge labels are matched only if the query specifies any
  label_t label[kMaxQueryVertices] = {};
  uint32_t out_adj[kMaxQueryVertices] = {};  // undirected: out_adj == in_adj (symmetric)
  uint32_t in_adj[kMaxQueryVertices] = {};
  std::vector<QueryEdge> edges;       // each logical edge once
  std::vector<raw_id_t> raw_ids;      // original ids, for reporting

  uint32_t neighbors(int u) const { return out_adj[u] | in_adj[u]; }
  int degree(int u) const { return __builtin_popcount(neighbors(u)); }  // distinct neighbors
  bool connected() const;
  int max_degree() const;
};

// Reads a query file 
QueryGraph read_query_file(const std::string& path, bool directed, LabelMap& vlabels, LabelMap& elabels,
                           label_t default_vlabel);

}  // namespace csm
