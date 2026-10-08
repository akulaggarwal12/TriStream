// Host-side data graph: label-partitioned sorted CSR.
#pragma once

#include <cstdint>
#include <vector>

#include "csm/common.hpp"

namespace csm {

struct EdgeRecord {
  vid_t u, v;
  label_t el; 
};

struct GraphReadStats {
  uint64_t vertex_lines = 0;
  uint64_t edge_lines = 0;
  uint64_t implicit_vertices = 0;   // referenced by an edge before/without a 'v' line
  uint64_t relabel_ignored = 0;     // second 'v' line for an existing vertex
  uint64_t malformed_lines = 0;
};

// Result of parsing the initial graph file, before CSR construction.
struct RawGraph {
  std::vector<label_t> vlabel;  
  std::vector<EdgeRecord> edges;
  bool any_edge_label = false;
  GraphReadStats stats;
};

struct HostCSR {
  std::vector<eid_t> offsets;    // n + 1
  std::vector<adj_t> keys;       // per vertex, sorted ascending: grouped by neighbor label, then id
  std::vector<label_t> elabels;  // parallel to keys; empty when the graph has no edge labels

  eid_t degree(vid_t v) const { return offsets[v + 1] - offsets[v]; }
  eid_t entries() const { return keys.size(); }
};

struct GraphBuildStats {
  uint64_t self_loops_dropped = 0;
  uint64_t duplicate_edges_dropped = 0;  // logical edges
  uint64_t edge_label_conflicts = 0;     // same (u,v) listed with different edge labels
  eid_t max_out_degree = 0;
  eid_t max_in_degree = 0;
  double build_ms = 0.0;
};

struct HostGraph {
  bool directed = false;
  bool has_edge_labels = false;
  vid_t n = 0;          // all vertices, including those first seen in the stream
  vid_t n_initial = 0;  // vertices known after reading the initial graph file
  uint64_t m = 0;       // logical edges (undirected edge counted once)
  std::vector<label_t> vlabel;
  HostCSR out;  // undirected: symmetric adjacency (each edge stored at both endpoints)
  HostCSR in;   // directed only; empty for undirected graphs
  GraphBuildStats build;

};

// Builds the CSR over n_total vertices 
HostGraph build_host_graph(RawGraph&& raw, bool directed, vid_t n_total, vid_t n_initial);

CSM_HD uint64_t mix_entry(uint64_t a, uint64_t b) {
  uint64_t h = a * 0x9E3779B97F4A7C15ull ^ (b + 0x632BE59BD9B4E019ull);
  h ^= h >> 29;
  h *= 0xBF58476D1CE4E5B9ull;
  h ^= h >> 32;
  return h;
}

}  // namespace csm
