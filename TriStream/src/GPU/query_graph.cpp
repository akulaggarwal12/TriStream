#include "csm/query_graph.hpp"

#include <utility>

#include "csm/text_scanner.hpp"

namespace csm {

bool QueryGraph::connected() const {
  if (k == 0) return false;
  uint32_t seen = 1u, frontier = 1u;
  while (frontier) {
    uint32_t next = 0;
    for (uint32_t f = frontier; f; f &= f - 1) next |= neighbors(__builtin_ctz(f));
    frontier = next & ~seen;
    seen |= next;
  }
  const uint32_t all = (k == 32) ? 0xFFFFFFFFu : ((1u << k) - 1u);
  return seen == all;
}

int QueryGraph::max_degree() const {
  int mx = 0;
  for (int u = 0; u < k; ++u) mx = degree(u) > mx ? degree(u) : mx;
  return mx;
}

QueryGraph read_query_file(const std::string& path, bool directed, LabelMap& vlabels, LabelMap& elabels,
                           label_t default_vlabel) {
  QueryGraph q;
  q.directed = directed;
  std::vector<label_t> labels;  // kInvalidLabel until a 'v' line is seen

  auto local_id = [&](std::string_view tok, size_t line) -> int {
    raw_id_t raw;
    if (!parse_raw_id(tok, raw)) fatal("query %s:%zu: bad vertex id '%.*s'", path.c_str(), line,
                                       static_cast<int>(tok.size()), tok.data());
    for (size_t i = 0; i < q.raw_ids.size(); ++i)
      if (q.raw_ids[i] == raw) return static_cast<int>(i);
    if (q.raw_ids.size() >= static_cast<size_t>(kMaxQueryVertices))
      fatal("query %s has more than %d vertices", path.c_str(), kMaxQueryVertices);
    q.raw_ids.push_back(raw);
    labels.push_back(kInvalidLabel);
    return static_cast<int>(q.raw_ids.size() - 1);
  };

  MappedFile f(path);
  LineScanner sc(f.begin(), f.end());
  while (sc.next_line()) {
    std::string_view t, a, b, c;
    sc.next_token(t);
    if (t == "t") continue;
    if (t == "v") {
      uint64_t raw_label;
      if (!sc.next_token(a) || !sc.next_token(b) || !parse_u64(b, raw_label))
        fatal("query %s:%zu: expected 'v <id> <label>'", path.c_str(), sc.line_no());
      const int u = local_id(a, sc.line_no());
      labels[u] = vlabels.get_or_insert(raw_label);
    } else if (t == "e") {
      if (!sc.next_token(a) || !sc.next_token(b))
        fatal("query %s:%zu: expected 'e <u> <v> [label]'", path.c_str(), sc.line_no());
      int u = local_id(a, sc.line_no());
      int v = local_id(b, sc.line_no());
      if (u == v) fatal("query %s:%zu: self loop", path.c_str(), sc.line_no());
      label_t el = kNoEdgeLabel;
      uint64_t raw_el;
      if (sc.next_token(c) && parse_u64(c, raw_el)) {
        el = elabels.get_or_insert(raw_el);
        q.has_edge_labels = true;
      }
      if (!directed && u > v) std::swap(u, v);
      if (q.out_adj[u] & (1u << v)) continue;  // duplicate edge
      if (q.edges.size() >= static_cast<size_t>(kMaxQueryEdges))
        fatal("query %s has more than %d edges", path.c_str(), kMaxQueryEdges);
      q.out_adj[u] |= 1u << v;
      q.in_adj[v] |= 1u << u;
      if (!directed) {
        q.out_adj[v] |= 1u << u;
        q.in_adj[u] |= 1u << v;
      }
      q.edges.push_back({static_cast<uint8_t>(u), static_cast<uint8_t>(v), el});
    } else {
      fatal("query %s:%zu: unknown line type '%.*s'", path.c_str(), sc.line_no(), static_cast<int>(t.size()),
            t.data());
    }
  }

  q.k = static_cast<int>(q.raw_ids.size());
  for (int u = 0; u < q.k; ++u) q.label[u] = labels[u] == kInvalidLabel ? default_vlabel : labels[u];
  if (q.k == 0 || q.edges.empty()) fatal("query %s has no edges", path.c_str());
  if (!q.connected()) fatal("query %s is not connected (CSM requires a connected query)", path.c_str());
  return q;
}

}  // namespace csm
