#include "csm/host_graph.hpp"

#include <algorithm>
#include <utility>

namespace csm {
namespace {

struct KeyLabel {
  adj_t key;
  label_t el;
  bool operator<(const KeyLabel& o) const { return key != o.key ? key < o.key : el < o.el; }
};

// Scatters (src -> dst) entries into a pre-sized CSR, then sorts and deduplicates list in parallel
uint64_t fill_sort_dedup(HostCSR& csr, vid_t n, const std::vector<EdgeRecord>& edges,
                         const std::vector<label_t>& vlabel, bool with_elabels, bool reverse,
                         bool symmetric, uint64_t& label_conflicts) {
  std::vector<eid_t> deg(static_cast<size_t>(n) + 1, 0);
  for (const EdgeRecord& e : edges) {
    if (e.u == e.v) continue;
    ++deg[reverse ? e.v : e.u];
    if (symmetric) ++deg[e.v];
  }
  std::vector<eid_t> start(static_cast<size_t>(n) + 1, 0);
  for (vid_t v = 0; v < n; ++v) start[v + 1] = start[v] + deg[v];

  std::vector<adj_t> keys(start[n]);
  std::vector<label_t> els(with_elabels ? start[n] : 0);
  std::vector<eid_t> cursor(start.begin(), start.end() - 1);
  auto put = [&](vid_t src, vid_t dst, label_t el) {
    const eid_t p = cursor[src]++;
    keys[p] = adj_key(vlabel[dst], dst);
    if (with_elabels) els[p] = el;
  };
  for (const EdgeRecord& e : edges) {
    if (e.u == e.v) continue;
    if (reverse) put(e.v, e.u, e.el);
    else put(e.u, e.v, e.el);
    if (symmetric) put(e.v, e.u, e.el);
  }
  std::vector<eid_t>().swap(cursor);

  // Sort + unique per vertex.
  uint64_t conflicts = 0;
#pragma omp parallel for schedule(dynamic, 1024) reduction(+ : conflicts)
  for (int64_t vi = 0; vi < static_cast<int64_t>(n); ++vi) {
    const vid_t v = static_cast<vid_t>(vi);
    const eid_t b = start[v], e = start[v + 1];
    if (e - b <= 1) {
      deg[v] = e - b;
      continue;
    }
    if (!with_elabels) {
      std::sort(keys.begin() + b, keys.begin() + e);
      deg[v] = static_cast<eid_t>(std::unique(keys.begin() + b, keys.begin() + e) - (keys.begin() + b));
    } else {
      std::vector<KeyLabel> tmp(e - b);
      for (eid_t i = b; i < e; ++i) tmp[i - b] = {keys[i], els[i]};
      std::sort(tmp.begin(), tmp.end());
      eid_t w = 0;
      for (size_t i = 0; i < tmp.size(); ++i) {
        if (w > 0 && tmp[i].key == keys[b + w - 1]) {
          if (tmp[i].el != els[b + w - 1]) ++conflicts;
          continue;
        }
        keys[b + w] = tmp[i].key;
        els[b + w] = tmp[i].el;
        ++w;
      }
      deg[v] = w;
    }
  }
  label_conflicts += conflicts;

  csr.offsets.assign(static_cast<size_t>(n) + 1, 0);
  for (vid_t v = 0; v < n; ++v) csr.offsets[v + 1] = csr.offsets[v] + deg[v];
  csr.keys.resize(csr.offsets[n]);
  if (with_elabels) csr.elabels.resize(csr.offsets[n]);
#pragma omp parallel for schedule(dynamic, 4096)
  for (int64_t vi = 0; vi < static_cast<int64_t>(n); ++vi) {
    const vid_t v = static_cast<vid_t>(vi);
    std::copy_n(keys.begin() + start[v], deg[v], csr.keys.begin() + csr.offsets[v]);
    if (with_elabels) std::copy_n(els.begin() + start[v], deg[v], csr.elabels.begin() + csr.offsets[v]);
  }
  return start[n] - csr.offsets[n];
}

eid_t max_degree(const HostCSR& csr) {
  eid_t mx = 0;
  for (size_t v = 0; v + 1 < csr.offsets.size(); ++v) mx = std::max(mx, csr.offsets[v + 1] - csr.offsets[v]);
  return mx;
}

}  // namespace

HostGraph build_host_graph(RawGraph&& raw, bool directed, vid_t n_total, vid_t n_initial) {
  WallTimer timer;
  HostGraph g;
  g.directed = directed;
  g.has_edge_labels = raw.any_edge_label;
  g.n = n_total;
  g.n_initial = n_initial;
  g.vlabel = std::move(raw.vlabel);
  if (g.vlabel.size() != n_total) fatal("internal: vlabel size %zu != n_total %u", g.vlabel.size(), n_total);
  for (vid_t v = 0; v < n_total; ++v)
    if (g.vlabel[v] == kInvalidLabel) fatal("internal: vertex %u has no label", v);

  for (const EdgeRecord& e : raw.edges)
    if (e.u == e.v) ++g.build.self_loops_dropped;

  uint64_t conflicts = 0;
  const uint64_t removed_out =
      fill_sort_dedup(g.out, n_total, raw.edges, g.vlabel, g.has_edge_labels, false, !directed, conflicts);
  if (directed) {
    uint64_t ignored = 0;  // conflicts already counted on the out side
    fill_sort_dedup(g.in, n_total, raw.edges, g.vlabel, g.has_edge_labels, true, false, ignored);
    g.m = g.out.entries();
    g.build.duplicate_edges_dropped = removed_out;
    g.build.edge_label_conflicts = conflicts;
    g.build.max_in_degree = max_degree(g.in);
  } else {
    g.m = g.out.entries() / 2;
    g.build.duplicate_edges_dropped = removed_out / 2;
    g.build.edge_label_conflicts = conflicts / 2;
  }
  g.build.max_out_degree = max_degree(g.out);
  if (!directed) g.build.max_in_degree = g.build.max_out_degree;

  std::vector<EdgeRecord>().swap(raw.edges);
  g.build.build_ms = timer.ms();
  return g;
}

}  // namespace csm
