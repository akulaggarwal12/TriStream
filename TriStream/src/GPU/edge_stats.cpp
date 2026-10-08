#include "csm/edge_stats.hpp"

#include <algorithm>

namespace csm {

double EdgeStats::fanout(label_t lp, uint32_t s, label_t lq) const {
  auto in = n.find(lp);
  auto im = m.find(key(lp, s, lq));
  if (in == n.end() || in->second == 0 || im == m.end()) return 0.0;
  return static_cast<double>(im->second) / static_cast<double>(in->second);
}

double EdgeStats::closure(label_t lp, uint32_t s, label_t lq) const {
  auto ip = n.find(lp), iq = n.find(lq);
  auto im = m.find(key(lp, s, lq));
  if (ip == n.end() || iq == n.end() || im == m.end()) return 0.0;
  return std::min(1.0, static_cast<double>(im->second) / (static_cast<double>(ip->second) * iq->second));
}

EdgeStats compute_edge_stats(const HostGraph& g) {
  EdgeStats st;
  for (vid_t v = 0; v < g.n; ++v) st.n[g.vlabel[v]]++;
  const int sides = g.directed ? 2 : 1;
  for (int s = 0; s < sides; ++s) {
    const HostCSR& c = s ? g.in : g.out;
    for (vid_t v = 0; v < g.n; ++v)
      for (eid_t i = c.offsets[v]; i < c.offsets[v + 1]; ++i)
        st.m[EdgeStats::key(g.vlabel[v], static_cast<uint32_t>(s), adj_label(c.keys[i]))]++;
  }
  return st;
}

}  // namespace csm
