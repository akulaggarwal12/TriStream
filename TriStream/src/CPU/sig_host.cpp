#include "csm/sig_host.hpp"

#include <algorithm>

namespace csm {

sig::Layout KeyLayout::view() const {
  sig::Layout L;
  L.n1 = n1;
  L.ns = ns;
  L.nm = nm;
  L.nf = nf;
  L.np = np;
  L.words = n1 + ns + np;
  L.nel = nel;
  L.nvl = static_cast<uint32_t>(lq.size());
  L.nelab = static_cast<uint32_t>(le.size());
  L.lq = lq.data();
  L.le = use_el ? le.data() : nullptr;
  L.s1 = s1.data();
  L.st = st.data();
  L.mid = mid.data();
  L.far = far.data();
  L.pair = pair.data();
  L.pm = pm.data();
  L.moff = moff.data();
  L.mfar = mfar.data();
  L.mslot = mslot.data();
  return L;
}

KeyLayout build_key_layout(const HostAdj& q, bool use_el, label_t num_vlabels) {
  using namespace sig;
  KeyLayout k;
  k.use_el = use_el;
  const size_t n = q.inc.size();
  std::vector<label_t> labs(q.label.begin(), q.label.end());
  std::sort(labs.begin(), labs.end());
  labs.erase(std::unique(labs.begin(), labs.end()), labs.end());
  label_t nvl = num_vlabels;
  for (label_t l : labs) nvl = std::max(nvl, l + 1);
  k.lq.assign(nvl, kNoQL);
  for (size_t i = 0; i < labs.size(); ++i) k.lq[labs[i]] = static_cast<uint8_t>(i);
  const int nql = static_cast<int>(labs.size());
  if (use_el) {
    std::vector<label_t> els;
    for (size_t v = 0; v < n; ++v)
      for (const IncEdge& e : q.inc[v]) els.push_back(e.el);
    std::sort(els.begin(), els.end());
    els.erase(std::unique(els.begin(), els.end()), els.end());
    k.le.assign(els.empty() ? 1 : els.back() + 1, kNoQL);
    for (size_t i = 0; i < els.size(); ++i) k.le[els[i]] = static_cast<uint8_t>(i);
    k.nel = std::max<int>(1, static_cast<int>(els.size()));
  }
  auto ql = [&](vid_t w) { return static_cast<int>(k.lq[q.label[w]]); };
  auto ei = [&](label_t el) { return use_el ? static_cast<int>(k.le[el]) : 0; };
  std::vector<uint32_t> deg(n);
  for (size_t v = 0; v < n; ++v) deg[v] = static_cast<uint32_t>(q.inc[v].size());
  k.s1.assign(static_cast<size_t>(nql) * 2 * k.nel, -1);
  k.st.assign(static_cast<size_t>(nql) * 2 * kNumT, -1);
  k.mid.assign(static_cast<size_t>(nql) * 2, -1);
  k.far.assign(static_cast<size_t>(nql) * 2, -1);
  int c1 = 0, cs = 0, cm = 0, cf = 0;
  for (size_t v = 0; v < n; ++v)
    for (const IncEdge& e : q.inc[v]) {
      int8_t& s = k.s1[(ql(e.w) * 2 + static_cast<int>(e.d)) * k.nel + ei(e.el)];
      if (s < 0) s = static_cast<int8_t>(c1++ % kMaxSlope);
      const int c = stair_class(deg[e.w]);
      if (c > 0) {
        int8_t& t = k.st[(ql(e.w) * 2 + static_cast<int>(e.d)) * kNumT + (c - 1)];
        if (t < 0) t = static_cast<int8_t>(cs++ % kMaxStair);
      }
      for (const IncEdge& f : q.inc[e.w]) {
        if (f.w == v) continue;
        int8_t& m = k.mid[ql(e.w) * 2 + static_cast<int>(e.d)];
        if (m < 0) m = static_cast<int8_t>(cm++ % kMaxMid);
        int8_t& x = k.far[ql(f.w) * 2 + static_cast<int>(f.d)];
        if (x < 0) x = static_cast<int8_t>(cf++ % kMaxFar);
      }
    }
  k.n1 = std::min(c1, kMaxSlope);
  k.ns = std::min(cs, kMaxStair);
  k.nm = std::min(cm, kMaxMid);
  k.nf = std::min(cf, kMaxFar);
  k.pair.assign(static_cast<size_t>(k.nm) * k.nf, -1);
  int cp = 0;
  for (size_t v = 0; v < n; ++v)
    for (const IncEdge& e : q.inc[v])
      for (const IncEdge& f : q.inc[e.w]) {
        if (f.w == v) continue;
        const int m = k.mid[ql(e.w) * 2 + static_cast<int>(e.d)];
        const int x = k.far[ql(f.w) * 2 + static_cast<int>(f.d)];
        int16_t& p = k.pair[m * k.nf + x];
        if (p < 0) p = static_cast<int16_t>(cp++ % kMaxPair);
      }
  k.np = std::min(cp, kMaxPair);
  k.folded = c1 > kMaxSlope || cs > kMaxStair || cm > kMaxMid || cf > kMaxFar || cp > kMaxPair;
  k.pm.assign(std::max(1, k.np), 0u);
  k.moff.assign(k.nm + 1, 0);
  for (int m = 0; m < k.nm; ++m) {
    for (int x = 0; x < k.nf; ++x) {
      const int p = k.pair[m * k.nf + x];
      if (p < 0) continue;
      k.pm[p] |= 1u << m;
      k.mfar.push_back(static_cast<uint8_t>(x));
      k.mslot.push_back(static_cast<uint8_t>(p));
    }
    k.moff[m + 1] = static_cast<uint16_t>(k.mfar.size());
  }
  if (k.mfar.empty()) {
    k.mfar.push_back(0);
    k.mslot.push_back(0);
  }
  if (k.pair.empty()) k.pair.push_back(-1);
  if (k.s1.empty()) k.s1.push_back(-1);
  if (k.st.empty()) k.st.push_back(-1);
  if (k.mid.empty()) k.mid.push_back(-1);
  if (k.far.empty()) k.far.push_back(-1);
  return k;
}

void compute_signatures(const HostAdj& g, bool use_el, const sig::Layout& L, SigTable& out) {
  using namespace sig;
  const size_t n = g.inc.size();
  out.n = n;
  out.words = static_cast<size_t>(L.words);
  out.rows.assign(std::max<size_t>(1, n * out.words), 0);
  out.degout.assign(n, 0);
  out.degin.assign(n, 0);
  for (size_t v = 0; v < n; ++v)
    for (const IncEdge& e : g.inc[v]) (e.d ? out.degin : out.degout)[v]++;
  const int off = L.n1 + L.ns;
  for (size_t v = 0; v < n; ++v) {
    uint32_t* r = out.rows.data() + v * out.words;
    for (const IncEdge& e : g.inc[v]) {
      const label_t lw = g.label[e.w];
      const int s = L.slope(lw, e.d, use_el ? e.el : kNoEdgeLabel);
      if (s >= 0) r[s]++;
      const int cls = stair_class(out.degout[e.w] + out.degin[e.w]);
      for (int j = 0; j < cls; ++j) {
        const int t = L.stair(lw, e.d, j);
        if (t >= 0) r[L.n1 + t]++;
      }
      const int m = L.mid_of(lw, e.d);
      if (m < 0) continue;
      for (const IncEdge& f : g.inc[e.w])  // 2-path v –e– w –f– x, x != v
        if (f.w != v) {
          const int p = L.pair_of(m, L.far_of(g.label[f.w], f.d));
          if (p >= 0) r[off + p]++;
        }
    }
  }
}

HostAdj query_adjacency(const QueryGraph& q) {
  HostAdj a;
  a.directed = q.directed;
  a.label.assign(q.label, q.label + q.k);
  a.inc.resize(q.k);
  for (const QueryEdge& e : q.edges) {
    a.inc[e.src].push_back({e.dst, 0, e.el});
    a.inc[e.dst].push_back({e.src, q.directed ? 1u : 0u, e.el});
  }
  return a;
}

}  // namespace csm
