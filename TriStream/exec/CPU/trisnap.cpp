// TriStream: TriSnap, the CPU-parallel execution of TriMatch
#include <omp.h>
#include <x86intrin.h>
#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <parallel/algorithm>
#include <string>
#include <vector>
#include "csm/edge_key.hpp"
#include "csm/edge_stats.hpp"
#include "csm/loader.hpp"
#include "csm/match_plan.hpp"
#include "csm/sig_host.hpp"
#include "csm/signature.hpp"

namespace csm {
namespace {

constexpr ts_t kInf = 0xFFFFFFFFu;
using ull = unsigned long long;

// Options of the CPU execution
struct Opt {
  LoadOptions load; // Loading 3 graphs: data, query and stream
  std::string name; // Name of the dataset
  bool verbose = false;  // print progress and stats
  size_t batch = 65536; // updates per micro-batch (b)
  int threads = 0; // Number of cpu threads
  uint32_t hub_tau = 256; // hub threshold tau of the S2 curvature mask
  double compact_ratio = 0.25; 
  int min_k = 5, explore = 16; // Used in the exploration for s1
  double time_limit = 0;      // time limit flag
};

std::atomic<bool> g_tle{false}; // time limit exceed flag.
uint64_t g_deadline = ~0ull; // never initially, set later, time when needs to stop.

// sorting based on parallel or sequential.
constexpr size_t kParSort = size_t(1) << 16; // A threshold, needed for faster sorting.
template <class It, class C>
void psort(It b, It e, C c) { // g++ special sort, works in case of parallelism.
  if (size_t(e - b) >= kParSort) __gnu_parallel::sort(b, e, c);
  else std::sort(b, e, c);
}
template <class It, class C>
void pstable(It b, It e, C c) { // keeps sorting stable. equal data in same order as before.
  if (size_t(e - b) >= kParSort) __gnu_parallel::stable_sort(b, e, c);
  else std::stable_sort(b, e, c);
}
template <class It>
void psort(It b, It e) { // simple sort.
  psort(b, e, std::less<typename std::iterator_traits<It>::value_type>());
}

struct DEnt {  // stream edges type.
  adj_t key; // label + vertex
  label_t el; // edge label
  ts_t ins, del; // time when it was inserted and deleted, initially del = infinite (alive)
};

struct Side {  // CSR type, for simple adjacency list.
  std::vector<eid_t> off; // CSR offsets, n + 1
  std::vector<adj_t> key; // label + vertex for that, sorted initially.
  std::vector<label_t> el;  // edge label, empty if no edge labels
  std::vector<ts_t> del; // deletion time for the edge
  std::vector<std::vector<DEnt>> delta; // list of stream edges.
};

// Versioned Graph timestamp method.
struct Graph {
  bool directed = false, has_el = false; // falgs for directed graph and edge labelled graph.
  vid_t n = 0; // number of vertices.
  std::vector<label_t> vl; // vertex label if exist.
  Side s[2];  // storing out edges and in edges, for undirected graph, only first index is used.
  uint64_t delta_entries = 0, dead_base = 0, base_entries = 0; // deciding the compactness of graph.

  const Side& side(int d) const { return s[directed ? d : 0]; } // returns the side of graph
  Side& side(int d) { return s[directed ? d : 0]; } // side only count if notion of direction exists. 


  // Uses two binary searches to find the range of correct label edges as sorting is done on label first.
  static void seg(const std::vector<adj_t>& keys, eid_t b, eid_t e, label_t l, eid_t& lo, eid_t& hi) { 
    lo = std::lower_bound(keys.begin() + b, keys.begin() + e, adj_key(l, 0)) - keys.begin();
    hi = std::lower_bound(keys.begin() + lo, keys.begin() + e, l == 0xFFFFFFFFu ? ~0ull : adj_key(l + 1, 0)) -
         keys.begin();
  }

  // Uses the same on streaming edges.
  static void dseg(const std::vector<DEnt>& d, label_t l, size_t& lo, size_t& hi) {
    auto cmp = [](const DEnt& x, adj_t k) { return x.key < k; };
    lo = std::lower_bound(d.begin(), d.end(), adj_key(l, 0), cmp) - d.begin();
    hi = std::lower_bound(d.begin() + lo, d.end(), l == 0xFFFFFFFFu ? ~0ull : adj_key(l + 1, 0), cmp) - d.begin();
  }

  // Checks whether their exist the edge btw a -> w for the given time.
  bool visible(int d, vid_t a, vid_t w, ts_t th, label_t el) const {
    const Side& S = side(d);
    const adj_t k = adj_key(vl[w], w);
    auto it = std::lower_bound(S.key.begin() + S.off[a], S.key.begin() + S.off[a + 1], k);
    if (it != S.key.begin() + S.off[a + 1] && *it == k) {
      const size_t i = it - S.key.begin();
      if (S.del[i] > th && (el == kAnyEdgeLabel || (S.el.empty() ? kNoEdgeLabel : S.el[i]) == el)) return true;
    }
    const auto& D = S.delta[a];
    auto jt = std::lower_bound(D.begin(), D.end(), k, [](const DEnt& x, adj_t kk) { return x.key < kk; });
    for (; jt != D.end() && jt->key == k; ++jt)
      if (jt->ins <= th && th < jt->del && (el == kAnyEdgeLabel || jt->el == el)) return true;
    return false;
  }
};

// Builds a versioned graph out of it.
Graph build_graph(HostGraph& h) {
  Graph g;
  g.directed = h.directed;
  g.has_el = h.has_edge_labels;
  g.n = h.n;
  g.vl = std::move(h.vlabel); // uses move to prevent copying of data, saving memory. directly using loader arrays
  HostCSR* src[2] = {&h.out, &h.in};
  for (int d = 0; d < (h.directed ? 2 : 1); ++d) {
    Side& S = g.s[d];
    // moving the data
    S.off = std::move(src[d]->offsets);
    S.key = std::move(src[d]->keys);
    S.el = std::move(src[d]->elabels);
    S.del.assign(S.key.size(), kInf);
    S.delta.assign(h.n, {});
    g.base_entries += S.key.size();
  }
  return g;
}

// A candidate list: base segment [bb, be) + delta segment [db, de) of one vertex / side / label.
struct List {
  const Side* S;
  vid_t a;
  eid_t bb, be;
  size_t db, de;
  uint64_t len() const { return (be - bb) + (de - db); }
  // candidate at position i if visible at θ with edge label el, else kInvalidVid
  vid_t at(uint64_t i, ts_t th, label_t el) const {
    if (i < be - bb) {
      const eid_t j = bb + i;
      if (S->del[j] <= th) return kInvalidVid;
      if (el != kAnyEdgeLabel && (S->el.empty() ? kNoEdgeLabel : S->el[j]) != el) return kInvalidVid;
      return adj_vertex(S->key[j]);
    }
    const DEnt& e = S->delta[a][db + (i - (be - bb))];
    if (!(e.ins <= th && th < e.del)) return kInvalidVid;
    if (el != kAnyEdgeLabel && e.el != el) return kInvalidVid;
    return adj_vertex(e.key);
  }
};

List list_of(const Graph& g, int d, vid_t a, label_t l) {
  List L;
  L.S = &g.side(d);
  L.a = a;
  Graph::seg(L.S->key, L.S->off[a], L.S->off[a + 1], l, L.bb, L.be);
  Graph::dseg(L.S->delta[a], l, L.db, L.de);
  return L;
}

// S2 derivative signatures
struct Sig {
  static constexpr uint32_t kNoSlot = 0xFFFFFFFFu;
  bool use_el = false;
  uint32_t tau = 256;
  std::vector<uint8_t> keep;  // label mask
  std::vector<uint32_t> slot;  // vertex changes to compressed row indexes. no label, so no row
  std::vector<uint32_t> rows, dout, din, prof; // rows hold the 1-hop and 2-hop signature counters, then out / in degrees and the hashed neighbour profile.
  std::vector<uint32_t> mask; std::vector<uint8_t> hub; // hub mask of the curvature groups, hub flags
  std::vector<uint8_t> stale;
  SigTable q; // the query's own signatures
  KeyLayout keys;
  sig::Layout L;
  bool kept(const Graph& g, vid_t v) const { return g.vl[v] < keep.size() && keep[g.vl[v]]; } // query projection: only vertices with a query label have a row
  uint32_t deg(vid_t v) const { const uint32_t k = slot[v]; return k == kNoSlot ? 0 : dout[k] + din[k]; } // returns total degree.
};

// checks whether the required edges were in the active window.
template <class F>
void for_incident(const Graph& g, const Sig& s, vid_t v, ts_t lo, ts_t hi, F&& f) {
  for (int d = 0; d < (g.directed ? 2 : 1); ++d) { // first out edges.
    const Side& S = g.s[d];
    for (eid_t i = S.off[v]; i < S.off[v + 1]; ++i) { // base neighbours
      if (S.del[i] < lo) continue;
      const vid_t w = adj_vertex(S.key[i]);
      if (!s.kept(g, w)) continue;
      f(w, static_cast<uint32_t>(d), S.el.empty() ? kNoEdgeLabel : S.el[i]);
    } 
    for (const DEnt& e : S.delta[v]) { // inserted neighbours.
      if (e.ins > hi || e.del < lo) continue;
      const vid_t w = adj_vertex(e.key);
      if (!s.kept(g, w)) continue;
      f(w, static_cast<uint32_t>(d), e.el);
    }
  }
}

// checks whether the edge is there or not.
bool in_window(const Graph& g, int d, vid_t a, vid_t w, ts_t lo, ts_t hi) {
  const Side& S = g.s[d];
  const adj_t k = adj_key(g.vl[w], w); // checking for the label.
  auto it = std::lower_bound(S.key.begin() + S.off[a], S.key.begin() + S.off[a + 1], k); // binary search on base graph.
  if (it != S.key.begin() + S.off[a + 1] && *it == k && S.del[it - S.key.begin()] >= lo) return true;
  const auto& D = S.delta[a]; // checking for the streaming edges.
  auto jt = std::lower_bound(D.begin(), D.end(), k, [](const DEnt& x, adj_t kk) { return x.key < kk; });
  for (; jt != D.end() && jt->key == k; ++jt)
    if (jt->ins <= hi && jt->del >= lo) return true;
  return false; // if any streaming or base graph doesnt have it
}

// S2 work starts
void refresh_deg(const Graph& g, Sig& s, vid_t v, ts_t lo, ts_t hi) {
  uint32_t o = 0, i = 0, pr[sig::kMaxFar] = {};
  for_incident(g, s, v, lo, hi, [&](vid_t w, uint32_t d, label_t) {
    (d ? i : o)++; // counting neighbours
    const int f = s.L.far_of(g.vl[w], d);
    if (f >= 0) pr[f]++; // Hashing the neighbours accordingly.
  });
  const size_t k = s.slot[v];
  s.dout[k] = o;
  s.din[k] = i;
  std::memcpy(&s.prof[k * s.L.nf], pr, sizeof(uint32_t) * s.L.nf);
  if (o + i > s.tau) s.hub[k] = 1;  // degree above tau: hub (sticky); curvature through it is masked
}

// refreshes the signature row of v for the window from lo to hi.
void refresh_row(const Graph& g, Sig& s, vid_t v, ts_t lo, ts_t hi) {
  using namespace sig;
  const Layout& Ly = s.L;
  const int offC = Ly.n1 + Ly.ns;
  uint32_t r[kMaxWords] = {};
  uint32_t m = 0;
  const size_t kv = s.slot[v];
  const bool vhub = s.hub[kv];
  for_incident(g, s, v, lo, hi, [&](vid_t w, uint32_t d, label_t el) {
    const label_t lw = g.vl[w];
    const int sl = Ly.slope(lw, d, s.use_el ? el : kNoEdgeLabel);
    if (sl >= 0) r[sl]++; // slope hashing, 1 hop
    const size_t kw = s.slot[w];
    const int cls = stair_class(s.dout[kw] + s.din[kw]); // stair hashing, 2 hop
    for (int j = 0; j < cls; ++j) {
      const int t = Ly.stair(lw, d, j);
      if (t >= 0) r[Ly.n1 + t]++; // doing the hashing
    }
    if (vhub) return; // if hub, flat curve, skip.
    const int grp = Ly.mid_of(lw, d); // curvature of the middle node, a->b<-c type.
    if (grp < 0) return;
    if (s.hub[kw]) {  // hub, skip it.
      m |= 1u << grp;
      return;
    }
    const uint32_t* pw = &s.prof[kw * Ly.nf];
    for (int x = Ly.moff[grp]; x < Ly.moff[grp + 1]; ++x) r[offC + Ly.mslot[x]] += pw[Ly.mfar[x]];
    // returning edges of w to v: w -> v (from w: out, d' = 0) and v -> w (from w: in, d' = 1); undirected: one
    const label_t lv = g.vl[v]; // getting label of v to check for it.
    if (!g.directed) {
      const int p0 = Ly.pair_of(grp, Ly.far_of(lv, 0));
      if (p0 >= 0) r[offC + p0] -= 1; // row layout: slope, staircase, then curvature pairs (middle group, far bucket)
    } else { // subtract the 2-paths v-w-v that return to v (x must differ from v)
      const int p0 = Ly.pair_of(grp, Ly.far_of(lv, 0)), p1 = Ly.pair_of(grp, Ly.far_of(lv, 1));
      if (p0 >= 0 && in_window(g, 0, w, v, lo, hi)) r[offC + p0] -= 1;  // w -> v
      if (p1 >= 0 && in_window(g, 1, w, v, lo, hi)) r[offC + p1] -= 1;  // v -> w
    }
  });
  std::memcpy(&s.rows[kv * Ly.words], r, sizeof(uint32_t) * Ly.words); // copy the row's counters (Ly.words, set by the query)
  s.mask[kv] = m; // curvature groups masked through hub neighbours
  s.stale[kv] = 0;
}

bool dominates(const Sig& s, vid_t v, int q) {
  const uint32_t* qr = s.q.row(q);
  const size_t k = s.slot[v];
  return sig::dominates(s.L, qr, s.q.degout[q], s.q.degin[q], &s.rows[k * s.L.words], s.dout[k], s.din[k], s.mask[k],
                        s.hub[k]); // tells us whether the vertex satisfies minimum signature needs of the query vertex.
}

inline void slice_of(uint64_t len, uint32_t j, uint32_t m, uint64_t& lo, uint64_t& hi) {
  const uint64_t units = (len + 31) / 32;
  if (units >= m) {
    lo = (j * units / m) * 32;
    hi = j + 1 == m ? len : ((j + 1) * units / m) * 32;
  } else {
    lo = j * len / m;
    hi = (j + 1) * len / m;
  }
  lo = std::min(lo, len);
  hi = std::min(hi, len);
}

// Work information
struct Task {
  uint32_t upd;  // index in the batch
  uint16_t plan; // anchor plan to compare.
  uint8_t bet, explore; // S1 flags: bet or explore if higher than betting threshold.
  uint8_t sm[4], sj[4];
  float cost;  // predicted chunks (whole task)
  float raw;
  uint64_t n2, lo, hi;  // level-2 list length, and this piece's range [lo, hi) of it (hi = ~0: open end)
  uint64_t budget; // for S1 bets, current threshold
};

struct Learn {  // S1 memory, for cost model for upcoming batches.
  std::vector<double> e, c, s, cand;  // necessary nodes vectors.
  std::vector<double> calA, calP, bets, aborts; // vectors storing the S1 cost model, actual pred work and betting and failed bets
  std::vector<uint8_t> ready; // is the plan ready for next batch search.
};

struct Stats { // necessary TriMatch stats and flags.
  uint64_t eff_ins = 0, eff_del = 0, noop = 0, anchored = 0, tested = 0, rejected = 0, probe_rej = 0, gate_rej = 0,
           gate_tested = 0, clo_fail = 0, tasks = 0, pieces = 0, split = 0, switched = 0, explored = 0, aborts = 0,
           bet_waste = 0, chunks = 0, s2_rows = 0, s2_sleeps = 0, s2_wakes = 0, s2_dormant_b = 0, s1_dormant_b = 0,
           flat_b = 0, compactions = 0, batches = 0, reentries = 0;
  double apply_ms = 0, s2_ms = 0, triage_ms = 0, s1_ms = 0, s3_ms = 0, search_ms = 0, compact_ms = 0;
};

struct Engine {
  const Opt& o; // settings of the CPU execution
  const QueryGraph& q; // query graph
  const MatchPlan& p; // anchor + order + levels info
  Graph& g; // Base graph csr + delta
  Sig& s; // S2 signature
  Stats st; // TriMatch stats
  Learn L; // S1 memory
  int P = 1; // no of threads.
  double cyc_per_ms = 1e6; // cpu cycle for time.
  bool s2_dormant = false; // S2 asleep: no maintenance, triage only as a count-only probe
  uint64_t batch_no = 0, gate_off_until = 0, clo_off_until = 0, s1_sleep_until = 0; // controller state: batch number, gate / closure off until, S1 asleep until
  int flat_streak = 0, wake_streak = 0, probe_every = 1; 
  uint64_t next_probe = 0;
  double s2_cost_per_upd = -1, reentry_per_ep = -1, bias = 1.0, last_probe_rate = 0;
  bool bias_known = false;
  std::vector<vid_t> backlog;
  bool any_gate = false;
  std::vector<vid_t> stale_l;
  std::vector<uint8_t> in_backlog; // endpoints recorded while S2 sleeps, refreshed exactly when it wakes

  Engine(const Opt& o_, const QueryGraph& q_, const MatchPlan& p_, Graph& g_, Sig& s_) // constructor
      : o(o_), q(q_), p(p_), g(g_), s(s_) {
    const size_t pk = size_t(p.num_plans) * p.k; // no of plan slots needed.
    L.e.assign(pk, 0); // level entered
    L.c.assign(pk, 0); // chunks worked
    L.s.assign(pk, 0); // survivors
    L.cand.assign(pk, 0); // candidates
    L.calA.assign(p.num_plans, 0); // All initialised at 0
    L.calP.assign(p.num_plans, 0);
    L.bets.assign(p.num_plans, 0);
    L.aborts.assign(p.num_plans, 0);
    L.ready.assign(p.num_plans, 0); // no bet since nothing is learned initially.
    in_backlog.assign(g.n, 0); // no s2 backlog
    any_gate = std::find(p.gate.begin(), p.gate.end(), 1) != p.gate.end();
    const uint64_t c0 = __rdtsc(); // cpu cycle
    WallTimer t; // starts time
    while (t.ms() < 5.0) { // spin wait 5 ms
    }
    cyc_per_ms = double(__rdtsc() - c0) / t.ms(); //cycle/ms
  }

  const PlanLevel& lv(int plan, int L_) const { return p.levels[size_t(plan) * p.k + L_]; } // Give the level of the plan

  // pivot list of level L under partial map phi (the shortest of the level's check lists)
  List pivot(int plan, int L_, const vid_t* phi, int& piv) const {
    const PlanLevel& l = lv(plan, L_);
    List best{};
    uint64_t bl = ~0ull;
    piv = 0;
    for (int c = 0; c < l.nchk; ++c) {
      const PlanCheck& ch = p.checks[l.cb + c];
      List x = list_of(g, ch.dir, phi[ch.pos], l.lq);
      if (x.len() < bl) {
        bl = x.len();
        best = x;
        piv = c;
      }
    }
    return best;
  }

  bool closure_ok(int a, vid_t x, vid_t y, ts_t th) const { // initial checking for closed loops, checking common edges.
    const int rb = p.creq_off[a], nr = p.creq_off[a + 1] - rb; // anchor requirements matching
    if (nr == 0) return true; // if not common edges
    const uint32_t dx = s.deg(x), dy = s.deg(y); // degree of x, y
    const bool xu = dx <= dy; // which node has fewer neighbours
    const vid_t a0 = xu ? x : y, b0 = xu ? y : x;
    uint32_t have[64] = {}; // neighbours, as sorted labelling
    for (int side = 0; side < (g.directed ? 2 : 1); ++side) { // first outgoing, then ingoing
      for (int r0 = 0; r0 < nr; ++r0) { // no of labels
        const label_t l = p.creq[rb + r0].l;
        if (r0 > 0 && p.creq[rb + r0 - 1].l == l) continue; // skip if previous label match
        const List c = list_of(g, side, a0, l);
        for (uint64_t i = 0; i < c.len(); ++i) {
          const vid_t w = c.at(i, th, kAnyEdgeLabel);
          if (w == kInvalidVid || w == b0) continue;
          uint32_t hx, hy;
          if (!g.directed) {
            hx = 1;
            hy = g.visible(0, b0, w, th, kAnyEdgeLabel) ? 1u : 0u;
          } else {
            if (side == 0) {
              hx = 1u | (g.visible(1, a0, w, th, kAnyEdgeLabel) ? 2u : 0u);
            } else {
              if (g.visible(0, a0, w, th, kAnyEdgeLabel)) continue;  // already counted on side 0
              hx = 2u;
            }
            hy = (g.visible(0, b0, w, th, kAnyEdgeLabel) ? 1u : 0u) | (g.visible(1, b0, w, th, kAnyEdgeLabel) ? 2u : 0u);
          }
          bool all = true;
          for (int r = 0; r < nr; ++r) {
            const ClosureReq& cq = p.creq[rb + r];
            const uint32_t nx = xu ? cq.need0 : cq.need1, ny = xu ? cq.need1 : cq.need0;
            if (cq.l == l && (hx & nx) == nx && (hy & ny) == ny) have[r]++;
            all &= have[r] >= cq.target;
          }
          if (all) return true;
        }
      }
    }
    for (int r = 0; r < nr; ++r)
      if (have[r] < p.creq[rb + r].target) return false; // if some requirement is missing, no match
    return true;
  }

  // leaf counting: the independent tail of leaves is counted by inclusion-exclusion instead of enumerated.
  uint64_t leaf_count(const LeafPlan& lp, const vid_t* phi, int nphi, ts_t th, uint64_t& work) const {
    uint64_t cntm[1u << kMaxLeaves] = {};
    int pvs[kMaxLeaves], ord[kMaxLeaves];
    List cs[kMaxLeaves];
    for (int i = 0; i < lp.ng; ++i) {
      int pv = 0;
      uint64_t bl = ~0ull;
      List c{};
      for (int y = 0; y < lp.gnc[i]; ++y) {
        const List t = list_of(g, lp.gdir[i][y], phi[lp.gpos[i][y]], lp.glab[i]);
        if (t.len() < bl) {
          bl = t.len();
          c = t;
          pv = y;
        }
      }
      if (bl < lp.gmul[i]) return 0;
      pvs[i] = pv;
      cs[i] = c;
      int x = i;
      for (; x > 0 && cs[ord[x - 1]].len() > bl; --x) ord[x] = ord[x - 1];
      ord[x] = i;
    }
    for (int oi = 0; oi < lp.ng; ++oi) {
      const int i = ord[oi], pv = pvs[i];
      const List& c = cs[i];
      uint64_t nv = 0;
      work += (c.len() + 31) / 32;
      for (uint64_t x = 0; x < c.len(); ++x) {
        const vid_t w = c.at(x, th, lp.gel[i][pv]);
        if (w == kInvalidVid) continue;
        bool used = false;
        for (int j = 0; j < nphi; ++j) used |= phi[j] == w;
        if (used) continue;
        bool ok = true;
        for (int y = 0; y < lp.gnc[i] && ok; ++y)
          if (y != pv) ok = g.visible(lp.gdir[i][y], phi[lp.gpos[i][y]], w, th, lp.gel[i][y]);
        if (!ok) continue;
        ++nv;
        uint32_t m = 1u << i;
        bool lowest = true;
        for (int j = 0; j < lp.ng; ++j) {
          if (j == i || lp.glab[j] != lp.glab[i]) continue;
          bool in = true;
          for (int y = 0; y < lp.gnc[j] && in; ++y) in = g.visible(lp.gdir[j][y], phi[lp.gpos[j][y]], w, th, lp.gel[j][y]);
          if (in) {
            m |= 1u << j;
            if (j < i) lowest = false;
          }
        }
        if (lowest) ++cntm[m];
      }
      if (nv < lp.gmul[i]) return 0;
    }
    uint64_t I[1u << kMaxLeaves] = {};
    const uint32_t full = 1u << lp.ng;
    for (uint32_t G = 1; G < full; ++G)
      for (uint32_t M = G; M < full; ++M)
        if ((M & G) == G) I[G] += cntm[M];
    uint64_t tot = 0;
    for (int t = 0; t < lp.nterm; ++t) {
      uint64_t v = static_cast<uint64_t>(static_cast<int64_t>(lp.coef[t]));
      for (int b = 0; b < lp.nblk[t]; ++b) v *= I[lp.bmask[t][b]];
      tot += v;
    }
    return tot;
  }

  // one piece on one thread. Returns the count and sets aborted if a bet overran its budget.
  uint64_t run_piece(const Task& tk, const Update& u, ts_t th, bool gate, uint64_t& work, bool& aborted,
                     std::vector<double>* smp) const {
    const int k = p.k, plan = tk.plan;
    vid_t phi[kMaxQueryVertices];
    phi[0] = u.u;
    phi[1] = u.v;
    // level-1 checks against level 0 at θ
    {
      const PlanLevel& l1 = lv(plan, 1);
      for (int c = 0; c < l1.nchk; ++c) {
        const PlanCheck& ch = p.checks[l1.cb + c];
        if (!g.visible(ch.dir, phi[ch.pos], phi[1], th, ch.el)) return 0;
      }
    }
    if (k == 2) return tk.lo == 0 ? 1 : 0;
    const int lfrom = p.leaf[plan].from;
    struct Fr {
      List lst;
      uint64_t pos, end;
      label_t el;
      int piv;
    } fr[kMaxQueryVertices];
    uint64_t cnt = 0;
    int L_ = 2;
    fr[2].lst = pivot(plan, 2, phi, fr[2].piv);
    fr[2].el = p.checks[lv(plan, 2).cb + fr[2].piv].el;
    fr[2].pos = tk.lo;
    fr[2].end = std::min<uint64_t>(tk.hi, fr[2].lst.len());
    work += (fr[2].end > fr[2].pos ? fr[2].end - fr[2].pos + 31 : 0) / 32;
    if (smp) {
      (*smp)[L_ * 4 + 0] += 1;
      (*smp)[L_ * 4 + 1] += double(fr[2].end - fr[2].pos);
      (*smp)[L_ * 4 + 3] += double((fr[2].end - fr[2].pos + 31) / 32);
    }
    uint32_t tick = 0;
    while (L_ >= 2) {
      if ((++tick & 1023u) == 0 && __rdtsc() > g_deadline) {
        g_tle.store(true, std::memory_order_relaxed);
        return cnt;
      }
      Fr& f = fr[L_];
      if (f.pos >= f.end) {
        --L_;
        continue;
      }
      const vid_t w = f.lst.at(f.pos++, th, f.el);
      if (w == kInvalidVid) continue;
      bool ok = true;
      for (int j = 0; j < L_ && ok; ++j) ok = phi[j] != w;
      const PlanLevel& l = lv(plan, L_);
      for (int c = 0; c < l.nchk && ok; ++c) {
        if (c == f.piv) continue;
        const PlanCheck& ch = p.checks[l.cb + c];
        ok = g.visible(ch.dir, phi[ch.pos], w, th, ch.el);
      }
      if (ok && gate && p.gate[size_t(plan) * k + L_]) {
        ok = dominates(s, w, l.q);
        if (!ok) gate_rej_local++;
        gate_tested_local++;
      }
      if (!ok) continue;
      if (smp) (*smp)[L_ * 4 + 2] += 1;
      if (L_ == k - 1) {
        ++cnt;
        continue;
      }
      if (L_ == lfrom - 1) {  // only leaves remain so count them directly
        phi[L_] = w;
        const uint64_t w0 = work;
        cnt += leaf_count(p.leaf[plan], phi, lfrom, th, work);
        if (smp) {
          (*smp)[lfrom * 4 + 0] += 1;
          (*smp)[lfrom * 4 + 1] += double(work - w0) * 32.0;
          (*smp)[lfrom * 4 + 3] += double(work - w0);
        }
        if (tk.bet && work > tk.budget) {
          aborted = true;
          return 0;
        }
        continue;
      }
      phi[L_] = w;
      ++L_;
      Fr& nf = fr[L_];
      nf.lst = pivot(plan, L_, phi, nf.piv);
      nf.el = p.checks[lv(plan, L_).cb + nf.piv].el;
      nf.pos = 0;
      nf.end = nf.lst.len();
      if (L_ >= 3 && L_ <= 6 && tk.sm[L_ - 3] > 1) slice_of(nf.lst.len(), tk.sj[L_ - 3], tk.sm[L_ - 3], nf.pos, nf.end);
      work += (nf.end - nf.pos + 31) / 32;
      if (smp) {
        (*smp)[L_ * 4 + 0] += 1;
        (*smp)[L_ * 4 + 1] += double(nf.end);
        (*smp)[L_ * 4 + 3] += double((nf.end + 31) / 32);
      }
      if (tk.bet && work > tk.budget) { // budget overloaded
        aborted = true;
        return 0;
      }
    }
    return cnt;
  }
  static thread_local uint64_t gate_rej_local, gate_tested_local;

  // S1 prediction for one plan
  float predict(int plan, uint64_t n2, float* raw = nullptr) const {
    const int k = p.k;
    const size_t b = size_t(plan) * k;
    const int lf = p.leaf[plan].from;
    double W = 0;
    for (int L_ = std::min(k - 1, lf); L_ >= 3; --L_) {
      const bool learned = L.e[b + L_] >= 8;
      double pc = p.prior_c[b + L_], ps = p.prior_s[b + L_];
      if (L_ == lf) {
        pc = 0;
        for (int z = lf; z < k; ++z) pc += p.prior_c[b + z];
        ps = 0;
      }
      const double c = learned ? std::max(1.0, L.c[b + L_] / L.e[b + L_]) : pc;
      const double sv = learned ? L.s[b + L_] / L.e[b + L_] : ps;
      W = c + sv * W;
    }
    const double r2 = L.cand[b + 2] >= 32 ? L.s[b + 2] / L.cand[b + 2] : p.prior_r[b + 2];
    double C = std::ceil(double(n2) / 32.0) + (k > 3 ? double(n2) * r2 * W : 0.0);
    const double kap = L.calP[plan] >= 8 ? std::clamp(L.calA[plan] / L.calP[plan], 1.0 / 64, 64.0) : 1.0;
    if (raw) *raw = static_cast<float>(std::max(1.0, C));
    return static_cast<float>(std::max(1.0, C * kap));
  }

  uint64_t s3_pieces(std::vector<Task>& T, double Bp, double W, double target, bool under) const {
    uint64_t nsplit = 0;
    std::vector<Task> pieces;
    pieces.reserve(T.size());
    for (const Task& tk : T) {
      double want = tk.cost > Bp ? std::ceil(tk.cost / Bp) : 1.0;
      if (under) want = std::max({want, std::floor(8.0 * target / double(T.size())), std::ceil(target * tk.cost / std::max(W, 1e-9))});
      want = std::min(want, 1024.0);
      if (tk.bet || tk.n2 < 1 || want < 2) {
        pieces.push_back(tk);
        continue;
      }
      const int b = int(tk.plan) * p.k;
      const int deep = std::min({p.k - 1, int(p.leaf[tk.plan].from) - 1, 6});
      auto est = [&](int L_) { return L.e[b + L_] >= 8 ? std::max(1.0, std::round(L.cand[b + L_] / L.e[b + L_])) : 8.0; };
      const uint32_t m2 = uint32_t(std::min<double>(want, double(tk.n2)));
      double r = std::ceil(want / m2);
      uint32_t m[4] = {1, 1, 1, 1}, deeper = 1;
      for (int L_ = 3; L_ <= deep && r > 1; ++L_) {
        const double cap = std::ceil(std::pow(r, 1.0 / double(deep - L_ + 1)));
        m[L_ - 3] = uint32_t(std::max(1.0, std::min({cap, est(L_), 255.0})));
        r = std::ceil(r / m[L_ - 3]);
        deeper *= m[L_ - 3];
      }
      const uint32_t np = m2 * deeper;
      if (np < 2) {
        pieces.push_back(tk);
        continue;
      }
      ++nsplit;
      for (uint32_t j = 0; j < np; ++j) {
        Task x = tk;
        uint32_t rest = j;
        for (int z = 3; z >= 0; --z) {
          x.sm[z] = uint8_t(m[z] > 1 ? m[z] : 0);
          x.sj[z] = uint8_t(m[z] > 1 ? rest % m[z] : 0);
          rest /= m[z];
        }
        const uint32_t j2 = rest;
        if (m2 > 1) {
          uint64_t lo, hi;
          slice_of(tk.n2, j2, m2, lo, hi);
          x.lo = lo;
          x.hi = j2 + 1 == m2 ? ~0ull : hi;
        }
        x.cost = tk.cost / float(np);
        pieces.push_back(x);
      }
    }
    T.swap(pieces);
    return nsplit;
  }

  uint64_t level2_len(int plan, const Update& u) const {
    if (p.k <= 2) return 1;
    vid_t phi[2] = {u.u, u.v};
    int piv;
    return pivot(plan, 2, phi, piv).len();
  }

  // S2 maintenance
  void s2_phase(const std::vector<vid_t>& E, ts_t lo, ts_t hi, bool defer = false) {
    if (E.empty()) return;
    std::vector<uint8_t> was_hub(E.size());
    std::vector<int> old_cls(E.size());
    for (size_t i = 0; i < E.size(); ++i) {
      was_hub[i] = s.hub[s.slot[E[i]]];
      old_cls[i] = sig::stair_class(s.deg(E[i]));
    }
#pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < E.size(); ++i) refresh_deg(g, s, E[i], lo, hi);
    if (defer) {
      if (any_gate) {
        std::vector<std::vector<vid_t>> nl(P);
#pragma omp parallel for schedule(dynamic, 64)
        for (size_t i = 0; i < E.size(); ++i) {
          auto& out = nl[omp_get_thread_num()];
          for_incident(g, s, E[i], lo, hi, [&](vid_t w, uint32_t, label_t) { out.push_back(w); });
        }
        for (auto& v : nl)
          for (vid_t w : v) {
            uint8_t& f = s.stale[s.slot[w]];
            if (!f) {
              f = 1;
              stale_l.push_back(w);
            }
          }
      }
#pragma omp parallel for schedule(dynamic, 16)
      for (size_t i = 0; i < E.size(); ++i) refresh_row(g, s, E[i], lo, hi);
      st.s2_rows += E.size();
      return;
    }
    // affected rows
    std::vector<std::vector<vid_t>> loc(P);
#pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < E.size(); ++i) {
      auto& out = loc[omp_get_thread_num()];
      out.push_back(E[i]);
      if (was_hub[i] && sig::stair_class(s.deg(E[i])) == old_cls[i]) continue;
      for_incident(g, s, E[i], lo, hi, [&](vid_t w, uint32_t, label_t) { out.push_back(w); });
    }
    std::vector<vid_t> A;
    for (auto& v : loc) A.insert(A.end(), v.begin(), v.end());
    psort(A.begin(), A.end());
    A.erase(std::unique(A.begin(), A.end()), A.end());
#pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < A.size(); ++i) {
      if (!std::binary_search(E.begin(), E.end(), A[i])) refresh_deg(g, s, A[i], lo, hi);
    }
#pragma omp parallel for schedule(dynamic, 16)
    for (size_t i = 0; i < A.size(); ++i) refresh_row(g, s, A[i], lo, hi);
    st.s2_rows += A.size();
    if (!stale_l.empty()) {
#pragma omp parallel for schedule(dynamic, 16)
      for (size_t i = 0; i < stale_l.size(); ++i)
        if (s.stale[s.slot[stale_l[i]]]) refresh_row(g, s, stale_l[i], lo, hi);
      st.s2_rows += stale_l.size();
      stale_l.clear();
    }
  }

  std::vector<vid_t> endpoints(const std::vector<Update>& B, const std::vector<uint8_t>& eff, bool del_only = false) const {
    std::vector<vid_t> E;
    for (size_t i = 0; i < B.size(); ++i)
      if (eff[i] && (!del_only || B[i].op == UpdateOp::DeleteEdge) && s.kept(g, B[i].u) && s.kept(g, B[i].v)) {
        E.push_back(B[i].u);
        E.push_back(B[i].v);
      }
    psort(E.begin(), E.end());
    E.erase(std::unique(E.begin(), E.end()), E.end());
    return E;
  }

  // one batch: apply, S2 maintenance, triage, S1, S3, search, control
  void process(const std::vector<Update>& B, ts_t t0, std::vector<uint64_t>& out) {
    const size_t nb = B.size();
    ++st.batches;
    ++batch_no;
    std::vector<uint8_t> eff(nb, 0);
    std::vector<ts_t> theta(nb);
    // ---- A. APPLY
    WallTimer ta;
    {
      std::vector<std::pair<uint64_t, uint32_t>> K(nb);
#pragma omp parallel for
      for (size_t i = 0; i < nb; ++i) K[i] = {canonical_edge(B[i].u, B[i].v, g.directed), uint32_t(i)};
      pstable(K.begin(), K.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
      std::vector<size_t> starts;
      for (size_t i = 0; i < nb; ++i)
        if (i == 0 || K[i].first != K[i - 1].first) starts.push_back(i);
      starts.push_back(nb);
      struct NewV {
        vid_t u, v;
        label_t el;
        ts_t ins, del;
      };
      std::vector<std::vector<NewV>> nv(P);
      std::vector<uint64_t> ci(P, 0), cd(P, 0), cn(P, 0), dead(P, 0);
      const size_t ngroups = starts.size() - 1;
#pragma omp parallel for schedule(dynamic, 256)
      for (size_t gi = 0; gi < ngroups; ++gi) {
        const int tid = omp_get_thread_num();
        const Update& f0 = B[K[starts[gi]].second];
        const vid_t cu = g.directed || f0.u < f0.v ? f0.u : f0.v, cv = g.directed || f0.u < f0.v ? f0.v : f0.u;
        // live version at batch start
        Side& S0 = g.s[0];
        Side& S1 = g.directed ? g.s[1] : g.s[0];
        auto base_pos = [&](Side& S, vid_t a, vid_t b) -> int64_t {
          const adj_t k = adj_key(g.vl[b], b);
          auto it = std::lower_bound(S.key.begin() + S.off[a], S.key.begin() + S.off[a + 1], k);
          return (it != S.key.begin() + S.off[a + 1] && *it == k) ? int64_t(it - S.key.begin()) : -1;
        };
        auto dpos = [&](Side& S, vid_t a, vid_t b) -> DEnt* {
          const adj_t k = adj_key(g.vl[b], b);
          for (DEnt& e : S.delta[a])
            if (e.key == k && e.del == kInf) return &e;
          return nullptr;
        };
        const int64_t bp = base_pos(S0, cu, cv);
        const bool base_live = bp >= 0 && S0.del[bp] == kInf;
        DEnt* dl = base_live ? nullptr : dpos(S0, cu, cv);
        int live = base_live ? 1 : (dl ? 2 : 0);  // 1 base, 2 old delta, 3 new version, 0 absent
        size_t newi = 0;
        for (size_t j = starts[gi]; j < starts[gi + 1]; ++j) {
          const uint32_t i = K[j].second;
          const ts_t t = t0 + i;
          const Update& up = B[i];
          theta[i] = up.op == UpdateOp::InsertEdge ? t : t - 1;
          if (up.op == UpdateOp::InsertEdge) {
            if (live) {
              ++cn[tid];
              continue;
            }
            nv[tid].push_back({cu, cv, up.el, t, kInf});
            newi = nv[tid].size() - 1;
            live = 3;
            eff[i] = 1;
            ++ci[tid];
          } else {
            if (!live) {
              ++cn[tid];
              continue;
            }
            if (live == 1) {
              S0.del[bp] = t;
              const int64_t mp = base_pos(S1, g.directed ? cv : cv, cu);  // mirror: in[v] (directed) or out[v]
              if (mp >= 0) S1.del[mp] = t;
              ++dead[tid];
            } else if (live == 2) {
              dl->del = t;
              DEnt* mir = dpos(S1, cv, cu);
              if (mir) mir->del = t;
            } else {
              nv[tid][newi].del = t;
            }
            live = 0;
            eff[i] = 1;
            ++cd[tid];
          }
        }
      }
      for (int t = 0; t < P; ++t) {
        st.eff_ins += ci[t];
        st.eff_del += cd[t];
        st.noop += cn[t];
        g.dead_base += dead[t];
      }
      // merge new versions into the per-vertex delta vectors (grouped by (side, vertex))
      std::vector<std::pair<uint64_t, DEnt>> ent;
      for (auto& v : nv)
        for (auto& x : v) {
          ent.push_back({(uint64_t(0) << 63) | x.u, DEnt{adj_key(g.vl[x.v], x.v), x.el, x.ins, x.del}});
          ent.push_back({(uint64_t(g.directed ? 1 : 0) << 63) | x.v, DEnt{adj_key(g.vl[x.u], x.u), x.el, x.ins, x.del}});
        }
      psort(ent.begin(), ent.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
      std::vector<size_t> gs;
      for (size_t i = 0; i < ent.size(); ++i)
        if (i == 0 || ent[i].first != ent[i - 1].first) gs.push_back(i);
      gs.push_back(ent.size());
      const size_t nvg = gs.size() - 1;
#pragma omp parallel for schedule(dynamic, 64)
      for (size_t gi = 0; gi < nvg; ++gi) {
        const uint64_t key = ent[gs[gi]].first;
        auto& D = g.s[key >> 63].delta[key & 0xFFFFFFFFu];
        for (size_t j = gs[gi]; j < gs[gi + 1]; ++j) D.push_back(ent[j].second);
        std::stable_sort(D.begin(), D.end(), [](const DEnt& a, const DEnt& b) { return a.key < b.key; });
      }
      g.delta_entries += ent.size();
    }
    st.apply_ms += ta.ms();
    const ts_t t1 = t0 + ts_t(nb) - 1;
    bool any_del = false;
    for (size_t i = 0; i < nb; ++i) any_del |= eff[i] && B[i].op == UpdateOp::DeleteEdge;

    // S2 maintenance
    WallTimer t2;
    const bool dormant = s2_dormant;
    size_t reentered = 0;
    {
      std::vector<vid_t> E = endpoints(B, eff);
      if (dormant) {
        for (vid_t v : E)
          if (!in_backlog[v]) {
            in_backlog[v] = 1;
            backlog.push_back(v);
          }
        ++st.s2_dormant_b;
      } else {
        if (!backlog.empty()) {  // exact re-entry of everything recorded while dormant
          reentered = backlog.size();
          E.insert(E.end(), backlog.begin(), backlog.end());
          for (vid_t v : backlog) in_backlog[v] = 0;
          backlog.clear();
          std::sort(E.begin(), E.end());
          E.erase(std::unique(E.begin(), E.end()), E.end());
          st.reentries += reentered;
        }
        s2_phase(E, t0, t1, !any_gate || batch_no < gate_off_until);
      }
    }
    const double s2_batch_ms = t2.ms();
    st.s2_ms += s2_batch_ms;

    // tasks + S2 triage
    WallTimer tt;
    const bool triage = !dormant;
    const bool probe = dormant && batch_no >= next_probe;
    const bool gate = !dormant && batch_no >= gate_off_until;
    const bool clo = batch_no >= clo_off_until && !p.creq.empty();
    std::vector<std::vector<Task>> tl(P);
    std::vector<uint64_t> an(P, 0), te(P, 0), rj(P, 0);
#pragma omp parallel for schedule(dynamic, 1024)
    for (size_t i = 0; i < nb; ++i) {
      if (!eff[i]) continue;
      const int tid = omp_get_thread_num();
      const Update& u = B[i];
      for (int a = 0; a < p.num_anchors; ++a) {
        const int pl = p.plan_off[a];
        const PlanLevel& l0 = lv(pl, 0);
        const PlanLevel& l1 = lv(pl, 1);
        if (g.vl[u.u] != l0.lq || g.vl[u.v] != l1.lq) continue;
        ++an[tid];
        if (triage || probe) {
          ++te[tid];
          const bool pass = dominates(s, u.u, l0.q) && dominates(s, u.v, l1.q);
          if (!pass) {
            ++rj[tid];
            if (triage) continue;  // probe: count only, keep the task
          }
        }
        Task tk{};
        tk.upd = uint32_t(i);
        tk.plan = uint16_t(pl);
        tk.hi = ~0ull;
        tl[tid].push_back(tk);
      }
    }
    std::vector<Task> T;
    uint64_t anchored = 0, tested = 0, rejected = 0;
    for (int t = 0; t < P; ++t) { // Anchor update only when timestamp matches
      T.insert(T.end(), tl[t].begin(), tl[t].end());
      anchored += an[t];
      tested += te[t];
      rejected += rj[t];
    }
    st.anchored += anchored;
    st.tested += triage ? tested : 0;
    st.rejected += triage ? rejected : 0;
    st.probe_rej += probe ? rejected : 0;
    st.triage_ms += tt.ms();

    // S1 PLAN
    WallTimer t1s;
    const bool s1_awake = batch_no >= s1_sleep_until;
    uint64_t sw = 0, ex = 0;
    double W = 0, M = 0;
    if (s1_awake) {
      std::vector<uint64_t> swl(P, 0), exl(P, 0);
#pragma omp parallel for schedule(dynamic, 256) reduction(+ : W) reduction(max : M)
      for (size_t i = 0; i < T.size(); ++i) {
        Task& tk = T[i];
        const Update& u = B[tk.upd];
        const int a = p.plan_anchor[tk.plan];
        const int ri = p.plan_off[a];
        tk.n2 = level2_len(ri, u);
        tk.cost = predict(ri, tk.n2, &tk.raw);
        const int np = p.plan_off[a + 1] - ri;
        if (np > 1 && p.k >= o.min_k) {
          int best = -1;
          float bc = tk.cost, br = tk.raw;
          for (int pl = ri + 1; pl < ri + np; ++pl) {
            if (!L.ready[pl]) continue;
            float rw;
            const float c = predict(pl, level2_len(pl, u), &rw);
            if (c < 0.7f * tk.cost && c < bc) {
              bc = c;
              br = rw;
              best = pl;
            }
          }
          if (best >= 0) {
            tk.plan = uint16_t(best);
            tk.bet = 1;
            tk.budget = std::max<uint64_t>(64, uint64_t(3.0f * bc));
            tk.cost = bc;
            tk.raw = br;
            ++swl[omp_get_thread_num()];
          } else if (o.explore > 0 && tk.cost <= 8.f &&
                     (mix_entry(batch_no * 1000003ull + i, 0xE7) % uint64_t(o.explore)) == 0) {
            tk.plan = uint16_t(ri + 1 + (i % (np - 1)));
            tk.bet = 1;
            tk.explore = 1;
            tk.budget = 64;
            ++exl[omp_get_thread_num()];
          }
          if (tk.plan != ri) tk.n2 = level2_len(tk.plan, u);
          if (tk.explore) tk.cost = predict(tk.plan, tk.n2, &tk.raw);
        }
        W += tk.cost;
        M = std::max<double>(M, tk.cost);
      }
      for (int t = 0; t < P; ++t) {
        sw += swl[t];
        ex += exl[t];
      }
      st.switched += sw;
      st.explored += ex;
    } else {
      ++st.s1_dormant_b;
      for (Task& tk : T) {
        tk.n2 = 0;
        tk.cost = 1;
      }
    }
    st.s1_ms += t1s.ms();

    // S3 Split
    WallTimer t3;
    const double Bp = std::max(64.0, W / (4.0 * P));
    const double target = 4.0 * P;
    const bool under = s1_awake && P > 1 && double(T.size()) < target && p.k >= 3 && M >= 64.0;
    const bool flat = !s1_awake || (M <= Bp && !under);
    if (flat) ++st.flat_b;
    if (s1_awake && !flat) st.split += s3_pieces(T, Bp, W, target, under);
    if (s1_awake && !flat)
      pstable(T.begin(), T.end(), [](const Task& a, const Task& b) { return a.cost > b.cost; });
    st.s3_ms += t3.ms();
    st.tasks += T.size();

    // S3 search: pieces heaviest first, taken dynamically by the threads
    WallTimer tf;
    std::vector<std::atomic<uint64_t>> cnt(nb);
    for (auto& c : cnt) c.store(0, std::memory_order_relaxed);
    const int k = p.k;
    std::vector<std::vector<double>> smp(P, std::vector<double>(size_t(p.num_plans) * k * 4, 0.0));
    std::vector<std::vector<double>> cal(P, std::vector<double>(size_t(p.num_plans) * 4, 0.0));
    std::vector<uint64_t> wk(P, 0), ab(P, 0), bw(P, 0), gr(P, 0), gt(P, 0), cf(P, 0);
    std::vector<double> cyc(P, 0.0), zcyc(P, 0.0);
    std::vector<uint64_t> zn(P, 0);
    std::vector<uint64_t> sp(P, 0);
#pragma omp parallel
    {
      const int tid = omp_get_thread_num();
      gate_rej_local = gate_tested_local = 0;
      std::vector<double> loc(size_t(k) * 4, 0.0);
#pragma omp for schedule(dynamic, 1)
      for (size_t i = 0; i < T.size(); ++i) {
        const uint64_t c0 = __rdtsc();
        if (c0 > g_deadline) g_tle.store(true, std::memory_order_relaxed);
        if (g_tle.load(std::memory_order_relaxed)) continue;
        const Task& tk = T[i];
        const Update& u = B[tk.upd];
        const ts_t th = theta[tk.upd];
        const int a = p.plan_anchor[tk.plan];
        if (clo && k >= 4 && !closure_ok(a, u.u, u.v, th)) {  // same verdict in every piece of a task
          if (tk.lo == 0 && !(tk.sj[0] | tk.sj[1] | tk.sj[2] | tk.sj[3])) ++cf[tid];
          const double dc = double(__rdtsc() - c0);
          cyc[tid] += dc;
          zcyc[tid] += dc;
          ++zn[tid];
          continue;
        }
        const bool sample = s1_awake && (mix_entry(i, batch_no) & 7) == 0 && tk.lo == 0 && tk.hi == ~0ull &&
                            !(tk.sm[0] | tk.sm[1] | tk.sm[2] | tk.sm[3]);
        if (sample) std::fill(loc.begin(), loc.end(), 0.0);
        uint64_t work = 0;
        bool aborted = false;
        uint64_t c = run_piece(tk, u, th, gate, work, aborted, sample ? &loc : nullptr);
        if (aborted) {  // bounded bet overran
          ++ab[tid];
          bw[tid] += work;
          Task ri = tk;
          ri.plan = p.plan_off[a];
          ri.bet = 0;
          ri.explore = 0;
          ri.lo = 0;
          ri.hi = ~0ull;
          ri.n2 = level2_len(ri.plan, u);
          ri.cost = predict(ri.plan, ri.n2, &ri.raw);
          std::vector<Task> rp{ri};
          if (k >= 3 && P > 1) sp[tid] += s3_pieces(rp, 64.0, ri.cost, 0.5 * P, true);
          for (const Task& x : rp) {
#pragma omp task firstprivate(x)
            {
              if (!g_tle.load(std::memory_order_relaxed)) {
                const int t2 = omp_get_thread_num();
                uint64_t w2 = 0;
                bool a2 = false;
                const uint64_t c2 = run_piece(x, B[x.upd], theta[x.upd], gate, w2, a2, nullptr);
                if (c2) cnt[x.upd].fetch_add(c2, std::memory_order_relaxed);
                wk[t2] += w2;
              }
            }
          }
          c = 0;
          cal[tid][size_t(tk.plan) * 4 + 3] += 1;
        }
        if (tk.bet) cal[tid][size_t(tk.plan) * 4 + 2] += 1;
        if (c) cnt[tk.upd].fetch_add(c, std::memory_order_relaxed);
        wk[tid] += work;
        if (sample && !aborted) {
          for (int L_ = 0; L_ < k; ++L_)
            for (int z = 0; z < 4; ++z) smp[tid][(size_t(tk.plan) * k + L_) * 4 + z] += loc[size_t(L_) * 4 + z];
          cal[tid][size_t(tk.plan) * 4 + 0] += double(work);
          cal[tid][size_t(tk.plan) * 4 + 1] += double(tk.raw);
        }
        const double dc = double(__rdtsc() - c0);
        cyc[tid] += dc;
        if (c == 0 && !aborted) {
          zcyc[tid] += dc;
          ++zn[tid];
        }
      }
      gr[tid] = gate_rej_local;
      gt[tid] = gate_tested_local;
    }
    const double search_ms = tf.ms();
    st.search_ms += search_ms;
    uint64_t work = 0, grej = 0, gtest = 0, cfail = 0, abort_n = 0;
    double cyc_sum = 0, zcyc_sum = 0;
    uint64_t zn_sum = 0;
    for (int t = 0; t < P; ++t) {
      zcyc_sum += zcyc[t];
      zn_sum += zn[t];
      work += wk[t];
      grej += gr[t];
      gtest += gt[t];
      cfail += cf[t];
      abort_n += ab[t];
      st.bet_waste += bw[t];
      cyc_sum += cyc[t];
    }
    for (int t = 0; t < P; ++t) st.split += sp[t];
    st.chunks += work;
    st.gate_rej += grej;
    st.gate_tested += gtest;
    st.clo_fail += cfail;
    st.aborts += abort_n;
    for (size_t i = 0; i < nb; ++i) out[t0 - 1 + i] = cnt[i].load(std::memory_order_relaxed);

    // Control
    const double decay = 0.5;
    if (s1_awake) {
      for (auto& x : L.e) x *= decay;
      for (auto& x : L.c) x *= decay;
      for (auto& x : L.s) x *= decay;
      for (auto& x : L.cand) x *= decay;
      for (int t = 0; t < P; ++t)
        for (int pl = 0; pl < p.num_plans; ++pl) {
          for (int L_ = 2; L_ < k; ++L_) {
            const double* z = &smp[t][(size_t(pl) * k + L_) * 4];
            L.e[size_t(pl) * k + L_] += z[0];
            L.cand[size_t(pl) * k + L_] += z[1];
            L.c[size_t(pl) * k + L_] += z[3];
            L.s[size_t(pl) * k + L_] += z[2];
          }
          L.calA[pl] = L.calA[pl] * (t == 0 ? decay : 1.0) + cal[t][size_t(pl) * 4 + 0];
          L.calP[pl] = L.calP[pl] * (t == 0 ? decay : 1.0) + cal[t][size_t(pl) * 4 + 1];
          L.bets[pl] += cal[t][size_t(pl) * 4 + 2];
          L.aborts[pl] += cal[t][size_t(pl) * 4 + 3];
        }
      for (int pl = 0; pl < p.num_plans; ++pl) {
        bool r = true;
        for (int L_ = 2; L_ <= std::min(k - 1, int(p.leaf[pl].from)); ++L_) r &= L.e[size_t(pl) * k + L_] >= 32;
        if (L.aborts[pl] > 0.05 * std::max(1.0, L.bets[pl])) r = false;
        L.ready[pl] = r;
      }
      // S1 dormancy: two flat batches in a row that are clearly flat or decision-free -> sleep 4 batches
      const bool idle = sw == 0 && ex == 0;
      flat_streak = (flat && (4.0 * M <= Bp || idle)) ? flat_streak + 1 : 0;
      if (flat_streak >= 2) {
        s1_sleep_until = batch_no + 1 + 4;
        flat_streak = 0;
      }
    }
    // S2 gate / closure off for the next batches when they reject < 0.2% of what they test
    if (gate && gtest >= 1000 && grej * 500 < gtest) gate_off_until = batch_no + 8;
    if (clo && anchored >= 1000 && cfail * 500 < anchored) clo_off_until = batch_no + 8;
    // S2 sleep / wake 
    {
      const double per_task = cyc_sum / cyc_per_ms / double(std::max<size_t>(1, T.size()));
      const double per_zero = zn_sum ? zcyc_sum / cyc_per_ms / double(zn_sum) : per_task;
      const double per_chunk = cyc_sum / cyc_per_ms / double(std::max<uint64_t>(1, work));
      const double s2_cpu = s2_batch_ms * P;
      if (dormant) {
        if (probe) {
          const double saving = double(rejected) * per_zero;
          const double rate = double(rejected) / double(std::max<uint64_t>(1, tested));
          last_probe_rate = rate;
          const double batch_cost = s2_cost_per_upd * double(nb);
          const double ep = reentry_per_ep >= 0 ? reentry_per_ep : 0.5 * std::max(0.0, s2_cost_per_upd);
          const double need = batch_cost + ep * double(backlog.size()) / 4.0;
          const bool evidence = tested >= 1000;
          const bool worth = s2_cost_per_upd >= 0 && evidence && saving * bias >= need;
          wake_streak = worth ? wake_streak + 1 : 0;
          probe_every = (evidence && s2_cost_per_upd >= 0 && saving * bias < 0.25 * need) ? std::min(16, 2 * probe_every) : 1;
          next_probe = batch_no + uint64_t(probe_every);
          if (wake_streak >= 2) {
            s2_dormant = false;
            wake_streak = 0;
            ++st.s2_wakes;
          }
        }
      } else if (reentered > 0) {  // the wake batch 
        const double regular = std::max(0.0, s2_cost_per_upd) * double(nb);
        const double ep = std::max(0.0, s2_cpu - regular) / double(reentered);
        reentry_per_ep = reentry_per_ep < 0 ? ep : 0.5 * reentry_per_ep + 0.5 * ep;
        if (last_probe_rate > 0 && tested >= 1000) {
          const double rho = std::clamp((double(rejected) / double(tested)) / last_probe_rate, 0.02, 1.0);
          bias = bias_known ? 0.5 * bias + 0.5 * rho : rho;
          bias_known = true;
        }
      } else if (nb) {
        const double c = s2_cpu / double(nb);
        s2_cost_per_upd = s2_cost_per_upd < 0 ? c : 0.5 * s2_cost_per_upd + 0.5 * c;
        if (tested >= 1000) {
          const double saving = double(rejected) * per_zero + double(grej) * per_chunk;
          if (saving < 0.5 * s2_cpu) {
            s2_dormant = true;
            wake_streak = 0;
            probe_every = 1;
            next_probe = batch_no + 1;
            ++st.s2_sleeps;
          }
        }
      }
    }
    // S2 end refresh 
    if (any_del && !dormant) {
      WallTimer te2;
      s2_phase(endpoints(B, eff, true), t1 + 1, t1 + 1, !any_gate || batch_no < gate_off_until);
      st.s2_ms += te2.ms();
    }
    maybe_compact(t1);
  }

  void maybe_compact(ts_t now) {
    if (double(g.delta_entries + g.dead_base) <= o.compact_ratio * double(std::max<uint64_t>(g.base_entries, 1024))) return;
    WallTimer tc;
    ++st.compactions;
    uint64_t total = 0;
    for (int d = 0; d < (g.directed ? 2 : 1); ++d) {
      Side& S = g.s[d];
      std::vector<eid_t> cntv(g.n + 1, 0);
#pragma omp parallel for schedule(dynamic, 1024)
      for (vid_t v = 0; v < g.n; ++v) {
        eid_t c = 0;
        for (eid_t i = S.off[v]; i < S.off[v + 1]; ++i) c += S.del[i] == kInf;
        for (const DEnt& e : S.delta[v]) c += e.del == kInf;
        cntv[v + 1] = c;
      }
      for (vid_t v = 0; v < g.n; ++v) cntv[v + 1] += cntv[v];
      std::vector<adj_t> nk(cntv[g.n]);
      std::vector<label_t> ne(S.el.empty() && !g.has_el ? 0 : cntv[g.n]);
#pragma omp parallel for schedule(dynamic, 1024)
      for (vid_t v = 0; v < g.n; ++v) {
        eid_t o2 = cntv[v], i = S.off[v];
        const eid_t ie = S.off[v + 1];
        const std::vector<DEnt>& D = S.delta[v];
        size_t j = 0;
        while (i < ie || j < D.size()) {
          if (i < ie && S.del[i] != kInf) {
            ++i;
            continue;
          }
          if (j < D.size() && D[j].del != kInf) {
            ++j;
            continue;
          }
          const bool fromb = j >= D.size() || (i < ie && S.key[i] < D[j].key);
          nk[o2] = fromb ? S.key[i] : D[j].key;
          if (!ne.empty()) ne[o2] = fromb ? (S.el.empty() ? kNoEdgeLabel : S.el[i]) : D[j].el;
          ++o2;
          if (fromb) ++i;
          else ++j;
        }
        if (!D.empty()) {
          S.delta[v].clear();
          S.delta[v].shrink_to_fit();
        }
      }
      S.off.swap(cntv);
      S.key.swap(nk);
      S.el.swap(ne);
      S.del.assign(S.key.size(), kInf);
      total += S.key.size();
    }
    g.base_entries = total;
    g.delta_entries = 0;
    g.dead_base = 0;
    (void)now;
    st.compact_ms += tc.ms();
  }
};
thread_local uint64_t Engine::gate_rej_local = 0;
thread_local uint64_t Engine::gate_tested_local = 0;

// Initial S2 rows 
void build_signatures(const Graph& g, Sig& s) {
  const size_t n = g.n;
  s.slot.assign(n, Sig::kNoSlot);
  size_t nk = 0;
  for (vid_t v = 0; v < n; ++v)
    if (s.kept(g, v)) s.slot[v] = static_cast<uint32_t>(nk++);
  s.rows.assign(std::max<size_t>(1, nk * s.L.words), 0);
  s.dout.assign(nk, 0);
  s.din.assign(nk, 0);
  s.prof.assign(std::max<size_t>(1, nk * s.L.nf), 0);
  s.mask.assign(nk, 0);
  s.hub.assign(nk, 0);
  s.stale.assign(nk, 0);
#pragma omp parallel for schedule(dynamic, 256)
  for (vid_t v = 0; v < n; ++v)
    if (s.kept(g, v)) refresh_deg(g, s, v, 0, 0);
#pragma omp parallel for schedule(dynamic, 64)
  for (vid_t v = 0; v < n; ++v)
    if (s.kept(g, v)) refresh_row(g, s, v, 0, 0);
}

void usage(const char* prog) {
  std::printf("usage: %s --graph FILE --stream FILE --query FILE [--directed] [--batch N] [--threads T] [--time-limit S]\n"
              "       [--name NAME] [--verbose]\n",
              prog);
}

// Short Display report
void rule() { std::printf("==========================================\n"); }
std::string dir_name(const std::string& p) {
  const size_t k = p.find_last_of('/');
  return k == std::string::npos ? "." : p.substr(0, k);
}
std::string base_name(const std::string& p) {
  const size_t k = p.find_last_of('/');
  return k == std::string::npos ? p : p.substr(k + 1);
}
// query file input
std::string short_path(const std::string& p, const std::string& graph_path) {
  const std::string d = dir_name(graph_path) + "/";
  return p.compare(0, d.size(), d) == 0 ? p.substr(d.size()) : base_name(p);
}
std::string dataset_name(const Opt& o) { return o.name.empty() ? base_name(dir_name(o.load.graph_path)) : o.name; }
int query_diameter(const QueryGraph& q) {  // Query diameter, longest shortest path in case of direction
  int diam = 0;
  for (int s = 0; s < q.k; ++s) {
    uint32_t seen = 1u << s, front = 1u << s;
    int d = 0;
    for (;;) {
      uint32_t next = 0;
      for (int u = 0; u < q.k; ++u)
        if (front >> u & 1u) next |= q.neighbors(u);
      next &= ~seen;
      if (!next) break;
      seen |= next;
      front = next;
      ++d;
    }
    diam = std::max(diam, d);
  }
  return diam;
}
int query_label_count(const QueryGraph& q) {
  int n = 0;
  for (int u = 0; u < q.k; ++u) {
    bool first = true;
    for (int w = 0; w < u; ++w) first &= q.label[w] != q.label[u];
    n += first;
  }
  return n;
}

}
}  // namespace csm

int main(int argc, char** argv) {
  using namespace csm;
  Opt o;
  for (int i = 1; i < argc; ++i) {
    auto need = [&](const char* f) -> std::string {
      if (i + 1 >= argc) fatal("%s needs a value", f);
      return argv[++i];
    };
    const char* a = argv[i];
    if (!std::strcmp(a, "--graph")) o.load.graph_path = need(a);
    else if (!std::strcmp(a, "--stream")) o.load.stream_path = need(a);
    else if (!std::strcmp(a, "--query")) o.load.query_path = need(a);
    else if (!std::strcmp(a, "--directed")) o.load.directed = true;
    else if (!std::strcmp(a, "--batch")) o.batch = std::stoull(need(a));
    else if (!std::strcmp(a, "--threads")) o.threads = std::stoi(need(a));
    else if (!std::strcmp(a, "--name")) o.name = need(a);
    else if (!std::strcmp(a, "--verbose")) o.verbose = true;
    else if (!std::strcmp(a, "--time-limit")) o.time_limit = std::stod(need(a));
    else if (!std::strcmp(a, "--help") || !std::strcmp(a, "-h")) {
      usage(argv[0]);
      return 0;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", a);
      usage(argv[0]);
      return 1;
    }
  }
  if (o.load.graph_path.empty() || o.load.stream_path.empty() || o.load.query_path.empty()) {
    usage(argv[0]);
    return 1;
  }
  if (o.batch == 0) fatal("--batch must be > 0");
  if (o.threads > 0) omp_set_num_threads(o.threads);
  const int P = omp_get_max_threads();

  if (!o.verbose) {
    rule();
    std::printf("TriStream (TriSnap, CPU)\n");
    rule();
    std::fflush(stdout);
  }
  // Simple times measurment for verbose, not actual cpu
  WallTimer t_all;
  Dataset data = load_dataset(o.load);
  const double load_ms = t_all.ms();
  const QueryGraph& q = data.query;
  if (!o.verbose) {
    std::printf("Dataset        : %s\n", dataset_name(o).c_str());
    std::printf("Nodes          : %u\n", data.graph.n_initial);
    std::printf("Edges          : %" PRIu64 " (%s)\n", data.graph_read.edge_lines,
                data.graph.directed ? "directed" : "undirected");
    std::printf("Query          : %s\n", short_path(o.load.query_path, o.load.graph_path).c_str());
    std::printf("Query details  : %d vertices | %zu edges | max degree %d | diameter %d | %d labels\n", q.k,
                q.edges.size(), q.max_degree(), query_diameter(q), query_label_count(q));
    std::printf("Stream         : %s\n", short_path(o.load.stream_path, o.load.graph_path).c_str());
    std::printf("Stream updates : %" PRIu64 " (%" PRIu64 " insertions, %" PRIu64 " deletions)\n", data.stream_orig_size,
                data.stream.stats.insertions, data.stream.stats.deletions);
    std::fflush(stdout);
  }
  WallTimer t_prep;
  EdgeStats es = compute_edge_stats(data.graph);
  MatchPlan plan = build_match_plan(q, &es, 3);
  Graph g = build_graph(data.graph);
  data.ids = IdMap(16);  // raw-id map
  Sig s;
  s.tau = o.hub_tau;
  s.use_el = q.has_edge_labels;
  {
    label_t nl = 0;
    for (vid_t v = 0; v < g.n; ++v) nl = std::max(nl, g.vl[v] + 1);
    for (int u = 0; u < q.k; ++u) nl = std::max(nl, q.label[u] + 1);
    s.keep.assign(nl, 0);
    for (int u = 0; u < q.k; ++u) s.keep[q.label[u]] = 1;
    s.keys = build_key_layout(query_adjacency(q), s.use_el, nl);
    s.L = s.keys.view();
  }
  compute_signatures(query_adjacency(q), s.use_el, s.L, s.q);
  build_signatures(g, s);
  const double prep_ms = t_prep.ms();

  Engine eng(o, q, plan, g, s);
  eng.P = P;
  // CPU time starts
  WallTimer t_stream;
  relevance_filter_stream(data, g.vl.data(), g.directed);
  const std::vector<Update>& U = data.stream.updates;
  const size_t N = U.size();
  std::vector<uint64_t> counts(N, 0);
  if (o.time_limit > 0) g_deadline = __rdtsc() + uint64_t(o.time_limit * 1000.0 * eng.cyc_per_ms);
  size_t processed = 0;
  for (size_t f = 0, bs = 2 * N > o.batch ? std::max<size_t>(1, o.batch / 16) : o.batch; f < N && !g_tle.load();
       f += bs, bs = std::min(o.batch, 2 * bs)) {
    const size_t nb = std::min(bs, N - f);
    std::vector<Update> B(U.begin() + f, U.begin() + f + nb);
    eng.process(B, ts_t(f + 1), counts);
    if (!g_tle.load()) processed = f + nb;
  }
  const bool tle = g_tle.load();
  const double stream_ms = t_stream.ms();
  const double e2e_ms = t_all.ms();  // whole run: parsing, preparation, S2 build, stream processing
  uint64_t pos = 0, neg = 0;
  for (size_t i = 0; i < N; ++i) (U[i].op == UpdateOp::InsertEdge ? pos : neg) += counts[i];

  const Stats& st = eng.st;
  const size_t orig = data.stream_orig_size ? data.stream_orig_size : N;
  if (!o.verbose) {  // the short report
    std::printf("Batches to CPU : %" PRIu64 " (batch size %zu; %zu of %zu updates can touch the query; %d threads)\n",
                st.batches, o.batch, N, orig, P);
    std::printf("Matches found  : %" PRIu64 " added, %" PRIu64 " deleted\n", pos, neg);
    if (tle)
      std::printf("Time limit     : %.1f s reached, %zu of %zu relevant updates processed (counts are lower bounds)\n",
                  o.time_limit, processed, N);
    std::printf("Time (CPU)     : %.2f ms\n", stream_ms);
    if (tle || stream_ms <= 0)
      std::printf("Speed          : n/a\n");
    else
      std::printf("Speed          : %.4e edges/sec\n", double(orig) / (stream_ms / 1000.0));
    std::printf("End-to-end time: %.2f ms\n", e2e_ms);
    rule();
    return 0;
  }
  std::printf("== TriSnap (CPU, %d threads) ==\n", P);
  std::printf("  dataset %s | V %u | E0 %" PRIu64 " | query k=%d m=%zu | updates %zu (%zu after relevance filter)\n",
              dataset_name(o).c_str(), g.n, data.graph.m, q.k, q.edges.size(), orig, N);
  std::printf("  batches %" PRIu64 " x %zu | matches ΔM+ %" PRIu64 " ΔM- %" PRIu64 "\n", st.batches, o.batch, pos, neg);
  std::printf("  time: load %.1f ms | prep %.1f ms | stream %.1f ms (relevance test %.1f, apply %.1f, S2 %.1f, triage %.1f,"
              " S1 %.1f, S3 %.1f, search %.1f, compact %.1f) | end-to-end %.1f ms\n",
              load_ms, prep_ms, stream_ms, data.stream_filter_ms, st.apply_ms, st.s2_ms, st.triage_ms, st.s1_ms, st.s3_ms, st.search_ms,
              st.compact_ms, e2e_ms);
  std::printf("  speed: %.3f M updates/s\n", double(orig) / (stream_ms / 1000.0) / 1e6);
  std::printf("  S2 keys: %d counters/vertex (slope %d, staircase %d, curvature %d pairs of %d x %d keys%s), %zu rows\n",
              s.L.words, s.L.n1, s.L.ns, s.L.np, s.L.nm, s.L.nf, s.keys.folded ? ", folded" : "", s.dout.size());
  if (tle) std::printf("  TIME LIMIT %.1f s reached after %zu of %zu updates: counts are incomplete\n", o.time_limit, processed, N);
  std::printf("RESULT system=TriSnap dataset=%s threads=%d directed=%d V=%u E0=%" PRIu64 " updates=%zu rel_updates=%zu"
              " ins=%" PRIu64 " del=%" PRIu64 " noop=%" PRIu64 " batch=%zu batches=%" PRIu64 " qk=%d qm=%zu"
              " dm_pos=%" PRIu64 " dm_neg=%" PRIu64 " load_ms=%.1f prep_ms=%.1f stream_ms=%.1f filter_ms=%.1f apply_ms=%.1f"
              " s2_ms=%.1f triage_ms=%.1f s1_ms=%.1f s3_ms=%.1f search_ms=%.1f compact_ms=%.1f e2e_ms=%.1f"
              " Mupd_per_s=%.3f anchored=%" PRIu64 " tested=%" PRIu64 " rejected=%" PRIu64
              " probe_rej=%" PRIu64 " gate_rej=%" PRIu64 " clo_fail=%" PRIu64 " tasks=%" PRIu64 " split=%" PRIu64
              " switched=%" PRIu64 " explored=%" PRIu64 " aborts=%" PRIu64 " bet_waste=%" PRIu64 " chunks=%" PRIu64
              " s2_rows=%" PRIu64 " s2_sleeps=%" PRIu64 " s2_wakes=%" PRIu64 " s2_dormant_b=%" PRIu64
              " reentries=%" PRIu64 " s1_dormant_b=%" PRIu64 " flat_b=%" PRIu64 " compactions=%" PRIu64
              " tle=%d processed=%zu\n",
              dataset_name(o).c_str(), P, g.directed ? 1 : 0, g.n, data.graph.m, orig, N, st.eff_ins, st.eff_del, st.noop,
              o.batch, st.batches, q.k, q.edges.size(), pos, neg, load_ms, prep_ms, stream_ms, data.stream_filter_ms, st.apply_ms, st.s2_ms,
              st.triage_ms, st.s1_ms, st.s3_ms, st.search_ms, st.compact_ms, e2e_ms,
              double(orig) / (stream_ms / 1000.0) / 1e6, st.anchored,
              st.tested, st.rejected, st.probe_rej, st.gate_rej, st.clo_fail, st.tasks, st.split, st.switched,
              st.explored, st.aborts, st.bet_waste, st.chunks, st.s2_rows, st.s2_sleeps, st.s2_wakes, st.s2_dormant_b,
              st.reentries, st.s1_dormant_b, st.flat_b, st.compactions, tle ? 1 : 0, processed);
  return 0;
}
