// the S2 dominance test and the candidate lists of the S3 search
#pragma once

#include <cstdint>

#include "csm/dynamic_graph.cuh"
#include "csm/match_plan.hpp"
#include "csm/signature.hpp"

namespace csm {
namespace mdev {

using ull = unsigned long long;
constexpr unsigned kFull = 0xFFFFFFFFu;

// S2 stats
struct S2Args {
  int triage, closure, gate;
  const uint32_t* rows;
  const uint32_t* degout;
  const uint32_t* degin;
  const uint32_t* mask;
  const uint8_t* hub;
  const uint32_t* slot;  // vertex -> CSR adjacency list
  const uint32_t* qrows;
  const uint32_t* qdo;
  const uint32_t* qdi;
  const ClosureReq* creq;
  const uint16_t* creq_off;
  const uint8_t* gates;
  int triage_probe;  // S2 dormant
  sig::Layout lay;
};

// w's signature dominate query vertex q's
__device__ __forceinline__ bool s2_ok(const S2Args& a, int q, vid_t w) {
  const uint32_t k = a.slot[w];
  if (k == 0xFFFFFFFFu) return true;  // no row (cannot happen for a label-matching candidate)
  return sig::dominates(a.lay, a.qrows + q * a.lay.words, a.qdo[q], a.qdi[q],
                        a.rows + static_cast<uint64_t>(k) * a.lay.words, a.degout[k], a.degin[k], a.mask[k], a.hub[k]);
}

__device__ __forceinline__ const BaseSideView& base_of(const DynGraphView& g, uint32_t side) {
  return side ? g.in_base : g.out_base;
}
__device__ __forceinline__ const DeltaSideView& delta_of(const DynGraphView& g, uint32_t side) {
  return side ? g.in_delta : g.out_delta;
}
__device__ __forceinline__ bool label_ok(label_t want, label_t have) { return want == kAnyEdgeLabel || want == have; }

__device__ inline bool edge_visible(const DynGraphView& g, uint32_t side, vid_t x, vid_t y, ts_t tp, label_t el) {
  const BaseSideView& b = base_of(g, side);
  const eid_t p = base_find(b, g.vlabel, x, y);
  if (p != kNotFound && b.del[p] > tp && label_ok(el, b.el ? b.el[p] : kNoEdgeLabel)) return true;
  const DeltaSideView& d = delta_of(g, side);
  if (d.size == 0) return false;
  const uint64_t key = pack_sk(x, y);
  for (uint64_t i = delta_lower_bound(d, key); i < d.size && d.sk[i] == key; ++i)
    if (d.ins[i] <= tp && tp < d.del[i] && label_ok(el, d.el[i])) return true;
  return false;
}

// Candidate list of one level
struct Cur {
  eid_t bb, be;
  uint64_t db, de;
  uint64_t pos;   // next chunk start (virtual index)
  uint32_t mask;  // validated, not yet explored candidates of the current chunk
  uint32_t side;  // 0 out-list, 1 in-list
  uint32_t piv;   // pivot check (relative to the level's first check)
};

__device__ __forceinline__ uint64_t cur_len(const Cur& c) { return (c.be - c.bb) + (c.de - c.db); }

__device__ __forceinline__ void list_of(const DynGraphView& g, uint32_t side, vid_t x, label_t lq, eid_t& bb,
                                        eid_t& be, uint64_t& db, uint64_t& de) {
  const BaseSideView& b = base_of(g, side);
  const eid_t s = b.offsets[x], e = b.offsets[x + 1];
  bb = lower_bound_key(b.keys, s, e, adj_key(lq, 0));
  be = lower_bound_key(b.keys, bb, e, adj_key(lq + 1, 0));
  const DeltaSideView& d = delta_of(g, side);
  if (d.size == 0) {
    db = de = 0;
    return;
  }
  db = delta_lower_bound(d, pack_sk(x, 0));
  de = delta_lower_bound(d, pack_sk(x + 1, 0));
}

// Reads virtual list entry idx
__device__ __forceinline__ bool read_cand(const DynGraphView& g, const Cur& c, uint64_t idx, label_t lq, ts_t tp,
                                          label_t pel, vid_t& w) {
  const uint64_t nb = c.be - c.bb;
  if (idx < nb) {
    const BaseSideView& b = base_of(g, c.side);
    const eid_t p = c.bb + idx;
    w = adj_vertex(b.keys[p]);
    return b.del[p] > tp && label_ok(pel, b.el ? b.el[p] : kNoEdgeLabel);
  }
  const DeltaSideView& d = delta_of(g, c.side);
  const uint64_t i = c.db + (idx - nb);
  w = static_cast<vid_t>(d.sk[i] & 0xFFFFFFFFu);
  return d.ins[i] <= tp && tp < d.del[i] && g.vlabel[w] == lq && label_ok(pel, d.el[i]);
}

} 
}  // namespace csm
