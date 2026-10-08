#include "csm/matcher.cuh"
#include "csm/match_device.cuh"

#include <cub/cub.cuh>

#include <algorithm>
#include <cmath>

namespace csm {
namespace {

using namespace mdev;
constexpr int kMatchBlock = 256;

// ctr_ layout
enum : int {
  kMcNext = 0, kMcNTasks,  // per batch: task-list cursor and length
  kMcTasks, kMcLabel, kMcS2, kMcAnchored, kMcCloFail, kMcCloChunks, kMcGateRej, kMcChunks, kMcTimeouts, kMcPos, kMcNeg,
  kMcGateTested, kMcExplore, kMcK0, kMcK1, kMcK2, kMcB0, kMcB1, kMcB2,
  kMcProbeAbort, kMcSwitched, kMcLocal,
  kMcBind, kMcWaste, kMcAbortCnt, kMcMaxCyc, kMcSumCyc,
  kMcDead, kMcProbeRej, kMcRerunN, kMcZeroCyc, kMcZeroN,
  kMcNum = 48
};
constexpr uint64_t kFullRange = 0xFFFFFFFFull;  // piece range word lo << 32 | hi; hi = 0xFFFFFFFF: open end
constexpr int kHist = 64;                       // log2-ratio histogram: bucket = floor(2·log2 r) + 32
constexpr ull kProbeBit = 1ull << 63;     // task word flag: a bet (any deviation from RI; bounded, abort-and-fallback)
constexpr ull kExploreBit = 1ull << 62;   // ... that is an exploration probe: its budget is exactly the probe budget
constexpr ull kTaskMask = kExploreBit - 1;  // update * plans + plan
// ns clock shared by all SMs (per-query time limit)
__device__ __forceinline__ ull gtime() {
  ull t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}


__host__ __device__ __forceinline__ void slice_of(uint64_t len, uint32_t j, uint32_t m, uint64_t& lo, uint64_t& hi) {
  const uint64_t units = (len + 31) / 32;
  if (units >= m) {
    lo = (j * units / m) * 32;
    hi = j + 1 == m ? len : ((j + 1) * units / m) * 32;
  } else {
    lo = j * len / m;
    hi = (j + 1) * len / m;
  }
  lo = lo < len ? lo : len;
  hi = hi < len ? hi : len;
}

// S1 Stats
struct S1Args {
  int plans, k, explore, min_k;
  const float* kappa;    // calibration factor per (plan, magnitude bucket): actual / raw predicted chunks, learned
  uint32_t* len2;        // out: level-2 list length of the chosen plan (S3 split input)
  unsigned salt;
  const uint16_t* plan_off;
  const uint16_t* plan_anchor;
  const uint8_t* plan_kind;
  const PlanLevel* levels;
  const PlanCheck* checks;
  const float* rc;       // learned chunks per entry, per (plan, level)
  const float* rs;       // learned survivors per entry, per (plan, level)
  const float* rr;       // learned survival probability of one candidate, per (plan, level)
  const uint8_t* ready;  // per plan: its learned rates rest on enough observations to be trusted
  const uint32_t* rows;  // S2 signature rows (nullptr without S2): local cardinality sketch
  const uint32_t* mask;
  const uint8_t* hub;
  const uint32_t* slot;  // vertex -> compact signature row
  sig::Layout lay;
  const LeafPlan* leaf;
};

// Guard rails
constexpr float kSwitchMargin = 0.7f;
constexpr float kExploreMaxCost = 8.f;
constexpr uint64_t kExploreMaxLen2 = 64;

// S1 cost model: predicted DFS chunks of running plan p for anchor (u,v) using mathematical fixed formula
// An estimate only: it never affects which matches are found.
__device__ float s1_predict(const DynGraphView& g, const S1Args& a, int p, vid_t u, vid_t v,
                            uint64_t* len2_out = nullptr, bool* local_out = nullptr) {
  const int k = a.k;
  if (len2_out) *len2_out = 0;
  if (local_out) *local_out = false;
  if (k == 2) return 0.f;
  const int base = p * k;
  const PlanLevel lv2 = a.levels[base + 2];
  uint64_t n2 = ~0ull;
  int piv = 0;
  for (int c = 0; c < lv2.nchk; ++c) {
    const PlanCheck ch = a.checks[lv2.cb + c];
    eid_t bb, be;
    uint64_t db, de;
    list_of(g, ch.dir, ch.pos == 0 ? u : v, lv2.lq, bb, be, db, de);
    const uint64_t len = static_cast<uint64_t>((be - bb) + (de - db));
    if (len < n2) {
      n2 = len;
      piv = c;
    }
  }
  if (len2_out) *len2_out = n2;
  const float c2 = ceilf(static_cast<float>(n2) / 32.f);
  if (k == 3) return c2;
  float l3 = -1.f;  // < 0: no local estimate
  const PlanLevel lv3 = a.levels[base + 3];
  const PlanCheck p2 = a.checks[lv2.cb + piv];
  const vid_t x = p2.pos == 0 ? u : v;  // the endpoint whose list feeds level 2
  for (int c = 0; c < lv3.nchk; ++c) {
    const PlanCheck ch = a.checks[lv3.cb + c];
    float est = -1.f;
    if (ch.pos < 2) {
      eid_t bb, be;
      uint64_t db, de;
      list_of(g, ch.dir, ch.pos == 0 ? u : v, lv3.lq, bb, be, db, de);
      est = static_cast<float>((be - bb) + (de - db));
    } else if (ch.pos == 2 && a.rows && n2 > 0 && a.slot[x] != 0xFFFFFFFFu && !a.hub[a.slot[x]]) {
      const int grp = a.lay.mid_of(lv2.lq, p2.dir);
      const int pr = a.lay.pair_of(grp, a.lay.far_of(lv3.lq, ch.dir));
      const uint64_t kx = a.slot[x];
      if (pr >= 0 && !((a.mask[kx] >> grp) & 1u)) {
        const uint32_t K = a.rows[kx * a.lay.words + a.lay.n1 + a.lay.ns + pr];
        est = static_cast<float>(K) / static_cast<float>(n2);
      }
    }
    if (est >= 0.f && (l3 < 0.f || est < l3)) l3 = est;
  }
  const int lf = a.leaf ? static_cast<int>(a.leaf[p].from) : k;
  const int top = min(k - 1, lf);
  float T3;
  if (l3 >= 0.f && local_out) *local_out = true;
  if (lf == 3) {
    T3 = a.rc[base + 3];
  } else if (l3 >= 0.f) {
    T3 = ceilf(l3 / 32.f);
    if (top > 3) {
      float T = a.rc[base + top];
      for (int L = top - 1; L >= 4; --L) T = a.rc[base + L] + a.rs[base + L] * T;
      T3 += l3 * a.rr[base + 3] * T;
    }
  } else {
    T3 = a.rc[base + top];
    for (int L = top - 1; L >= 3; --L) T3 = a.rc[base + L] + a.rs[base + L] * T3;
  }
  return c2 + static_cast<float>(n2) * a.rr[base + 2] * T3;
}

// S1: one thread per surviving task
__global__ void k_s1(DynGraphView g, DeviceStreamView s, size_t first, S1Args a, ull n, ull* list, float* cost,
                     ull* ctr) {
  ull nk[3] = {0, 0, 0}, nb[3] = {0, 0, 0}, nexp = 0, nsw = 0, nloc = 0;
  const ull P = static_cast<ull>(a.plans);
  for (ull i = blockIdx.x * static_cast<ull>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<ull>(gridDim.x) * blockDim.x) {
    const ull t = list[i];
    const ull upd = t / P;
    const int p0 = static_cast<int>(t % P);
    const int an = a.plan_anchor[p0];
    const int pb = a.plan_off[an], np = a.plan_off[an + 1] - pb;
    const size_t j = first + upd;
    const vid_t u = s.u[j], v = s.v[j];
    int bp = pb;  // RI: the default plan
    bool loc = false;
    auto calibrated = [&](int p, float raw) { return a.kappa[p] * raw; };
    const float c_ri = calibrated(pb, s1_predict(g, a, pb, u, v, nullptr, &loc));
    nloc += loc;
    float best = c_ri;
    // a switch is only considered where a better order can pay (k >= min_k) 
    const bool eligible = a.k >= a.min_k;
    for (int p = pb + 1; eligible && p < pb + np; ++p) {
      if (!a.ready[p]) continue;
      bool lp = false;
      const float c = calibrated(p, s1_predict(g, a, p, u, v, nullptr, &lp));
      if (!(loc && lp)) continue;
      if (c < kSwitchMargin * best) {
        best = c;
        bp = p;
      }
    }
    nsw += bp != pb;
    const uint32_t h = sig::mix32(t ^ (static_cast<uint64_t>(a.salt) << 40));
    const ull nexp_before = nexp;
    if (eligible && np > 1 && a.explore > 0 && h % a.explore == 0 && c_ri <= kExploreMaxCost) {
      const int ep = pb + 1 + static_cast<int>((h / a.explore) % (np - 1));
      uint64_t len2 = 0;
      s1_predict(g, a, ep, u, v, &len2);
      if (len2 <= kExploreMaxLen2) {
        bp = ep;
        ++nexp;
      }
    }
    const bool explored = bp != pb && nexp_before != nexp;
    uint64_t len2 = 0;
    const float raw = s1_predict(g, a, bp, u, v, &len2);
    best = a.kappa[bp] * raw;
    list[i] = (upd * P + bp) |
              (bp != pb ? kProbeBit : 0ull) |  // every deviation is a bounded bet
              (explored ? kExploreBit : 0ull);
    cost[i] = best;
    if (a.len2) a.len2[i] = len2 > 0xFFFFFFFFull ? 0xFFFFFFFFu : static_cast<uint32_t>(len2);
    ++nk[a.plan_kind[bp]];
    ++nb[best < 2.f ? 0 : (best < 64.f ? 1 : 2)];
  }
  for (int x = 0; x < 3; ++x) {
    if (nk[x]) atomicAdd(&ctr[kMcK0 + x], nk[x]);
    if (nb[x]) atomicAdd(&ctr[kMcB0 + x], nb[x]);
  }
  if (nexp) atomicAdd(&ctr[kMcExplore], nexp);
  if (nsw) atomicAdd(&ctr[kMcSwitched], nsw);
  if (nloc) atomicAdd(&ctr[kMcLocal], nloc);
}


// Warp-uniform
__device__ void init_level(const DynGraphView& g, const PlanLevel& lv, const PlanCheck* checks, vid_t my_m,
                           int lane, Cur& cur) {
  uint64_t best_len = ~0ull;
  uint32_t best_c = 0;
  for (int c0 = 0; c0 < lv.nchk; c0 += 32) {
    const int c = c0 + lane;
    const bool has = c < lv.nchk;
    const PlanCheck ch = has ? checks[lv.cb + c] : PlanCheck{0, 0, 0, 0};
    const vid_t x = __shfl_sync(kFull, my_m, ch.pos);
    uint64_t len = ~0ull;
    if (has) {
      eid_t bb, be;
      uint64_t db, de;
      list_of(g, ch.dir, x, lv.lq, bb, be, db, de);
      len = (be - bb) + (de - db);
    }
    uint32_t cc = static_cast<uint32_t>(c);
    for (int off = 16; off; off >>= 1) {
      const uint64_t ol = __shfl_xor_sync(kFull, len, off);
      const uint32_t oc = __shfl_xor_sync(kFull, cc, off);
      if (ol < len || (ol == len && oc < cc)) {
        len = ol;
        cc = oc;
      }
    }
    if (len < best_len) {
      best_len = len;
      best_c = cc;
    }
  }
  const PlanCheck ch = checks[lv.cb + best_c];
  const vid_t x = __shfl_sync(kFull, my_m, ch.pos);
  list_of(g, ch.dir, x, lv.lq, cur.bb, cur.be, cur.db, cur.de);
  cur.pos = 0;
  cur.mask = 0;
  cur.side = ch.dir;
  cur.piv = best_c;
}

// S2 edge closure (warp-uniform)
__device__ bool closure_ok(const DynGraphView& g, const S2Args& sa, int a, vid_t u, vid_t v, ts_t tp, int lane,
                           ull& chunks) {
  const int rb = sa.creq_off[a], nr = sa.creq_off[a + 1] - rb;
  if (nr == 0) return true;
  const uint32_t ku = sa.slot[u], kv = sa.slot[v];  // no row -> degree 0: only the scan side / cap change (sound)
  const uint32_t du = ku == 0xFFFFFFFFu ? 0 : sa.degout[ku] + sa.degin[ku];
  const uint32_t dv = kv == 0xFFFFFFFFu ? 0 : sa.degout[kv] + sa.degin[kv];
  const bool xu = du <= dv;
  const vid_t x = xu ? u : v, y = xu ? v : u;
  const uint32_t my_target = lane < nr ? sa.creq[rb + lane].target : 0;
  uint32_t have = 0;
  const int sides = g.directed ? 2 : 1;
  for (int side = 0; side < sides; ++side) {
    for (int r0 = 0; r0 < nr; ++r0) {
      const label_t l = sa.creq[rb + r0].l;
      if (r0 > 0 && sa.creq[rb + r0 - 1].l == l) continue;  // requirements are sorted by label
      Cur c;
      list_of(g, side, x, l, c.bb, c.be, c.db, c.de);
      c.side = side;
      c.pos = 0;
      c.mask = 0;
      c.piv = 0;
      const uint64_t len = cur_len(c);
      for (uint64_t pos = 0; pos < len; pos += 32) {
        ++chunks;
        const uint64_t idx = pos + lane;
        vid_t w = kInvalidVid;
        bool ok = idx < len && read_cand(g, c, idx, l, tp, kAnyEdgeLabel, w) && w != y;
        uint32_t hx = 0, hy = 0;  // bit 0: endpoint -> w, bit 1: w -> endpoint
        if (ok) {
          if (!g.directed) {
            hx = 1;
            hy = edge_visible(g, 0, y, w, tp, kAnyEdgeLabel) ? 1u : 0u;
          } else {
            if (side == 0) {
              hx = 1u | (edge_visible(g, 1, x, w, tp, kAnyEdgeLabel) ? 2u : 0u);
            } else {
              ok = !edge_visible(g, 0, x, w, tp, kAnyEdgeLabel);  // x->w too: already counted on side 0
              hx = 2u;
            }
            if (ok)
              hy = (edge_visible(g, 0, y, w, tp, kAnyEdgeLabel) ? 1u : 0u) |
                   (edge_visible(g, 1, y, w, tp, kAnyEdgeLabel) ? 2u : 0u);
          }
        }
        for (int r = 0; r < nr; ++r) {
          const ClosureReq q = sa.creq[rb + r];
          const uint32_t nx = xu ? q.need0 : q.need1, ny = xu ? q.need1 : q.need0;
          const bool hit = ok && q.l == l && (hx & nx) == nx && (hy & ny) == ny;
          const uint32_t n = __popc(__ballot_sync(kFull, hit));
          if (lane == r) have += n;
        }
        if (__ballot_sync(kFull, lane < nr && have < my_target) == 0) return true;
      }
    }
  }
  return __ballot_sync(kFull, lane < nr && have < my_target) == 0;
}

// Leaf counting (warp-uniform)
__device__ ull leaf_count_warp(const DynGraphView& g, const LeafPlan& lp, vid_t my_m, vid_t wL, ts_t tp, int lane,
                               ull& work) {
  const int L = lp.from - 1;
  const uint32_t full = 1u << lp.ng;
  ull cm = 0;  // lane mm (< full) holds the number of distinct candidates whose membership mask is mm
  int pvs[kMaxLeaves], ord[kMaxLeaves];
  uint64_t bls[kMaxLeaves];
  for (int i = 0; i < lp.ng; ++i) {
    int pv = 0;
    uint64_t bl = ~0ull;
    for (int y = 0; y < lp.gnc[i]; ++y) {
      const vid_t py = lp.gpos[i][y] == L ? wL : __shfl_sync(kFull, my_m, lp.gpos[i][y]);
      eid_t bb, be;
      uint64_t db, de;
      list_of(g, lp.gdir[i][y], py, lp.glab[i], bb, be, db, de);
      const uint64_t ly = static_cast<uint64_t>((be - bb) + (de - db));
      if (ly < bl) {
        bl = ly;
        pv = y;
      }
    }
    if (bl < lp.gmul[i]) return 0;
    pvs[i] = pv;
    bls[i] = bl;
    int x = i;
    for (; x > 0 && bls[ord[x - 1]] > bl; --x) ord[x] = ord[x - 1];
    ord[x] = i;
  }
  for (int oi = 0; oi < lp.ng; ++oi) {
    const int i = ord[oi], pv = pvs[i];
    uint32_t nv = 0;
    const vid_t pi = lp.gpos[i][pv] == L ? wL : __shfl_sync(kFull, my_m, lp.gpos[i][pv]);
    Cur c;
    list_of(g, lp.gdir[i][pv], pi, lp.glab[i], c.bb, c.be, c.db, c.de);
    c.side = lp.gdir[i][pv];
    c.pos = 0;
    c.mask = 0;
    c.piv = 0;
    const uint64_t len = cur_len(c);
    for (uint64_t pos = 0; pos < len; pos += 32) {
      ++work;
      const uint64_t idx = pos + lane;
      vid_t w = kInvalidVid;
      bool ok = idx < len && read_cand(g, c, idx, lp.glab[i], tp, lp.gel[i][pv], w);
      for (int j = 0; j < L; ++j) ok &= (w != __shfl_sync(kFull, my_m, j));  // injective with the prefix
      ok &= (w != wL);
      for (int y = 0; y < lp.gnc[i]; ++y) {
        if (y == pv) continue;
        const vid_t py = lp.gpos[i][y] == L ? wL : __shfl_sync(kFull, my_m, lp.gpos[i][y]);
        if (ok) ok = edge_visible(g, lp.gdir[i][y], py, w, tp, lp.gel[i][y]);
      }
      uint32_t m = 1u << i;
      bool lowest = true;
      for (int j = 0; j < lp.ng; ++j) {
        if (j == i || lp.glab[j] != lp.glab[i]) continue;
        bool in = ok;
        for (int y = 0; y < lp.gnc[j]; ++y) {
          const vid_t py = lp.gpos[j][y] == L ? wL : __shfl_sync(kFull, my_m, lp.gpos[j][y]);
          if (in) in = edge_visible(g, lp.gdir[j][y], py, w, tp, lp.gel[j][y]);
        }
        if (in) {
          m |= 1u << j;
          if (j < i) lowest = false;
        }
      }
      const bool take = ok && lowest;
      nv += __popc(__ballot_sync(kFull, ok));
      for (uint32_t mm = 1; mm < full; ++mm) {
        const uint32_t n = __popc(__ballot_sync(kFull, take && m == mm));
        if (lane == static_cast<int>(mm)) cm += n;
      }
    }
    if (nv < lp.gmul[i]) return 0;
  }
  if (__ballot_sync(kFull, cm != 0) == 0) return 0;
  ull cnt[1u << kMaxLeaves];
  for (uint32_t mm = 0; mm < full; ++mm) cnt[mm] = __shfl_sync(kFull, cm, mm);
  ull tot = 0;
  if (lane == 0) {
    ull I[1u << kMaxLeaves];
    for (uint32_t G = 0; G < full; ++G) {
      I[G] = 0;
      for (uint32_t M = G; M < full; ++M)
        if ((M & G) == G) I[G] += cnt[M];
    }
    for (int t = 0; t < lp.nterm; ++t) {
      ull v = static_cast<ull>(static_cast<long long>(lp.coef[t]));
      for (int b = 0; b < lp.nblk[t]; ++b) v *= I[lp.bmask[t][b]];
      tot += v;
    }
  }
  return __shfl_sync(kFull, tot, 0);
}

// Phase 3: one thread per (update, anchor)
__global__ void k_triage(DeviceStreamView s, size_t first, uint32_t count, const uint8_t* __restrict__ eff,
                         const label_t* __restrict__ vlabel, const PlanLevel* __restrict__ levels, int k, int anchors,
                         const uint16_t* __restrict__ plan_off, int plans, S2Args sa, ull* ctr, ull* list) {
  const int lane = threadIdx.x & 31;
  const ull total = static_cast<ull>(count) * anchors;
  const ull stride = static_cast<ull>(gridDim.x) * blockDim.x;
  ull n_tasks = 0, n_label = 0, n_s2 = 0, n_probe = 0;
  for (ull t0 = static_cast<ull>(blockIdx.x) * blockDim.x + (threadIdx.x & ~31u); t0 < total; t0 += stride) {
    const ull t = t0 + lane;
    bool keep = false;
    ull out = 0;
    if (t < total) {
      const uint32_t upd = static_cast<uint32_t>(t / anchors);
      const int a = static_cast<int>(t % anchors);
      const int p0 = plan_off[a];
      out = static_cast<ull>(upd) * plans + p0;
      if (eff[upd]) {
        ++n_tasks;
        const vid_t u = s.u[first + upd], v = s.v[first + upd];
        const PlanLevel l0 = levels[p0 * k], l1 = levels[p0 * k + 1];
        if (vlabel[u] == l0.lq && vlabel[v] == l1.lq) {
          ++n_label;
          const bool s2ok = !(sa.triage || sa.triage_probe) || (s2_ok(sa, l0.q, u) && s2_ok(sa, l1.q, v));
          keep = sa.triage ? s2ok : true;  // a probe (dormant S2) never prunes
          n_probe += sa.triage_probe && !s2ok;
          n_s2 += keep;
        }
      }
    }
    const unsigned m = __ballot_sync(kFull, keep);
    ull base = 0;
    if (lane == 0 && m) base = atomicAdd(&ctr[kMcNTasks], static_cast<ull>(__popc(m)));
    base = __shfl_sync(kFull, base, 0);
    if (keep) list[base + __popc(m & ((1u << lane) - 1u))] = out;
  }
  if (n_tasks) atomicAdd(&ctr[kMcTasks], n_tasks);
  if (n_label) atomicAdd(&ctr[kMcLabel], n_label);
  if (n_s2) atomicAdd(&ctr[kMcS2], n_s2);
  if (n_probe) atomicAdd(&ctr[kMcProbeRej], n_probe);
}

__global__ void __launch_bounds__(kMatchBlock)
    k_match(DynGraphView g, DeviceStreamView s, size_t first, const ull* __restrict__ list,
            const PlanLevel* __restrict__ levels, const PlanCheck* __restrict__ checks,
            const LeafPlan* __restrict__ leafs, int k, int plans,
            const uint16_t* __restrict__ plan_anchor, const uint16_t* __restrict__ plan_off, S2Args sa,
            ull* ctr, ull* counts, uint8_t* tmo, ull* lstat, ull* probe_stat,
            ull probe_budget, float bet_scale, const float* __restrict__ cost, const uint64_t* __restrict__ range,
            const uint64_t* __restrict__ slice, ull* rerun_list,
            const float* __restrict__ kappa, double* calib, ull* hist, ull deadline_rel) {
  const int lane = threadIdx.x & 31;
  const ull total = ctr[kMcNTasks];
  ull n_anchored = 0, n_clofail = 0, n_clochunks = 0, n_gate = 0, n_gtest = 0, n_chunks = 0, n_tmo = 0, n_pos = 0,
      n_neg = 0, n_bind = 0, n_waste = 0, n_abort_cnt = 0, cyc_sum = 0, cyc_max = 0, n_dead = 0, zcyc = 0, zn = 0;
  // per-query time limit: every warp works until its own copy of the deadline (0 = none)
  ull t_dead = ~0ull;
  if (deadline_rel) {
    ull now = lane == 0 ? gtime() : 0;
    t_dead = __shfl_sync(kFull, now, 0) + deadline_rel;
  }

  for (;;) {
    ull ti = 0;
    if (lane == 0) ti = atomicAdd(&ctr[kMcNext], 1ull);
    ti = __shfl_sync(kFull, ti, 0);
    if (ti >= total) break;
    const ull word = list[ti];
    bool probe = (word & kProbeBit) != 0;
    const ull task = word & kTaskMask;
    // a model-driven switch for bettting
    ull my_budget =
        !probe ? 0ull
               : ((word & kExploreBit) ? probe_budget
                                       : max(probe_budget, cost ? static_cast<ull>(ceilf(bet_scale * cost[ti])) : 0ull));
    const uint32_t upd = static_cast<uint32_t>(task / plans);
    int plan = static_cast<int>(task % plans);
    const int a = plan_anchor[plan];
    const long long t_start = clock64();
    bool rerun = false;  // this attempt re-runs an aborted bet with RI
    if (__shfl_sync(kFull, lane == 0 ? (gtime() > t_dead ? 1 : 0) : 0, 0)) {  // query time limit reached
      if (lane == 0) tmo[static_cast<uint32_t>(task / plans)] = 1;
      ++n_dead;
      continue;
    }
  attempt:
    ull le = 0, lc = 0, ls = 0, ll = 0;  // S1 learning
    // S3 piece
    const int L0 = 2;
    const uint64_t rw = range ? range[ti] : kFullRange;
    const uint64_t r_lo = rw >> 32, r_hi = (rw & kFullRange) == kFullRange ? ~0ull : (rw & kFullRange);
    const uint64_t sl = slice ? slice[ti] : 0ull;
    const bool first_piece = r_lo == 0 && (sl & 0x00FF00FF00FF00FFull) == 0;

    const size_t j = first + upd;
    const ts_t t = static_cast<ts_t>(j + 1);
    const bool is_del = s.op[j] == static_cast<uint8_t>(UpdateOp::DeleteEdge);
    const ts_t tp = is_del ? t - 1 : t;  // deletion: count in G_{t-1}; insertion: in G_t
    const vid_t u = s.u[j], v = s.v[j];
    const PlanLevel* lvs = levels + plan * k;
    const int lfrom = leafs ? static_cast<int>(leafs[plan].from) : k;  // leaf counting from this level (k: none)

    // Anchor checks
    const PlanLevel l1 = lvs[1];
    bool ok = true;
    for (int c = 0; ok && c < l1.nchk; ++c) {
      const PlanCheck ch = checks[l1.cb + c];
      ok = edge_visible(g, ch.dir, u, v, tp, ch.el);
    }
    if (!ok) continue;
    n_anchored += first_piece;  // a split task is counted once
    // same verdict in every piece
    if (sa.closure && first_piece && !closure_ok(g, sa, a, u, v, tp, lane, n_clochunks)) {
      n_clofail += first_piece;
      zcyc += static_cast<ull>(clock64() - t_start);
      ++zn;
      continue;
    }

    ull cnt = 0, work = 0;
    bool aborted = false, dead = false;
    if (k == 2) {
      cnt = 1;
    } else {
      vid_t my_m = lane == 0 ? u : (lane == 1 ? v : 0); 
      eid_t s_bb = 0, s_be = 0;     // saved Cur of level `lane`
      uint64_t s_db = 0, s_de = 0, s_pos = 0, s_end = 0;
      uint64_t cend = r_hi;
      uint32_t s_mask = 0, s_side = 0, s_piv = 0;

      int L = L0;
      PlanLevel lv = lvs[L0];
      Cur cur;
      init_level(g, lv, checks, my_m, lane, cur);
      cur.pos = r_lo;  // piece: start of its range at level L0 (0 for a whole task)
      if (lane == L) {
        ++le;
        ll += cur_len(cur);
      }
      label_t pel = checks[lv.cb + cur.piv].el;
      for (;;) {
        if (cur.mask == 0) {
          const uint64_t cl = cur_len(cur);
          const uint64_t len = cend < cl ? cend : cl;  // piece: end of its range at level L0
          if (cur.pos >= len) {  // level exhausted -> backtrack
            if (L == L0) break;
            --L;
            cur.bb = __shfl_sync(kFull, s_bb, L);
            cur.be = __shfl_sync(kFull, s_be, L);
            cur.db = __shfl_sync(kFull, s_db, L);
            cur.de = __shfl_sync(kFull, s_de, L);
            cur.pos = __shfl_sync(kFull, s_pos, L);
            cur.mask = __shfl_sync(kFull, s_mask, L);
            cur.side = __shfl_sync(kFull, s_side, L);
            cur.piv = __shfl_sync(kFull, s_piv, L);
            cend = __shfl_sync(kFull, s_end, L);
            lv = lvs[L];
            pel = checks[lv.cb + cur.piv].el;
            continue;
          }
          if (++work > my_budget && my_budget) {
            aborted = true;
            break;
          }
          if ((work & 15) == 0 && t_dead != ~0ull &&
              __shfl_sync(kFull, lane == 0 ? (gtime() > t_dead ? 1 : 0) : 0, 0)) {  // query time limit
            aborted = dead = true;
            break;
          }
          // scan one chunk: lane i validates list entry pos + i
          const uint64_t idx = cur.pos + lane;
          vid_t w = kInvalidVid;
          bool good = idx < len && read_cand(g, cur, idx, lv.lq, tp, pel, w);
          for (int m = 0; m < L; ++m) good &= (w != __shfl_sync(kFull, my_m, m));  // injectivity
          for (int c = 0; c < lv.nchk; ++c) {
            if (c == static_cast<int>(cur.piv)) continue;
            const PlanCheck ch = checks[lv.cb + c];
            const vid_t x = __shfl_sync(kFull, my_m, ch.pos);
            if (good) good = edge_visible(g, ch.dir, x, w, tp, ch.el);
          }
          uint32_t mask = __ballot_sync(kFull, good);
          if (sa.gate && sa.gates[plan * k + L]) {  // S2 gate: can w host q_L's remaining neighbourhood?
            if (good) good = s2_ok(sa, lv.q, w);
            const uint32_t gated = __ballot_sync(kFull, good);
            n_gtest += __popc(mask);
            n_gate += __popc(mask & ~gated);
            mask = gated;
          }
          if (lane == L) {
            ++lc;
            ls += __popc(mask);
          }
          cur.pos += 32;
          if (L == k - 1) {
            cnt += __popc(mask);  // last level: count, never descend
          } else if (L == lfrom - 1) {  // only leaves remain: count their injective assignments per candidate
            uint32_t rest = mask;
            while (rest) {
              const int bit = __ffs(rest) - 1;
              rest &= rest - 1;
              vid_t wl = kInvalidVid;
              read_cand(g, cur, cur.pos - 32 + bit, lv.lq, tp, pel, wl);
              const ull w0 = work;
              cnt += leaf_count_warp(g, leafs[plan], my_m, wl, tp, lane, work);
              if (lane == lfrom) {
                ++le;
                lc += work - w0;
                ll += (work - w0) * 32;
              }
            }
          } else {
            cur.mask = mask;
            n_bind += __popc(mask);  // intermediate partial matches that will be extended
          }
          continue;
        }
        // descend into the lowest remaining candidate of the chunk
        const int bit = __ffs(cur.mask) - 1;
        cur.mask &= cur.mask - 1;
        vid_t w = kInvalidVid;
        read_cand(g, cur, cur.pos - 32 + bit, lv.lq, tp, pel, w);
        if (lane == L) {
          s_bb = cur.bb;
          s_be = cur.be;
          s_db = cur.db;
          s_de = cur.de;
          s_pos = cur.pos;
          s_mask = cur.mask;
          s_side = cur.side;
          s_piv = cur.piv;
          s_end = cend;
          my_m = w;
        }
        ++L;
        lv = lvs[L];
        init_level(g, lv, checks, my_m, lane, cur);
        cend = ~0ull;
        if (L >= 3 && L <= 6 && ((sl >> (16 * (L - 3) + 8)) & 255u) > 1) {
          uint64_t slo, shi;
          slice_of(cur_len(cur), static_cast<uint32_t>((sl >> (16 * (L - 3))) & 255u),
                   static_cast<uint32_t>((sl >> (16 * (L - 3) + 8)) & 255u), slo, shi);
          cur.pos = slo;
          cend = shi;
        }
        if (lane == L) {
          ++le;
          ll += cur_len(cur);
        }
        pel = checks[lv.cb + cur.piv].el;
      }
    }
    n_chunks += work;
    if (probe && lane == 0) atomicAdd(&probe_stat[2 * plan], 1ull);  // probes run with this plan
    if (probe && aborted && !dead) {
      // bounded regret
      n_waste += work;
      n_abort_cnt += cnt != 0;
      if (lane == 0) {
        atomicAdd(&probe_stat[2 * plan + 1], 1ull);
        atomicAdd(&ctr[kMcProbeAbort], 1ull);
      }
      if (rerun_list) {
        if (lane == 0) rerun_list[atomicAdd(&ctr[kMcRerunN], 1ull)] = static_cast<ull>(upd) * plans + plan_off[a];
        continue;
      }
      plan = plan_off[a];
      probe = false;
      my_budget = 0;
      rerun = true;
      goto attempt;
    }
    {  // task time (all attempts)
      const ull dt = static_cast<ull>(clock64() - t_start);
      cyc_sum += dt;
      cyc_max = max(cyc_max, dt);
      if (cnt == 0 && !aborted) {
        zcyc += dt;
        ++zn;
      }
    }
    if (lane == 0) {
      if (cnt) atomicAdd(&counts[upd], cnt);
      if (aborted) tmo[upd] = 1;
      if (calib && cost && !aborted && !rerun && (ti & 7) == 0) {  // S1 calibration + error distribution (1/8)
        const float c = cost[ti];
        atomicAdd(&calib[2 * plan], static_cast<double>(work));
        atomicAdd(&calib[2 * plan + 1], static_cast<double>(c / fmaxf(kappa[plan], 1e-6f)));
        const float r = (static_cast<float>(work) + 1.f) / (c + 1.f);
        const int b = min(kHist - 1, max(0, static_cast<int>(floorf(2.f * log2f(r))) + kHist / 2));
        atomicAdd(&hist[b], 1ull);
      }
    }
    if (rw != kFullRange && lane == 2) le = 0;  // only part of level 2 seen: no level-2 learning from it
    if (lane >= 3 && lane <= 6 && ((sl >> (16 * (lane - 3) + 8)) & 255u) > 1) le = 0;
    if (lstat && (ti & 7) == 0 && lane >= 2 && lane < k && le) {  // 1/8 sample of tasks feeds S1
      ull* st = lstat + (static_cast<ull>(plan) * k + lane) * 4;
      atomicAdd(st + 0, le);
      atomicAdd(st + 1, lc);
      atomicAdd(st + 2, ls);
      atomicAdd(st + 3, ll);
    }
    n_tmo += aborted;
    n_dead += dead;
    if (is_del) n_neg += cnt;
    else n_pos += cnt;
  }
  if (lane == 0) {
    if (n_dead) atomicAdd(&ctr[kMcDead], n_dead);
    if (n_anchored) atomicAdd(&ctr[kMcAnchored], n_anchored);
    if (n_clofail) atomicAdd(&ctr[kMcCloFail], n_clofail);
    if (n_clochunks) atomicAdd(&ctr[kMcCloChunks], n_clochunks);
    if (n_gate) atomicAdd(&ctr[kMcGateRej], n_gate);
    if (n_gtest) atomicAdd(&ctr[kMcGateTested], n_gtest);
    if (n_chunks) atomicAdd(&ctr[kMcChunks], n_chunks);
    if (n_tmo) atomicAdd(&ctr[kMcTimeouts], n_tmo);
    if (n_pos) atomicAdd(&ctr[kMcPos], n_pos);
    if (n_neg) atomicAdd(&ctr[kMcNeg], n_neg);
    if (n_bind) atomicAdd(&ctr[kMcBind], n_bind);
    if (n_waste) atomicAdd(&ctr[kMcWaste], n_waste);
    if (n_abort_cnt) atomicAdd(&ctr[kMcAbortCnt], n_abort_cnt);
    if (cyc_sum) atomicAdd(&ctr[kMcSumCyc], cyc_sum);
    if (cyc_max) atomicMax(&ctr[kMcMaxCyc], cyc_max);
    if (zcyc) atomicAdd(&ctr[kMcZeroCyc], zcyc);
    if (zn) atomicAdd(&ctr[kMcZeroN], zn);
  }
}

// S3 counting
__global__ void k_s3_count(const ull* list, const float* cost, const uint32_t* len2, ull n, float B,
                           int max_pieces, float W, float target, int k, int plans, const LeafPlan* leaf,
                           const float* rl, uint64_t* np, uint64_t* shape) {
  const ull i = blockIdx.x * static_cast<ull>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  uint32_t m2 = 1, mz[4] = {1, 1, 1, 1};
  const ull word = list[i];
  const uint32_t n2 = len2[i];
  if (!(word & kProbeBit) && n2 >= 1) {
    float want = cost[i] > B ? ceilf(cost[i] / B) : 1.f;
    if (static_cast<float>(n) < target)
      want = fmaxf(want, fmaxf(floorf(8.f * target / static_cast<float>(n)), ceilf(target * cost[i] / fmaxf(W, 1e-6f))));
    want = fminf(want, static_cast<float>(max_pieces));
    if (want >= 2.f) {
      const int p = static_cast<int>((word & kTaskMask) % static_cast<ull>(plans));
      const int deep = min(min(k - 1, static_cast<int>(leaf[p].from) - 1), 6);
      m2 = static_cast<uint32_t>(fminf(want, static_cast<float>(n2)));
      float r = ceilf(want / static_cast<float>(m2));
      for (int L = 3; L <= deep && r > 1.f; ++L) {
        const float cap = ceilf(powf(r, 1.f / static_cast<float>(deep - L + 1)));
        mz[L - 3] = static_cast<uint32_t>(fmaxf(1.f, fminf(fminf(cap, fmaxf(1.f, rintf(rl[p * k + L]))), 255.f)));
        r = ceilf(r / static_cast<float>(mz[L - 3]));
      }
    }
  }
  np[i] = static_cast<uint64_t>(m2) * mz[0] * mz[1] * mz[2] * mz[3];
  shape[i] = static_cast<uint64_t>(m2) | (static_cast<uint64_t>(mz[0]) << 16) | (static_cast<uint64_t>(mz[1]) << 24) |
             (static_cast<uint64_t>(mz[2]) << 32) | (static_cast<uint64_t>(mz[3]) << 40);
}

// S3 filling
__global__ void k_s3_fill(const ull* list, const float* cost, const uint32_t* len2, const uint64_t* np,
                          const uint64_t* shape, const uint64_t* off, ull n, ull* out_task, float* out_cost,
                          uint64_t* out_range, uint64_t* out_slice, uint32_t* out_idx) {
  const ull i = blockIdx.x * static_cast<ull>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  const uint32_t P = static_cast<uint32_t>(np[i]), n2 = len2[i];
  const uint64_t sh = shape[i];
  const uint32_t m2 = static_cast<uint32_t>(sh & 0xFFFFu);
  const uint32_t mz[4] = {static_cast<uint32_t>((sh >> 16) & 255u), static_cast<uint32_t>((sh >> 24) & 255u),
                          static_cast<uint32_t>((sh >> 32) & 255u), static_cast<uint32_t>((sh >> 40) & 255u)};
  const uint64_t o = off[i];
  for (uint32_t j = 0; j < P; ++j) {
    out_task[o + j] = list[i];
    out_cost[o + j] = cost[i] / static_cast<float>(P);
    out_idx[o + j] = static_cast<uint32_t>(o + j);
    uint32_t rest = j;
    uint64_t sw = 0;
    for (int z = 3; z >= 0; --z)
      if (mz[z] > 1) {
        sw |= (static_cast<uint64_t>(mz[z]) << (16 * z + 8)) | (static_cast<uint64_t>(rest % mz[z]) << (16 * z));
        rest /= mz[z];
      }
    const uint32_t j2 = rest;
    if (m2 == 1) {
      out_range[o + j] = kFullRange;
    } else {
      uint64_t lo, hi;
      slice_of(n2, j2, m2, lo, hi);
      out_range[o + j] = (lo << 32) | (j2 + 1 == m2 ? kFullRange : hi);
    }
    out_slice[o + j] = sw;
  }
}

__global__ void k_s3_gather(const uint32_t* idx, ull n, const ull* task_in, const uint64_t* range_in,
                            const uint64_t* slice_in, ull* task_out, uint64_t* range_out, uint64_t* slice_out) {
  const ull i = blockIdx.x * static_cast<ull>(blockDim.x) + threadIdx.x;
  if (i >= n) return;
  task_out[i] = task_in[idx[i]];
  range_out[i] = range_in[idx[i]];
  slice_out[i] = slice_in[idx[i]];
}

}  // namespace

Matcher::Matcher(const MatchPlan& plan, const S1Config& s1, const S3Config& s3)
    : k_(plan.k), anchors_(plan.num_anchors), plans_(plan.num_plans), s1_(s1), s3_(s3) {
  if (k_ < 2 || k_ > kMaxQueryVertices) fatal("matcher: query must have 2..%d vertices", kMaxQueryVertices);
  levels_.upload(plan.levels.data(), plan.levels.size());
  checks_.upload(plan.checks.data(), plan.checks.size());
  leaf_.upload(plan.leaf.data(), plan.leaf.size());
  creq_.upload(plan.creq.data(), plan.creq.size());
  creq_off_.upload(plan.creq_off.data(), plan.creq_off.size());
  gate_.upload(plan.gate.data(), plan.gate.size());
  any_gate_ = std::find(plan.gate.begin(), plan.gate.end(), 1) != plan.gate.end();
  plan_off_.upload(plan.plan_off.data(), plan.plan_off.size());
  plan_anchor_.upload(plan.plan_anchor.data(), plan.plan_anchor.size());
  plan_kind_.upload(plan.plan_kind.data(), plan.plan_kind.size());
  prior_c_ = plan.prior_c;
  prior_s_ = plan.prior_s;
  prior_r_ = plan.prior_r;
  lfrom_.resize(plans_);
  for (int p = 0; p < plans_; ++p) {
    lfrom_[p] = plan.leaf[p].from;
    if (lfrom_[p] < k_) {
      float c = 0.f;
      for (int z = lfrom_[p]; z < k_; ++z) c += plan.prior_c[static_cast<size_t>(p) * k_ + z];
      prior_c_[static_cast<size_t>(p) * k_ + lfrom_[p]] = c;
      prior_s_[static_cast<size_t>(p) * k_ + lfrom_[p]] = 0.f;
    }
  }
  const size_t pk = static_cast<size_t>(plans_) * k_;
  dE_.assign(pk, 0.0);
  dC_.assign(pk, 0.0);
  dS_.assign(pk, 0.0);
  dLen_.assign(pk, 0.0);
  rate_c_.upload(prior_c_.data(), pk);
  rate_s_.upload(prior_s_.data(), pk);
  rate_r_.upload(prior_r_.data(), pk);
  {
    const std::vector<float> l32(pk, 8.f);
    rate_l_.upload(l32.data(), pk);
  }
  ready_.resize(plans_);
  ready_.fill_bytes(0);  // nothing is trusted before the first observations
  probe_stat_.resize(2 * static_cast<size_t>(plans_));
  probe_stat_.fill_bytes(0);
  dProbe_.assign(plans_, 0.0);
  dAbort_.assign(plans_, 0.0);
  lstat_.resize(pk * 4);
  lstat_.fill_bytes(0);
  ctr_.resize(kMcNum);
  ctr_.fill_bytes(0);
  const std::vector<float> ones(static_cast<size_t>(plans_), 1.f);
  kappa_.upload(ones.data(), ones.size());
  calib_.resize(2 * ones.size());
  calib_.fill_bytes(0);
  dCalA_.assign(ones.size(), 0.0);
  dCalP_.assign(ones.size(), 0.0);
  hist_.resize(kHist);
  hist_.fill_bytes(0);
  int dev = 0, sms = 0, per_sm = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz_, cudaDevAttrClockRate, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, k_match, kMatchBlock, 0));
  grid_ = sms * std::max(per_sm, 1);
}

bool Matcher::s2_needed_next() const { return batch_no_ >= s2_sleep_until_; }

// S2 triage
void Matcher::end_batch_s2(double s2_ms, uint64_t reentry_eps, uint64_t backlog_eps) {
  if (!b_s2_used_) return;
  // triage's saving in this batch: (tasks it rejected - or, while dormant, WOULD have rejected on the stale rows,
  // measured by the count-only probe) x the batch's GPU time per task
  const double per_task = b_kernel_ms_ / static_cast<double>(std::max<ull>(1, b_anchored_ + (b_probe_ ? 0 : b_rejected_)));
  const double per_zero = (b_zn_ && b_cyc_) ? b_kernel_ms_ * (static_cast<double>(b_zcyc_) / static_cast<double>(b_cyc_)) /
                                                  static_cast<double>(b_zn_)
                                            : per_task;
  const double saving = static_cast<double>(b_rejected_) * per_zero;
  const double rate = static_cast<double>(b_rejected_) / static_cast<double>(std::max<ull>(1, b_tested_));
  if (b_probe_) {
    // dormant: wake only when the probe's saving
    constexpr double kWakeHorizon = 4.0;
    last_probe_rate_ = rate;
    const double batch_cost = s2_ms_per_update_ * static_cast<double>(b_count_);
    const double per_ep = reentry_ms_per_ep_ >= 0.0 ? reentry_ms_per_ep_ : 0.5 * s2_ms_per_update_;  // prior
    const double reentry = per_ep * static_cast<double>(backlog_eps);
    if (!b_probe_ran_) {  // backed-off batch
      s2_sleep_until_ = batch_no_ + 1;
      ++host_.s2_probe_skips;
      return;
    }
    const double need = batch_cost + reentry / kWakeHorizon;
    const bool evidence = b_tested_ >= s1_.s2_min_tested;
    const bool worth = s2_cost_known_ && evidence && saving * probe_bias_ >= need;
    wake_streak_ = worth ? wake_streak_ + 1 : 0;
    // probe back-off 
    probe_every_ = (evidence && s2_cost_known_ && saving * probe_bias_ < 0.25 * need) ? std::min(16, 2 * probe_every_) : 1;
    next_probe_batch_ = batch_no_ + static_cast<uint64_t>(probe_every_) - 1;
    if (wake_streak_ >= 2) {
      s2_sleep_until_ = batch_no_;  // awake from the next batch on
      wake_streak_ = 0;
      ++host_.s2_wakes;
    } else {
      s2_sleep_until_ = batch_no_ + 1;  // keep sleeping
    }
    return;
  }
  if (reentry_eps > 0) {
    // first awake batch after a sleep.
    const double regular = s2_ms_per_update_ * static_cast<double>(b_count_);
    const double ep = std::max(0.0, s2_ms - regular) / static_cast<double>(reentry_eps);
    reentry_ms_per_ep_ = reentry_ms_per_ep_ < 0.0 ? ep : 0.5 * reentry_ms_per_ep_ + 0.5 * ep;
    if (last_probe_rate_ > 0.0 && b_tested_ >= s1_.s2_min_tested) {
      const double rho = std::min(1.0, std::max(0.02, rate / last_probe_rate_));
      probe_bias_ = bias_known_ ? 0.5 * probe_bias_ + 0.5 * rho : rho;  // the first measurement replaces the prior 1
      bias_known_ = true;
    }
    return;
  }
  if (b_count_) {
    const double c = s2_ms / static_cast<double>(b_count_);
    s2_ms_per_update_ = s2_cost_known_ ? 0.5 * s2_ms_per_update_ + 0.5 * c : c;
    s2_cost_known_ = true;
  }
  if (b_tested_ < s1_.s2_min_tested) return;
  // every ROW-READING use counts
  const double per_chunk = b_kernel_ms_ / static_cast<double>(std::max<ull>(1, b_chunks_));
  const double saving_all = saving + static_cast<double>(b_gate_rej_) * per_chunk;
  if (saving_all < 0.5 * s2_ms) {
    s2_sleep_until_ = batch_no_ + 1;  // asleep from the next batch on; the probes decide when to wake
    wake_streak_ = 0;
    probe_every_ = 1;                 // the first dormant batch probes
    next_probe_batch_ = batch_no_;
    ++host_.s2_sleeps;
  }
}

bool Matcher::gate_active_next() const {
  // a plan set without any gated level (e.g. every 3-vertex query) never reads a non-endpoint row
  return any_gate_ && batch_no_ >= gate_off_until_;
}

void Matcher::set_s2(const SigTable& q, const sig::Layout& lay) {
  lay_ = lay;
  qrows_.upload(q.rows.data(), q.rows.size());
  qdo_.upload(q.degout.data(), q.degout.size());
  qdi_.upload(q.degin.data(), q.degin.size());
}

// S1 learning
void Matcher::learn() {
  const size_t pk = static_cast<size_t>(plans_) * k_;
  std::vector<ull> b(pk * 4);
  CUDA_CHECK(cudaMemcpy(b.data(), lstat_.data(), b.size() * sizeof(ull), cudaMemcpyDeviceToHost));
  lstat_.fill_bytes(0);
  const double lam = s1_.decay;
  std::vector<float> rc(pk), rs(pk), rr(pk), rl(pk);
  for (size_t x = 0; x < pk; ++x) {  // decayed sums: D <- λ·D + (this batch)
    dE_[x] = lam * dE_[x] + static_cast<double>(b[4 * x + 0]);
    dC_[x] = lam * dC_[x] + static_cast<double>(b[4 * x + 1]);
    dS_[x] = lam * dS_[x] + static_cast<double>(b[4 * x + 2]);
    dLen_[x] = lam * dLen_[x] + static_cast<double>(b[4 * x + 3]);
    rc[x] = dE_[x] > 0 ? static_cast<float>(dC_[x] / dE_[x]) : prior_c_[x];      // chunks per entry
    rs[x] = dE_[x] > 0 ? static_cast<float>(dS_[x] / dE_[x]) : prior_s_[x];      // survivors per entry
    rr[x] = dLen_[x] > 0 ? static_cast<float>(dS_[x] / dLen_[x]) : prior_r_[x];  // survival probability
    rl[x] = dE_[x] >= 8.0 ? static_cast<float>(dLen_[x] / dE_[x]) : 8.f;
  }
  rate_l_.upload(rl.data(), pk);
  rate_c_.upload(rc.data(), pk);
  rate_s_.upload(rs.data(), pk);
  rate_r_.upload(rr.data(), pk);
  // a plan is trusted once every one of its search levels has >= 32 
  std::vector<ull> ps(2 * static_cast<size_t>(plans_));
  CUDA_CHECK(cudaMemcpy(ps.data(), probe_stat_.data(), ps.size() * sizeof(ull), cudaMemcpyDeviceToHost));
  probe_stat_.fill_bytes(0);
  std::vector<uint8_t> ready(plans_, 1);
  for (int p = 0; p < plans_; ++p) {
    dProbe_[p] = lam * dProbe_[p] + static_cast<double>(ps[2 * p]);
    dAbort_[p] = lam * dAbort_[p] + static_cast<double>(ps[2 * p + 1]);
    for (int L = 2; L <= std::min(k_ - 1, lfrom_[p]); ++L)
      if (dE_[static_cast<size_t>(p) * k_ + L] < 32.0) ready[p] = 0;
    if (dAbort_[p] > 0.05 * dProbe_[p]) ready[p] = 0;
  }
  ready_.upload(ready.data(), ready.size());
  // calibration needed for next batch. 
  std::vector<double> cb(2 * static_cast<size_t>(plans_));
  CUDA_CHECK(cudaMemcpy(cb.data(), calib_.data(), cb.size() * sizeof(double), cudaMemcpyDeviceToHost));
  calib_.fill_bytes(0);
  std::vector<float> kap(plans_);
  auto clampk = [](double x) { return static_cast<float>(std::min(64.0, std::max(1.0 / 64, x))); };
  for (int p = 0; p < plans_; ++p) {
    dCalA_[p] = lam * dCalA_[p] + cb[2 * p];
    dCalP_[p] = lam * dCalP_[p] + cb[2 * p + 1];
    kap[p] = dCalP_[p] >= 8.0 ? clampk(dCalA_[p] / dCalP_[p]) : 1.f;
  }
  kappa_.upload(kap.data(), kap.size());
}

// W and max of the predicted costs of the first n tasks (one sync)
void Matcher::cost_stats(uint64_t n, float& W, float& M) {
  size_t bytes = 0;
  wsum_.resize(2);
  CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, bytes, cost_.data(), wsum_.data(), static_cast<int>(n)));
  if (sort_tmp_.size() < bytes) sort_tmp_.resize(bytes);
  CUDA_CHECK(cub::DeviceReduce::Sum(sort_tmp_.data(), bytes, cost_.data(), wsum_.data(), static_cast<int>(n)));
  bytes = 0;
  CUDA_CHECK(cub::DeviceReduce::Max(nullptr, bytes, cost_.data(), wsum_.data() + 1, static_cast<int>(n)));
  if (sort_tmp_.size() < bytes) sort_tmp_.resize(bytes);
  CUDA_CHECK(cub::DeviceReduce::Max(sort_tmp_.data(), bytes, cost_.data(), wsum_.data() + 1, static_cast<int>(n)));
  float wm[2] = {0.f, 0.f};
  CUDA_CHECK(cudaMemcpy(wm, wsum_.data(), sizeof(wm), cudaMemcpyDeviceToHost));
  W = wm[0];
  M = wm[1];
}

// heaviest predicted task first sorted by cost, descending
void Matcher::sort_by_cost(uint64_t n) {
  tasks_sorted_.resize(n);
  cost_sorted_.resize(n);
  len2_sorted_.resize(n);
  size_t bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(nullptr, bytes, cost_.data(), cost_sorted_.data(),
                                                       tasks_.data(), tasks_sorted_.data(), static_cast<int>(n)));
  if (sort_tmp_.size() < bytes) sort_tmp_.resize(bytes);
  CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(sort_tmp_.data(), bytes, cost_.data(), cost_sorted_.data(),
                                                       tasks_.data(), tasks_sorted_.data(), static_cast<int>(n)));
  bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(nullptr, bytes, cost_.data(), cost_sorted_.data(),
                                                       len2_.data(), len2_sorted_.data(), static_cast<int>(n)));
  if (sort_tmp_.size() < bytes) sort_tmp_.resize(bytes);
  CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(sort_tmp_.data(), bytes, cost_.data(), cost_sorted_.data(),
                                                       len2_.data(), len2_sorted_.data(), static_cast<int>(n)));
  tasks_.swap(tasks_sorted_);
  cost_.swap(cost_sorted_);
  len2_.swap(len2_sorted_);
}

// S3: cost-bounded splitting of the heaviest-first sorted task list 
uint64_t Matcher::split(uint64_t n, float B, float W, float target) {
  size_t bytes = 0;
  const int blocks = static_cast<int>(std::min<ull>((n + 255) / 256, 1u << 20));
  npieces_.resize(n);
  shape_.resize(n);
  k_s3_count<<<blocks, 256>>>(tasks_.data(), cost_.data(), len2_.data(), n, B, s3_.max_pieces, W, target, k_, plans_,
                              leaf_.data(), rate_l_.data(), npieces_.data(), shape_.data());
  CUDA_CHECK_LAUNCH();
  poff_.resize(n + 1);
  CUDA_CHECK(cudaMemset(poff_.data(), 0, sizeof(uint64_t)));
  bytes = 0;
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, bytes, npieces_.data(), poff_.data() + 1, static_cast<int>(n)));
  if (sort_tmp_.size() < bytes) sort_tmp_.resize(bytes);
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(sort_tmp_.data(), bytes, npieces_.data(), poff_.data() + 1,
                                           static_cast<int>(n)));
  uint64_t total = 0;
  CUDA_CHECK(cudaMemcpy(&total, poff_.data() + n, sizeof(uint64_t), cudaMemcpyDeviceToHost));
  if (total == n) return n;  // nothing split
  tasks_sorted_.resize(total);
  cost_sorted_.resize(total);
  range_sorted_.resize(total);
  slice_sorted_.resize(total);
  pidx_.resize(total);
  k_s3_fill<<<blocks, 256>>>(tasks_.data(), cost_.data(), len2_.data(), npieces_.data(), shape_.data(), poff_.data(),
                             n, tasks_sorted_.data(), cost_sorted_.data(), range_sorted_.data(), slice_sorted_.data(),
                             pidx_.data());
  CUDA_CHECK_LAUNCH();
  // re-sort pieces heaviest-first (LPT over pieces)
  cost_.resize(total);
  pidx_sorted_.resize(total);
  bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(nullptr, bytes, cost_sorted_.data(), cost_.data(), pidx_.data(),
                                                       pidx_sorted_.data(), static_cast<int>(total)));
  if (sort_tmp_.size() < bytes) sort_tmp_.resize(bytes);
  CUDA_CHECK(cub::DeviceRadixSort::SortPairsDescending(sort_tmp_.data(), bytes, cost_sorted_.data(), cost_.data(),
                                                       pidx_.data(), pidx_sorted_.data(), static_cast<int>(total)));
  tasks_.resize(total);
  range_.resize(total);
  slice_.resize(total);
  k_s3_gather<<<static_cast<int>(std::min<ull>((total + 255) / 256, 1u << 20)), 256>>>(
      pidx_sorted_.data(), total, tasks_sorted_.data(), range_sorted_.data(), slice_sorted_.data(), tasks_.data(),
      range_.data(), slice_.data());
  CUDA_CHECK_LAUNCH();
  std::vector<uint64_t> h(n);
  CUDA_CHECK(cudaMemcpy(h.data(), npieces_.data(), n * sizeof(uint64_t), cudaMemcpyDeviceToHost));
  for (uint64_t x : h)
    if (x > 1) {
      ++host_.s3_split;
      host_.s3_pieces += x;
    }
  return total;
}

void Matcher::match_batch(const DynGraphView& g, const DeviceStreamView& s, size_t first, size_t count,
                          const uint8_t* eff, const SigView* sv) {
  h_counts_.assign(count, 0);
  h_tmo_.assign(count, 0);
  if (count == 0) return;
  // the controller decides whether the S2 gate / closure are worth running in this batch
  const bool gate_on = batch_no_ >= gate_off_until_;
  const bool clo_on = batch_no_ >= clo_off_until_;
  if (!gate_on || !clo_on) ++host_.s2_off_batches;
  const bool s2 = sv != nullptr;
  S2Args sa{};
  probe_ran_ = s2 && s2_probe_ && batch_no_ >= next_probe_batch_;  // probe back-off
  if (s2 && s2_probe_) {
    // S2 dormant
    sa = S2Args{0, clo_on, 0, sv->rows, sv->degout, sv->degin, sv->mask, sv->hub, sv->slot,
                qrows_.data(),
                qdo_.data(), qdi_.data(), creq_.data(), creq_off_.data(), gate_.data(), probe_ran_ ? 1 : 0};
  } else if (s2) {
    sa = S2Args{1, clo_on, gate_on, sv->rows, sv->degout, sv->degin, sv->mask,
                sv->hub, sv->slot, qrows_.data(), qdo_.data(), qdi_.data(), creq_.data(), creq_off_.data(),
                gate_.data(), 0};
  }
  sa.lay = lay_;
  const uint64_t total = static_cast<uint64_t>(count) * anchors_;
  counts_.resize(count);
  tmo_.resize(count);
  tasks_.resize(total);
  cost_.resize(std::max<uint64_t>(total, 1));
  rerun_.resize(std::max<uint64_t>(total, 1));
  GpuTimer timer;
  timer.start();
  counts_.fill_bytes(0);
  tmo_.fill_bytes(0);
  CUDA_CHECK(cudaMemset(ctr_.data() + kMcNext, 0, 2 * sizeof(ull)));      // cursor + list length
  const int tgrid = static_cast<int>(std::min<uint64_t>((total + 255) / 256, 1u << 16));
  k_triage<<<std::max(tgrid, 1), 256>>>(s, first, static_cast<uint32_t>(count), eff, g.vlabel, levels_.data(), k_,
                                         anchors_, plan_off_.data(), plans_, sa, ctr_.data(), tasks_.data());
  CUDA_CHECK_LAUNCH();
  ull n = 0;
  CUDA_CHECK(cudaMemcpy(&n, ctr_.data() + kMcNTasks, sizeof(ull), cudaMemcpyDeviceToHost));
  // S1 dormancy
  const bool s1a = batch_no_ >= s1_sleep_until_;
  if (!s1a) ++host_.s1_dormant_batches;
  if (s1a && n > 0) {
    GpuTimer t1;
    t1.start();
    const bool sig = sv != nullptr && !s2_probe_;  // stale rows are not read for estimates either
    len2_.resize(n);
    const S1Args a{plans_, k_, s1_.explore, s1_.min_k, kappa_.data(), len2_.data(),
                   static_cast<unsigned>(batch_no_), plan_off_.data(), plan_anchor_.data(), plan_kind_.data(),
                   levels_.data(), checks_.data(), rate_c_.data(), rate_s_.data(), rate_r_.data(), ready_.data(),
                   sig ? sv->rows : nullptr, sig ? sv->mask : nullptr, sig ? sv->hub : nullptr,
                   sig ? sv->slot : nullptr, lay_, leaf_.data()};
    k_s1<<<static_cast<int>(std::min<ull>((n + 255) / 256, 1u << 16)), 256>>>(g, s, first, a, n, tasks_.data(),
                                                                              cost_.data(), ctr_.data());
    CUDA_CHECK_LAUNCH();
    host_.s1_ms += t1.stop_ms();
  }
  bool pieces = false;
  if (s1a && n > 0) {
    // the batch's predicted cost profile decides what S1 / S3 still have to do
    GpuTimer t1;
    t1.start();
    float W = 0.f, M = 0.f;
    cost_stats(n, W, M);
    const double warps = static_cast<double>(grid_) * kMatchBlock / 32.0;
    const float B = static_cast<float>(std::max<double>(static_cast<double>(s3_.min_piece), W / (s3_.share * warps)));
    host_.s1_ms += t1.stop_ms();
    // FLAT batch: no task can exceed the piece budget
    const float target = static_cast<float>(s3_.share * warps);
    const bool under = static_cast<double>(n) < target && k_ >= 3 && M >= static_cast<float>(s3_.min_piece);
    const bool flat = M <= B && !under;
    if (flat) ++host_.flat_batches;
    // a flat batch in which S1 decided nothing
    ull c[kMcNum];
    CUDA_CHECK(cudaMemcpy(c, ctr_.data(), sizeof(c), cudaMemcpyDeviceToHost));
    const ull dev = c[kMcK1] + c[kMcK2] + c[kMcSwitched] + c[kMcExplore];
    const bool idle = dev == last_s1_dev_;
    last_s1_dev_ = dev;
    flat_streak_ = (flat && (4.f * M <= B || idle)) ? flat_streak_ + 1 : 0;
    if (flat_streak_ >= 2) {  // two clearly flat batches in a row: sleep for the next 4
      s1_sleep_until_ = batch_no_ + 1 + 4;
      flat_streak_ = 0;
    }
    if (!flat && n > 1) {
      GpuTimer ts;
      ts.start();
      sort_by_cost(n);
      host_.s1_ms += ts.stop_ms();
    }
    if (!flat && n > 0 && k_ >= 3) {  // S3: split predicted-heavy tasks before launch
      GpuTimer t3;
      t3.start();
      const ull m = split(n, B, W, under ? target : 0.f);
      if (m != n) {
        pieces = true;
        n = m;
        CUDA_CHECK(cudaMemcpy(ctr_.data() + kMcNTasks, &n, sizeof(ull), cudaMemcpyHostToDevice));
      }
      host_.s3_ms += t3.stop_ms();
      ++s3_batches_;
      s3_budget_sum_ += B;
    }
  }
  k_match<<<grid_, kMatchBlock>>>(g, s, first, tasks_.data(), levels_.data(), checks_.data(), leaf_.data(), k_, plans_,
                                  plan_anchor_.data(), plan_off_.data(), sa, ctr_.data(), counts_.data(),
                                  tmo_.data(), s1a ? lstat_.data() : nullptr, probe_stat_.data(), s1_.probe_budget,
                                  s1_.bet_scale, s1a ? cost_.data() : nullptr, pieces ? range_.data() : nullptr,
                                  pieces ? slice_.data() : nullptr, rerun_.data(), kappa_.data(),
                                  s1a ? calib_.data() : nullptr, hist_.data(), deadline_rel_);
  CUDA_CHECK_LAUNCH();
  {
    ull R = 0;
    CUDA_CHECK(cudaMemcpy(&R, ctr_.data() + kMcRerunN, sizeof(ull), cudaMemcpyDeviceToHost));
    if (R > 0) {
      CUDA_CHECK(cudaMemset(ctr_.data() + kMcRerunN, 0, sizeof(ull)));
      tasks_.resize(R);
      cost_.resize(R);
      len2_.resize(R);
      CUDA_CHECK(cudaMemcpy(tasks_.data(), rerun_.data(), R * sizeof(ull), cudaMemcpyDeviceToDevice));
      ull m = R;
      CUDA_CHECK(cudaMemset(ctr_.data() + kMcNext, 0, sizeof(ull)));
      CUDA_CHECK(cudaMemcpy(ctr_.data() + kMcNTasks, &m, sizeof(ull), cudaMemcpyHostToDevice));
      const S1Args a2{plans_, k_, 0, 1 << 20, kappa_.data(), len2_.data(), static_cast<unsigned>(batch_no_),
                      plan_off_.data(), plan_anchor_.data(), plan_kind_.data(), levels_.data(), checks_.data(),
                      rate_c_.data(), rate_s_.data(), rate_r_.data(), ready_.data(), nullptr, nullptr, nullptr, nullptr,
                      lay_, leaf_.data()};
      k_s1<<<static_cast<int>(std::min<ull>((R + 255) / 256, 1u << 16)), 256>>>(g, s, first, a2, R, tasks_.data(),
                                                                                cost_.data(), ctr_.data());
      CUDA_CHECK_LAUNCH();
      float W2 = 0.f, M2 = 0.f;
      cost_stats(R, W2, M2);
      const double warps2 = static_cast<double>(grid_) * kMatchBlock / 32.0;
      const float B2 = static_cast<float>(std::max<double>(static_cast<double>(s3_.min_piece), W2 / (s3_.share * warps2)));
      if (R > 1) sort_by_cost(R);
      bool pieces2 = false;
      if (k_ >= 3) {
        m = split(R, B2, W2, static_cast<float>(s3_.share * warps2));
        if (m != R) {
          pieces2 = true;
          CUDA_CHECK(cudaMemcpy(ctr_.data() + kMcNTasks, &m, sizeof(ull), cudaMemcpyHostToDevice));
        }
      }
      k_match<<<grid_, kMatchBlock>>>(g, s, first, tasks_.data(), levels_.data(), checks_.data(), leaf_.data(), k_,
                                      plans_, plan_anchor_.data(), plan_off_.data(), sa, ctr_.data(), counts_.data(),
                                      tmo_.data(), nullptr, probe_stat_.data(), s1_.probe_budget, s1_.bet_scale,
                                      cost_.data(), pieces2 ? range_.data() : nullptr, pieces2 ? slice_.data() : nullptr,
                                      nullptr, kappa_.data(), nullptr, hist_.data(), deadline_rel_);
      CUDA_CHECK_LAUNCH();
    }
  }
  CUDA_CHECK(cudaMemcpy(h_counts_.data(), counts_.data(), count * sizeof(ull), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_tmo_.data(), tmo_.data(), count, cudaMemcpyDeviceToHost));
  b_kernel_ms_ = timer.stop_ms();
  host_.kernel_ms += b_kernel_ms_;
  for (size_t i = 0; i < count; ++i) {
    host_.updates_with_matches += h_counts_[i] != 0;
    host_.timed_out_updates += h_tmo_[i];
  }
  if (s1a) learn();
  {  // this batch's triage evidence, for the controller's S2-maintenance decision (end_batch_s2)
    ull c[kMcNum];
    CUDA_CHECK(cudaMemcpy(c, ctr_.data(), sizeof(c), cudaMemcpyDeviceToHost));
    b_tested_ = sv ? c[kMcLabel] - last_label_ : 0;
    b_rejected_ = !sv ? 0 : s2_probe_ ? c[kMcProbeRej] - last_probe_ : (c[kMcLabel] - last_label_) - (c[kMcS2] - last_s2pass_);
    b_probe_ = s2_probe_;
    b_probe_ran_ = probe_ran_;
    b_count_ = count;
    last_probe_ = c[kMcProbeRej];
    b_gate_rej_ = c[kMcGateRej] - last_gate_rej2_;
    b_chunks_ = c[kMcChunks] - last_chunks2_;
    last_gate_rej2_ = c[kMcGateRej];
    last_chunks2_ = c[kMcChunks];
    b_anchored_ = c[kMcAnchored] - last_anch_;
    b_cyc_ = c[kMcSumCyc] - last_cyc_;
    b_zcyc_ = c[kMcZeroCyc] - last_zcyc_;
    b_zn_ = c[kMcZeroN] - last_zn_;
    last_cyc_ = c[kMcSumCyc];
    last_zcyc_ = c[kMcZeroCyc];
    last_zn_ = c[kMcZeroN];
    b_s2_used_ = sv != nullptr;
    last_label_ = c[kMcLabel];
    last_s2pass_ = c[kMcS2];
    last_anch_ = c[kMcAnchored];
  }
  {  // control: switch an S2 use off for 7 batches when it rejected < 0.2% of what it tested
    ull c[kMcNum];
    CUDA_CHECK(cudaMemcpy(c, ctr_.data(), sizeof(c), cudaMemcpyDeviceToHost));
    const ull gt = c[kMcGateTested] - last_gate_tested_, gr = c[kMcGateRej] - last_gate_rej_;
    const ull an = c[kMcAnchored] - last_anchored_, cf = c[kMcCloFail] - last_clo_fail_;
    if (gate_on && gt >= 1000 && gr * 500 < gt) gate_off_until_ = batch_no_ + 8;
    if (clo_on && an >= 1000 && cf * 500 < an) clo_off_until_ = batch_no_ + 8;
    last_gate_tested_ = c[kMcGateTested];
    last_gate_rej_ = c[kMcGateRej];
    last_anchored_ = c[kMcAnchored];
    last_clo_fail_ = c[kMcCloFail];
  }
  ++batch_no_;
}

MatchStats Matcher::stats() const {
  ull c[kMcNum];
  CUDA_CHECK(cudaMemcpy(c, ctr_.data(), sizeof(c), cudaMemcpyDeviceToHost));
  MatchStats m = host_;
  m.tasks = c[kMcTasks];
  m.label_pass = c[kMcLabel];
  m.s2_pass = c[kMcS2];
  m.anchored = c[kMcAnchored];
  m.closure_fail = c[kMcCloFail];
  m.closure_chunks = c[kMcCloChunks];
  m.gate_tested = c[kMcGateTested];
  m.gate_rejects = c[kMcGateRej];
  m.chunks = c[kMcChunks];
  m.timeouts = c[kMcTimeouts];
  m.matches_pos = c[kMcPos];
  m.matches_neg = c[kMcNeg];
  for (int x = 0; x < 3; ++x) {
    m.kind_chosen[x] = c[kMcK0 + x];
    m.size_class[x] = c[kMcB0 + x];
  }
  m.explored = c[kMcExplore];
  m.switched = c[kMcSwitched];
  m.local_used = c[kMcLocal];
  m.probe_aborts = c[kMcProbeAbort];
  m.bindings = c[kMcBind];
  m.dead_tasks = c[kMcDead];
  m.bet_waste = c[kMcWaste];
  m.abort_counted = c[kMcAbortCnt];
  m.max_task_ms = static_cast<double>(c[kMcMaxCyc]) / clock_khz_;
  m.sum_task_ms = static_cast<double>(c[kMcSumCyc]) / clock_khz_;
  m.s3_budget_avg = s3_batches_ ? s3_budget_sum_ / s3_batches_ : 0.0;
  {  // ratio percentiles from the half-octave histogram (bucket centre)
    ull h[kHist];
    CUDA_CHECK(cudaMemcpy(h, hist_.data(), sizeof(h), cudaMemcpyDeviceToHost));
    ull tot = 0;
    for (int b = 0; b < kHist; ++b) tot += h[b];
    m.ratio_n = tot;
    auto pct = [&](double q) {
      ull acc = 0;
      for (int b = 0; b < kHist; ++b) {
        acc += h[b];
        if (acc >= q * tot) return std::pow(2.0, (b - kHist / 2 + 0.5) / 2.0);
      }
      return 0.0;
    };
    if (tot) {
      m.ratio_p50 = pct(0.5);
      m.ratio_p90 = pct(0.9);
      m.ratio_p99 = pct(0.99);
    }
  }
  return m;
}

}  // namespace csm
