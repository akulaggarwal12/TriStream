#include "csm/sig_store.cuh"

#include <cub/cub.cuh>

#include <numeric>

namespace csm {
namespace {

using namespace sig;
using ull = unsigned long long;
constexpr int kBlk = 256;
constexpr int kWpb = kBlk / 32;
enum : int { kCEnd = 0, kCAff, kCDef, kCStale, kCSkip, kCNum = 8 };

struct Win {  // entry visible iff ins <= hi && del > lo (base entries have ins = 0)
  ts_t lo, hi;
};
constexpr uint32_t kNoSlot = 0xFFFFFFFFu;
struct Proj {  // query projection: only neighbours with a query label count (n = 0: no projection)
  const uint8_t* mask;
  const label_t* labels;
  int n;
  const uint32_t* slot;  // vertex -> compact row
  __device__ __forceinline__ bool keep(label_t l) const { return n == 0 || mask[l]; }
};

__device__ __forceinline__ const BaseSideView& bside(const DynGraphView& g, int s) { return s ? g.in_base : g.out_base; }
__device__ __forceinline__ const DeltaSideView& dside(const DynGraphView& g, int s) {
  return s ? g.in_delta : g.out_delta;
}
__device__ __forceinline__ bool dvis(const DeltaSideView& d, uint64_t i, Win w) { return d.ins[i] <= w.hi && d.del[i] > w.lo; }
__device__ __forceinline__ label_t bel(const BaseSideView& b, eid_t p) { return b.el ? b.el[p] : kNoEdgeLabel; }

// For a visible delta entry: is it the first visible version of its key (fk) / of its (key, edge label) (fke)?
// Several versions of one edge can be visible in a union window (insert, delete, re-insert in one batch).
__device__ void firsts(const DynGraphView& g, int s, vid_t v, uint64_t i, Win win, bool use_el, bool& fk, bool& fke) {
  const DeltaSideView& d = dside(g, s);
  const BaseSideView& b = bside(g, s);
  const vid_t w = static_cast<vid_t>(d.sk[i] & 0xFFFFFFFFu);
  const label_t el = use_el ? d.el[i] : kNoEdgeLabel;
  fk = fke = true;
  const eid_t p = base_find(b, g.vlabel, v, w);
  if (p != kNotFound && b.del[p] > win.lo) {
    fk = false;
    if ((use_el ? bel(b, p) : kNoEdgeLabel) == el) fke = false;
  }
  for (uint64_t j = i; j > 0 && d.sk[j - 1] == d.sk[i]; --j)
    if (dvis(d, j - 1, win)) {
      fk = false;
      if ((use_el ? d.el[j - 1] : kNoEdgeLabel) == el) fke = false;
    }
}

// Does x's side-s list contain y in the window?
__device__ bool edge_in(const DynGraphView& g, int s, vid_t x, vid_t y, Win win) {
  const BaseSideView& b = bside(g, s);
  const eid_t p = base_find(b, g.vlabel, x, y);
  if (p != kNotFound && b.del[p] > win.lo) return true;
  const DeltaSideView& d = dside(g, s);
  if (d.size == 0) return false;
  const uint64_t key = pack_sk(x, y);
  for (uint64_t i = delta_lower_bound(d, key); i < d.size && d.sk[i] == key; ++i)
    if (dvis(d, i, win)) return true;
  return false;
}

// Warp-strided visit of v's visible entries on side s: f(w, el, fk, fke). With a projection only the query-label
// segments of the (label-partitioned) base list are scanned, and delta entries of other labels are skipped.
template <class F>
__device__ void for_each_nb(const DynGraphView& g, int s, vid_t v, Win win, bool use_el, int lane, Proj pj, F f) {
  const BaseSideView& b = bside(g, s);
  if (pj.n == 0) {
    for (eid_t p = b.offsets[v] + lane; p < b.offsets[v + 1]; p += 32)
      if (b.del[p] > win.lo) f(adj_vertex(b.keys[p]), bel(b, p), true, true);
  } else {
    eid_t lo = b.offsets[v];
    const eid_t e = b.offsets[v + 1];
    for (int li = 0; li < pj.n && lo < e; ++li) {
      const label_t l = pj.labels[li];
      const eid_t bb = lower_bound_key(b.keys, lo, e, adj_key(l, 0));
      const eid_t be = lower_bound_key(b.keys, bb, e, adj_key(l + 1, 0));
      for (eid_t p = bb + lane; p < be; p += 32)
        if (b.del[p] > win.lo) f(adj_vertex(b.keys[p]), bel(b, p), true, true);
      lo = be;
    }
  }
  const DeltaSideView& d = dside(g, s);
  if (d.size == 0) return;
  const uint64_t db = delta_lower_bound(d, pack_sk(v, 0)), de = delta_lower_bound(d, pack_sk(v + 1, 0));
  for (uint64_t i = db + lane; i < de; i += 32) {
    if (!dvis(d, i, win)) continue;
    if (!pj.keep(g.vlabel[static_cast<vid_t>(d.sk[i] & 0xFFFFFFFFu)])) continue;
    bool fk, fke;
    firsts(g, s, v, i, win, use_el, fk, fke);
    if (fk || fke) f(static_cast<vid_t>(d.sk[i] & 0xFFFFFFFFu), d.el[i], fk, fke);
  }
}

// Order 0 + order 1 slope + profile of the listed vertices
__global__ void k_order1(DynGraphView g, const vid_t* list, uint64_t count, Win win, int use_el, uint32_t tau,
                         uint32_t* degout, uint32_t* degin, uint32_t* prof, uint32_t* rows, uint8_t* hub,
                         uint8_t* expand, int deferred, Proj pj, Layout L) {
  __shared__ uint32_t sm[kWpb][2 + kMaxFar + kMaxSlope];
  const int lane = threadIdx.x & 31;
  uint32_t* a = sm[threadIdx.x >> 5];
  const uint64_t nw = static_cast<uint64_t>(gridDim.x) * kWpb;
  for (uint64_t i = blockIdx.x * static_cast<uint64_t>(kWpb) + (threadIdx.x >> 5); i < count; i += nw) {
    const vid_t v = list ? list[i] : static_cast<vid_t>(i);
    for (int x = lane; x < 2 + kMaxFar + kMaxSlope; x += 32) a[x] = 0;
    __syncwarp();
    const int sides = g.directed ? 2 : 1;
    for (int s = 0; s < sides; ++s)
      for_each_nb(g, s, v, win, use_el != 0, lane, pj, [&](vid_t w, label_t el, bool fk, bool fke) {
        const label_t lw = g.vlabel[w];
        if (fk) {
          atomicAdd(&a[s], 1u);
          const int f = L.far_of(lw, s);
          if (f >= 0) atomicAdd(&a[2 + f], 1u);
        }
        if (fke) {
          const int sl = L.slope(lw, s, use_el ? el : kNoEdgeLabel);
          if (sl >= 0) atomicAdd(&a[2 + kMaxFar + sl], 1u);
        }
      });
    __syncwarp();
    const uint64_t kv = pj.slot[v];
    if (lane == 0) {
      const uint32_t old_deg = degout[kv] + degin[kv], new_deg = a[0] + a[1];
      if (expand) {
        const bool cls = stair_class(old_deg) != stair_class(new_deg);
        expand[i] = cls ? 1 : (!hub[kv] ? (deferred ? 2 : 1) : 0);
      }
      degout[kv] = a[0];
      degin[kv] = a[1];
      if (new_deg > tau) hub[kv] = 1;
    }
    for (int x = lane; x < L.nf; x += 32) prof[kv * L.nf + x] = a[2 + x];
    for (int x = lane; x < L.n1; x += 32) rows[kv * L.words + x] = a[2 + kMaxFar + x];
    __syncwarp();
  }
}

// Order 1 staircase + order 2 curvature 
__global__ void k_order2(DynGraphView g, const vid_t* list, uint64_t count, Win win, const uint32_t* degout,
                         const uint32_t* degin, const uint32_t* prof, uint32_t* rows, const uint8_t* hub,
                         uint32_t* mask, uint8_t* stale, Proj pj, Layout L) {
  __shared__ uint32_t sm[kWpb][kMaxStair + kMaxPair + 1];
  const int lane = threadIdx.x & 31;
  uint32_t* a = sm[threadIdx.x >> 5];
  constexpr int kMask = kMaxStair + kMaxPair;
  const uint64_t nw = static_cast<uint64_t>(gridDim.x) * kWpb;
  for (uint64_t i = blockIdx.x * static_cast<uint64_t>(kWpb) + (threadIdx.x >> 5); i < count; i += nw) {
    const vid_t v = list ? list[i] : static_cast<vid_t>(i);
    for (int x = lane; x <= kMask; x += 32) a[x] = 0;
    __syncwarp();
    const uint64_t kv = pj.slot[v];
    const bool vhub = hub[kv] != 0;
    const label_t lv = g.vlabel[v];
    const int sides = g.directed ? 2 : 1;
    for (int s = 0; s < sides; ++s)
      for_each_nb(g, s, v, win, false, lane, pj, [&](vid_t w, label_t, bool fk, bool) {
        if (!fk) return;
        const label_t lw = g.vlabel[w];
        const uint64_t kw = pj.slot[w];
        const int cls = stair_class(degout[kw] + degin[kw]);
        for (int j = 0; j < cls; ++j) {
          const int t = L.stair(lw, s, j);
          if (t >= 0) atomicAdd(&a[t], 1u);
        }
        if (vhub) return;
        const int grp = L.mid_of(lw, s);
        if (grp < 0) return;
        if (hub[kw]) {
          atomicOr(&a[kMask], 1u << grp);
          return;
        }
        uint32_t* c = a + kMaxStair;
        for (int x = L.moff[grp]; x < L.moff[grp + 1]; ++x) {
          const uint32_t pw = prof[kw * L.nf + L.mfar[x]];
          if (pw) atomicAdd(&c[L.mslot[x]], pw);
        }
        // remove the 2-paths that come back to v (x = v):
        const int p0 = L.pair_of(grp, L.far_of(lv, 0)), p1 = L.pair_of(grp, L.far_of(lv, 1));
        if (!pj.keep(lv)) {
        } else if (!g.directed) {
          if (p0 >= 0) atomicSub(&c[p0], 1u);
        } else {
          if (p1 >= 0 && (s == 0 || edge_in(g, 0, v, w, win))) atomicSub(&c[p1], 1u);  // v→w: v is in-nb of w
          if (p0 >= 0 && (s == 1 || edge_in(g, 1, v, w, win))) atomicSub(&c[p0], 1u);  // w→v: v is out-nb of w
        }
      });
    __syncwarp();
    uint32_t* r = rows + kv * L.words;
    for (int x = lane; x < L.ns; x += 32) r[L.n1 + x] = a[x];
    for (int x = lane; x < L.np; x += 32) r[L.n1 + L.ns + x] = vhub ? 0u : a[kMaxStair + x];
    if (lane == 0) {
      mask[kv] = vhub ? 0u : a[kMask];  // exact row: only hub groups stay masked
      stale[kv] = 0;
    }
    __syncwarp();
  }
}

__global__ void k_endpoints(DeviceStreamView s, size_t first, uint32_t count, const uint8_t* eff, const label_t* vlabel,
                            Proj pj, vid_t* out, ull* ctr, int del_only) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count || !eff[i]) return;
  if (del_only && s.op[first + i] != static_cast<uint8_t>(UpdateOp::DeleteEdge)) return;
  // projection: an update changes a readable (projected) row only if both endpoints carry query labels
  if (!pj.keep(vlabel[s.u[first + i]]) || !pj.keep(vlabel[s.v[first + i]])) {
    atomicAdd(&ctr[kCSkip], 1ull);
    return;
  }
  const ull p = atomicAdd(&ctr[kCEnd], 2ull);
  out[p] = s.u[first + i];
  out[p + 1] = s.v[first + i];
}

__global__ void k_count(const vid_t* ends, uint64_t m, const uint8_t* expand, const uint32_t* degout,
                        const uint32_t* degin, const uint32_t* slot, uint64_t* cnt) {
  const uint64_t i = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
  if (i >= m) return;
  const uint32_t k = slot[ends[i]];
  cnt[i] = 1 + (expand[i] == 1 ? static_cast<uint64_t>(degout[k]) + degin[k] : 0);
}

__global__ void k_fill(DynGraphView g, const vid_t* ends, uint64_t m, const uint8_t* expand, const uint64_t* off,
                       Win win, Proj pj, vid_t* aff) {
  __shared__ uint32_t pos[kWpb];
  const int lane = threadIdx.x & 31, wib = threadIdx.x >> 5;
  const uint64_t nw = static_cast<uint64_t>(gridDim.x) * kWpb;
  for (uint64_t i = blockIdx.x * static_cast<uint64_t>(kWpb) + wib; i < m; i += nw) {
    const vid_t v = ends[i];
    vid_t* out = aff + off[i];
    if (lane == 0) {
      out[0] = v;
      pos[wib] = 1;
    }
    __syncwarp();
    if (expand[i] == 1) {
      const int sides = g.directed ? 2 : 1;
      for (int s = 0; s < sides; ++s)
        for_each_nb(g, s, v, win, false, lane, pj, [&](vid_t w, label_t, bool fk, bool) {
          if (fk) out[atomicAdd(&pos[wib], 1u)] = w;
        });
    }
    __syncwarp();
  }
}

__device__ __forceinline__ bool in_sorted(const vid_t* a, uint64_t m, vid_t x) {
  uint64_t lo = 0, hi = m;
  while (lo < hi) {
    const uint64_t mid = (lo + hi) >> 1;
    if (a[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo < m && a[lo] == x;
}

// Deferred mode: warp per endpoint a with expand == 2. 
__global__ void k_defer(DynGraphView g, const vid_t* ends, uint64_t m, const uint8_t* expand, Win win, uint32_t* mask,
                        uint8_t* stale, ull* ctr, Proj pj, Layout L) {
  const int lane = threadIdx.x & 31;
  const uint64_t nw = static_cast<uint64_t>(gridDim.x) * kWpb;
  ull n_def = 0;
  for (uint64_t i = blockIdx.x * static_cast<uint64_t>(kWpb) + (threadIdx.x >> 5); i < m; i += nw) {
    if (expand[i] != 2) continue;
    const vid_t a = ends[i];
    const label_t la = g.vlabel[a];
    const int sides = g.directed ? 2 : 1;
    for (int s = 0; s < sides; ++s) {
      const int gm = L.mid_of(la, g.directed ? 1u - s : 0u);
      if (gm < 0) continue;
      const uint32_t bit = 1u << gm;
      for_each_nb(g, s, a, win, false, lane, pj, [&](vid_t y, label_t, bool fk, bool) {
        if (!fk || in_sorted(ends, m, y)) return;
        const uint32_t ky = pj.slot[y];
        atomicOr(&mask[ky], bit);
        stale[ky] = 1;
        ++n_def;
      });
    }
  }
  if (n_def) atomicAdd(&ctr[kCDef], n_def);
}

__global__ void k_collect_stale(const uint8_t* stale, uint64_t nk, const vid_t* vert, vid_t* out, ull* ctr) {
  const uint64_t k = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
  if (k < nk && stale[k]) out[atomicAdd(&ctr[kCStale], 1ull)] = vert[k];
}

// keep flag per vertex (query label, or every vertex without a projection) -> compact rows
__global__ void k_keep(const label_t* vlabel, vid_t n, Proj pj, uint32_t* flag) {
  const vid_t v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v < n) flag[v] = pj.keep(vlabel[v]) ? 1u : 0u;
}
__global__ void k_slots(vid_t n, const uint32_t* flag, uint32_t* slot, vid_t* vert) {
  const vid_t v = blockIdx.x * blockDim.x + threadIdx.x;  // slot holds the exclusive prefix sum of the flags
  if (v >= n) return;
  if (flag[v]) vert[slot[v]] = v;
  else slot[v] = kNoSlot;
}

inline int grid_for(uint64_t items, int per_block) {
  uint64_t b = (items + per_block - 1) / per_block;
  return static_cast<int>(std::max<uint64_t>(1, std::min<uint64_t>(b, 1u << 20)));
}

}  // namespace

void SigStore::run(const DynGraphView& g, const vid_t* list, uint64_t count, ts_t lo, ts_t hi, bool expand,
                   bool deferred) {
  const Win win{lo, hi};
  const Proj pj{pmask_.data(), plabels_.data(), pn_, slot_.data()};
  k_order1<<<grid_for(count, kWpb), kBlk>>>(g, list, count, win, use_el_, tau_, degout_.data(), degin_.data(),
                                             prof_.data(), rows_.data(), hub_.data(),
                                             expand ? expand_.data() : nullptr, deferred ? 1 : 0, pj, lay_);
  CUDA_CHECK_LAUNCH();
  if (!expand) {  // full build: every row
    k_order2<<<grid_for(count, kWpb), kBlk>>>(g, list, count, win, degout_.data(), degin_.data(), prof_.data(),
                                               rows_.data(), hub_.data(), mask_.data(), stale_.data(), pj, lay_);
    CUDA_CHECK_LAUNCH();
    return;
  }
  if (deferred) {  // mask + mark before the exact recomputations below (a row recomputed later is exact again)
    CUDA_CHECK(cudaMemset(ctr_.data() + kCDef, 0, sizeof(ull)));
    k_defer<<<grid_for(count, kWpb), kBlk>>>(g, list, count, expand_.data(), win, mask_.data(), stale_.data(),
                                              ctr_.data(), pj, lay_);
    CUDA_CHECK_LAUNCH();
    ull d = 0;
    CUDA_CHECK(cudaMemcpy(&d, ctr_.data() + kCDef, sizeof(ull), cudaMemcpyDeviceToHost));
    st_.deferred_rows += d;
    if (d) maybe_stale_ = true;
  }
  // refreshed set = endpoints ∪ neighbours of expanded endpoints, deduplicated
  cnt_.resize(count);
  off_.resize(count + 1);
  k_count<<<grid_for(count, kBlk), kBlk>>>(list, count, expand_.data(), degout_.data(), degin_.data(), slot_.data(),
                                          cnt_.data());
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaMemset(off_.data(), 0, sizeof(uint64_t)));
  size_t bytes = 0;
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, bytes, cnt_.data(), off_.data() + 1, count));
  if (tmp_.size() < bytes) tmp_.resize(bytes);
  CUDA_CHECK(cub::DeviceScan::InclusiveSum(tmp_.data(), bytes, cnt_.data(), off_.data() + 1, count));
  uint64_t total = 0;
  CUDA_CHECK(cudaMemcpy(&total, off_.data() + count, sizeof(uint64_t), cudaMemcpyDeviceToHost));
  aff_.resize(total);
  aff_sorted_.resize(total);
  k_fill<<<grid_for(count, kWpb), kBlk>>>(g, list, count, expand_.data(), off_.data(), win, pj, aff_.data());
  CUDA_CHECK_LAUNCH();
  bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, bytes, aff_.data(), aff_sorted_.data(), static_cast<int>(total)));
  if (tmp_.size() < bytes) tmp_.resize(bytes);
  CUDA_CHECK(cub::DeviceRadixSort::SortKeys(tmp_.data(), bytes, aff_.data(), aff_sorted_.data(), static_cast<int>(total)));
  bytes = 0;
  CUDA_CHECK(cub::DeviceSelect::Unique(nullptr, bytes, aff_sorted_.data(), aff_.data(), ctr_.data() + kCAff,
                                       static_cast<int>(total)));
  if (tmp_.size() < bytes) tmp_.resize(bytes);
  CUDA_CHECK(cub::DeviceSelect::Unique(tmp_.data(), bytes, aff_sorted_.data(), aff_.data(), ctr_.data() + kCAff,
                                       static_cast<int>(total)));
  ull affected = 0;
  CUDA_CHECK(cudaMemcpy(&affected, ctr_.data() + kCAff, sizeof(ull), cudaMemcpyDeviceToHost));
  k_order2<<<grid_for(affected, kWpb), kBlk>>>(g, aff_.data(), affected, win, degout_.data(), degin_.data(),
                                                prof_.data(), rows_.data(), hub_.data(), mask_.data(), stale_.data(),
                                                pj, lay_);
  CUDA_CHECK_LAUNCH();
  st_.endpoint_rows += count;
  st_.affected_rows += affected;
}

void SigStore::set_projection(const std::vector<uint8_t>& label_mask) {
  std::vector<label_t> labels;
  for (size_t l = 0; l < label_mask.size(); ++l)
    if (label_mask[l]) labels.push_back(static_cast<label_t>(l));
  if (labels.empty() || labels.size() == label_mask.size()) {  // no labels given, or every label is a query label
    pn_ = 0;
    return;
  }
  pmask_.upload(label_mask.data(), label_mask.size());
  plabels_.upload(labels.data(), labels.size());
  pn_ = static_cast<int>(labels.size());
}

void SigStore::set_layout(const KeyLayout& k) {
  lay_ = k.view();
  klq_.upload(k.lq.data(), k.lq.size());
  kle_.upload(k.le.data(), k.le.size());
  ks1_.upload(k.s1.data(), k.s1.size());
  kst_.upload(k.st.data(), k.st.size());
  kmid_.upload(k.mid.data(), k.mid.size());
  kfar_.upload(k.far.data(), k.far.size());
  kpair_.upload(k.pair.data(), k.pair.size());
  kpm_.upload(k.pm.data(), k.pm.size());
  kmoff_.upload(k.moff.data(), k.moff.size());
  kmfar_.upload(k.mfar.data(), k.mfar.size());
  kmslot_.upload(k.mslot.data(), k.mslot.size());
  lay_.lq = klq_.data();
  lay_.le = k.use_el ? kle_.data() : nullptr;
  lay_.s1 = ks1_.data();
  lay_.st = kst_.data();
  lay_.mid = kmid_.data();
  lay_.far = kfar_.data();
  lay_.pair = kpair_.data();
  lay_.pm = kpm_.data();
  lay_.moff = kmoff_.data();
  lay_.mfar = kmfar_.data();
  lay_.mslot = kmslot_.data();
}

void SigStore::build(const DynGraphView& g, ts_t t) {
  GpuTimer timer;
  timer.start();
  // compact rows for the query-label vertices: flag -> exclusive scan = slot; vert = inverse
  {
    DevBuf<uint32_t> flag;
    flag.resize(std::max<size_t>(1, n_));
    slot_.resize(std::max<size_t>(1, n_));
    vert_.resize(std::max<size_t>(1, n_));
    k_keep<<<grid_for(n_, kBlk), kBlk>>>(g.vlabel, n_, Proj{pmask_.data(), plabels_.data(), pn_, nullptr}, flag.data());
    CUDA_CHECK_LAUNCH();
    size_t bytes = 0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, bytes, flag.data(), slot_.data(), static_cast<int>(n_)));
    if (tmp_.size() < bytes) tmp_.resize(bytes);
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(tmp_.data(), bytes, flag.data(), slot_.data(), static_cast<int>(n_)));
    uint32_t last_slot = 0, last_flag = 0;
    if (n_) {
      CUDA_CHECK(cudaMemcpy(&last_slot, slot_.data() + n_ - 1, 4, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(&last_flag, flag.data() + n_ - 1, 4, cudaMemcpyDeviceToHost));
    }
    nk_ = n_ ? static_cast<uint64_t>(last_slot) + last_flag : 0;
    k_slots<<<grid_for(n_, kBlk), kBlk>>>(n_, flag.data(), slot_.data(), vert_.data());
    CUDA_CHECK_LAUNCH();
  }
  const size_t nk = std::max<uint64_t>(1, nk_);
  rows_.resize(std::max<size_t>(1, nk * lay_.words));
  prof_.resize(std::max<size_t>(1, nk * lay_.nf));
  degout_.resize(nk);
  degin_.resize(nk);
  mask_.resize((nk + 3) & ~static_cast<size_t>(3));  // whole words for the 32-bit atomicOr
  hub_.resize(nk);
  stale_.resize(nk);
  for (auto* b : {&degout_, &degin_}) b->fill_bytes(0);
  mask_.fill_bytes(0);
  hub_.fill_bytes(0);
  stale_.fill_bytes(0);
  maybe_stale_ = false;
  ctr_.resize(kCNum);
  ctr_.fill_bytes(0);
  if (nk_) run(g, vert_.data(), nk_, t, t, false, false);
  st_.build_ms += timer.stop_ms();
}

void SigStore::note_dormant(const DeviceStreamView& s, size_t first, size_t count, const uint8_t* eff,
                            const label_t* vlabel) {
  if (count == 0) return;
  GpuTimer timer;
  timer.start();
  dirty_.resize(dirty_n_ + 2 * count, true);
  const ull base = dirty_n_;
  CUDA_CHECK(cudaMemcpy(ctr_.data() + kCEnd, &base, sizeof(ull), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(ctr_.data() + kCSkip, 0, sizeof(ull)));
  k_endpoints<<<grid_for(count, kBlk), kBlk>>>(s, first, static_cast<uint32_t>(count), eff, vlabel,
                                                Proj{pmask_.data(), plabels_.data(), pn_, slot_.data()}, dirty_.data(),
                                                ctr_.data(), 0);
  CUDA_CHECK_LAUNCH();
  ull m = 0, sk = 0;
  CUDA_CHECK(cudaMemcpy(&m, ctr_.data() + kCEnd, sizeof(ull), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&sk, ctr_.data() + kCSkip, sizeof(ull), cudaMemcpyDeviceToHost));
  skipped_ += sk;
  dirty_n_ = m;
  ++st_.dormant_batches;
  st_.refresh_ms += timer.stop_ms();
}

void SigStore::refresh(const DynGraphView& g, const DeviceStreamView& s, size_t first, size_t count,
                       const uint8_t* eff, ts_t lo, ts_t hi, bool deferred, bool del_only) {
  if (count == 0 && dirty_n_ == 0) return;
  GpuTimer timer;
  timer.start();
  ends_.resize(2 * count + dirty_n_);
  ends_sorted_.resize(2 * count + dirty_n_);
  const ull d0 = dirty_n_;
  if (d0) {  // exact re-entry after dormancy: the recorded endpoints are refreshed together with this batch's
    CUDA_CHECK(cudaMemcpy(ends_.data(), dirty_.data(), d0 * sizeof(vid_t), cudaMemcpyDeviceToDevice));
    ++st_.reentries;
    st_.reentry_endpoints += d0;
    dirty_n_ = 0;
  }
  CUDA_CHECK(cudaMemcpy(ctr_.data() + kCEnd, &d0, sizeof(ull), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(ctr_.data() + kCSkip, 0, sizeof(ull)));
  k_endpoints<<<grid_for(count, kBlk), kBlk>>>(s, first, static_cast<uint32_t>(count), eff, g.vlabel,
                                                Proj{pmask_.data(), plabels_.data(), pn_, slot_.data()}, ends_.data(),
                                                ctr_.data(), del_only ? 1 : 0);
  CUDA_CHECK_LAUNCH();
  ull m = 0, sk = 0;
  CUDA_CHECK(cudaMemcpy(&m, ctr_.data() + kCEnd, sizeof(ull), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&sk, ctr_.data() + kCSkip, sizeof(ull), cudaMemcpyDeviceToHost));
  skipped_ += sk;
  if (m > 0) {
    size_t bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(nullptr, bytes, ends_.data(), ends_sorted_.data(), static_cast<int>(m)));
    if (tmp_.size() < bytes) tmp_.resize(bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortKeys(tmp_.data(), bytes, ends_.data(), ends_sorted_.data(), static_cast<int>(m)));
    bytes = 0;
    CUDA_CHECK(cub::DeviceSelect::Unique(nullptr, bytes, ends_sorted_.data(), ends_.data(), ctr_.data() + kCEnd,
                                         static_cast<int>(m)));
    if (tmp_.size() < bytes) tmp_.resize(bytes);
    CUDA_CHECK(cub::DeviceSelect::Unique(tmp_.data(), bytes, ends_sorted_.data(), ends_.data(), ctr_.data() + kCEnd,
                                         static_cast<int>(m)));
    CUDA_CHECK(cudaMemcpy(&m, ctr_.data() + kCEnd, sizeof(ull), cudaMemcpyDeviceToHost));
    expand_.resize(m);
    run(g, ends_.data(), m, lo, hi, true, deferred);
  }
  ++st_.phases;
  st_.deferred_phases += deferred;
  st_.refresh_ms += timer.stop_ms();
}

void SigStore::heal(const DynGraphView& g, ts_t lo, ts_t hi) {
  if (!maybe_stale_ || n_ == 0) return;
  GpuTimer timer;
  timer.start();
  aff_.resize(std::max<uint64_t>(1, nk_));
  CUDA_CHECK(cudaMemset(ctr_.data() + kCStale, 0, sizeof(ull)));
  k_collect_stale<<<grid_for(nk_, kBlk), kBlk>>>(stale_.data(), nk_, vert_.data(), aff_.data(), ctr_.data());
  CUDA_CHECK_LAUNCH();
  ull m = 0;
  CUDA_CHECK(cudaMemcpy(&m, ctr_.data() + kCStale, sizeof(ull), cudaMemcpyDeviceToHost));
  if (m > 0) {
    k_order2<<<grid_for(m, kWpb), kBlk>>>(g, aff_.data(), m, Win{lo, hi}, degout_.data(), degin_.data(), prof_.data(),
                                           rows_.data(), hub_.data(), mask_.data(), stale_.data(),
                                           Proj{pmask_.data(), plabels_.data(), pn_, slot_.data()}, lay_);
    CUDA_CHECK_LAUNCH();
  }
  maybe_stale_ = false;
  ++st_.heals;
  st_.healed_rows += m;
  st_.heal_ms += timer.stop_ms();
}

SigView SigStore::view() const {
  return SigView{rows_.data(), degout_.data(), degin_.data(), mask_.data(), hub_.data(), slot_.data()};
}

SigStats SigStore::stats() const {
  SigStats s = st_;
  std::vector<uint8_t> h(nk_);
  if (nk_) CUDA_CHECK(cudaMemcpy(h.data(), hub_.data(), nk_, cudaMemcpyDeviceToHost));
  s.hubs = static_cast<uint64_t>(std::accumulate(h.begin(), h.end(), 0ull));
  if (nk_) CUDA_CHECK(cudaMemcpy(h.data(), stale_.data(), nk_, cudaMemcpyDeviceToHost));
  s.stale_end = static_cast<uint64_t>(std::accumulate(h.begin(), h.end(), 0ull));
  return s;
}

}  // namespace csm
