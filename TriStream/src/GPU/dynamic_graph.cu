#include "csm/dynamic_graph.cuh"

#include <cub/cub.cuh>
#include <thrust/execution_policy.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/merge.h>
#include <thrust/remove.h>
#include <thrust/tuple.h>

#include <algorithm>

#include "csm/host_graph.hpp"

namespace csm {
namespace {

using ull = unsigned long long;

// counters_ layout
enum : int {
  kCtrRecords = 0,   // new versions emitted in the current batch (reset per batch)
  kCtrEffIns = 1,
  kCtrEffDel = 2,
  kCtrNoopIns = 3,
  kCtrNoopDel = 4,
  kCtrBaseDead = 5,  // dead out-side base entries since last compaction
  kCtrBroken = 7,    // symmetry violations found while resolving (must stay 0)
  kCtrAlive = 8,     // scratch for compaction
  kNumCounters = 16
};

constexpr int kBlock = 256;

inline int blocks_for(uint64_t items, int block = kBlock) {
  uint64_t b = (items + block - 1) / block;
  if (b < 1) b = 1;
  if (b > (1u << 30)) b = 1u << 30;
  return static_cast<int>(b);
}

inline int bit_width(uint64_t x) {  // smallest b with 2^b > x
  int b = 0;
  while (b < 64 && (x >> b) != 0) ++b;
  return b;
}

// batch kernels

__global__ void k_group_keys(DeviceStreamView s, size_t first, uint32_t count, uint32_t directed, uint64_t* keys,
                             uint32_t* vals) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const size_t j = first + i;
  keys[i] = canonical_edge(s.u[j], s.v[j], directed != 0);
  vals[i] = i;
}

// One thread per edge group replays the group's updates in timestamp order.
__global__ void k_resolve(DynGraphView g, DeviceStreamView s, size_t first, uint32_t count, const uint64_t* keys,
                          const uint32_t* vals, uint8_t* eff, uint64_t* rec_edge, ts_t* rec_ins, ts_t* rec_del,
                          label_t* rec_el, ull* ctr) {
  const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const uint64_t key = keys[i];
  if (i > 0 && keys[i - 1] == key) return;  // not the head of its group
  const vid_t a = static_cast<vid_t>(key >> 32), b = static_cast<vid_t>(key & 0xFFFFFFFFu);
  const bool directed = g.directed != 0;
  const BaseSideView& rb = directed ? g.in_base : g.out_base;     // side holding the reverse entry b->a
  const DeltaSideView& rd = directed ? g.in_delta : g.out_delta;

  // version of the edge is alive at the start of the batch
  enum { kNone, kBase, kDelta, kNew } state = kNone;
  eid_t pf = base_find(g.out_base, g.vlabel, a, b), pr = kNotFound;
  uint64_t df = kNotFound, dr = kNotFound;
  ull broken = 0;
  if (pf != kNotFound && g.out_base.del[pf] == kAliveTs) {
    state = kBase;
    pr = base_find(rb, g.vlabel, b, a);
    if (pr == kNotFound || rb.del[pr] != kAliveTs) ++broken;
  } else {
    df = delta_find_open(g.out_delta, a, b);
    if (df != kNotFound) {
      state = kDelta;
      dr = delta_find_open(rd, b, a);
      if (dr == kNotFound) ++broken;
    }
  }

  ts_t open_ins = 0;
  label_t open_el = kNoEdgeLabel;
  ull e_ins = 0, e_del = 0, n_ins = 0, n_del = 0, dead = 0;
  auto emit = [&](ts_t ins, ts_t del, label_t el) {
    const ull slot = atomicAdd(&ctr[kCtrRecords], 1ull);
    rec_edge[slot] = key;
    rec_ins[slot] = ins;
    rec_del[slot] = del;
    rec_el[slot] = el;
  };

  for (uint32_t j = i; j < count && keys[j] == key; ++j) {
    const uint32_t idx = vals[j];  // stable sort
    const size_t sj = first + idx;
    const ts_t t = static_cast<ts_t>(sj + 1);
    uint32_t e = 0;
    if (s.op[sj] == static_cast<uint8_t>(UpdateOp::InsertEdge)) {
      if (state == kNone) {
        state = kNew;
        open_ins = t;
        open_el = s.el[sj];
        e = 1;
        ++e_ins;
      } else {
        ++n_ins;
      }
    } else if (state == kNone) {
      ++n_del;
    } else {
      if (state == kBase) {
        g.out_base.del[pf] = t;
        if (pr != kNotFound) rb.del[pr] = t;
        dead += directed ? 1 : 2;
      } else if (state == kDelta) {
        g.out_delta.del[df] = t;
        if (dr != kNotFound) rd.del[dr] = t;
      } else {
        emit(open_ins, t, open_el);  // inserted and deleted inside this batch
      }
      state = kNone;
      e = 1;
      ++e_del;
    }
    eff[idx] = static_cast<uint8_t>(e);
  }
  if (state == kNew) emit(open_ins, kAliveTs, open_el);

  if (e_ins) atomicAdd(&ctr[kCtrEffIns], e_ins);
  if (e_del) atomicAdd(&ctr[kCtrEffDel], e_del);
  if (n_ins) atomicAdd(&ctr[kCtrNoopIns], n_ins);
  if (n_del) atomicAdd(&ctr[kCtrNoopDel], n_del);
  if (dead) atomicAdd(&ctr[kCtrBaseDead], dead);
  if (broken) atomicAdd(&ctr[kCtrBroken], broken);
}

__global__ void k_expand(uint64_t num, const uint64_t* rec_edge, int mode, uint64_t* nsk, uint32_t* nidx) {
  const uint64_t r = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
  if (r >= num) return;
  const vid_t a = static_cast<vid_t>(rec_edge[r] >> 32), b = static_cast<vid_t>(rec_edge[r] & 0xFFFFFFFFu);
  if (mode == 0) {
    nsk[r] = pack_sk(a, b);
    nidx[r] = static_cast<uint32_t>(r);
  } else if (mode == 1) {
    nsk[r] = pack_sk(b, a);
    nidx[r] = static_cast<uint32_t>(r);
  } else {
    nsk[2 * r] = pack_sk(a, b);
    nidx[2 * r] = static_cast<uint32_t>(r);
    nsk[2 * r + 1] = pack_sk(b, a);
    nidx[2 * r + 1] = static_cast<uint32_t>(r);
  }
}

__global__ void k_gather(uint64_t num, const uint32_t* idx, const ts_t* rec_ins, const ts_t* rec_del,
                         const label_t* rec_el, ts_t* nins, ts_t* ndel, label_t* nel) {
  const uint64_t k = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
  if (k >= num) return;
  const uint32_t r = idx[k];
  nins[k] = rec_ins[r];
  ndel[k] = rec_del[r];
  nel[k] = rec_el[r];
}

struct DeadBefore {  // delta version that died before the current batch
  ts_t t0;
  template <class Tuple>
  __host__ __device__ bool operator()(const Tuple& x) const {
    const ts_t d = thrust::get<2>(x);
    return d != kAliveTs && d <= t0;
  }
};

// compaction kernels

__device__ __forceinline__ uint64_t make_comp(vid_t src, label_t l, vid_t dst, int bv, int bl) {
  return (static_cast<uint64_t>(src) << (bv + bl)) | (static_cast<uint64_t>(l) << bv) | dst;
}

// Warp per vertex
__global__ void k_comp_base(BaseSideView b, vid_t n, int bv, int bl, uint64_t* comp, label_t* cel) {
  const uint64_t warp = (blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x) / 32;
  const int lane = threadIdx.x & 31;
  const uint64_t warps = (gridDim.x * static_cast<uint64_t>(blockDim.x)) / 32;
  for (uint64_t v = warp; v < n; v += warps) {
    for (eid_t p = b.offsets[v] + lane; p < b.offsets[v + 1]; p += 32) {
      const adj_t k = b.keys[p];
      comp[p] = b.del[p] == kAliveTs ? make_comp(static_cast<vid_t>(v), adj_label(k), adj_vertex(k), bv, bl) : ~0ull;
      cel[p] = b.el ? b.el[p] : kNoEdgeLabel;
    }
  }
}

__global__ void k_comp_delta(DeltaSideView d, const label_t* vlabel, int bv, int bl, uint64_t* comp, label_t* cel) {
  const uint64_t i = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
  if (i >= d.size) return;
  const vid_t src = static_cast<vid_t>(d.sk[i] >> 32), dst = static_cast<vid_t>(d.sk[i] & 0xFFFFFFFFu);
  comp[i] = d.del[i] == kAliveTs ? make_comp(src, vlabel[dst], dst, bv, bl) : ~0ull;
  cel[i] = d.el[i];
}

// Sorted composites -> degree histogram, adjacency keys.
__global__ void k_comp_unpack(const uint64_t* comp, uint64_t alive, int bv, int bl, eid_t* deg, adj_t* keys) {
  const uint64_t i = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x;
  if (i >= alive) return;
  const uint64_t c = comp[i];
  const vid_t src = static_cast<vid_t>(c >> (bv + bl));
  const label_t l = static_cast<label_t>((c >> bv) & ((1ull << bl) - 1));
  const vid_t dst = static_cast<vid_t>(c & ((1ull << bv) - 1));
  atomicAdd(reinterpret_cast<ull*>(&deg[src]), 1ull);
  keys[i] = adj_key(l, dst);
}

__global__ void k_count_alive(const uint64_t* comp, uint64_t total, ull* alive) {
  ull local = 0;
  for (uint64_t i = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x; i < total;
       i += gridDim.x * static_cast<uint64_t>(blockDim.x))
    local += comp[i] != ~0ull;
  if (local) atomicAdd(alive, local);
}

// snapshot fingerprint kernels

__device__ __forceinline__ void account(vid_t src, vid_t dst, int mode, ull& edges, ull& hash, ull& entries) {
  ++entries;
  if (mode == 0 && src > dst) return;
  ++edges;
  hash += edge_print(mode == 2 ? pack_sk(dst, src) : pack_sk(src, dst));
}

__global__ void k_print_base(BaseSideView b, vid_t n, ts_t t, int mode, ull* acc) {
  const uint64_t warp = (blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x) / 32;
  const int lane = threadIdx.x & 31;
  const uint64_t warps = (gridDim.x * static_cast<uint64_t>(blockDim.x)) / 32;
  ull edges = 0, hash = 0, entries = 0;
  for (uint64_t v = warp; v < n; v += warps)
    for (eid_t p = b.offsets[v] + lane; p < b.offsets[v + 1]; p += 32)
      if (b.del[p] > t) account(static_cast<vid_t>(v), adj_vertex(b.keys[p]), mode, edges, hash, entries);
  atomicAdd(&acc[0], edges);
  atomicAdd(&acc[1], hash);
  atomicAdd(&acc[2], entries);
}

__global__ void k_print_delta(DeltaSideView d, ts_t t, int mode, ull* acc) {
  ull edges = 0, hash = 0, entries = 0;
  for (uint64_t i = blockIdx.x * static_cast<uint64_t>(blockDim.x) + threadIdx.x; i < d.size;
       i += gridDim.x * static_cast<uint64_t>(blockDim.x))
    if (d.ins[i] <= t && d.del[i] > t)
      account(static_cast<vid_t>(d.sk[i] >> 32), static_cast<vid_t>(d.sk[i] & 0xFFFFFFFFu), mode, edges, hash,
              entries);
  atomicAdd(&acc[0], edges);
  atomicAdd(&acc[1], hash);
  atomicAdd(&acc[2], entries);
}

void grow_tmp(DevBuf<uint8_t>& tmp, size_t bytes) {
  if (tmp.size() < bytes) tmp.resize(bytes);
}

}  // namespace

// DynamicGraph

void DynamicGraph::upload(const HostGraph& g) {
  n_ = g.n;
  directed_ = g.directed;
  has_edge_labels_ = g.has_edge_labels;
  vlabel_.upload(g.vlabel.data(), g.vlabel.size());
  label_t max_label = 0;
  for (label_t l : g.vlabel) max_label = std::max(max_label, l);
  bits_v_ = bit_width(n_);
  bits_l_ = std::max(1, bit_width(max_label));
  if (2 * bits_v_ + bits_l_ > 64)
    fatal("compaction key does not fit 64 bits (%d vertex bits, %d label bits)", bits_v_, bits_l_);

  auto load_side = [&](Side& s, const HostCSR& csr) {
    s.offsets.upload(csr.offsets.data(), csr.offsets.size());
    s.keys.upload(csr.keys.data(), csr.keys.size());
    s.del.resize(csr.keys.size());
    s.del.fill_bytes(0xFF);  // kAliveTs
    if (has_edge_labels_) s.el.upload(csr.elabels.data(), csr.elabels.size());
    s.base_entries = csr.keys.size();
    s.dsize = 0;
  };
  load_side(out_, g.out);
  if (directed_) load_side(in_, g.in);
  counters_.resize(kNumCounters);
  counters_.fill_bytes(0);
  applied_ = 0;
  stats_ = DynamicStats{};
}

BaseSideView DynamicGraph::base_side_view(const Side& s) const {
  return BaseSideView{s.offsets.data(), s.keys.data(), s.del.data(), has_edge_labels_ ? s.el.data() : nullptr,
                      s.base_entries};
}

DeltaSideView DynamicGraph::delta_side_view(const Side& s) const {
  return DeltaSideView{s.dsk.data(), s.dins.data(), s.ddel.data(), s.del_.data(), s.dsize};
}

DynGraphView DynamicGraph::view() const {
  DynGraphView v{};
  v.n = n_;
  v.directed = directed_;
  v.has_edge_labels = has_edge_labels_;
  v.vlabel = vlabel_.data();
  v.out_base = base_side_view(out_);
  v.out_delta = delta_side_view(out_);
  v.in_base = directed_ ? base_side_view(in_) : v.out_base;
  v.in_delta = directed_ ? delta_side_view(in_) : v.out_delta;
  return v;
}

void DynamicGraph::apply_batch(const DeviceStreamView& s, size_t first, size_t count) {
  if (first != applied_) fatal("internal: batch starts at %zu but %u updates were applied", first, applied_);
  if (count == 0) return;
  const uint32_t c = static_cast<uint32_t>(count);
  gkeys_.resize(c);
  gkeys_sorted_.resize(c);
  gvals_.resize(c);
  gvals_sorted_.resize(c);
  eff_.resize(c);
  rec_edge_.resize(c);
  rec_ins_.resize(c);
  rec_del_.resize(c);
  rec_el_.resize(c);

  // 1. group updates by edge 
  k_group_keys<<<blocks_for(c), kBlock>>>(s, first, c, directed_, gkeys_.data(), gvals_.data());
  CUDA_CHECK_LAUNCH();
  const int end_bit = 32 + bits_v_;
  size_t tmp_bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, tmp_bytes, gkeys_.data(), gkeys_sorted_.data(), gvals_.data(),
                                             gvals_sorted_.data(), static_cast<int>(c), 0, end_bit));
  grow_tmp(cub_tmp_, tmp_bytes);
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(cub_tmp_.data(), tmp_bytes, gkeys_.data(), gkeys_sorted_.data(),
                                             gvals_.data(), gvals_sorted_.data(), static_cast<int>(c), 0, end_bit));

  // 2. replay each edge's updates in order
  CUDA_CHECK(cudaMemset(counters_.data() + kCtrRecords, 0, sizeof(ull)));
  k_resolve<<<blocks_for(c), kBlock>>>(view(), s, first, c, gkeys_sorted_.data(), gvals_sorted_.data(), eff_.data(),
                                       rec_edge_.data(), rec_ins_.data(), rec_del_.data(), rec_el_.data(),
                                       counters_.data());
  CUDA_CHECK_LAUNCH();
  ull records = 0;
  CUDA_CHECK(cudaMemcpy(&records, counters_.data() + kCtrRecords, sizeof(ull), cudaMemcpyDeviceToHost));

  // 3. merge the new versions into the delta of each side
  const ts_t t0 = static_cast<ts_t>(first);
  if (directed_) {
    merge_new_records(out_, records, t0, 0);
    merge_new_records(in_, records, t0, 1);
  } else {
    merge_new_records(out_, records, t0, 2);
  }
  applied_ += c;
  stats_.max_delta_entries = std::max<uint64_t>(stats_.max_delta_entries, out_.dsize);
}

void DynamicGraph::merge_new_records(Side& s, uint64_t num_records, ts_t t0, int mode) {
  const uint64_t k = mode == 2 ? 2 * num_records : num_records;
  if (k > 0) {
    s.nsk.resize(k);
    s.nsk_sorted.resize(k);
    s.nidx.resize(k);
    s.nidx_sorted.resize(k);
    s.nins.resize(k);
    s.ndel.resize(k);
    s.nel.resize(k);
    k_expand<<<blocks_for(num_records), kBlock>>>(num_records, rec_edge_.data(), mode, s.nsk.data(), s.nidx.data());
    CUDA_CHECK_LAUNCH();
    size_t tmp_bytes = 0;
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, tmp_bytes, s.nsk.data(), s.nsk_sorted.data(), s.nidx.data(),
                                               s.nidx_sorted.data(), static_cast<int>(k)));
    grow_tmp(cub_tmp_, tmp_bytes);
    CUDA_CHECK(cub::DeviceRadixSort::SortPairs(cub_tmp_.data(), tmp_bytes, s.nsk.data(), s.nsk_sorted.data(),
                                               s.nidx.data(), s.nidx_sorted.data(), static_cast<int>(k)));
    k_gather<<<blocks_for(k), kBlock>>>(k, s.nidx_sorted.data(), rec_ins_.data(), rec_del_.data(), rec_el_.data(),
                                        s.nins.data(), s.ndel.data(), s.nel.data());
    CUDA_CHECK_LAUNCH();
  }

  // Drop versions that died before this batch 
  uint64_t m = s.dsize;
  if (m > 0) {
    auto old_begin = thrust::make_zip_iterator(thrust::make_tuple(s.dsk.data(), s.dins.data(), s.ddel.data(),
                                                                  s.del_.data()));
    auto old_end = thrust::remove_if(thrust::device, old_begin, old_begin + m, DeadBefore{t0});
    m = static_cast<uint64_t>(old_end - old_begin);
  }
  if (k == 0) {
    s.dsize = m;
    return;
  }
  const uint64_t total = m + k;
  s.dsk2.resize(total);
  s.dins2.resize(total);
  s.ddel2.resize(total);
  s.del2.resize(total);
  thrust::merge_by_key(
      thrust::device, s.dsk.data(), s.dsk.data() + m, s.nsk_sorted.data(), s.nsk_sorted.data() + k,
      thrust::make_zip_iterator(thrust::make_tuple(s.dins.data(), s.ddel.data(), s.del_.data())),
      thrust::make_zip_iterator(thrust::make_tuple(s.nins.data(), s.ndel.data(), s.nel.data())), s.dsk2.data(),
      thrust::make_zip_iterator(thrust::make_tuple(s.dins2.data(), s.ddel2.data(), s.del2.data())));
  s.dsk.swap(s.dsk2);
  s.dins.swap(s.dins2);
  s.ddel.swap(s.ddel2);
  s.del_.swap(s.del2);
  s.dsize = total;
}

bool DynamicGraph::maybe_compact() {
  ull dead = 0;
  CUDA_CHECK(cudaMemcpy(&dead, counters_.data() + kCtrBaseDead, sizeof(ull), cudaMemcpyDeviceToHost));
  const double garbage = static_cast<double>(out_.dsize + dead);
  if (garbage <= compact_ratio_ * static_cast<double>(std::max<eid_t>(out_.base_entries, 1024))) return false;
  compact();
  return true;
}

void DynamicGraph::compact() {
  WallTimer timer;
  compact_side(out_);
  if (directed_) compact_side(in_);
  CUDA_CHECK(cudaMemset(counters_.data() + kCtrBaseDead, 0, sizeof(ull)));
  CUDA_CHECK(cudaDeviceSynchronize());
  ++stats_.compactions;
  stats_.compaction_ms += timer.ms();
}

// New base = every entry visible "now" (alive base + alive delta), re-sorted by (src, label, dst).
void DynamicGraph::compact_side(Side& s) {
  const uint64_t E = s.base_entries, D = s.dsize, T = E + D;
  DevBuf<uint64_t> comp, comp_sorted;
  DevBuf<label_t> cel, cel_sorted;
  comp.resize(T);
  comp_sorted.resize(T);
  cel.resize(T);
  cel_sorted.resize(T);
  const BaseSideView b = base_side_view(s);
  const DeltaSideView d = delta_side_view(s);
  if (E > 0) {
    k_comp_base<<<blocks_for(static_cast<uint64_t>(n_) * 32), kBlock>>>(b, n_, bits_v_, bits_l_, comp.data(), cel.data());
    CUDA_CHECK_LAUNCH();
  }
  if (D > 0) {
    k_comp_delta<<<blocks_for(D), kBlock>>>(d, vlabel_.data(), bits_v_, bits_l_, comp.data() + E, cel.data() + E);
    CUDA_CHECK_LAUNCH();
  }
  CUDA_CHECK(cudaMemset(counters_.data() + kCtrAlive, 0, sizeof(ull)));
  k_count_alive<<<blocks_for(T), kBlock>>>(comp.data(), T, counters_.data() + kCtrAlive);
  CUDA_CHECK_LAUNCH();
  ull alive = 0;
  CUDA_CHECK(cudaMemcpy(&alive, counters_.data() + kCtrAlive, sizeof(ull), cudaMemcpyDeviceToHost));

  const int end_bit = 2 * bits_v_ + bits_l_;
  size_t tmp_bytes = 0;
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(nullptr, tmp_bytes, comp.data(), comp_sorted.data(), cel.data(),
                                             cel_sorted.data(), static_cast<int>(T), 0, end_bit));
  grow_tmp(cub_tmp_, tmp_bytes);
  CUDA_CHECK(cub::DeviceRadixSort::SortPairs(cub_tmp_.data(), tmp_bytes, comp.data(), comp_sorted.data(), cel.data(),
                                             cel_sorted.data(), static_cast<int>(T), 0, end_bit));
  comp.release();
  cel.release();

  // degrees -> offsets; composites -> keys
  DevBuf<eid_t> deg;
  deg.resize(static_cast<size_t>(n_) + 1);
  deg.fill_bytes(0);
  DevBuf<adj_t> keys;
  keys.resize(alive);
  if (alive > 0) {
    k_comp_unpack<<<blocks_for(alive), kBlock>>>(comp_sorted.data(), alive, bits_v_, bits_l_, deg.data(), keys.data());
    CUDA_CHECK_LAUNCH();
  }
  s.offsets.resize(static_cast<size_t>(n_) + 1);
  tmp_bytes = 0;
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, tmp_bytes, deg.data(), s.offsets.data(), n_ + 1));
  grow_tmp(cub_tmp_, tmp_bytes);
  CUDA_CHECK(cub::DeviceScan::ExclusiveSum(cub_tmp_.data(), tmp_bytes, deg.data(), s.offsets.data(), n_ + 1));

  s.keys.swap(keys);
  s.del.resize(alive);
  s.del.fill_bytes(0xFF);
  if (has_edge_labels_) {
    s.el.resize(alive);
    if (alive) CUDA_CHECK(cudaMemcpy(s.el.data(), cel_sorted.data(), alive * sizeof(label_t), cudaMemcpyDeviceToDevice));
  }
  s.base_entries = alive;
  s.dsize = 0;
}

SnapshotPrint DynamicGraph::snapshot_print(ts_t t) const {
  DevBuf<ull> acc;
  acc.resize(6);
  acc.fill_bytes(0);
  const int base_blocks = blocks_for(static_cast<uint64_t>(n_) * 32);
  const int out_mode = directed_ ? 1 : 0;
  k_print_base<<<std::min(base_blocks, 8192), kBlock>>>(base_side_view(out_), n_, t, out_mode, acc.data());
  CUDA_CHECK_LAUNCH();
  if (out_.dsize) {
    k_print_delta<<<std::min(blocks_for(out_.dsize), 8192), kBlock>>>(delta_side_view(out_), t, out_mode, acc.data());
    CUDA_CHECK_LAUNCH();
  }
  if (directed_) {
    k_print_base<<<std::min(base_blocks, 8192), kBlock>>>(base_side_view(in_), n_, t, 2, acc.data() + 3);
    CUDA_CHECK_LAUNCH();
    if (in_.dsize) {
      k_print_delta<<<std::min(blocks_for(in_.dsize), 8192), kBlock>>>(delta_side_view(in_), t, 2, acc.data() + 3);
      CUDA_CHECK_LAUNCH();
    }
  }
  ull h[6];
  CUDA_CHECK(cudaMemcpy(h, acc.data(), sizeof(h), cudaMemcpyDeviceToHost));
  SnapshotPrint p;
  p.edges = h[0];
  p.hash = h[1];
  p.out_entries = h[2];
  p.in_edges = directed_ ? h[3] : h[0];
  p.in_hash = directed_ ? h[4] : h[1];
  return p;
}

DynamicStats DynamicGraph::stats() const {
  ull c[kNumCounters];
  CUDA_CHECK(cudaMemcpy(c, counters_.data(), sizeof(c), cudaMemcpyDeviceToHost));
  DynamicStats s = stats_;
  s.eff_insertions = c[kCtrEffIns];
  s.eff_deletions = c[kCtrEffDel];
  s.noop_insertions = c[kCtrNoopIns];
  s.noop_deletions = c[kCtrNoopDel];
  if (c[kCtrBroken]) std::fprintf(stderr, "[csm] ERROR: %llu adjacency symmetry violations while applying updates\n", c[kCtrBroken]);
  return s;
}

}  // namespace csm
