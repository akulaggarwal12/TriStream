// The dynamic, versioned GPU graph: every task reads the edge versions visible in its snapshot.
#pragma once

#include <cstdint>

#include "csm/common.hpp"
#include "csm/dev_buffer.cuh"
#include "csm/device_graph.cuh"
#include "csm/edge_key.hpp"

namespace csm {

constexpr ts_t kAliveTs = 0xFFFFFFFFu;
constexpr eid_t kNotFound = ~eid_t(0);

struct BaseSideView {
  const eid_t* offsets;  // n + 1
  const adj_t* keys;     // sorted 
  ts_t* del;             // deletion time
  const label_t* el;     // edge labels
  eid_t entries;
};

struct DeltaSideView {
  const uint64_t* sk;  // sorted
  const ts_t* ins;
  ts_t* del;
  const label_t* el;
  uint64_t size;
};

struct DynGraphView {
  vid_t n;
  uint32_t directed;
  uint32_t has_edge_labels;
  const label_t* vlabel;
  BaseSideView out_base, in_base;    
  DeltaSideView out_delta, in_delta;
};

#ifdef __CUDACC__
// Position of (src -> dst) in the base list
__device__ __forceinline__ eid_t base_find(const BaseSideView& b, const label_t* vlabel, vid_t src, vid_t dst) {
  const adj_t k = adj_key(vlabel[dst], dst);
  const eid_t e = b.offsets[src + 1];
  const eid_t p = lower_bound_key(b.keys, b.offsets[src], e, k);
  return (p < e && b.keys[p] == k) ? p : kNotFound;
}
// First delta index with sk >= x.
__device__ __forceinline__ uint64_t delta_lower_bound(const DeltaSideView& d, uint64_t x) {
  uint64_t lo = 0, hi = d.size;
  while (lo < hi) {
    const uint64_t mid = lo + ((hi - lo) >> 1);
    if (d.sk[mid] < x) lo = mid + 1;
    else hi = mid;
  }
  return lo;
}
// Index of the version of (src -> dst) that is alive
__device__ __forceinline__ uint64_t delta_find_open(const DeltaSideView& d, vid_t src, vid_t dst) {
  const uint64_t x = pack_sk(src, dst);
  for (uint64_t i = delta_lower_bound(d, x); i < d.size && d.sk[i] == x; ++i)
    if (d.del[i] == kAliveTs) return i;
  return kNotFound;
}
#endif

struct DynamicStats {
  uint64_t eff_insertions = 0;
  uint64_t eff_deletions = 0;
  uint64_t noop_insertions = 0;  // edge already present
  uint64_t noop_deletions = 0;   // edge absent
  uint64_t compactions = 0;
  uint64_t max_delta_entries = 0;
  double compaction_ms = 0.0;
};

// Visible-edge fingerprint of snapshot 
struct SnapshotPrint {
  uint64_t edges = 0;
  uint64_t hash = 0;
  uint64_t out_entries = 0;  // undirected graphs have double edges
  uint64_t in_edges = 0;     // in edges for directed only
  uint64_t in_hash = 0;
};

class DynamicGraph {
 public:
  explicit DynamicGraph(double compact_ratio = 0.25) : compact_ratio_(compact_ratio) {}

  void upload(const HostGraph& g);
  DynGraphView view() const;

  // Applies stream updates
  void apply_batch(const DeviceStreamView& s, size_t first, size_t count);
  // Compacts if delta + dead base entries exceed compact_ratio of the base
  bool maybe_compact();
  void compact();

  SnapshotPrint snapshot_print(ts_t t) const;
  DynamicStats stats() const;
  // Effect flags of the last applied batch
  const uint8_t* batch_effects() const { return eff_.data(); }

 private:
  struct Side {
    DevBuf<eid_t> offsets;
    DevBuf<adj_t> keys;
    DevBuf<ts_t> del;
    DevBuf<label_t> el;
    eid_t base_entries = 0;
    // delta 
    DevBuf<uint64_t> dsk, dsk2;
    DevBuf<ts_t> dins, dins2, ddel, ddel2;
    DevBuf<label_t> del_, del2;
    uint64_t dsize = 0;
    // new records of the current batch
    DevBuf<uint64_t> nsk, nsk_sorted;
    DevBuf<uint32_t> nidx, nidx_sorted;
    DevBuf<ts_t> nins, ndel;
    DevBuf<label_t> nel;
  };

  BaseSideView base_side_view(const Side& s) const;
  DeltaSideView delta_side_view(const Side& s) const;
  void merge_new_records(Side& s, uint64_t num_new, ts_t batch_start, int mode);
  void compact_side(Side& s);

  double compact_ratio_;
  vid_t n_ = 0;
  bool directed_ = false;
  bool has_edge_labels_ = false;
  int bits_v_ = 0, bits_l_ = 0;  
  DevBuf<label_t> vlabel_;
  Side out_, in_;
  ts_t applied_ = 0;  

  // batch scratch
  DevBuf<uint64_t> gkeys_, gkeys_sorted_;
  DevBuf<uint32_t> gvals_, gvals_sorted_;
  DevBuf<uint8_t> eff_;
  DevBuf<uint64_t> rec_edge_;  
  DevBuf<ts_t> rec_ins_, rec_del_;
  DevBuf<label_t> rec_el_;
  DevBuf<unsigned long long> counters_;  
  DevBuf<uint8_t> cub_tmp_;
  DynamicStats stats_;
};

}  // namespace csm
