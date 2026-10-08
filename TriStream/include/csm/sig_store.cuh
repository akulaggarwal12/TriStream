// S2 maintenance on the GPU

#pragma once

#include <cstdint>
#include <vector>

#include "csm/dev_buffer.cuh"
#include "csm/device_graph.cuh"
#include "csm/dynamic_graph.cuh"
#include "csm/sig_host.hpp"
#include "csm/signature.hpp"

namespace csm {

// slot[v] = compact row of v
struct SigView {
  const uint32_t* rows;    // nk * kWords (nk = query-label vertices)
  const uint32_t* degout;
  const uint32_t* degin;
  const uint32_t* mask;
  const uint8_t* hub;
  const uint32_t* slot;    // vertex -> compact row
};

struct SigStats {
  double build_ms = 0.0, refresh_ms = 0.0, heal_ms = 0.0;
  uint64_t phases = 0, endpoint_rows = 0, affected_rows = 0, hubs = 0;
  uint64_t deferred_phases = 0;  // phases run in deferred (exact-or-abstain) mode
  uint64_t dormant_batches = 0, reentries = 0, reentry_endpoints = 0;  // S2 dormancy
  uint64_t deferred_rows = 0;    // endpoint -> neighbour group deferrals (mask bit + stale flag instead of a refresh)
  uint64_t heals = 0, healed_rows = 0;
  uint64_t stale_end = 0;        // stale rows at the end of the run
};

class SigStore {
 public:
  SigStore(vid_t n, uint32_t tau, bool use_el) : n_(n), tau_(tau), use_el_(use_el) {}
  // query projection: label_mask[l] = 1 for every query label (dense label ids); an empty mask = no projection.
  void set_projection(const std::vector<uint8_t>& label_mask);
  void set_layout(const KeyLayout& k);
  const sig::Layout& layout() const { return lay_; }
  uint64_t skipped_updates() const { return skipped_; }

  void build(const DynGraphView& g, ts_t t);  // all rows on snapshot t
  // deferred = exact-or-abstain mode for this phase (see above)
  void refresh(const DynGraphView& g, const DeviceStreamView& s, size_t first, size_t count, const uint8_t* eff,
               ts_t lo, ts_t hi, bool deferred = false, bool del_only = false);
  // recompute every stale row exactly on window (lo, hi]; no-op when nothing can be stale
  void heal(const DynGraphView& g, ts_t lo, ts_t hi);
  // dormant batch: no maintenance, only record the endpoints of its effective (projected) updates
  void note_dormant(const DeviceStreamView& s, size_t first, size_t count, const uint8_t* eff, const label_t* vlabel);
  uint64_t dirty_count() const { return dirty_n_; }  // endpoints recorded while dormant (with duplicates)
  SigView view() const;

  SigStats stats() const;

 private:
  void run(const DynGraphView& g, const vid_t* list, uint64_t count, ts_t lo, ts_t hi, bool expand, bool deferred);

  vid_t n_;
  uint32_t tau_;
  bool use_el_;
  DevBuf<uint32_t> rows_, degout_, degin_, prof_;
  DevBuf<uint32_t> slot_;     // vertex -> compact row (kNoSlot: no query label, never read)
  DevBuf<vid_t> vert_;        // compact row -> vertex
  uint64_t nk_ = 0;           // rows stored
  DevBuf<uint32_t> mask_; DevBuf<uint8_t> hub_, stale_;  // mask_ is padded to a multiple of 4 bytes (32-bit atomicOr)
  sig::Layout lay_;
  DevBuf<uint8_t> klq_, kle_, kmfar_, kmslot_;
  DevBuf<int8_t> ks1_, kst_, kmid_, kfar_;
  DevBuf<int16_t> kpair_;
  DevBuf<uint32_t> kpm_;
  DevBuf<uint16_t> kmoff_;
  bool maybe_stale_ = false;
  DevBuf<uint8_t> pmask_;     // per label: 1 = query label
  DevBuf<label_t> plabels_;   // the query labels, ascending
  int pn_ = 0;                // 0 = no projection
  uint64_t skipped_ = 0;      // effective updates skipped by the projection (an endpoint without a query label)
  DevBuf<vid_t> dirty_;       // endpoints recorded while dormant (with duplicates)
  uint64_t dirty_n_ = 0;
  DevBuf<vid_t> ends_, ends_sorted_, aff_, aff_sorted_;
  DevBuf<uint64_t> cnt_, off_;
  DevBuf<uint8_t> expand_;
  DevBuf<uint8_t> tmp_;
  DevBuf<unsigned long long> ctr_;
  SigStats st_;
};

}  // namespace csm
