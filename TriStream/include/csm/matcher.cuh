// Exact GPU matcher of TriWarp: S2 triage, S1 cost model, S3 split and search, and the controller
#pragma once

#include <cstdint>
#include <vector>

#include "csm/dev_buffer.cuh"
#include "csm/device_graph.cuh"
#include "csm/dynamic_graph.cuh"
#include "csm/match_device.cuh"
#include "csm/match_plan.hpp"
#include "csm/sig_host.hpp"
#include "csm/sig_store.cuh"

namespace csm {

struct S1Config {
  float decay = 0.5f;       // dynamic decay weight of the past batches
  int explore = 16;         // 1 of `explore` tiny tasks probes another plan (keeps every plan's rates alive)
  uint64_t probe_budget = 64;  // minimum chunks a bet may spend before it is aborted and re-run with RI
  float bet_scale = 3.f;       // a bet may spend max(probe_budget, bet_scale x its predicted cost) chunks
  int min_k = 5;               // plan switches / exploration only for queries with >= min_k vertices (trust_all: any)
  uint64_t s2_min_tested = 1000;  // S2 sleep / wake decisions need this many label-tested tasks in the batch
                                  // (1 lets the controller act on tiny graphs)
};

struct S3Config {
  uint64_t min_piece = 64;     // lower bound of the piece budget B (chunks)
  float share = 4.f;           // B = max(min_piece, W / (share · warps))
  int max_pieces = 1024;       // per task
};

struct MatchStats {
  uint64_t tasks = 0;          // (effective update, anchor) pairs
  uint64_t label_pass = 0;     // whose endpoint labels match the anchor
  uint64_t s2_pass = 0;        // that also pass the S2 triage (= label_pass when triage is off)
  uint64_t anchored = 0;       // whose level-1 query edges are visible in G_θ
  uint64_t closure_fail = 0;   // rejected by the S2 edge closure
  uint64_t closure_chunks = 0; // 32-entry chunks scanned by the closure test
  uint64_t gate_tested = 0;    // DFS candidates that reached the signature gate
  uint64_t gate_rejects = 0;   // DFS candidates that passed every edge check but failed the signature gate
  uint64_t chunks = 0;         // DFS candidate chunks (32 list entries) scanned = search work units
  uint64_t timeouts = 0;       // tasks aborted by the per-task budget (their updates are flagged)
  uint64_t matches_pos = 0;    // total ΔM⁺
  uint64_t matches_neg = 0;    // total ΔM⁻
  uint64_t updates_with_matches = 0;
  uint64_t timed_out_updates = 0;
  double kernel_ms = 0.0;
  // S1
  uint64_t kind_chosen[3] = {0, 0, 0};  // tasks run with an RI / rare-first / leaf-last plan
  uint64_t size_class[3] = {0, 0, 0};   // predicted tiny (< 2 chunks) / normal / heavy (>= 64 chunks)
  uint64_t explored = 0;                // exploration probes (tiny tasks, untrusted plan)
  uint64_t switched = 0;                // tasks moved off RI by the cost model (also run as bounded bets)
  uint64_t local_used = 0;              // predictions that used an S2 / endpoint local estimate at level 3
  uint64_t probe_aborts = 0;            // bets that overran their budget and were re-run with RI
  uint64_t s2_off_batches = 0;          // batches in which the controller had switched the gate or closure off
  double s1_ms = 0.0, s3_ms = 0.0;
  uint64_t bindings = 0;      // partial matches extended 
  uint64_t bet_waste = 0;      // chunks spent by aborted bets (the realised regret of S1, beyond the baseline)
  uint64_t abort_counted = 0;  // aborted bets that had already counted matches (their partial count is discarded)
  double ratio_p50 = 0, ratio_p90 = 0, ratio_p99 = 0;  // (actual + 1) / (prediction + 1), sampled tasks
  uint64_t ratio_n = 0;
  uint64_t s3_split = 0, s3_pieces = 0;  // tasks split by S3, pieces they became
  double s3_budget_avg = 0;              // average piece budget B (chunks) over batches with splitting
  double max_task_ms = 0, sum_task_ms = 0;  // longest single task / piece, and the sum over all (GPU clock)
  uint64_t dead_tasks = 0;      // tasks stopped or never started because the per-query time limit was reached
  uint64_t flat_batches = 0;    // batches in which S1 / S3 had nothing to decide (no sort, no split)
  uint64_t s2_sleeps = 0;       // times the controller put S2 to sleep (cost-benefit)
  uint64_t s2_wakes = 0;        // times a count-only probe found S2 worth waking
  uint64_t s2_probe_skips = 0;  // dormant batches without a probe (back-off)
  uint64_t s1_dormant_batches = 0;  // batches in which S1 slept (no prediction at all)
};

class Matcher {
 public:
  Matcher(const MatchPlan& plan, const S1Config& s1 = S1Config{}, const S3Config& s3 = S3Config{});
  // per-query time limit
  void set_deadline(uint64_t rel_ns) { deadline_rel_ = rel_ns; }
  void set_s2(const SigTable& qsig, const sig::Layout& lay);
  // controller: S2-maintenance decision
  bool gate_active_next() const;
  // S2 dormancy
  bool s2_needed_next() const;
  void end_batch_s2(double s2_ms, uint64_t reentry_eps = 0, uint64_t backlog_eps = 0);
  // the next match_batch runs with S2 dormant
  void set_s2_probe(bool on) { s2_probe_ = on; }

  // Matches updates
  void match_batch(const DynGraphView& g, const DeviceStreamView& s, size_t first, size_t count, const uint8_t* eff,
                   const SigView* sv);

  MatchStats stats() const;

 private:
  void learn();  // fold this batch's per-(plan, level) work into the decayed rates
  uint64_t split(uint64_t n, float B, float W, float target);  // S3: returns the number of task pieces now in tasks_ / cost_ / range_
  void cost_stats(uint64_t n, float& W, float& M);
  void sort_by_cost(uint64_t n);

  int k_, anchors_, plans_;
  S1Config s1_;
  S3Config s3_;
  uint64_t deadline_rel_ = 0;
  uint64_t s1_sleep_until_ = 0;
  int flat_streak_ = 0;
  unsigned long long last_s1_dev_ = 0;  // plan deviations (other plan / switch / exploration) seen by S1 so far
  uint64_t s2_sleep_until_ = 0;
  unsigned long long b_tested_ = 0, b_rejected_ = 0, b_anchored_ = 0, last_label_ = 0, last_s2pass_ = 0, last_anch_ = 0;
  bool b_s2_used_ = false, b_probe_ = false, s2_probe_ = false;
  unsigned long long last_probe_ = 0, b_gate_rej_ = 0, b_chunks_ = 0, last_gate_rej2_ = 0, last_chunks2_ = 0;
  size_t b_count_ = 0;
  unsigned long long last_cyc_ = 0, last_zcyc_ = 0, last_zn_ = 0, b_cyc_ = 0, b_zcyc_ = 0, b_zn_ = 0;
  double s2_ms_per_update_ = 0.0;   // S2 maintenance per update while awake (re-entry batches excluded)
  bool s2_cost_known_ = false;       // the first awake measurement initialises it (not averaged with 0)
  double reentry_ms_per_ep_ = -1.0;  // re-entry cost per recorded endpoint (< 0: not measured yet)
  double probe_bias_ = 1.0;          // true / probe reject rate, measured at each wake (stale rows over-reject)
  bool bias_known_ = false;
  double last_probe_rate_ = 0.0;     // reject rate of the last probe batch
  int wake_streak_ = 0;              // consecutive probe batches that met the wake condition
  int probe_every_ = 1;              // probe back-off: a dormant batch probes every probe_every_ batches
  uint64_t next_probe_batch_ = 0;
  bool probe_ran_ = false, b_probe_ran_ = false;
  double b_kernel_ms_ = 0.0;
  DevBuf<float> kappa_;                   // per plan calibration factor
  DevBuf<double> calib_;                   // this batch, per plan {Σ actual, Σ raw predicted} (sampled)
  std::vector<double> dCalA_, dCalP_;
  DevBuf<unsigned long long> hist_;        // log2 ratio histogram (half-octave buckets)
  DevBuf<uint32_t> len2_, len2_sorted_;
  DevBuf<float> wsum_;
  DevBuf<uint64_t> npieces_, range_, range_sorted_, poff_;
  DevBuf<uint32_t> pidx_, pidx_sorted_;
  uint64_t s3_batches_ = 0;
  double s3_budget_sum_ = 0;
  int clock_khz_ = 1;
  bool any_gate_ = false;
  uint64_t batch_no_ = 0, gate_off_until_ = 0, clo_off_until_ = 0;
  uint64_t last_gate_tested_ = 0, last_gate_rej_ = 0, last_anchored_ = 0, last_clo_fail_ = 0;
  DevBuf<uint16_t> plan_off_, plan_anchor_;
  DevBuf<uint8_t> plan_kind_;
  std::vector<float> prior_c_, prior_s_, prior_r_;
  std::vector<int> lfrom_;
  std::vector<double> dE_, dC_, dS_, dLen_;  // decayed per-(plan, level) sums: entries, chunks, survivors, list length
  DevBuf<float> rate_c_, rate_s_, rate_r_, rate_l_;
  DevBuf<uint64_t> shape_, slice_, slice_sorted_;
  DevBuf<unsigned long long> rerun_;
  DevBuf<uint8_t> ready_;
  DevBuf<unsigned long long> probe_stat_;  // aborted probes to re-run; per plan {probes, aborts}
  std::vector<double> dProbe_, dAbort_;
  DevBuf<unsigned long long> lstat_;          // this batch: per (plan, level) x {entries, chunks, survivors, length}
  DevBuf<float> cost_, cost_sorted_;
  DevBuf<unsigned long long> tasks_sorted_;
  DevBuf<uint8_t> sort_tmp_;
  DevBuf<PlanLevel> levels_;
  DevBuf<LeafPlan> leaf_;  // per plan, the trailing leaves counted by inclusion-exclusion
  DevBuf<PlanCheck> checks_;
  DevBuf<ClosureReq> creq_;
  DevBuf<uint16_t> creq_off_;
  DevBuf<uint8_t> gate_;
  DevBuf<uint32_t> qrows_, qdo_, qdi_;
  sig::Layout lay_;
  DevBuf<unsigned long long> tasks_;
  DevBuf<unsigned long long> counts_;
  DevBuf<uint8_t> tmo_;
  DevBuf<unsigned long long> ctr_;
  std::vector<uint64_t> h_counts_;
  std::vector<uint8_t> h_tmo_;
  MatchStats host_;
  int grid_ = 0;
};

}  // namespace csm
