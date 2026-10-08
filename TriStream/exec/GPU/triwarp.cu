// TriStream: TriWarp, the GPU-parallel execution of TriMatch
#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <thread>
#include <string>
#include <vector>
#include "csm/cuda_check.cuh"
#include "csm/device_graph.cuh"
#include "csm/dynamic_graph.cuh"
#include "csm/edge_stats.hpp"
#include "csm/loader.hpp"
#include "csm/match_plan.hpp"
#include "csm/matcher.cuh"
#include "csm/sig_store.cuh"

namespace {

struct Args {
  csm::LoadOptions load;
  std::string name;  // taken from the graph file's folder
  int device = 0;
  bool gpu_info_only = false;
  size_t batch = 65536;
  bool verbose = false;
  double compact_ratio = 0.25;
  uint32_t hub_tau = 256;         // hub threshold
  csm::S1Config s1cfg;
  csm::S3Config s3cfg;
  int s1_orders = 3;
  double time_limit = 0.0;        // per-query limit on stream processing
};

void usage(const char* prog) {
  std::fprintf(stderr,
               "usage: %s --graph FILE --stream FILE --query FILE [--directed] [--batch N] [--time-limit S]\n"
               "       [--name NAME] [--verbose] [--device N] [--gpu-info]\n",
               prog);
  std::exit(EXIT_FAILURE);
}

Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    auto need = [&](const char* flag) -> const char* {
      if (i + 1 >= argc) csm::fatal("%s needs a value", flag);
      return argv[++i];
    };
    if (!std::strcmp(argv[i], "--graph")) a.load.graph_path = need("--graph");
    else if (!std::strcmp(argv[i], "--stream")) a.load.stream_path = need("--stream");
    else if (!std::strcmp(argv[i], "--query")) a.load.query_path = need("--query");
    else if (!std::strcmp(argv[i], "--name")) a.name = need("--name");
    else if (!std::strcmp(argv[i], "--directed")) a.load.directed = true;
    else if (!std::strcmp(argv[i], "--default-label")) {
      a.load.has_default_vlabel = true;
      a.load.default_vlabel = std::stoull(need("--default-label"));
    } else if (!std::strcmp(argv[i], "--batch")) a.batch = std::stoull(need("--batch"));
    else if (!std::strcmp(argv[i], "--verbose")) a.verbose = true;
    else if (!std::strcmp(argv[i], "--time-limit")) a.time_limit = std::stod(need("--time-limit"));
    else if (!std::strcmp(argv[i], "--device")) a.device = std::stoi(need("--device"));
    else if (!std::strcmp(argv[i], "--gpu-info")) a.gpu_info_only = true;
    else {
      std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
      usage(argv[0]);
    }
  }
  if (a.batch == 0 || a.batch > (1u << 30)) csm::fatal("--batch must be in [1, 2^30]");
  return a;
}

void print_gpu(int dev, FILE* out) {
  cudaDeviceProp p{};
  CUDA_CHECK(cudaGetDeviceProperties(&p, dev));
  int rt = 0, drv = 0;
  cudaRuntimeGetVersion(&rt);
  cudaDriverGetVersion(&drv);
  std::fprintf(out, "== GPU %d: %s ==\n", dev, p.name);
  std::fprintf(out, "  compute capability  : %d.%d   SMs: %d   warp: %d\n", p.major, p.minor, p.multiProcessorCount,
               p.warpSize);
  std::fprintf(out, "  global memory       : %.2f GB   L2: %.1f MB   shared/block: %zu KB (opt-in %zu KB)\n",
               p.totalGlobalMem / 1073741824.0, p.l2CacheSize / 1048576.0, p.sharedMemPerBlock / 1024,
               p.sharedMemPerBlockOptin / 1024);
  std::fprintf(out, "  max threads/SM      : %d   regs/SM: %d\n", p.maxThreadsPerMultiProcessor,
               p.regsPerMultiprocessor);
  std::fprintf(out, "  CUDA runtime/driver : %d / %d\n", rt, drv);
}

double gpu_used_gb() {
  size_t free_b = 0, total_b = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
  return (total_b - free_b) / 1073741824.0;
}

// short report
std::string base_name(const std::string& p) {
  const size_t s = p.find_last_of('/');
  return s == std::string::npos ? p : p.substr(s + 1);
}

std::string dir_name(const std::string& p) {
  const size_t s = p.find_last_of('/');
  return s == std::string::npos ? "." : p.substr(0, s);
}

// name of the graph
std::string dataset_name(const Args& a) {
  if (!a.name.empty()) return a.name;
  const std::string d = base_name(dir_name(a.load.graph_path));
  return (d.empty() || d == "." || d == "..") ? base_name(a.load.graph_path) : d;
}

// name of query
std::string short_path(const std::string& p, const std::string& graph_path) {
  const std::string d = dir_name(graph_path) + "/";
  return p.compare(0, d.size(), d) == 0 ? p.substr(d.size()) : base_name(p);
}

int query_diameter(const csm::QueryGraph& q) {  // longest shortest path, edge direction ignored
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

int query_label_count(const csm::QueryGraph& q) {
  int n = 0;
  for (int u = 0; u < q.k; ++u) {
    bool first = true;
    for (int w = 0; w < u; ++w) first &= q.label[w] != q.label[u];
    n += first;
  }
  return n;
}

void rule() { std::printf("==========================================\n"); }

}  // namespace

int main(int argc, char** argv) {
  const Args args = parse(argc, argv);
  csm::WallTimer wall;
  if (args.gpu_info_only) {
    CUDA_CHECK(cudaSetDevice(args.device));
    print_gpu(args.device, stdout);
    return 0;
  }
  if (args.load.graph_path.empty() || args.load.stream_path.empty() || args.load.query_path.empty()) usage(argv[0]);
  if (!args.verbose) {
    rule();
    std::printf("TriStream (TriWarp, GPU)\n");
    rule();
    std::fflush(stdout);
  }
  // Initiating gpu early to save time
  cudaError_t gpu_init_err = cudaSuccess;
  std::thread gpu_init([&] {
    gpu_init_err = cudaSetDevice(args.device);
    if (gpu_init_err == cudaSuccess) gpu_init_err = cudaFree(nullptr);
  });
  wall.reset();

  // loading graphs
  csm::Dataset data = csm::load_dataset(args.load);
  const double load_ms = wall.ms();
  gpu_init.join();
  CUDA_CHECK(gpu_init_err);
  CUDA_CHECK(cudaSetDevice(args.device));  // this thread's current device
  if (args.verbose) {
    print_gpu(args.device, stdout);
    csm::print_dataset_report(data, stdout);
  } else {
    const csm::QueryGraph& q = data.query;
    std::printf("Dataset        : %s\n", dataset_name(args).c_str());
    std::printf("Nodes          : %u\n", data.graph.n_initial);
    std::printf("Edges          : %" PRIu64 " (%s)\n", data.graph_read.edge_lines,
                data.graph.directed ? "directed" : "undirected");
    std::printf("Query          : %s\n", short_path(args.load.query_path, args.load.graph_path).c_str());
    std::printf("Query details  : %d vertices | %zu edges | max degree %d | diameter %d | %d labels\n", q.k,
                q.edges.size(), q.max_degree(), query_diameter(q), query_label_count(q));
    std::printf("Stream         : %s\n", short_path(args.load.stream_path, args.load.graph_path).c_str());
    std::printf("Stream updates : %" PRIu64 " (%" PRIu64 " insertions, %" PRIu64 " deletions)\n",
                data.stream_orig_size, data.stream.stats.insertions, data.stream.stats.deletions);
    std::fflush(stdout);
  }

  csm::WallTimer t_up;
  csm::DynamicGraph dyn(args.compact_ratio);
  dyn.upload(data.graph);
  CUDA_CHECK(cudaDeviceSynchronize());
  const double upload_ms = t_up.ms();
  // After all loading, starting the time
  csm::WallTimer t_in;
  csm::relevance_filter_stream(data, data.graph.vlabel.data(), data.graph.directed);
  csm::DeviceStream dstream;
  dstream.upload(data.stream);
  CUDA_CHECK(cudaDeviceSynchronize());
  const double stream_in_ms = t_in.ms();
  if (args.verbose) std::printf("== Upload ==\n  %.2f ms, GPU memory in use %.2f GB\n", upload_ms, gpu_used_gb());

  // S1 cost model: edge-type statistics and the matching plans (RI, rare-first, leaf-last)
  const csm::EdgeStats estats = csm::compute_edge_stats(data.graph);
  const csm::MatchPlan plan = csm::build_match_plan(data.query, &estats, args.s1_orders);
  if (args.verbose) csm::print_match_plan(plan, data.query, stdout);
  csm::Matcher matcher(plan, args.s1cfg, args.s3cfg);

  // S2: derivative signatures of the query and of the data vertices
  csm::SigTable qsig;
  // S2 query projection: mask of the query labels
  std::vector<uint8_t> lmask;
  {
    csm::label_t nl = 0;
    for (csm::vid_t v = 0; v < data.graph.n; ++v) nl = std::max(nl, data.graph.vlabel[v] + 1);
    for (int u = 0; u < data.query.k; ++u) nl = std::max(nl, data.query.label[u] + 1);
    lmask.assign(nl, 0);
    for (int u = 0; u < data.query.k; ++u) lmask[data.query.label[u]] = 1;
  }
  const bool use_el = data.query.has_edge_labels;
  const csm::KeyLayout keys = csm::build_key_layout(csm::query_adjacency(data.query), use_el,
                                                    static_cast<csm::label_t>(lmask.size()));
  csm::compute_signatures(csm::query_adjacency(data.query), use_el, keys.view(), qsig);
  csm::SigStore s2(data.graph.n, args.hub_tau, use_el);
  s2.set_projection(lmask);
  s2.set_layout(keys);
  s2.build(dyn.view(), 0);
  matcher.set_s2(qsig, s2.layout());
  if (args.verbose)
    std::printf("== S2 signatures ==\n  %d counters/vertex (slope %d, staircase %d, curvature %d pairs of %d x %d keys%s),"
                " hub tau %u, build %.2f ms\n",
                keys.n1 + keys.ns + keys.np, keys.n1, keys.ns, keys.np, keys.nm, keys.nf, keys.folded ? ", folded" : "",
                args.hub_tau, s2.stats().build_ms);
  uint64_t prev_del = 0;
  double s2_ms = 0.0;

  // stream the updates in micro-batches (warm-up: b/16, doubling up to b)
  const size_t N = data.stream.size();
  size_t batches = 0;
  for (size_t f = 0, bs = (N > args.batch ? std::max<size_t>(1, args.batch / 16) : args.batch); f < N; f += bs, bs = std::min(args.batch, 2 * bs)) ++batches;

  // Free the host copies: the GPU holds the graph and the stream from here on.
  {
    auto drop = [](auto& v) { std::decay_t<decltype(v)>().swap(v); };
    drop(data.graph.out.offsets); drop(data.graph.out.keys); drop(data.graph.out.elabels);
    drop(data.graph.in.offsets); drop(data.graph.in.keys); drop(data.graph.in.elabels);
    drop(data.graph.vlabel);
    data.ids = csm::IdMap(16);
    drop(data.stream.updates);
    drop(data.stream_raw);
  }

  if (args.verbose)
    std::printf("== Streaming: %zu updates, batch %zu, %zu batches ==\n", N, args.batch, batches);
  csm::GpuTimer gt;
  double apply_ms = 0.0, compact_ms = 0.0, peak_gb = gpu_used_gb();
  bool tle = false;
  size_t processed = N;
  csm::WallTimer t_limit;
  size_t sent_b = 0;
  for (size_t b = 0, first = 0, bs = (N > args.batch ? std::max<size_t>(1, args.batch / 16) : args.batch); b < batches;
       ++b, first += bs, bs = std::min(args.batch, 2 * bs)) {
    const size_t count = std::min(bs, N - first);
    if (args.time_limit > 0) {  // per-query time limit on stream processing
      CUDA_CHECK(cudaDeviceSynchronize());
      const double left_ms = args.time_limit * 1000.0 - t_limit.ms();
      if (left_ms <= 0) {
        tle = true;
        processed = first;
        break;
      }
      matcher.set_deadline(static_cast<uint64_t>(left_ms * 1e6));
    }
    ++sent_b;
    gt.start();
    dyn.apply_batch(dstream.view(), first, count);
    apply_ms += gt.stop_ms();
    // control: S2-maintenance mode (deferred or exact)
    const bool deferred = !matcher.gate_active_next();
    // S2 dormancy: 
    const bool dormant = !matcher.s2_needed_next();
    double s2_batch_ms = 0.0;
    if (dormant) {
      gt.start();
      s2.note_dormant(dstream.view(), first, count, dyn.batch_effects(), dyn.view().vlabel);
      const double ms = gt.stop_ms();
      s2_ms += ms;
    }
    uint64_t reentry_eps = 0;  // endpoints recorded while dormant, re-entered by this batch's refresh
    if (!dormant) {  // phase A: union window (t0, t_end] -> sound for every snapshot of the batch
      reentry_eps = s2.dirty_count();
      gt.start();
      s2.refresh(dyn.view(), dstream.view(), first, count, dyn.batch_effects(), static_cast<csm::ts_t>(first),
                 static_cast<csm::ts_t>(first + count), deferred);
      if (!deferred) s2.heal(dyn.view(), static_cast<csm::ts_t>(first), static_cast<csm::ts_t>(first + count));
      s2_batch_ms = gt.stop_ms();
      s2_ms += s2_batch_ms;
    }
    {  // batch snapshot
      const csm::SigView sv = s2.view();
      // dormant S2: triage runs as a count-only probe
      matcher.set_s2_probe(dormant);
      matcher.match_batch(dyn.view(), dstream.view(), first, count, dyn.batch_effects(), &sv);
      matcher.set_s2_probe(false);
    }
    if (!dormant) {  // phase B: the batch deleted an edge -> refresh the rows on G_end
      const uint64_t del_now = dyn.stats().eff_deletions;
      if (del_now != prev_del) {
        gt.start();
        const csm::ts_t te = static_cast<csm::ts_t>(first + count);
        s2.refresh(dyn.view(), dstream.view(), first, count, dyn.batch_effects(), te, te, deferred, true);
        const double ms = gt.stop_ms();
        s2_ms += ms;
        s2_batch_ms += ms;
      }
    }
    matcher.end_batch_s2(s2_batch_ms, reentry_eps, s2.dirty_count());
    prev_del = dyn.stats().eff_deletions;
    peak_gb = std::max(peak_gb, gpu_used_gb());

    csm::WallTimer t_c;
    const bool compacted = dyn.maybe_compact();
    if (compacted) {
      compact_ms += t_c.ms();
      peak_gb = std::max(peak_gb, gpu_used_gb());
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  // end of the timed stream phase
  const double loop_wall_ms = t_limit.ms();
  const double stream_wall_ms = stream_in_ms + loop_wall_ms;
  const double total_ms = wall.ms();

  const csm::DynamicStats st = dyn.stats();
  const csm::SnapshotPrint final_print = dyn.snapshot_print(static_cast<csm::ts_t>(N));
  const double stream_ms = apply_ms + compact_ms;
  const double ups = stream_ms > 0 ? N / (stream_ms / 1000.0) : 0.0;
  const csm::MatchStats ms = matcher.stats();
  tle = tle || ms.dead_tasks > 0;
  const double proc_ms = stream_ms + ms.kernel_ms;  // apply + compaction + matching (+ result copy)
  const double proc_ups = proc_ms > 0 ? N / (proc_ms / 1000.0) : 0.0;

  const csm::SigStats ss = s2.stats();

  if (!args.verbose) {  // the short report
    const uint64_t n_upd = data.stream_orig_size;
    const size_t sent = tle ? sent_b : batches;
    // stream processing: relevance test of every update + stream copy + apply + S2 + matching
    const double gpu_ms = stream_wall_ms;  // wall clock of the whole stream phase (see above)
    std::printf("Batches to GPU : %zu (batch size %zu; %zu of %" PRIu64 " updates can touch the query)\n", sent,
                args.batch, N, n_upd);
    std::printf("Matches found  : %" PRIu64 " added, %" PRIu64 " deleted\n", ms.matches_pos, ms.matches_neg);
    if (tle)
      std::printf("Time limit     : %.1f s reached, %zu of %zu relevant updates processed (counts are lower bounds)\n",
                  args.time_limit, processed, N);
    std::printf("Time (GPU)     : %.2f ms\n", gpu_ms);
    if (tle || gpu_ms <= 0)
      std::printf("Speed          : n/a\n");
    else
      std::printf("Speed          : %.4e edges/sec\n", n_upd / (gpu_ms / 1000.0));
    // whole run
    std::printf("End-to-end time: %.2f ms\n", total_ms);
    rule();
  }

  if (args.verbose) {  // the full report
    std::printf("== Stream result ==\n");
    std::printf("  effective: %" PRIu64 " insertions, %" PRIu64 " deletions | no-op: %" PRIu64
                " duplicate insertions, %" PRIu64 " deletions of absent edges\n",
                st.eff_insertions, st.eff_deletions, st.noop_insertions, st.noop_deletions);
    std::printf("  edges: %" PRIu64 " -> %" PRIu64 "   compactions: %" PRIu64 "   max delta entries: %" PRIu64 "\n",
                data.graph.m, final_print.edges, st.compactions, st.max_delta_entries);
    std::printf("  time: load %.1f ms | upload %.1f ms | stream in (relevance test + copy) %.1f ms | stream apply %.1f ms"
                " + compaction %.1f ms | end-to-end %.1f ms\n",
                load_ms, upload_ms, stream_in_ms, apply_ms, compact_ms, total_ms);
    std::printf("  speed: %.3f M edge-updates/s (GPU stream processing)   peak GPU memory %.2f GB\n", ups / 1e6, peak_gb);
    std::printf("== Match result (exact; S1 / S2 / S3 as configured below) ==\n");
    std::printf("  ΔM+ %" PRIu64 "   ΔM- %" PRIu64 "   updates with matches %" PRIu64 "   timed-out updates %" PRIu64 "\n",
                ms.matches_pos, ms.matches_neg, ms.updates_with_matches, ms.timed_out_updates);
    std::printf("  tasks %" PRIu64 " (effective update x anchor), anchored %" PRIu64 ", candidate chunks %" PRIu64
                ", aborted tasks %" PRIu64 "\n", ms.tasks, ms.anchored, ms.chunks, ms.timeouts);
    std::printf("  time: match %.1f ms | apply+compact+match %.1f ms | %.3f M updates/s end-to-end on GPU\n",
                ms.kernel_ms, proc_ms, proc_ups / 1e6);
    std::printf("== S2: derivative signatures ==\n");
    std::printf("  maintenance: build %.1f ms, refresh %.1f ms over %" PRIu64 " phases; rows refreshed %" PRIu64
                " (endpoints %" PRIu64 "), hubs %" PRIu64 " (tau %u)\n",
                ss.build_ms, s2_ms, ss.phases, ss.affected_rows, ss.endpoint_rows, ss.hubs, args.hub_tau);
    const double cut = ms.label_pass ? 100.0 * (ms.label_pass - ms.s2_pass) / ms.label_pass : 0.0;
    const uint64_t searched = ms.anchored - ms.closure_fail;
    std::printf("  tasks %" PRIu64 " -> labels %" PRIu64 " -> S2 triage %" PRIu64 " (removes %.2f%% of label survivors)"
                " -> anchor edges ok %" PRIu64 " -> closure ok %" PRIu64 " (closure removes %" PRIu64 ", %" PRIu64
                " chunks) -> searched\n", ms.tasks, ms.label_pass, ms.s2_pass, cut, ms.anchored, searched,
                ms.closure_fail, ms.closure_chunks);
    std::printf("  DFS: %" PRIu64 " candidate chunks, %" PRIu64 " bindings (extended partial matches), %" PRIu64
                " candidates rejected by the signature gate\n", ms.chunks, ms.bindings, ms.gate_rejects);
    std::printf("  maintenance policy auto: %" PRIu64 " of %" PRIu64 " phases deferred, %" PRIu64 " group deferrals, %"
                PRIu64 " heals (%" PRIu64 " rows, %.1f ms), %" PRIu64 " stale rows at the end\n",
                ss.deferred_phases, ss.phases, ss.deferred_rows, ss.heals, ss.healed_rows, ss.heal_ms, ss.stale_end);
    std::printf("  dormancy: %" PRIu64 " batches without maintenance (the controller put S2 to sleep %" PRIu64 " times), %" PRIu64
                " exact re-entries over %" PRIu64 " recorded endpoints; count-only probes woke S2 %" PRIu64 " times\n",
                ss.dormant_batches, ms.s2_sleeps, ss.reentries, ss.reentry_endpoints, ms.s2_wakes);
    std::printf("== S1: cost model (plans %d for %d anchors, decay %.2f, explore 1/%d) ==\n", plan.num_plans,
                plan.num_anchors, args.s1cfg.decay, args.s1cfg.explore);
    std::printf("  plans chosen: RI %" PRIu64 " | rare-first %" PRIu64 " | leaf-last %" PRIu64 " (switched by the model %"
                PRIu64 ", exploration probes %" PRIu64 ", bets aborted and re-run with RI %" PRIu64 ")\n",
                ms.kind_chosen[0], ms.kind_chosen[1], ms.kind_chosen[2], ms.switched, ms.explored, ms.probe_aborts);
    std::printf("  local level-3 estimate used for %" PRIu64 " tasks (S2 curvature / endpoint lists)\n", ms.local_used);
    std::printf("  predicted size: tiny %" PRIu64 " | normal %" PRIu64 " | heavy %" PRIu64
                " | S2 gate/closure switched off in %" PRIu64 " batches | S1 time %.1f ms\n",
                ms.size_class[0], ms.size_class[1], ms.size_class[2], ms.s2_off_batches, ms.s1_ms);
    std::printf("  calibrated error (actual+1)/(pred+1) over %" PRIu64 " sampled tasks: p50 %.2f p90 %.2f p99 %.2f |"
                " regret: %" PRIu64 " chunks spent by aborted bets (%" PRIu64 " had counted matches, discarded)\n",
                ms.ratio_n, ms.ratio_p50, ms.ratio_p90, ms.ratio_p99, ms.bet_waste, ms.abort_counted);
    std::printf("  flat batches (no sort, no split: nothing to decide) %" PRIu64 "\n", ms.flat_batches);
    std::printf("== S3: cost-bounded splitting ==\n");
    std::printf("  split %" PRIu64 " tasks into %" PRIu64 " pieces, avg piece budget %.0f chunks, S3 time %.1f ms |"
                " longest task/piece %.2f ms, sum of task times %.1f ms\n", ms.s3_split, ms.s3_pieces,
                ms.s3_budget_avg, ms.s3_ms, ms.max_task_ms, ms.sum_task_ms);
    if (tle)
      std::printf("== TIME LIMIT %.1f s reached: %zu of %zu updates processed, %" PRIu64
                  " tasks stopped; affected counts are lower bounds / unknown (flagged) ==\n",
                  args.time_limit, processed, N, ms.dead_tasks);

    std::printf("RESULT dataset=%s directed=%d V=%u E0=%" PRIu64 " E_end=%" PRIu64 " updates=%zu ins=%" PRIu64
                " del=%" PRIu64 " noop=%" PRIu64 " batch=%zu batches=%zu load_ms=%.1f upload_ms=%.1f stream_in_ms=%.1f apply_ms=%.1f"
                " compact_ms=%.1f e2e_ms=%.1f Mupd_per_s=%.3f compactions=%" PRIu64 " peak_gpu_gb=%.2f"
                " qk=%d qm=%zu match_ms=%.1f dm_pos=%" PRIu64 " dm_neg=%" PRIu64 " tasks=%" PRIu64 " anchored=%" PRIu64
                " chunks=%" PRIu64 " tmo_upd=%" PRIu64 " proc_Mupd_per_s=%.3f"
                " s2_ms=%.1f s2_rows=%" PRIu64 " label_pass=%" PRIu64 " s2_pass=%" PRIu64
                " clo_fail=%" PRIu64 " clo_chunks=%" PRIu64 " gate_rej=%" PRIu64
                " plans=%d k_ri=%" PRIu64 " k_rare=%" PRIu64 " k_leaf=%" PRIu64 " sz_tiny=%" PRIu64 " sz_norm=%" PRIu64
                " sz_heavy=%" PRIu64 " s2_off_b=%" PRIu64 " s1_ms=%.1f s1_sw=%" PRIu64
                " s1_explore=%" PRIu64 " s1_abort=%" PRIu64 " s1_local=%" PRIu64 " bind=%" PRIu64
                " s2_def_phases=%" PRIu64 " s2_def_rows=%" PRIu64 " s2_heals=%" PRIu64 " s2_healed=%" PRIu64
                " s2_stale_end=%" PRIu64 " ratio_p50=%.2f ratio_p90=%.2f ratio_p99=%.2f bet_waste=%" PRIu64
                " abort_cnt=%" PRIu64 " s3_split=%" PRIu64 " s3_pieces=%" PRIu64 " s3_B=%.0f s3_ms=%.1f"
                " max_task_ms=%.3f sum_task_ms=%.1f s2_skip=%" PRIu64 " tle=%d processed=%zu dead=%" PRIu64
                " flat_b=%" PRIu64 " s1_dormant=%" PRIu64
                " s2_dormant=%" PRIu64 " s2_sleeps=%" PRIu64 " s2_wakes=%" PRIu64 " s2_probe_skips=%" PRIu64 " s2_reentries=%" PRIu64
                " rel_filter=%d rel_updates=%zu rel_edges_dropped=%" PRIu64 " rel_ms=%.1f gpu_ms=%.2f gpu_parts_ms=%.2f\n",
                dataset_name(args).c_str(), data.graph.directed ? 1 : 0, data.graph.n, data.graph.m, final_print.edges,
                static_cast<size_t>(data.stream_orig_size ? data.stream_orig_size : N),
                st.eff_insertions, st.eff_deletions, st.noop_insertions + st.noop_deletions, args.batch, batches,
                load_ms, upload_ms, stream_in_ms, apply_ms, compact_ms, total_ms, ups / 1e6, st.compactions, peak_gb,
                data.query.k, data.query.edges.size(), ms.kernel_ms,
                ms.matches_pos, ms.matches_neg, ms.tasks, ms.anchored, ms.chunks, ms.timed_out_updates, proc_ups / 1e6,
                s2_ms, ss.affected_rows,
                ms.label_pass, ms.s2_pass, ms.closure_fail, ms.closure_chunks, ms.gate_rejects,
                plan.num_plans, ms.kind_chosen[0], ms.kind_chosen[1],
                ms.kind_chosen[2], ms.size_class[0], ms.size_class[1], ms.size_class[2], ms.s2_off_batches, ms.s1_ms,
                ms.switched, ms.explored, ms.probe_aborts, ms.local_used, ms.bindings,
                ss.deferred_phases, ss.deferred_rows, ss.heals, ss.healed_rows, ss.stale_end,
                ms.ratio_p50, ms.ratio_p90, ms.ratio_p99, ms.bet_waste, ms.abort_counted,
                ms.s3_split, ms.s3_pieces, ms.s3_budget_avg, ms.s3_ms,
                ms.max_task_ms, ms.sum_task_ms, s2.skipped_updates(), tle ? 1 : 0, processed,
                ms.dead_tasks,
                ms.flat_batches, ms.s1_dormant_batches, ss.dormant_batches, ms.s2_sleeps, ms.s2_wakes, ms.s2_probe_skips, ss.reentries,
                data.relevance_filtered ? 1 : 0, N, data.rel_edges_dropped, data.rel_filter_ms,
                stream_wall_ms, stream_in_ms + apply_ms + compact_ms + s2_ms + ms.kernel_ms);
  }
  return EXIT_SUCCESS;
}
