#include "csm/loader.hpp"

#include <omp.h>

#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <cinttypes>

#include "csm/text_scanner.hpp"

namespace csm {
namespace {

constexpr int kMaxWarnings = 5;

void warn_malformed(const char* what, const std::string& path, size_t line, uint64_t& counter) {
  if (++counter <= kMaxWarnings) std::fprintf(stderr, "[csm] warning: %s %s:%zu malformed line skipped\n", what,
                                              path.c_str(), line);
}

// Dense id for a raw vertex token; new vertices get kInvalidLabel until resolved.
// minus_one: the token came from the CaLiG deletion syntax "e -(u+1) -(v+1)", so the id is token - 1.
bool vertex_ref(std::string_view tok, IdMap& ids, std::vector<label_t>& vlabel, vid_t& out, bool minus_one = false) {
  raw_id_t raw;
  if (!parse_raw_id(tok, raw)) return false;
  if (minus_one) {
    if (raw.hi == 0 && raw.lo == 0) return false;  // "-0" is not a valid encoded id
    if (raw.lo == 0) --raw.hi;
    --raw.lo;
  }
  bool created = false;
  out = ids.get_or_insert(raw, &created);
  if (created) vlabel.push_back(kInvalidLabel);
  return true;
}

bool optional_edge_label(LineScanner& sc, LabelMap& elabels, label_t& el) {
  std::string_view tok;
  uint64_t raw;
  el = kNoEdgeLabel;
  if (!sc.next_token(tok)) return false;
  if (!parse_u64(tok, raw)) return false;
  el = elabels.get_or_insert(raw);
  return true;
}

// 'v <id> <label>' for graph and stream files.
bool vertex_line(LineScanner& sc, IdMap& ids, LabelMap& vlabels, std::vector<label_t>& vlabel,
                 uint64_t& relabel_ignored) {
  std::string_view a, b;
  uint64_t raw_label;
  vid_t id;
  if (!sc.next_token(a) || !sc.next_token(b) || !parse_u64(b, raw_label)) return false;
  if (!vertex_ref(a, ids, vlabel, id)) return false;
  const label_t l = vlabels.get_or_insert(raw_label);
  if (vlabel[id] == kInvalidLabel) vlabel[id] = l;
  else if (vlabel[id] != l) ++relabel_ignored;
  return true;
}

void graph_line(LineScanner& sc, size_t lno, const std::string& path, IdMap& ids, LabelMap& vlabels, LabelMap& elabels,
                RawGraph& g) {
  std::string_view t, a, b;
  sc.next_token(t);
  if (t == "e") {
    ++g.stats.edge_lines;
    vid_t u, v;
    if (!sc.next_token(a) || !sc.next_token(b) || !vertex_ref(a, ids, g.vlabel, u) ||
        !vertex_ref(b, ids, g.vlabel, v)) {
      warn_malformed("graph", path, lno, g.stats.malformed_lines);
      return;
    }
    label_t el;
    if (optional_edge_label(sc, elabels, el)) g.any_edge_label = true;
    g.edges.push_back({u, v, el});
  } else if (t == "v") {
    ++g.stats.vertex_lines;
    if (!vertex_line(sc, ids, vlabels, g.vlabel, g.stats.relabel_ignored))
      warn_malformed("graph", path, lno, g.stats.malformed_lines);
  } else if (t == "t") {
    std::string_view n_tok, m_tok;
    uint64_t n = 0, m = 0;
    if (sc.next_token(n_tok) && parse_u64(n_tok, n)) {
      ids.reserve(n);
      g.vlabel.reserve(n);
    }
    if (sc.next_token(m_tok) && parse_u64(m_tok, m)) g.edges.reserve(m);
  } else {
    warn_malformed("graph", path, lno, g.stats.malformed_lines);
  }
}

struct GraphRec {
  uint64_t a, b, el;
  uint32_t lno;
  uint8_t kind, has_el;
};

RawGraph read_graph_file(const std::string& path, IdMap& ids, LabelMap& vlabels, LabelMap& elabels) {
  RawGraph g;
  MappedFile f(path);
  g.edges.reserve(f.size() / 20);
  const int T = std::max(1, omp_get_max_threads());
  std::vector<const char*> cut(T + 1);
  cut[0] = f.begin();
  cut[T] = f.end();
  for (int c = 1; c < T; ++c) {
    const char* p = f.begin() + f.size() * static_cast<size_t>(c) / static_cast<size_t>(T);
    if (p < cut[c - 1]) p = cut[c - 1];
    const char* nl = p < f.end() ? static_cast<const char*>(std::memchr(p, '\n', static_cast<size_t>(f.end() - p))) : nullptr;
    cut[c] = nl ? nl + 1 : f.end();
  }
  std::vector<std::vector<GraphRec>> recs(T);
  std::vector<size_t> lines(T + 1, 0);
#pragma omp parallel for num_threads(T) schedule(static, 1)
  for (int c = 0; c < T; ++c) {
    LineScanner sc(cut[c], cut[c + 1]);
    std::vector<GraphRec>& R = recs[c];
    R.reserve(static_cast<size_t>(cut[c + 1] - cut[c]) / 16 + 1);
    while (sc.next_line()) {
      GraphRec r{};
      r.lno = static_cast<uint32_t>(sc.line_no());
      const char* lb = sc.line_begin();
      std::string_view t, a, b, e;
      raw_id_t ra, rb;
      uint64_t x = 0;
      r.kind = 2;
      if (sc.next_token(t) && sc.next_token(a) && sc.next_token(b)) {
        if (t == "e" && parse_raw_id(a, ra) && parse_raw_id(b, rb) && ra.hi == 0 && rb.hi == 0) {
          r.a = ra.lo;
          r.b = rb.lo;
          if (!sc.next_token(e)) r.kind = 0;
          else if (parse_u64(e, x)) {
            r.kind = 0;
            r.has_el = 1;
            r.el = x;
          }
        } else if (t == "v" && parse_u64(b, x) && parse_raw_id(a, ra) && ra.hi == 0) {
          r.kind = 1;
          r.a = ra.lo;
          r.b = x;
        }
      }
      if (r.kind == 2) r.a = reinterpret_cast<uint64_t>(lb);
      R.push_back(r);
    }
    lines[c + 1] = sc.line_no();
  }
  for (int c = 0; c < T; ++c) lines[c + 1] += lines[c];
  for (int c = 0; c < T; ++c) {
    for (const GraphRec& r : recs[c]) {
      if (r.kind == 0) {
        ++g.stats.edge_lines;
        bool cr = false;
        const vid_t u = ids.get_or_insert(raw_id_t{0, r.a}, &cr);
        if (cr) g.vlabel.push_back(kInvalidLabel);
        const vid_t v = ids.get_or_insert(raw_id_t{0, r.b}, &cr);
        if (cr) g.vlabel.push_back(kInvalidLabel);
        label_t el = kNoEdgeLabel;
        if (r.has_el) {
          el = elabels.get_or_insert(r.el);
          g.any_edge_label = true;
        }
        g.edges.push_back({u, v, el});
      } else if (r.kind == 1) {
        ++g.stats.vertex_lines;
        bool cr = false;
        const vid_t id = ids.get_or_insert(raw_id_t{0, r.a}, &cr);
        if (cr) g.vlabel.push_back(kInvalidLabel);
        const label_t l = vlabels.get_or_insert(r.b);
        if (g.vlabel[id] == kInvalidLabel) g.vlabel[id] = l;
        else if (g.vlabel[id] != l) ++g.stats.relabel_ignored;
      } else {
        const char* lb = reinterpret_cast<const char*>(r.a);
        const char* le = static_cast<const char*>(std::memchr(lb, '\n', static_cast<size_t>(cut[c + 1] - lb)));
        LineScanner sc(lb, le ? le : cut[c + 1]);
        sc.next_line();
        graph_line(sc, lines[c] + r.lno, path, ids, vlabels, elabels, g);
      }
    }
    std::vector<GraphRec>().swap(recs[c]);
  }
  return g;
}

UpdateStream read_stream_file(const std::string& path, IdMap& ids, LabelMap& vlabels, LabelMap& elabels,
                              std::vector<label_t>& vlabel, bool& any_edge_label) {
  UpdateStream s;
  const size_t n_before = ids.size();
  MappedFile f(path);
  s.updates.reserve(f.size() / 24);
  LineScanner sc(f.begin(), f.end());
  uint64_t relabel_ignored = 0;
  while (sc.next_line()) {
    std::string_view t, a, b;
    sc.next_token(t);
    if (t == "e" || t == "+e" || t == "-e") {
      if (!sc.next_token(a) || !sc.next_token(b)) {
        warn_malformed("stream", path, sc.line_no(), s.stats.malformed_lines);
        continue;
      }
      UpdateOp op = (t == "-e") ? UpdateOp::DeleteEdge : UpdateOp::InsertEdge;
      const bool ma = strip_minus(a), mb = strip_minus(b);
      if (ma != mb || (ma && op == UpdateOp::DeleteEdge)) {
        warn_malformed("stream", path, sc.line_no(), s.stats.malformed_lines);
        continue;
      }
      if (ma) op = UpdateOp::DeleteEdge;  // CaLiG syntax "e -(u+1) -(v+1)" deletes edge (u, v)
      vid_t u, v;
      if (!vertex_ref(a, ids, vlabel, u, ma) || !vertex_ref(b, ids, vlabel, v, ma)) {
        warn_malformed("stream", path, sc.line_no(), s.stats.malformed_lines);
        continue;
      }
      label_t el;
      if (optional_edge_label(sc, elabels, el)) any_edge_label = true;
      if (u == v) {
        ++s.stats.self_loops_dropped;
        continue;
      }
      s.updates.push_back({u, v, el, op});
      if (op == UpdateOp::InsertEdge) ++s.stats.insertions;
      else ++s.stats.deletions;
    } else if (t == "v" || t == "+v") {
      ++s.stats.vertex_lines;
      if (!vertex_line(sc, ids, vlabels, vlabel, relabel_ignored))
        warn_malformed("stream", path, sc.line_no(), s.stats.malformed_lines);
    } else if (t == "-v") {
      ++s.stats.vertex_deletions_ignored;
    } else {
      warn_malformed("stream", path, sc.line_no(), s.stats.malformed_lines);
    }
  }
  s.stats.new_vertices = ids.size() - n_before;
  return s;
}

}  // namespace

namespace {
// Relevance table over (label(x), label(y)) of the query edge
struct RelTable {
  size_t nl;
  bool dense;
  std::vector<uint8_t> table;
  std::unordered_map<uint64_t, uint8_t> sparse;  // used instead of the table when there are many labels
  RelTable(const QueryGraph& q, size_t nlabels, bool directed) : nl(nlabels), dense(nlabels <= 4096) {
    if (dense) table.assign(nl * nl, 0);
    for (const QueryEdge& e : q.edges) {
      put(q.label[e.src], q.label[e.dst]);
      if (!directed) put(q.label[e.dst], q.label[e.src]);
    }
  }
  void put(label_t a, label_t b) {
    const uint64_t key = static_cast<uint64_t>(a) * nl + b;
    if (dense) table[key] = 1;
    else sparse[key] = 1;
  }
  bool operator()(label_t a, label_t b) const {
    const uint64_t key = static_cast<uint64_t>(a) * nl + b;
    return dense ? table[key] != 0 : sparse.count(key) != 0;
  }
};

// The per-update test is independent for every update.
void filter_stream_with(Dataset& d, const RelTable& rel, const label_t* vlabel) {
  WallTimer tf;
  const std::vector<Update>& U = d.stream.updates;
  const size_t n = U.size();
  const size_t par_min = size_t{1} << 16;
  const int T = n >= par_min ? std::max(1, omp_get_max_threads()) : 1;
  const size_t chunk = (n + T - 1) / T;
  std::vector<uint8_t> keep(n);
  std::vector<size_t> off(static_cast<size_t>(T) + 1, 0);
#pragma omp parallel for num_threads(T) schedule(static, 1)
  for (int t = 0; t < T; ++t) {
    const size_t lo = std::min(n, static_cast<size_t>(t) * chunk), hi = std::min(n, lo + chunk);
    size_t c = 0;
    for (size_t i = lo; i < hi; ++i) c += keep[i] = rel(vlabel[U[i].u], vlabel[U[i].v]) ? 1 : 0;
    off[t + 1] = c;
  }
  for (int t = 0; t < T; ++t) off[t + 1] += off[t];
  const size_t total = off[T];
  if (total != n) {  // otherwise nothing to map: keep the original stream
    std::vector<Update> kept(total);
#pragma omp parallel for num_threads(T) schedule(static, 1)
    for (int t = 0; t < T; ++t) {
      const size_t lo = std::min(n, static_cast<size_t>(t) * chunk), hi = std::min(n, lo + chunk);
      size_t j = off[t];
      for (size_t i = lo; i < hi; ++i)
        if (keep[i]) kept[j++] = U[i];
    }
    d.stream.updates.swap(kept);
    d.stream_raw.swap(kept);
  }
  d.stream_filter_pending = false;
  d.stream_filter_ms = tf.ms();
}
}  // namespace

void relevance_filter_stream(Dataset& d, const label_t* vlabel, bool directed) {
  if (!d.stream_filter_pending) return;
  WallTimer tf;
  const RelTable rel(d.query, d.vlabels.size(), directed);
  filter_stream_with(d, rel, vlabel);
  d.stream_filter_ms = tf.ms();  // table + pass
}

Dataset load_dataset(const LoadOptions& opt) {
  Dataset d;
  WallTimer t;

  RawGraph raw = read_graph_file(opt.graph_path, d.ids, d.vlabels, d.elabels);
  d.times.graph_read_ms = t.ms();
  const uint64_t graph_lines = raw.stats.vertex_lines + raw.stats.edge_lines;
  if (raw.stats.malformed_lines > 100 && raw.stats.malformed_lines * 10 > graph_lines)
    fatal("%s: %" PRIu64 " of %" PRIu64 " lines are malformed - the file is not in 'v id label' / 'e u v' format "
          "(corrupted or wrongly converted dataset)",
          opt.graph_path.c_str(), raw.stats.malformed_lines, graph_lines);
  const vid_t n_initial = static_cast<vid_t>(d.ids.size());

  t.reset();
  if (!opt.stream_path.empty())
    d.stream = read_stream_file(opt.stream_path, d.ids, d.vlabels, d.elabels, raw.vlabel, raw.any_edge_label);
  d.times.stream_read_ms = t.ms();
  const StreamStats& ss = d.stream.stats;
  const uint64_t stream_lines = ss.insertions + ss.deletions + ss.vertex_lines + ss.malformed_lines;
  if (ss.malformed_lines > 100 && ss.malformed_lines * 10 > stream_lines)
    fatal("%s: %" PRIu64 " of %" PRIu64 " lines are malformed - the file is not a valid update stream",
          opt.stream_path.c_str(), ss.malformed_lines, stream_lines);

  // Labels actually declared by the data ('v' lines of graph and stream), before the query adds its own.
  std::vector<uint8_t> declared(d.vlabels.size(), 0);
  for (label_t l : raw.vlabel)
    if (l != kInvalidLabel) declared[l] = 1;
  const size_t data_labels = static_cast<size_t>(std::count(declared.begin(), declared.end(), 1));

  t.reset();
  if (!opt.query_path.empty())  // query vertices without 'v' line keep kInvalidLabel until resolved below
    d.query = read_query_file(opt.query_path, opt.directed, d.vlabels, d.elabels, kInvalidLabel);
  d.times.query_read_ms = t.ms();

  // Default label for vertices without a 'v' line
  std::vector<label_t> query_labels;
  for (int u = 0; u < d.query.k; ++u)
    if (d.query.label[u] != kInvalidLabel &&
        std::find(query_labels.begin(), query_labels.end(), d.query.label[u]) == query_labels.end())
      query_labels.push_back(d.query.label[u]);
  const bool needed = std::find(raw.vlabel.begin(), raw.vlabel.end(), kInvalidLabel) != raw.vlabel.end();
  if (opt.has_default_vlabel) {
    d.default_vlabel = d.vlabels.get_or_insert(opt.default_vlabel);
  } else if (data_labels == 1) {
    d.default_vlabel = static_cast<label_t>(std::find(declared.begin(), declared.end(), 1) - declared.begin());
  } else if (data_labels == 0 && query_labels.size() == 1) {
    d.default_vlabel = query_labels[0];
  } else {
    d.default_vlabel = d.vlabels.get_or_insert(0);
    if (needed)
      std::fprintf(stderr, "[csm] warning: cannot infer a default vertex label; unlabeled vertices get raw label 0 "
                           "(override with --default-label)\n");
  }

  for (label_t& l : raw.vlabel) {
    if (l == kInvalidLabel) {
      l = d.default_vlabel;
      ++raw.stats.implicit_vertices;
    }
  }
  for (int u = 0; u < d.query.k; ++u)
    if (d.query.label[u] == kInvalidLabel) d.query.label[u] = d.default_vlabel;
  d.graph_read = raw.stats;

  // Relevance pre-filter
  d.stream_orig_size = d.stream.updates.size();
  if (d.query.k >= 2 && !d.query.edges.empty()) {
    WallTimer tf;
    const RelTable rel(d.query, d.vlabels.size(), opt.directed);
    const size_t e0 = raw.edges.size();
    raw.edges.erase(std::remove_if(raw.edges.begin(), raw.edges.end(),
                                   [&](const EdgeRecord& e) { return !rel(raw.vlabel[e.u], raw.vlabel[e.v]); }),
                    raw.edges.end());
    d.rel_edges_dropped = e0 - raw.edges.size();
    d.relevance_filtered = true;
    d.rel_filter_ms = tf.ms();
    d.stream_filter_pending = true;  // the driver filters the stream inside its timer
  }

  const vid_t n_total = static_cast<vid_t>(d.ids.size());
  d.graph = build_host_graph(std::move(raw), opt.directed, n_total, n_initial);
  d.times.csr_build_ms = d.graph.build.build_ms;

  // A query label that no data vertex carries means zero matches - almost always a dataset mistake.
  std::vector<uint8_t> present(d.vlabels.size(), 0);
  for (label_t l : d.graph.vlabel) present[l] = 1;
  for (int u = 0; u < d.query.k; ++u)
    if (!present[d.query.label[u]])
      std::fprintf(stderr, "[csm] WARNING: query vertex %d has label %" PRIu64
                           " which no data vertex carries -> the query can never match\n",
                   u, d.vlabels.raw_of(d.query.label[u]));
  return d;
}

void print_dataset_report(const Dataset& d, FILE* out) {
  const HostGraph& g = d.graph;
  const GraphReadStats& r = d.graph_read;
  std::fprintf(out, "== Data graph (%s) ==\n", g.directed ? "directed" : "undirected");
  if (d.relevance_filtered)
    std::fprintf(out, "  S1 relevance filter : dropped %" PRIu64 " initial edges and %" PRIu64 " of %" PRIu64
                      " updates whose label pair matches no query edge (%.1f ms)\n",
                 d.rel_edges_dropped, d.stream_orig_size - d.stream.updates.size(), d.stream_orig_size,
                 d.rel_filter_ms + d.stream_filter_ms);
  std::fprintf(out, "  vertices            : %u initial, %u incl. stream-new\n", g.n_initial, g.n);
  std::fprintf(out, "  edges (logical)     : %" PRIu64 "\n", g.m);
  std::fprintf(out, "  adjacency entries   : out %" PRIu64 ", in %" PRIu64 "\n", g.out.entries(),
               g.directed ? g.in.entries() : g.out.entries());
  std::fprintf(out, "  avg / max degree    : %.2f / out %" PRIu64 " in %" PRIu64 "\n",
               g.n ? static_cast<double>(g.out.entries()) / g.n : 0.0, g.build.max_out_degree,
               g.build.max_in_degree);
  std::fprintf(out, "  vertex labels       : %zu distinct   edge labels: %zu distinct%s\n", d.vlabels.size(),
               d.elabels.num_real_labels(), g.has_edge_labels ? "" : " (graph unlabeled on edges)");
  std::fprintf(out, "  dropped             : %" PRIu64 " self loops, %" PRIu64 " duplicate edges, %" PRIu64
                    " edge-label conflicts\n",
               g.build.self_loops_dropped, g.build.duplicate_edges_dropped, g.build.edge_label_conflicts);
  std::fprintf(out, "  file lines          : %" PRIu64 " v, %" PRIu64 " e, %" PRIu64 " malformed; %" PRIu64
                    " vertices without 'v' line (got default label), %" PRIu64 " relabels ignored\n",
               r.vertex_lines, r.edge_lines, r.malformed_lines, r.implicit_vertices, r.relabel_ignored);

  // Label histogram (top 8) - tells immediately whether label-based pruning can work.
  std::vector<uint64_t> hist(d.vlabels.size(), 0);
  for (vid_t v = 0; v < g.n; ++v) ++hist[g.vlabel[v]];
  std::vector<label_t> order(hist.size());
  for (label_t l = 0; l < order.size(); ++l) order[l] = l;
  std::sort(order.begin(), order.end(), [&](label_t a, label_t b) { return hist[a] > hist[b]; });
  std::fprintf(out, "  label histogram     :");
  for (size_t i = 0; i < order.size() && i < 8; ++i)
    std::fprintf(out, " [%" PRIu64 "]=%" PRIu64, d.vlabels.raw_of(order[i]), hist[order[i]]);
  std::fprintf(out, "%s\n", order.size() > 8 ? " ..." : "");

  const StreamStats& s = d.stream.stats;
  std::fprintf(out, "== Update stream ==\n");
  std::fprintf(out, "  updates             : %zu (%" PRIu64 " insertions, %" PRIu64 " deletions)\n",
               d.stream.size(), s.insertions, s.deletions);
  std::fprintf(out, "  new vertices        : %" PRIu64 "  vertex lines: %" PRIu64 "  -v ignored: %" PRIu64
                    "  self loops dropped: %" PRIu64 "  malformed: %" PRIu64 "\n",
               s.new_vertices, s.vertex_lines, s.vertex_deletions_ignored, s.self_loops_dropped,
               s.malformed_lines);

  const QueryGraph& q = d.query;
  std::fprintf(out, "== Query ==\n");
  std::fprintf(out, "  vertices / edges    : %d / %zu   max degree: %d   edge labels: %s\n", q.k, q.edges.size(),
               q.max_degree(), q.has_edge_labels ? "yes" : "no");
  for (int u = 0; u < q.k; ++u) {
    std::fprintf(out, "  q%-2d label=%-6" PRIu64 " out=", u, d.vlabels.raw_of(q.label[u]));
    for (uint32_t m = q.out_adj[u]; m; m &= m - 1) std::fprintf(out, "%d,", __builtin_ctz(m));
    if (q.directed) {
      std::fprintf(out, " in=");
      for (uint32_t m = q.in_adj[u]; m; m &= m - 1) std::fprintf(out, "%d,", __builtin_ctz(m));
    }
    std::fprintf(out, "\n");
  }
  std::fprintf(out, "== Load times (ms) ==\n  graph %.1f  stream %.1f  query %.1f  csr-build %.1f\n",
               d.times.graph_read_ms, d.times.stream_read_ms, d.times.query_read_ms, d.times.csr_build_ms);
}

}  // namespace csm
