#include "csm/match_plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <tuple>

#include "csm/edge_stats.hpp"

namespace csm {
namespace {

// Label of query edge src -> dst (undirected: the edge {src,dst}).
label_t query_edge_label(const QueryGraph& q, int src, int dst) {
  for (const QueryEdge& e : q.edges) {
    if (e.src == src && e.dst == dst) return e.el;
    if (!q.directed && e.src == dst && e.dst == src) return e.el;
  }
  fatal("internal: query edge %d-%d not found", src, dst);
}

// Expected size of the candidate list of c when reached from matched neighbour p (edge-type statistics).
double fan(const QueryGraph& q, const EdgeStats* st, int p, int c) {
  if (!st) return 1.0;
  const uint32_t side = (!q.directed || ((q.out_adj[p] >> c) & 1u)) ? 0u : 1u;  // p->c: out-list of p
  return st->fanout(q.label[p], side, q.label[c]);
}

// kind 0 RI, 1 rare-first, 2 leaf-last. Every prefix stays connected.
void make_order(const QueryGraph& q, const EdgeStats* st, int kind, int qa, int qb, int* order) {
  order[0] = qa;
  order[1] = qb;
  uint32_t done = (1u << qa) | (1u << qb);
  for (int i = 2; i < q.k; ++i) {
    int best = -1;
    double best_key[3] = {0, 0, 0};
    for (int c = 0; c < q.k; ++c) {
      if (done & (1u << c)) continue;
      const uint32_t back_mask = q.neighbors(c) & done;
      if (!back_mask) continue;
      const double back = __builtin_popcount(back_mask), deg = q.degree(c);
      double key[3];  // larger is better, compared lexicographically
      if (kind == 1) {
        double f = 1e300;
        for (int p = 0; p < q.k; ++p)
          if ((back_mask >> p) & 1u) f = std::min(f, fan(q, st, p, c));
        key[0] = -f;
        key[1] = back;
        key[2] = deg;
      } else {
        key[0] = kind == 2 ? (q.degree(c) >= 2 ? 1.0 : 0.0) : 0.0;  // leaf-last: any core vertex beats a leaf
        key[1] = back;
        key[2] = deg;
      }
      if (best < 0 || std::lexicographical_compare(best_key, best_key + 3, key, key + 3)) {
        best = c;
        std::memcpy(best_key, key, sizeof(key));
      }
    }
    if (best < 0) fatal("internal: query is not connected");
    order[i] = best;
    done |= 1u << best;
  }
}

// A countable leaf
bool countable_leaf(const QueryGraph& q, int c, int qa, int qb) {
  if (c == qa || c == qb || q.degree(c) != 1) return false;
  const int p = __builtin_ctz(q.neighbors(c));
  if (!q.directed) return true;
  return (((q.out_adj[p] >> c) & 1u) + ((q.out_adj[c] >> p) & 1u)) == 1;
}

// Moves the countable leaves to the end, keeping the relative order of everything else
void postpone_leaves(const QueryGraph& q, int qa, int qb, int* order) {
  int tmp[kMaxQueryVertices], n = 0;
  for (int i = 0; i < q.k; ++i)
    if (!countable_leaf(q, order[i], qa, qb)) tmp[n++] = order[i];
  for (int i = 0; i < q.k; ++i)
    if (countable_leaf(q, order[i], qa, qb)) tmp[n++] = order[i];
  std::memcpy(order, tmp, sizeof(int) * q.k);
}

// Set partitions of r leaves -> inclusion-exclusion terms for the number of injective assignments.
void leaf_terms(int r, const int* grp, LeafPlan& lp) {
  int blk[kMaxLeaves];
  lp.nterm = 0;
  std::function<void(int, int)> rec = [&](int i, int nb) {
    if (i == r) {
      int64_t coef = 1;
      uint8_t masks[kMaxLeaves] = {};
      for (int b = 0; b < nb; ++b) {
        int sz = 0;
        for (int j = 0; j < r; ++j)
          if (blk[j] == b) {
            ++sz;
            masks[b] |= static_cast<uint8_t>(1u << grp[j]);
          }
        for (int f = 1; f < sz; ++f) coef *= -f;  // (-1)^(sz-1) (sz-1)!
      }
      const int t = lp.nterm++;
      lp.coef[t] = static_cast<int8_t>(coef);
      lp.nblk[t] = static_cast<uint8_t>(nb);
      for (int b = 0; b < nb; ++b) lp.bmask[t][b] = masks[b];
      return;
    }
    for (int b = 0; b <= nb; ++b) {
      blk[i] = b;
      rec(i + 1, b == nb ? nb + 1 : nb);
    }
  };
  rec(0, 0);
}

void emit_plan(const QueryGraph& q, const EdgeStats* st, const int* order, int kind, int anchor, MatchPlan& p) {
  for (int i = 0; i < q.k; ++i) {
    const int c = order[i];
    PlanLevel lv{};
    lv.q = static_cast<uint8_t>(c);
    lv.lq = q.label[c];
    lv.cb = static_cast<uint16_t>(p.checks.size());
    std::vector<double> fcs, pcs;  // per check: fan-out and closure probability (S1 prior)
    for (int j = 0; j < i; ++j) {
      const int pj = order[j];
      auto add = [&](uint8_t dir, label_t el) {
        p.checks.push_back({static_cast<uint8_t>(j), dir, 0, el});
        if (st) {
          fcs.push_back(st->fanout(q.label[pj], dir, q.label[c]));
          pcs.push_back(st->closure(q.label[pj], dir, q.label[c]));
        }
      };
      if (!q.directed) {
        if (q.out_adj[pj] & (1u << c)) add(0, q.has_edge_labels ? query_edge_label(q, pj, c) : kAnyEdgeLabel);
        continue;
      }
      if (q.out_adj[pj] & (1u << c)) add(0, q.has_edge_labels ? query_edge_label(q, pj, c) : kAnyEdgeLabel);
      if (q.out_adj[c] & (1u << pj)) add(1, q.has_edge_labels ? query_edge_label(q, c, pj) : kAnyEdgeLabel);
    }
    lv.nchk = static_cast<uint8_t>(p.checks.size() - lv.cb);
    if (i > 0 && lv.nchk == 0) fatal("internal: level %d of the plan has no edge to earlier levels", i);
    p.levels.push_back(lv);
    // the pivot is the check with the smallest expected list
    double f = 1e300, pi = 1.0;
    size_t piv = 0;
    for (size_t x = 0; x < fcs.size(); ++x)
      if (fcs[x] < f) {
        f = fcs[x];
        piv = x;
      }
    for (size_t x = 0; x < pcs.size(); ++x)
      if (x != piv) pi *= pcs[x];
    if (!st || fcs.empty()) {
      p.prior_c.push_back(1.0f);
      p.prior_s.push_back(1.0f);
      p.prior_r.push_back(1.0f);
    } else {
      p.prior_c.push_back(static_cast<float>(std::max(1.0, std::ceil(f / 32.0))));
      p.prior_s.push_back(static_cast<float>(f * pi));
      p.prior_r.push_back(static_cast<float>(pi));
    }
  }
  // S2 signature gate
  uint32_t prefix = 0;
  for (int i = 0; i < q.k; ++i) {
    prefix |= 1u << order[i];
    p.gate.push_back(i >= 2 && i < q.k - 1 && (q.neighbors(order[i]) & ~prefix) != 0 ? 1 : 0);
  }
  // leaf counting: the maximal suffix of countable leaves
  LeafPlan lp;
  lp.from = static_cast<uint8_t>(q.k);
  int from = q.k;
  uint32_t tail = 0;
  while (from - 1 >= 3 && q.k - (from - 1) <= kMaxLeaves) {
    const int c = order[from - 1];
    if (c == order[0] || c == order[1] || (q.neighbors(c) & tail) != 0) break;
    if (p.levels[p.levels.size() - q.k + from - 1].nchk > kMaxLeafChecks) break;
    tail |= 1u << c;
    --from;
  }
  if (q.k - from >= 2) {
    lp.from = static_cast<uint8_t>(from);
    lp.r = static_cast<uint8_t>(q.k - from);
    int grp[kMaxLeaves];
    for (int j = 0; j < lp.r; ++j) {
      const int L = from + j;
      const PlanLevel& tl = p.levels[p.levels.size() - q.k + L];
      std::vector<std::tuple<uint8_t, uint8_t, label_t>> cs;
      for (int c = 0; c < tl.nchk; ++c) {
        const PlanCheck& ch = p.checks[tl.cb + c];
        cs.emplace_back(ch.pos, ch.dir, ch.el);
      }
      std::sort(cs.begin(), cs.end());
      const label_t lab = q.label[order[L]];
      int g = 0;
      for (; g < lp.ng; ++g) {
        if (lp.glab[g] != lab || lp.gnc[g] != cs.size()) continue;
        bool same = true;
        for (size_t x = 0; x < cs.size() && same; ++x)
          same = lp.gpos[g][x] == std::get<0>(cs[x]) && lp.gdir[g][x] == std::get<1>(cs[x]) && lp.gel[g][x] == std::get<2>(cs[x]);
        if (same) break;
      }
      if (g == lp.ng) {
        for (size_t x = 0; x < cs.size(); ++x) {
          lp.gpos[g][x] = std::get<0>(cs[x]);
          lp.gdir[g][x] = std::get<1>(cs[x]);
          lp.gel[g][x] = std::get<2>(cs[x]);
        }
        lp.gnc[g] = static_cast<uint8_t>(cs.size());
        lp.glab[g] = lab;
        ++lp.ng;
      }
      grp[j] = g;
      ++lp.gmul[g];
    }
    leaf_terms(lp.r, grp, lp);
  }
  p.leaf.push_back(lp);
  p.plan_anchor.push_back(static_cast<uint16_t>(anchor));
  p.plan_kind.push_back(static_cast<uint8_t>(kind));
  ++p.num_plans;
}

void add_anchor(const QueryGraph& q, const EdgeStats* st, int max_orders, int qa, int qb, MatchPlan& p) {
  const int anchor = p.num_anchors;
  std::vector<std::vector<int>> seen;
  for (int kind = 0; kind < std::max(1, std::min(max_orders, 3)); ++kind) {
    int order[kMaxQueryVertices];
    make_order(q, st, kind, qa, qb, order);
    postpone_leaves(q, qa, qb, order);
    std::vector<int> o(order, order + q.k);
    if (std::find(seen.begin(), seen.end(), o) != seen.end()) continue;  // duplicate order
    seen.push_back(o);
    emit_plan(q, st, order, kind, anchor, p);
  }
  if (p.checks.size() > 0xFFFF) fatal("match plan too large (%zu checks)", p.checks.size());
  p.plan_off.push_back(static_cast<uint16_t>(p.num_plans));

  // S2 edge closure requirements of this anchor.
  auto bits = [&](int e, int c) -> uint8_t {
    if (!q.directed) return static_cast<uint8_t>((q.out_adj[e] >> c) & 1u);
    return static_cast<uint8_t>(((q.out_adj[e] >> c) & 1u) | (((q.out_adj[c] >> e) & 1u) << 1));
  };
  std::vector<ClosureReq> third, reqs;
  for (int c = 0; q.k >= 4 && c < q.k; ++c) {
    if (c == qa || c == qb) continue;
    const uint8_t n0 = bits(qa, c), n1 = bits(qb, c);
    if (n0 && n1) third.push_back({q.label[c], n0, n1, 0, 0});
  }
  for (const ClosureReq& r : third) {
    bool dup = false;
    for (const ClosureReq& x : reqs) dup |= x.l == r.l && x.need0 == r.need0 && x.need1 == r.need1;
    if (dup) continue;
    ClosureReq t = r;
    for (const ClosureReq& c : third)
      t.target += c.l == r.l && (c.need0 & r.need0) == r.need0 && (c.need1 & r.need1) == r.need1;
    reqs.push_back(t);
  }
  std::stable_sort(reqs.begin(), reqs.end(), [](const ClosureReq& x, const ClosureReq& y) { return x.l < y.l; });
  p.creq.insert(p.creq.end(), reqs.begin(), reqs.end());
  p.creq_off.push_back(static_cast<uint16_t>(p.creq.size()));
  ++p.num_anchors;
}

}  // namespace

MatchPlan build_match_plan(const QueryGraph& q, const EdgeStats* stats, int max_orders) {
  MatchPlan p;
  p.k = q.k;
  p.creq_off.push_back(0);
  p.plan_off.push_back(0);
  for (const QueryEdge& e : q.edges) {
    add_anchor(q, stats, max_orders, e.src, e.dst, p);
    if (!q.directed) add_anchor(q, stats, max_orders, e.dst, e.src, p);
  }
  return p;
}

void print_match_plan(const MatchPlan& p, const QueryGraph& q, FILE* out) {
  static const char* kKind[3] = {"RI", "rare-first", "leaf-last"};
  std::fprintf(out, "== Match plan: %d anchors, %d plans x %d levels, %zu checks (%s query) ==\n", p.num_anchors,
               p.num_plans, p.k, p.checks.size(), q.directed ? "directed" : "undirected");
  const int show = p.num_anchors < 3 ? p.num_anchors : 3;
  for (int a = 0; a < show; ++a) {
    std::fprintf(out, "  anchor %d: u->q%d v->q%d   closure:", a, p.levels[p.plan_off[a] * p.k].q,
                 p.levels[p.plan_off[a] * p.k + 1].q);
    for (int r = p.creq_off[a]; r < p.creq_off[a + 1]; ++r)
      std::fprintf(out, " [L%u n0=%u n1=%u >=%u]", p.creq[r].l, p.creq[r].need0, p.creq[r].need1, p.creq[r].target);
    std::fprintf(out, "\n");
    for (int pl = p.plan_off[a]; pl < p.plan_off[a + 1]; ++pl) {
      std::fprintf(out, "    plan %d (%s):", pl, kKind[p.plan_kind[pl]]);
      for (int i = 0; i < p.k; ++i) {
        const PlanLevel& lv = p.levels[pl * p.k + i];
        std::fprintf(out, " q%d[", lv.q);
        for (int c = 0; c < lv.nchk; ++c) {
          const PlanCheck& ch = p.checks[lv.cb + c];
          std::fprintf(out, "%s%s%d", c ? "," : "", ch.dir ? "<" : ">", ch.pos);
        }
        std::fprintf(out, "]%s", p.gate[pl * p.k + i] ? "g" : "");
      }
      if (p.leaf[pl].from < p.k)
        std::fprintf(out, "   count-tail from level %d (%d vertices, %d groups)", p.leaf[pl].from, p.leaf[pl].r, p.leaf[pl].ng);
      std::fprintf(out, "\n");
    }
  }
  if (show < p.num_anchors) std::fprintf(out, "  ... (%d more anchors)\n", p.num_anchors - show);
}

}  // namespace csm
