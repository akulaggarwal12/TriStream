// Matching plans per anchor: orders (chosen by S1), S2 closure and gate, leaf counting of the S3 search
#pragma once

#include <cstdint>
#include <vector>

#include "csm/common.hpp"
#include "csm/query_graph.hpp"

namespace csm {

constexpr label_t kAnyEdgeLabel = 0xFFFFFFFFu;

// Required query edges
struct PlanCheck {
  uint8_t pos;
  uint8_t dir;
  uint16_t pad;
  label_t el;  // required edge label
};

struct PlanLevel {
  uint8_t q;      // query vertex matched at this level
  uint8_t nchk;   // number of checks against earlier levels
  uint16_t cb;    // first check
  label_t lq;     // vertex label of q
};

// S2 edge closure 
struct ClosureReq {
  label_t l;
  uint8_t need0, need1, target, pad;
};

// Leaf counting 
constexpr int kMaxLeaves = 5;       // leaves counted per plan 
constexpr int kMaxLeafTerms = 52;   // Bell(4)
constexpr int kMaxLeafChecks = 8;
struct LeafPlan {
  uint8_t from = 0, r = 0, ng = 0, nterm = 0;
  uint8_t gpos[kMaxLeaves][kMaxLeafChecks] = {}, gdir[kMaxLeaves][kMaxLeafChecks] = {};  // per group: parent level, direction of the parent check
  label_t glab[kMaxLeaves] = {}, gel[kMaxLeaves][kMaxLeafChecks] = {};  // per group: vertex label, required edge label
  uint8_t gnc[kMaxLeaves] = {};
  uint8_t gmul[kMaxLeaves] = {};
  int8_t coef[kMaxLeafTerms] = {};
  uint8_t nblk[kMaxLeafTerms] = {};
  uint8_t bmask[kMaxLeafTerms][kMaxLeaves] = {};  // per term, per block: bitmask of the groups in the block
};

struct MatchPlan {
  int k = 0;             // query vertices = levels per plan
  int num_anchors = 0;
  int num_plans = 0;
  std::vector<uint16_t> plan_off;     // num_anchors + 1
  std::vector<uint16_t> plan_anchor;  // num_plans
  std::vector<uint8_t> plan_kind;     // num_plans: 0 RI, 1 rare-first, 2 leaf-last
  std::vector<PlanLevel> levels;  // num_plans * k, plan p uses levels p*k to +k levels
  std::vector<PlanCheck> checks;
  // S1 priorities
  std::vector<float> prior_c, prior_s, prior_r;
  // S2 triage: edge closure and signature gate
  std::vector<ClosureReq> creq;     // per anchor, sorted by label
  std::vector<uint16_t> creq_off;   // num_anchors + 1
  std::vector<uint8_t> gate;        // gate flag
  std::vector<LeafPlan> leaf;       // num_plans (leaf.from = k: no leaf counting)
};

struct EdgeStats;
// S1 stats
MatchPlan build_match_plan(const QueryGraph& q, const EdgeStats* stats = nullptr, int max_orders = 3);

// Human-readable dump (orders and checks), for the log.
void print_match_plan(const MatchPlan& p, const QueryGraph& q, FILE* out);

}  // namespace csm
