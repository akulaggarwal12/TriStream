// S2 — Discrete Derivative Signatures (DDS)
#pragma once

#include <cstdint>

#include "csm/common.hpp"

namespace csm {
namespace sig {

constexpr int kMaxSlope = 32;
constexpr int kMaxStair = 32;
constexpr int kMaxMid = 32;
constexpr int kMaxFar = 32;
constexpr int kMaxPair = 128;
constexpr int kMaxWords = kMaxSlope + kMaxStair + kMaxPair;
constexpr uint8_t kNoQL = 0xFF;

// Staircase thresholds
constexpr int kNumT = 8;
CSM_HD uint32_t threshold(int j) {  // 2, 3, 4, 5, 6, 8, 12, 16
  return j < 5 ? static_cast<uint32_t>(j + 2) : (j == 5 ? 8u : (j == 6 ? 12u : 16u));
}
// Number of thresholds reached by degree d (0..kNumT).
CSM_HD int stair_class(uint32_t d) {
  int c = 0;
  while (c < kNumT && d >= threshold(c)) ++c;
  return c;
}

CSM_HD uint32_t mix32(uint64_t x) {
  x ^= x >> 33;
  x *= 0xFF51AFD7ED558CCDull;
  x ^= x >> 33;
  x *= 0xC4CEB9FE1A85EC53ull;
  x ^= x >> 33;
  return static_cast<uint32_t>(x);
}
struct Layout {
  int n1 = 0, ns = 0, nm = 0, nf = 0, np = 0, words = 0;
  int nel = 1;
  uint32_t nvl = 0, nelab = 0;
  const uint8_t* lq = nullptr;
  const uint8_t* le = nullptr;
  const int8_t* s1 = nullptr;
  const int8_t* st = nullptr;
  const int8_t* mid = nullptr;
  const int8_t* far = nullptr;
  const int16_t* pair = nullptr;
  const uint32_t* pm = nullptr;
  const uint16_t* moff = nullptr;
  const uint8_t* mfar = nullptr;
  const uint8_t* mslot = nullptr;

  CSM_HD int ql(label_t l) const { return l < nvl ? lq[l] : kNoQL; }
  CSM_HD int slope(label_t l, uint32_t d, label_t el) const {
    const int q = ql(l);
    if (q == kNoQL) return -1;
    int e = 0;
    if (le) {
      if (el >= nelab || le[el] == kNoQL) return -1;
      e = le[el];
    }
    return s1[(q * 2 + static_cast<int>(d)) * nel + e];
  }
  CSM_HD int stair(label_t l, uint32_t d, int j) const {
    const int q = ql(l);
    return q == kNoQL ? -1 : st[(q * 2 + static_cast<int>(d)) * kNumT + j];
  }
  CSM_HD int mid_of(label_t l, uint32_t d) const {
    const int q = ql(l);
    return q == kNoQL ? -1 : mid[q * 2 + static_cast<int>(d)];
  }
  CSM_HD int far_of(label_t l, uint32_t d) const {
    const int q = ql(l);
    return q == kNoQL ? -1 : far[q * 2 + static_cast<int>(d)];
  }
  CSM_HD int pair_of(int m, int f) const { return (m < 0 || f < 0) ? -1 : pair[m * nf + f]; }
};

// The S2 test
CSM_HD bool dominates(const Layout& L, const uint32_t* qrow, uint32_t qdo, uint32_t qdi, const uint32_t* vrow,
                      uint32_t vdo, uint32_t vdi, uint32_t vmask, uint32_t vhub) {
  if (vdo < qdo || vdi < qdi) return false;
  const int first = L.n1 + L.ns;
  for (int i = 0; i < first; ++i)
    if (vrow[i] < qrow[i]) return false;
  if (vhub) return true;
  for (int p = 0; p < L.np; ++p) {
    if (vmask & L.pm[p]) continue;
    if (vrow[first + p] < qrow[first + p]) return false;
  }
  return true;
}

}  // namespace sig
}  // namespace csm
