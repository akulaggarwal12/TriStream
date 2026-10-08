#include "csm/id_map.hpp"

#include <algorithm>

namespace csm {

IdMap::IdMap(size_t expected) { reserve(expected); }

uint64_t IdMap::hash(raw_id_t x) {
  // splitmix64 finalizer over both 64-bit halves
  uint64_t h = x.lo ^ (x.hi * 0x9E3779B97F4A7C15ull);
  h ^= h >> 30;
  h *= 0xBF58476D1CE4E5B9ull;
  h ^= h >> 27;
  h *= 0x94D049BB133111EBull;
  h ^= h >> 31;
  return h;
}

void IdMap::reserve(size_t expected) {
  size_t cap = 16;
  while (cap < expected * 2) cap <<= 1;  // load factor <= 0.5
  if (cap > keys_.size()) rehash(cap);
  raw_of_.reserve(expected);
}

void IdMap::rehash(size_t new_capacity) {
  std::vector<raw_id_t> old_keys = std::move(keys_);
  std::vector<vid_t> old_vals = std::move(vals_);
  keys_.assign(new_capacity, raw_id_t{});
  vals_.assign(new_capacity, kInvalidVid);
  mask_ = new_capacity - 1;
  for (size_t i = 0; i < old_vals.size(); ++i) {
    if (old_vals[i] == kInvalidVid) continue;
    size_t p = hash(old_keys[i]) & mask_;
    while (vals_[p] != kInvalidVid) p = (p + 1) & mask_;
    keys_[p] = old_keys[i];
    vals_[p] = old_vals[i];
  }
}

vid_t IdMap::get_or_insert(raw_id_t raw, bool* created) {
  if (raw.hi == 0 && raw.lo < kDirectMax) {
    if (raw.lo >= direct_.size())
      direct_.resize(std::min<uint64_t>(kDirectMax, std::max<uint64_t>(raw.lo + 1, direct_.size() * 2)), kInvalidVid);
    vid_t& slot = direct_[raw.lo];
    if (slot != kInvalidVid) {
      if (created) *created = false;
      return slot;
    }
    if (raw_of_.size() >= kInvalidVid) fatal("more than 2^32-1 vertices are not supported");
    slot = static_cast<vid_t>(raw_of_.size());
    raw_of_.push_back(raw);
    if (created) *created = true;
    return slot;
  }
  if ((raw_of_.size() + 1) * 2 > keys_.size()) rehash(keys_.size() * 2);
  size_t p = hash(raw) & mask_;
  while (vals_[p] != kInvalidVid) {
    if (keys_[p] == raw) {
      if (created) *created = false;
      return vals_[p];
    }
    p = (p + 1) & mask_;
  }
  if (raw_of_.size() >= kInvalidVid) fatal("more than 2^32-1 vertices are not supported");
  const vid_t id = static_cast<vid_t>(raw_of_.size());
  keys_[p] = raw;
  vals_[p] = id;
  raw_of_.push_back(raw);
  if (created) *created = true;
  return id;
}

vid_t IdMap::find(raw_id_t raw) const {
  if (raw.hi == 0 && raw.lo < kDirectMax) return raw.lo < direct_.size() ? direct_[raw.lo] : kInvalidVid;
  size_t p = hash(raw) & mask_;
  while (vals_[p] != kInvalidVid) {
    if (keys_[p] == raw) return vals_[p];
    p = (p + 1) & mask_;
  }
  return kInvalidVid;
}

LabelMap::LabelMap(bool reserve_none) : reserve_none_(reserve_none) {
  if (reserve_none_) raw_of_.push_back(UINT64_MAX);  // dense id 0 == kNoEdgeLabel
}

label_t LabelMap::get_or_insert(uint64_t raw) {
  auto it = map_.find(raw);
  if (it != map_.end()) return it->second;
  const label_t id = static_cast<label_t>(raw_of_.size());
  map_.emplace(raw, id);
  raw_of_.push_back(raw);
  return id;
}

label_t LabelMap::find(uint64_t raw) const {
  auto it = map_.find(raw);
  return it == map_.end() ? kInvalidLabel : it->second;
}

}  // namespace csm
