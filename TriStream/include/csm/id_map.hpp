// Mapping of raw file tokens to dense ids.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "csm/common.hpp"

namespace csm {

struct raw_id_t {
  uint64_t hi = 0, lo = 0;
  bool operator==(const raw_id_t& o) const { return hi == o.hi && lo == o.lo; }
  bool operator!=(const raw_id_t& o) const { return !(*this == o); }
};
class IdMap {
 public:
  explicit IdMap(size_t expected = 1024);

  // Dense id of `raw`, inserting it if unseen.
  vid_t get_or_insert(raw_id_t raw, bool* created = nullptr);
  vid_t find(raw_id_t raw) const;  // kInvalidVid if absent
  void reserve(size_t expected);

  size_t size() const { return raw_of_.size(); }
  raw_id_t raw_of(vid_t v) const { return raw_of_[v]; }

 private:
  static uint64_t hash(raw_id_t x);
  void rehash(size_t new_capacity);

  std::vector<raw_id_t> keys_;
  std::vector<vid_t> vals_;  // kInvalidVid marks an empty slot
  
  static constexpr uint64_t kDirectMax = uint64_t{1} << 26;
  std::vector<vid_t> direct_;
  size_t mask_ = 0;
  std::vector<raw_id_t> raw_of_;
};

// Raw numeric labels -> dense label ids.
class LabelMap {
 public:
  explicit LabelMap(bool reserve_none);

  label_t get_or_insert(uint64_t raw);
  label_t find(uint64_t raw) const; 
  size_t size() const { return raw_of_.size(); }
  size_t num_real_labels() const { return raw_of_.size() - (reserve_none_ ? 1 : 0); }
  uint64_t raw_of(label_t l) const { return raw_of_[l]; }

 private:
  bool reserve_none_;
  std::unordered_map<uint64_t, label_t> map_;
  std::vector<uint64_t> raw_of_;
};

}  // namespace csm
